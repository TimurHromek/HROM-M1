import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
import math
import re # Added for punctuation fix
import logging
from collections import defaultdict # Keep for CheckpointLoader

# Rich library imports
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.status import Status
from rich.rule import Rule
import string # For punctuation check (though not directly used in the fix, it's available)

# Setup Rich logging
console = Console(highlight=False) # Global console for consistent output
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s', # Simpler format for RichHandler
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, show_level=True, log_time_format="[%H:%M:%S]")],
    force=True
)

# Configuration for Inference
CONFIG = {
    "dim": 768,
    "n_layers": 8,
    "n_heads": 8,
    "ff_dim": 2048, # ff_dim for each expert
    "dropout": 0.1, # Used in model definition, though inactive in eval mode
    "max_seq_len": 512,
    "vocab_size": 32000, # Will be updated by tokenizer if different
    "tokenizer_name": "hrom_moe_tokenizer.json",
    "checkpoint_dir": "checkpoints_moe",

    # --- MoE Specific Configuration ---
    "num_experts": 8,
    "top_k_experts": 2,

    # --- Generation Specific Configuration ---
    "max_new_tokens_generation": 5000,
    "temperature_generation": 0.7,
    "top_k_generation": 40,
}

if CONFIG["top_k_experts"] > CONFIG["num_experts"]:
    logging.warning(f"top_k_experts ({CONFIG['top_k_experts']}) > num_experts ({CONFIG['num_experts']}). Setting top_k_experts to num_experts.")
    CONFIG["top_k_experts"] = CONFIG["num_experts"]


# --- Model Definition (HROM, HROMBlock_MoE, HROMAttention, SwiGLU, RoPE, Expert, MoELayer) ---

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        if seq_len == 0:
             return torch.empty((0, self.inv_freq.shape[0] * 2), device=self.inv_freq.device)
        if freqs.ndim == 1 and seq_len > 0 :
            freqs = freqs.unsqueeze(0)
        
        if freqs.shape[0] != seq_len and seq_len > 0:
             freqs = freqs.reshape(seq_len, -1)
        elif seq_len == 0:
            return torch.empty((0, self.inv_freq.shape[0]*2), device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(pos, t):
    pos = pos.to(t.device, dtype=t.dtype)
    pos = pos.unsqueeze(0).unsqueeze(0)

    tensor_seq_len = t.shape[2]
    pos_seq_len = pos.shape[2]

    if pos_seq_len < tensor_seq_len:
         t_rotated_part = t[:, :, :pos_seq_len, :]
         pos_truncated = pos[:, :, :pos_seq_len, :]
         
         cos_pos = pos_truncated.cos()
         sin_pos = pos_truncated.sin()

         t_rotated_part = (t_rotated_part * cos_pos) + (rotate_half(t_rotated_part) * sin_pos)
         t_unrotated_part = t[:, :, pos_seq_len:, :]
         return torch.cat([t_rotated_part, t_unrotated_part], dim=2)
    elif pos_seq_len > tensor_seq_len:
         pos = pos[:, :, :tensor_seq_len, :]

    if pos.shape[-1] != t.shape[-1]:
        logging.error(f"Mismatched head dimensions for RoPE: pos ({pos.shape[-1]}) vs t ({t.shape[-1]})")
        raise ValueError("Rotary embedding head dimension must match tensor head dimension.")
    if pos.shape[2] != t.shape[2]:
        logging.error(f"Mismatched sequence lengths for RoPE: pos ({pos.shape[2]}) vs t ({t.shape[2]})")
        raise ValueError("Rotary embedding sequence length must match tensor sequence length after adjustments.")

    cos_pos = pos.cos()
    sin_pos = pos.sin()
    rotated_t = (t * cos_pos) + (rotate_half(t) * sin_pos)
    return rotated_t


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)

class HROMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = CONFIG["dim"]
        self.n_heads = CONFIG["n_heads"]
        self.head_dim = self.dim // self.n_heads
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        pos = self.rotary(T)
        q = apply_rotary_pos_emb(pos, q)
        k = apply_rotary_pos_emb(pos, k)
        
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + mask
            
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
        attn_probs = self.dropout(attn_probs)
        
        output = attn_probs @ v
        output = output.transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(output)

class Expert(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * ff_dim)
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        hidden = self.fc1(x)
        activated_hidden = self.activation(hidden)
        return self.fc2(activated_hidden)

class MoELayer(nn.Module):
    def __init__(self, dim, ff_dim, num_experts, top_k):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(dim, ff_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        
        gate_logits = self.gate(x_reshaped)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_weights_norm = top_k_gate_probs / (top_k_gate_probs.sum(dim=-1, keepdim=True) + 1e-6)

        final_output = torch.zeros_like(x_reshaped)
        
        for i in range(self.num_experts):
            token_indices_for_expert_i, position_in_top_k = torch.where(top_k_indices == i)
            
            if token_indices_for_expert_i.numel() > 0:
                tokens_for_this_expert = x_reshaped[token_indices_for_expert_i]
                weights_for_this_expert = top_k_weights_norm[token_indices_for_expert_i, position_in_top_k]
                expert_output = self.experts[i](tokens_for_this_expert)
                weighted_expert_output = expert_output * weights_for_this_expert.unsqueeze(-1)
                final_output.index_add_(0, token_indices_for_expert_i, weighted_expert_output.to(final_output.dtype))

        final_output = final_output.reshape(batch_size, seq_len, dim)
        return final_output


class HROMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HROMAttention()
        self.moe_layer = MoELayer(
            dim=CONFIG["dim"],
            ff_dim=CONFIG["ff_dim"],
            num_experts=CONFIG["num_experts"],
            top_k=CONFIG["top_k_experts"]
        )
        self.norm1 = nn.LayerNorm(CONFIG["dim"])
        self.norm2 = nn.LayerNorm(CONFIG["dim"])
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        residual1 = x
        normed_x1 = self.norm1(x)
        attn_output = self.attn(normed_x1, mask)
        x = residual1 + self.dropout(attn_output)

        residual2 = x
        normed_x2 = self.norm2(x)
        ff_output = self.moe_layer(normed_x2)
        x = residual2 + self.dropout(ff_output)
        return x


class HROM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG["vocab_size"], CONFIG["dim"])
        self.blocks = nn.ModuleList([HROMBlock() for _ in range(CONFIG["n_layers"])])
        self.norm = nn.LayerNorm(CONFIG["dim"])
        self.head = nn.Linear(CONFIG["dim"], CONFIG["vocab_size"])
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
             torch.nn.init.zeros_(module.bias)
             torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        x = self.embed(input_ids)
        x = self.dropout(x)

        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        additive_causal_mask = torch.zeros_like(causal_mask, dtype=x.dtype)
        additive_causal_mask.masked_fill_(causal_mask, float('-inf'))
        
        combined_mask = additive_causal_mask.unsqueeze(0).unsqueeze(0) 

        if attention_mask is not None:
            pad_mask_additive = (1.0 - attention_mask.to(x.dtype)) * torch.finfo(x.dtype).min
            pad_mask_additive = pad_mask_additive.unsqueeze(1).unsqueeze(2)
            
            if combined_mask.shape[0] == 1 and B > 1:
                combined_mask = combined_mask.expand(B, -1, -1, -1)
            
            combined_mask = combined_mask + pad_mask_additive

        combined_mask = combined_mask.to(dtype=x.dtype)

        for block in self.blocks:
            x = block(x, combined_mask)
        
        x = self.norm(x)
        logits = self.head(x)
        return logits


class TokenizerLoader:
    def __init__(self):
        self.tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Ensure it's trained and available.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
         tokenizer.pad_id = tokenizer.token_to_id("<pad>")
         tokenizer.eos_id = tokenizer.token_to_id("</s>")
         tokenizer.bos_id = tokenizer.token_to_id("<s>")
         tokenizer.user_id = tokenizer.token_to_id("<user>")
         tokenizer.assistant_id = tokenizer.token_to_id("<assistant>")
         return tokenizer


class CheckpointLoader:
    def __init__(self):
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        if not os.path.isdir(self.checkpoint_dir):
            logging.warning(f"Checkpoint directory '{self.checkpoint_dir}' does not exist.")

    def load_latest(self, model, checkpoint_path=None):
        if checkpoint_path:
            if not os.path.exists(checkpoint_path):
                logging.error(f"Specified checkpoint path '{checkpoint_path}' does not exist.")
                return False
            latest_checkpoint_path = checkpoint_path
            logging.info(f"Loading specified checkpoint from: {latest_checkpoint_path}")
        else:
            try:
                prefix_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
                checkpoints = []
                if not os.path.isdir(self.checkpoint_dir):
                    logging.info(f"No checkpoint directory at '{self.checkpoint_dir}'. Cannot load latest.")
                    return False

                for f_name in os.listdir(self.checkpoint_dir):
                     if f_name.endswith(".pt") and f_name.startswith(f"hrom_{prefix_base}"):
                          filepath = os.path.join(self.checkpoint_dir, f_name)
                          checkpoints.append((filepath, os.path.getmtime(filepath)))

                if not checkpoints:
                    logging.info(f"No valid .pt checkpoints found in '{self.checkpoint_dir}' starting with 'hrom_{prefix_base}'. Cannot load.")
                    return False

                checkpoints.sort(key=lambda x: x[1], reverse=True)
                latest_checkpoint_path, _ = checkpoints[0]
                logging.info(f"Loading latest checkpoint (by mtime): {latest_checkpoint_path}")

            except Exception as e:
                logging.error(f"Error finding latest checkpoint: {e}. Cannot load.", exc_info=True)
                return False

        try:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_checkpoint_path, map_location=map_location)

            loaded_config = checkpoint.get("config", {})
            critical_keys = ["dim", "n_layers", "n_heads", "ff_dim", "vocab_size", "max_seq_len",
                             "tokenizer_name", "num_experts", "top_k_experts"]
            if loaded_config:
                mismatched_keys = []
                for key in critical_keys:
                    loaded_val = loaded_config.get(key)
                    current_val = CONFIG.get(key)
                    if key == "vocab_size" and hasattr(model, 'embed') and model.embed.num_embeddings != loaded_val and loaded_val is not None:
                         logging.warning(f"Checkpoint vocab_size ({loaded_val}) differs from initial model embed layer ({model.embed.num_embeddings}). This might be fine if vocab_size is updated from tokenizer later or if this checkpoint has the correct vocab_size.")
                    elif key == "vocab_size":
                        pass
                    elif loaded_val is not None and loaded_val != current_val:
                        mismatched_keys.append((key, loaded_val, current_val))

                if mismatched_keys:
                    console.print(Panel(Text("--- CONFIG MISMATCH DETECTED (Loading Checkpoint) ---", style="bold yellow")))
                    for key, loaded_val, current_val in mismatched_keys:
                        logging.warning(f"  - {key}: Checkpoint='{loaded_val}', Current='{current_val}'")
                    logging.warning("Proceeding with loading, but this may impact model performance or cause errors if critical arch params changed.")
            else:
                logging.warning("Checkpoint does not contain configuration info. Cannot check for mismatches.")

            if "vocab_size" in loaded_config and loaded_config["vocab_size"] != CONFIG["vocab_size"]:
                logging.info(f"Updating CONFIG['vocab_size'] from {CONFIG['vocab_size']} to {loaded_config['vocab_size']} based on checkpoint.")
                CONFIG["vocab_size"] = loaded_config["vocab_size"]
                if hasattr(model, 'embed') and model.embed.num_embeddings != CONFIG["vocab_size"]:
                    logging.warning(f"Model embedding layer size ({model.embed.num_embeddings}) still differs from new CONFIG vocab_size ({CONFIG['vocab_size']}). Strict loading might fail. Ensure model is re-initialized if vocab_size changes significantly.")

            model.load_state_dict(checkpoint['model'], strict=True)
            logging.info(f"Model weights loaded successfully from {latest_checkpoint_path}.")
            return True

        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at '{latest_checkpoint_path}'.")
            return False
        except RuntimeError as e:
            logging.error(f"Failed to load model state_dict: {e}. Model architecture (esp. vocab_size) may have changed or checkpoint is incompatible.")
            return False
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}.", exc_info=True)
            return False

# Regex to fix spacing before common punctuation marks
PUNCTUATION_FIX_REGEX = re.compile(r'\s+([?.!,;:\'\"])')

def _print_chat_header(console_instance: Console):
    """Prints the welcome panel and current settings."""
    console_instance.print(Panel(
        Text.from_markup(
            "[bold green]HROM-MoE Chatbot[/bold green]\n\n"
            "Type your message and press Enter.\n"
            "Available commands:\n"
            "  `clear` - Clear the screen.\n"
            "  `temp <value>` - Set generation temperature (e.g., `temp 0.7`).\n"
            "  `top-k <value>` - Set top-k sampling (e.g., `top-k 40`).\n"
            "  `quit`, `exit`, `bye` - End the session."
        ),
        title="[cyan]Welcome![/cyan]",
        expand=False,
        border_style="blue"
    ))
    current_config_display = (
        f"Temp: [cyan]{CONFIG['temperature_generation']:.2f}[/cyan], "
        f"Top-K: [cyan]{CONFIG['top_k_generation']}[/cyan], "
        f"Max New Tokens: [cyan]{CONFIG['max_new_tokens_generation']}[/cyan]"
    )
    console_instance.print(Panel(Text.from_markup(current_config_display), title="[yellow]Current Generation Settings[/yellow]", border_style="yellow", expand=False))
    console_instance.print(Rule(style="blue"))


def chat_ui(model, tokenizer, device, console_instance: Console):
    model.eval()
    _print_chat_header(console_instance)

    while True:
        try:
            user_prompt_text = console_instance.input(Text.from_markup("[b steel_blue]You:[/b steel_blue] "))
            command_lower = user_prompt_text.lower().strip()

            if command_lower in ["quit", "exit", "bye"]:
                console_instance.print(Text.from_markup("[b green]HROM:[/b green] Goodbye! 👋"))
                break
            
            if command_lower == "clear":
                console_instance.clear()
                _print_chat_header(console_instance) # Re-print header after clearing
                continue

            elif command_lower.startswith("temp "):
                parts = user_prompt_text.split()
                if len(parts) == 2:
                    try:
                        new_temp = float(parts[1])
                        if new_temp > 0:
                            CONFIG["temperature_generation"] = new_temp
                            console_instance.print(Text.from_markup(f"[italic yellow]Temperature set to {new_temp:.2f}[/italic yellow]"))
                            # Update settings display
                            _print_chat_header(console_instance) # Simplest way to show updated settings
                        else:
                            console_instance.print(Text.from_markup("[italic red]Error: Temperature must be > 0.[/italic red]"))
                    except ValueError:
                        console_instance.print(Text.from_markup("[italic red]Error: Invalid temperature value.[/italic red]"))
                else:
                    console_instance.print(Text.from_markup("[italic red]Usage: temp <value>[/italic red]"))
                console_instance.print(Rule(style="blue")) # Keep rule after command output
                continue

            elif command_lower.startswith("top-k "):
                parts = user_prompt_text.split()
                if len(parts) == 2:
                    try:
                        new_top_k = int(parts[1])
                        if new_top_k >= 0:
                            CONFIG["top_k_generation"] = new_top_k
                            console_instance.print(Text.from_markup(f"[italic yellow]Top-K set to {new_top_k}[/italic yellow]"))
                            _print_chat_header(console_instance) # Simplest way to show updated settings
                        else:
                            console_instance.print(Text.from_markup("[italic red]Error: Top-K must be >= 0.[/italic red]"))
                    except ValueError:
                        console_instance.print(Text.from_markup("[italic red]Error: Invalid Top-K value.[/italic red]"))
                else:
                    console_instance.print(Text.from_markup("[italic red]Usage: top-k <value>[/italic red]"))
                console_instance.print(Rule(style="blue")) # Keep rule after command output
                continue
            
            if not user_prompt_text.strip(): # If it was not a command and is empty
                continue

            prompt_ids_user_part = tokenizer.encode(user_prompt_text, add_special_tokens=False).ids
            
            current_generated_ids = [tokenizer.bos_id, tokenizer.user_id] + \
                                    prompt_ids_user_part + \
                                    [tokenizer.eos_id, tokenizer.assistant_id]
            
            response_display_ids = [] 
            displayed_text_len = 0    

            assistant_response_text = Text.from_markup("[b green]HROM:[/b green] ")
            
            with Live(assistant_response_text, console=console_instance, refresh_per_second=12, vertical_overflow="visible") as live:
                with torch.no_grad():
                    for step in range(CONFIG["max_new_tokens_generation"]):
                        model_input_ids_list = current_generated_ids[-CONFIG["max_seq_len"]:]
                        input_tensor = torch.tensor([model_input_ids_list], dtype=torch.long, device=device)
                        attention_mask = torch.ones_like(input_tensor, device=device)

                        try:
                            logits = model(input_tensor, attention_mask=attention_mask)
                            next_token_logits = logits[:, -1, :]
                        except Exception as e:
                             logging.error(f"Model forward pass failed during generation: {e}", exc_info=True)
                             error_msg = Text(f"\n[Internal Error: Model forward pass failed]", style="red")
                             if assistant_response_text.spans:
                                 assistant_response_text.append_text(error_msg)
                             else:
                                 assistant_response_text = error_msg
                             live.update(assistant_response_text)
                             break 

                        temp = CONFIG["temperature_generation"]
                        if temp > 0 and temp != 1.0: # Avoid division by zero if temp somehow becomes 0
                            next_token_logits = next_token_logits / temp

                        top_k_val = CONFIG["top_k_generation"]
                        if top_k_val > 0 and top_k_val < next_token_logits.size(-1):
                            v, _ = torch.topk(next_token_logits, top_k_val, dim=-1)
                            threshold_val = v[:, -1].unsqueeze(-1)
                            next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold_val, -float('Inf'))

                        probs = torch.softmax(next_token_logits, dim=-1)
                        if torch.isnan(probs).any() or torch.isinf(probs).any():
                             logging.warning(f"NaN/Inf detected in probabilities at step {step}. Using uniform distribution as fallback.")
                             probs = torch.ones_like(probs) / probs.size(-1)

                        next_token_id = torch.multinomial(probs, num_samples=1).item()
                        
                        if next_token_id == tokenizer.eos_id:
                            logging.debug(f"EOS token ({tokenizer.eos_id}) generated at step {step+1}. Stopping.")
                            break
                        
                        current_generated_ids.append(next_token_id)
                        response_display_ids.append(next_token_id)
                        
                        full_decoded_response_segment = tokenizer.decode(response_display_ids, skip_special_tokens=True)
                        
                        new_text_to_display = full_decoded_response_segment[displayed_text_len:]
                        
                        if new_text_to_display:
                            # Apply punctuation fix
                            new_text_to_display = PUNCTUATION_FIX_REGEX.sub(r'\1', new_text_to_display)
                            
                            assistant_response_text.append(new_text_to_display)
                            live.update(assistant_response_text)
                            displayed_text_len = len(full_decoded_response_segment)

                        if step == CONFIG["max_new_tokens_generation"] - 1:
                             logging.debug("Max new tokens reached.")
            
            console_instance.print(Rule(style="blue"))

        except KeyboardInterrupt:
            console_instance.print(Text.from_markup("\n[b green]HROM:[/b green] Session interrupted by user. Goodbye! 👋"))
            break
        except Exception as e:
            logging.error(f"An error occurred in the chat loop: {e}", exc_info=True)
            console_instance.print(Panel(Text(f"An unexpected error occurred: {e}", style="bold red"), title="[red]Error[/red]", border_style="red"))
            console_instance.print(Rule(style="blue"))


if __name__ == "__main__":
    console.print(Rule("[bold yellow]HROM-MoE Inference Engine Starting Up[/bold yellow]"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer_instance = None
    with Status("[cyan]Loading tokenizer...", console=console, spinner="dots"):
        try:
            tokenizer_loader = TokenizerLoader()
            tokenizer_instance = tokenizer_loader.get_tokenizer()
            if CONFIG["vocab_size"] != tokenizer_instance.get_vocab_size():
                logging.info(f"Updating CONFIG['vocab_size'] from {CONFIG['vocab_size']} to {tokenizer_instance.get_vocab_size()} based on tokenizer.")
                CONFIG["vocab_size"] = tokenizer_instance.get_vocab_size()
            logging.info(f"Tokenizer loaded. Vocab size: {CONFIG['vocab_size']}")
        except Exception as e:
            logging.error(f"Failed to load tokenizer: {e}", exc_info=True)
            console.print(Panel(Text(f"Fatal Error: Could not load tokenizer.\n{e}", style="bold red"), title="[red]Startup Error[/red]"))
            exit(1)

    model_instance = None
    with Status("[cyan]Initializing HROM-MoE model...", console=console, spinner="dots"):
        try:
            model_instance = HROM().to(device)
            total_params = sum(p.numel() for p in model_instance.parameters())
            trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
            logging.info(f"Model initialized. Total params: {total_params/1e6:.2f}M. Trainable: {trainable_params/1e6:.2f}M")
        except Exception as e:
            logging.error(f"Failed to initialize model: {e}", exc_info=True)
            console.print(Panel(Text(f"Fatal Error: Could not initialize model.\n{e}", style="bold red"), title="[red]Startup Error[/red]"))
            exit(1)

    with Status("[cyan]Loading model checkpoint...", console=console, spinner="dots") as status:
        checkpoint_loader = CheckpointLoader()
        loaded_successfully = checkpoint_loader.load_latest(model_instance)

        if not loaded_successfully:
            logging.error("Could not load model checkpoint. Exiting.")
            console.print(Panel(Text(f"Fatal Error: Could not load model checkpoint from '{CONFIG['checkpoint_dir']}'.\nPlease ensure checkpoints exist or specify a path.", style="bold red"), title="[red]Startup Error[/red]"))
            exit(1)
        
        model_instance.to(device)

    console.print(Rule("[bold green]Model Ready[/bold green]"))
    chat_ui(model_instance, tokenizer_instance, device, console)
