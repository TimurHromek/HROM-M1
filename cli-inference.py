import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for softmax in MoE
from tokenizers import Tokenizer, models, pre_tokenizers, decoders # Removed trainers, processors
import math
import re
import logging
from collections import defaultdict # Keep for CheckpointLoader optimizer state loading (though we'll simplify)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
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
    # "moe_load_balancing_coeff": 0.01 # Not needed for inference
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
    pos = pos.unsqueeze(0).unsqueeze(1)
    tensor_seq_len = t.shape[2]
    pos_seq_len = pos.shape[2]

    if pos_seq_len < tensor_seq_len:
         # logging.debug(f"RoPE Warning: pos sequence length ({pos_seq_len}) is shorter than tensor sequence length ({tensor_seq_len}). Using truncated tensor length for RoPE.")
         t_rotated = t[:, :, :pos_seq_len, :]
         pos = pos[:, :, :pos_seq_len, :]
         cos_pos = pos.cos()
         sin_pos = pos.sin()
         t_rotated = (t_rotated * cos_pos) + (rotate_half(t_rotated) * sin_pos)
         t_unrotated = t[:, :, pos_seq_len:, :]
         return torch.cat([t_rotated, t_unrotated], dim=2)
    elif pos_seq_len > tensor_seq_len:
         pos = pos[:, :, :tensor_seq_len, :]

    if pos.shape[-1] != t.shape[-1]:
        logging.error(f"Mismatched dimensions for RoPE: pos ({pos.shape[-1]}) vs t ({t.shape[-1]})")
        raise ValueError("Rotary embedding dimension must match head dimension.")

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
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_scores = attn_scores + mask
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(dtype=x.dtype)
        attn_probs = self.dropout(attn_probs) # Dropout is inactive during model.eval()
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
    def __init__(self, dim, ff_dim, num_experts, top_k): # Removed load_balancing_coeff
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        # self.load_balancing_coeff = load_balancing_coeff # Not needed for inference

        self.experts = nn.ModuleList([Expert(dim, ff_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        # num_tokens = x_reshaped.shape[0] # Not explicitly needed without loss calc

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

        # Load balancing loss calculation removed for inference
        final_output = final_output.reshape(batch_size, seq_len, dim)
        return final_output # Only return final_output


class HROMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HROMAttention()
        self.moe_layer = MoELayer(
            dim=CONFIG["dim"],
            ff_dim=CONFIG["ff_dim"],
            num_experts=CONFIG["num_experts"],
            top_k=CONFIG["top_k_experts"]
            # load_balancing_coeff not passed
        )
        self.norm1 = nn.LayerNorm(CONFIG["dim"])
        self.norm2 = nn.LayerNorm(CONFIG["dim"])
        self.dropout = nn.Dropout(CONFIG["dropout"])

    def forward(self, x, mask=None):
        residual1 = x
        normed_x1 = self.norm1(x)
        attn_output = self.attn(normed_x1, mask)
        x = residual1 + self.dropout(attn_output) # Dropout inactive during model.eval()

        residual2 = x
        normed_x2 = self.norm2(x)
        # moe_aux_loss is not returned by MoELayer anymore
        ff_output = self.moe_layer(normed_x2)
        x = residual2 + self.dropout(ff_output) # Dropout inactive during model.eval()
        return x # Only return x


class HROM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(CONFIG["vocab_size"], CONFIG["dim"])
        self.blocks = nn.ModuleList([HROMBlock() for _ in range(CONFIG["n_layers"])])
        self.norm = nn.LayerNorm(CONFIG["dim"])
        self.head = nn.Linear(CONFIG["dim"], CONFIG["vocab_size"])
        self.dropout = nn.Dropout(CONFIG["dropout"]) # Applied to embeddings
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
        x = self.dropout(x) # Dropout inactive during model.eval() if model.eval() is called

        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(1) # Shape: (1, 1, T, T)

        if attention_mask is not None:
            # Ensure attention_mask is correctly broadcastable for addition
            # Expected shape for pad_mask (B, 1, 1, T) or (B, 1, T_kv, T_q)
            # Here, T_kv = T_q = T
            pad_mask = (1.0 - attention_mask.to(torch.float32)) * torch.finfo(torch.float32).min
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2) # Shape: (B, 1, 1, T)
            if combined_mask.shape[0] == 1 and B > 1 : # If causal_mask is (1,1,T,T) and B > 1
                combined_mask = combined_mask.expand(B, -1, -1, -1) # Expand to (B,1,T,T)
            combined_mask = combined_mask + pad_mask # Add padding mask

        combined_mask = combined_mask.to(dtype=x.dtype)

        # total_moe_aux_loss = 0.0 # Not needed for inference
        for block in self.blocks:
            # block_moe_aux_loss is not returned by HROMBlock anymore
            x = block(x, combined_mask)
            # total_moe_aux_loss += block_moe_aux_loss # Not needed

        x = self.norm(x)
        logits = self.head(x)

        # avg_moe_aux_loss = total_moe_aux_loss / CONFIG["n_layers"] if CONFIG["n_layers"] > 0 else 0.0 # Not needed
        return logits # Only return logits


class TokenizerLoader:
    def __init__(self):
        self.tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
        # self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"] # For reference

    def _clean_text(self, text): # Kept for potential use, though not used in basic loop
        text = str(text)
        text = re.sub(r'_comma_', ',', text)
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Ensure it's trained and available.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
         # Store token IDs directly in the tokenizer object for easy access in generation
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

    def _parse_step_from_filename(self, filename_part):
        match_epoch_step = re.search(r'epoch\d+_step(\d+)', filename_part)
        if match_epoch_step:
            return int(match_epoch_step.group(1))
        match_step = re.search(r'(\d+)', filename_part) # General step number
        if match_step:
            return int(match_step.group(1))
        if "final_step" in filename_part: # Specific case for "final_stepNUM"
            match_final = re.search(r'final_step(\d+)', filename_part)
            if match_final:
                return int(match_final.group(1))
        return 0

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
                # Adjusted regex to be more flexible with step naming (e.g., final_step, epochX_stepY, or just digits)
                pattern_str = rf"hrom_{re.escape(prefix_base)}_step([\w\d_]+)_(\d{{8}}_\d{{6}})\.pt"
                pattern = re.compile(pattern_str)

                checkpoints = []
                if not os.path.isdir(self.checkpoint_dir):
                    logging.info(f"No checkpoint directory at '{self.checkpoint_dir}'. Cannot load latest.")
                    return False

                for f_name in os.listdir(self.checkpoint_dir):
                     match = pattern.match(f_name)
                     if match:
                          filepath = os.path.join(self.checkpoint_dir, f_name)
                          checkpoints.append((filepath, os.path.getmtime(filepath)))

                if not checkpoints:
                    logging.info(f"No valid checkpoints found in '{self.checkpoint_dir}' matching pattern. Cannot load.")
                    return False

                checkpoints.sort(key=lambda x: x[1], reverse=True) # Sort by modification time, newest first
                latest_checkpoint_path, _ = checkpoints[0]
                logging.info(f"Loading latest checkpoint from: {latest_checkpoint_path}")

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
                    # Special handling for vocab_size, as current CONFIG might not be updated yet
                    if key == "vocab_size" and hasattr(model, 'embed') and model.embed.num_embeddings != loaded_val:
                         logging.warning(f"Checkpoint vocab_size ({loaded_val}) differs from initial model embed layer ({model.embed.num_embeddings}). This might be fine if vocab_size is updated from tokenizer later.")
                    elif key == "vocab_size": # If model not fully built or vocab_size check already covered
                        pass
                    elif loaded_val != current_val:
                        mismatched_keys.append((key, loaded_val, current_val))

                if mismatched_keys:
                    logging.warning("--- CONFIG MISMATCH DETECTED (Loading Checkpoint) ---")
                    for key, loaded_val, current_val in mismatched_keys:
                        logging.warning(f"  - {key}: Checkpoint='{loaded_val}', Current='{current_val}'")
                    logging.warning("Proceeding with loading, but this may impact model performance or cause errors if critical arch params changed.")
            else:
                logging.warning("Checkpoint does not contain configuration info. Cannot check for mismatches.")

            # Update CONFIG's vocab_size from checkpoint if available and different from initial, before loading state_dict
            if "vocab_size" in loaded_config and CONFIG["vocab_size"] != loaded_config["vocab_size"]:
                logging.info(f"Updating CONFIG['vocab_size'] from {CONFIG['vocab_size']} to {loaded_config['vocab_size']} based on checkpoint.")
                CONFIG["vocab_size"] = loaded_config["vocab_size"]
            model.load_state_dict(checkpoint['model'], strict=True) # Use strict=True for inference
            logging.info(f"Model weights loaded successfully from {latest_checkpoint_path}.")
            return True

        except FileNotFoundError:
            logging.error(f"Checkpoint file not found at '{latest_checkpoint_path}'.")
            return False
        except RuntimeError as e:
            logging.error(f"Failed to load model state_dict: {e}. Model architecture may have changed or vocab_size mismatch.")
            return False
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}.", exc_info=True)
            return False


def generate_response(model, tokenizer, prompt_text, max_new_tokens=100, temperature=0.7, top_k=50):
    model.eval()
    device = next(model.parameters()).device

    # Prepare prompt
    # Example structure: <s> <user> Hello </s> <assistant>
    # If prompt_text is "Hello", it becomes "<s> <user> Hello </s> <assistant>"
    # Tokenizer should not add BOS/EOS by default via encode for manual control here.
    
    # Clean the prompt text (optional, depends on how tokenizer was trained)
    # cleaned_prompt = TokenizerLoader()._clean_text(prompt_text) # If cleaning is desired
    cleaned_prompt = prompt_text

    # Construct the input sequence
    # Start with BOS, then user token, then prompt, then EOS, then assistant token
    prompt_ids = tokenizer.encode(cleaned_prompt, add_special_tokens=False).ids
    
    input_ids_list = [tokenizer.bos_id, tokenizer.user_id] + prompt_ids + [tokenizer.eos_id, tokenizer.assistant_id]
    
    generated_ids = list(input_ids_list)
    logging.debug(f"Initial input to model (tokenized): {generated_ids}")
    logging.debug(f"Initial input to model (decoded): '{tokenizer.decode(generated_ids)}'")


    with torch.no_grad():
        for step in range(max_new_tokens):
            # Trim input_ids to max_seq_len if they exceed it
            current_input_ids_trimmed = generated_ids[-CONFIG["max_seq_len"]:]
            input_tensor = torch.tensor([current_input_ids_trimmed], dtype=torch.long, device=device)
            
            # Create attention mask for padding (if any, though unlikely with trimming here for generation)
            # For generation, typically all tokens in current_input_ids_trimmed are attended to.
            attention_mask = torch.ones_like(input_tensor, device=device)

            try:
                logits = model(input_tensor, attention_mask=attention_mask) # Model returns only logits
                next_token_logits = logits[:, -1, :] # Logits for the last token position
            except Exception as e:
                 logging.error(f"Model forward pass failed during generation: {e}", exc_info=True)
                 break

            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k filtering
            if top_k > 0 and top_k < next_token_logits.size(-1):
                v, _ = torch.topk(next_token_logits, top_k, dim=-1)
                threshold_val = v[:, -1].unsqueeze(-1)
                next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold_val, -float('Inf'))

            probs = torch.softmax(next_token_logits, dim=-1)
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                 logging.warning(f"NaN/Inf detected in probabilities at step {step}. Using uniform distribution as fallback.")
                 probs = torch.ones_like(probs) / probs.size(-1)

            next_token_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_token_id)

            if next_token_id == tokenizer.eos_id:
                logging.debug(f"EOS token ({tokenizer.eos_id}) generated at step {step+1}. Stopping.")
                break
            if step == max_new_tokens - 1:
                 logging.debug("Max new tokens reached.")
                 if generated_ids[-1] != tokenizer.eos_id: # Append EOS if not already there
                     generated_ids.append(tokenizer.eos_id)

    # Extract only the generated response part (after the initial prompt + assistant cue)
    response_ids = generated_ids[len(input_ids_list):]
    
    # Decode, skip special tokens like <s>, </s>, <user>, <assistant> in the final output string
    decoded_text = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    return decoded_text


if __name__ == "__main__":
    logging.info("--- HROM-MoE Inference ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Tokenizer
    logging.info("Loading tokenizer...")
    try:
        tokenizer_loader = TokenizerLoader()
        tokenizer = tokenizer_loader.get_tokenizer()
        # Update CONFIG vocab_size if tokenizer's vocab size is different
        # This is crucial before model initialization if checkpoint doesn't provide vocab_size or if loading fresh
        if CONFIG["vocab_size"] != tokenizer.get_vocab_size():
            logging.info(f"Updating CONFIG['vocab_size'] from {CONFIG['vocab_size']} to {tokenizer.get_vocab_size()} based on tokenizer.")
            CONFIG["vocab_size"] = tokenizer.get_vocab_size()
        logging.info(f"Tokenizer loaded. Vocab size: {CONFIG['vocab_size']}")
    except Exception as e:
        logging.error(f"Failed to load tokenizer: {e}", exc_info=True)
        exit()

    # 2. Initialize Model
    logging.info("Initializing HROM-MoE model...")
    model = HROM().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model initialized. Total params: {total_params/1e6:.2f}M")


    # 3. Load Checkpoint
    logging.info("Loading model checkpoint...")
    checkpoint_loader = CheckpointLoader()
    # You can specify a direct path to a checkpoint if you don't want the "latest":
    # specific_checkpoint_file = "checkpoints_moe/hrom_moe_stepYOUR_STEP_HERE_TIMESTAMP.pt"
    # loaded_successfully = checkpoint_loader.load_latest(model, checkpoint_path=specific_checkpoint_file)
    loaded_successfully = checkpoint_loader.load_latest(model) # Loads latest from CONFIG["checkpoint_dir"]

    if not loaded_successfully:
        logging.error("Could not load model checkpoint. Exiting.")
        exit()
    
    model.to(device) # Ensure model is on the correct device after loading

    # 4. Interactive Chat Loop
    logging.info("Model ready. Starting interactive chat session.")
    logging.info("Type 'quit', 'exit', or 'bye' to end the session.")
    print("\n--- HROM Chatbot ---")
    history_turns = [] # Optional: to maintain a short history for context

    while True:
        try:
            user_prompt = input("You: ")
            if user_prompt.lower() in ["quit", "exit", "bye"]:
                print("HROM: Goodbye!")
                break
            if not user_prompt.strip():
                continue

            # Basic history mechanism (optional, can be made more sophisticated)
            # For this simple version, we are not explicitly feeding multi-turn history back into the prompt_text.
            # The `generate_response` function currently takes a single `prompt_text`.
            # To handle history, you would need to format `prompt_text` to include previous turns.
            # E.g., prompt_text = "<s> <user> Hi </s> <assistant> Hello! </s> <user> How are you? </s>"
            # For now, each input is treated as a new conversation start.

            response = generate_response(model, tokenizer, user_prompt,
                                         max_new_tokens=150, temperature=0.7, top_k=40)
            print(f"HROM: {response}")

        except KeyboardInterrupt:
            print("\nHROM: Session interrupted. Goodbye!")
            break
        except Exception as e:
            logging.error(f"An error occurred during chat: {e}", exc_info=True)
            print("HROM: Sorry, an error occurred.")
