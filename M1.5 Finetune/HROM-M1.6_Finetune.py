# HROM-M1.10_Finetune_Standalone_IndentFix.py
"""
Standalone fine-tuning script with the IndentationError fixed in the
HROMAttention class. This version should now execute correctly on Windows
(with torch.compile disabled).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import math

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
BASE_CONFIG = { "dim": 768, "n_layers": 8, "n_heads": 8, "ff_dim": 2048, "dropout": 0.1, "max_seq_len": 512, "vocab_size": 32000, "num_experts": 8, "top_k_experts": 2, "moe_load_balancing_coeff": 0.01 }
FINETUNE_CONFIG = {
    "base_model_checkpoint": "checkpoints_moe/HROM-M1.pt",
    "tokenizer_path": os.path.join("tokenizer", "hrom_moe_tokenizer.json"),
    "finetuned_checkpoint_dir": "checkpoints_moe_finetuned_alpaca",
    "dataset_name": "yahma/alpaca-cleaned",
    "prompt_template": {
        "with_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "without_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
    },
    "max_seq_len": 512, "num_epochs": 2, "batch_size": 8, "grad_accum_steps": 4,
    "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_steps": 100, "max_grad_norm": 1.0,
    "use_torch_compile": False, "amp_dtype": torch.bfloat16, "num_workers": 12,
    "save_every_n_steps": 500,
}

# --- MODEL DEFINITION ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim): super().__init__(); inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim)); self.register_buffer("inv_freq", inv_freq)
    def forward(self, seq_len): t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq); freqs = torch.einsum("i, j -> i j", t, self.inv_freq); return torch.cat((freqs, freqs), dim=-1)
def rotate_half(x): x1, x2 = x.chunk(2, dim=-1); return torch.cat((-x2, x1), dim=-1)
def apply_rotary_pos_emb(pos, t): cos_pos = pos.cos().unsqueeze(0).unsqueeze(1); sin_pos = pos.sin().unsqueeze(0).unsqueeze(1); return (t * cos_pos) + (rotate_half(t) * sin_pos)
class SwiGLU(nn.Module):
    def forward(self, x): x, gate = x.chunk(2, dim=-1); return x * F.gelu(gate)

class HROMAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.n_heads = config["dim"], config["n_heads"]
        self.head_dim = self.dim // self.n_heads
        self.qkv = nn.Linear(self.dim, 3 * self.dim)
        self.proj = nn.Linear(self.dim, self.dim)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        pos = self.rotary(T)
        q, k = apply_rotary_pos_emb(pos, q), apply_rotary_pos_emb(pos, k)
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # --- CORRECTED INDENTATION ---
        if mask is not None:
            attn_scores = attn_scores + mask
            
        attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(x.dtype)
        attn_probs = self.dropout(attn_probs)
        output = (attn_probs @ v).transpose(1, 2).reshape(B, T, self.dim)
        return self.proj(output)

class Expert(nn.Module):
    def __init__(self, config): super().__init__(); self.fc1 = nn.Linear(config["dim"], 2 * config["ff_dim"]); self.activation, self.fc2 = SwiGLU(), nn.Linear(config["ff_dim"], config["dim"])
    def forward(self, x): return self.fc2(self.activation(self.fc1(x)))
class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__(); self.num_experts, self.top_k, self.load_balancing_coeff = config["num_experts"], config["top_k_experts"], config["moe_load_balancing_coeff"]
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)]); self.gate = nn.Linear(config["dim"], self.num_experts)
    def forward(self, x):
        B, T, C = x.shape; x_reshaped = x.reshape(-1, C); gate_logits = self.gate(x_reshaped); weights, indices = torch.topk(gate_logits, self.top_k, dim=-1); weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype); final_output = torch.zeros_like(x_reshaped)
        for i in range(self.num_experts):
            token_mask, top_k_pos = torch.where(indices == i)
            if token_mask.numel() > 0: expert_inputs = x_reshaped[token_mask]; expert_weights = weights[token_mask, top_k_pos].unsqueeze(-1); expert_output = self.experts[i](expert_inputs) * expert_weights; final_output.index_add_(0, token_mask, expert_output)
        router_probs = F.softmax(gate_logits, dim=-1, dtype=torch.float); chosen_expert_mask = torch.zeros_like(router_probs); chosen_expert_mask.scatter_(1, indices, 1); fraction_tokens_per_expert = chosen_expert_mask.mean(dim=0); mean_router_probs_per_expert = router_probs.mean(dim=0); load_balancing_loss = self.load_balancing_coeff * self.num_experts * torch.sum(fraction_tokens_per_expert * mean_router_probs_per_expert); return final_output.reshape(B, T, C), load_balancing_loss
class HROMBlock(nn.Module):
    def __init__(self, config): super().__init__(); self.attn, self.moe_layer = HROMAttention(config), MoELayer(config); self.norm1, self.norm2 = nn.LayerNorm(config["dim"]), nn.LayerNorm(config["dim"]); self.dropout = nn.Dropout(config["dropout"])
    def forward(self, x, mask=None): x = x + self.dropout(self.attn(self.norm1(x), mask)); ff_output, moe_aux_loss = self.moe_layer(self.norm2(x)); x = x + self.dropout(ff_output); return x, moe_aux_loss
class HROM(nn.Module):
    def __init__(self, config): super().__init__(); self.config = config; self.embed = nn.Embedding(config["vocab_size"], config["dim"]); self.blocks = nn.ModuleList([HROMBlock(config) for _ in range(config["n_layers"])]); self.norm, self.head = nn.LayerNorm(config["dim"]), nn.Linear(config["dim"], config["vocab_size"]); self.dropout = nn.Dropout(config["dropout"])
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape; x = self.dropout(self.embed(input_ids)); causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device, dtype=torch.bool), diagonal=1)
        combined_mask = causal_mask | (attention_mask[:, None, :] == 0) if attention_mask is not None else causal_mask
        attn_mask = torch.zeros(B, 1, T, T, device=input_ids.device, dtype=x.dtype).masked_fill_(combined_mask.unsqueeze(1), float('-inf'))
        total_moe_aux_loss = 0.0
        for block in self.blocks: x, block_moe_aux_loss = block(x, attn_mask); total_moe_aux_loss += block_moe_aux_loss
        x = self.norm(x); logits = self.head(x); avg_moe_aux_loss = total_moe_aux_loss / self.config["n_layers"] if self.config["n_layers"] > 0 else 0.0
        return logits, avg_moe_aux_loss

# --- FINE-TUNING LOGIC ---
# (The rest of the script is unchanged and should be correct)
# ... (InstructionDataset, FineTuner, load_pretrained_model, main) ...
class InstructionDataset(Dataset):
    def __init__(self, config, tokenizer): self.config, self.tokenizer = config, tokenizer; self.max_length, self.prompt_template = config["max_seq_len"], config["prompt_template"]; self.pad_id, self.ignore_index = tokenizer.token_to_id("<pad>"), -100; self.processed_data = self._load_and_process_data()
    def _load_and_process_data(self):
        logging.info(f"Loading '{self.config['dataset_name']}'..."); dataset = load_dataset(self.config['dataset_name'], split="train"); processed_data = []
        for item in tqdm(dataset, desc="Processing dataset"):
            instruction, input_text, output_text = item.get('instruction', ''), item.get('input', ''), item.get('output', '')
            prompt_text = self.prompt_template["with_input"].format(instruction=instruction, input=input_text) if input_text else self.prompt_template["without_input"].format(instruction=instruction)
            full_text_with_tokens = "<s>" + prompt_text + output_text + "</s>"
            tokenized_full = self.tokenizer.encode(full_text_with_tokens, add_special_tokens=False)
            if len(tokenized_full.ids) > self.max_length: continue
            tokenized_prompt = self.tokenizer.encode("<s>" + prompt_text, add_special_tokens=False)
            processed_data.append({"input_ids": tokenized_full.ids, "prompt_len": len(tokenized_prompt.ids)})
        logging.info(f"Finished processing. Kept {len(processed_data)}/{len(dataset)} samples."); return processed_data
    def __len__(self): return len(self.processed_data)
    def __getitem__(self, idx): item = self.processed_data[idx]; input_ids, prompt_len = item["input_ids"], item["prompt_len"]; inputs = torch.tensor(input_ids[:-1], dtype=torch.long); labels = torch.tensor(input_ids[1:], dtype=torch.long); labels[:prompt_len-1] = self.ignore_index; return {"input_ids": inputs, "labels": labels}
    def collate_fn(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch); inputs_padded = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long); labels_padded = torch.full((len(batch), max_len), self.ignore_index, dtype=torch.long); attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, item in enumerate(batch): seq_len = len(item["input_ids"]); inputs_padded[i, :seq_len], labels_padded[i, :seq_len] = item["input_ids"], item["labels"]; attention_mask[i, :seq_len] = 1
        return {"input_ids": inputs_padded, "labels": labels_padded, "attention_mask": attention_mask}
class FineTuner:
    def __init__(self, config, model, tokenizer, dataloader):
        self.config, self.model, self.tokenizer, self.dataloader = config, model, tokenizer, dataloader; self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); self.amp_dtype = config.get("amp_dtype", torch.float16); self.optimizer, self.scheduler = self._setup_optimizer_and_scheduler(); self.criterion = nn.CrossEntropyLoss(ignore_index=-100); self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda')); self.model.to(self.device)
        if config["use_torch_compile"] and hasattr(torch, "compile"):
            logging.info(f"Attempting to apply torch.compile...")
            try: self.model = torch.compile(self.model); logging.info("torch.compile() successful.")
            except Exception as e: logging.warning(f"torch.compile() failed: {e}. Running in eager mode.")
        os.makedirs(config["finetuned_checkpoint_dir"], exist_ok=True); self.global_step = 0
    def _setup_optimizer_and_scheduler(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"], fused=(self.device.type == 'cuda')); num_training_steps = len(self.dataloader) // self.config["grad_accum_steps"] * self.config["num_epochs"]
        def lr_lambda(current_step):
            if current_step < self.config["warmup_steps"]: return float(current_step) / float(max(1, self.config["warmup_steps"]))
            progress = float(current_step - self.config["warmup_steps"]) / float(max(1, num_training_steps - self.config["warmup_steps"])); return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return optimizer, LambdaLR(optimizer, lr_lambda)
    def _save_checkpoint(self, name):
        save_path = os.path.join(self.config["finetuned_checkpoint_dir"], name); state_dict = self.model.state_dict(); torch.save({"model": state_dict, "config": self.config}, save_path); logging.info(f"Saved checkpoint to {save_path}")
    def run(self):
        logging.info("Starting fine-tuning..."); self.model.train(); optimizer_step = 0
        for epoch in range(self.config["num_epochs"]):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{self.config['num_epochs']}")
            for i, batch in enumerate(pbar):
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=(self.device.type == 'cuda')):
                    outputs, moe_aux_loss = self.model(batch["input_ids"].to(self.device), attention_mask=batch["attention_mask"].to(self.device))
                    main_loss = self.criterion(outputs.view(-1, outputs.size(-1)), batch["labels"].to(self.device).view(-1))
                    total_loss = main_loss + moe_aux_loss
                scaled_loss = self.scaler.scale(total_loss / self.config["grad_accum_steps"]); scaled_loss.backward()
                if (i + 1) % self.config["grad_accum_steps"] == 0:
                    self.scaler.unscale_(self.optimizer); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["max_grad_norm"]); self.scaler.step(self.optimizer); self.scaler.update(); self.scheduler.step(); self.optimizer.zero_grad(set_to_none=True); optimizer_step += 1
                pbar.set_postfix({"loss": f"{main_loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})
                if self.config["save_every_n_steps"] > 0 and optimizer_step > 0 and optimizer_step % self.config["save_every_n_steps"] == 0: self._save_checkpoint(f"checkpoint_step_{optimizer_step}.pt")
            self._save_checkpoint(f"finetuned_epoch_{epoch + 1}.pt")
        logging.info("Fine-tuning finished.")

# --- MAIN EXECUTION ---
def load_pretrained_model(model_config, fine_tune_config):
    logging.info(f"Loading tokenizer from: {fine_tune_config['tokenizer_path']}");
    if not os.path.exists(fine_tune_config['tokenizer_path']): logging.error("Tokenizer not found."); return None, None
    tokenizer = Tokenizer.from_file(fine_tune_config['tokenizer_path'])
    if model_config['vocab_size'] != tokenizer.get_vocab_size(): logging.warning("Updating vocab_size to match tokenizer."); model_config['vocab_size'] = tokenizer.get_vocab_size()
    logging.info("Instantiating HROM model..."); model = HROM(config=model_config)
    checkpoint_path = fine_tune_config['base_model_checkpoint']; logging.info(f"Loading weights from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path): logging.error(f"Checkpoint not found at {checkpoint_path}."); return None, None
    try: checkpoint = torch.load(checkpoint_path, map_location='cpu'); model.load_state_dict(checkpoint['model'], strict=True); logging.info("Weights loaded successfully.")
    except Exception as e: logging.error(f"Error loading weights: {e}"); return None, None
    return model, tokenizer
def main():
    model, tokenizer = load_pretrained_model(BASE_CONFIG, FINETUNE_CONFIG)
    if not model or not tokenizer: return
    dataset = InstructionDataset(FINETUNE_CONFIG, tokenizer)
    dataloader = DataLoader(dataset, batch_size=FINETUNE_CONFIG["batch_size"], collate_fn=dataset.collate_fn, shuffle=True, num_workers=FINETUNE_CONFIG["num_workers"], pin_memory=True, prefetch_factor=2 if FINETUNE_CONFIG["num_workers"] > 0 else None)
    tuner = FineTuner(FINETUNE_CONFIG, model, tokenizer, dataloader)
    tuner.run()

if __name__ == "__main__":
    main()