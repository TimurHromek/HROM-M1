import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, disable_caching
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import math
import re
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict
import logging
import random

# --- Rich TUI Imports ---
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, SpinnerColumn
from rich.text import Text
from rich.table import Table
from rich.logging import RichHandler
from rich.columns import Columns
from rich.markup import escape

# Setup Rich Console and Logging
console = Console(force_terminal=True, tab_size=4)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False, markup=True)]
)


# Configuration
CONFIG = {
    "dim": 768,
    "n_layers": 8,
    "n_heads": 8,
    "ff_dim": 2048,
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 16,
    "checkpoint_interval": 2000,
    "debug_interval": 400, # Interval for logging metrics, not for debug generation anymore
    "datasets": ["daily_dialog", "empathetic_dialogues", "blended_skill_talk", "AlekseyKorshuk/persona-chat", "papahawk/conversational-01"],
    "tokenizer_name": "hrom_moe_tokenizer.json",
    "checkpoint_dir": "checkpoints_moe",
    "vocab_size": 32000,
    "tokenizer_train_samples_per_dataset": 50000,
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "max_turns": 8,
    "max_checkpoints": 5,
    "num_epochs": 30,
    "grad_accum_steps": 8,
    "num_experts": 8,
    "top_k_experts": 2,
    "moe_load_balancing_coeff": 0.01
}

if CONFIG["top_k_experts"] > CONFIG["num_experts"]:
    logging.warning(f"top_k_experts ({CONFIG['top_k_experts']}) > num_experts ({CONFIG['num_experts']}). Setting top_k_experts to num_experts.")
    CONFIG["top_k_experts"] = CONFIG["num_experts"]


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        if seq_len == 0:
             return torch.empty((0, self.inv_freq.shape[0] * 2), device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i, j -> i j", t, self.inv_freq)
        if freqs.shape[0] != seq_len:
             freqs = freqs.reshape(seq_len, -1)
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
         logging.warning(f"RoPE Warning: pos sequence length ({pos_seq_len}) is shorter than tensor sequence length ({tensor_seq_len}). Using truncated tensor length for RoPE.")
         t_rotated = t[:, :, :pos_seq_len, :]
         pos_truncated = pos[:, :, :pos_seq_len, :]
         cos_pos = pos_truncated.cos()
         sin_pos = pos_truncated.sin()
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
    def __init__(self, dim, ff_dim, num_experts, top_k, load_balancing_coeff):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancing_coeff = load_balancing_coeff
        self.experts = nn.ModuleList([Expert(dim, ff_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_reshaped = x.reshape(-1, dim)
        num_tokens = x_reshaped.shape[0]
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

        chosen_expert_mask = torch.zeros_like(gate_probs, device=x.device)
        chosen_expert_mask.scatter_(1, top_k_indices, 1)
        fraction_tokens_per_expert = chosen_expert_mask.mean(dim=0)
        mean_router_probs_per_expert = gate_probs.mean(dim=0)
        load_balancing_loss = self.load_balancing_coeff * self.num_experts * \
                              torch.sum(fraction_tokens_per_expert * mean_router_probs_per_expert)
        final_output = final_output.reshape(batch_size, seq_len, dim)
        return final_output, load_balancing_loss


class HROMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = HROMAttention()
        self.moe_layer = MoELayer(
            dim=CONFIG["dim"],
            ff_dim=CONFIG["ff_dim"],
            num_experts=CONFIG["num_experts"],
            top_k=CONFIG["top_k_experts"],
            load_balancing_coeff=CONFIG["moe_load_balancing_coeff"]
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
        ff_output, moe_aux_loss = self.moe_layer(normed_x2)
        x = residual2 + self.dropout(ff_output)
        return x, moe_aux_loss


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
        combined_mask = None
        causal_mask = torch.triu(torch.ones(T, T, device=input_ids.device) * float('-inf'), diagonal=1)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        if attention_mask is not None:
            pad_mask = (1.0 - attention_mask.to(torch.float32)) * torch.finfo(torch.float32).min
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
            combined_mask = combined_mask + pad_mask
        combined_mask = combined_mask.to(dtype=x.dtype)

        total_moe_aux_loss = 0.0
        for block in self.blocks:
            x, block_moe_aux_loss = block(x, combined_mask)
            total_moe_aux_loss += block_moe_aux_loss
        x = self.norm(x)
        logits = self.head(x)
        avg_moe_aux_loss = total_moe_aux_loss / CONFIG["n_layers"] if CONFIG["n_layers"] > 0 else 0.0
        return logits, avg_moe_aux_loss


class TokenizerTrainer:
    def __init__(self):
        self.tokenizer = Tokenizer(models.BPE())
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
        self.tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
        self.tokenizer_dir = os.path.dirname(self.tokenizer_path)

    def _clean_text(self, text):
        text = str(text)
        text = re.sub(r'_comma_', ',', text)
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def train(self, dataset_names):
        logging.info("Starting tokenizer training...")
        text_samples = []
        samples_per_dataset = CONFIG['tokenizer_train_samples_per_dataset']

        if "daily_dialog" in dataset_names:
            logging.info(f"Loading daily_dialog for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                dd_dataset = load_dataset("daily_dialog", split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info("Processing daily_dialog...")
                for entry in dd_dataset:
                    formatted_dialogue = []
                    dialogue = entry['dialog'][:CONFIG["max_turns"]]
                    for i, utterance in enumerate(dialogue):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                             formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process daily_dialog for tokenizer: {e}")

        if "empathetic_dialogues" in dataset_names:
            logging.info(f"Loading empathetic_dialogues for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 3}]", trust_remote_code=True)
                logging.info("Processing empathetic_dialogues...")
                grouped_by_conv = defaultdict(list)
                for entry in ed_dataset: grouped_by_conv[entry['conv_id']].append(entry)
                processed_conv_count = 0
                for conv_id, entries in grouped_by_conv.items():
                    if processed_conv_count >= samples_per_dataset: break
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    formatted_dialogue = []
                    if sorted_entries[0]['context']:
                         cleaned_context = self._clean_text(sorted_entries[0]['context'])
                         if cleaned_context: formatted_dialogue.append(f"<user> {cleaned_context}")
                    last_role = '<user>' if formatted_dialogue else None
                    for entry in sorted_entries:
                        cleaned_utterance = self._clean_text(entry['utterance'])
                        if cleaned_utterance:
                            current_role = '<assistant>' if last_role == '<user>' else '<user>'
                            formatted_dialogue.append(f"{current_role} {cleaned_utterance}")
                            last_role = current_role
                    formatted_dialogue = formatted_dialogue[:CONFIG["max_turns"]]
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
                        processed_conv_count += 1
            except Exception as e:
                logging.error(f"Failed to load or process empathetic_dialogues for tokenizer: {e}")

        if "blended_skill_talk" in dataset_names:
            logging.info(f"Loading blended_skill_talk for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                bst_dataset = load_dataset("blended_skill_talk", split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info("Processing blended_skill_talk...")
                for entry in bst_dataset:
                    formatted_dialogue = []
                    dialogue_turns_raw = list(entry['previous_utterance'])
                    if entry.get('free_turker_utterance'): dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'): dialogue_turns_raw.append(entry['guided_turker_utterance'])
                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]]
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance: formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue: text_samples.append(" </s> ".join(formatted_dialogue))
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for tokenizer: {e}")

        if "AlekseyKorshuk/persona-chat" in dataset_names:
            pc_dataset_name = "AlekseyKorshuk/persona-chat"
            logging.info(f"Loading {pc_dataset_name} for tokenizer training (max {samples_per_dataset} dialogues)...")
            try:
                pc_dataset = load_dataset(pc_dataset_name, split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info(f"Processing {pc_dataset_name}...")
                for entry in pc_dataset:
                    if 'utterances' in entry and entry['utterances']:
                        history = entry['utterances'][-1]['history'][:CONFIG["max_turns"]]
                        formatted_dialogue = []
                        for i, utterance in enumerate(history):
                             role = "<user>" if i % 2 == 0 else "<assistant>"
                             cleaned_utterance = self._clean_text(utterance)
                             if cleaned_utterance: formatted_dialogue.append(f"{role} {cleaned_utterance}")
                        if formatted_dialogue: text_samples.append(" </s> ".join(formatted_dialogue))
                    else: logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry}")
            except Exception as e:
                logging.error(f"Failed to load or process {pc_dataset_name} for tokenizer: {e}")

        if "papahawk/conversational-01" in dataset_names:
            ph_dataset_name = "papahawk/conversational-01"
            logging.info(f"Loading {ph_dataset_name} for tokenizer training (max {samples_per_dataset} entries)...")
            try:
                ph_dataset = load_dataset(ph_dataset_name, split=f"train[:{samples_per_dataset}]", trust_remote_code=True)
                logging.info(f"Processing {ph_dataset_name} for tokenizer...")
                for entry in ph_dataset:
                    instruction = self._clean_text(entry.get('instruction', ''))
                    response = self._clean_text(entry.get('response', ''))
                    formatted_pair = []
                    if instruction: formatted_pair.append(f"<user> {instruction}")
                    if response and instruction: formatted_pair.append(f"<assistant> {response}")
                    if len(formatted_pair) == 2: text_samples.append(" </s> ".join(formatted_pair))
                    elif len(formatted_pair) == 1: text_samples.append(formatted_pair[0])
            except Exception as e:
                logging.error(f"Failed to load or process {ph_dataset_name} for tokenizer: {e}")

        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading and paths.")

        os.makedirs(self.tokenizer_dir, exist_ok=True)
        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(vocab_size=CONFIG["vocab_size"], special_tokens=self.special_tokens, min_frequency=2, show_progress=True)
        def text_iterator():
            for sample in text_samples: yield sample
        self.tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(text_samples))
        eos_token_id = self.tokenizer.token_to_id("</s>") or self.tokenizer.token_to_id("<pad>") or 0
        self.tokenizer.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>", special_tokens=[("</s>", eos_token_id)])
        logging.info(f"Saving tokenizer to {self.tokenizer_path}")
        self.tokenizer.save(self.tokenizer_path)
        logging.info("Tokenizer training complete.")

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Train tokenizer first.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
         return tokenizer


class CombinedChatDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        if None in [self.pad_id, self.eos_id, self.bos_id, self.user_id, self.assistant_id]:
            missing = [n for n,v in zip(["pad","eos","bos","user","assistant"], [self.pad_id, self.eos_id, self.bos_id, self.user_id, self.assistant_id]) if v is None]
            raise ValueError(f"Tokenizer missing critical special token IDs: {missing}.")
        self.max_length = CONFIG["max_seq_len"]
        self._clean_text = TokenizerTrainer()._clean_text
        self.all_processed_conversations = []

        if "daily_dialog" in CONFIG["datasets"]:
            logging.info("Loading and processing daily_dialog dataset...")
            try:
                dd_dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)
                for entry in dd_dataset:
                    conv = []
                    for i, utt in enumerate(entry['dialog'][:CONFIG["max_turns"]]):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned = self._clean_text(utt)
                        if cleaned: conv.append({'role': role, 'text': cleaned})
                    if conv: self.all_processed_conversations.append(conv)
            except Exception as e: logging.error(f"Failed to load/process daily_dialog: {e}")

        if "empathetic_dialogues" in CONFIG["datasets"]:
            logging.info("Loading and processing empathetic_dialogues dataset...")
            try:
                ed_dataset = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
                conv_gr = defaultdict(list)
                for entry in ed_dataset: conv_gr[entry['conv_id']].append(entry)
                for cid, entries in conv_gr.items():
                    conv = []
                    s_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    if s_entries[0]['context']:
                        ctx = self._clean_text(s_entries[0]['context'])
                        if ctx: conv.append({'role': '<user>', 'text': ctx})
                    lr = conv[-1]['role'] if conv else None
                    for entry in s_entries:
                        txt = self._clean_text(entry['utterance'])
                        if not txt: continue
                        cr = '<assistant>' if lr == '<user>' else '<user>'
                        conv.append({'role': cr, 'text': txt}); lr = cr
                    if conv: self.all_processed_conversations.append(conv[:CONFIG["max_turns"]])
            except Exception as e: logging.error(f"Failed to load/process empathetic_dialogues: {e}")

        if "blended_skill_talk" in CONFIG["datasets"]:
            logging.info("Loading and processing blended_skill_talk dataset...")
            try:
                bst_dataset = load_dataset("blended_skill_talk", split="train", trust_remote_code=True)
                for entry in bst_dataset:
                    conv = []
                    raw_turns = list(entry['previous_utterance'])
                    if entry.get('free_turker_utterance'): raw_turns.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'): raw_turns.append(entry['guided_turker_utterance'])
                    for i, utt in enumerate(raw_turns[:CONFIG["max_turns"]]):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned = self._clean_text(utt)
                        if cleaned: conv.append({'role': role, 'text': cleaned})
                    if conv: self.all_processed_conversations.append(conv)
            except Exception as e: logging.error(f"Failed to load/process blended_skill_talk: {e}")

        if "AlekseyKorshuk/persona-chat" in CONFIG["datasets"]:
            logging.info("Loading and processing AlekseyKorshuk/persona-chat dataset...")
            try:
                pc_dataset = load_dataset("AlekseyKorshuk/persona-chat", split="train", trust_remote_code=True)
                for entry in pc_dataset:
                    conv = []
                    if 'utterances' in entry and entry['utterances']:
                        hist = entry['utterances'][-1]['history'][:CONFIG["max_turns"]]
                        for i, utt in enumerate(hist):
                            role = "<user>" if i % 2 == 0 else "<assistant>"
                            cleaned = self._clean_text(utt)
                            if cleaned: conv.append({'role': role, 'text': cleaned})
                    if conv: self.all_processed_conversations.append(conv)
            except Exception as e: logging.error(f"Failed to load/process AlekseyKorshuk/persona-chat: {e}")

        if "papahawk/conversational-01" in CONFIG["datasets"]:
            logging.info("Loading and processing papahawk/conversational-01 dataset...")
            try:
                ph_dataset = load_dataset("papahawk/conversational-01", split="train", trust_remote_code=True)
                for entry in ph_dataset:
                    instr = self._clean_text(entry.get('instruction', ''))
                    resp = self._clean_text(entry.get('response', ''))
                    if instr and resp:
                        self.all_processed_conversations.append([
                            {'role': '<user>', 'text': instr},
                            {'role': '<assistant>', 'text': resp}
                        ])
            except Exception as e: logging.error(f"Failed to load/process papahawk/conversational-01: {e}")


        logging.info(f"Total processed conversations from all datasets: {len(self.all_processed_conversations)}")
        if not self.all_processed_conversations:
             raise ValueError("No processed conversations were created from any dataset.")
        logging.info("Shuffling combined dataset...")
        random.shuffle(self.all_processed_conversations)

    def __len__(self):
        return len(self.all_processed_conversations)

    def __getitem__(self, idx):
        conversation = self.all_processed_conversations[idx]
        formatted_ids = [self.bos_id]
        for turn in conversation:
            role_id = self.user_id if turn['role'] == '<user>' else self.assistant_id
            try:
                utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids
            except Exception as e:
                 logging.error(f"Error encoding text at index {idx}, turn '{escape(str(turn))}': {e}")
                 utterance_ids = []
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                if len(formatted_ids) + 1 + 1 <= self.max_length:
                     formatted_ids.extend([role_id, self.eos_id])
                break
            formatted_ids.extend([role_id] + utterance_ids + [self.eos_id])

        if len(formatted_ids) > self.max_length:
             formatted_ids = formatted_ids[:self.max_length]
             if formatted_ids and (formatted_ids[-1] == self.user_id or formatted_ids[-1] == self.assistant_id):
                  formatted_ids.pop()
             if formatted_ids and formatted_ids[-1] != self.eos_id:
                 if len(formatted_ids) == self.max_length: formatted_ids[-1] = self.eos_id
                 elif len(formatted_ids) < self.max_length : formatted_ids.append(self.eos_id)

        if len(formatted_ids) < 2:
             logging.warning(f"Sequence at index {idx} is too short after processing (<2 tokens): {formatted_ids}. Skipping.")
             return None
        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]
        if len(input_ids) == 0:
            logging.warning(f"Sequence at index {idx} resulted in empty input_ids after slicing. Skipping.")
            return None
        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def collate_fn(batch, pad_token_id_from_dataset):
        batch = [item for item in batch if item is not None]
        if not batch:
            logging.warning("Collate_fn received an entirely empty batch after filtering Nones.")
            return None
        max_len = max(len(item["input_ids"]) for item in batch if "input_ids" in item) if batch else 0
        if max_len == 0:
            logging.warning("Collate_fn: max_len is 0 after processing batch items.")
            return None

        pad_id = pad_token_id_from_dataset
        inputs, labels, masks = [], [], []
        for item in batch:
            input_len = len(item["input_ids"])
            pad_len = max_len - input_len
            inputs.append(item["input_ids"] + [pad_id] * pad_len)
            labels.append(item["labels"] + [pad_id] * pad_len)
            masks.append([1] * input_len + [0] * pad_len)
        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)
        }

class HROMTrainer:
    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.use_amp = (self.device.type == "cuda" and hasattr(torch.cuda.amp, "GradScaler")) # Keep this check for general AMP availability
        self.amp_dtype = torch.bfloat16 if (self.use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        logging.info(f"Automatic Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}. Using dtype: {self.amp_dtype if self.use_amp else 'N/A'}")

        if self.use_amp:
            # Corrected GradScaler initialization
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            class DummyScaler:
                def __init__(self): pass
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
            self.scaler = DummyScaler()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=CONFIG["learning_rate"], betas=(0.9, 0.95),
            weight_decay=0.1, fused=(self.device.type == "cuda")
        )
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            logging.warning("<pad> token ID not found in tokenizer, using fallback ID: 0")
            self.pad_id = 0
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.base_lr = CONFIG["learning_rate"]
        self.warmup_steps = CONFIG["warmup_steps"]

    def _adjust_learning_rate(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else: lr = self.base_lr
        for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return lr

    def train_step(self, batch):
        autocast_context = torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)
        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            outputs, moe_aux_loss = self.model(input_ids, attention_mask=attention_mask)
            logits_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            main_loss = self.criterion(logits_flat.float(), labels_flat)
            total_loss = main_loss + moe_aux_loss
            scaled_loss = total_loss / CONFIG["grad_accum_steps"]

        self.scaler.scale(scaled_loss).backward()
        return main_loss.item(), moe_aux_loss.item()

    def clip_and_step(self, current_optimizer_step):
         current_lr = self._adjust_learning_rate(current_optimizer_step)
         self.scaler.unscale_(self.optimizer)
         torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
         self.scaler.step(self.optimizer)
         self.scaler.update()
         self.optimizer.zero_grad(set_to_none=True)
         return current_lr


class SafetyManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm", "die", "attack", "rape", "molest", "exploit", "terror"]
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        for word in self.bad_words:
             ids_s = tokenizer.encode(f" {word}", add_special_tokens=False).ids
             if ids_s: self.bad_word_ids.append(ids_s)
             ids_ns = tokenizer.encode(word, add_special_tokens=False).ids
             if ids_ns and ids_ns != ids_s: self.bad_word_ids.append(ids_ns)
             if not ids_s and not ids_ns: logging.warning(f"Could not encode bad word '{escape(word)}' - skipping.")
        self.eos_id = self.tokenizer.token_to_id("</s>") or 0
        self.bos_id = self.tokenizer.token_to_id("<s>") or 0
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        self.pad_id = self.tokenizer.token_to_id("<pad>") or 0

    def contains_sequence(self, tokens, seq):
        if not seq or not tokens or len(tokens) < len(seq): return False
        sl = len(seq)
        for i in range(len(tokens) - sl + 1):
            if tokens[i : i + sl] == seq: return True
        return False

    def content_filter(self, text_ids):
        if not isinstance(text_ids, list): return True
        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                try:
                    dw = self.tokenizer.decode(bad_ids)
                except Exception:
                    dw = "unknown (decoding error)"
                logging.warning(f"Unsafe content detected: Found sequence for '{escape(dw)}' (IDs: {bad_ids}). Blocking generation.")
                return False
        return True

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device
        decoded_bos = self.tokenizer.decode([self.bos_id])
        if decoded_bos and prompt.startswith(decoded_bos):
            prompt = prompt[len(decoded_bos):].strip()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        input_ids = [self.bos_id] + prompt_ids
        if self.assistant_id is not None:
            if not input_ids or input_ids[-1] not in [self.assistant_id, self.user_id, self.eos_id]:
                input_ids.append(self.assistant_id)
            elif input_ids and input_ids[-1] in [self.user_id, self.eos_id]:
                input_ids.append(self.assistant_id)
            elif not input_ids:
                 input_ids.extend([self.user_id, self.eos_id, self.assistant_id])
        else: logging.error("Assistant token ID is None. Cannot properly cue model for generation.")


        generated_ids = list(input_ids)
        with torch.no_grad():
            for step in range(max_new_tokens):
                current_ids_trimmed = generated_ids[-CONFIG["max_seq_len"]:]
                current_tensor = torch.tensor([current_ids_trimmed], device=device)
                attn_mask = torch.ones_like(current_tensor, device=device)
                try:
                    outputs, _ = self.model(current_tensor, attention_mask=attn_mask)
                    next_token_logits = outputs[:, -1, :]
                except Exception as e:
                     logging.error(f"Model forward pass failed during generation: {e}", exc_info=True); break
                if temperature > 0 and temperature != 1.0: next_token_logits /= temperature
                if top_k > 0 and top_k < next_token_logits.size(-1):
                    v, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < v[:, -1].unsqueeze(-1), -float('Inf'))
                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                     logging.warning(f"NaN/Inf in probabilities at step {step}. Using uniform fallback."); probs = torch.ones_like(probs) / probs.size(-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
                potential_seq_chk = generated_ids[len(input_ids):] + [next_token_id]
                if not self.content_filter(potential_seq_chk):
                    logging.warning(f"Unsafe token ID {next_token_id} ('{escape(self.tokenizer.decode([next_token_id]))}') blocked. Stopping."); break
                generated_ids.append(next_token_id)
                if next_token_id == self.eos_id: break
                if step == max_new_tokens - 1 and generated_ids[-1] != self.eos_id and self.eos_id is not None: generated_ids.append(self.eos_id)
        self.model.train()
        response_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    def debug_generation(self, prompt="<user> Tell me about your hobbies. </s>"):
         # This method now also returns prompt and response for UI display
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         
         original_prompt_for_return = str(prompt) # Capture before any modifications

         if not prompt.strip().startswith(("<user>", "<assistant>")):
             prompt = f"<user> {prompt.strip()}"
         if not prompt.strip().endswith("</s>"):
             prompt = f"{prompt.strip()} </s>"
         
         # The prompt sent to generate_safely will have <s> and <assistant> appended internally
         # For logging and UI, we use the user-facing prompt with <user>/<assistant> and </s>
         prompt_for_generation = prompt 

         generated_response = self.generate_safely(prompt_for_generation, max_new_tokens=60, temperature=0.7, top_k=50)
         
         logging.info(f"Prompt Sent (raw to model after formatting): '{escape(prompt_for_generation)}'")
         logging.info(f"Generated Response: '{escape(generated_response)}'")
         logging.info(f"--- End Debug Generation ---\n")
         
         return original_prompt_for_return, generated_response


class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def save(self, model, optimizer, step_info):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
        step_str = str(step_info).replace(" ", "_")
        filename = f"hrom_{prefix_base}_step{step_str}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step_info": step_info, "config": CONFIG}
        logging.info(f"Saving checkpoint to {path}...")
        try:
            torch.save(state, path)
            logging.info(f"Checkpoint saved successfully: {filename}")
            self._cleanup_old_checkpoints()
        except Exception as e: logging.error(f"Failed to save checkpoint '{path}': {e}", exc_info=True)

    def _parse_step_from_filename(self, filename_part):
        m_es = re.search(r'epoch\d+_step(\d+)', filename_part)
        if m_es: return int(m_es.group(1))
        m_s = re.search(r'(\d+)', filename_part)
        if m_s: return int(m_s.group(1))
        return 0

    def _cleanup_old_checkpoints(self):
        max_chk = CONFIG.get("max_checkpoints", 5)
        if max_chk <= 0: return
        try:
            p_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern = re.compile(rf"hrom_{re.escape(p_base)}_step([\w\d_]+)_(\d{{8}}_\d{{6}})\.pt")
            chks = []
            for f_name in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f_name)
                 if match: chks.append((os.path.join(self.checkpoint_dir, f_name), os.path.getmtime(os.path.join(self.checkpoint_dir, f_name))))
            chks.sort(key=lambda x: x[1])
            num_del = len(chks) - max_chk
            if num_del > 0:
                logging.info(f"Max checkpoints ({max_chk}) reached. Deleting {num_del} oldest ones.")
                for i in range(num_del):
                    try: os.remove(chks[i][0]); logging.info(f"Removed old checkpoint: {chks[i][0]}")
                    except OSError as e: logging.error(f"Error removing old checkpoint {chks[i][0]}: {e}")
        except Exception as e: logging.error(f"Error during checkpoint cleanup: {e}", exc_info=True)

    def load_latest(self, model, optimizer):
        try:
            p_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern = re.compile(rf"hrom_{re.escape(p_base)}_step([\w\d_]+)_(\d{{8}}_\d{{6}})\.pt")
            chks = []
            for f_name in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f_name)
                 if match: chks.append((os.path.join(self.checkpoint_dir, f_name), os.path.getmtime(os.path.join(self.checkpoint_dir, f_name)), match.group(1)))
            if not chks: logging.info(f"No valid checkpoints found in '{self.checkpoint_dir}'. Starting fresh."); return 0
            chks.sort(key=lambda x: x[1], reverse=True)
            latest_path, _, latest_step_str = chks[0]
            logging.info(f"Loading latest checkpoint from: {latest_path}")
            map_loc = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(latest_path, map_location=map_loc)

            loaded_cfg = checkpoint.get("config", {})
            crit_keys = ["dim","n_layers","n_heads","ff_dim","vocab_size","max_seq_len","tokenizer_name","num_experts","top_k_experts"]
            if loaded_cfg:
                mismatched = [(k, loaded_cfg.get(k), CONFIG.get(k)) for k in crit_keys if loaded_cfg.get(k) != CONFIG.get(k)]
                if mismatched:
                    logging.warning("--- CONFIG MISMATCH DETECTED (Loading Checkpoint) ---")
                    for k, lv, cv in mismatched: logging.warning(f"  - {k}: Checkpoint='{lv}', Current='{cv}'")
            else: logging.warning("Checkpoint does not contain config. Cannot check for mismatches.")

            try: model.load_state_dict(checkpoint['model'], strict=True)
            except RuntimeError as e: logging.error(f"Failed to load model state_dict: {e}. Starting fresh."); return 0
            try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
                 for state_val in optimizer.state.values():
                    for k, v in state_val.items():
                        if isinstance(v, torch.Tensor): state_val[k] = v.to(map_loc)
            except Exception as e: logging.warning(f"Could not load optimizer state_dict: {e}. Optimizer state reset."); optimizer.state = defaultdict(dict)

            step_info_ld = checkpoint.get('step_info', 0)
            start_opt_step = step_info_ld + 1 if isinstance(step_info_ld, int) else (self._parse_step_from_filename(step_info_ld) + 1 if self._parse_step_from_filename(step_info_ld) > 0 else 0)
            if isinstance(step_info_ld, str) and start_opt_step == 0 and "epoch" in step_info_ld.lower():
                 logging.warning(f"Loaded epoch checkpoint '{escape(step_info_ld)}' but could not parse optimizer step.")
            logging.info(f"Checkpoint loaded. Resuming from info '{escape(str(step_info_ld))}'. Next optimizer_step: {start_opt_step}.")
            return start_opt_step
        except FileNotFoundError: logging.info(f"No checkpoint directory '{self.checkpoint_dir}'. Starting fresh."); return 0
        except Exception as e: logging.error(f"Error loading checkpoint: {e}. Starting fresh.", exc_info=True); return 0


def train():
    logging.info("Starting HROM-MoE training process...")
    logging.info(f"Initial Configuration: {escape(str(CONFIG))}")

    tokenizer_trainer = TokenizerTrainer()
    if not os.path.exists(tokenizer_trainer.tokenizer_path):
        logging.info(f"Tokenizer '{CONFIG['tokenizer_name']}' not found. Training new tokenizer...")
        try: tokenizer_trainer.train(list(set(CONFIG["datasets"])))
        except Exception as e: logging.error(f"Tokenizer training error: {e}", exc_info=True); return
    else: logging.info(f"Loading existing tokenizer from {tokenizer_trainer.tokenizer_path}")

    try:
        tokenizer = tokenizer_trainer.get_tokenizer()
        CONFIG['pad_token_id'] = tokenizer.token_to_id("<pad>")
        CONFIG['bos_token_id'] = tokenizer.token_to_id("<s>")
        CONFIG['eos_token_id'] = tokenizer.token_to_id("</s>")
        if None in [CONFIG['pad_token_id'],CONFIG['bos_token_id'],CONFIG['eos_token_id'],tokenizer.token_to_id("<user>"),tokenizer.token_to_id("<assistant>")]:
            raise ValueError("Critical special tokens missing from tokenizer.")
        logging.info(f"Tokenizer loaded. Vocab: {tokenizer.get_vocab_size()}. PAD: {CONFIG['pad_token_id']}, BOS: {CONFIG['bos_token_id']}, EOS: {CONFIG['eos_token_id']}")
    except (FileNotFoundError, ValueError) as e: logging.error(f"Tokenizer loading error: {e}. Cannot continue.", exc_info=True); return

    logging.info("Initializing HROM-MoE model...")
    if CONFIG['vocab_size'] != tokenizer.get_vocab_size():
         logging.warning(f"CONFIG vocab_size ({CONFIG['vocab_size']}) != tokenizer ({tokenizer.get_vocab_size()}). Updating CONFIG.")
         CONFIG['vocab_size'] = tokenizer.get_vocab_size()
    model = HROM()
    params_info = f"Total params: {sum(p.numel() for p in model.parameters()):,} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M)"
    logging.info(f"HROM-MoE Model initialized. {params_info}")

    logging.info("Setting up dataset and dataloader...")
    try:
         dataset = CombinedChatDataset(tokenizer)
         if len(dataset) == 0: logging.error("Dataset empty. Cannot train."); return

         collate_wrapper = lambda batch: CombinedChatDataset.collate_fn(batch, dataset.pad_id)

         cpu_count = os.cpu_count() or 1
         num_workers = min(4, cpu_count // 2) if torch.cuda.is_available() else min(2, cpu_count // 2)
         num_workers = max(0, num_workers)
         logging.info(f"Using num_workers: {num_workers} for DataLoader.")

         dataloader = DataLoader(
             dataset, batch_size=CONFIG["batch_size"], collate_fn=collate_wrapper,
             shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
             prefetch_factor=2 if num_workers > 0 else None, drop_last=False
         )
    except Exception as e: logging.error(f"Dataset/Dataloader init error: {e}", exc_info=True); return

    trainer_obj = HROMTrainer(model, tokenizer)
    checkpoint_manager = CheckpointManager()
    safety = SafetyManager(model, tokenizer)
    start_optimizer_step = checkpoint_manager.load_latest(model, trainer_obj.optimizer)
    model.to(trainer_obj.device)
    logging.info(f"Starting/Resuming training from optimizer step {start_optimizer_step}")
    optimizer_step = start_optimizer_step
    accum_main_loss, accum_aux_loss = 0.0, 0.0

    batches_per_epoch = len(dataloader) if len(dataloader) > 0 else 1
    start_epoch = (optimizer_step * CONFIG["grad_accum_steps"]) // batches_per_epoch if batches_per_epoch > 0 else 0

    # Variables to store the last debug inference for the UI
    last_debug_prompt = "N/A (Pending first epoch-end inference)"
    last_debug_response = "N/A"

    # --- Rich TUI Setup ---
    layout = Layout(name="root")
    layout.split_column(
        Layout(Panel(Text("HROM-MoE Training Monitor", justify="center", style="bold white on blue")), name="header", size=3),
        Layout(name="progress_bars_layout", size=8),
        Layout(name="metrics_panel_layout", size=7),
        Layout(name="log_panel_layout", ratio=1) # This panel will hold the debug inference table
    )
    layout["progress_bars_layout"].split_row(
        Layout(name="overall_progress_panel"),
        Layout(name="epoch_progress_panel")
    )

    overall_progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console)
    epoch_progress = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeRemainingColumn(), TimeElapsedColumn(), console=console)

    metrics_table = Table.grid(expand=True, padding=(0, 1))
    metrics_table.add_column("Metric", style="cyan", no_wrap=True)
    metrics_table.add_column("Value", style="magenta")
    metrics_data = {
        "Epoch": f"-/{CONFIG['num_epochs']}", "Batch": f"-/{batches_per_epoch}",
        "Opt Step": str(optimizer_step), "LR": f"{trainer_obj.base_lr:.2e}",
        "Main Loss": "N/A", "Aux Loss": "N/A", "Total Loss": "N/A"
    }
    for key, value in metrics_data.items(): metrics_table.add_row(key, value)

    layout["overall_progress_panel"].update(Panel(overall_progress, title="Overall Progress (Epochs)"))
    layout["epoch_progress_panel"].update(Panel(epoch_progress, title="Current Epoch Progress (Batches)"))
    layout["metrics_panel_layout"].update(Panel(metrics_table, title="Live Metrics"))

    # Setup for the debug inference display panel
    debug_inference_table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
    debug_inference_table.add_column("Prompt", style="cyan", overflow="fold", ratio=1)
    debug_inference_table.add_column("Response", style="green", overflow="fold", ratio=2)
    debug_inference_table.add_row(escape(last_debug_prompt), escape(last_debug_response))
    layout["log_panel_layout"].update(Panel(debug_inference_table, title="[b]Last Debug Inference[/b]", border_style="dim green"))


    with Live(layout, console=console, refresh_per_second=4, vertical_overflow="fold") as live:
        overall_task = overall_progress.add_task("[green]Epochs", total=CONFIG['num_epochs'], completed=start_epoch)
        epoch_task = epoch_progress.add_task("[cyan]Batches", total=batches_per_epoch)

        model.train()
        current_total_batch_steps = optimizer_step * CONFIG["grad_accum_steps"]

        debug_chat_prompts = [
            "<user> Hi there! How are you doing today? </s>",
            "<user> What are your favorite topics to talk about? </s>",
            "<user> Tell me a short story. </s>",
            "<user> What did you learn recently? </s>",
            "<user> What's something interesting you know? </s>",
            "<user> Can you help me with a creative idea? </s>",
            "<user> What's the weather like in your world? </s>"
        ]

        for epoch in range(start_epoch, CONFIG['num_epochs']):
            logging.info(f"--- Starting Epoch {epoch+1}/{CONFIG['num_epochs']} (Optimizer step: {optimizer_step}) ---")
            epoch_main_loss_sum, epoch_aux_loss_sum, epoch_batches_processed = 0.0, 0.0, 0

            epoch_progress.reset(epoch_task) # Reset for new epoch
            epoch_progress.update(epoch_task, total=batches_per_epoch, completed=0, description=f"[cyan]E{epoch+1} Batches")

            metrics_data["Epoch"] = f"{epoch+1}/{CONFIG['num_epochs']}"
            current_epoch_start_opt_step = optimizer_step

            for i, batch in enumerate(dataloader):
                if batch is None or not batch["input_ids"].numel():
                     logging.warning(f"[yellow]Skipping {'None' if batch is None else 'empty'} batch at index {i} in epoch {epoch+1}.[/yellow]")
                     if batches_per_epoch > 0 : epoch_progress.update(epoch_task, advance=1)
                     continue

                main_loss_val, aux_loss_val = trainer_obj.train_step(batch)

                if main_loss_val is None or math.isnan(main_loss_val) or math.isinf(main_loss_val) or \
                   aux_loss_val is None or math.isnan(aux_loss_val) or math.isinf(aux_loss_val):
                    logging.error(f"[bold red]NaN/Inf loss detected. Main: {main_loss_val}, Aux: {aux_loss_val}. OptStep {optimizer_step}. Stopping.[/bold red]")
                    checkpoint_manager.save(model, trainer_obj.optimizer, f"error_loss_nan_inf_step{optimizer_step}")
                    return

                accum_main_loss += main_loss_val
                accum_aux_loss += aux_loss_val
                epoch_main_loss_sum += main_loss_val
                epoch_aux_loss_sum += aux_loss_val
                epoch_batches_processed += 1
                current_total_batch_steps += 1

                if current_total_batch_steps % CONFIG["grad_accum_steps"] == 0:
                    current_lr = trainer_obj.clip_and_step(optimizer_step)
                    avg_main_loss = accum_main_loss / CONFIG["grad_accum_steps"]
                    avg_aux_loss = accum_aux_loss / CONFIG["grad_accum_steps"]
                    accum_main_loss, accum_aux_loss = 0.0, 0.0

                    metrics_data["Opt Step"] = str(optimizer_step)
                    metrics_data["LR"] = f"{current_lr:.2e}"
                    metrics_data["Main Loss"] = f"{avg_main_loss:.4f}"
                    metrics_data["Aux Loss"] = f"{avg_aux_loss:.4f}"
                    metrics_data["Total Loss"] = f"{avg_main_loss + avg_aux_loss:.4f}"

                    if optimizer_step % CONFIG["debug_interval"] == 0: # Log metrics at debug_interval
                        logging.info(
                            f"E {epoch+1} | OptSt {optimizer_step} | TotalBatchSt {current_total_batch_steps} | "
                            f"AvgMainL: {avg_main_loss:.4f} | AvgAuxL: {avg_aux_loss:.4f} | LR: {current_lr:.2e}"
                        )
                    # Removed debug generation from here, it's now per-epoch
                    if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
                        checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step)
                    optimizer_step += 1

                epoch_progress.update(epoch_task, advance=1)
                metrics_data["Batch"] = f"{i+1}/{batches_per_epoch}"

                new_metrics_table = Table.grid(expand=True, padding=(0,1))
                new_metrics_table.add_column("Metric", style="cyan", no_wrap=True)
                new_metrics_table.add_column("Value", style="magenta")
                for key, value in metrics_data.items(): new_metrics_table.add_row(key, value)
                layout["metrics_panel_layout"].update(Panel(new_metrics_table, title="Live Metrics"))


            overall_progress.update(overall_task, advance=1)
            avg_epoch_main = epoch_main_loss_sum / epoch_batches_processed if epoch_batches_processed > 0 else 0
            avg_epoch_aux = epoch_aux_loss_sum / epoch_batches_processed if epoch_batches_processed > 0 else 0
            logging.info(
                f"--- Finished Epoch {epoch+1}/{CONFIG['num_epochs']} --- "
                f"Avg Epoch MainL: {avg_epoch_main:.4f} | Avg Epoch AuxL: {avg_epoch_aux:.4f} | "
                f"Opt Steps this epoch: {optimizer_step - current_epoch_start_opt_step} (approx)"
            )
            checkpoint_manager.save(model, trainer_obj.optimizer, f"epoch{epoch+1}_step{optimizer_step}")
            
            # Perform debug inference at the end of the epoch
            current_debug_prompt = random.choice(debug_chat_prompts)
            last_debug_prompt, last_debug_response = safety.debug_generation(current_debug_prompt)

            # Update the debug inference table in the UI
            new_debug_display_table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
            new_debug_display_table.add_column("Prompt", style="cyan", overflow="fold", ratio=1)
            new_debug_display_table.add_column("Response", style="green", overflow="fold", ratio=2)
            new_debug_display_table.add_row(escape(last_debug_prompt), escape(last_debug_response))
            layout["log_panel_layout"].update(Panel(new_debug_display_table, title="[b]Last Debug Inference[/b]", border_style="dim green"))


        logging.info(f"[bold green]Training finished after {CONFIG['num_epochs']} target epochs. Final optimizer step: {optimizer_step}.[/bold green]")

    logging.info("Saving final model state...")
    checkpoint_manager.save(model, trainer_obj.optimizer, f"final_step{optimizer_step}")
    
    # Final debug generation and UI update if Live is not active anymore (or to ensure it's shown)
    final_debug_prompt = "<user> The training is complete. How do you feel? </s>"
    last_debug_prompt, last_debug_response = safety.debug_generation(final_debug_prompt)
    
    # Create the table for console output if Live context is exited
    final_debug_display_table = Table(show_header=True, header_style="bold magenta", box=None, expand=True)
    final_debug_display_table.add_column("Prompt", style="cyan", overflow="fold", ratio=1)
    final_debug_display_table.add_column("Response", style="green", overflow="fold", ratio=2)
    final_debug_display_table.add_row(escape(last_debug_prompt), escape(last_debug_response))

    if 'live' in locals() and live.is_started:
        layout["log_panel_layout"].update(Panel(final_debug_display_table, title="[b]Last Debug Inference (Final)[/b]", border_style="dim green"))
        # Keep live active for a moment to show final state
        try:
            live.refresh() # Refresh to show the final update
            # console.input("Press Enter to exit...") # Optional: wait for user
        except Exception:
            pass # Live might already be stopped
    else: # If Live context already exited, print to console
        console.print(Panel(final_debug_display_table, title="[b]Last Debug Inference (Final)[/b]", border_style="dim green"))


if __name__ == "__main__":
    train()
