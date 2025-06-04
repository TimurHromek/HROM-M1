import os
# Set parallelism env var *before* importing tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F # Added for softmax in MoE
from torch.utils.data import Dataset, DataLoader
# Import necessary dataset functions, including concatenate_datasets if needed later
from datasets import load_dataset, disable_caching, concatenate_datasets
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders
import math
import re
from datetime import datetime
from contextlib import nullcontext
from collections import defaultdict
import logging
import random # For shuffling combined data

# Disable caching for datasets if needed, helps ensure reprocessing
# disable_caching()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

# Configuration
CONFIG = {
    "dim": 768,
    "n_layers": 8,
    "n_heads": 8,
    "ff_dim": 2048, # This will be the ff_dim for each expert
    "dropout": 0.1,
    "max_seq_len": 512,
    "batch_size": 16,
    "checkpoint_interval": 2000,
    "debug_interval": 400,
    "datasets": ["daily_dialog", "empathetic_dialogues", "blended_skill_talk", "AlekseyKorshuk/persona-chat", "papahawk/conversational-01"], # <-- Added papahawk/conversational-01
    "tokenizer_name": "hrom_moe_tokenizer.json", # Changed tokenizer name for MoE version
    "checkpoint_dir": "checkpoints_moe", # Changed checkpoint dir for MoE version
    "vocab_size": 32000,
    "tokenizer_train_samples_per_dataset": 50000, # Keep lower for faster testing if needed
    "learning_rate": 2e-5,
    "warmup_steps": 1000,
    "max_turns": 8, # For multi-turn datasets, papahawk is treated as 2 turns
    "max_checkpoints": 5,
    "num_epochs": 30,
    "grad_accum_steps": 8,

    # --- MoE Specific Configuration ---
    "num_experts": 8,             # Number of experts in each MoE layer
    "top_k_experts": 2,           # Number of experts to route to for each token
    "moe_load_balancing_coeff": 0.01 # Coefficient for the load balancing loss
}

# Ensure top_k is not more than num_experts
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
         logging.warning(f"RoPE Warning: pos sequence length ({pos_seq_len}) is shorter than tensor sequence length ({tensor_seq_len}). Using truncated tensor length for RoPE.")
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
        return x * nn.functional.gelu(gate) # Changed from F.silu(gate) to F.gelu(gate) to match original code

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

# --- MoE Components ---
class Expert(nn.Module):
    """A simple feed-forward network for an expert in the MoE layer."""
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * ff_dim) # Input to SwiGLU
        self.activation = SwiGLU()
        self.fc2 = nn.Linear(ff_dim, dim)    # Output of SwiGLU feeds into this

    def forward(self, x):
        hidden = self.fc1(x)
        activated_hidden = self.activation(hidden)
        return self.fc2(activated_hidden)

class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k gating."""
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
        x_reshaped = x.reshape(-1, dim) # (B*T, C) or (num_tokens, dim)
        num_tokens = x_reshaped.shape[0]

        # 1. Gating mechanism
        gate_logits = self.gate(x_reshaped) # (num_tokens, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1) # (num_tokens, num_experts)

        # 2. Select Top-K experts
        top_k_gate_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1) # (num_tokens, top_k)

        # Normalize top_k_gate_probs so they sum to 1 for each token's selected experts
        top_k_weights_norm = top_k_gate_probs / (top_k_gate_probs.sum(dim=-1, keepdim=True) + 1e-6) # (num_tokens, top_k)

        # 3. Dispatch tokens to experts and combine outputs
        final_output = torch.zeros_like(x_reshaped) # (num_tokens, dim)

        for i in range(self.num_experts):
            # Find tokens routed to expert i
            token_indices_for_expert_i, position_in_top_k = torch.where(top_k_indices == i)

            if token_indices_for_expert_i.numel() > 0:
                tokens_for_this_expert = x_reshaped[token_indices_for_expert_i] # (num_tokens_for_expert_i, dim)
                # Get the normalized weights for these tokens for this expert
                weights_for_this_expert = top_k_weights_norm[token_indices_for_expert_i, position_in_top_k] # (num_tokens_for_expert_i)

                expert_output = self.experts[i](tokens_for_this_expert) # (num_tokens_for_expert_i, dim)
                weighted_expert_output = expert_output * weights_for_this_expert.unsqueeze(-1)

                # Accumulate weighted outputs
                final_output.index_add_(0, token_indices_for_expert_i, weighted_expert_output.to(final_output.dtype))

        # 4. Load balancing loss (Mixtral-style)
        chosen_expert_mask = torch.zeros_like(gate_probs, device=x.device) # (num_tokens, num_experts)
        chosen_expert_mask.scatter_(1, top_k_indices, 1) # Mark chosen experts with 1

        fraction_tokens_per_expert = chosen_expert_mask.mean(dim=0) # (num_experts,)
        mean_router_probs_per_expert = gate_probs.mean(dim=0) # (num_experts,)

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


# --- Tokenizer Training ---
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
        text = re.sub(r'[^\w\s.,!?\'\-:;<>"]', '', text) # Allow < and > for special tokens
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
                # empathetic_dialogues is structured with multiple entries per conv_id
                # So we need to fetch more raw entries to get `samples_per_dataset` actual conversations
                ed_dataset = load_dataset("empathetic_dialogues", split=f"train[:{samples_per_dataset * 3}]", trust_remote_code=True) # Fetch more due to grouping
                logging.info("Processing empathetic_dialogues...")
                grouped_by_conv = defaultdict(list)
                for entry in ed_dataset:
                    grouped_by_conv[entry['conv_id']].append(entry)

                processed_conv_count = 0
                for conv_id, entries in grouped_by_conv.items():
                    if processed_conv_count >= samples_per_dataset:
                        break
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    formatted_dialogue = []
                    if sorted_entries[0]['context']:
                         cleaned_context = self._clean_text(sorted_entries[0]['context'])
                         if cleaned_context:
                              formatted_dialogue.append(f"<user> {cleaned_context}")
                    last_role = '<user>' if formatted_dialogue else None # Determine based on context
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
                    # Correctly access turns including free_turker_utterance and guided_turker_utterance
                    dialogue_turns_raw = list(entry['previous_utterance']) # Make a mutable copy
                    if entry.get('free_turker_utterance'): # This is usually the user's last turn
                        dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'): # This is usually the system's last turn
                         dialogue_turns_raw.append(entry['guided_turker_utterance'])

                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]]
                    # BST turn structure: User, Bot, User, Bot ...
                    # The 'previous_utterance' list alternates.
                    # If 'free_turker_utterance' is present, it's a user turn.
                    # If 'guided_turker_utterance' is present, it's an agent turn.
                    # A common pattern is previous_utterance ends with Agent, then free_turker (User), then guided_turker (Agent).
                    # Let's assume simple alternation for the combined list.
                    for i, utterance in enumerate(turns_to_process):
                        role = "<user>" if i % 2 == 0 else "<assistant>" # This might need adjustment based on exact BST turn structure
                        cleaned_utterance = self._clean_text(utterance)
                        if cleaned_utterance:
                            formatted_dialogue.append(f"{role} {cleaned_utterance}")
                    if formatted_dialogue:
                        text_samples.append(" </s> ".join(formatted_dialogue))
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
                        # Get the history from the last utterance object
                        history = entry['utterances'][-1]['history']
                        history = history[:CONFIG["max_turns"]] # Limit turns
                        formatted_dialogue = []
                        for i, utterance in enumerate(history):
                             role = "<user>" if i % 2 == 0 else "<assistant>" # Assuming alternating roles
                             cleaned_utterance = self._clean_text(utterance)
                             if cleaned_utterance:
                                  formatted_dialogue.append(f"{role} {cleaned_utterance}")
                        if formatted_dialogue:
                            text_samples.append(" </s> ".join(formatted_dialogue))
                    else:
                        logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry}")
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
                    if instruction:
                        formatted_pair.append(f"<user> {instruction}")
                    if response and instruction: # Only add assistant if there was a user part
                        formatted_pair.append(f"<assistant> {response}")
                    
                    if len(formatted_pair) == 2: # Ensure we have a user-assistant pair
                        text_samples.append(" </s> ".join(formatted_pair))
                    elif len(formatted_pair) == 1: # If only user instruction
                         text_samples.append(formatted_pair[0]) # append "<user> instruction"
            except Exception as e:
                logging.error(f"Failed to load or process {ph_dataset_name} for tokenizer: {e}")

        logging.info(f"Total text samples for tokenizer training: {len(text_samples)}")
        if not text_samples:
            raise ValueError("No text samples collected for tokenizer training. Check dataset loading and paths.")

        os.makedirs(self.tokenizer_dir, exist_ok=True)
        logging.info(f"Training BPE tokenizer with vocab size {CONFIG['vocab_size']}...")
        trainer = trainers.BpeTrainer(
            vocab_size=CONFIG["vocab_size"],
            special_tokens=self.special_tokens,
            min_frequency=2,
            show_progress=True
        )
        def text_iterator():
            for sample in text_samples:
                yield sample
        self.tokenizer.train_from_iterator(text_iterator(), trainer=trainer, length=len(text_samples))

        eos_token_id = self.tokenizer.token_to_id("</s>")
        if eos_token_id is None:
            logging.warning("</s> token not found! Using <pad> as fallback for post-processor.")
            eos_token_id = self.tokenizer.token_to_id("<pad>") or 0 # Ensure it's not None

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A </s>", # This adds </s> to single sequences
            pair="$A </s> $B </s>", # This adds </s> to pairs
            special_tokens=[("</s>", eos_token_id)],
        )
        logging.info(f"Saving tokenizer to {self.tokenizer_path}")
        self.tokenizer.save(self.tokenizer_path)
        logging.info("Tokenizer training complete.")

    def get_tokenizer(self):
         if not os.path.exists(self.tokenizer_path):
              raise FileNotFoundError(f"Tokenizer file not found at {self.tokenizer_path}. Train tokenizer first.")
         tokenizer = Tokenizer.from_file(self.tokenizer_path)
         # Ensure all special tokens are actually in the tokenizer's vocab
         required_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<user>", "<assistant>"]
         for token in required_tokens:
              if tokenizer.token_to_id(token) is None:
                   # This is critical, if a special token isn't there, it can't be used.
                   raise ValueError(f"Crucial special token '{token}' not found in loaded tokenizer '{self.tokenizer_path}'!")
         return tokenizer

# --- Dataset Loading and Processing ---
class CombinedChatDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")

        if None in [self.pad_id, self.eos_id, self.bos_id, self.user_id, self.assistant_id]:
            missing = [name for name, val in zip(["pad", "eos", "bos", "user", "assistant"], [self.pad_id, self.eos_id, self.bos_id, self.user_id, self.assistant_id]) if val is None]
            raise ValueError(f"Tokenizer is missing critical special token IDs: {missing}. Tokenizer path: {self.tokenizer.model_path if hasattr(self.tokenizer, 'model_path') else 'N/A'}")

        self.max_length = CONFIG["max_seq_len"]
        self._clean_text = TokenizerTrainer()._clean_text # Use the same cleaning logic
        self.all_processed_conversations = []

        if "daily_dialog" in CONFIG["datasets"]:
            logging.info("Loading and processing daily_dialog dataset...")
            try:
                dd_dataset = load_dataset("daily_dialog", split="train", trust_remote_code=True)
                logging.info(f"Processing {len(dd_dataset)} daily_dialog conversations...")
                for entry in dd_dataset:
                    conversation = []
                    dialogue = entry['dialog'][:CONFIG["max_turns"]]
                    if not dialogue: continue
                    for i, utterance in enumerate(dialogue):
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_text = self._clean_text(utterance)
                        if cleaned_text:
                            conversation.append({'role': role, 'text': cleaned_text})
                    if conversation:
                        self.all_processed_conversations.append(conversation)
            except Exception as e:
                 logging.error(f"Failed to load or process daily_dialog for training: {e}")

        if "empathetic_dialogues" in CONFIG["datasets"]:
            logging.info("Loading and processing empathetic_dialogues dataset...")
            try:
                ed_dataset = load_dataset("empathetic_dialogues", split="train", trust_remote_code=True)
                logging.info("Grouping empathetic_dialogues by conversation ID...")
                conversations_grouped = defaultdict(list)
                for entry in ed_dataset:
                    conversations_grouped[entry['conv_id']].append(entry)
                logging.info(f"Processing {len(conversations_grouped)} empathetic_dialogues conversations...")
                for conv_id, entries in conversations_grouped.items():
                    conversation = []
                    sorted_entries = sorted(entries, key=lambda x: x['utterance_idx'])
                    if sorted_entries[0]['context']:
                        context_text = self._clean_text(sorted_entries[0]['context'])
                        if context_text:
                             conversation.append({'role': '<user>', 'text': context_text})
                    last_role = conversation[-1]['role'] if conversation else None
                    for entry in sorted_entries:
                         text = self._clean_text(entry['utterance'])
                         if not text: continue
                         current_role = '<assistant>' if last_role == '<user>' else '<user>'
                         conversation.append({'role': current_role, 'text': text})
                         last_role = current_role
                    conversation = conversation[:CONFIG["max_turns"]]
                    if conversation:
                        self.all_processed_conversations.append(conversation)
            except Exception as e:
                logging.error(f"Failed to load or process empathetic_dialogues for training: {e}")

        if "blended_skill_talk" in CONFIG["datasets"]:
            logging.info("Loading and processing blended_skill_talk dataset...")
            try:
                bst_dataset = load_dataset("blended_skill_talk", split="train", trust_remote_code=True)
                logging.info(f"Processing {len(bst_dataset)} blended_skill_talk conversations...")
                for entry in bst_dataset:
                    conversation = []
                    dialogue_turns_raw = list(entry['previous_utterance'])
                    if entry.get('free_turker_utterance'):
                        dialogue_turns_raw.append(entry['free_turker_utterance'])
                    if entry.get('guided_turker_utterance'):
                         dialogue_turns_raw.append(entry['guided_turker_utterance'])
                    if not dialogue_turns_raw: continue

                    turns_to_process = dialogue_turns_raw[:CONFIG["max_turns"]]
                    for i, utterance in enumerate(turns_to_process):
                        # Simplified role assignment, assuming alternation.
                        # For BST, the exact roles might depend on how 'previous_utterance' mixes with 'free_turker' and 'guided_turker'.
                        # A common pattern: prev_utterance (alternating), free_turker_utterance (user), guided_turker_utterance (agent).
                        # This simplified alternation should be mostly correct for a combined list.
                        role = "<user>" if i % 2 == 0 else "<assistant>"
                        cleaned_text = self._clean_text(utterance)
                        if cleaned_text:
                            conversation.append({'role': role, 'text': cleaned_text})
                    if conversation:
                        self.all_processed_conversations.append(conversation)
            except Exception as e:
                logging.error(f"Failed to load or process blended_skill_talk for training: {e}")

        if "AlekseyKorshuk/persona-chat" in CONFIG["datasets"]:
            pc_dataset_name = "AlekseyKorshuk/persona-chat"
            logging.info(f"Loading and processing {pc_dataset_name} dataset...")
            try:
                pc_dataset = load_dataset(pc_dataset_name, split="train", trust_remote_code=True)
                logging.info(f"Processing {len(pc_dataset)} {pc_dataset_name} conversations...")
                for entry in pc_dataset:
                    conversation = []
                    if 'utterances' in entry and entry['utterances']:
                        history = entry['utterances'][-1]['history']
                        history = history[:CONFIG["max_turns"]] # Limit turns
                        for i, utterance in enumerate(history):
                            role = "<user>" if i % 2 == 0 else "<assistant>"
                            cleaned_text = self._clean_text(utterance)
                            if cleaned_text:
                                conversation.append({'role': role, 'text': cleaned_text})
                        if conversation:
                            self.all_processed_conversations.append(conversation)
                    else:
                         logging.warning(f"Skipping {pc_dataset_name} entry due to unexpected structure: {entry.keys()}")
            except Exception as e:
                logging.error(f"Failed to load or process {pc_dataset_name} for training: {e}")

        if "papahawk/conversational-01" in CONFIG["datasets"]:
            ph_dataset_name = "papahawk/conversational-01"
            logging.info(f"Loading and processing {ph_dataset_name} dataset...")
            try:
                ph_dataset = load_dataset(ph_dataset_name, split="train", trust_remote_code=True)
                logging.info(f"Processing {len(ph_dataset)} {ph_dataset_name} entries...")
                for entry in ph_dataset:
                    instruction = self._clean_text(entry.get('instruction', ''))
                    response = self._clean_text(entry.get('response', ''))

                    if instruction and response: # Only process if both instruction and response exist
                        # Treat as a two-turn conversation
                        conversation = [
                            {'role': '<user>', 'text': instruction},
                            {'role': '<assistant>', 'text': response}
                        ]
                        # CONFIG["max_turns"] is not strictly applied here as each entry is 2 turns.
                        # If it were a multi-turn format from this dataset, truncation would apply.
                        self.all_processed_conversations.append(conversation)
                    # else:
                        # Optionally log skipped entries if instruction or response is missing
                        # logging.debug(f"Skipping entry from {ph_dataset_name} due to missing instruction or response.")
            except Exception as e:
                logging.error(f"Failed to load or process {ph_dataset_name} for training: {e}")


        logging.info(f"Total processed conversations from all datasets: {len(self.all_processed_conversations)}")
        if not self.all_processed_conversations:
             raise ValueError("No processed conversations were created from any dataset. Check dataset paths and processing logic.")
        logging.info("Shuffling combined dataset...")
        random.shuffle(self.all_processed_conversations)

    def __len__(self):
        return len(self.all_processed_conversations)

    def __getitem__(self, idx):
        conversation = self.all_processed_conversations[idx]
        formatted_ids = [self.bos_id] # Start with BOS

        for turn in conversation:
            role_id = self.user_id if turn['role'] == '<user>' else self.assistant_id
            try:
                # Encode utterance without adding any special tokens (like BOS/EOS) automatically by tokenizer.encode
                utterance_ids = self.tokenizer.encode(turn['text'], add_special_tokens=False).ids
            except Exception as e:
                 logging.error(f"Error encoding text at index {idx}, turn '{turn}': {e}")
                 utterance_ids = [] # Skip problematic turn

            # Check space: current length + role_id + utterance_ids + eos_id
            if len(formatted_ids) + 1 + len(utterance_ids) + 1 > self.max_length:
                # If only role + EOS can fit, add them and break
                if len(formatted_ids) + 1 + 1 <= self.max_length:
                     formatted_ids.append(role_id)
                     formatted_ids.append(self.eos_id) # Add EOS if this is the last possible token
                break # Sequence is full

            formatted_ids.append(role_id)
            formatted_ids.extend(utterance_ids)
            formatted_ids.append(self.eos_id) # Add EOS after each turn

        # Truncate if still too long (e.g. if last utterance was very long)
        if len(formatted_ids) > self.max_length:
             formatted_ids = formatted_ids[:self.max_length]
             # Ensure last token is not a role_id if truncated abruptly
             if formatted_ids and (formatted_ids[-1] == self.user_id or formatted_ids[-1] == self.assistant_id):
                  formatted_ids.pop() # Remove trailing role_id
             # Re-check length and ensure it ends with EOS if possible and space allows
             if formatted_ids and formatted_ids[-1] != self.eos_id:
                 if len(formatted_ids) == self.max_length: # If full, replace last token with EOS
                     formatted_ids[-1] = self.eos_id
                 # elif len(formatted_ids) < self.max_length: # If space, append EOS
                 #    formatted_ids.append(self.eos_id) # This case is less likely due to above logic

        # Ensure sequence has at least BOS and one other token before slicing for input/label
        if len(formatted_ids) < 2: # e.g., only [bos_id] or [bos_id, eos_id] after truncation
             logging.warning(f"Sequence at index {idx} is too short after processing (<2 tokens): {formatted_ids}. Skipping.")
             # Try to return a minimal valid item to avoid None, or handle None in collate_fn
             # For now, let collate_fn handle potential Nones.
             return None

        input_ids = formatted_ids[:-1]
        labels = formatted_ids[1:]

        if len(input_ids) == 0: # Should be caught by len(formatted_ids) < 2
            logging.warning(f"Sequence at index {idx} resulted in empty input_ids after slicing. Skipping.")
            return None

        return {"input_ids": input_ids, "labels": labels}

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in batch if item is not None] # Filter out None items
        if not batch:
            # If all items in batch were None, return None or an empty dict
            # to be handled by the training loop
            logging.warning("Collate_fn received an entirely empty batch after filtering Nones.")
            return None # Or: return {"input_ids": torch.empty(0), "labels": torch.empty(0), "attention_mask": torch.empty(0)}

        max_len = 0
        for item in batch:
            if "input_ids" in item and len(item["input_ids"]) > max_len:
                max_len = len(item["input_ids"])

        if max_len == 0: # If all valid items had empty input_ids (should not happen with __getitem__ checks)
            logging.warning("Collate_fn: max_len is 0 after processing batch items.")
            return None

        try:
            # It's better to pass pad_id or get it from a global config/tokenizer instance
            # rather than reloading from file in every collate_fn call.
            # For simplicity, keeping current structure but flagging as potential optimization.
            tokenizer_path = os.path.join("tokenizer", CONFIG["tokenizer_name"])
            # This can be slow if called frequently. Consider passing tokenizer/pad_id.
            tokenizer = Tokenizer.from_file(tokenizer_path)
            pad_id = tokenizer.token_to_id("<pad>")
            if pad_id is None: raise ValueError("<pad> token not found in tokenizer for collate_fn")
        except Exception as e:
            logging.error(f"Collate Error: Failed to load tokenizer or get pad_id ('{CONFIG['tokenizer_name']}'): {e}. Using pad_id=0 as fallback.")
            pad_id = 0 # Fallback pad_id

        inputs, labels, masks = [], [], []
        for item in batch:
            input_len = len(item["input_ids"])
            pad_len = max_len - input_len

            inputs.append(item["input_ids"] + [pad_id] * pad_len)
            labels.append(item["labels"] + [pad_id] * pad_len) # Use pad_id for labels too for CrossEntropyLoss ignore_index
            masks.append([1] * input_len + [0] * pad_len)

        return {
            "input_ids": torch.tensor(inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long)
        }

# --- Trainer, Safety Manager, Checkpoint Manager ---

class HROMTrainer:
    def __init__(self, model, tokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        self.model = model.to(self.device)

        self.use_amp = (self.device.type == "cuda" and hasattr(torch.cuda.amp, "GradScaler"))
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        logging.info(f"Automatic Mixed Precision (AMP): {'Enabled' if self.use_amp else 'Disabled'}")

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused= (self.device.type == "cuda") # fused=True can be faster on CUDA
        )
        self.tokenizer = tokenizer # Store the tokenizer instance
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
             # This should ideally not happen if tokenizer loading is robust
             self.pad_id = CONFIG.get("pad_token_id", 0) # Fallback from global config if available
             logging.warning(f"<pad> token ID not found in provided tokenizer, using fallback ID: {self.pad_id}")

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        self.base_lr = CONFIG["learning_rate"]
        self.warmup_steps = CONFIG["warmup_steps"]

    def _adjust_learning_rate(self, step):
        if self.warmup_steps > 0 and step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            # Optional: Add cosine decay after warmup
            # progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            # lr = self.base_lr * (0.5 * (1.0 + math.cos(math.pi * progress)))
            lr = self.base_lr # For now, constant after warmup
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_step(self, batch):
        if self.use_amp:
            amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        # Ensure nullcontext is properly used if AMP is disabled
        autocast_context = torch.cuda.amp.autocast(dtype=amp_dtype, enabled=self.use_amp) if self.use_amp else nullcontext()

        with autocast_context:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs, moe_aux_loss = self.model(input_ids, attention_mask=attention_mask)

            logits_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)

            # Ensure logits are float32 for CrossEntropyLoss if using AMP with float16
            # For bfloat16, this might not be strictly necessary but doesn't hurt.
            main_loss = self.criterion(logits_flat.float(), labels_flat)
            total_loss = main_loss + moe_aux_loss

            # Scale loss for gradient accumulation
            scaled_loss = total_loss / CONFIG["grad_accum_steps"]

        if self.use_amp and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return main_loss.item(), moe_aux_loss.item() # Return unscaled losses for logging

    def clip_and_step(self, current_optimizer_step): # Renamed step to current_optimizer_step for clarity
         current_lr = self._adjust_learning_rate(current_optimizer_step) # Pass optimizer step for LR scheduling
         if self.use_amp and self.scaler:
             self.scaler.unscale_(self.optimizer) # Unscale before clipping
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             self.scaler.step(self.optimizer)
             self.scaler.update()
         else:
             torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
             self.optimizer.step()
         self.optimizer.zero_grad(set_to_none=True) # More memory efficient
         return current_lr


class SafetyManager:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.bad_words = ["kill", "murder", "suicide", "hate", "abuse", "violence", "illegal", "harm", "die", "attack", "rape", "molest", "exploit", "terror"]
        self.bad_word_ids = []
        logging.info("Initializing safety manager...")
        for word in self.bad_words:
             # Try encoding with and without leading space as tokenization can vary
             ids_with_space = tokenizer.encode(f" {word}", add_special_tokens=False).ids
             if ids_with_space:
                self.bad_word_ids.append(ids_with_space)
                logging.debug(f"Encoded bad word ' {word}' to IDs: {ids_with_space}")

             ids_no_space = tokenizer.encode(word, add_special_tokens=False).ids
             if ids_no_space and ids_no_space != ids_with_space: # Avoid duplicates if space makes no difference
                  self.bad_word_ids.append(ids_no_space)
                  logging.debug(f"Encoded bad word '{word}' to IDs: {ids_no_space}")

             if not ids_with_space and not ids_no_space:
                logging.warning(f"Could not encode bad word '{word}' - skipping.")

        # Get critical token IDs
        self.eos_id = self.tokenizer.token_to_id("</s>")
        self.bos_id = self.tokenizer.token_to_id("<s>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.assistant_id = self.tokenizer.token_to_id("<assistant>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")

        # Log errors if critical tokens are missing
        if self.eos_id is None: logging.error("</s> token ID not found in SafetyManager!"); self.eos_id = 0 # Fallback
        if self.bos_id is None: logging.error("<s> token ID not found in SafetyManager!"); self.bos_id = 0 # Fallback
        if self.user_id is None: logging.error("<user> token ID not found in SafetyManager!")
        if self.assistant_id is None: logging.error("<assistant> token ID not found in SafetyManager!")
        if self.pad_id is None: logging.error("<pad> token ID not found in SafetyManager!"); self.pad_id = 0 # Fallback

    def contains_sequence(self, tokens, seq):
        if not seq or not tokens or len(tokens) < len(seq): return False
        seq_len = len(seq)
        for i in range(len(tokens) - seq_len + 1):
            if tokens[i : i + seq_len] == seq: return True
        return False

    def content_filter(self, text_ids):
        if not isinstance(text_ids, list):
            logging.warning(f"Content filter received non-list input: {type(text_ids)}")
            return True # Default to safe if input is unexpected
        for bad_ids in self.bad_word_ids:
            if self.contains_sequence(text_ids, bad_ids):
                try:
                    detected_word = self.tokenizer.decode(bad_ids)
                except Exception:
                    detected_word = "unknown (decoding error)"
                logging.warning(f"Unsafe content detected: Found sequence for '{detected_word}' (IDs: {bad_ids}). Blocking generation.")
                return False # Unsafe
        return True # Safe

    def generate_safely(self, prompt, max_new_tokens=50, temperature=0.7, top_k=50):
        self.model.eval()
        device = next(self.model.parameters()).device

        # Ensure prompt starts with BOS, add user/assistant roles correctly
        # Example prompt: "<user> Hello there! </s>"
        # Or simply "Hello there!" -> will be wrapped.

        # Tokenize the input prompt
        # Remove <s> if already present, as we add it.
        if prompt.startswith(self.tokenizer.decode([self.bos_id])):
            prompt = prompt[len(self.tokenizer.decode([self.bos_id])):].strip()

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False).ids
        input_ids = [self.bos_id] + prompt_ids

        # Ensure the prompt ends with an assistant token to cue the model for response
        if self.assistant_id is not None:
            if not input_ids or input_ids[-1] != self.assistant_id: # Check if last token is already assistant
                # Also check if it ends with user_id, if so, add assistant
                if input_ids and input_ids[-1] == self.user_id:
                    input_ids.append(self.assistant_id)
                elif input_ids and input_ids[-1] == self.eos_id: # e.g. <user> text </s>
                    input_ids.append(self.assistant_id)
                elif not input_ids: # Empty prompt
                    input_ids.extend([self.user_id, self.eos_id, self.assistant_id]) # Default to user -> assistant
                else: # General case, append assistant
                    input_ids.append(self.assistant_id)
        else:
            logging.error("Assistant token ID is None. Cannot properly cue model for generation.")
            return "Error: Assistant token not found."

        generated_ids = list(input_ids) # Start with the prepared input_ids
        logging.debug(f"Starting safe generation with initial IDs: {generated_ids} (decoded: '{self.tokenizer.decode(generated_ids)}')")

        with torch.no_grad():
            for step in range(max_new_tokens):
                # Prepare input tensor, ensuring it fits max_seq_len
                current_input_ids_trimmed = generated_ids[-CONFIG["max_seq_len"]:]
                current_input_tensor = torch.tensor([current_input_ids_trimmed], device=device)
                attention_mask = torch.ones_like(current_input_tensor, device=device)

                try:
                    outputs, _ = self.model(current_input_tensor, attention_mask=attention_mask) # Ignore aux_loss
                    next_token_logits = outputs[:, -1, :] # Logits for the last token
                except Exception as e:
                     logging.error(f"Model forward pass failed during generation: {e}", exc_info=True)
                     break # Stop generation on error

                # Apply temperature
                if temperature > 0 and temperature != 1.0: # Avoid division by zero or no-op
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0 and top_k < next_token_logits.size(-1):
                    v, _ = torch.topk(next_token_logits, top_k, dim=-1)
                    # Create a mask for tokens not in top-k
                    # Ensure threshold is taken from the correct dimension if batch_size > 1 (here B=1)
                    threshold_val = v[:, -1].unsqueeze(-1) # Get the k-th largest logit value
                    # Set logits not in top-k to -inf
                    next_token_logits = next_token_logits.masked_fill(next_token_logits < threshold_val, -float('Inf'))

                # Get probabilities and sample
                probs = torch.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or torch.isinf(probs).any():
                     logging.warning(f"NaN/Inf detected in probabilities at step {step}. Using uniform distribution as fallback.")
                     # Fallback to uniform distribution over the vocabulary
                     probs = torch.ones_like(probs) / probs.size(-1)

                next_token_id = torch.multinomial(probs, num_samples=1).item()

                # --- Safety Check BEFORE appending the token ---
                # Check the potential sequence if this token were added
                potential_sequence_for_check = generated_ids[len(input_ids):] + [next_token_id] # Check only the generated part
                if not self.content_filter(potential_sequence_for_check):
                    logging.warning(f"Unsafe token ID {next_token_id} ('{self.tokenizer.decode([next_token_id])}') blocked PRE-APPEND. Stopping generation.")
                    # Optionally, try to sample a different token or end generation.
                    # For now, just stop.
                    break

                generated_ids.append(next_token_id)

                if next_token_id == self.eos_id:
                    logging.debug(f"EOS token ({self.eos_id}) generated at step {step+1}. Stopping.")
                    break
                if step == max_new_tokens - 1: # Max length reached
                     logging.debug("Max new tokens reached.")
                     if generated_ids[-1] != self.eos_id and self.eos_id is not None:
                         generated_ids.append(self.eos_id) # Append EOS if not already there
        self.model.train() # Set model back to training mode

        # Extract only the generated response part (after the initial input_ids)
        response_ids = generated_ids[len(input_ids):]
        # Decode, skipping special tokens like <s>, </s>, <user>, <assistant> in the final output string
        decoded_text = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
        return decoded_text

    def debug_generation(self, prompt="<user> Tell me about your hobbies. </s>"): # Ensure prompt ends with </s> for consistency
         logging.info(f"\n--- Debug Generation & Safety Check ---")
         # Standardize prompt format slightly for consistency in logging/testing
         if not prompt.strip().startswith("<user>") and not prompt.strip().startswith("<assistant>"):
             prompt = f"<user> {prompt.strip()}" # Default to user prompt
         if not prompt.strip().endswith("</s>"):
             prompt = f"{prompt.strip()} </s>"

         # The generate_safely method handles BOS and final assistant cueing.
         generated_response = self.generate_safely(prompt, max_new_tokens=60, temperature=0.7, top_k=50)
         logging.info(f"Prompt Sent: '{prompt}'")
         logging.info(f"Generated Response: '{generated_response}'")
         logging.info(f"--- End Debug Generation ---\n")


class CheckpointManager:
    def __init__(self):
        self.checkpoint_dir = CONFIG["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logging.info(f"Checkpoint directory set to: {self.checkpoint_dir}")

    def save(self, model, optimizer, step_info): # step_info can be int or string like "epochX_stepY"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract base name for MoE checkpoints, e.g., "moe" from "checkpoints_moe"
        prefix_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
        step_str = str(step_info).replace(" ", "_") # Sanitize step_info for filename

        filename = f"hrom_{prefix_base}_step{step_str}_{timestamp}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step_info": step_info, # Store the original step_info
            "config": CONFIG # Save current config for reference
        }
        logging.info(f"Saving checkpoint to {path}...")
        try:
            torch.save(state, path)
            logging.info(f"Checkpoint saved successfully: {filename}")
            self._cleanup_old_checkpoints()
        except Exception as e:
            logging.error(f"Failed to save checkpoint '{path}': {e}", exc_info=True)

    def _parse_step_from_filename(self, filename_part):
        # Tries to extract a numerical step from strings like "12000" or "epoch3_step12000"
        match_epoch_step = re.search(r'epoch\d+_step(\d+)', filename_part)
        if match_epoch_step:
            return int(match_epoch_step.group(1))
        match_step = re.search(r'(\d+)', filename_part)
        if match_step:
            return int(match_step.group(1))
        return 0 # Fallback if no numeric step found

    def _cleanup_old_checkpoints(self):
        max_checkpoints = CONFIG.get("max_checkpoints", 5)
        if max_checkpoints <= 0: return # Disabled

        try:
            prefix_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            # Regex to match checkpoint filenames: hrom_moe_step(digits or epochX_stepY or final_stepZ)_timestamp.pt
            pattern_str = rf"hrom_{re.escape(prefix_base)}_step([\w\d_]+)_(\d{{8}}_\d{{6}})\.pt"
            pattern = re.compile(pattern_str)
            
            checkpoints = []
            for f_name in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f_name)
                 if match:
                      filepath = os.path.join(self.checkpoint_dir, f_name)
                      # Use file modification time for sorting actual save time
                      # Step info can be used for tie-breaking or specific logic if needed
                      checkpoints.append((filepath, os.path.getmtime(filepath)))

            # Sort by modification time (oldest first)
            checkpoints.sort(key=lambda x: x[1])

            num_to_delete = len(checkpoints) - max_checkpoints
            if num_to_delete > 0:
                logging.info(f"Max checkpoints ({max_checkpoints}) reached. Deleting {num_to_delete} oldest ones.")
                for i in range(num_to_delete):
                    file_to_remove, _ = checkpoints[i]
                    try:
                        os.remove(file_to_remove)
                        logging.info(f"Removed old checkpoint: {file_to_remove}")
                    except OSError as e:
                        logging.error(f"Error removing old checkpoint {file_to_remove}: {e}")
        except Exception as e:
            logging.error(f"Error during checkpoint cleanup: {e}", exc_info=True)

    def load_latest(self, model, optimizer):
        try:
            prefix_base = os.path.basename(self.checkpoint_dir).replace("checkpoints_", "")
            pattern_str = rf"hrom_{re.escape(prefix_base)}_step([\w\d_]+)_(\d{{8}}_\d{{6}})\.pt"
            pattern = re.compile(pattern_str)

            checkpoints = []
            for f_name in os.listdir(self.checkpoint_dir):
                 match = pattern.match(f_name)
                 if match:
                      filepath = os.path.join(self.checkpoint_dir, f_name)
                      # Use modification time to find the truly latest saved file
                      checkpoints.append((filepath, os.path.getmtime(filepath), match.group(1))) # path, mtime, step_info_str

            if not checkpoints:
                logging.info(f"No valid checkpoints found in '{self.checkpoint_dir}' matching pattern. Starting fresh.")
                return 0 # No checkpoint to load, start from step 0

            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: x[1], reverse=True)
            latest_checkpoint_path, _, latest_step_info_str = checkpoints[0]

            logging.info(f"Loading latest checkpoint from: {latest_checkpoint_path}")
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
                    if loaded_val != current_val: # Handles cases where key might be missing in one
                        mismatched_keys.append((key, loaded_val, current_val))
                if mismatched_keys:
                    logging.warning("--- CONFIG MISMATCH DETECTED (Loading Checkpoint) ---")
                    for key, loaded_val, current_val in mismatched_keys:
                        logging.warning(f"  - {key}: Checkpoint='{loaded_val}', Current='{current_val}'")
                    logging.warning("Proceeding with loading, but this may impact model performance or cause errors if critical arch params changed.")
            else:
                logging.warning("Checkpoint does not contain configuration info. Cannot check for mismatches.")

            try:
                 model.load_state_dict(checkpoint['model'], strict=True)
            except RuntimeError as e:
                 logging.error(f"Failed to load model state_dict: {e}. This often happens if model architecture changed or vocab_size is different. Starting fresh.")
                 return 0 # Cannot recover model state, start fresh

            try:
                 optimizer.load_state_dict(checkpoint['optimizer'])
                 # Move optimizer states to current device
                 for state_val in optimizer.state.values(): # state is a defaultdict
                    for k, v in state_val.items():
                        if isinstance(v, torch.Tensor):
                            try:
                                state_val[k] = v.to(map_location)
                            except Exception as e_opt_move:
                                logging.error(f"Failed to move optimizer tensor '{k}' to device: {e_opt_move}")
            except Exception as e:
                 logging.warning(f"Could not load optimizer state_dict: {e}. Optimizer state will be reset.")
                 # Reset optimizer state if loading fails
                 optimizer.state = defaultdict(dict)


            # Determine starting optimizer step
            # The 'step_info' could be an int (optimizer step) or a string (e.g., "epochX_stepY", "final_stepZ")
            step_info_loaded = checkpoint.get('step_info', 0)
            start_optimizer_step = 0
            if isinstance(step_info_loaded, int):
                start_optimizer_step = step_info_loaded + 1 # Resume from next step
            elif isinstance(step_info_loaded, str):
                # Try to parse numeric step from string for continuation
                # e.g., "epoch2_step10000" -> 10000, "final_step20000" -> 20000
                parsed_step = self._parse_step_from_filename(step_info_loaded)
                start_optimizer_step = parsed_step + 1 if parsed_step > 0 else 0 # If parsing fails, might start from 0 or a previous point
                if parsed_step == 0 and "epoch" in step_info_loaded.lower(): # If it was an epoch save but couldn't parse step, log it
                    logging.warning(f"Loaded epoch checkpoint '{step_info_loaded}' but could not parse specific optimizer step. Optimizer step count might reset or be inaccurate for LR scheduling if not careful.")

            logging.info(f"Checkpoint loaded. Resuming from (or after) saved info '{step_info_loaded}'. Effective next optimizer_step: {start_optimizer_step}.")
            return start_optimizer_step

        except FileNotFoundError:
            logging.info(f"No checkpoint directory or files at '{self.checkpoint_dir}'. Starting fresh.")
            return 0
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}. Starting fresh.", exc_info=True)
            return 0


# --- Training Function ---

def train():
    logging.info("Starting HROM-MoE training process...")
    logging.info(f"Initial Configuration: {CONFIG}")

    tokenizer_trainer = TokenizerTrainer()
    tokenizer_path = tokenizer_trainer.tokenizer_path
    if not os.path.exists(tokenizer_path):
        logging.info(f"Tokenizer '{CONFIG['tokenizer_name']}' not found at '{tokenizer_path}'. Training new tokenizer...")
        try:
            # Pass only unique dataset names to tokenizer trainer
            tokenizer_datasets = list(set(CONFIG["datasets"]))
            tokenizer_trainer.train(tokenizer_datasets)
        except Exception as e:
             logging.error(f"Critical error during tokenizer training: {e}", exc_info=True)
             return # Cannot proceed without a tokenizer
    else:
        logging.info(f"Loading existing tokenizer from {tokenizer_path}")

    try:
        tokenizer = tokenizer_trainer.get_tokenizer()
        # Update global config with actual token IDs from the loaded tokenizer
        CONFIG['pad_token_id'] = tokenizer.token_to_id("<pad>")
        CONFIG['bos_token_id'] = tokenizer.token_to_id("<s>")
        CONFIG['eos_token_id'] = tokenizer.token_to_id("</s>")
        # Check if all critical tokens were loaded
        if None in [CONFIG['pad_token_id'], CONFIG['bos_token_id'], CONFIG['eos_token_id'],
                    tokenizer.token_to_id("<user>"), tokenizer.token_to_id("<assistant>")]:
            raise ValueError("One or more critical special tokens (<pad>, <s>, </s>, <user>, <assistant>) are missing from the tokenizer after loading.")
        logging.info(f"Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size()}. PAD ID: {CONFIG['pad_token_id']}, BOS ID: {CONFIG['bos_token_id']}, EOS ID: {CONFIG['eos_token_id']}")
    except (FileNotFoundError, ValueError) as e:
         logging.error(f"Critical error loading tokenizer: {e}. Cannot continue.", exc_info=True)
         return

    logging.info("Initializing HROM-MoE model...")
    if CONFIG['vocab_size'] != tokenizer.get_vocab_size():
         logging.warning(
             f"CONFIG vocab_size ({CONFIG['vocab_size']}) differs from tokenizer vocab_size ({tokenizer.get_vocab_size()}). "
             f"Updating CONFIG vocab_size to match tokenizer: {tokenizer.get_vocab_size()}."
         )
         CONFIG['vocab_size'] = tokenizer.get_vocab_size()
    model = HROM()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"HROM-MoE Model initialized. Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    logging.info(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")

    logging.info("Setting up combined dataset and dataloader...")
    try:
         logging.info("Pre-checking specified datasets for availability...")
         for ds_name in CONFIG["datasets"]:
              logging.info(f"Attempting to quick-load '{ds_name}' to check cache/availability...")
              try:
                  # Load a tiny slice to check if dataset is accessible and caches are populated
                  _ = load_dataset(ds_name, split="train[:1%]", download_mode="reuse_cache_if_exists", trust_remote_code=True)
                  logging.info(f"Successfully quick-loaded '{ds_name}'.")
              except Exception as e:
                  logging.warning(f"Could not pre-check/quick-load dataset '{ds_name}': {e}. Full load will proceed but might take time or fail.")

         dataset = CombinedChatDataset(tokenizer) # Pass tokenizer instance
         if len(dataset) == 0:
             logging.error("Dataset is empty after processing all specified sources. Cannot train.")
             return

         # Determine num_workers carefully
         cpu_count = os.cpu_count()
         num_workers = 0 # Default to 0 for main process loading
         if cpu_count and cpu_count > 1:
            if torch.cuda.is_available(): # More workers if GPU is bottlenecked by CPU
                num_workers = min(4, cpu_count // 2)
            else: # Fewer if CPU is doing both compute and loading
                num_workers = min(2, cpu_count // 2)
         num_workers = max(0, num_workers) # Ensure non-negative

         logging.info(f"Using num_workers: {num_workers} for DataLoader.")

         dataloader = DataLoader(
             dataset,
             batch_size=CONFIG["batch_size"],
             collate_fn=CombinedChatDataset.collate_fn, # Static method, no instance needed
             shuffle=True,
             num_workers=num_workers,
             pin_memory=torch.cuda.is_available(), # Pin memory if using CUDA
             prefetch_factor=2 if num_workers > 0 else None, # Prefetch if using multiple workers
             drop_last=False # Process all data, even if last batch is smaller
         )
    except Exception as e:
         logging.error(f"Failed to initialize dataset/dataloader: {e}", exc_info=True)
         return

    logging.info("Initializing Trainer, Checkpoint Manager, and Safety Manager...")
    trainer_obj = HROMTrainer(model, tokenizer) # Pass tokenizer instance
    checkpoint_manager = CheckpointManager()
    safety = SafetyManager(model, tokenizer) # Pass tokenizer instance

    start_optimizer_step = checkpoint_manager.load_latest(model, trainer_obj.optimizer)
    model.to(trainer_obj.device) # Ensure model is on the correct device after loading state

    logging.info(f"Starting/Resuming training from optimizer step {start_optimizer_step}")
    optimizer_step = start_optimizer_step
    
    accum_main_loss_for_log = 0.0
    accum_aux_loss_for_log = 0.0
    
    # Estimate current batch step and epoch based on loaded optimizer_step
    # This is an approximation if dataloader length varies or if not all epochs are full
    batches_per_epoch_est = len(dataloader) if len(dataloader) > 0 else 1 # Avoid division by zero
    current_total_batch_steps = optimizer_step * CONFIG["grad_accum_steps"]
    start_epoch = current_total_batch_steps // batches_per_epoch_est if batches_per_epoch_est > 0 else 0

    try:
        if len(dataloader) == 0: raise ValueError("DataLoader has zero length, cannot train.")
        total_optimizer_steps_estimate = (len(dataloader) * CONFIG["num_epochs"]) // CONFIG["grad_accum_steps"]
        logging.info(f"Dataset size: {len(dataset)} samples, Batches per epoch: {len(dataloader)}")
        logging.info(f"Gradient Accumulation Steps: {CONFIG['grad_accum_steps']}, Effective Batch Size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
        logging.info(f"Target Epochs: {CONFIG['num_epochs']}, Estimated Total Optimizer Steps: {total_optimizer_steps_estimate}")
    except Exception as e:
        logging.warning(f"Could not fully estimate training steps due to: {e}")

    model.train() # Ensure model is in training mode
    for epoch in range(start_epoch, CONFIG["num_epochs"]):
        logging.info(f"--- Starting Epoch {epoch+1}/{CONFIG['num_epochs']} (Optimizer step: {optimizer_step}) ---")
        epoch_main_loss_sum = 0.0
        epoch_aux_loss_sum = 0.0
        epoch_batches_processed = 0 # Batches processed within this epoch execution

        # Skip batches if resuming mid-epoch (simplified: we restart epoch if start_epoch > 0)
        # More precise mid-epoch resumption would require saving/loading dataloader state or batch index.
        # For now, if start_epoch > 0, previous epochs are considered complete.

        for i, batch in enumerate(dataloader):
            if batch is None: # Should be handled by collate_fn returning None for entirely bad batches
                 logging.warning(f"Skipping None batch at index {i} in epoch {epoch+1}. This might indicate data issues.")
                 continue
            if not batch["input_ids"].numel(): # Check if batch tensors are empty
                logging.warning(f"Skipping batch with empty tensors at index {i} in epoch {epoch+1}.")
                continue

            main_loss_val, aux_loss_val = trainer_obj.train_step(batch)

            valid_loss = True
            if main_loss_val is None or math.isnan(main_loss_val) or math.isinf(main_loss_val):
                 logging.error(f"NaN/Inf main loss detected: {main_loss_val}. Aux: {aux_loss_val}. Optimizer Step {optimizer_step}. Stopping training.")
                 checkpoint_manager.save(model, trainer_obj.optimizer, f"error_main_loss_nan_inf_step{optimizer_step}")
                 valid_loss = False
            if aux_loss_val is None or math.isnan(aux_loss_val) or math.isinf(aux_loss_val):
                 logging.warning(f"NaN/Inf auxiliary loss detected: {aux_loss_val}. Main: {main_loss_val}. Optimizer Step {optimizer_step}. This is problematic.")
                 # If main_loss was also bad, we're already stopping.
                 # If only aux_loss is bad, the total_loss will be NaN/Inf, potentially corrupting gradients.
                 # Consider setting aux_loss_val = 0.0 if this becomes a frequent issue and main_loss is fine.
                 if not valid_loss: return # Already stopping from main_loss error

            if not valid_loss: return # Stop training if critical loss error

            accum_main_loss_for_log += main_loss_val
            accum_aux_loss_for_log += aux_loss_val
            epoch_main_loss_sum += main_loss_val
            epoch_aux_loss_sum += aux_loss_val
            epoch_batches_processed += 1
            current_total_batch_steps += 1 # This tracks raw batches processed by train_step

            if current_total_batch_steps % CONFIG["grad_accum_steps"] == 0:
                current_lr = trainer_obj.clip_and_step(optimizer_step) # Perform optimizer step

                avg_main_loss_accum = accum_main_loss_for_log / CONFIG["grad_accum_steps"]
                avg_aux_loss_accum = accum_aux_loss_for_log / CONFIG["grad_accum_steps"]
                accum_main_loss_for_log = 0.0 # Reset accumulators
                accum_aux_loss_for_log = 0.0

                if optimizer_step % CONFIG["debug_interval"] == 0:
                    logging.info(
                        f"E {epoch+1} | OptSt {optimizer_step} | TotalBatchSt {current_total_batch_steps} | "
                        f"AvgMainL: {avg_main_loss_accum:.4f} | AvgAuxL: {avg_aux_loss_accum:.4f} | LR: {current_lr:.2e}"
                    )
                    # Perform debug generation less frequently to save time
                    if optimizer_step > 0 and optimizer_step % (CONFIG["debug_interval"] * 10) == 0: # e.g., every 5 * 400 = 2000 opt steps
                         safety.debug_generation("<user> Hi there! How are you doing today? </s>")

                if optimizer_step > 0 and optimizer_step % CONFIG["checkpoint_interval"] == 0:
                    logging.info(f"Checkpoint interval reached at optimizer step {optimizer_step}.")
                    checkpoint_manager.save(model, trainer_obj.optimizer, optimizer_step) # Save with optimizer_step
                    # Optionally run debug generation after checkpointing
                    # safety.debug_generation("<user> What is the capital of France? </s>")
                
                optimizer_step += 1 # Increment optimizer_step *after* an optimization step

        avg_epoch_main_loss = epoch_main_loss_sum / epoch_batches_processed if epoch_batches_processed > 0 else 0
        avg_epoch_aux_loss = epoch_aux_loss_sum / epoch_batches_processed if epoch_batches_processed > 0 else 0
        logging.info(
            f"--- Finished Epoch {epoch+1}/{CONFIG['num_epochs']} --- "
            f"Avg Epoch MainL: {avg_epoch_main_loss:.4f} | Avg Epoch AuxL: {avg_epoch_aux_loss:.4f} | "
            f"Optimizer Steps this epoch: {optimizer_step - (start_optimizer_step if epoch == start_epoch else current_epoch_start_opt_step)} (approx)"
        )
        # Save checkpoint at the end of each epoch
        # Use a string that includes epoch and current optimizer step
        checkpoint_manager.save(model, trainer_obj.optimizer, f"epoch{epoch+1}_step{optimizer_step}")
        safety.debug_generation("<user> That was an interesting epoch. What did you learn? </s>")
        
        # For next epoch's calculation of "Optimizer Steps this epoch"
        current_epoch_start_opt_step = optimizer_step 


    logging.info(f"Training finished after {CONFIG['num_epochs']} target epochs. Final optimizer step: {optimizer_step}.")
    logging.info("Saving final model state...")
    checkpoint_manager.save(model, trainer_obj.optimizer, f"final_step{optimizer_step}")
    safety.debug_generation("<user> The training is complete. How do you feel? </s>")


if __name__ == "__main__":
    # For reproducibility, consider setting random seeds early
    # random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(42)
    train()
