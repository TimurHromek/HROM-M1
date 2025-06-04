# HROM-M1

**HROM-M1** is a transformer-based Mixture-of-Experts (MoE) language model built entirely in PyTorch by me, *Timur Hromek*, a 15-year-old self-taught developer. It's designed for multi-turn, persona-aware dialogue with a focus on safety, modularity, and extensibility.

This implementation includes top-k expert routing, rotary position embeddings, SwiGLU activations, and a custom tokenizer, along with built-in safety filters and checkpoint management.

## Features

- Mixture-of-Experts (MoE) with 8 experts and top-2 routing per token.
- Transformer architecture with 8 layers, 8 heads, and RoPE (rotary positional embeddings).
- SwiGLU activation for efficient MLP computation.
- Multi-dataset training support, including:
  - `daily_dialog`
  - `empathetic_dialogues`
  - `blended_skill_talk`
  - `persona-chat`
- Custom tokenizer using Byte-Pair Encoding (BPE).
- `SafetyManager` for blocking unsafe generations using token-level filtering.
- `CheckpointManager` with rotating save slots and auto-recovery.
- AMP (mixed precision) and gradient accumulation support.

## Model Specs

| Hyperparameter            | Value          |
|--------------------------|----------------|
| Embedding Size (dim)     | 768            |
| Layers                   | 8              |
| Attention Heads          | 8              |
| Expert FF Dim            | 2048           |
| Number of Experts        | 8              |
| Top-k Experts            | 2              |
| Vocabulary Size          | 32,000         |
| Max Sequence Length      | 512 tokens     |
| Dropout                  | 0.1            |
| Batch Size               | 16             |
| Learning Rate            | 2e-5           |
| Optimizer                | AdamW          |
| Epochs                   | 30             |
| Grad Accumulation Steps  | 8              |

## Architecture Overview

- `HROMBlock`: Transformer block with attention and MoE feedforward.
- `MoELayer`: Routes tokens to top-k experts and applies load balancing loss.
- `Expert`: Lightweight FFN with SwiGLU nonlinearity.
- `SafetyManager`: Filters generations using predefined token patterns.
- `TokenizerTrainer`: Builds a BPE tokenizer from dialogue data.
- `CheckpointManager`: Rotates and auto-recovers checkpoints.

## Safety

The model includes a basic content filter that blocks sequences containing unsafe keywords by checking token IDs. Unsafe generations are interrupted before completion.

## Installation

```bash
git clone https://github.com/yourusername/HROM-M1.git
cd HROM-M1
pip install -r requirements.txt
```

## Training

```bash
python HROM-M1.py
```

The tokenizer will auto-train if not found. Dialogue datasets are pulled via HuggingFace `datasets`.
