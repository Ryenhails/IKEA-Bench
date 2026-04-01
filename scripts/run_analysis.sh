#!/bin/bash
# Example: Run mechanistic analysis on IKEA-Bench
#
# Layer 1: CKA + linear probe + cross-modal retrieval
# Layer 2: LLM hidden state cosine similarity
# Layer 3: Attention routing analysis

# Layer 1: Representation analysis (all 4 analysis models)
python analysis/layer1_representation.py \
    --data-dir data \
    --output results/layer1_representation.json

# Layer 2: Hidden state analysis (per model)
python analysis/layer2_hidden_states.py \
    --model Qwen3-VL-8B \
    --data-dir data \
    --n 100 \
    --output results/layer2_hidden_states.json

# Layer 3: Attention analysis (Qwen3-VL-8B)
python analysis/layer3_attention.py \
    --data-dir data \
    --n_attn 100 \
    --output results/layer3_attention.json
