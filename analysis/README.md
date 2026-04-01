# Mechanistic Analysis Scripts

Three-layer mechanistic analysis for understanding how vision-language models process assembly instructions in IKEA-Bench.

## Overview

| Layer | Script | What it measures |
|-------|--------|-----------------|
| Layer 1 | `layer1_representation.py` | CKA + linear probe + cross-modal retrieval (representation alignment) |
| Layer 2 | `layer2_hidden_states.py` | LLM hidden state cosine similarity (modality influence on prediction) |
| Layer 3 | `layer3_attention.py` | Attention routing analysis (per-modality attention shares) |

Shared utilities live in `representation_utils.py`.

## Layer 1: Representation Alignment

Extracts ViT and merger/projector representations for diagram and video frame images, then measures:

- **CKA** (Centered Kernel Alignment): How similar are diagram vs. video representations?
- **Linear probe**: Can a logistic regression distinguish same-step vs. different-step video frames?
- **Cross-modal retrieval**: Given a diagram, can we retrieve the correct video frame by cosine similarity?

```bash
python analysis/layer1_representation.py \
    --models all \
    --data-dir data/ \
    --cache-dir /path/to/hf_cache \
    --output results/representation_analysis.json
```

## Layer 2: Hidden State Similarity

For T1 questions under Visual and V+T conditions, extracts the LLM's last-layer hidden states and computes cosine similarity between the last-token (prediction) representation and per-modality (video, diagram, text) token representations. A drop in diagram similarity under V+T indicates that text descriptions redirect the model away from diagram features.

```bash
python analysis/layer2_hidden_states.py \
    --model Qwen3-VL-8B \
    --data-dir data/ \
    --cache-dir /path/to/hf_cache \
    --output results/repr_similarity_Qwen3_VL_8B.json \
    --n 100
```

## Layer 3: Attention Routing

Runs on Qwen3-VL-8B. Monkey-patches flash-attention layers to capture last-token attention weights without the O(seq^2) memory cost. Three sub-analyses:

1. **Cross-modal attention shift**: Compares per-modality attention shares under Visual vs. V+T on T1 questions.
2. **Masking ablation**: Replaces text descriptions with neutral placeholders; if accuracy recovers to Visual level, the text content (not just its presence) caused the mode switch.
3. **D1 video attention balance**: Measures whether attention is split evenly between clip A and clip B, regardless of whether they show the same or different steps.

```bash
python analysis/layer3_attention.py \
    --data-dir data/ \
    --cache-dir /path/to/hf_cache \
    --output results/attention_analysis.json \
    --n_attn 100 \
    --n_d1 100
```

Add `--skip_masking` to skip the masking ablation (which requires generation and is slow), or `--skip_attention` to skip the attention distribution analysis.

## Environment Variables

- `HF_HOME`: HuggingFace cache directory (default: `~/.cache/huggingface`). Can also be set via `--cache-dir`.

## Dependencies

All scripts require the `ikea_bench` package (specifically `eval_benchmark.py` for prompt building). Install from the project root or ensure `ikea_bench/` is on your Python path.
