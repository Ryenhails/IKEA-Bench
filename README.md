<div align="center">

# IKEA-Bench

**Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment**

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/Ryenhails/ikea-bench)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/XXXX.XXXXX)

</div>

## Overview

IKEA-Bench is a benchmark for evaluating how well vision-language models (VLMs) can align **assembly instruction diagrams** (like IKEA manuals) with **real-world assembly videos**. This *cross-depiction alignment* task requires bridging the gap between schematic line drawings and photorealistic video — a challenge that current VLMs struggle with.

The benchmark includes:
- **1,623 questions** across **6 task types** in a 2D taxonomy (Cross-Modal Alignment x Procedural Reasoning)
- **29 furniture products** from 6 categories, with 98 assembly videos
- **3 alignment strategies**: Visual (baseline), Visual+Text, Text Only
- **19 VLMs** evaluated from 8 model families
- **Three-layer mechanistic analysis** probing why models fail at diagram understanding

### Task Taxonomy

| | Cross-Modal Alignment | Procedural Reasoning |
|---|---|---|
| **Recognition** | T1: Step Recognition (320) | T3: Progress Tracking (334) |
| **Verification** | T2: Action Verification (350) | T4: Next-Step Prediction (204) |
| **Diagnostic** | D1: Video Discrimination (350) | D2: Instruction Comprehension (65) |

## Leaderboard

Results on the **Visual (baseline)** setting. Accuracy (%) per task type.

| Model | Params | T1 | T2 | T3 | T4 | D1 | D2 | Avg |
|-------|--------|-----|-----|-----|-----|-----|-----|-----|
| Gemini-3-Flash | - | 65.3 | 68.6 | 65.6 | 43.1 | 71.1 | 81.5 | 65.9 |
| Gemini-3.1-Pro | - | 62.8 | 65.1 | 65.0 | 41.7 | 67.4 | 76.9 | 63.2 |
| Qwen3.5-27B | 27B | 59.4 | 62.9 | 59.3 | 41.2 | 63.7 | 70.8 | 59.6 |
| Qwen3.5-9B | 9B | 57.8 | 63.7 | 46.7 | 38.2 | 63.1 | 58.5 | 54.7 |
| InternVL3.5-38B | 38B | 54.4 | 61.4 | 47.3 | 37.7 | 61.4 | 67.7 | 55.0 |
| Qwen3-VL-8B | 8B | 53.1 | 56.6 | 49.4 | 39.7 | 58.3 | 58.5 | 52.6 |
| MiniCPM-V-4.5 | 8B | 49.7 | 55.7 | 41.0 | 32.8 | 50.0 | 50.8 | 46.7 |
| Qwen2.5-VL-7B | 7B | 49.1 | 50.9 | 35.0 | 36.8 | 46.0 | 53.8 | 45.3 |
| Qwen3-VL-30B-A3B | 30B | 48.8 | 58.3 | 50.6 | 34.3 | 60.0 | 56.9 | 51.5 |
| GLM-4.1V-9B | 9B | 48.4 | 55.7 | 43.7 | 35.8 | 50.3 | 47.7 | 46.9 |
| Qwen3.5-2B | 2B | 44.4 | 56.6 | 32.9 | 36.3 | 51.4 | 36.9 | 43.1 |
| Gemma3-27B | 27B | 43.1 | 55.7 | 37.1 | 31.4 | 53.7 | 41.5 | 43.8 |
| Qwen2.5-VL-3B | 3B | 42.8 | 51.1 | 35.6 | 28.9 | 48.3 | 52.3 | 43.2 |
| Qwen3-VL-2B | 2B | 42.2 | 50.0 | 29.6 | 34.8 | 50.0 | 26.2 | 38.8 |
| InternVL3.5-8B | 8B | 39.4 | 53.7 | 36.5 | 31.4 | 49.4 | 50.8 | 43.5 |
| Gemma3-4B | 4B | 39.4 | 50.3 | 27.8 | 29.4 | 47.7 | 20.0 | 35.8 |
| Gemma3-12B | 12B | 35.3 | 49.7 | 35.9 | 28.4 | 49.1 | 32.3 | 38.5 |
| LLaVA-OV-8B | 8B | 35.3 | 46.3 | 27.8 | 29.4 | 41.4 | 27.7 | 34.7 |
| InternVL3.5-2B | 2B | 33.4 | 50.3 | 29.9 | 23.0 | 48.6 | 20.0 | 34.2 |
| *Random baseline* | - | 25.0 | 50.0 | 25.0 | 25.0 | 50.0 | 25.0 | 33.3 |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/YOUR_ORG/IKEA-Bench.git
cd IKEA-Bench
pip install -r requirements.txt
```

### 2. Data Setup

The entire dataset (~300MB) is hosted on [HuggingFace](https://huggingface.co/datasets/Ryenhails/ikea-bench), including all images.

```bash
# One command — downloads everything
python setup_data.py
```

Or directly via Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("Ryenhails/ikea-bench", repo_type="dataset", local_dir="data")
```

### 3. Run Evaluation

```bash
# Evaluate a model
python -m ikea_bench.eval \
    --model qwen3-vl-8b \
    --setting baseline \
    --input data/qa_benchmark.json \
    --data-dir data \
    --output results/qwen3-vl-8b_baseline.json

# Available settings: baseline, text_grounding, text_only
# See all supported models:
python -m ikea_bench.eval --help
```

**Gemini API evaluation:**

```bash
export GEMINI_API_KEY=your_key_here
python -m ikea_bench.eval_gemini \
    --setting baseline \
    --data-dir data \
    --output results/gemini_baseline.json
```

### 4. Custom Model Evaluation

To evaluate your own model, you can either:

**Option A:** Add your model to the registry in `ikea_bench/models/registry.py` and implement a loader in `ikea_bench/models/__init__.py`.

**Option B:** Use the prompt builder directly:

```python
from ikea_bench.prompts import build_prompt_and_images
from ikea_bench.utils import extract_answer
import json

data_dir = "data"
with open(f"{data_dir}/qa_benchmark.json") as f:
    questions = json.load(f)

for q in questions:
    content, images = build_prompt_and_images(q, setting="baseline", data_dir=data_dir)
    # content is a list of {"type": "text", "text": ...} or {"type": "image", "image": PIL.Image}
    # Feed to your model, then:
    response = your_model(content, images)
    pred = extract_answer(response, valid_labels=[o["label"] for o in q["options"]])
```

## Mechanistic Analysis

The three-layer analysis probes *why* VLMs struggle with assembly diagrams:

| Layer | What it probes | Scripts |
|-------|----------------|---------|
| **Layer 1**: Representation | CKA, linear probe, cross-modal retrieval on ViT/merger features | `analysis/layer1_representation.py` |
| **Layer 2**: Hidden States | Cosine similarity between LLM hidden states and per-modality tokens | `analysis/layer2_hidden_states.py` |
| **Layer 3**: Attention | Per-modality attention routing across decoder layers | `analysis/layer3_attention.py` |

See [`analysis/README.md`](analysis/README.md) for details and usage.

## Data Construction

The benchmark is deterministically constructed from ground-truth annotations. To reproduce:

```bash
# Build QA from annotations
python data_construction/build_qa.py --data-dir data

# Extract video frames
python data_construction/extract_frames.py --data-dir data

# Remove step numbers from manual images
python data_construction/remove_step_numbers.py --data-dir data
```

## Supported Models

| Family | Models |
|--------|--------|
| Qwen2.5-VL | 3B, 7B |
| Qwen3-VL | 2B, 8B, 30B-A3B |
| Qwen3.5 | 2B, 9B, 27B |
| InternVL3.5 | 2B, 8B, 38B |
| Gemma3 | 4B, 12B, 27B |
| GLM-4.1V | 9B |
| LLaVA-OV | 8B |
| MiniCPM-V | 4.5 |

## Project Structure

```
IKEA-Bench/
├── ikea_bench/              # Core evaluation package
│   ├── eval.py              # Main evaluation CLI
│   ├── eval_gemini.py       # Gemini API evaluation
│   ├── prompts.py           # Prompt construction
│   ├── utils.py             # Shared utilities
│   └── models/              # Model loading & inference
│       ├── __init__.py      # Load & run dispatch
│       └── registry.py      # Model registry
├── analysis/                # Mechanistic analysis
│   ├── layer1_representation.py
│   ├── layer2_hidden_states.py
│   ├── layer3_attention.py
│   └── representation_utils.py
├── data_construction/       # Benchmark construction
│   ├── build_qa.py
│   ├── extract_frames.py
│   └── remove_step_numbers.py
├── setup_data.py            # Data download & preparation
├── scripts/                 # Example run scripts
├── results/                 # Evaluation outputs
├── requirements.txt
└── LICENSE                  # CC-BY-4.0
```

## Citation

If you use IKEA-Bench in your research, please cite:

```bibtex
@article{liu2026ikeabench,
  title={Benchmarking and Mechanistic Analysis of Vision-Language Models for Cross-Depiction Assembly Instruction Alignment},
  author={Liu, Zhuchenyang and Zhang, Yao and Xiao, Yu},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

Please also cite the source dataset:

```bibtex
@inproceedings{liu2024ikeamanualsatwork,
  title={IKEA Manuals at Work: 4D Grounding of Assembly Instructions on Internet Videos},
  author={Liu, Yunong and Eyzaguirre, Cristobal and Li, Manling and Khanna, Shubh and Niebles, Juan Carlos and Ravi, Vineeth and Mishra, Saumitra and Liu, Weiyu and Wu, Jiajun},
  booktitle={NeurIPS 2024 Datasets and Benchmarks},
  year={2024}
}
```

## License

This project is licensed under [CC-BY-4.0](LICENSE). The benchmark annotations, text descriptions, and evaluation code are released under this license. Original IKEA manual images are sourced from the [IKEA Manuals at Work](https://github.com/yunongLiu1/IKEA-Manuals-at-Work) dataset (also CC-BY-4.0).

## Acknowledgments

This benchmark is built upon the [IKEA Manuals at Work](https://github.com/yunongLiu1/IKEA-Manuals-at-Work) dataset by Liu et al. (NeurIPS 2024). We thank the authors for releasing their dataset under an open license.
