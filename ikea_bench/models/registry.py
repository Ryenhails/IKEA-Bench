"""Model registry for IKEA-Bench supported VLMs."""

import os

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    # === Qwen2.5-VL family ===
    "qwen2.5-vl-3b": {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2.5vl",
    },
    "qwen2.5-vl-7b": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2.5vl",
    },
    # === Qwen3-VL family (Instruct) ===
    "qwen3-vl-2b": {
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",
        "family": "qwen3vl",
    },
    "qwen3-vl-8b": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3vl",
    },
    "qwen3-vl-30b-a3b": {
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "family": "qwen3vl_moe",
    },
    # === Qwen3.5 family (unified multimodal) ===
    "qwen3.5-2b": {
        "model_id": "Qwen/Qwen3.5-2B",
        "family": "qwen3.5",
    },
    "qwen3.5-9b": {
        "model_id": "Qwen/Qwen3.5-9B",
        "family": "qwen3.5",
    },
    "qwen3.5-27b": {
        "model_id": "Qwen/Qwen3.5-27B",
        "family": "qwen3.5",
    },
    # === InternVL3.5 family ===
    "internvl3.5-2b": {
        "model_id": "OpenGVLab/InternVL3_5-2B",
        "family": "internvl",
    },
    "internvl3.5-8b": {
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "family": "internvl",
    },
    "internvl3.5-38b": {
        "model_id": "OpenGVLab/InternVL3_5-38B",
        "family": "internvl",
    },
    # === Gemma3 family ===
    "gemma3-4b": {
        "model_id": "google/gemma-3-4b-it",
        "family": "gemma3",
    },
    "gemma3-12b": {
        "model_id": "google/gemma-3-12b-it",
        "family": "gemma3",
    },
    "gemma3-27b": {
        "model_id": "google/gemma-3-27b-it",
        "family": "gemma3",
    },
    # === Other families ===
    "llava-ov-1.5-8b": {
        "model_id": "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        "family": "llava_ov",
    },
    "minicpm-v-4.5": {
        "model_id": "openbmb/MiniCPM-V-4_5",
        "family": "minicpm",
    },
    "glm-4.1v-9b": {
        "model_id": "THUDM/GLM-4.1V-9B-Thinking",
        "family": "glm4v",
    },
    "molmo2-8b": {
        "model_id": "allenai/Molmo2-8B",
        "family": "molmo",
    },
    "phi4-mm": {
        "model_id": "microsoft/Phi-4-multimodal-instruct",
        "family": "phi4",
    },
    # === Flagship models (multi-GPU) ===
    "qwen3.5-397b-a17b": {
        "model_id": "Qwen/Qwen3.5-397B-A17B",
        "family": "qwen3.5",
    },
    "qwen3-vl-235b-a22b": {
        "model_id": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "family": "qwen3vl_moe",
    },
    "internvl3-78b": {
        "model_id": "OpenGVLab/InternVL3-78B",
        "family": "internvl",
    },
    "llama4-scout-17b-16e": {
        "model_id": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "family": "llama4",
    },
}


def get_cache_dir() -> str:
    """Return the HuggingFace cache directory from environment or default."""
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
