"""
Shared utilities for representation-level analyses (CKA, linear probe, repr similarity).
Supports 4 analysis models: Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3.5-VL-9B, InternVL3.5-8B.
"""
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional

BASE = Path(__file__).resolve().parent.parent

ANALYSIS_MODELS = {
    "Qwen2.5-VL-7B": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2.5vl",
    },
    "Qwen3-VL-8B": {
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen3vl",
    },
    "Qwen3.5-VL-9B": {
        "model_id": "Qwen/Qwen3.5-9B",
        "family": "qwen3.5",
    },
    "InternVL3.5-8B": {
        "model_id": "OpenGVLab/InternVL3_5-8B",
        "family": "internvl",
    },
}


def get_cache_dir():
    """Return the HuggingFace cache directory from env or default."""
    return os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


# ------------------------------------------------------
# Model loading
# ------------------------------------------------------
def load_analysis_model(model_name, cache_dir=None):
    """Load one of the 4 analysis models. Returns (model, processor_or_tokenizer, family)."""
    cfg = ANALYSIS_MODELS[model_name]
    family = cfg["family"]
    if cache_dir is None:
        cache_dir = get_cache_dir()

    if family == "qwen2.5vl":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=cache_dir)
        model.eval()
        return model, processor, family

    elif family == "qwen3vl":
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=cache_dir)
        model.eval()
        return model, processor, family

    elif family == "qwen3.5":
        from transformers import AutoModelForImageTextToText, AutoProcessor
        model = AutoModelForImageTextToText.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="sdpa",
        )
        processor = AutoProcessor.from_pretrained(cfg["model_id"], cache_dir=cache_dir)
        model.eval()
        return model, processor, family

    elif family == "internvl":
        from transformers import AutoModel, AutoTokenizer, PreTrainedModel
        # Patch for newer transformers compatibility with InternVL custom code
        if not hasattr(PreTrainedModel, '_original_mark_tied'):
            _orig = PreTrainedModel.mark_tied_weights_as_initialized
            def _safe_mark_tied(self, *args, **kwargs):
                if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
                    self.all_tied_weights_keys = getattr(self, '_tied_weights_keys', None) or {}
                return _orig(self, *args, **kwargs)
            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark_tied
            PreTrainedModel._original_mark_tied = True

        model = AutoModel.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            use_flash_attn=True, trust_remote_code=True,
        ).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_id"], cache_dir=cache_dir,
            trust_remote_code=True, use_fast=False,
        )
        # Initialize img_context_token_id (normally set in chat() method)
        if model.img_context_token_id is None:
            model.img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        return model, tokenizer, family


# ------------------------------------------------------
# ViT structure discovery
# ------------------------------------------------------
def _get_visual(model, family):
    """Get the visual encoder module, handling Qwen's nested structure."""
    if family == "internvl":
        return model.vision_model
    # All Qwen families: ForConditionalGeneration.model.visual
    return model.model.visual


def _find_vit_last_block(model, family):
    """Find the last ViT transformer block for hooking."""
    if family == "internvl":
        return model.vision_model.encoder.layers[-1]
    # All Qwen families use model.model.visual.blocks
    return model.model.visual.blocks[-1]


def _find_merger(model, family):
    """Find the merger/projector module."""
    if family == "internvl":
        return model.mlp1
    return model.model.visual.merger


def print_model_structure(model, family, model_name):
    """Print key structural info for verification."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name} (family={family})")

    vit_block = _find_vit_last_block(model, family)
    merger = _find_merger(model, family)
    print(f"  ViT last block: {type(vit_block).__name__}")
    print(f"  Merger/Projector: {type(merger).__name__}")

    if family == "internvl":
        n_vit = len(model.vision_model.encoder.layers)
        n_llm = len(model.language_model.model.layers)
        vit_dim = model.vision_model.config.hidden_size
        llm_dim = model.language_model.config.hidden_size
    else:
        # All Qwen families: model.model.visual.blocks, model.model.language_model.layers
        visual = model.model.visual
        n_vit = len(visual.blocks)
        n_llm = len(model.model.language_model.layers)
        vit_dim = model.config.vision_config.hidden_size
        llm_dim = model.config.text_config.hidden_size

    print(f"  ViT layers: {n_vit}, ViT hidden: {vit_dim}")
    print(f"  LLM layers: {n_llm}, LLM hidden: {llm_dim}")
    print(f"{'='*60}\n")
    return {"n_vit": n_vit, "n_llm": n_llm, "vit_dim": vit_dim, "llm_dim": llm_dim}


# ------------------------------------------------------
# Image preprocessing for InternVL
# ------------------------------------------------------
def _internvl_dynamic_preprocess(image, min_num=1, max_num=6, image_size=448):
    """Dynamic tiling for InternVL -- same as eval_benchmark.py."""
    from torchvision import transforms as T

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set()
    for n in range(1, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = (1, 1)
    best_ratio_diff = float('inf')
    area = orig_width * orig_height
    for ratio in target_ratios:
        target_aspect = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio

    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized = image.resize((target_width, target_height))
    processed_blocks = []
    for i in range(best_ratio[1]):
        for j in range(best_ratio[0]):
            box = (j * image_size, i * image_size,
                   (j + 1) * image_size, (i + 1) * image_size)
            processed_blocks.append(resized.crop(box))

    if blocks != 1:
        thumbnail = image.resize((image_size, image_size))
        processed_blocks.append(thumbnail)

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    pixel_values = torch.stack([transform(block) for block in processed_blocks])
    return pixel_values


# ------------------------------------------------------
# ViT representation extraction
# ------------------------------------------------------
def extract_vit_representations(model, processor, family, image_paths, verbose=True):
    """Extract ViT last-layer and merger/projector output for each image.

    Returns:
        vit_reprs:    dict[path -> tensor(d_vit,)]    -- pre-merger, average-pooled
        merger_reprs: dict[path -> tensor(d_merger,)] -- post-merger, average-pooled
    """
    vit_reprs = {}
    merger_reprs = {}

    # Hook on ViT last block
    _hook_output = {}
    vit_last = _find_vit_last_block(model, family)
    def _vit_hook(module, input, output):
        if isinstance(output, tuple):
            _hook_output['vit'] = output[0].detach()
        else:
            _hook_output['vit'] = output.detach()
    handle = vit_last.register_forward_hook(_vit_hook)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    try:
        for idx, path in enumerate(image_paths):
            if verbose and idx % 200 == 0:
                print(f"  Extracting [{idx}/{len(image_paths)}]...")

            img = Image.open(path).convert("RGB")
            _hook_output.clear()

            with torch.no_grad():
                if family in ("qwen2.5vl", "qwen3vl", "qwen3.5"):
                    image_inputs = processor.image_processor(images=[img], return_tensors="pt")
                    pixel_values = image_inputs["pixel_values"].to(device, dtype=dtype)
                    grid_thw = image_inputs["image_grid_thw"].to(device)
                    visual_out = model.model.visual(pixel_values, grid_thw=grid_thw)
                    # Handle BaseModelOutputWithPooling or raw tensor
                    if hasattr(visual_out, 'pooler_output') and visual_out.pooler_output is not None:
                        merger_out = visual_out.pooler_output
                    elif hasattr(visual_out, 'last_hidden_state'):
                        merger_out = visual_out.last_hidden_state
                    elif isinstance(visual_out, torch.Tensor):
                        merger_out = visual_out
                    else:
                        raise TypeError(f"Unexpected visual output type: {type(visual_out)}")
                    if merger_out.dim() == 3:
                        merger_out = merger_out.reshape(-1, merger_out.shape[-1])
                    merger_reprs[path] = merger_out.float().mean(dim=0).cpu()

                elif family == "internvl":
                    pv = _internvl_dynamic_preprocess(img).to(device, dtype=dtype)
                    mlp1_out = model.extract_feature(pv)
                    if mlp1_out.dim() == 3:
                        mlp1_out = mlp1_out.reshape(-1, mlp1_out.shape[-1])
                    merger_reprs[path] = mlp1_out.float().mean(dim=0).cpu()

                # ViT last-layer output (from hook)
                if 'vit' in _hook_output:
                    vit_out = _hook_output['vit']
                    if vit_out.dim() == 3:
                        # (batch, seq, hidden) or (n_tiles, seq, hidden)
                        vit_out = vit_out.reshape(-1, vit_out.shape[-1])
                    # InternVL: first token is CLS, skip it
                    if family == "internvl":
                        n_patches_per_tile = (448 // 14) ** 2  # 1024
                        n_tiles = vit_out.shape[0] // (n_patches_per_tile + 1)
                        if n_tiles >= 1:
                            # Remove CLS from each tile
                            chunks = vit_out.reshape(n_tiles, n_patches_per_tile + 1, -1)
                            vit_out = chunks[:, 1:, :].reshape(-1, chunks.shape[-1])
                    vit_reprs[path] = vit_out.float().mean(dim=0).cpu()

    finally:
        handle.remove()

    if verbose:
        print(f"  Extracted {len(vit_reprs)} ViT reprs, {len(merger_reprs)} merger reprs")
        if vit_reprs:
            sample = next(iter(vit_reprs.values()))
            print(f"  ViT repr dim: {sample.shape[0]}")
        if merger_reprs:
            sample = next(iter(merger_reprs.values()))
            print(f"  Merger repr dim: {sample.shape[0]}")

    return vit_reprs, merger_reprs


# ------------------------------------------------------
# CKA computation
# ------------------------------------------------------
def linear_cka(X, Y):
    """Compute Linear CKA (Kornblith et al., 2019) between centered matrices.

    Args:
        X: (n, d1) numpy array
        Y: (n, d2) numpy array
    Returns:
        cka: float in [0, 1]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = np.linalg.norm(YtX, 'fro') ** 2
    denominator = np.linalg.norm(XtX, 'fro') * np.linalg.norm(YtY, 'fro')

    if denominator < 1e-10:
        return 0.0
    return float(numerator / denominator)


def bootstrap_cka(X, Y, n_bootstrap=1000, seed=42):
    """Compute CKA with bootstrap 95% CI over examples (rows)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    cka_full = linear_cka(X, Y)

    boot_vals = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_vals.append(linear_cka(X[idx], Y[idx]))

    boot_vals = np.array(boot_vals)
    ci_low = float(np.percentile(boot_vals, 2.5))
    ci_high = float(np.percentile(boot_vals, 97.5))
    return {"cka": cka_full, "ci_low": ci_low, "ci_high": ci_high, "n": n}


# ------------------------------------------------------
# Data collection from benchmark
# ------------------------------------------------------
def collect_image_data(benchmark_path=None, data_dir=None):
    """Collect all unique images with metadata from qa_benchmark.json.

    Returns:
        all_image_paths: list of unique image paths (diagrams + video frames)
        step_pairs: dict[(product, step_id) -> {"diagram": path, "video_frames": set(paths)}]
        image_meta: dict[path -> {"type": "diagram"|"video", "product": str, "step_id": int}]
    """
    if benchmark_path is None:
        if data_dir is None:
            data_dir = BASE / "data"
        benchmark_path = Path(data_dir) / "benchmark" / "qa_benchmark.json"

    with open(benchmark_path) as f:
        data = json.load(f)

    image_meta = {}
    step_pairs = {}

    for q in data:
        product = q["product"]

        # Video frames (T1, T2, T3, T4)
        if "video_frames" in q:
            step_id = q.get("answer_step_id", None)
            if step_id is not None:
                for fp in q["video_frames"]:
                    if fp not in image_meta:
                        image_meta[fp] = {"type": "video", "product": product, "step_id": step_id}
                    key = (product, step_id)
                    if key not in step_pairs:
                        step_pairs[key] = {"diagram": None, "video_frames": set()}
                    step_pairs[key]["video_frames"].add(fp)

        # Diagram images from options (T1, T2, T4 -- options with "image" key)
        if "options" in q:
            correct_label = q.get("answer", "")
            for opt in q["options"]:
                if "image" in opt:
                    img_path = opt["image"]
                    opt_step_id = opt.get("step_id", None)
                    if opt_step_id is not None and img_path not in image_meta:
                        image_meta[img_path] = {"type": "diagram", "product": product, "step_id": opt_step_id}
                    # If correct option, assign as diagram for this step
                    if opt.get("label") == correct_label and opt_step_id is not None:
                        key = (product, q.get("answer_step_id", opt_step_id))
                        if key not in step_pairs:
                            step_pairs[key] = {"diagram": None, "video_frames": set()}
                        step_pairs[key]["diagram"] = img_path

        # T3 (2a): manual_images -- dict mapping step_id -> image_path
        if "manual_images" in q and isinstance(q["manual_images"], dict):
            for sid_str, img_path in q["manual_images"].items():
                sid = int(sid_str)
                if img_path not in image_meta:
                    image_meta[img_path] = {"type": "diagram", "product": product, "step_id": sid}
                key = (product, sid)
                if key not in step_pairs:
                    step_pairs[key] = {"diagram": None, "video_frames": set()}
                if step_pairs[key]["diagram"] is None:
                    step_pairs[key]["diagram"] = img_path

        # D2 (2c): step_images -- list of {label, step_id, image}
        if "step_images" in q:
            for si in q["step_images"]:
                if "image" in si:
                    img_path = si["image"]
                    sid = si.get("step_id", None)
                    if sid is not None and img_path not in image_meta:
                        image_meta[img_path] = {"type": "diagram", "product": product, "step_id": sid}

        # D1 (1c): video_frames_a, video_frames_b -- step info in metadata
        for vf_key in ("video_frames_a", "video_frames_b"):
            if vf_key in q:
                step_key = "step_a" if vf_key == "video_frames_a" else "step_b"
                meta_step = q.get("metadata", {}).get(step_key, None)
                for fp in q[vf_key]:
                    if fp not in image_meta:
                        image_meta[fp] = {"type": "video", "product": product,
                                          "step_id": meta_step}

    # Convert sets to sorted lists
    for key in step_pairs:
        step_pairs[key]["video_frames"] = sorted(step_pairs[key]["video_frames"])

    # Filter step_pairs to those with both diagram and video frames
    valid_pairs = {k: v for k, v in step_pairs.items()
                   if v["diagram"] is not None and len(v["video_frames"]) > 0}

    all_image_paths = sorted(image_meta.keys())

    print(f"Collected {len(all_image_paths)} unique images "
          f"({sum(1 for v in image_meta.values() if v['type'] == 'diagram')} diagrams, "
          f"{sum(1 for v in image_meta.values() if v['type'] == 'video')} video frames)")
    print(f"Valid step pairs (diagram + video): {len(valid_pairs)}")

    return all_image_paths, valid_pairs, image_meta


def get_product_split(image_meta, test_ratio=0.2, seed=42):
    """Split products into train/test sets for linear probing.
    Every 5th product (sorted) goes to test.
    """
    products = sorted(set(m["product"] for m in image_meta.values()))
    test_products = set(products[i] for i in range(4, len(products), 5))
    train_products = set(products) - test_products
    print(f"Product split: {len(train_products)} train, {len(test_products)} test")
    return train_products, test_products
