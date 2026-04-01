"""
Layer 2: LLM Hidden State Cosine Similarity Analysis.

For each of 4 analysis models, under Visual and V+T conditions on T1 questions:
  - Extract LLM last-layer hidden states
  - Compute cosine similarity between last-token representation and
    per-modality (video/diagram/text) token representations
  - Compare Visual vs V+T to verify that diagram information decreases
    in the prediction representation when text is added

Usage:
  python analysis/layer2_hidden_states.py --model Qwen3-VL-8B --n 100
"""
import json
import gc
import time
import argparse
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from representation_utils import (
    ANALYSIS_MODELS, BASE,
    load_analysis_model, print_model_structure,
    _internvl_dynamic_preprocess,
    get_cache_dir,
)

# Import eval_benchmark from the ikea_bench package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ikea_bench"))
from eval_benchmark import (
    build_prompt_and_images, _ensure_imports,
)


# ------------------------------------------------------
# Token categorization (shared with layer3_attention.py)
# ------------------------------------------------------
def find_vision_spans_qwen(input_ids, processor):
    """Find vision spans in Qwen model input_ids."""
    tok = processor.tokenizer
    vs_id = tok.convert_tokens_to_ids("<|vision_start|>")
    ve_id = tok.convert_tokens_to_ids("<|vision_end|>")

    ids = input_ids[0].tolist()
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] == vs_id:
            start = i
            while i < len(ids) and ids[i] != ve_id:
                i += 1
            end = i + 1 if i < len(ids) else i
            spans.append((start, end))
        i += 1
    return spans


def find_vision_spans_internvl(input_ids, model):
    """Find vision spans in InternVL input_ids."""
    img_ctx_id = model.img_context_token_id
    ids = input_ids[0].tolist()
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] == img_ctx_id:
            start = i
            while i < len(ids) and ids[i] == img_ctx_id:
                i += 1
            spans.append((start, i))
        else:
            i += 1
    return spans


def make_token_type_array(seq_len, spans, n_video, n_diagram):
    """Assign token types: 0=text, 1=video, 2=diagram."""
    token_type = np.zeros(seq_len, dtype=np.int32)
    for i, (start, end) in enumerate(spans):
        if i < n_video:
            token_type[start:end] = 1  # video
        elif i < n_video + n_diagram:
            token_type[start:end] = 2  # diagram
    return token_type


# ------------------------------------------------------
# LLM last-layer hook
# ------------------------------------------------------
def get_llm_last_layer(model, family):
    """Get the last LLM decoder layer for hooking."""
    if family in ("qwen2.5vl", "qwen3vl", "qwen3.5"):
        # All Qwen: ForConditionalGeneration.model.language_model.layers
        return model.model.language_model.layers[-1]
    elif family == "internvl":
        return model.language_model.model.layers[-1]


# ------------------------------------------------------
# Tokenization per family
# ------------------------------------------------------
def tokenize_qwen(processor, content, images, family="qwen3vl"):
    """Tokenize for Qwen family models."""
    messages = [{"role": "user", "content": content}]
    chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if family == "qwen3.5":
        chat_kwargs["enable_thinking"] = False
    text = processor.apply_chat_template(messages, **chat_kwargs)
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def tokenize_internvl(model, tokenizer, content, images):
    """Tokenize for InternVL -- build pixel_values + input_ids manually."""
    # Process images
    all_pixel_values = []
    num_patches_list = []
    if images:
        for img in images:
            pv = _internvl_dynamic_preprocess(img, min_num=1, max_num=6)
            all_pixel_values.append(pv)
            num_patches_list.append(pv.shape[0])
        pixel_values = torch.cat(all_pixel_values, dim=0).to(
            dtype=torch.bfloat16, device=model.device
        )
    else:
        pixel_values = None

    # Build prompt
    prompt_parts = []
    for c in content:
        if c["type"] == "text":
            prompt_parts.append(c["text"])
        elif c["type"] == "image":
            prompt_parts.append("<image>")
    full_prompt = "\n".join(prompt_parts)

    return pixel_values, num_patches_list, full_prompt


# ------------------------------------------------------
# Forward with LLM last-layer hook
# ------------------------------------------------------
def forward_with_repr_hook(model, processor_or_tokenizer, family,
                           content, images, n_video, n_diagram):
    """Run forward pass and extract per-modality representation similarities.

    Returns dict with cosine similarities between last token repr and
    average-pooled modality representations at LLM last layer.
    """
    _hook_output = {}
    llm_last = get_llm_last_layer(model, family)

    def _hook(module, input, output):
        if isinstance(output, tuple):
            _hook_output['hidden'] = output[0].detach()
        else:
            _hook_output['hidden'] = output.detach()

    handle = llm_last.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            if family in ("qwen2.5vl", "qwen3vl", "qwen3.5"):
                inputs = tokenize_qwen(processor_or_tokenizer, content, images, family)
                inputs = inputs.to(model.device)
                model(**inputs)

                input_ids = inputs["input_ids"]
                spans = find_vision_spans_qwen(input_ids, processor_or_tokenizer)
                seq_len = input_ids.shape[1]

            elif family == "internvl":
                pixel_values, num_patches_list, prompt = tokenize_internvl(
                    model, processor_or_tokenizer, content, images
                )
                # Use model's internal tokenization for chat
                # We need input_ids to find vision spans
                response, history, input_ids_full = _internvl_forward_for_repr(
                    model, processor_or_tokenizer, pixel_values, num_patches_list, prompt
                )
                spans = find_vision_spans_internvl(input_ids_full, model)
                seq_len = input_ids_full.shape[1]

    finally:
        handle.remove()

    if 'hidden' not in _hook_output:
        return None

    hidden = _hook_output['hidden'][0]  # (seq_len, d)
    token_type = make_token_type_array(seq_len, spans, n_video, n_diagram)

    # Last token representation
    last_repr = hidden[-1].float()  # (d,)

    # Per-modality average representations
    result = {}
    for modality, type_id in [("video", 1), ("diagram", 2), ("text", 0)]:
        mask = torch.from_numpy(token_type == type_id).to(hidden.device)
        if mask.sum() > 0:
            modality_repr = hidden[mask].float().mean(dim=0)  # (d,)
            cos_sim = F.cosine_similarity(last_repr.unsqueeze(0),
                                          modality_repr.unsqueeze(0)).item()
            result[f"cos_{modality}"] = cos_sim
            result[f"n_tokens_{modality}"] = int(mask.sum())
        else:
            result[f"cos_{modality}"] = None
            result[f"n_tokens_{modality}"] = 0

    return result


def _internvl_forward_for_repr(model, tokenizer, pixel_values, num_patches_list, prompt):
    """Run InternVL forward pass and return input_ids for span detection."""
    # Build the full input using model's internal chat method
    # We need to intercept the input_ids before generation
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    # Replace <image> with proper tokens
    image_tokens = ''
    for i, n_patches in enumerate(num_patches_list):
        image_tokens += (
            f'{IMG_START_TOKEN}'
            f'{IMG_CONTEXT_TOKEN * model.num_image_token * n_patches}'
            f'{IMG_END_TOKEN}'
        )
        if i < len(num_patches_list) - 1:
            image_tokens += '\n'

    # Insert image tokens
    prompt_with_images = prompt.replace('<image>', image_tokens, len(num_patches_list))

    # Tokenize
    model_inputs = tokenizer(prompt_with_images, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs.get('attention_mask',
                                      torch.ones_like(input_ids)).to(model.device)

    # Get input embeddings and splice in vision tokens
    input_embeds = model.language_model.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        vit_embeds = model.extract_feature(pixel_values)
        # Handle image_flags
        B, N, C = input_embeds.shape
        input_embeds_flat = input_embeds.reshape(B * N, C)
        input_ids_flat = input_ids.reshape(B * N)
        selected = (input_ids_flat == model.img_context_token_id)

        if selected.sum() > 0 and vit_embeds.numel() > 0:
            vit_embeds_flat = vit_embeds.reshape(-1, C)
            # Ensure sizes match
            n_selected = selected.sum().item()
            n_vit = vit_embeds_flat.shape[0]
            if n_selected == n_vit:
                input_embeds_flat[selected] = vit_embeds_flat.to(input_embeds_flat.dtype)
            else:
                # Size mismatch -- truncate
                min_n = min(n_selected, n_vit)
                indices = selected.nonzero(as_tuple=True)[0][:min_n]
                input_embeds_flat[indices] = vit_embeds_flat[:min_n].to(input_embeds_flat.dtype)

        input_embeds = input_embeds_flat.reshape(B, N, C)

    # Forward pass (no generation, just get hidden states)
    model.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
    )

    return None, None, input_ids


# ------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Layer 2: LLM hidden state cosine similarity analysis"
    )
    parser.add_argument("--model", required=True, help="Model name from ANALYSIS_MODELS")
    parser.add_argument("--data-dir", default=str(BASE / "data"),
                        help="Path to data directory containing benchmark/qa_benchmark.json")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache directory (default: HF_HOME or ~/.cache/huggingface)")
    parser.add_argument("--output", default=None,
                        help="Output JSON (default: results/repr_similarity_{model}.json)")
    parser.add_argument("--n", type=int, default=100, help="Number of T1 questions to analyze")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = get_cache_dir()

    if args.output is None:
        safe_name = args.model.replace(".", "_").replace("-", "_")
        args.output = str(BASE / "results" / f"repr_similarity_{safe_name}.json")

    input_path = Path(args.data_dir) / "benchmark" / "qa_benchmark.json"

    _ensure_imports()

    # Load questions
    with open(input_path) as f:
        data = json.load(f)
    t1_questions = [q for q in data if q["type"] == "1a"][:args.n]
    print(f"Loaded {len(t1_questions)} T1 questions")

    # Load model
    print(f"\nLoading {args.model}...")
    model, processor, family = load_analysis_model(args.model, cache_dir=args.cache_dir)
    info = print_model_structure(model, family, args.model)

    results = {
        "model": args.model,
        "model_info": info,
        "n_questions": len(t1_questions),
        "settings": {},
    }

    for setting in ["baseline", "text_grounding"]:
        print(f"\n--- Setting: {setting} ---")
        per_question = []

        for qi, q in enumerate(t1_questions):
            if qi % 20 == 0:
                print(f"  [{qi}/{len(t1_questions)}]...")

            content, images = build_prompt_and_images(q, setting)
            n_video = len(q.get("video_frames", []))
            n_diagram = sum(1 for o in q["options"] if "image" in o) if setting != "text_only" else 0

            sim = forward_with_repr_hook(
                model, processor, family,
                content, images, n_video, n_diagram
            )

            if sim is not None:
                sim["question_id"] = q["id"]
                per_question.append(sim)

        # Aggregate
        if per_question:
            agg = {}
            for key in ["cos_video", "cos_diagram", "cos_text"]:
                vals = [pq[key] for pq in per_question if pq.get(key) is not None]
                if vals:
                    agg[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)),
                        "n": len(vals),
                    }

            results["settings"][setting] = {
                "aggregate": agg,
                "per_question": per_question,
            }

            print(f"  Results ({setting}):")
            for key, val in agg.items():
                print(f"    {key}: {val['mean']:.4f} +/- {val['std']:.4f} (n={val['n']})")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Comparison table
    print(f"\n{'='*60}")
    print(f"COMPARISON: {args.model}")
    print(f"{'='*60}")
    print(f"{'Modality':<12} {'Visual':>12} {'V+T':>12} {'Delta':>12}")
    print("-" * 48)
    for mod in ["cos_video", "cos_diagram", "cos_text"]:
        vis = results["settings"].get("baseline", {}).get("aggregate", {}).get(mod, {})
        vt = results["settings"].get("text_grounding", {}).get("aggregate", {}).get(mod, {})
        v_mean = vis.get("mean", float('nan'))
        vt_mean = vt.get("mean", float('nan'))
        delta = vt_mean - v_mean
        label = mod.replace("cos_", "")
        print(f"{label:<12} {v_mean:>12.4f} {vt_mean:>12.4f} {delta:>+12.4f}")


if __name__ == "__main__":
    main()
