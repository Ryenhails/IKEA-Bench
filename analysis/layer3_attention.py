"""
Layer 3: Attention Routing Analysis.

Runs on Qwen3-VL-8B as typical model. Three analyses:
  1. Cross-modal attention shift -- Visual vs V+T attention distribution on T1
  2. Text masking ablation -- V+T with descriptions neutralized, compare accuracy
  3. D1 video attention -- attention balance between two video clips

Usage:
  python analysis/layer3_attention.py \
    --data-dir ../data \
    --output results/attention_analysis.json \
    --n_attn 100 --n_d1 100
"""
import json
import argparse
import gc
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

BASE = Path(__file__).resolve().parent.parent

# Import eval_benchmark from the ikea_bench package
sys.path.insert(0, str(BASE / "ikea_bench"))
from eval_benchmark import (
    build_prompt_and_images, extract_answer,
    _ensure_imports, MODEL_CONFIGS,
)

from representation_utils import get_cache_dir

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Target layers for attention analysis (evenly spaced across 36 layers)
TARGET_LAYERS = [0, 7, 15, 23, 31, 35]


# ------------------------------------------------------
# Model loading
# ------------------------------------------------------
def load_model(cache_dir=None, attn_impl="flash_attention_2"):
    """Load Qwen3-VL-8B with specified attention implementation."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    if cache_dir is None:
        cache_dir = get_cache_dir()

    print(f"Loading Qwen3-VL-8B with attn_implementation={attn_impl}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    model.eval()
    print(f"Model loaded. Device: {model.device}")
    return model, processor


# ------------------------------------------------------
# Token categorization
# ------------------------------------------------------
def get_vision_token_ids(processor):
    """Get special vision token IDs."""
    tok = processor.tokenizer
    return {
        "vision_start": tok.convert_tokens_to_ids("<|vision_start|>"),
        "vision_end": tok.convert_tokens_to_ids("<|vision_end|>"),
        "image_pad": tok.convert_tokens_to_ids("<|image_pad|>"),
    }


def find_vision_spans(input_ids, vtok_ids):
    """Find vision token spans. Returns list of (start_idx, end_idx) tuples."""
    ids = input_ids[0].tolist()
    spans = []
    i = 0
    while i < len(ids):
        if ids[i] == vtok_ids["vision_start"]:
            start = i
            while i < len(ids) and ids[i] != vtok_ids["vision_end"]:
                i += 1
            end = i + 1 if i < len(ids) else i
            spans.append((start, end))
        i += 1
    return spans


def make_token_type_array(seq_len, vision_spans, n_type1, n_type2):
    """Create token type array. Type1/Type2 map to first/second group of vision spans.

    Returns: np.array of shape (seq_len,) with values:
      0 = text tokens
      1 = type1 vision tokens (e.g., video frames)
      2 = type2 vision tokens (e.g., diagram images)
    """
    token_type = np.zeros(seq_len, dtype=np.int32)
    for i, (start, end) in enumerate(vision_spans):
        if i < n_type1:
            token_type[start:end] = 1
        elif i < n_type1 + n_type2:
            token_type[start:end] = 2
    return token_type


def count_images_t1(q, setting):
    """Count video frames and diagram images for T1 (type 1a)."""
    n_video = len(q.get("video_frames", []))
    n_diagram = 0
    if setting != "text_only":
        n_diagram = sum(1 for o in q["options"] if "image" in o)
    return n_video, n_diagram


def count_images_d1(q):
    """Count video frames for D1 (type 1c). Returns (n_clip_a, n_clip_b)."""
    n_a = len(q.get("video_frames_a", []))
    n_b = len(q.get("video_frames_b", []))
    return n_a, n_b


# ------------------------------------------------------
# Tokenize input
# ------------------------------------------------------
def tokenize_prompt(processor, content, images):
    """Tokenize content into model inputs. Returns inputs dict."""
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text],
        images=images if images else None,
        padding=True,
        return_tensors="pt",
    )
    return inputs


# ------------------------------------------------------
# Attention extraction via hooks (flash-attention compatible)
# ------------------------------------------------------
# Strategy: Use flash_attention_2 for the actual forward pass (memory-efficient),
# but hook into the attention module to capture Q, K AFTER RoPE.
# Then manually compute attention for ONLY the last query token:
#   attn = softmax(q_last @ K^T / sqrt(d))  -- O(seq) memory, not O(seq^2).
#
# This gives us the same attention distribution as eager attention for the
# last token, without storing the full (seq x seq) attention matrix.
# ------------------------------------------------------

_attn_store = {}


def _make_qk_hook(layer_idx):
    """Hook that intercepts the attention module's forward to capture Q, K.

    We monkey-patch the forward to capture Q, K after RoPE but before
    the attention function. Then compute last-token attention manually.
    """
    def hook(module, args, kwargs):
        # This is a pre-forward hook with kwargs (register_forward_pre_hook with_kwargs=True)
        # We can't easily intercept Q, K here since they're computed inside forward.
        # Instead, we'll use a wrapper approach.
        pass
    return hook


def _patch_attention_forward(attn_module, layer_idx):
    """Wrap the attention module's forward to capture last-token attention.

    Replaces self_attn.forward with a wrapper that:
    1. Computes Q, K, V and applies RoPE (same as original)
    2. Manually computes last-token attention from Q[-1] @ K^T
    3. Calls the original attention function (flash) for the actual output
    4. Returns the original output
    """
    original_forward = attn_module.forward

    def patched_forward(hidden_states, position_embeddings, attention_mask=None,
                        past_key_values=None, cache_position=None, **kwargs):
        # Step 1: Compute Q, K (same as original forward lines 9-17)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn_module.head_dim)

        query_states = attn_module.q_norm(attn_module.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = attn_module.k_norm(attn_module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
        query_states_rope, key_states_rope = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Step 2: Compute last-token attention manually
        # q_last: (1, num_heads, 1, head_dim)
        q_last = query_states_rope[:, :, -1:, :]
        # k_all: (1, num_kv_heads, seq_len, head_dim) -- need to repeat for GQA
        k_for_attn = key_states_rope
        num_groups = attn_module.config.num_attention_heads // attn_module.config.num_key_value_heads
        if num_groups > 1:
            k_for_attn = k_for_attn.repeat_interleave(num_groups, dim=1)

        # attn_scores: (1, num_heads, 1, seq_len)
        attn_scores = torch.matmul(q_last, k_for_attn.transpose(-2, -1)) * attn_module.scaling

        # Apply causal mask for the last position (all positions visible)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32)
        # Store: (num_heads, seq_len)
        _attn_store[layer_idx] = attn_probs[0, :, 0, :].detach().cpu()

        del q_last, k_for_attn, attn_scores, attn_probs, query_states_rope, key_states_rope
        del query_states, key_states

        # Step 3: Call original forward (which uses flash attention)
        return original_forward(
            hidden_states, position_embeddings, attention_mask,
            past_key_values, cache_position, **kwargs
        )

    attn_module._original_forward = original_forward
    attn_module.forward = patched_forward


def _unpatch_attention_forward(attn_module):
    """Restore original forward method."""
    if hasattr(attn_module, '_original_forward'):
        attn_module.forward = attn_module._original_forward
        del attn_module._original_forward


def register_attention_hooks(model):
    """Patch target layers' attention modules to capture last-token attention.
    Returns list of patched modules (for unpatching).

    Compatible with flash_attention_2 -- no eager attention needed.
    """
    layers = model.model.language_model.layers
    patched = []

    print(f"  Patching {len(TARGET_LAYERS)} layers for attention capture: {TARGET_LAYERS}")
    for layer_idx in TARGET_LAYERS:
        attn_module = layers[layer_idx].self_attn
        _patch_attention_forward(attn_module, layer_idx)
        patched.append(attn_module)
    return patched


def remove_attention_hooks(patched_modules):
    """Restore original forward methods."""
    for attn_module in patched_modules:
        _unpatch_attention_forward(attn_module)


def extract_attention_ratios(model, inputs, token_type):
    """Forward pass with patched attention, compute ratios.

    Patched layers compute last-token attention manually (O(seq) memory),
    while using flash attention for the actual forward computation.
    """
    global _attn_store
    _attn_store = {}

    with torch.no_grad():
        model(**inputs)

    type_names = {0: "text", 1: "type1", 2: "type2"}
    per_layer = {}

    for layer_idx in TARGET_LAYERS:
        if layer_idx not in _attn_store:
            continue
        last_attn = _attn_store[layer_idx].float()  # (heads, seq_len)
        avg_attn = last_attn.mean(dim=0).numpy()  # (seq_len,)

        layer_ratios = {}
        for type_id, name in type_names.items():
            mask = token_type == type_id
            if mask.sum() > 0:
                layer_ratios[name] = float(avg_attn[mask].sum())
            else:
                layer_ratios[name] = 0.0

        total = sum(layer_ratios.values())
        if total > 0:
            layer_ratios = {k: v / total for k, v in layer_ratios.items()}
        per_layer[layer_idx] = layer_ratios

    avg_ratios = {}
    for name in type_names.values():
        vals = [per_layer[l].get(name, 0.0) for l in per_layer]
        avg_ratios[name] = float(np.mean(vals)) if vals else 0.0

    _attn_store = {}
    torch.cuda.empty_cache()

    return avg_ratios, per_layer


# ------------------------------------------------------
# Generation
# ------------------------------------------------------
def generate_answer(model, processor, content, images, max_new_tokens=1024):
    """Generate model response."""
    inputs = tokenize_prompt(processor, content, images)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    generated = output_ids[0, inputs["input_ids"].shape[1]:]
    response = processor.decode(generated, skip_special_tokens=True).strip()

    del output_ids, inputs
    torch.cuda.empty_cache()
    return response


# ------------------------------------------------------
# Masked prompt: V+T with descriptions neutralized
# ------------------------------------------------------
def build_prompt_masked(q):
    """Build V+T prompt but replace text descriptions with neutral placeholder.

    This tests causality: if accuracy recovers to Visual level when semantic
    content of descriptions is removed, the text content (not just presence)
    caused the mode switch.
    """
    content, images = build_prompt_and_images(q, "text_grounding")

    masked_content = []
    for item in content:
        if item["type"] == "text":
            t = item["text"]
            if t.startswith("(Description of the diagram above:"):
                masked_content.append(
                    {"type": "text", "text": "(Step description omitted.)"}
                )
            else:
                masked_content.append(item)
        else:
            masked_content.append(item)

    return masked_content, images


# ------------------------------------------------------
# Analysis 1: Cross-modal attention shift on T1
# ------------------------------------------------------
def analysis_1_attention(model, processor, t1_questions, args):
    """Compare attention distribution between Visual and V+T on T1.

    Uses forward hooks to capture attention from TARGET_LAYERS only.
    Memory-efficient: only stores last-token attention slices.
    """
    vtok_ids = get_vision_token_ids(processor)
    results = []
    n = min(args.n_attn, len(t1_questions))

    print(f"\n=== Analysis 1: Attention Distribution (T1, n={n}) ===")

    # Register hooks
    handles = register_attention_hooks(model)

    for i, q in enumerate(t1_questions[:n]):
        qid = q["id"]
        gt = q["answer"]
        t0 = time.time()

        record = {"id": qid, "product": q["product"], "answer_gt": gt}

        for setting in ["baseline", "text_grounding"]:
            content, images = build_prompt_and_images(q, setting)
            inputs = tokenize_prompt(processor, content, images)
            seq_len = inputs["input_ids"].shape[1]

            n_video, n_diagram = count_images_t1(q, setting)
            token_type = make_token_type_array(
                seq_len,
                find_vision_spans(inputs["input_ids"], vtok_ids),
                n_video, n_diagram,
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            avg_ratios, per_layer = extract_attention_ratios(
                model, inputs, token_type
            )

            # Rename keys for clarity
            setting_key = "visual" if setting == "baseline" else "vt"
            record[f"{setting_key}_attn"] = {
                "video": avg_ratios.get("type1", 0.0),
                "diagram": avg_ratios.get("type2", 0.0),
                "text": avg_ratios.get("text", 0.0),
            }
            record[f"{setting_key}_seq_len"] = seq_len
            record[f"{setting_key}_n_video"] = n_video
            record[f"{setting_key}_n_diagram"] = n_diagram

            del inputs
            torch.cuda.empty_cache()
            gc.collect()

        dt = time.time() - t0
        v_diag = record.get("visual_attn", {}).get("diagram", -1)
        vt_diag = record.get("vt_attn", {}).get("diagram", -1)
        print(
            f"  [{i+1}/{n}] {qid}: "
            f"V_diagram={v_diag:.3f} VT_diagram={vt_diag:.3f} "
            f"({dt:.1f}s)"
        )
        results.append(record)

    # Remove hooks
    remove_attention_hooks(handles)

    # Aggregate
    valid = [r for r in results if "visual_attn" in r and "vt_attn" in r]
    summary = {}
    for setting_key in ["visual", "vt"]:
        for tok_type in ["video", "diagram", "text"]:
            vals = [r[f"{setting_key}_attn"][tok_type] for r in valid]
            summary[f"{setting_key}_{tok_type}_mean"] = float(np.mean(vals))
            summary[f"{setting_key}_{tok_type}_std"] = float(np.std(vals))

    summary["n_valid"] = len(valid)
    summary["n_skipped"] = n - len(valid)

    print(f"\n  Summary ({len(valid)} valid questions):")
    print(f"    Visual:  video={summary['visual_video_mean']:.3f}  "
          f"diagram={summary['visual_diagram_mean']:.3f}  "
          f"text={summary['visual_text_mean']:.3f}")
    print(f"    V+T:     video={summary['vt_video_mean']:.3f}  "
          f"diagram={summary['vt_diagram_mean']:.3f}  "
          f"text={summary['vt_text_mean']:.3f}")

    return {"summary": summary, "per_question": results}


# ------------------------------------------------------
# Analysis 1b: Masking ablation on T1
# ------------------------------------------------------
def analysis_1b_masking(model, processor, t1_questions, args):
    """Run V+T with text descriptions replaced by neutral placeholder.

    Compare accuracy: Visual vs V+T vs V+T-masked.
    If V+T-masked approx Visual -> text content caused the degradation (causal evidence).
    """
    results = []
    n = len(t1_questions)  # Run on ALL T1 questions

    print(f"\n=== Analysis 1b: Masking Ablation (T1, n={n}) ===")

    for i, q in enumerate(t1_questions):
        qid = q["id"]
        gt = q["answer"]
        valid_labels = [o["label"] for o in q["options"]]
        t0 = time.time()

        # V+T-masked: descriptions replaced with neutral text
        content_masked, images_masked = build_prompt_masked(q)
        response = generate_answer(
            model, processor, content_masked, images_masked,
            max_new_tokens=args.max_new_tokens,
        )
        pred = extract_answer(response, valid_labels)
        correct = pred == gt

        results.append({
            "id": qid,
            "product": q["product"],
            "answer_gt": gt,
            "answer_pred_masked": pred,
            "correct_masked": correct,
            "raw_response_masked": response[:200],
        })

        dt = time.time() - t0
        status = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{n}] {qid}: pred={pred} gt={gt} {status} ({dt:.1f}s)")

        if (i + 1) % 50 == 0:
            n_correct = sum(1 for r in results if r["correct_masked"])
            print(f"  --- Running accuracy: {n_correct}/{i+1} = {n_correct/(i+1)*100:.1f}% ---")

        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    n_correct = sum(1 for r in results if r["correct_masked"])
    n_parsed = sum(1 for r in results if r["answer_pred_masked"] is not None)
    accuracy = n_correct / n if n > 0 else 0

    # Load existing results for comparison
    visual_acc = None
    vt_acc = None
    results_dir = BASE / "results"
    for tag, setting in [("visual", "baseline"), ("vt", "text_grounding")]:
        fpath = results_dir / f"full_qwen3-vl-8b_{setting}.json"
        if fpath.exists():
            existing = json.load(open(fpath))
            t1_existing = [r for r in existing if r["type"] == "1a"]
            n_c = sum(1 for r in t1_existing if r["correct"])
            acc = n_c / len(t1_existing) if t1_existing else 0
            if tag == "visual":
                visual_acc = acc
            else:
                vt_acc = acc

    summary = {
        "visual_accuracy": visual_acc,
        "vt_accuracy": vt_acc,
        "vt_masked_accuracy": accuracy,
        "vt_masked_n_correct": n_correct,
        "vt_masked_n_parsed": n_parsed,
        "vt_masked_n_total": n,
    }

    print(f"\n  Masking Ablation Results:")
    print(f"    Visual (baseline):     {visual_acc*100:.1f}%" if visual_acc else "    Visual: N/A")
    print(f"    V+T (text_grounding):  {vt_acc*100:.1f}%" if vt_acc else "    V+T: N/A")
    print(f"    V+T-masked:            {accuracy*100:.1f}%")
    if visual_acc and vt_acc:
        recovery = (accuracy - vt_acc) / (visual_acc - vt_acc) * 100 if visual_acc != vt_acc else 0
        summary["recovery_pct"] = recovery
        print(f"    Recovery: {recovery:.1f}% (100% = fully recovers to Visual level)")

    return {"summary": summary, "per_question": results}


# ------------------------------------------------------
# Analysis 2: D1 video attention
# ------------------------------------------------------
def analysis_2_d1(model, processor, d1_questions, args):
    """Measure attention balance between clip A and clip B in D1.

    If attention to both clips is roughly equal regardless of same/different,
    the model cannot differentiate video content -> explains near-chance D1.
    """
    vtok_ids = get_vision_token_ids(processor)
    results = []
    n = min(args.n_d1, len(d1_questions))

    print(f"\n=== Analysis 2: D1 Video Attention (n={n}) ===")

    # Register hooks
    handles = register_attention_hooks(model)

    for i, q in enumerate(d1_questions[:n]):
        qid = q["id"]
        gt = q["answer"]
        is_same = q["metadata"].get("is_positive", None)
        t0 = time.time()

        content, images = build_prompt_and_images(q, "baseline")
        inputs = tokenize_prompt(processor, content, images)
        seq_len = inputs["input_ids"].shape[1]

        n_clip_a, n_clip_b = count_images_d1(q)
        # type1 = clip A, type2 = clip B
        token_type = make_token_type_array(
            seq_len,
            find_vision_spans(inputs["input_ids"], vtok_ids),
            n_clip_a, n_clip_b,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        avg_ratios, per_layer = extract_attention_ratios(model, inputs, token_type)

        clip_a_attn = avg_ratios.get("type1", 0.0)
        clip_b_attn = avg_ratios.get("type2", 0.0)
        total_video = clip_a_attn + clip_b_attn
        balance = clip_a_attn / total_video if total_video > 0 else 0.5

        results.append({
            "id": qid,
            "is_same": is_same,
            "clip_a_attn": clip_a_attn,
            "clip_b_attn": clip_b_attn,
            "balance": balance,  # 0.5 = perfectly balanced
            "text_attn": avg_ratios.get("text", 0.0),
            "n_clip_a": n_clip_a,
            "n_clip_b": n_clip_b,
        })

        dt = time.time() - t0
        label = "SAME" if is_same else "DIFF"
        print(f"  [{i+1}/{n}] {qid} ({label}): "
              f"clipA={clip_a_attn:.3f} clipB={clip_b_attn:.3f} "
              f"balance={balance:.3f} ({dt:.1f}s)")

        del inputs
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate by same/different
    same_qs = [r for r in results if r["is_same"] is True]
    diff_qs = [r for r in results if r["is_same"] is False]

    def agg(qs, label):
        if not qs:
            return {}
        balances = [r["balance"] for r in qs]
        return {
            "n": len(qs),
            "balance_mean": float(np.mean(balances)),
            "balance_std": float(np.std(balances)),
            "clip_a_mean": float(np.mean([r["clip_a_attn"] for r in qs])),
            "clip_b_mean": float(np.mean([r["clip_b_attn"] for r in qs])),
        }

    summary = {
        "same": agg(same_qs, "SAME"),
        "different": agg(diff_qs, "DIFF"),
        "all": agg(results, "ALL"),
    }

    # Remove hooks
    remove_attention_hooks(handles)

    print(f"\n  D1 Video Attention Summary:")
    for label in ["same", "different", "all"]:
        s = summary[label]
        if s:
            print(f"    {label.upper()} (n={s['n']}): "
                  f"balance={s['balance_mean']:.3f}+/-{s['balance_std']:.3f}")

    return {"summary": summary, "per_question": results}


# ------------------------------------------------------
# Analysis 3: Difficulty-controlled comparison (CPU only)
# ------------------------------------------------------
def analysis_3_controlled(t1_questions, attn_results, args):
    """Compare attention patterns for correct vs incorrect, controlled for difficulty.

    Uses ALL 17 models' T1 results to define per-question difficulty.
    Only looks at medium-difficulty questions (answered correctly by 5-12 models).
    """
    print(f"\n=== Analysis 3: Difficulty-Controlled Comparison ===")

    # Load all models' T1 baseline results
    results_dir = BASE / "results"
    model_names = [
        "qwen2.5-vl-3b", "qwen2.5-vl-7b",
        "qwen3-vl-2b", "qwen3-vl-8b", "qwen3-vl-30b-a3b",
        "qwen3.5-2b", "qwen3.5-9b", "qwen3.5-27b",
        "internvl3.5-2b", "internvl3.5-8b", "internvl3.5-38b",
        "gemma3-4b", "gemma3-12b", "gemma3-27b",
        "glm-4.1v-9b", "llava-ov-1.5-8b", "minicpm-v-4.5",
    ]

    # Count per-question correctness across models
    question_difficulty = defaultdict(int)  # qid -> num models correct
    for mname in model_names:
        fpath = results_dir / f"full_{mname}_baseline.json"
        if not fpath.exists():
            print(f"  Warning: {fpath.name} not found, skipping")
            continue
        data = json.load(open(fpath))
        for r in data:
            if r["type"] == "1a" and r.get("correct"):
                question_difficulty[r["id"]] += 1

    # Medium difficulty: 5-12 models correct (out of 17)
    medium_qids = {
        qid for qid, count in question_difficulty.items()
        if 5 <= count <= 12
    }
    print(f"  Total T1 questions: {len(t1_questions)}")
    print(f"  Medium difficulty (5-12 models correct): {len(medium_qids)}")

    # Cross-reference with attention data
    attn_by_id = {r["id"]: r for r in attn_results if "visual_attn" in r}
    medium_with_attn = [
        attn_by_id[qid] for qid in medium_qids if qid in attn_by_id
    ]
    print(f"  Medium difficulty with attention data: {len(medium_with_attn)}")

    if not medium_with_attn:
        print("  No data for controlled comparison.")
        return {"summary": {"n_medium": 0}, "per_question": []}

    # Load Qwen3-VL-8B baseline results to know correct/incorrect
    qwen_results = {}
    fpath = results_dir / "full_qwen3-vl-8b_baseline.json"
    if fpath.exists():
        for r in json.load(open(fpath)):
            if r["type"] == "1a":
                qwen_results[r["id"]] = r.get("correct", False)

    correct_attn = []
    incorrect_attn = []
    for r in medium_with_attn:
        qid = r["id"]
        is_correct = qwen_results.get(qid, None)
        if is_correct is None:
            continue
        attn = r["visual_attn"]
        entry = {
            "diagram": attn["diagram"],
            "video": attn["video"],
            "text": attn["text"],
            "difficulty": question_difficulty.get(qid, 0),
        }
        if is_correct:
            correct_attn.append(entry)
        else:
            incorrect_attn.append(entry)

    def agg_attn(entries, label):
        if not entries:
            return {}
        return {
            "n": len(entries),
            "diagram_mean": float(np.mean([e["diagram"] for e in entries])),
            "diagram_std": float(np.std([e["diagram"] for e in entries])),
            "video_mean": float(np.mean([e["video"] for e in entries])),
            "video_std": float(np.std([e["video"] for e in entries])),
            "text_mean": float(np.mean([e["text"] for e in entries])),
            "text_std": float(np.std([e["text"] for e in entries])),
            "avg_difficulty": float(np.mean([e["difficulty"] for e in entries])),
        }

    summary = {
        "n_medium": len(medium_qids),
        "n_with_attn": len(medium_with_attn),
        "correct": agg_attn(correct_attn, "correct"),
        "incorrect": agg_attn(incorrect_attn, "incorrect"),
    }

    print(f"\n  Correct (n={summary['correct'].get('n', 0)}):")
    if summary["correct"]:
        print(f"    diagram={summary['correct']['diagram_mean']:.3f}+/-{summary['correct']['diagram_std']:.3f}  "
              f"video={summary['correct']['video_mean']:.3f}  "
              f"avg_difficulty={summary['correct']['avg_difficulty']:.1f}")
    print(f"  Incorrect (n={summary['incorrect'].get('n', 0)}):")
    if summary["incorrect"]:
        print(f"    diagram={summary['incorrect']['diagram_mean']:.3f}+/-{summary['incorrect']['diagram_std']:.3f}  "
              f"video={summary['incorrect']['video_mean']:.3f}  "
              f"avg_difficulty={summary['incorrect']['avg_difficulty']:.1f}")

    return {"summary": summary, "per_question_correct": correct_attn, "per_question_incorrect": incorrect_attn}


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Layer 3: Attention Routing Analysis")
    parser.add_argument("--data-dir", default=str(BASE / "data"),
                        help="Path to data directory containing benchmark/qa_benchmark.json")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache directory (default: HF_HOME or ~/.cache/huggingface)")
    parser.add_argument("--output", default=str(BASE / "results" / "attention_analysis.json"),
                        help="Output JSON path")
    parser.add_argument("--n_attn", type=int, default=100,
                        help="Max T1 questions for attention analysis")
    parser.add_argument("--n_d1", type=int, default=100,
                        help="Max D1 questions for video attention analysis")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--skip_masking", action="store_true",
                        help="Skip masking ablation (slow)")
    parser.add_argument("--skip_attention", action="store_true",
                        help="Skip attention analysis")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = get_cache_dir()

    input_path = Path(args.data_dir) / "benchmark" / "qa_benchmark.json"

    # Load benchmark
    with open(input_path) as f:
        all_questions = json.load(f)

    t1_questions = [q for q in all_questions if q["type"] == "1a"]
    d1_questions = [q for q in all_questions if q["type"] == "1c"]
    print(f"Loaded {len(all_questions)} questions: T1={len(t1_questions)}, D1={len(d1_questions)}")

    # Ensure PIL is loaded
    _ensure_imports()

    # Load model (flash_attention_2 -- attention captured via monkey-patched hooks)
    model, processor = load_model(cache_dir=args.cache_dir, attn_impl="flash_attention_2")

    all_results = {
        "model": "Qwen3-VL-8B-Instruct",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Analysis 1: Attention distribution (runs if n_attn > 0)
    if args.n_attn > 0 and not args.skip_attention:
        a1 = analysis_1_attention(model, processor, t1_questions, args)
        all_results["analysis_1_attention"] = a1

        # Analysis 3: Difficulty-controlled (uses Analysis 1 data, CPU only)
        a3 = analysis_3_controlled(
            t1_questions, a1["per_question"], args
        )
        all_results["analysis_3_controlled"] = a3
    else:
        print("\n=== Skipping Analysis 1 (attention) ===")

    # Analysis 2: D1 video attention (runs if n_d1 > 0)
    if args.n_d1 > 0:
        a2 = analysis_2_d1(model, processor, d1_questions, args)
        all_results["analysis_2_d1"] = a2
    else:
        print("\n=== Skipping Analysis 2 (D1, n_d1=0) ===")

    # Analysis 1b: Masking ablation
    if not args.skip_masking:
        a1b = analysis_1b_masking(model, processor, t1_questions, args)
        all_results["analysis_1b_masking"] = a1b
    else:
        print("\n=== Skipping Analysis 1b (masking) ===")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # Print key numbers
    print("\n" + "=" * 60)
    print("KEY RESULTS SUMMARY")
    print("=" * 60)
    if "analysis_1_attention" in all_results:
        s = all_results["analysis_1_attention"]["summary"]
        print(f"\nAttention Shift (Visual -> V+T):")
        print(f"  Diagram attention: {s['visual_diagram_mean']:.3f} -> {s['vt_diagram_mean']:.3f} "
              f"(delta={s['vt_diagram_mean'] - s['visual_diagram_mean']:+.3f})")
        print(f"  Text attention:    {s['visual_text_mean']:.3f} -> {s['vt_text_mean']:.3f} "
              f"(delta={s['vt_text_mean'] - s['visual_text_mean']:+.3f})")
    if "analysis_1b_masking" in all_results:
        s = all_results["analysis_1b_masking"]["summary"]
        print(f"\nMasking Ablation (T1 accuracy):")
        if s["visual_accuracy"] is not None:
            print(f"  Visual:     {s['visual_accuracy']*100:.1f}%")
        if s["vt_accuracy"] is not None:
            print(f"  V+T:        {s['vt_accuracy']*100:.1f}%")
        print(f"  V+T-masked: {s['vt_masked_accuracy']*100:.1f}%")
        if "recovery_pct" in s:
            print(f"  Recovery:   {s['recovery_pct']:.1f}%")
    if "analysis_2_d1" in all_results:
        s = all_results["analysis_2_d1"]["summary"]
        print(f"\nD1 Video Attention Balance:")
        for label in ["same", "different"]:
            if s.get(label):
                print(f"  {label.upper()}: balance={s[label]['balance_mean']:.3f}+/-{s[label]['balance_std']:.3f}")


if __name__ == "__main__":
    main()
