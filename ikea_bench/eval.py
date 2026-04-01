"""IKEA-Bench VLM evaluation -- main CLI entry point.

Supports multiple models and evaluation settings:
  - baseline:       Video frames + manual images (as specified by each question)
  - text_grounding: Baseline + text descriptions appended
  - text_only:      Video frames + text descriptions (no manual images)
  - random:         Random answer (no model call)

Usage:
  python -m ikea_bench.eval \\
    --model qwen2.5-vl-7b \\
    --setting baseline \\
    --input data/benchmark/qa_benchmark.json \\
    --output results/qwen25vl7b_baseline.json \\
    [--max_new_tokens 128] [--seed 42] [--cache-dir /path/to/cache]
"""

import json
import argparse
import random
import time
from pathlib import Path

from .utils import extract_answer
from .prompts import build_prompt_and_images
from .models import load_model, run_single
from .models.registry import MODEL_CONFIGS


def main():
    parser = argparse.ArgumentParser(description="IKEA-Bench VLM Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()) + ["random"],
                        help="Model to evaluate")
    parser.add_argument("--setting", type=str, default="baseline",
                        choices=["baseline", "text_grounding", "text_only", "random"])
    parser.add_argument("--input", type=str, required=True,
                        help="Path to QA JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save results JSON")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root data directory containing images and descriptions "
                             "(default: <project_root>/data)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace cache directory (default: $HF_HOME or ~/.cache/huggingface)")
    args = parser.parse_args()

    # Load questions
    with open(args.input) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Random baseline
    if args.model == "random" or args.setting == "random":
        rng = random.Random(args.seed)
        results = []
        for q in questions:
            valid = [opt["label"] for opt in q["options"]]
            pred = rng.choice(valid)
            results.append({
                "id": q["id"],
                "type": q["type"],
                "product": q["product"],
                "answer_gt": q["answer"],
                "answer_pred": pred,
                "correct": pred == q["answer"],
                "raw_response": pred,
            })
    else:
        # Load model
        print(f"Loading model: {args.model}")
        model, proc, family = load_model(args.model, cache_dir=args.cache_dir)
        model.eval()
        print(f"Model loaded ({family})")

        results = []
        t0 = time.time()
        for i, q in enumerate(questions):
            content, images = build_prompt_and_images(q, args.setting, data_dir=args.data_dir)
            valid_labels = [opt["label"] for opt in q["options"]]

            try:
                raw = run_single(model, proc, family, content, images, args.max_new_tokens)
            except Exception as e:
                print(f"  ERROR on {q['id']}: {e}")
                raw = ""

            pred = extract_answer(raw, valid_labels)
            correct = pred == q["answer"] if pred else False

            results.append({
                "id": q["id"],
                "type": q["type"],
                "product": q["product"],
                "answer_gt": q["answer"],
                "answer_pred": pred,
                "correct": correct,
                "raw_response": raw,
            })

            # Log every question for debugging
            prompt_text = " | ".join(c["text"][:60] for c in content if c["type"] == "text")
            mark = "OK" if correct else "WRONG"
            print(f"  [{i+1}/{len(questions)}] {q['id'][:12]} type={q['type']} "
                  f"gt={q['answer']} pred={pred} {mark}")
            print(f"    prompt: {prompt_text[:200]}")
            print(f"    output: {raw[:200]}")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                acc_so_far = sum(r["correct"] for r in results) / len(results)
                print(f"  --- [{i+1}/{len(questions)}] acc={acc_so_far:.1%} ({elapsed:.0f}s) ---")

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Model: {args.model} | Setting: {args.setting}")
    print(f"Total: {len(results)} | Parsed: {sum(1 for r in results if r['answer_pred'] is not None)}")
    total_correct = sum(r["correct"] for r in results)
    print(f"Accuracy: {total_correct}/{len(results)} = {total_correct/len(results):.1%}")

    # Per-type breakdown
    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    print(f"\nPer-type accuracy:")
    for t in sorted(by_type.keys()):
        rs = by_type[t]
        acc = sum(r["correct"] for r in rs) / len(rs)
        chance = 0.5 if t in ("1b", "1c") else 0.25
        print(f"  {t}: {acc:.1%} ({sum(r['correct'] for r in rs)}/{len(rs)}) [chance={chance:.0%}]")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
