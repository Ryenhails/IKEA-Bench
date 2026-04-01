"""IKEA-Bench evaluation using Google Gemini API (proprietary model).

Reuses prompt construction and answer extraction from the ikea_bench package.
Calls Gemini via the google-genai SDK.

Usage:
  export GEMINI_API_KEY=your_key_here
  python -m ikea_bench.eval_gemini \\
    --setting baseline \\
    --input data/benchmark/qa_benchmark.json \\
    --output results/gemini31pro_baseline.json

  # Run all 3 settings:
  python -m ikea_bench.eval_gemini --setting all
"""

import json
import os
import io
import time
import argparse
from pathlib import Path

from PIL import Image

from .prompts import build_prompt_and_images
from .utils import _ensure_imports, extract_answer, get_data_dir


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------
def init_gemini_client():
    """Initialize the Google GenAI client."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package not found. Install with:\n"
            "  pip install google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GEMINI_API_KEY environment variable")

    client = genai.Client(api_key=api_key)
    return client, types


def pil_to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    """Convert PIL Image to bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def call_gemini(client, types, content_list, model_name="gemini-3.1-pro"):
    """Call Gemini API with interleaved text/image content.

    Parameters
    ----------
    client : genai.Client
    types : google.genai.types module
    content_list : list[dict]
        List of ``{"type": "text", "text": ...}`` or
        ``{"type": "image", "image": PIL.Image}`` dicts.
    model_name : str

    Returns
    -------
    str
        Response text.
    """
    # Build parts list for Gemini
    parts = []
    for c in content_list:
        if c["type"] == "text":
            parts.append(c["text"])
        elif c["type"] == "image":
            img = c["image"]
            img_bytes = pil_to_bytes(img)
            parts.append(types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/jpeg",
            ))

    response = client.models.generate_content(
        model=model_name,
        contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
        ),
    )

    if response.text:
        return response.text.strip()
    return ""


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_eval(client, types, questions, setting, model_name, output_path,
             existing_results=None, data_dir=None):
    """Run evaluation for a single setting."""
    _ensure_imports()

    # Start from existing results if resuming
    results = list(existing_results) if existing_results else []
    t0 = time.time()
    errors = 0
    consecutive_errors = 0
    rate_limit_waits = 0

    for i, q in enumerate(questions):
        content, images = build_prompt_and_images(q, setting, data_dir=data_dir)
        valid_labels = [opt["label"] for opt in q["options"]]

        # Retry with exponential backoff for rate limits
        raw = ""
        success = False
        for attempt in range(5):
            try:
                raw = call_gemini(client, types, content, model_name)
                success = True
                consecutive_errors = 0
                break
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                    wait = min(120, 2 ** attempt * 5)
                    rate_limit_waits += 1
                    print(f"  Rate limited (attempt {attempt+1}), waiting {wait}s...")
                    time.sleep(wait)
                elif "safety" in err_str or "block" in err_str:
                    print(f"  BLOCKED on {q['id']}: {e}")
                    raw = ""
                    success = True  # not a retriable error
                    consecutive_errors = 0
                    break
                else:
                    print(f"  ERROR on {q['id']}: {e}")
                    errors += 1
                    consecutive_errors += 1
                    raw = ""
                    break

        # If all 5 retries failed, count as error
        if not success:
            errors += 1
            consecutive_errors += 1

        pred = extract_answer(raw, valid_labels)
        correct = pred == q["answer"] if pred else False

        results.append({
            "id": q["id"],
            "type": q["type"],
            "product": q["product"],
            "answer_gt": q["answer"],
            "answer_pred": pred,
            "correct": correct,
            "raw_response": raw[:500],  # truncate for storage
        })

        mark = "OK" if correct else "WRONG"
        print(f"  [{i+1}/{len(questions)}] {q['id'][:12]} type={q['type']} "
              f"gt={q['answer']} pred={pred} {mark}")

        # Save every 10 questions to minimize loss on crash
        if (i + 1) % 10 == 0:
            _save_results(results, output_path, setting, model_name)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            acc_so_far = sum(r["correct"] for r in results) / len(results)
            print(f"  --- [{i+1}/{len(questions)}] acc={acc_so_far:.1%} "
                  f"({elapsed:.0f}s, errors={errors}, rl_waits={rate_limit_waits}) ---")

        # Abort if 10 consecutive errors (API key invalid, quota exhausted, etc.)
        if consecutive_errors >= 10:
            print(f"\n  ABORTING: {consecutive_errors} consecutive errors. "
                  f"Check API key / quota. Saving {len(results)} results.")
            _save_results(results, output_path, setting, model_name)
            return results

    # Final save
    _save_results(results, output_path, setting, model_name)
    return results


def _save_results(results, output_path, setting, model_name):
    """Save results and print summary."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    total_correct = sum(r["correct"] for r in results)
    parsed = sum(1 for r in results if r["answer_pred"] is not None)
    print(f"\n  [{model_name} / {setting}] "
          f"{total_correct}/{len(results)} = {total_correct/len(results):.1%} "
          f"(parsed={parsed}/{len(results)})")

    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    for t in sorted(by_type.keys()):
        rs = by_type[t]
        acc = sum(r["correct"] for r in rs) / len(rs)
        chance = 0.5 if t in ("1b", "1c") else 0.25
        print(f"    {t}: {acc:.1%} ({sum(r['correct'] for r in rs)}/{len(rs)}) "
              f"[chance={chance:.0%}]")
    print(f"  Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="IKEA-Bench Gemini API Evaluation")
    parser.add_argument("--setting", type=str, default="baseline",
                        choices=["baseline", "text_grounding", "text_only", "all"],
                        help="Evaluation setting (or 'all' for all 3)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to QA JSON file")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default="gemini-3.1-pro",
                        help="Gemini model ID")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root data directory for images (default: <project_root>/data)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial results")
    args = parser.parse_args()

    # Load questions
    with open(args.input) as f:
        questions = json.load(f)
    print(f"Loaded {len(questions)} questions from {args.input}")

    # Init client
    client, types = init_gemini_client()
    print(f"Gemini client initialized (model: {args.model})")

    # Determine settings to run
    if args.setting == "all":
        settings = ["baseline", "text_grounding", "text_only"]
    else:
        settings = [args.setting]

    safe_model = args.model.replace(".", "_").replace("-", "_")

    for setting in settings:
        output_path = Path(args.output_dir) / f"{safe_model}_{setting}.json"

        # Resume support
        if args.resume and output_path.exists():
            with open(output_path) as f:
                existing = json.load(f)
            done_ids = {r["id"] for r in existing}
            remaining = [q for q in questions if q["id"] not in done_ids]
            if not remaining:
                print(f"\n{setting}: already complete ({len(existing)} results)")
                continue
            print(f"\n{setting}: resuming ({len(existing)} done, {len(remaining)} remaining)")
        else:
            existing = []
            remaining = questions

        print(f"\n{'='*60}")
        print(f"Setting: {setting} ({len(remaining)} questions)")
        print(f"{'='*60}")

        # Pass existing results so saves include everything
        run_eval(
            client, types, remaining, setting, args.model, str(output_path),
            existing_results=existing, data_dir=args.data_dir,
        )


if __name__ == "__main__":
    main()
