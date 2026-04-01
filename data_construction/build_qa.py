"""IKEA-Bench QA Construction — 2x3 Taxonomy (Quality-First).

Generates 6 task types from GT annotations:
  Dim 1 (Cross-Modal Alignment): 1a, 1b, 1c
  Dim 2 (Procedural Reasoning):  2a, 2b, 2c

Design principles:
  - Quality > Quantity: every question is natural and non-redundant
  - All answers deterministic from GT annotations
  - Hard negatives: adjacent steps only
  - No artificial inflation (no backward variants, no skip triplets)
"""

import argparse
import json
import random
import hashlib
import itertools
from pathlib import Path
from collections import defaultdict

random.seed(42)

# Question phrasing variants
PHRASINGS = {
    "1a": [
        "Which manual step is being performed in these video frames?",
        "These video frames show part of the assembly process. Which instruction diagram matches this action?",
        "Match the video clip to the correct manual step.",
        "The person in the video is following one of these instructions. Which one?",
    ],
    "1b": [
        "Does the action in these video frames match the instruction shown in this manual diagram?",
        "Is the person in the video correctly following this instruction diagram?",
    ],
    "1c": [
        "Are these two video clips showing the same assembly step?",
        "Both clips show someone assembling the same product. Are they at the same stage?",
    ],
    "2a": [
        "Which assembly step is currently being performed in these video frames?",
        "Looking at the assembly state in these frames, which step is this?",
        "How far along is the furniture assembly in these video frames?",
    ],
    "2b": [
        "After the action shown in these video frames, which manual step should be performed next?",
        "What should be done immediately after the action shown here?",
    ],
    "2c": [
        "These three manual step diagrams are shown in random order. What is the correct assembly sequence?",
        "Arrange these instruction steps in the proper assembly order.",
    ],
}


def make_binary_options(positive_text, negative_text, is_positive):
    """Create binary options with randomized A/B order to avoid position bias.

    50% of the time the order is flipped (A=negative, B=positive).
    Returns (options_list, answer_label).
    """
    if random.random() < 0.5:
        # Normal order: A=positive, B=negative
        options = [
            {"label": "A", "text": positive_text},
            {"label": "B", "text": negative_text},
        ]
        answer = "A" if is_positive else "B"
    else:
        # Flipped order: A=negative, B=positive
        options = [
            {"label": "A", "text": negative_text},
            {"label": "B", "text": positive_text},
        ]
        answer = "B" if is_positive else "A"
    return options, answer


def load_data(data_dir):
    with open(data_dir / "data.json") as f:
        return json.load(f)


def get_video_id(url):
    return url.split("/watch?v=")[-1]


def get_manual_images(manual_dir, category, name):
    base = manual_dir / category / name
    if not base.exists():
        return {}
    result = {}
    for step_dir in sorted(base.iterdir()):
        if step_dir.is_dir() and step_dir.name.startswith("step_"):
            step_idx = int(step_dir.name.split("_")[1])
            pngs = list(step_dir.glob("*.png"))
            if pngs:
                result[step_idx] = str(pngs[0])
    return result


def get_frame_paths(frames_dir, category, name, step_id, video_id):
    frame_dir = frames_dir / category / name / f"step{step_id}" / video_id
    if not frame_dir.exists():
        return []
    return sorted(str(p) for p in frame_dir.glob("*.jpg"))


def make_id(*args):
    s = "|".join(str(a) for a in args)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def sample_distractors(correct_id, all_ids, n=3):
    """Sample n distractors, preferring adjacent IDs (hard negatives)."""
    candidates = [x for x in all_ids if x != correct_id]
    if len(candidates) <= n:
        return candidates
    candidates.sort(key=lambda x: abs(x - correct_id))
    close = candidates[:min(n + 2, len(candidates))]
    if len(close) > n:
        selected = close[:n - 1] + [random.choice(close[n - 1:])]
    else:
        selected = close[:n]
    return selected


def estimate_visual_tokens(n_video_frames, n_manual_images):
    return n_video_frames * 299 + n_manual_images * 280


def _diverse_sample(questions, n):
    """Downsample while maximizing product diversity."""
    if len(questions) <= n:
        return questions
    by_product = defaultdict(list)
    for q in questions:
        by_product[q["product"]].append(q)
    # Round-robin across products
    result = []
    products = sorted(by_product.keys())
    idx = {p: 0 for p in products}
    while len(result) < n:
        added = False
        for p in products:
            if idx[p] < len(by_product[p]) and len(result) < n:
                result.append(by_product[p][idx[p]])
                idx[p] += 1
                added = True
        if not added:
            break
    return result


def cap_binary_task(questions, target_per_side):
    """Cap a binary task to target_per_side positives + target_per_side negatives."""
    pos = [q for q in questions if q["metadata"]["is_positive"]]
    neg = [q for q in questions if not q["metadata"]["is_positive"]]
    if len(pos) > target_per_side:
        pos = _diverse_sample(pos, target_per_side)
    if len(neg) > target_per_side:
        neg = _diverse_sample(neg, target_per_side)
    n_balanced = min(len(pos), len(neg))
    return pos[:n_balanced] + neg[:n_balanced]


# ============================================================
# 1a: Step Recognition — V + 4xM -> 4-way MC (image)
# ============================================================

def build_1a(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]
        manual_imgs = get_manual_images(manual_dir, category, name)
        if len(manual_imgs) < 3:
            continue
        step_ids = sorted(manual_imgs.keys())

        for step in item["steps"]:
            sid = step["step_id"]
            if sid not in manual_imgs:
                continue
            for vid_info in step["video"]:
                video_id = get_video_id(vid_info["video_id"])
                frames = get_frame_paths(frames_dir, category, name, sid, video_id)
                if not frames:
                    continue
                distractors = sample_distractors(sid, step_ids, n=3)
                if len(distractors) < 2:
                    continue

                options = [(sid, manual_imgs[sid])]
                for d in distractors:
                    options.append((d, manual_imgs[d]))
                options = options[:4]
                random.shuffle(options)
                correct_idx = next(i for i, (s, _) in enumerate(options) if s == sid)

                questions.append({
                    "id": make_id("1a", name, sid, video_id),
                    "type": "1a",
                    "dimension": "alignment",
                    "task": "step_recognition",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["1a"]),
                    "video_frames": frames,
                    "options": [
                        {"label": chr(65 + i), "step_id": s, "image": img}
                        for i, (s, img) in enumerate(options)
                    ],
                    "answer": chr(65 + correct_idx),
                    "answer_step_id": sid,
                    "visual_tokens_est": estimate_visual_tokens(len(frames), len(options)),
                    "metadata": {
                        "video_id": video_id,
                        "step_start": vid_info["step_start"],
                        "step_end": vid_info["step_end"],
                    }
                })
    return questions


# ============================================================
# 1b: Action Verification — V + 1xM -> Binary
# ============================================================

def build_1b(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]
        manual_imgs = get_manual_images(manual_dir, category, name)
        if len(item["steps"]) < 2:
            continue

        for step in item["steps"]:
            sid = step["step_id"]
            if sid not in manual_imgs:
                continue
            for vid_info in step["video"]:
                video_id = get_video_id(vid_info["video_id"])
                frames = get_frame_paths(frames_dir, category, name, sid, video_id)
                if not frames:
                    continue

                # Positive
                opts_1b_pos, ans_1b_pos = make_binary_options(
                    "Yes, the video matches this manual step",
                    "No, the video shows a different step",
                    is_positive=True,
                )
                questions.append({
                    "id": make_id("1b+", name, sid, video_id),
                    "type": "1b",
                    "dimension": "alignment",
                    "task": "action_verification",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["1b"]),
                    "video_frames": frames,
                    "manual_step_image": manual_imgs[sid],
                    "manual_step_id": sid,
                    "options": opts_1b_pos,
                    "answer": ans_1b_pos,
                    "visual_tokens_est": estimate_visual_tokens(len(frames), 1),
                    "metadata": {
                        "video_id": video_id,
                        "shown_step": sid, "actual_step": sid,
                        "is_positive": True,
                    }
                })

                # Negative (adjacent step only)
                adjacent = [s for s in sorted(manual_imgs.keys())
                            if s != sid and abs(s - sid) == 1]
                if not adjacent:
                    adjacent = [s for s in sorted(manual_imgs.keys()) if s != sid]
                if not adjacent:
                    continue
                wrong_sid = random.choice(adjacent)

                opts_1b_neg, ans_1b_neg = make_binary_options(
                    "Yes, the video matches this manual step",
                    "No, the video shows a different step",
                    is_positive=False,
                )
                questions.append({
                    "id": make_id("1b-", name, sid, video_id, wrong_sid),
                    "type": "1b",
                    "dimension": "alignment",
                    "task": "action_verification",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["1b"]),
                    "video_frames": frames,
                    "manual_step_image": manual_imgs[wrong_sid],
                    "manual_step_id": wrong_sid,
                    "options": opts_1b_neg,
                    "answer": ans_1b_neg,
                    "visual_tokens_est": estimate_visual_tokens(len(frames), 1),
                    "metadata": {
                        "video_id": video_id,
                        "shown_step": wrong_sid, "actual_step": sid,
                        "is_positive": False,
                    }
                })
    return questions


# ============================================================
# 1c: Cross-View Matching — V1 + V2 -> Binary
# ============================================================

def build_1c(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]

        # Positive: same step, different videos
        for step in item["steps"]:
            sid = step["step_id"]
            videos = step["video"]
            if len(videos) < 2:
                continue
            for i in range(len(videos)):
                for j in range(i + 1, len(videos)):
                    vid_a = get_video_id(videos[i]["video_id"])
                    vid_b = get_video_id(videos[j]["video_id"])
                    frames_a = get_frame_paths(frames_dir, category, name, sid, vid_a)
                    frames_b = get_frame_paths(frames_dir, category, name, sid, vid_b)
                    if not frames_a or not frames_b:
                        continue
                    opts_1c_pos, ans_1c_pos = make_binary_options(
                        "Yes, they show the same step",
                        "No, they show different steps",
                        is_positive=True,
                    )
                    questions.append({
                        "id": make_id("1c+", name, sid, vid_a, vid_b),
                        "type": "1c",
                        "dimension": "alignment",
                        "task": "cross_view_matching",
                        "product": name,
                        "category": category,
                        "question": random.choice(PHRASINGS["1c"]),
                        "video_frames_a": frames_a,
                        "video_frames_b": frames_b,
                        "options": opts_1c_pos,
                        "answer": ans_1c_pos,
                        "visual_tokens_est": estimate_visual_tokens(
                            len(frames_a) + len(frames_b), 0),
                        "metadata": {
                            "video_a": vid_a, "video_b": vid_b,
                            "step_a": sid, "step_b": sid,
                            "is_positive": True,
                        }
                    })

        # Negative: adjacent steps only
        steps_with_frames = []
        for step in item["steps"]:
            sid = step["step_id"]
            for vid_info in step["video"]:
                vid_id = get_video_id(vid_info["video_id"])
                frames = get_frame_paths(frames_dir, category, name, sid, vid_id)
                if frames:
                    steps_with_frames.append((sid, vid_id, frames))

        for i in range(len(steps_with_frames)):
            for j in range(i + 1, len(steps_with_frames)):
                sa, va, fa = steps_with_frames[i]
                sb, vb, fb = steps_with_frames[j]
                if sa == sb or abs(sa - sb) > 1:
                    continue
                opts_1c_neg, ans_1c_neg = make_binary_options(
                    "Yes, they show the same step",
                    "No, they show different steps",
                    is_positive=False,
                )
                questions.append({
                    "id": make_id("1c-", name, sa, va, sb, vb),
                    "type": "1c",
                    "dimension": "alignment",
                    "task": "cross_view_matching",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["1c"]),
                    "video_frames_a": fa,
                    "video_frames_b": fb,
                    "options": opts_1c_neg,
                    "answer": ans_1c_neg,
                    "visual_tokens_est": estimate_visual_tokens(
                        len(fa) + len(fb), 0),
                    "metadata": {
                        "video_a": va, "video_b": vb,
                        "step_a": sa, "step_b": sb,
                        "is_positive": False,
                    }
                })

    # Balance positive/negative
    pos = [q for q in questions if q["metadata"]["is_positive"]]
    neg = [q for q in questions if not q["metadata"]["is_positive"]]
    n_balanced = min(len(pos), len(neg))
    if len(pos) > n_balanced:
        pos = _diverse_sample(pos, n_balanced)
    if len(neg) > n_balanced:
        neg = _diverse_sample(neg, n_balanced)
    return pos + neg


# ============================================================
# 2a: Progress Tracking — V + all M -> 4-way MC
# ============================================================

def build_2a(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]
        manual_imgs = get_manual_images(manual_dir, category, name)
        if len(item["steps"]) < 3:
            continue
        step_ids = sorted(manual_imgs.keys())

        for step in item["steps"]:
            sid = step["step_id"]
            for vid_info in step["video"]:
                video_id = get_video_id(vid_info["video_id"])
                frames = get_frame_paths(frames_dir, category, name, sid, video_id)
                if not frames:
                    continue
                distractors = sample_distractors(sid, step_ids, n=3)
                if len(distractors) < 2:
                    continue

                options_raw = [sid] + distractors
                random.shuffle(options_raw)
                correct_idx = options_raw.index(sid)

                questions.append({
                    "id": make_id("2a", name, sid, video_id),
                    "type": "2a",
                    "dimension": "procedural",
                    "task": "progress_tracking",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["2a"]),
                    "video_frames": frames,
                    "manual_images": {
                        str(s): manual_imgs[s] for s in step_ids if s in manual_imgs
                    },
                    "options": [
                        {"label": chr(65 + i), "step_id": s, "text": f"Step {s + 1}"}
                        for i, s in enumerate(options_raw)
                    ],
                    "answer": chr(65 + correct_idx),
                    "answer_step_id": sid,
                    "visual_tokens_est": estimate_visual_tokens(
                        len(frames), len(manual_imgs)),
                    "metadata": {"video_id": video_id}
                })
    return questions


# ============================================================
# 2b: Next Step Prediction — V + 4xM -> 4-way MC (image)
# Forward only: "what comes next?"
# ============================================================

def build_2b(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]
        manual_imgs = get_manual_images(manual_dir, category, name)
        n_steps = len(item["steps"])
        if n_steps < 3:
            continue
        step_ids = sorted(manual_imgs.keys())

        for i in range(n_steps - 1):
            step = item["steps"][i]
            next_step = item["steps"][i + 1]
            sid = step["step_id"]
            next_sid = next_step["step_id"]
            if next_sid not in manual_imgs:
                continue

            for vid_info in step["video"]:
                video_id = get_video_id(vid_info["video_id"])
                frames = get_frame_paths(frames_dir, category, name, sid, video_id)
                if not frames:
                    continue

                other_steps = [s for s in step_ids if s != sid and s != next_sid]
                distractors = sample_distractors(next_sid, other_steps + [next_sid], n=3)
                distractors = [d for d in distractors if d != next_sid][:3]
                if len(distractors) < 2:
                    continue

                options = [(next_sid, manual_imgs[next_sid])]
                for d in distractors:
                    if d in manual_imgs:
                        options.append((d, manual_imgs[d]))
                if len(options) < 3:
                    continue
                options = options[:4]
                random.shuffle(options)
                correct_idx = next(
                    idx for idx, (s, _) in enumerate(options) if s == next_sid)

                questions.append({
                    "id": make_id("2b", name, sid, video_id),
                    "type": "2b",
                    "dimension": "procedural",
                    "task": "next_step_prediction",
                    "product": name,
                    "category": category,
                    "question": random.choice(PHRASINGS["2b"]),
                    "video_frames": frames,
                    "options": [
                        {"label": chr(65 + i), "step_id": s, "image": img}
                        for i, (s, img) in enumerate(options)
                    ],
                    "answer": chr(65 + correct_idx),
                    "answer_step_id": next_sid,
                    "visual_tokens_est": estimate_visual_tokens(
                        len(frames), len(options)),
                    "metadata": {
                        "video_id": video_id,
                        "current_step": sid,
                    }
                })
    return questions


# ============================================================
# 2c: Sequence Ordering — 3xM (consecutive) -> 4-way MC (perm)
# Consecutive triplets only — no skip, no inflation
# ============================================================

def build_2c(data, frames_dir, manual_dir):
    questions = []
    for item in data:
        category, name = item["category"], item["name"]
        manual_imgs = get_manual_images(manual_dir, category, name)
        step_ids = sorted(manual_imgs.keys())
        if len(step_ids) < 3:
            continue

        for start in range(len(step_ids) - 2):
            triplet = step_ids[start:start + 3]
            if not all(t in manual_imgs for t in triplet):
                continue

            # Shuffle display order so images are not in correct order
            display_order = list(range(3))
            random.shuffle(display_order)
            # Correct answer: the permutation that recovers chronological order
            correct_display = tuple(display_order.index(i) for i in range(3))

            all_perms = list(itertools.permutations(range(3)))
            wrong_perms = [p for p in all_perms if p != correct_display]
            selected_wrong = random.sample(wrong_perms, min(3, len(wrong_perms)))

            all_options = [correct_display] + selected_wrong
            random.shuffle(all_options)
            correct_idx = all_options.index(correct_display)

            questions.append({
                "id": make_id("2c", name, start),
                "type": "2c",
                "dimension": "procedural",
                "task": "sequence_ordering",
                "product": name,
                "category": category,
                "question": random.choice(PHRASINGS["2c"]),
                "step_images": [
                    {
                        "label": f"Image {i + 1}",
                        "step_id": triplet[display_order[i]],
                        "image": manual_imgs[triplet[display_order[i]]],
                    }
                    for i in range(3)
                ],
                "options": [
                    {
                        "label": chr(65 + i),
                        "text": f"Image {p[0]+1} -> Image {p[1]+1} -> Image {p[2]+1}",
                    }
                    for i, p in enumerate(all_options)
                ],
                "answer": chr(65 + correct_idx),
                "visual_tokens_est": estimate_visual_tokens(0, 3),
                "metadata": {
                    "triplet_step_ids": triplet,
                    "display_order": display_order,
                }
            })

    return questions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="IKEA-Bench QA Construction (2x3 Taxonomy)")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root data directory (default: <repo>/data)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    frames_dir = data_dir / "qa_frames"
    manual_dir = data_dir / "manual_img"
    output_dir = data_dir / "benchmark"

    data = load_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("IKEA-Bench QA Construction (2x3, Quality-First)")
    print("=" * 60)

    builders = [
        ("1a", "Step Recognition", build_1a),
        ("1b", "Action Verification", build_1b),
        ("1c", "Cross-View Matching", build_1c),
        ("2a", "Progress Tracking", build_2a),
        ("2b", "Next Step Prediction", build_2b),
        ("2c", "Sequence Ordering", build_2c),
    ]

    all_questions = []
    stats = {}

    # Cap binary tasks to keep dimensions balanced
    # Target: ~200 per side for 1b/1c -> ~400 each, comparable to 1a(320)
    BINARY_CAPS = {"1b": 175, "1c": 175}

    for task_id, task_name, builder in builders:
        qs = builder(data, frames_dir, manual_dir)
        raw_count = len(qs)
        if task_id in BINARY_CAPS:
            qs = cap_binary_task(qs, BINARY_CAPS[task_id])
        print(f"  {task_id} ({task_name}): {raw_count} raw -> {len(qs)} final")
        all_questions.extend(qs)
        stats[task_id] = {"raw": raw_count, "final": len(qs)}

    # Summary
    dim1 = sum(1 for q in all_questions if q["dimension"] == "alignment")
    dim2 = sum(1 for q in all_questions if q["dimension"] == "procedural")
    products = set(q["product"] for q in all_questions)

    print(f"\n  Total: {len(all_questions)}")
    print(f"  Dim 1 (Alignment):  {dim1}")
    print(f"  Dim 2 (Procedural): {dim2}")
    print(f"  Products: {len(products)}")

    # Visual token stats
    tokens = [q["visual_tokens_est"] for q in all_questions]
    print(f"\n  Visual tokens — min: {min(tokens)}, max: {max(tokens)}, "
          f"avg: {sum(tokens) / len(tokens):.0f}")

    # Per-product
    by_product = defaultdict(int)
    for q in all_questions:
        by_product[q["product"]] += 1

    # Save
    out_path = output_dir / "qa_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(all_questions, f, indent=2)
    print(f"\n  Saved: {out_path}")

    stats_path = output_dir / "qa_benchmark_stats.json"
    with open(stats_path, "w") as f:
        json.dump({
            "total": len(all_questions),
            "by_type": stats,
            "by_dimension": {"alignment": dim1, "procedural": dim2},
            "by_product": dict(by_product),
            "products_covered": len(products),
        }, f, indent=2)
    print(f"  Stats: {stats_path}")


if __name__ == "__main__":
    main()
