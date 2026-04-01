"""Extract video frames for QA construction.

For each (product, step, video) tuple:
- Sample 4-8 frames from within [step_start, step_end]
- Skip first/last 10% of step duration (transition avoidance)
- Short step (<10s): 4 frames
- Long step (>10s): 8 frames
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path


def get_video_id(url):
    return url.split("/watch?v=")[-1]


def extract_step_frames(video_path, step_start, step_end, fps, n_frames):
    """Extract n_frames uniformly from the middle 80% of [step_start, step_end]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    duration = step_end - step_start
    margin = duration * 0.1
    safe_start = step_start + margin
    safe_end = step_end - margin

    if safe_end <= safe_start:
        safe_start = step_start
        safe_end = step_end

    timestamps = np.linspace(safe_start, safe_end, n_frames)
    frames = []

    for t in timestamps:
        frame_idx = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append((t, frame))

    cap.release()
    return frames


def run_extraction(data_dir):
    """Run frame extraction on all videos in data_dir."""
    video_dir = data_dir / "videos"
    frames_dir = data_dir / "qa_frames"

    with open(data_dir / "data.json") as f:
        data = json.load(f)

    frames_dir.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "success": 0, "missing_video": 0}

    for item in data:
        category = item["category"]
        name = item["name"]

        for step in item["steps"]:
            step_id = step["step_id"]

            for vid_info in step["video"]:
                video_id = get_video_id(vid_info["video_id"])
                video_path = video_dir / category / name / video_id / f"{video_id}.mp4"

                stats["total"] += 1

                if not video_path.exists():
                    stats["missing_video"] += 1
                    continue

                step_start = vid_info["step_start"]
                step_end = vid_info["step_end"]
                duration = step_end - step_start
                fps = vid_info["fps"]

                n_frames = 4 if duration < 10 else 8

                frames = extract_step_frames(video_path, step_start, step_end, fps, n_frames)

                if not frames:
                    continue

                # Save frames
                out_dir = frames_dir / category / name / f"step{step_id}" / video_id
                out_dir.mkdir(parents=True, exist_ok=True)

                for i, (t, frame) in enumerate(frames):
                    out_path = out_dir / f"frame_{i:02d}_t{t:.1f}s.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

                stats["success"] += 1

    print(f"Done. Total step-video pairs: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Missing video: {stats['missing_video']}")

    # Save stats
    with open(frames_dir / "extraction_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract video frames for IKEA-Bench QA construction")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root data directory (default: <repo>/data)")
    args = parser.parse_args()

    run_extraction(args.data_dir.resolve())


if __name__ == "__main__":
    main()
