"""IKEA-Bench Data Setup

Downloads the complete dataset from HuggingFace, including:
  - qa_benchmark.json (1,623 questions)
  - step_descriptions.json (132 text descriptions)
  - manual_img/ (133 assembly instruction diagrams)
  - qa_frames/ (2,570 video frames)

Usage:
  python setup_data.py                    # Download to ./data
  python setup_data.py --data-dir /path   # Download to custom path
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="IKEA-Bench Data Setup -- download dataset from HuggingFace")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Target directory for dataset (default: <repo>/data)")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run:")
        print("  pip install huggingface_hub")
        return

    print("IKEA-Bench Data Setup")
    print(f"Target directory: {data_dir}")
    print(f"Downloading from: https://huggingface.co/datasets/Ryenhails/ikea-bench")
    print()

    snapshot_download(
        repo_id="Ryenhails/ikea-bench",
        repo_type="dataset",
        local_dir=str(data_dir),
    )

    # Validate
    print(f"\n{'='*60}")
    print("Validating download...")

    checks = {
        "qa_benchmark.json": data_dir / "qa_benchmark.json",
        "step_descriptions.json": data_dir / "step_descriptions.json",
        "manual_img/": data_dir / "manual_img",
        "qa_frames/": data_dir / "qa_frames",
    }

    all_ok = True
    for label, path in checks.items():
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {label}")
        if not exists:
            all_ok = False

    if (data_dir / "manual_img").exists():
        n_diagrams = sum(1 for _ in (data_dir / "manual_img").rglob("*.png"))
        print(f"  Manual diagrams: {n_diagrams} files")

    if (data_dir / "qa_frames").exists():
        n_frames = sum(1 for _ in (data_dir / "qa_frames").rglob("*.jpg"))
        print(f"  Video frames: {n_frames} files")

    if all_ok:
        print(f"\nSetup complete! You can now run evaluation:")
        print(f"  python -m ikea_bench.eval \\")
        print(f"    --model qwen3-vl-8b \\")
        print(f"    --setting baseline \\")
        print(f"    --input {data_dir}/qa_benchmark.json \\")
        print(f"    --data-dir {data_dir} \\")
        print(f"    --output results/output.json")
    else:
        print("\nSome files are missing. Check the output above.")


if __name__ == "__main__":
    main()
