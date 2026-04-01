"""
Layer 1 representation analysis: CKA + Linear Probe + Cross-Modal Retrieval.

Processes 4 analysis models sequentially:
  1. Extract ViT last-layer + merger representations for all images
  2. Compute CKA between diagram and video representations
  3. Run video frame same/different linear probe
  4. Run diagram-to-video cross-modal retrieval

Usage:
  python analysis/layer1_representation.py [--models all|Qwen3-VL-8B,...]
"""
import json
import gc
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from representation_utils import (
    ANALYSIS_MODELS, BASE,
    load_analysis_model, print_model_structure,
    extract_vit_representations,
    linear_cka, bootstrap_cka,
    collect_image_data, get_product_split,
    get_cache_dir,
)


# ------------------------------------------------------
# CKA Analysis
# ------------------------------------------------------
def compute_cka_analysis(step_pairs, vit_reprs, merger_reprs):
    """Compute CKA between diagram and video representations at both levels.

    For each step with matched (diagram, video_frames):
      - diagram repr: ViT/merger output for diagram image
      - video repr: average of ViT/merger outputs for all video frames of that step
    """
    results = {}

    for level_name, reprs in [("vit", vit_reprs), ("merger", merger_reprs)]:
        diag_vecs = []
        video_vecs = []
        used_steps = []

        for (product, step_id), pair in sorted(step_pairs.items()):
            diag_path = pair["diagram"]
            video_paths = pair["video_frames"]

            if diag_path not in reprs:
                continue
            video_available = [p for p in video_paths if p in reprs]
            if not video_available:
                continue

            diag_vecs.append(reprs[diag_path].numpy())
            video_avg = torch.stack([reprs[p] for p in video_available]).mean(dim=0)
            video_vecs.append(video_avg.numpy())
            used_steps.append(f"{product}_step{step_id}")

        if len(diag_vecs) < 10:
            print(f"  WARNING: only {len(diag_vecs)} valid steps for CKA at {level_name} level")
            results[level_name] = {"cka": None, "n": len(diag_vecs)}
            continue

        X_diag = np.stack(diag_vecs)   # (n_steps, d)
        X_video = np.stack(video_vecs) # (n_steps, d)

        cka_result = bootstrap_cka(X_diag, X_video)
        cka_result["level"] = level_name
        results[level_name] = cka_result
        print(f"  CKA ({level_name}): {cka_result['cka']:.4f} "
              f"[{cka_result['ci_low']:.4f}, {cka_result['ci_high']:.4f}] "
              f"(n={cka_result['n']} steps)")

    return results


# ------------------------------------------------------
# Video Linear Probe
# ------------------------------------------------------
def build_video_pairs(image_meta, reprs, train_products, test_products,
                      max_train=20000, max_test=4000, neg_ratio=4, seed=42):
    """Build same/different step pairs for video probe.

    Positive pairs: same step (same product).
    Negative pairs: different steps within same product (hard negatives).
    Ratio: 1:neg_ratio (positive:negative).
    """
    rng = np.random.RandomState(seed)

    # Group video frames by (product, step_id)
    step_frames = defaultdict(list)
    for path, meta in image_meta.items():
        if meta["type"] == "video" and path in reprs and meta["step_id"] is not None:
            step_frames[(meta["product"], meta["step_id"])].append(path)

    def _make_pairs(products, max_pairs):
        pos_pairs = []
        neg_pairs = []

        product_steps = defaultdict(list)
        for (prod, sid), frames in step_frames.items():
            if prod in products and len(frames) >= 2:
                product_steps[prod].append((sid, frames))

        # Positive pairs: sample from same step
        for prod, steps in product_steps.items():
            for sid, frames in steps:
                n = len(frames)
                for i in range(n):
                    for j in range(i + 1, n):
                        pos_pairs.append((frames[i], frames[j], 1))

        # Negative pairs: different steps within same product
        for prod, steps in product_steps.items():
            if len(steps) < 2:
                continue
            for i, (sid_i, frames_i) in enumerate(steps):
                for j, (sid_j, frames_j) in enumerate(steps):
                    if i >= j:
                        continue
                    for fi in frames_i:
                        for fj in frames_j:
                            neg_pairs.append((fi, fj, 0))

        # Sample to target sizes
        n_pos = min(len(pos_pairs), max_pairs // (1 + neg_ratio))
        n_neg = min(len(neg_pairs), n_pos * neg_ratio)

        if n_pos > 0:
            pos_idx = rng.choice(len(pos_pairs), size=n_pos, replace=False)
            pos_pairs = [pos_pairs[i] for i in pos_idx]
        if n_neg > 0:
            neg_idx = rng.choice(len(neg_pairs), size=n_neg, replace=False)
            neg_pairs = [neg_pairs[i] for i in neg_idx]

        all_pairs = pos_pairs + neg_pairs
        rng.shuffle(all_pairs)
        return all_pairs

    train_pairs = _make_pairs(train_products, max_train)
    test_pairs = _make_pairs(test_products, max_test)

    return train_pairs, test_pairs


def run_video_probe(image_meta, vit_reprs, merger_reprs, train_products, test_products):
    """Train and evaluate video frame same/different linear probe."""
    results = {}

    for level_name, reprs in [("vit", vit_reprs), ("merger", merger_reprs)]:
        train_pairs, test_pairs = build_video_pairs(
            image_meta, reprs, train_products, test_products
        )

        if len(train_pairs) < 100 or len(test_pairs) < 50:
            print(f"  WARNING: insufficient video pairs at {level_name} level "
                  f"(train={len(train_pairs)}, test={len(test_pairs)})")
            results[level_name] = {"accuracy": None, "n_train": len(train_pairs), "n_test": len(test_pairs)}
            continue

        def _pairs_to_xy(pairs):
            X_list = []
            y_list = []
            for p1, p2, label in pairs:
                r1 = reprs[p1].numpy()
                r2 = reprs[p2].numpy()
                # Concatenation: [r1; r2; |r1-r2|]
                feat = np.concatenate([r1, r2, np.abs(r1 - r2)])
                X_list.append(feat)
                y_list.append(label)
            return np.stack(X_list), np.array(y_list)

        X_train, y_train = _pairs_to_xy(train_pairs)
        X_test, y_test = _pairs_to_xy(test_pairs)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auroc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auroc = None

        n_pos_train = int(y_train.sum())
        n_pos_test = int(y_test.sum())

        results[level_name] = {
            "accuracy": float(acc),
            "auroc": float(auroc) if auroc is not None else None,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "n_pos_train": n_pos_train,
            "n_neg_train": len(train_pairs) - n_pos_train,
            "n_pos_test": n_pos_test,
            "n_neg_test": len(test_pairs) - n_pos_test,
        }
        auroc_str = f"{auroc:.4f}" if auroc else "N/A"
        print(f"  Video probe ({level_name}): acc={acc:.4f}, auroc={auroc_str} "
              f"(train={len(train_pairs)}, test={len(test_pairs)})")

    return results


# ------------------------------------------------------
# Cross-Modal Diagram Retrieval
# ------------------------------------------------------
def run_diagram_retrieval(step_pairs, image_meta, vit_reprs, merger_reprs):
    """Cross-modal retrieval: diagram query -> video frame gallery.

    For each diagram, find its nearest video frame neighbor by cosine similarity.
    Recall@k: does the correct step's video frame appear in top-k?
    """
    results = {}

    for level_name, reprs in [("vit", vit_reprs), ("merger", merger_reprs)]:
        # Build gallery: all video frames with representations
        gallery_paths = []
        gallery_labels = []
        gallery_vecs = []
        for path, meta in sorted(image_meta.items()):
            if meta["type"] == "video" and path in reprs and meta["step_id"] is not None:
                gallery_paths.append(path)
                gallery_labels.append((meta["product"], meta["step_id"]))
                gallery_vecs.append(reprs[path].numpy())

        if not gallery_vecs:
            results[level_name] = {"recall@1": None, "n_queries": 0}
            continue

        gallery_matrix = np.stack(gallery_vecs)  # (n_gallery, d)
        # L2 normalize for cosine similarity
        gallery_norms = np.linalg.norm(gallery_matrix, axis=1, keepdims=True)
        gallery_matrix_normed = gallery_matrix / (gallery_norms + 1e-8)

        # Build queries: one per step (diagram)
        query_paths = []
        query_labels = []
        query_vecs = []
        for (product, step_id), pair in sorted(step_pairs.items()):
            diag_path = pair["diagram"]
            if diag_path and diag_path in reprs:
                query_paths.append(diag_path)
                query_labels.append((product, step_id))
                query_vecs.append(reprs[diag_path].numpy())

        if not query_vecs:
            results[level_name] = {"recall@1": None, "n_queries": 0}
            continue

        query_matrix = np.stack(query_vecs)  # (n_queries, d)
        query_norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix_normed = query_matrix / (query_norms + 1e-8)

        # Compute similarities
        sim = query_matrix_normed @ gallery_matrix_normed.T  # (n_queries, n_gallery)

        # Evaluate recall@k
        recall_at = {1: 0, 5: 0, 10: 0}
        for qi in range(len(query_labels)):
            q_label = query_labels[qi]
            ranked_idx = np.argsort(-sim[qi])
            for k in recall_at:
                top_k_labels = [gallery_labels[ranked_idx[j]] for j in range(min(k, len(ranked_idx)))]
                if q_label in top_k_labels:
                    recall_at[k] += 1

        n_q = len(query_labels)
        result = {
            f"recall@{k}": v / n_q for k, v in recall_at.items()
        }
        result["n_queries"] = n_q
        result["n_gallery"] = len(gallery_labels)

        results[level_name] = result
        print(f"  Diagram retrieval ({level_name}): "
              f"R@1={result['recall@1']:.4f}, R@5={result['recall@5']:.4f}, "
              f"R@10={result['recall@10']:.4f} "
              f"(queries={n_q}, gallery={len(gallery_labels)})")

    return results


# ------------------------------------------------------
# Main
# ------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Layer 1: CKA + linear probe + cross-modal retrieval analysis"
    )
    parser.add_argument("--models", default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--data-dir", default=str(BASE / "data"),
                        help="Path to data directory containing benchmark/qa_benchmark.json")
    parser.add_argument("--cache-dir", default=None,
                        help="HuggingFace model cache directory (default: HF_HOME or ~/.cache/huggingface)")
    parser.add_argument("--output", default=str(BASE / "results" / "representation_analysis.json"),
                        help="Output JSON path")
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = get_cache_dir()

    if args.models == "all":
        model_names = list(ANALYSIS_MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    # Collect image data
    print("Collecting image data from benchmark...")
    all_image_paths, step_pairs, image_meta = collect_image_data(data_dir=args.data_dir)
    train_products, test_products = get_product_split(image_meta)

    # Load existing results if any
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for model_name in model_names:
        if model_name in all_results:
            print(f"\nSkipping {model_name} (already in results)")
            continue

        print(f"\n{'#'*60}")
        print(f"Processing: {model_name}")
        print(f"{'#'*60}")
        t0 = time.time()

        # Load model
        model, processor, family = load_analysis_model(model_name, cache_dir=args.cache_dir)
        info = print_model_structure(model, family, model_name)

        # Extract representations
        print(f"\nExtracting representations for {len(all_image_paths)} images...")
        t_ext = time.time()
        vit_reprs, merger_reprs = extract_vit_representations(
            model, processor, family, all_image_paths
        )
        print(f"  Extraction time: {time.time()-t_ext:.1f}s")

        # CKA Analysis
        print(f"\nCKA Analysis...")
        cka_results = compute_cka_analysis(step_pairs, vit_reprs, merger_reprs)

        # Video Linear Probe
        print(f"\nVideo Linear Probe...")
        probe_results = run_video_probe(
            image_meta, vit_reprs, merger_reprs, train_products, test_products
        )

        # Cross-Modal Retrieval
        print(f"\nCross-Modal Diagram Retrieval...")
        retrieval_results = run_diagram_retrieval(
            step_pairs, image_meta, vit_reprs, merger_reprs
        )

        # Save results
        all_results[model_name] = {
            "model_info": info,
            "cka": cka_results,
            "video_probe": probe_results,
            "diagram_retrieval": retrieval_results,
            "n_images_extracted": len(vit_reprs),
            "n_step_pairs": len(step_pairs),
            "time_seconds": time.time() - t0,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

        # Unload model
        del model, processor, vit_reprs, merger_reprs
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Model unloaded. Total time for {model_name}: {time.time()-t0:.1f}s")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for name, res in all_results.items():
        print(f"\n{name}:")
        for level in ["vit", "merger"]:
            cka = res["cka"].get(level, {})
            vp = res["video_probe"].get(level, {})
            dr = res["diagram_retrieval"].get(level, {})
            cka_val = f"{cka['cka']:.4f}" if cka.get('cka') is not None else "N/A"
            vp_acc = f"{vp['accuracy']:.4f}" if vp.get('accuracy') is not None else "N/A"
            dr_r1 = f"{dr['recall@1']:.4f}" if dr.get('recall@1') is not None else "N/A"
            print(f"  [{level:6s}] CKA={cka_val}  VideoProbe={vp_acc}  DiagRetrieval_R@1={dr_r1}")


if __name__ == "__main__":
    main()
