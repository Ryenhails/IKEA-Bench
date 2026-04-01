"""Remove step number labels from IKEA manual step images.

Strategy:
1. Backup all originals to manual_img_original/
2. Detect the bold step number in the top-left corner using connected component analysis
3. White out ONLY the number region + small margin, preserving all diagram content
4. Save cleaned images back to the original paths

The number is always a bold black digit (1-2 chars) positioned in the top-left corner
of a white-background IKEA manual instruction diagram.
"""

import argparse
import os
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

# The step number is always within this region (relative to image size)
# Using a small window to avoid catching diagram content
SCAN_HEIGHT_FRAC = 0.12   # top 12% of image
SCAN_WIDTH_FRAC = 0.15    # left 15% of image
DARK_THRESHOLD = 100       # pixel value below this = "dark" (bold black text)
MARGIN_PX = 15             # extra whitespace margin around detected number
MIN_COMPONENT_PIXELS = 30  # ignore tiny noise


def detect_step_number(img_gray):
    """Detect the step number in the top-left corner.

    Returns (y_min, y_max, x_min, x_max) bounding box or None.
    """
    arr = np.array(img_gray)
    h, w = arr.shape

    # Scan region
    rh = max(int(h * SCAN_HEIGHT_FRAC), 80)
    rw = max(int(w * SCAN_WIDTH_FRAC), 80)
    region = arr[:rh, :rw]

    # Binary mask of dark pixels
    dark_mask = region < DARK_THRESHOLD

    if not dark_mask.any():
        return None

    # Connected component labeling
    labeled, n_components = ndimage.label(dark_mask)
    if n_components == 0:
        return None

    # Find the component closest to the top-left corner (0,0)
    # The step number should be the topmost-leftmost significant component
    best_label = None
    best_score = float('inf')
    component_boxes = []

    for label_id in range(1, n_components + 1):
        component_mask = labeled == label_id
        n_pixels = component_mask.sum()
        if n_pixels < MIN_COMPONENT_PIXELS:
            continue

        rows, cols = np.where(component_mask)
        y_min, y_max = rows.min(), rows.max()
        x_min, x_max = cols.min(), cols.max()

        # Score: distance from top-left (prefer top and left)
        score = y_min + x_min
        component_boxes.append({
            'label': label_id,
            'y_min': y_min, 'y_max': y_max,
            'x_min': x_min, 'x_max': x_max,
            'n_pixels': n_pixels,
            'score': score,
        })

    if not component_boxes:
        return None

    # Sort by score (top-left preference)
    component_boxes.sort(key=lambda c: c['score'])
    best = component_boxes[0]

    # The number might consist of multiple close components (e.g., "1" and "0" for "10")
    # Merge components that are close to the best one (within 2x its height horizontally)
    num_height = best['y_max'] - best['y_min']
    num_width = best['x_max'] - best['x_min']
    merge_x_threshold = max(num_width * 2, 60)
    merge_y_threshold = max(num_height * 0.5, 20)

    merged_y_min = best['y_min']
    merged_y_max = best['y_max']
    merged_x_min = best['x_min']
    merged_x_max = best['x_max']

    for comp in component_boxes[1:]:
        # Check if this component is horizontally adjacent to the current merged box
        y_overlap = (comp['y_min'] <= merged_y_max + merge_y_threshold and
                     comp['y_max'] >= merged_y_min - merge_y_threshold)
        x_close = comp['x_min'] <= merged_x_max + merge_x_threshold

        if y_overlap and x_close:
            merged_y_min = min(merged_y_min, comp['y_min'])
            merged_y_max = max(merged_y_max, comp['y_max'])
            merged_x_min = min(merged_x_min, comp['x_min'])
            merged_x_max = max(merged_x_max, comp['x_max'])

    # Sanity check: the number should not be too large
    # A step number is typically <100px tall and <150px wide
    box_h = merged_y_max - merged_y_min
    box_w = merged_x_max - merged_x_min
    if box_h > 120 or box_w > 180:
        # Probably caught diagram content -- fall back to just the best component
        return (best['y_min'], best['y_max'], best['x_min'], best['x_max'])

    return (merged_y_min, merged_y_max, merged_x_min, merged_x_max)


def process_image(src_path):
    """Remove step number from a single image. Returns True if modified."""
    img = Image.open(src_path)

    # Handle RGBA/palette images
    if img.mode == 'RGBA':
        # White background composite
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert('RGB')
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    img_gray = img.convert('L')
    bbox = detect_step_number(img_gray)

    if bbox is None:
        return False

    y_min, y_max, x_min, x_max = bbox

    # Apply margin
    y_min = max(0, y_min - MARGIN_PX)
    y_max = min(img.height - 1, y_max + MARGIN_PX)
    x_min = max(0, x_min - MARGIN_PX)
    x_max = min(img.width - 1, x_max + MARGIN_PX)

    # White out the number region
    arr = np.array(img)
    arr[y_min:y_max+1, x_min:x_max+1] = 255

    result = Image.fromarray(arr)
    result.save(src_path)
    return True


def run_removal(data_dir):
    """Run step number removal on all manual images in data_dir."""
    src_dir = os.path.join(str(data_dir), "manual_img")
    backup_dir = os.path.join(str(data_dir), "manual_img_original")

    # Step 1: Backup
    if os.path.exists(backup_dir):
        print(f"Backup already exists at {backup_dir}, skipping backup.")
    else:
        print(f"Backing up {src_dir} -> {backup_dir}")
        shutil.copytree(src_dir, backup_dir)
        print("Backup complete.")

    # Step 2: Process all images
    processed = 0
    modified = 0
    skipped = 0
    errors = []

    for root, dirs, files in sorted(os.walk(src_dir)):
        for f in sorted(files):
            if not f.endswith('.png'):
                continue
            path = os.path.join(root, f)
            rel = path.replace(src_dir + '/', '')
            processed += 1

            try:
                if process_image(path):
                    modified += 1
                    print(f"  [OK] {rel}")
                else:
                    skipped += 1
                    print(f"  [SKIP] {rel} (no number detected)")
            except Exception as e:
                errors.append((rel, str(e)))
                print(f"  [ERR] {rel}: {e}")

    print(f"\nDone: {processed} images processed, {modified} modified, "
          f"{skipped} skipped, {len(errors)} errors")
    if errors:
        print("Errors:")
        for rel, err in errors:
            print(f"  {rel}: {err}")

    return {"processed": processed, "modified": modified,
            "skipped": skipped, "errors": len(errors)}


def main():
    parser = argparse.ArgumentParser(
        description="Remove step number labels from IKEA manual images")
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root data directory (default: <repo>/data)")
    args = parser.parse_args()

    run_removal(args.data_dir.resolve())


if __name__ == "__main__":
    main()
