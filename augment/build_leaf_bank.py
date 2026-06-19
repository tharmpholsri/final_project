"""
Build the leaf bank for copy-paste augmentation.

Reads `annotations/instances_train_set.json` (the hand-labelled canopy-
train crops) and extracts each leaf annotation as a standalone "stamp":
a tight RGB cutout + binary mask + metadata. These stamps are sampled
at training time by `copy_paste.py` and pasted into other crops to
simulate a denser canopy.

Output layout
-------------
    leaf_bank/
    ├── leaves/<idx>.png          RGB cutout, background pixels zeroed
    ├── masks/<idx>.png           binary 0/255 mask matching leaves/<idx>.png
    ├── metadata.json             list of dicts (one per stamp)
    └── qa_grid.png               (optional) 50-random visual sanity grid

Metadata schema (per stamp):
    {
      "idx":            "0001",       # zero-padded filename stem
      "source_file":    "PS_Tray_075_18_p2.png",
      "source_stage":   "late",       # early/mid/late/canopy
      "source_tray":    75,
      "source_timestep": 18,
      "ann_id":         142,          # annotation id in the source COCO
      "bbox":           [x0, y0, w, h],
      "area_px":        3437,         # mask pixel count
      "aspect_ratio":   1.57,         # cutout width / height
      "cutout_h":       55,
      "cutout_w":       86
    }

Usage
-----
    PY=/opt/miniconda3/envs/final_project/bin/python

    # default — read instances_train_set.json, write leaf_bank/
    $PY -m final_project.augment.build_leaf_bank

    # custom paths
    $PY -m final_project.augment.build_leaf_bank \\
        --coco annotations/instances_train_set.json \\
        --images-dir crops_full/images \\
        --out leaf_bank

    # skip the QA grid
    $PY -m final_project.augment.build_leaf_bank --no-qa-grid
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from final_project.eval.metrics import stage_from_filename


# ════════════════════════════════════════════════════════════════════
# Stamp extraction
# ════════════════════════════════════════════════════════════════════
def parse_tray_timestep(filename: str) -> tuple[int, int]:
    """Recover (tray_id, timestep) from a crop filename like
    'PS_Tray_075_18_p2.png'. Returns (-1, -1) on any failure."""
    m = re.match(r"PS_Tray_(\d+)_(\d+)_p\d+", filename)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


def crop_to_bbox(
    image: np.ndarray, mask: np.ndarray, bbox: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image + mask to the COCO bbox [x, y, w, h], rounded to int and
    clamped to image bounds."""
    H, W = mask.shape[:2]
    x, y, w, h = bbox
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(W, int(np.ceil(x + w)))
    y1 = min(H, int(np.ceil(y + h)))
    return image[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def make_stamp(
    image: np.ndarray, mask01: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the mask to the RGB cutout — outside-leaf pixels go to 0."""
    cutout = image.copy()
    cutout[mask01 == 0] = 0
    mask255 = (mask01 * 255).astype(np.uint8)
    return cutout, mask255


# ════════════════════════════════════════════════════════════════════
# QA grid
# ════════════════════════════════════════════════════════════════════
def render_qa_grid(
    stamps: list[tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    n_samples: int = 50,
    cols: int = 10,
    cell_px: int = 96,
    bg_color: tuple = (40, 40, 40),
) -> None:
    """Render up to n_samples random stamps as a single PNG grid for
    quick visual review. Each cell is `cell_px`×`cell_px` with the
    cutout letterboxed inside."""
    n_samples = min(n_samples, len(stamps))
    sample = random.sample(stamps, n_samples)
    rows = (n_samples + cols - 1) // cols
    grid = np.full(
        (rows * cell_px, cols * cell_px, 3), bg_color, dtype=np.uint8
    )

    for i, (cutout, _mask) in enumerate(sample):
        r, c = divmod(i, cols)
        h, w = cutout.shape[:2]
        scale = (cell_px - 4) / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        resized = np.array(
            Image.fromarray(cutout).resize((new_w, new_h), Image.BILINEAR)
        )
        y0 = r * cell_px + (cell_px - new_h) // 2
        x0 = c * cell_px + (cell_px - new_w) // 2
        grid[y0:y0 + new_h, x0:x0 + new_w] = resized

    Image.fromarray(grid).save(out_path)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a leaf bank from a COCO instance-segmentation file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--coco", default="annotations/instances_train_set.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--out", default="leaf_bank")
    ap.add_argument("--min-area", type=int, default=200,
                    help="drop stamps smaller than this (mask pixel count)")
    ap.add_argument("--no-qa-grid", action="store_true",
                    help="skip rendering qa_grid.png")
    ap.add_argument("--qa-samples", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    coco = COCO(args.coco)
    images_dir = Path(args.images_dir)
    out = Path(args.out)
    out_leaves = out / "leaves"
    out_masks = out / "masks"
    out_leaves.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    n_anns = len(coco.anns)
    n_imgs = len(coco.imgs)
    print(f"COCO: {n_imgs} images, {n_anns} annotations")
    print(f"images dir: {images_dir}")
    print(f"out dir:    {out}")

    metadata: list[dict] = []
    stamps_for_qa: list[tuple[np.ndarray, np.ndarray]] = []
    n_skipped_small = 0
    n_skipped_empty = 0

    # Cache loaded source images so we don't re-decode for every leaf
    # (most source crops have ~6 leaves each)
    img_cache: dict[int, np.ndarray] = {}

    sorted_ann_ids = sorted(coco.anns.keys())
    for stamp_idx, ann_id in enumerate(sorted_ann_ids, start=1):
        ann = coco.anns[ann_id]
        img_info = coco.imgs[ann["image_id"]]
        file_name = img_info["file_name"]

        if ann["image_id"] not in img_cache:
            img_cache[ann["image_id"]] = np.array(
                Image.open(images_dir / file_name).convert("RGB")
            )
        image = img_cache[ann["image_id"]]

        mask01 = coco.annToMask(ann).astype(np.uint8)  # (H, W) 0/1

        if mask01.sum() < args.min_area:
            n_skipped_small += 1
            continue

        cutout_img, cutout_mask01 = crop_to_bbox(image, mask01, ann["bbox"])
        if cutout_mask01.sum() == 0:
            n_skipped_empty += 1
            continue

        stamp_img, stamp_mask = make_stamp(cutout_img, cutout_mask01)
        h, w = stamp_img.shape[:2]

        tray_id, timestep = parse_tray_timestep(file_name)

        idx_str = f"{stamp_idx:04d}"
        Image.fromarray(stamp_img).save(out_leaves / f"{idx_str}.png")
        Image.fromarray(stamp_mask).save(out_masks / f"{idx_str}.png")

        metadata.append({
            "idx": idx_str,
            "source_file": file_name,
            "source_stage": stage_from_filename(file_name) or "unknown",
            "source_tray": tray_id,
            "source_timestep": timestep,
            "ann_id": int(ann_id),
            "bbox": [float(v) for v in ann["bbox"]],
            "area_px": int(mask01.sum()),
            "aspect_ratio": float(w / max(h, 1)),
            "cutout_h": int(h),
            "cutout_w": int(w),
        })

        if len(stamps_for_qa) < args.qa_samples * 4:
            # collect more than needed so random sample is well-mixed
            stamps_for_qa.append((stamp_img, stamp_mask))

        if stamp_idx % 100 == 0 or stamp_idx == n_anns:
            print(f"  processed {stamp_idx}/{n_anns}  kept {len(metadata)}")

    # ── Write metadata.json ──────────────────────────────────────────
    meta_path = out / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # ── QA grid ─────────────────────────────────────────────────────
    if not args.no_qa_grid and stamps_for_qa:
        qa_path = out / "qa_grid.png"
        render_qa_grid(stamps_for_qa, qa_path, n_samples=args.qa_samples)
        print(f"  qa grid → {qa_path}")

    # ── Summary by stage ────────────────────────────────────────────
    print()
    print("=== leaf bank summary ===")
    print(f"  kept stamps:     {len(metadata):>5}")
    print(f"  skipped (small): {n_skipped_small:>5}  (min_area={args.min_area})")
    print(f"  skipped (empty): {n_skipped_empty:>5}")
    print(f"  metadata:        {meta_path}")

    stage_counts: dict[str, int] = {}
    area_by_stage: dict[str, list[int]] = {}
    for m in metadata:
        s = m["source_stage"]
        stage_counts[s] = stage_counts.get(s, 0) + 1
        area_by_stage.setdefault(s, []).append(m["area_px"])

    print()
    print("  per-stage breakdown:")
    print(f"  {'stage':<10}  {'count':>6}  {'min_area':>10}  {'med_area':>10}  {'max_area':>10}")
    for s in ("early", "mid", "late", "canopy", "unknown"):
        if s not in stage_counts:
            continue
        areas = sorted(area_by_stage[s])
        print(f"  {s:<10}  {stage_counts[s]:>6}  "
              f"{areas[0]:>10,}  {areas[len(areas) // 2]:>10,}  {areas[-1]:>10,}")


if __name__ == "__main__":
    main()
