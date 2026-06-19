"""
Crop each plant instance into its own image, using the provided plant-level
instance masks. Does NOT modify the original dataset.

Inputs  (relative to --root, read-only):
  images/   RGB photos
  masks/    plant-level instance masks (0 = bg, 1..N = plant instances)
  trays/    tray-cell masks (0 = tray frame, 1..20 = pot cell IDs)
            — needed only when --with-pots is set

Outputs (in --out, default ./crops):
  images/      RGB crops, one per plant instance
  masks/       binary plant mask for that crop (0 = bg, 255 = this plant)
  pots/        binary pot-cell mask for that crop          (--with-pots)
               — 0 = outside pot, 255 = inside this plant's pot cell
               — defines the region a pasted leaf is allowed to land in
                 during copy-paste augmentation
  metadata.csv mapping back to source (filename, plant_id, pot_id, bbox, ...)

Naming: <orig_stem>_p<plant_id>.png
   e.g. PS_Tray_080_15.png with plant_id=3  ->  PS_Tray_080_15_p3.png

Usage:
  python crop_plants.py --root Dataset/lettuce_PACE176 --out crops
  python crop_plants.py --root Dataset/lettuce_PACE176 --out crops --padding 80
  python crop_plants.py --root Dataset/lettuce_PACE176 --out crops --limit 5  # dry test
  python crop_plants.py --root Dataset/lettuce_PACE176 --out crops --with-pots
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Dataset/lettuce_PACE176",
                    help="dataset root (read-only)")
    ap.add_argument("--out", default="crops",
                    help="output folder (will be created)")
    ap.add_argument("--padding", type=int, default=50,
                    help="pixels of padding around plant bbox")
    ap.add_argument("--min-area", type=int, default=100,
                    help="skip plant instances smaller than this (pixels)")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process this many images (for testing)")
    ap.add_argument("--with-pots", action="store_true",
                    help="also crop pot-cell masks from trays/ — needed "
                         "for copy-paste augmentation (writes out/pots/)")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out_img = out / "images"
    out_msk = out / "masks"
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)

    img_dir = root / "images"
    mask_dir = root / "masks"
    tray_dir = root / "trays"

    if args.with_pots:
        out_pots = out / "pots"
        out_pots.mkdir(parents=True, exist_ok=True)
        if not tray_dir.exists():
            raise FileNotFoundError(
                f"--with-pots needs {tray_dir} but it doesn't exist"
            )

    names = sorted(p.name for p in img_dir.glob("*.png")
                   if (mask_dir / p.name).exists())
    if args.limit > 0:
        names = names[:args.limit]
    print(f"processing {len(names)} images, padding={args.padding}, "
          f"min_area={args.min_area}")

    meta_path = out / "metadata.csv"
    n_crops = 0
    n_skipped_small = 0
    n_skipped_empty = 0
    n_no_pot = 0

    header = [
        "crop_filename", "source_image", "tray_id", "timestep",
        "plant_id", "plant_area_px",
        "src_h", "src_w",
        "bbox_y0", "bbox_x0", "bbox_y1", "bbox_x1",
        "crop_y0", "crop_x0", "crop_y1", "crop_x1",
        "crop_h", "crop_w",
        "padding",
    ]
    if args.with_pots:
        header += ["pot_id", "pot_area_px_in_crop"]

    with meta_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i, name in enumerate(names, 1):
            if i % 20 == 0 or i == len(names):
                print(f"  {i}/{len(names)}  crops so far: {n_crops}")

            img = np.array(Image.open(img_dir / name).convert("RGB"))
            mask = np.array(Image.open(mask_dir / name))
            H, W = mask.shape[:2]

            tray_mask = None
            if args.with_pots:
                tray_path = tray_dir / name
                if tray_path.exists():
                    tray_mask = np.array(Image.open(tray_path))
                # tray_mask may legitimately be None if a tray file is
                # missing — handle per-plant below.

            # parse tray / timestep from name like "PS_Tray_080_15.png"
            stem = Path(name).stem
            parts = stem.split("_")
            try:
                tray_id = int(parts[2])
                timestep = int(parts[3])
            except (IndexError, ValueError):
                tray_id, timestep = -1, -1

            plant_ids = [v for v in np.unique(mask) if v != 0]
            if not plant_ids:
                n_skipped_empty += 1
                continue

            for pid in plant_ids:
                region = (mask == pid)
                area = int(region.sum())
                if area < args.min_area:
                    n_skipped_small += 1
                    continue

                ys, xs = np.where(region)
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())

                cy0 = max(0, y0 - args.padding)
                cx0 = max(0, x0 - args.padding)
                cy1 = min(H, y1 + args.padding + 1)
                cx1 = min(W, x1 + args.padding + 1)

                crop_img = img[cy0:cy1, cx0:cx1]
                crop_mask_full = mask[cy0:cy1, cx0:cx1]
                crop_mask_bin = np.where(crop_mask_full == pid, 255, 0).astype(np.uint8)

                crop_name = f"{stem}_p{int(pid)}.png"
                Image.fromarray(crop_img).save(out_img / crop_name)
                Image.fromarray(crop_mask_bin).save(out_msk / crop_name)

                # ── pot mask (optional, --with-pots) ───────────────────
                pot_id = 0
                pot_area = 0
                if args.with_pots and tray_mask is not None:
                    # Pick the pot ID most frequently overlapped by this
                    # plant's pixels — robust against irregular shapes /
                    # plants whose bbox centre falls outside the pot.
                    pot_ids_under_plant = tray_mask[region]
                    nonzero = pot_ids_under_plant[pot_ids_under_plant > 0]
                    if nonzero.size > 0:
                        vals, counts = np.unique(nonzero, return_counts=True)
                        pot_id = int(vals[counts.argmax()])
                        crop_pot_bin = np.where(
                            tray_mask[cy0:cy1, cx0:cx1] == pot_id,
                            255, 0,
                        ).astype(np.uint8)
                        pot_area = int((crop_pot_bin > 0).sum())
                        Image.fromarray(crop_pot_bin).save(out_pots / crop_name)
                    else:
                        # plant pixels fell entirely on tray frame (id=0) —
                        # very unusual; write an empty pot mask so file count
                        # matches images/ and downstream code can still load it
                        empty = np.zeros(crop_img.shape[:2], dtype=np.uint8)
                        Image.fromarray(empty).save(out_pots / crop_name)
                        n_no_pot += 1

                row = [
                    crop_name, name, tray_id, timestep,
                    int(pid), area,
                    H, W,
                    y0, x0, y1, x1,
                    cy0, cx0, cy1, cx1,
                    crop_img.shape[0], crop_img.shape[1],
                    args.padding,
                ]
                if args.with_pots:
                    row += [pot_id, pot_area]
                w.writerow(row)
                n_crops += 1

    print()
    print(f"done.")
    print(f"  crops written       : {n_crops}")
    print(f"  skipped (too small) : {n_skipped_small}  (min_area={args.min_area})")
    print(f"  skipped (empty mask): {n_skipped_empty}")
    if args.with_pots:
        print(f"  pots written        : {n_crops}  (empty if pot id unresolved)")
        print(f"  no-pot fallback     : {n_no_pot}  (plants fully on tray frame)")
    print(f"  output              : {out.resolve()}")
    print(f"  metadata            : {meta_path.resolve()}")


if __name__ == "__main__":
    main()
