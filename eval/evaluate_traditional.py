"""Evaluate distance-transform watershed segmentation."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from final_project.eval.metrics import (
    compute_counting_metrics,
    compute_per_stage_metrics,
    compute_segmentation_ap,
    format_per_stage_table,
)
from final_project.models.traditional import (
    TraditionalParams,
    instances_to_predictions,
    segment_leaves,
)


def run_dataset(
    coco_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    params: TraditionalParams,
    verbose: bool = False,
) -> list[dict]:
    coco = COCO(str(coco_path))
    predictions: list[dict] = []
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    t0 = time.time()
    for i, img_id in enumerate(coco.imgs, start=1):
        info = coco.imgs[img_id]
        img = np.array(Image.open(images_dir / info["file_name"]).convert("RGB"))
        mask = np.array(Image.open(masks_dir / info["file_name"]).convert("L"))

        instances = segment_leaves(img, mask, params)
        predictions.extend(
            instances_to_predictions(instances, image_id=img_id)
        )
        if verbose and i % 10 == 0:
            print(f"  {i}/{len(coco.imgs)}")

    if verbose:
        print(f"  done in {time.time() - t0:.1f}s "
              f"({len(coco.imgs)} images, {len(predictions)} predictions)")
    return predictions




SWEEP_GRID = {
    "sigma_dist":     [1.0, 2.0, 3.0, 4.0, 5.0],
    "min_distance":   [15, 20, 30, 40, 50],
    "sigma_gradient": [1.0, 2.0, 3.0, 4.0],
    "min_inst_size":  [100, 200, 300, 500],
}


def sweep(
    coco_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    out_dir: Path,
) -> tuple[TraditionalParams, dict]:
    keys = list(SWEEP_GRID.keys())
    combos = list(itertools.product(*[SWEEP_GRID[k] for k in keys]))
    print(f"sweeping {len(combos)} combinations on val ({coco_path}) ...")

    rows = []
    best: tuple[float, TraditionalParams] = (-1.0, TraditionalParams())
    t_start = time.time()

    for i, combo in enumerate(combos, start=1):
        params = TraditionalParams(**dict(zip(keys, combo)))
        preds = run_dataset(coco_path, images_dir, masks_dir, params)
        ap_metrics = compute_segmentation_ap(coco_path, preds)
        count_metrics = compute_counting_metrics(coco_path, preds)

        row = {
            **params.as_dict(),
            "AP50": ap_metrics["AP50"],
            "AP": ap_metrics["AP"],
            "MAE": count_metrics["MAE"],
            "RMSE": count_metrics["RMSE"],
        }
        rows.append(row)

        if row["AP50"] > best[0]:
            best = (row["AP50"], params)
            print(f"  [{i:>3}/{len(combos)}] new best AP50={row['AP50']:.3f}  "
                  f"{params.as_dict()}")
        elif i % 25 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(combos) - i)
            print(f"  [{i:>3}/{len(combos)}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "traditional_sweep.csv"
    import csv
    with sweep_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nfull sweep results -> {sweep_csv}")
    print(f"best AP50 = {best[0]:.3f} with params {best[1].as_dict()}")
    return best[1], {"best_AP50": best[0], "n_combinations": len(combos)}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coco', default='annotations/instances_validation.json')
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--masks-dir",  default="crops_full/masks")
    ap.add_argument('--out', default='results/traditional_val.json')
    ap.add_argument('--sweep', action='store_true')

    ap.add_argument("--sigma-dist",     type=float, default=3.0)
    ap.add_argument("--min-distance",   type=int,   default=30)
    ap.add_argument("--sigma-gradient", type=float, default=2.0)
    ap.add_argument("--min-inst-size",  type=int,   default=300)

    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        best_params, sweep_info = sweep(
            coco_path=args.coco,
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            out_dir=out_path.parent,
        )
        best_path = out_path.parent / "traditional_best_params.json"
        best_path.write_text(json.dumps({
            **best_params.as_dict(),
            "best_AP50": sweep_info["best_AP50"],
            "tuned_on": str(args.coco),
        }, indent=2))
        print(f"best params -> {best_path}")
        params = best_params
    else:
        params = TraditionalParams(
            sigma_dist=args.sigma_dist,
            min_distance=args.min_distance,
            sigma_gradient=args.sigma_gradient,
            min_inst_size=args.min_inst_size,
        )

    print(f"\nrunning final eval with params: {params.as_dict()}")
    preds = run_dataset(args.coco, args.images_dir, args.masks_dir, params, verbose=True)
    out_path.write_text(json.dumps(preds))
    print(f"\npredictions -> {out_path}")

    ap_metrics = compute_segmentation_ap(args.coco, preds)
    count_metrics = compute_counting_metrics(args.coco, preds)
    per_stage = compute_per_stage_metrics(args.coco, preds)

    print("\n=== OVERALL ===")
    print(f"  AP50       : {ap_metrics['AP50']:.3f}")
    print(f"  AP         : {ap_metrics['AP']:.3f}")
    print(f"  MAE count  : {count_metrics['MAE']:.2f}")
    print(f"  RMSE count : {count_metrics['RMSE']:.2f}")
    print(f"  DiC (bias) : {count_metrics['DiC_mean']:+.2f}")
    print(f"  GT total   : {count_metrics['gt_total']}  pred total: {count_metrics['pred_total']}")

    print("\n=== PER GROWTH STAGE ===")
    print(format_per_stage_table(per_stage))

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "params": params.as_dict(),
        "overall": {**ap_metrics, **count_metrics},
        "per_stage": per_stage,
    }, indent=2))
    print(f"\nmetrics summary -> {metrics_path}")


if __name__ == "__main__":
    main()
