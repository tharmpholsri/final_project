from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from dataclasses import asdict
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
from final_project.models.traditional import instances_to_predictions
from final_project.models.traditional_v2 import EdgeAwareParams
from final_project.models.traditional_v2_adaptive import (
    DEFAULT_AREA_THRESHOLD,
    AdaptiveParams,
    classify_tier,
    segment_leaves_adaptive,
)







GRID_SMALL = {
    "sigma_dist":        [2.0, 3.0, 4.0],         
    "min_distance":      [20, 30, 40],            
    "shadow_percentile": [25.0, 30.0],            
    "w_shadow":          [0.15, 0.30],            
    "min_inst_size":     [100],                   
}


GRID_LARGE = {
    "sigma_dist":        [4.0, 5.0, 6.0],         
    "min_distance":      [50, 60, 70, 80],        
    "shadow_percentile": [5.0, 10.0, 15.0],       
    "w_shadow":          [0.10, 0.20, 0.30],      
    "min_inst_size":     [500, 1000],             
}


def make_params(values: dict) -> EdgeAwareParams:
    w_shadow = values["w_shadow"]
    remaining = max(0.0, 1.0 - w_shadow)
    return EdgeAwareParams(
        sigma_dist=values["sigma_dist"],
        min_distance=int(values["min_distance"]),
        shadow_percentile=values["shadow_percentile"],
        w_shadow=w_shadow,
        w_gradient=remaining * 0.6,
        w_canny=remaining * 0.4,
        min_inst_size=int(values["min_inst_size"]),
    )







def run_dataset(
    coco_path: str | Path,
    images_dir: str | Path,
    masks_dir: str | Path,
    params: AdaptiveParams,
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
        instances = segment_leaves_adaptive(img, mask, params)
        predictions.extend(instances_to_predictions(instances, image_id=img_id))
        if verbose and i % 10 == 0:
            print(f"  {i}/{len(coco.imgs)}")
    if verbose:
        print(f"  done in {time.time() - t0:.1f}s "
              f"({len(coco.imgs)} images, {len(predictions)} predictions)")
    return predictions







def split_images_by_tier(coco: COCO, masks_dir: Path, area_threshold: int) -> dict[str, list[int]]:
    by_tier: dict[str, list[int]] = {"small": [], "large": []}
    for img_id, info in coco.imgs.items():
        mask = np.array(Image.open(masks_dir / info["file_name"]).convert("L"))
        tier = classify_tier(mask, area_threshold)
        by_tier[tier].append(img_id)
    return by_tier


def sweep_one_tier(
    coco_path: str | Path,
    images_dir: Path,
    masks_dir: Path,
    grid: dict,
    tier_name: str,
    tier_image_ids: list[int],
    out_dir: Path,
) -> tuple[EdgeAwareParams, float, dict]:
    """Tune parameters for one plant-size group."""
    coco = COCO(str(coco_path))
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"\n=== sweeping {len(combos)} combos for tier='{tier_name}' "
          f"on {len(tier_image_ids)} images ===")

    
    from tempfile import NamedTemporaryFile
    sub_data = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "categories": coco.dataset["categories"],
        "images": [im for im in coco.dataset["images"] if im["id"] in tier_image_ids],
        "annotations": [a for a in coco.dataset["annotations"] if a["image_id"] in tier_image_ids],
    }
    with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(sub_data, f)
        sub_path = f.name

    best: tuple[float, EdgeAwareParams, dict] = (-1.0, EdgeAwareParams(), {})
    rows = []
    t_start = time.time()

    for i, combo in enumerate(combos, start=1):
        values = dict(zip(keys, combo))
        params = make_params(values)

        preds = []
        for img_id in tier_image_ids:
            info = coco.imgs[img_id]
            img = np.array(Image.open(images_dir / info["file_name"]).convert("RGB"))
            mask = np.array(Image.open(masks_dir / info["file_name"]).convert("L"))
            from final_project.models.traditional_v2 import segment_leaves_edge_aware
            instances = segment_leaves_edge_aware(img, mask, params=params)
            preds.extend(instances_to_predictions(instances, image_id=img_id))

        ap = compute_segmentation_ap(sub_path, preds)
        ct = compute_counting_metrics(sub_path, preds)
        row = {**values, "tier": tier_name, "AP50": ap["AP50"], "AP": ap["AP"],
               "MAE": ct["MAE"], "RMSE": ct["RMSE"], "pred_total": ct["pred_total"],
               "gt_total": ct["gt_total"]}
        rows.append(row)

        if row["AP50"] > best[0]:
            best = (row["AP50"], params, values)
            print(f"  [{i:>3}/{len(combos)}] new best AP50={row['AP50']:.3f}  {values}")
        elif i % 10 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(combos) - i)
            print(f"  [{i:>3}/{len(combos)}] elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    Path(sub_path).unlink(missing_ok=True)

    
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"traditional_v2_adaptive_sweep_{tier_name}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  tier results -> {csv_path}")
    return best[1], best[0], best[2]







def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--masks-dir",  default="crops_full/masks")
    ap.add_argument("--out", default="results/traditional_v2_adaptive_val.json")
    ap.add_argument("--area-threshold", type=int, default=DEFAULT_AREA_THRESHOLD)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument('--best')
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)

    coco = COCO(args.coco)
    by_tier = split_images_by_tier(coco, masks_dir, args.area_threshold)
    print(f"area threshold = {args.area_threshold} px")
    print(f"  small tier: {len(by_tier['small'])} images")
    print(f"  large tier: {len(by_tier['large'])} images")

    if args.sweep:
        small_params, small_ap, small_vals = sweep_one_tier(
            args.coco, images_dir, masks_dir,
            GRID_SMALL, "small", by_tier["small"], out_path.parent,
        )
        large_params, large_ap, large_vals = sweep_one_tier(
            args.coco, images_dir, masks_dir,
            GRID_LARGE, "large", by_tier["large"], out_path.parent,
        )
        adaptive = AdaptiveParams(
            small_params=small_params,
            large_params=large_params,
            area_threshold=args.area_threshold,
        )
        best_path = out_path.parent / "traditional_v2_adaptive_best_params.json"
        best_path.write_text(json.dumps({
            "area_threshold": args.area_threshold,
            "small": {**asdict(small_params), "best_AP50": small_ap, "values": small_vals},
            "large": {**asdict(large_params), "best_AP50": large_ap, "values": large_vals},
            "tuned_on": str(args.coco),
        }, indent=2))
        print(f"\nbest params -> {best_path}")
        print(f"small tier best AP50: {small_ap:.3f}")
        print(f"large tier best AP50: {large_ap:.3f}")
    elif args.best:
        cfg = json.loads(Path(args.best).read_text())
        adaptive = AdaptiveParams(
            small_params=EdgeAwareParams(**{k: v for k, v in cfg["small"].items()
                                            if k in EdgeAwareParams.__dataclass_fields__}),
            large_params=EdgeAwareParams(**{k: v for k, v in cfg["large"].items()
                                            if k in EdgeAwareParams.__dataclass_fields__}),
            area_threshold=cfg.get("area_threshold", DEFAULT_AREA_THRESHOLD),
        )
        print(f"loaded best params from {args.best}")
    else:
        adaptive = AdaptiveParams(area_threshold=args.area_threshold)
        print("using AdaptiveParams defaults")

    
    print(f"\nrunning final eval ...")
    preds = run_dataset(args.coco, images_dir, masks_dir, adaptive, verbose=True)
    out_path.write_text(json.dumps(preds))
    print(f"predictions -> {out_path}")

    ap_metrics = compute_segmentation_ap(args.coco, preds)
    count_metrics = compute_counting_metrics(args.coco, preds)
    per_stage = compute_per_stage_metrics(args.coco, preds)

    print("\n=== OVERALL (V2-adaptive) ===")
    print(f"  AP50       : {ap_metrics['AP50']:.3f}")
    print(f"  AP         : {ap_metrics['AP']:.3f}")
    print(f"  MAE count  : {count_metrics['MAE']:.2f}")
    print(f"  RMSE count : {count_metrics['RMSE']:.2f}")
    print(f"  DiC (bias) : {count_metrics['DiC_mean']:+.2f}")
    print(f"  GT total   : {count_metrics['gt_total']}  pred total: {count_metrics['pred_total']}")

    print("\n=== PER GROWTH STAGE (V2-adaptive) ===")
    print(format_per_stage_table(per_stage))

    metrics_path = out_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({
        "area_threshold": adaptive.area_threshold,
        "small_params": adaptive.small_params.as_dict(),
        "large_params": adaptive.large_params.as_dict(),
        "overall": {**ap_metrics, **count_metrics},
        "per_stage": per_stage,
    }, indent=2))
    print(f"\nmetrics summary -> {metrics_path}")


if __name__ == "__main__":
    main()
