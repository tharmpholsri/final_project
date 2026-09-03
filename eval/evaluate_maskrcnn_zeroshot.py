"""Evaluate the COCO-pretrained Mask R-CNN zero-shot baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)

from final_project.eval.metrics import (
    compute_counting_metrics,
    compute_per_stage_metrics,
    compute_segmentation_ap,
    format_per_stage_table,
)
from final_project.models.mask_rcnn import pick_device


COCO_POTTED_PLANT_ID = 64


def build_coco_maskrcnn():
    """COCO-pretrained Mask R-CNN, no head replacement (91 original classes)."""
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights)
    return model


def run_inference(
    model,
    coco_path: str | Path,
    images_dir: str | Path,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float,
    mode: str,  # "all-classes" | "plant-only"
) -> list[dict]:
    """
    Run model on every image in `coco_path` and return COCO-format
    prediction dicts, filtered by score_threshold (and class if mode=plant-only).
    """
    coco = COCO(str(coco_path))
    images_dir = Path(images_dir)
    predictions: list[dict] = []

    model.eval()
    for i, img_id in enumerate(coco.imgs, start=1):
        info = coco.imgs[img_id]
        img_pil = Image.open(images_dir / info["file_name"]).convert("RGB")
        arr = np.array(img_pil)
        tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).to(device)
        )

        with torch.no_grad():
            out = model([tensor])[0]

        scores = out["scores"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        masks_prob = out["masks"].cpu().numpy().squeeze(1)  # (N, H, W)

        keep = scores >= score_threshold
        if mode == "plant-only":
            keep &= labels == COCO_POTTED_PLANT_ID

        for idx in np.where(keep)[0]:
            binary = (masks_prob[idx] > mask_threshold).astype(np.uint8)
            if binary.sum() < 30:
                continue
            ys, xs = np.where(binary)
            if len(ys) == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
            rle = mask_utils.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("utf-8")
            predictions.append({
                "image_id": img_id,
                "category_id": 1,                # our target class id
                "segmentation": rle,
                "bbox": bbox,
                "area": int(binary.sum()),
                "score": float(scores[idx]),
            })

    return predictions


def sweep_score_thresholds(
    model, coco_path, images_dir, device,
    thresholds: Iterable[float], mode: str, mask_threshold: float,
) -> list[dict]:
    """Run inference once, then evaluate at multiple score thresholds."""
    rows = []
    for thr in thresholds:
        preds = run_inference(
            model, coco_path, images_dir, device,
            score_threshold=thr, mask_threshold=mask_threshold, mode=mode,
        )
        if not preds:
            rows.append({"threshold": thr, "AP50": 0.0, "MAE": 0.0,
                         "pred_total": 0, "mode": mode})
            continue
        ap = compute_segmentation_ap(coco_path, preds)
        ct = compute_counting_metrics(coco_path, preds)
        rows.append({
            "threshold": thr,
            "mode": mode,
            "AP50": ap["AP50"],
            "AP": ap["AP"],
            "MAE": ct["MAE"],
            "RMSE": ct["RMSE"],
            "DiC_mean": ct["DiC_mean"],
            "pred_total": ct["pred_total"],
        })
        print(f"  thr={thr:.2f}  {mode:12}  AP50={ap['AP50']:.3f}  "
              f"MAE={ct['MAE']:.2f}  pred={ct['pred_total']}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--out", default="results/maskrcnn_zeroshot_val.json")
    ap.add_argument('--score-threshold', type=float, default=None)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument('--mode', choices=['all-classes', 'plant-only'], default=None)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(f"loading COCO-pretrained Mask R-CNN (no head replacement) ...")
    model = build_coco_maskrcnn().to(device)
    print(f"model params: {sum(p.numel() for p in model.parameters()):,}")

    modes = [args.mode] if args.mode else ["all-classes", "plant-only"]

    if args.sweep or args.score_threshold is None:
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        all_rows = []
        for mode in modes:
            print(f"\n=== sweep mode={mode} ===")
            rows = sweep_score_thresholds(
                model, args.coco, args.images_dir, device,
                thresholds=thresholds, mode=mode,
                mask_threshold=args.mask_threshold,
            )
            all_rows.extend(rows)

        out_dir = Path(args.out).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        import csv
        sweep_csv = out_dir / "maskrcnn_zeroshot_sweep.csv"
        with sweep_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader(); w.writerows(all_rows)
        print(f"\nsweep results -> {sweep_csv}")

        best = max(all_rows, key=lambda r: r["AP50"])
        print(f"\nbest combo: mode={best['mode']}  threshold={best['threshold']}  "
              f"AP50={best['AP50']:.3f}  MAE={best['MAE']:.2f}")
    else:
        for mode in modes:
            print(f"\n=== single run mode={mode} threshold={args.score_threshold} ===")
            preds = run_inference(
                model, args.coco, args.images_dir, device,
                score_threshold=args.score_threshold,
                mask_threshold=args.mask_threshold,
                mode=mode,
            )
            out_path = Path(args.out).with_stem(Path(args.out).stem + f"_{mode}")
            out_path.write_text(json.dumps(preds))
            print(f"  predictions -> {out_path}  ({len(preds)} instances)")

            if preds:
                ap = compute_segmentation_ap(args.coco, preds)
                ct = compute_counting_metrics(args.coco, preds)
                per_stage = compute_per_stage_metrics(args.coco, preds)
                print(f"  AP50: {ap['AP50']:.3f}  AP: {ap['AP']:.3f}  "
                      f"MAE: {ct['MAE']:.2f}  pred_total: {ct['pred_total']}")
                print()
                print(format_per_stage_table(per_stage))


if __name__ == "__main__":
    main()
