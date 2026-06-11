"""
Evaluate a trained Mask R-CNN checkpoint on the PACE val or test set.

Mirrors `evaluate_maskrcnn_zeroshot.py` but loads a checkpoint that was
fine-tuned on CVPPP via `train_maskrcnn.py`. The model is built with
`build_maskrcnn(num_classes=2)` (heads already swapped for our 2-class
task) and the saved `state_dict` is loaded on top.

Outputs
-------
  - `<out>` (JSON): list of COCO-format prediction dicts, ready to feed
    into pycocotools eval or the visualisation pipeline used for the
    traditional methods.
  - Stdout: overall metrics (AP50, AP, MAE, RMSE, DiC, prediction total)
    plus the per-stage table (early / mid / late / canopy).
  - Optional: `--sweep` mode iterates over a list of score thresholds
    and picks the best by val AP50 (do this on val, not test).

Usage
-----
    PY=/opt/miniconda3/envs/final_project/bin/python

    # default: run best checkpoint on val
    $PY -m final_project.eval.evaluate_maskrcnn

    # final test eval with chosen score threshold
    $PY -m final_project.eval.evaluate_maskrcnn \\
        --coco annotations/instances_test_set.json \\
        --out  results/maskrcnn_cvppp_test.json \\
        --score-threshold 0.5

    # sweep score thresholds on val (one inference pass, multiple eval)
    $PY -m final_project.eval.evaluate_maskrcnn --sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from final_project.eval.metrics import (
    compute_counting_metrics,
    compute_per_stage_metrics,
    compute_segmentation_ap,
    format_per_stage_table,
)
from final_project.models.mask_rcnn import build_maskrcnn, pick_device


# ════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ════════════════════════════════════════════════════════════════════
def load_trained_model(
    checkpoint_path: str | Path,
    num_classes: int,
    device: torch.device,
    trainable_backbone_layers: int = 3,
) -> torch.nn.Module:
    """Build a Mask R-CNN with our 2-class heads and load the trained weights."""
    model = build_maskrcnn(
        num_classes=num_classes,
        pretrained=False,  # we are loading our own weights
        trainable_backbone_layers=trainable_backbone_layers,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        meta = {k: state.get(k) for k in ("epoch", "best_ap50")}
    else:
        # bare state_dict
        model.load_state_dict(state)
        meta = {}
    model.to(device).eval()
    return model, meta


# ════════════════════════════════════════════════════════════════════
# Inference
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    coco_path: str | Path,
    images_dir: str | Path,
    device: torch.device,
    score_threshold: float,
    mask_threshold: float = 0.5,
    min_mask_area: int = 30,
) -> list[dict]:
    """Run the model on every image in `coco_path` and return COCO predictions.

    The model is expected to be in eval mode and on `device`. Predictions
    below `score_threshold` are dropped, masks are binarised at
    `mask_threshold`, and tiny masks below `min_mask_area` are filtered
    out (same as the zero-shot baseline for comparability).
    """
    coco = COCO(str(coco_path))
    images_dir = Path(images_dir)
    predictions: list[dict] = []

    for img_id in coco.imgs:
        info = coco.imgs[img_id]
        img_pil = Image.open(images_dir / info["file_name"]).convert("RGB")
        arr = np.array(img_pil)
        tensor = (
            torch.from_numpy(arr)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .to(device)
        )

        out = model([tensor])[0]
        scores = out["scores"].cpu().numpy()
        masks_prob = out["masks"].cpu().numpy().squeeze(1)  # (N, H, W)

        keep = scores >= score_threshold
        for idx in np.where(keep)[0]:
            binary = (masks_prob[idx] > mask_threshold).astype(np.uint8)
            if binary.sum() < min_mask_area:
                continue
            ys, xs = np.where(binary)
            if len(ys) == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            bbox = [
                float(x0), float(y0),
                float(x1 - x0 + 1), float(y1 - y0 + 1),
            ]
            rle = mask_utils.encode(np.asfortranarray(binary))
            rle["counts"] = rle["counts"].decode("utf-8")
            predictions.append({
                "image_id": img_id,
                "category_id": 1,
                "segmentation": rle,
                "bbox": bbox,
                "area": int(binary.sum()),
                "score": float(scores[idx]),
            })

    return predictions


# ════════════════════════════════════════════════════════════════════
# Eval helpers
# ════════════════════════════════════════════════════════════════════
def compute_all_metrics(coco_path: str | Path, predictions: list[dict]) -> dict:
    """Bundle AP, counting, and per-stage metrics into one dict."""
    if not predictions:
        return {
            "AP50": 0.0, "AP": 0.0, "AP75": 0.0,
            "MAE": 0.0, "RMSE": 0.0, "DiC_mean": 0.0,
            "pred_total": 0,
            "per_stage": {},
        }
    ap = compute_segmentation_ap(coco_path, predictions)
    ct = compute_counting_metrics(coco_path, predictions)
    per_stage = compute_per_stage_metrics(coco_path, predictions)
    return {
        "AP50": float(ap["AP50"]),
        "AP": float(ap["AP"]),
        "AP75": float(ap["AP75"]),
        "MAE": float(ct["MAE"]),
        "RMSE": float(ct["RMSE"]),
        "DiC_mean": float(ct["DiC_mean"]),
        "pred_total": int(ct["pred_total"]),
        "per_stage": per_stage,
    }


def print_summary(metrics: dict, header: str = "") -> None:
    if header:
        print(f"\n=== {header} ===")
    print(
        f"  AP50={metrics['AP50']:.4f}  "
        f"AP={metrics['AP']:.4f}  "
        f"AP75={metrics['AP75']:.4f}  "
        f"MAE={metrics['MAE']:.2f}  "
        f"RMSE={metrics['RMSE']:.2f}  "
        f"DiC={metrics['DiC_mean']:+.2f}  "
        f"pred={metrics['pred_total']}"
    )
    if metrics["per_stage"]:
        print()
        print(format_per_stage_table(metrics["per_stage"]))


# ════════════════════════════════════════════════════════════════════
# Sweep mode
# ════════════════════════════════════════════════════════════════════
def sweep_score_thresholds(
    model: torch.nn.Module,
    coco_path: str | Path,
    images_dir: str | Path,
    device: torch.device,
    thresholds: Iterable[float],
    mask_threshold: float,
    min_mask_area: int,
) -> tuple[list[dict], dict]:
    """Run inference once at the lowest threshold, then re-eval at each higher.

    Returns (rows for CSV, best row).
    """
    thresholds = sorted(thresholds)
    base_thr = thresholds[0]
    print(f"running inference at base threshold {base_thr:.2f} ...")
    preds_all = run_inference(
        model, coco_path, images_dir, device,
        score_threshold=base_thr,
        mask_threshold=mask_threshold,
        min_mask_area=min_mask_area,
    )
    print(f"  {len(preds_all)} raw predictions at thr={base_thr:.2f}")

    rows: list[dict] = []
    for thr in thresholds:
        preds = [p for p in preds_all if p["score"] >= thr]
        if not preds:
            rows.append({"threshold": thr, "AP50": 0.0, "AP": 0.0,
                         "MAE": 0.0, "pred_total": 0})
            continue
        ap = compute_segmentation_ap(coco_path, preds)
        ct = compute_counting_metrics(coco_path, preds)
        rows.append({
            "threshold": thr,
            "AP50": float(ap["AP50"]),
            "AP": float(ap["AP"]),
            "MAE": float(ct["MAE"]),
            "RMSE": float(ct["RMSE"]),
            "DiC_mean": float(ct["DiC_mean"]),
            "pred_total": int(ct["pred_total"]),
        })
        print(f"  thr={thr:.2f}  AP50={ap['AP50']:.4f}  "
              f"MAE={ct['MAE']:.2f}  pred={ct['pred_total']}")

    best = max(rows, key=lambda r: r["AP50"])
    print(f"\nbest threshold: {best['threshold']:.2f}  "
          f"AP50={best['AP50']:.4f}  MAE={best['MAE']:.2f}")
    return rows, best


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate a trained Mask R-CNN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")

    # Checkpoint
    ap.add_argument("--checkpoint",
                    default="checkpoints/maskrcnn_cvppp_best.pth")
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--trainable-backbone-layers", type=int, default=3)

    # Inference
    ap.add_argument("--score-threshold", type=float, default=0.5)
    ap.add_argument("--mask-threshold", type=float, default=0.5)
    ap.add_argument("--min-mask-area", type=int, default=30)

    # Sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep score thresholds 0.05–0.7")
    ap.add_argument(
        "--sweep-thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7",
        help="comma-separated thresholds for --sweep",
    )

    # Outputs
    ap.add_argument("--out", default="results/maskrcnn_cvppp_val.json",
                    help="path to write predictions JSON")
    ap.add_argument("--metrics-out", default=None,
                    help="optional path to write summary metrics JSON "
                         "(default: alongside --out with .metrics.json)")
    ap.add_argument("--sweep-csv", default=None,
                    help="optional path to write sweep CSV "
                         "(default: alongside --out as _sweep.csv)")

    # Device
    ap.add_argument("--device", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    model, meta = load_trained_model(
        args.checkpoint,
        num_classes=args.num_classes,
        device=device,
        trainable_backbone_layers=args.trainable_backbone_layers,
    )
    if meta.get("epoch") is not None:
        print(f"  loaded from epoch {meta['epoch'] + 1}  "
              f"(best AP50 at train time: {meta.get('best_ap50', 0.0):.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Sweep mode ───────────────────────────────────────────────────
    if args.sweep:
        thresholds = [float(x) for x in args.sweep_thresholds.split(",")]
        rows, best = sweep_score_thresholds(
            model, args.coco, args.images_dir, device,
            thresholds=thresholds,
            mask_threshold=args.mask_threshold,
            min_mask_area=args.min_mask_area,
        )
        sweep_csv = (
            Path(args.sweep_csv) if args.sweep_csv
            else out_path.with_name(out_path.stem + "_sweep.csv")
        )
        with sweep_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"sweep results → {sweep_csv}")

        # Save final predictions at the best threshold
        best_preds = run_inference(
            model, args.coco, args.images_dir, device,
            score_threshold=best["threshold"],
            mask_threshold=args.mask_threshold,
            min_mask_area=args.min_mask_area,
        )
        out_path.write_text(json.dumps(best_preds))
        print(f"predictions at best threshold → {out_path}  "
              f"({len(best_preds)} instances)")

        metrics = compute_all_metrics(args.coco, best_preds)
        print_summary(metrics, header=f"best (thr={best['threshold']:.2f})")
        metrics_path = (
            Path(args.metrics_out) if args.metrics_out
            else out_path.with_suffix(".metrics.json")
        )
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"metrics → {metrics_path}")
        return

    # ── Single threshold mode ────────────────────────────────────────
    preds = run_inference(
        model, args.coco, args.images_dir, device,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_mask_area=args.min_mask_area,
    )
    out_path.write_text(json.dumps(preds))
    print(f"predictions → {out_path}  ({len(preds)} instances)")

    metrics = compute_all_metrics(args.coco, preds)
    print_summary(metrics, header=f"thr={args.score_threshold:.2f}")

    metrics_path = (
        Path(args.metrics_out) if args.metrics_out
        else out_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics → {metrics_path}")


if __name__ == "__main__":
    main()
