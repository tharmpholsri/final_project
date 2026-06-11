"""
Evaluation metrics for leaf instance segmentation.

Wraps `pycocotools.cocoeval` to compute:
    - AP @ IoU=0.5         (AP50) — primary segmentation metric
    - AP @ IoU=[0.5:0.95]  (mAP)  — COCO-style mean AP
    - per-image mean IoU
    - MAE / RMSE on leaf counts

Designed to be reused by every method (traditional, Mask R-CNN, YOLO),
so reported numbers are directly comparable.
"""

from __future__ import annotations

import io
import json
import re
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =====================================================================
# Core metric: AP via pycocotools
# =====================================================================


def compute_segmentation_ap(
    coco_gt_path: str | Path,
    predictions: list[dict] | str | Path,
    verbose: bool = False,
) -> dict:
    """
    Compute COCO-style segmentation AP.

    Args:
        coco_gt_path: path to ground-truth COCO JSON.
        predictions: list of COCO result dicts (each with image_id,
            category_id, segmentation, score), OR a path to a JSON
            file containing such a list.
        verbose: if True, print pycocotools' default summary.

    Returns:
        dict with keys AP, AP50, AP75, AP_small, AP_medium, AP_large,
        AR_max1, AR_max10, AR_max100, AR_small, AR_medium, AR_large.
    """
    coco_gt = COCO(str(coco_gt_path))

    if isinstance(predictions, (str, Path)):
        with open(predictions) as f:
            predictions = json.load(f)
    if not predictions:
        return {k: 0.0 for k in _AP_KEYS}

    # pycocotools' loadRes mutates internal state; we want the standard path.
    with redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(predictions)

    cocoEval = COCOeval(coco_gt, coco_dt, iouType="segm")
    target_stream = None if verbose else io.StringIO()
    if verbose:
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    else:
        with redirect_stdout(target_stream):
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

    return {k: float(v) for k, v in zip(_AP_KEYS, cocoEval.stats)}


_AP_KEYS = [
    "AP",        # mean over IoU 0.5–0.95
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_max1",
    "AR_max10",
    "AR_max100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


# =====================================================================
# Counting metrics
# =====================================================================


def compute_counting_metrics(
    coco_gt_path: str | Path,
    predictions: list[dict] | str | Path,
    score_threshold: float = 0.0,
) -> dict:
    """
    MAE / RMSE between predicted leaf count and ground-truth leaf count.

    Args:
        score_threshold: only count predictions with score >= this.
            Set 0 for traditional (no real confidence), tune for ML.

    Returns:
        dict with MAE, RMSE, DiC_mean (mean signed error), and
        per-image (gt, pred) pairs.
    """
    coco_gt = COCO(str(coco_gt_path))
    if isinstance(predictions, (str, Path)):
        with open(predictions) as f:
            predictions = json.load(f)

    gt_counts: dict[int, int] = {iid: 0 for iid in coco_gt.imgs}
    for ann in coco_gt.dataset["annotations"]:
        gt_counts[ann["image_id"]] += 1

    pred_counts: dict[int, int] = {iid: 0 for iid in coco_gt.imgs}
    for pred in predictions:
        if pred.get("score", 1.0) < score_threshold:
            continue
        pred_counts[pred["image_id"]] = pred_counts.get(pred["image_id"], 0) + 1

    gts = np.array([gt_counts[iid] for iid in sorted(coco_gt.imgs)])
    preds = np.array([pred_counts.get(iid, 0) for iid in sorted(coco_gt.imgs)])
    errors = preds - gts

    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "DiC_mean": float(np.mean(errors)),  # signed bias
        "abs_DiC": float(np.mean(np.abs(errors))),
        "n_images": int(len(gts)),
        "gt_total": int(gts.sum()),
        "pred_total": int(preds.sum()),
    }


# =====================================================================
# Per-stage breakdown (proposal's main contribution)
# =====================================================================


def stage_from_filename(filename: str) -> str | None:
    """Extract growth-stage bucket from PACE filename PS_Tray_<TRAY>_<T>_p<P>.png."""
    m = re.match(r"PS_Tray_\d+_(\d+)_p\d+\.png", filename)
    if not m:
        return None
    t = int(m.group(1))
    if t <= 8:
        return "early"     # t=5
    if t <= 14:
        return "mid"       # t=10, 12
    if t <= 18:
        return "late"      # t=15, 17
    return "canopy"        # t=19, 20


def compute_per_stage_metrics(
    coco_gt_path: str | Path,
    predictions: list[dict] | str | Path,
    score_threshold: float = 0.0,
) -> dict[str, dict]:
    """
    Break AP50 and MAE down by growth stage (early / mid / late / canopy).

    This is the central analysis the proposal commits to.
    """
    coco_gt = COCO(str(coco_gt_path))
    if isinstance(predictions, (str, Path)):
        with open(predictions) as f:
            predictions = json.load(f)

    # Map image_id → stage
    img_id_to_stage: dict[int, str] = {}
    stage_to_img_ids: dict[str, list[int]] = defaultdict(list)
    for img in coco_gt.dataset["images"]:
        stage = stage_from_filename(img["file_name"])
        if stage is None:
            continue
        img_id_to_stage[img["id"]] = stage
        stage_to_img_ids[stage].append(img["id"])

    out: dict[str, dict] = {}
    for stage, img_ids in stage_to_img_ids.items():
        # Subset ground-truth annotations
        sub_gt = {
            "info": coco_gt.dataset.get("info", {}),
            "licenses": coco_gt.dataset.get("licenses", []),
            "categories": coco_gt.dataset["categories"],
            "images": [im for im in coco_gt.dataset["images"] if im["id"] in img_ids],
            "annotations": [a for a in coco_gt.dataset["annotations"] if a["image_id"] in img_ids],
        }
        sub_preds = [p for p in predictions if p["image_id"] in img_ids]

        if not sub_gt["images"]:
            out[stage] = {"AP50": 0.0, "MAE": 0.0, "n_images": 0}
            continue

        # Use a temp file because COCO() loads from path
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(sub_gt, f)
            tmp_path = f.name
        try:
            ap = compute_segmentation_ap(tmp_path, sub_preds)
            cnt = compute_counting_metrics(tmp_path, sub_preds, score_threshold)
            out[stage] = {
                "AP": ap["AP"],
                "AP50": ap["AP50"],
                "MAE": cnt["MAE"],
                "RMSE": cnt["RMSE"],
                "DiC_mean": cnt["DiC_mean"],
                "n_images": cnt["n_images"],
                "gt_total": cnt["gt_total"],
                "pred_total": cnt["pred_total"],
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return out


# =====================================================================
# Pretty printing
# =====================================================================


STAGE_ORDER = ["early", "mid", "late", "canopy"]


def format_per_stage_table(per_stage: dict[str, dict]) -> str:
    rows = [
        f"{'stage':<8} {'n_img':>6} {'gt':>5} {'pred':>5} "
        f"{'AP50':>6} {'AP':>6} {'MAE':>6} {'RMSE':>6} {'DiC':>6}"
    ]
    rows.append("-" * len(rows[0]))
    for stage in STAGE_ORDER:
        if stage not in per_stage:
            continue
        m = per_stage[stage]
        rows.append(
            f"{stage:<8} {m['n_images']:>6} {m['gt_total']:>5} {m['pred_total']:>5} "
            f"{m['AP50']:>6.3f} {m['AP']:>6.3f} "
            f"{m['MAE']:>6.2f} {m['RMSE']:>6.2f} {m['DiC_mean']:>+6.2f}"
        )
    return "\n".join(rows)
