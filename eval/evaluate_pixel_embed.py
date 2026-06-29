"""
Evaluate a trained pixel-embedding checkpoint.

Mirrors `evaluate_maskrcnn.py` so the final headline tables can be
filled in the same way. Inference works in four steps per image:

  1. Forward pass → semantic logits + 1-D tag map (full image
     resolution).
  2. Determine the foreground pixels — by default, use the provided
     plant mask (`crops_full/masks/<crop>.png`) as the prior, optionally
     intersected with the model's own semantic prediction. Supervisor
     approved this in meeting 2 ("OK to assume background is segmented").
  3. Cluster the foreground tags with mean-shift (sklearn). Each
     resulting cluster is one leaf instance.
  4. Encode each cluster mask as a COCO prediction and feed into the
     same `compute_segmentation_ap` / `compute_counting_metrics` /
     `compute_per_stage_metrics` we use for every other method.

Two-step protocol — sweep on val, evaluate on test
--------------------------------------------------
    # 1. Sweep bandwidth on val to pick the operating point
    python -m final_project.eval.evaluate_pixel_embed --sweep \\
        --checkpoint checkpoints/pixel_embed_best.pth \\
        --coco annotations/instances_validation.json \\
        --out results/pixel_embed_val.json

    # 2. Eval on test with the chosen bandwidth
    python -m final_project.eval.evaluate_pixel_embed \\
        --checkpoint checkpoints/pixel_embed_best.pth \\
        --coco annotations/instances_test_set.json \\
        --bandwidth 0.5 \\
        --out results/pixel_embed_test.json
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
from sklearn.cluster import MeanShift

from final_project.eval.metrics import (
    compute_counting_metrics,
    compute_per_stage_metrics,
    compute_segmentation_ap,
    format_per_stage_table,
)
from final_project.models.pixel_embed import build_pixel_embed_model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ════════════════════════════════════════════════════════════════════
# Checkpoint loading
# ════════════════════════════════════════════════════════════════════
def load_trained_model(
    checkpoint_path: str | Path,
    device: torch.device,
    trainable_backbone_layers: int = 3,
    head_channels: int = 256,
    embedding_dim: int = 16,
) -> torch.nn.Module:
    """Build the pixel-embed model and load weights from a checkpoint."""
    model = build_pixel_embed_model(
        pretrained=False,
        trainable_backbone_layers=trainable_backbone_layers,
        head_channels=head_channels,
        embedding_dim=embedding_dim,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
        meta = {k: state.get(k) for k in ("epoch", "best_val_loss")}
    else:
        model.load_state_dict(state)
        meta = {}
    model.to(device).eval()
    return model, meta


# ════════════════════════════════════════════════════════════════════
# Foreground + clustering
# ════════════════════════════════════════════════════════════════════
def load_plant_mask(
    file_name: str,
    plant_mask_dir: Path,
    target_shape: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Load the plant mask for a crop. Returns binary (H, W) or None
    if the file does not exist."""
    p = plant_mask_dir / Path(file_name).name
    if not p.exists():
        return None
    arr = np.array(Image.open(p))
    fg = (arr > 0).astype(np.uint8)
    if target_shape is not None and fg.shape != target_shape:
        return None  # refuse to misalign
    return fg


def cluster_tags_kd(
    tag_map: np.ndarray,
    foreground: np.ndarray,
    bandwidth: float,
    min_pixels: int = 100,
    max_fit_points: int = 30000,
    rng: np.random.RandomState | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Cluster foreground tag vectors with mean-shift; return per-cluster masks.

    Accepts either a 2-D tag map (H, W)  — legacy 1-D embedding —
    or a 3-D tag map (D, H, W) — multi-dim embedding (v3).

    Returns:
        masks: list of (H, W) uint8 binary masks, one per cluster.
        labels: full-resolution per-foreground-pixel label array.
    """
    if tag_map.ndim == 2:
        H, W = tag_map.shape
        D = 1
    elif tag_map.ndim == 3:
        D, H, W = tag_map.shape
    else:
        raise ValueError(f"tag_map must be (H,W) or (D,H,W); got {tag_map.shape}")

    ys, xs = np.where(foreground > 0)
    if ys.size == 0:
        return [], None

    if tag_map.ndim == 2:
        fg_tags = tag_map[ys, xs].reshape(-1, 1).astype(np.float32)
    else:
        fg_tags = tag_map[:, ys, xs].T.astype(np.float32)        # (N_fg, D)

    # Subsample for speed during the cluster-fit step.
    if fg_tags.shape[0] > max_fit_points:
        rng = rng or np.random.RandomState(0)
        sel = rng.choice(fg_tags.shape[0], max_fit_points, replace=False)
        fit_tags = fg_tags[sel]
    else:
        fit_tags = fg_tags

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    try:
        ms.fit(fit_tags)
    except ValueError:
        # bandwidth too small for the data — bail out cleanly
        return [], None
    labels_all = ms.predict(fg_tags)

    masks: list[np.ndarray] = []
    for cid in np.unique(labels_all):
        idx = np.where(labels_all == cid)[0]
        if idx.size < min_pixels:
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        m[ys[idx], xs[idx]] = 1
        masks.append(m)
    return masks, labels_all


def predictions_for_image(
    image_id: int,
    cluster_masks: list[np.ndarray],
    semantic_prob: np.ndarray,
) -> list[dict]:
    """Encode each cluster mask as a COCO-format prediction. The 'score'
    field is the mean semantic-foreground probability over the cluster,
    which serves as a per-instance confidence proxy."""
    preds: list[dict] = []
    for m in cluster_masks:
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bbox = [float(x0), float(y0),
                float(x1 - x0 + 1), float(y1 - y0 + 1)]
        rle = mask_utils.encode(np.asfortranarray(m))
        rle["counts"] = rle["counts"].decode("utf-8")
        score = float(semantic_prob[m > 0].mean())
        preds.append({
            "image_id": image_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": bbox,
            "area": int(m.sum()),
            "score": score,
        })
    return preds


# ════════════════════════════════════════════════════════════════════
# Inference loop
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    coco_path: str | Path,
    images_dir: str | Path,
    plant_mask_dir: str | Path,
    device: torch.device,
    bandwidth: float,
    semantic_threshold: float = 0.5,
    use_semantic_for_fg: bool = True,
    min_pixels: int = 100,
    max_fit_points: int = 30000,
) -> list[dict]:
    """Forward + cluster + format predictions for an entire dataset."""
    coco = COCO(str(coco_path))
    images_dir = Path(images_dir)
    plant_mask_dir = Path(plant_mask_dir)
    rng = np.random.RandomState(0)

    all_preds: list[dict] = []
    for img_id in coco.imgs:
        info = coco.imgs[img_id]
        file_name = info["file_name"]

        img_pil = Image.open(images_dir / file_name).convert("RGB")
        arr = np.array(img_pil)
        H, W = arr.shape[:2]
        tensor = (
            torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0).to(device)
        )

        outputs = model([tensor])[0]
        semantic_logits = outputs["semantic"].cpu().numpy()
        tag_map = outputs["embedding"].cpu().numpy()
        semantic_prob = 1.0 / (1.0 + np.exp(-semantic_logits))

        # Build foreground mask
        plant_fg = load_plant_mask(file_name, plant_mask_dir, target_shape=(H, W))
        if plant_fg is None:
            # CVPPP image (no plant mask) — fall back to semantic only
            foreground = (semantic_prob > semantic_threshold).astype(np.uint8)
        elif use_semantic_for_fg:
            foreground = (plant_fg.astype(bool)
                          & (semantic_prob > semantic_threshold)).astype(np.uint8)
        else:
            foreground = plant_fg

        cluster_masks, _ = cluster_tags_kd(
            tag_map, foreground,
            bandwidth=bandwidth,
            min_pixels=min_pixels,
            max_fit_points=max_fit_points,
            rng=rng,
        )
        all_preds.extend(predictions_for_image(
            img_id, cluster_masks, semantic_prob,
        ))

    return all_preds


# ════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ════════════════════════════════════════════════════════════════════
def compute_all_metrics(
    coco_path: str | Path, predictions: list[dict]
) -> dict:
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
        f"  AP50={metrics['AP50']:.4f}  AP={metrics['AP']:.4f}  "
        f"AP75={metrics['AP75']:.4f}  MAE={metrics['MAE']:.2f}  "
        f"RMSE={metrics['RMSE']:.2f}  DiC={metrics['DiC_mean']:+.2f}  "
        f"pred={metrics['pred_total']}"
    )
    if metrics["per_stage"]:
        print()
        print(format_per_stage_table(metrics["per_stage"]))


# ════════════════════════════════════════════════════════════════════
# Sweep
# ════════════════════════════════════════════════════════════════════
def sweep_bandwidths(
    model: torch.nn.Module,
    coco_path: str | Path,
    images_dir: str | Path,
    plant_mask_dir: str | Path,
    device: torch.device,
    bandwidths: Iterable[float],
    semantic_threshold: float,
    use_semantic_for_fg: bool,
    min_pixels: int,
) -> tuple[list[dict], dict]:
    """Run inference for each bandwidth; pick the one with the best AP50."""
    rows: list[dict] = []
    for bw in bandwidths:
        print(f"  bandwidth = {bw:.3f} ...")
        preds = run_inference(
            model, coco_path, images_dir, plant_mask_dir, device,
            bandwidth=bw,
            semantic_threshold=semantic_threshold,
            use_semantic_for_fg=use_semantic_for_fg,
            min_pixels=min_pixels,
        )
        if not preds:
            rows.append({"bandwidth": bw, "AP50": 0.0, "AP": 0.0,
                         "MAE": 0.0, "pred_total": 0})
            continue
        ap = compute_segmentation_ap(coco_path, preds)
        ct = compute_counting_metrics(coco_path, preds)
        rows.append({
            "bandwidth": bw,
            "AP50": float(ap["AP50"]),
            "AP": float(ap["AP"]),
            "MAE": float(ct["MAE"]),
            "RMSE": float(ct["RMSE"]),
            "DiC_mean": float(ct["DiC_mean"]),
            "pred_total": int(ct["pred_total"]),
        })
        print(f"    AP50={ap['AP50']:.4f}  MAE={ct['MAE']:.2f}  "
              f"pred={ct['pred_total']}")
    best = max(rows, key=lambda r: r["AP50"])
    print(f"\nbest bandwidth: {best['bandwidth']:.3f}  "
          f"AP50={best['AP50']:.4f}  MAE={best['MAE']:.2f}")
    return rows, best


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate a trained pixel-embedding checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--plant-mask-dir", default="crops_full/masks")

    # Checkpoint
    ap.add_argument("--checkpoint", default="checkpoints/pixel_embed_best.pth")
    ap.add_argument("--trainable-backbone-layers", type=int, default=3)
    ap.add_argument("--head-channels", type=int, default=256)
    ap.add_argument("--embedding-dim", type=int, default=16,
                    help="must match the dim the checkpoint was trained with")

    # Clustering
    ap.add_argument("--bandwidth", type=float, default=0.5,
                    help="mean-shift bandwidth for 1-D tag clustering")
    ap.add_argument("--semantic-threshold", type=float, default=0.5)
    ap.add_argument("--use-semantic-for-fg", action="store_true", default=True,
                    help="intersect plant mask with semantic prediction")
    ap.add_argument("--no-semantic-for-fg", dest="use_semantic_for_fg",
                    action="store_false")
    ap.add_argument("--min-pixels", type=int, default=100,
                    help="drop clusters smaller than this")

    # Sweep
    ap.add_argument("--sweep", action="store_true",
                    help="sweep bandwidth and pick the best by AP50")
    ap.add_argument("--sweep-bandwidths",
                    default="0.3,0.5,0.75,1.0,1.5,2.0,3.0,5.0",
                    help="comma-separated bandwidths for --sweep. "
                         "v3 default range widened for D-dim embedding space.")

    # Outputs
    ap.add_argument("--out", default="results/pixel_embed_val.json")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--sweep-csv", default=None)

    # Device
    ap.add_argument("--device", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")

    model, meta = load_trained_model(
        args.checkpoint, device,
        trainable_backbone_layers=args.trainable_backbone_layers,
        head_channels=args.head_channels,
        embedding_dim=args.embedding_dim,
    )
    if meta.get("epoch") is not None:
        print(f"  loaded from epoch {meta['epoch'] + 1}  "
              f"(best val_loss at train time: "
              f"{meta.get('best_val_loss', float('inf')):.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Sweep mode ───────────────────────────────────────────────────
    if args.sweep:
        bandwidths = [float(x) for x in args.sweep_bandwidths.split(",")]
        rows, best = sweep_bandwidths(
            model, args.coco, args.images_dir, args.plant_mask_dir, device,
            bandwidths=bandwidths,
            semantic_threshold=args.semantic_threshold,
            use_semantic_for_fg=args.use_semantic_for_fg,
            min_pixels=args.min_pixels,
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

        # Re-run final predictions at the best bandwidth
        best_preds = run_inference(
            model, args.coco, args.images_dir, args.plant_mask_dir, device,
            bandwidth=best["bandwidth"],
            semantic_threshold=args.semantic_threshold,
            use_semantic_for_fg=args.use_semantic_for_fg,
            min_pixels=args.min_pixels,
        )
        out_path.write_text(json.dumps(best_preds))
        print(f"predictions at best bandwidth → {out_path}  "
              f"({len(best_preds)} instances)")

        metrics = compute_all_metrics(args.coco, best_preds)
        print_summary(metrics,
                      header=f"best (bandwidth={best['bandwidth']:.3f})")
        metrics_path = (
            Path(args.metrics_out) if args.metrics_out
            else out_path.with_suffix(".metrics.json")
        )
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"metrics → {metrics_path}")
        return

    # ── Single bandwidth mode ────────────────────────────────────────
    preds = run_inference(
        model, args.coco, args.images_dir, args.plant_mask_dir, device,
        bandwidth=args.bandwidth,
        semantic_threshold=args.semantic_threshold,
        use_semantic_for_fg=args.use_semantic_for_fg,
        min_pixels=args.min_pixels,
    )
    out_path.write_text(json.dumps(preds))
    print(f"predictions → {out_path}  ({len(preds)} instances)")

    metrics = compute_all_metrics(args.coco, preds)
    print_summary(metrics, header=f"bandwidth={args.bandwidth:.3f}")

    metrics_path = (
        Path(args.metrics_out) if args.metrics_out
        else out_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"metrics → {metrics_path}")


if __name__ == "__main__":
    main()
