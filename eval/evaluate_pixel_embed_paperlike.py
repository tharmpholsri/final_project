"""Evaluate a paper-like 1-D associative-embedding model.

This decoder follows Newell, Huang, and Deng (2016), section 3.4:
threshold the foreground heatmap, build a histogram of scalar tags, find
identifier peaks with 1-D non-maximum suppression, and assign every
foreground pixel to its nearest identifier.  A 1-D MeanShift decoder is
also available as a controlled post-processing comparison.

Train a compatible checkpoint (pretrained backbone, scalar tag, raw MSE
foreground heatmap) with::

    python -m final_project.train.train_pixel_embed \
        --embedding-dim 1 --detection-loss mse \
        --ckpt-name pixel_embed_paperlike

Evaluate on validation before using the selected settings on test::

    python -m final_project.eval.evaluate_pixel_embed_paperlike \
        --checkpoint checkpoints/pixel_embed_paperlike_best.pth \
        --coco annotations/instances_validation.json \
        --decoder hist_nms \
        --out results/pixel_embed_paperlike_val.json

Tune histogram resolution and NMS prominence on validation only::

    python -m final_project.eval.evaluate_pixel_embed_paperlike \
        --checkpoint checkpoints/pixel_embed_paperlike_best.pth \
        --decoder hist_nms --sweep \
        --sweep-bins 128,256,512 \
        --sweep-prominence 0.01,0.02,0.05,0.10
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from final_project.eval.evaluate_pixel_embed import (
    cluster_tags_kd,
    compute_all_metrics,
    load_plant_mask,
    load_trained_model,
    pick_device,
    predictions_for_image,
    print_summary,
)


def decode_histogram_nms(
    tag_map: np.ndarray,
    foreground: np.ndarray,
    bins: int = 256,
    smooth_sigma: float = 2.0,
    peak_prominence: float = 0.02,
    peak_distance: int = 5,
    clip_percentile: float = 0.5,
    min_pixels: int = 100,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Decode scalar tags using histogram peaks and nearest-tag assignment.

    ``peak_prominence`` is a fraction of the largest smoothed histogram
    count. ``clip_percentile`` clips symmetric tag outliers only when
    estimating the histogram range; every foreground pixel is still assigned.
    """
    if tag_map.ndim == 3 and tag_map.shape[0] == 1:
        tag_map = tag_map[0]
    if tag_map.ndim != 2:
        raise ValueError(
            f"paper-like decoding requires one tag per pixel; got {tag_map.shape}"
        )
    if bins < 3:
        raise ValueError("bins must be at least 3")
    if not 0.0 <= clip_percentile < 50.0:
        raise ValueError("clip_percentile must be in [0, 50)")

    height, width = tag_map.shape
    ys, xs = np.where(foreground > 0)
    labels_image = np.full((height, width), -1, dtype=np.int32)
    if ys.size == 0:
        return [], labels_image

    tags = tag_map[ys, xs].astype(np.float64)
    finite = np.isfinite(tags)
    if not finite.all():
        ys, xs, tags = ys[finite], xs[finite], tags[finite]
    if tags.size == 0:
        return [], labels_image

    low = float(np.percentile(tags, clip_percentile))
    high = float(np.percentile(tags, 100.0 - clip_percentile))
    if not np.isfinite(low + high) or high - low < 1e-8:
        identifiers = np.array([float(np.mean(tags))], dtype=np.float64)
    else:
        hist, edges = np.histogram(tags, bins=bins, range=(low, high))
        smooth = gaussian_filter1d(
            hist.astype(np.float64), sigma=max(smooth_sigma, 0.0)
        )
        prominence = max(float(smooth.max()) * peak_prominence, 0.0)

        # Padding permits a mode at either edge of the clipped range to be
        # selected; scipy.find_peaks otherwise excludes array endpoints.
        padded = np.pad(smooth, (1, 1), mode="constant")
        peaks, _ = find_peaks(
            padded,
            prominence=prominence,
            distance=max(int(peak_distance), 1),
        )
        peaks = peaks - 1
        peaks = peaks[(peaks >= 0) & (peaks < bins)]
        if peaks.size == 0:
            peaks = np.array([int(np.argmax(smooth))])
        centers = 0.5 * (edges[:-1] + edges[1:])
        identifiers = centers[peaks]

    labels = np.abs(tags[:, None] - identifiers[None, :]).argmin(axis=1)
    masks: list[np.ndarray] = []
    kept_label = 0
    for cluster_id in range(len(identifiers)):
        member = labels == cluster_id
        if int(member.sum()) < min_pixels:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ys[member], xs[member]] = 1
        masks.append(mask)
        labels_image[ys[member], xs[member]] = kept_label
        kept_label += 1
    return masks, labels_image


def foreground_mask(
    source: str,
    semantic_score: np.ndarray,
    threshold: float,
    plant_mask: np.ndarray | None,
) -> np.ndarray:
    semantic = semantic_score > threshold
    if source == "semantic":
        return semantic.astype(np.uint8)
    if plant_mask is None:
        return semantic.astype(np.uint8)
    if source == "plant":
        return plant_mask.astype(np.uint8)
    return (plant_mask.astype(bool) & semantic).astype(np.uint8)


@torch.no_grad()
def run_inference(args: argparse.Namespace, model: torch.nn.Module, device) -> list[dict]:
    coco = COCO(str(args.coco))
    images_dir = Path(args.images_dir)
    plant_mask_dir = Path(args.plant_mask_dir)
    rng = np.random.RandomState(0)
    predictions: list[dict] = []

    for image_id in coco.imgs:
        info = coco.imgs[image_id]
        file_name = info["file_name"]
        image = np.array(Image.open(images_dir / file_name).convert("RGB"))
        height, width = image.shape[:2]
        tensor = (
            torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).to(device)
        )
        output = model([tensor])[0]
        raw_detection = output["semantic"].cpu().numpy()
        tag_map = output["embedding"].cpu().numpy()
        if tag_map.shape[0] != 1:
            raise ValueError(
                f"checkpoint has {tag_map.shape[0]} tag channels; expected 1"
            )

        if args.detection_activation == "sigmoid":
            detection_score = 1.0 / (1.0 + np.exp(-raw_detection))
        else:
            detection_score = raw_detection
        plant = load_plant_mask(file_name, plant_mask_dir, (height, width))
        foreground = foreground_mask(
            args.foreground_source,
            detection_score,
            args.detection_threshold,
            plant,
        )

        if args.decoder == "hist_nms":
            masks, _ = decode_histogram_nms(
                tag_map,
                foreground,
                bins=args.hist_bins,
                smooth_sigma=args.hist_smooth_sigma,
                peak_prominence=args.peak_prominence,
                peak_distance=args.peak_distance,
                clip_percentile=args.tag_clip_percentile,
                min_pixels=args.min_pixels,
            )
        else:
            masks, _ = cluster_tags_kd(
                tag_map[0],
                foreground,
                bandwidth=args.bandwidth,
                min_pixels=args.min_pixels,
                max_fit_points=args.max_fit_points,
                rng=rng,
            )

        # COCO evaluation uses scores for ranking and does not require them
        # to lie in [0, 1]. Preserve raw MSE scores rather than clipping or
        # normalising per image, both of which can damage cross-image ranking.
        confidence = detection_score
        predictions.extend(
            predictions_for_image(image_id, masks, confidence)
        )
    return predictions


def parse_csv_values(value: str, cast, name: str) -> list:
    """Parse and validate a non-empty comma-separated CLI grid."""
    try:
        values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def run_histogram_sweep(
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[list[dict], dict, list[dict]]:
    """Tune histogram bins and NMS prominence on validation AP50."""
    bins_grid = parse_csv_values(args.sweep_bins, int, "--sweep-bins")
    prominence_grid = parse_csv_values(
        args.sweep_prominence, float, "--sweep-prominence"
    )
    if any(value < 3 for value in bins_grid):
        raise ValueError("all --sweep-bins values must be at least 3")
    if any(value < 0.0 for value in prominence_grid):
        raise ValueError("all --sweep-prominence values must be non-negative")

    original_bins = args.hist_bins
    original_prominence = args.peak_prominence
    rows: list[dict] = []
    best_predictions: list[dict] = []
    best_row: dict | None = None

    try:
        for bins, prominence in product(bins_grid, prominence_grid):
            args.hist_bins = bins
            args.peak_prominence = prominence
            print(f"  bins={bins}  prominence={prominence:g} ...")
            predictions = run_inference(args, model, device)
            metrics = compute_all_metrics(args.coco, predictions)
            row = {
                "hist_bins": bins,
                "peak_prominence": prominence,
                "AP50": metrics["AP50"],
                "AP": metrics["AP"],
                "AP75": metrics["AP75"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "DiC_mean": metrics["DiC_mean"],
                "pred_total": metrics["pred_total"],
            }
            rows.append(row)
            print(
                f"    AP50={row['AP50']:.4f}  MAE={row['MAE']:.2f}  "
                f"pred={row['pred_total']}"
            )
            if best_row is None or row["AP50"] > best_row["AP50"]:
                best_row = row
                best_predictions = predictions
    finally:
        args.hist_bins = original_bins
        args.peak_prominence = original_prominence

    assert best_row is not None
    return rows, best_row, best_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paper-like scalar associative-embedding evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco", default="annotations/instances_validation.json")
    parser.add_argument("--images-dir", default="crops_full/images")
    parser.add_argument("--plant-mask-dir", default="crops_full/masks")
    parser.add_argument("--out", default="results/pixel_embed_paperlike_val.json")
    parser.add_argument("--metrics-out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--architecture",
        choices=["p2", "decoder"],
        default="p2",
        help="must match the architecture used during training",
    )
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=64)

    parser.add_argument("--decoder", choices=["hist_nms", "meanshift"],
                        default="hist_nms")
    parser.add_argument("--foreground-source",
                        choices=["semantic", "plant", "intersection"],
                        default="semantic")
    parser.add_argument("--detection-activation", choices=["raw", "sigmoid"],
                        default="raw",
                        help="raw matches an MSE-trained paper-like heatmap")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=100)

    parser.add_argument("--hist-bins", type=int, default=256)
    parser.add_argument("--hist-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--peak-prominence", type=float, default=0.02,
                        help="fraction of maximum smoothed histogram count")
    parser.add_argument("--peak-distance", type=int, default=5,
                        help="minimum separation between histogram peaks in bins")
    parser.add_argument("--tag-clip-percentile", type=float, default=0.5)

    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--max-fit-points", type=int, default=30000)

    parser.add_argument("--sweep", action="store_true",
                        help="tune histogram bins and peak prominence by AP50")
    parser.add_argument("--sweep-bins", default="128,256,512")
    parser.add_argument("--sweep-prominence", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--sweep-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else pick_device()
    model, meta = load_trained_model(
        args.checkpoint,
        device,
        trainable_backbone_layers=args.trainable_backbone_layers,
        head_channels=args.head_channels,
        decoder_channels=args.decoder_channels,
        embedding_dim=1,
        architecture=args.architecture,
    )
    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"decoder: {args.decoder}")
    if args.detection_activation == "raw" and args.detection_threshold == 0.5:
        print(
            "NOTE: raw MSE heatmap threshold 0.5 is the natural midpoint "
            "for targets 0/1, but it should still be validated on val data."
        )
    if meta.get("epoch") is not None:
        print(f"checkpoint epoch: {meta['epoch'] + 1}")

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.sweep:
        if args.decoder != "hist_nms":
            raise ValueError("--sweep currently applies only to --decoder hist_nms")
        rows, best, predictions = run_histogram_sweep(args, model, device)
        sweep_path = (
            Path(args.sweep_csv)
            if args.sweep_csv
            else output_path.with_name(output_path.stem + "_sweep.csv")
        )
        sweep_path.parent.mkdir(parents=True, exist_ok=True)
        with sweep_path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"best: bins={best['hist_bins']}  "
            f"prominence={best['peak_prominence']:g}  AP50={best['AP50']:.4f}"
        )
        print(f"sweep -> {sweep_path}")
    else:
        predictions = run_inference(args, model, device)

    output_path.write_text(json.dumps(predictions))

    metrics = compute_all_metrics(args.coco, predictions)
    print_summary(metrics, header=f"paper-like / {args.decoder}")
    metrics_path = (
        Path(args.metrics_out)
        if args.metrics_out
        else output_path.with_suffix(".metrics.json")
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"predictions -> {output_path}")
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
