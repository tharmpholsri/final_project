"""Visualise paper-like 1-D tag heatmaps and Histogram/NMS peaks.

This script is for failure analysis of the paper-like associative
embedding models. It renders one figure per image:

  image | GT foreground | predicted foreground | scalar tag heatmap |
  foreground-tag histogram with detected peaks | final Histogram/NMS masks

This helps answer whether failures come from:
  - foreground pixels being missing, or
  - tag values for different leaves not forming separable histogram peaks.
"""

from __future__ import annotations

import argparse
import colorsys
import html
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from final_project.eval.evaluate_pixel_embed import (
    load_plant_mask,
    load_trained_model,
    pick_device,
)
from final_project.eval.evaluate_pixel_embed_paperlike import (
    decode_histogram_nms,
    foreground_mask,
)


def distinct_colors(n: int) -> np.ndarray:
    hues = (np.arange(n) * 0.61803398875) % 1.0
    return np.array([colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues])


def stage_of(filename: str) -> str | None:
    m = re.match(r"PS_Tray_\d+_(\d+)_p\d+\.png", filename)
    if not m:
        return None
    t = int(m.group(1))
    if t <= 8:
        return "early"
    if t <= 14:
        return "mid"
    if t <= 18:
        return "late"
    return "canopy"


def overlay_binary(img: np.ndarray, mask: np.ndarray, color, alpha=0.45) -> np.ndarray:
    out = img.astype(np.float32) / 255.0
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.array(color)
    return np.clip(out, 0, 1)


def overlay_instances(img: np.ndarray, masks: list[np.ndarray], alpha=0.5) -> np.ndarray:
    out = img.astype(np.float32) / 255.0
    colors = distinct_colors(len(masks))
    for mask, color in zip(masks, colors):
        m = mask.astype(bool)
        out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 1)


def gt_foreground(coco: COCO, img_id: int, shape: tuple[int, int]) -> np.ndarray:
    H, W = shape
    fg = np.zeros((H, W), dtype=np.uint8)
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=img_id)):
        fg |= coco.annToMask(ann).astype(np.uint8)
    return fg


def histogram_debug(
    tag_map: np.ndarray,
    foreground: np.ndarray,
    bins: int,
    smooth_sigma: float,
    peak_prominence: float,
    peak_distance: int,
    clip_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if tag_map.ndim == 3 and tag_map.shape[0] == 1:
        tag_map = tag_map[0]
    tags = tag_map[foreground > 0].astype(np.float64)
    tags = tags[np.isfinite(tags)]
    if tags.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    low = float(np.percentile(tags, clip_percentile))
    high = float(np.percentile(tags, 100.0 - clip_percentile))
    if not np.isfinite(low + high) or high - low < 1e-8:
        center = np.array([float(np.mean(tags))])
        return np.array([tags.size]), center, np.array([tags.size]), np.array([0])

    hist, edges = np.histogram(tags, bins=bins, range=(low, high))
    smooth = gaussian_filter1d(hist.astype(np.float64), sigma=max(smooth_sigma, 0.0))
    prominence = max(float(smooth.max()) * peak_prominence, 0.0)
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
    return hist, centers, smooth, peaks


@torch.no_grad()
def render_one(
    model: torch.nn.Module,
    coco: COCO,
    img_id: int,
    images_dir: Path,
    plant_mask_dir: Path,
    out_path: Path,
    device: torch.device,
    architecture: str,
    foreground_source: str,
    detection_activation: str,
    detection_threshold: float,
    hist_bins: int,
    hist_smooth_sigma: float,
    peak_prominence: float,
    peak_distance: int,
    tag_clip_percentile: float,
    min_pixels: int,
) -> tuple[int, int, int]:
    info = coco.imgs[img_id]
    file_name = info["file_name"]
    img = np.array(Image.open(images_dir / file_name).convert("RGB"))
    H, W = img.shape[:2]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(device)

    output = model([tensor])[0]
    raw_detection = output["semantic"].detach().cpu().numpy()
    tag_map = output["embedding"].detach().cpu().numpy()
    if tag_map.shape[0] != 1:
        raise ValueError(f"expected 1-D tags, got embedding shape {tag_map.shape}")
    tag_2d = tag_map[0]

    if detection_activation == "sigmoid":
        detection_score = 1.0 / (1.0 + np.exp(-raw_detection))
    elif detection_activation == "raw":
        detection_score = raw_detection
    else:
        raise ValueError(f"unknown detection activation {detection_activation!r}")

    gt_fg = gt_foreground(coco, img_id, (H, W))
    plant = load_plant_mask(file_name, plant_mask_dir, (H, W))
    foreground = foreground_mask(
        foreground_source,
        detection_score,
        detection_threshold,
        plant,
    )
    masks, _ = decode_histogram_nms(
        tag_map,
        foreground,
        bins=hist_bins,
        smooth_sigma=hist_smooth_sigma,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        clip_percentile=tag_clip_percentile,
        min_pixels=min_pixels,
    )
    hist, centers, smooth, peaks = histogram_debug(
        tag_2d,
        foreground,
        bins=hist_bins,
        smooth_sigma=hist_smooth_sigma,
        peak_prominence=peak_prominence,
        peak_distance=peak_distance,
        clip_percentile=tag_clip_percentile,
    )

    n_gt = len(coco.getAnnIds(imgIds=img_id))
    n_pred = len(masks)
    n_peaks = int(len(peaks))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    axes[0].imshow(img)
    axes[0].set_title(f"image\n{file_name}")

    axes[1].imshow(overlay_binary(img, gt_fg, color=(0.0, 1.0, 0.0)))
    axes[1].set_title(f"GT foreground\nGT leaves: {n_gt}")

    axes[2].imshow(overlay_binary(img, foreground, color=(1.0, 0.0, 0.0)))
    axes[2].set_title(
        f"pred foreground\nsource={foreground_source}, thr={detection_threshold}"
    )

    tag_vis = np.ma.masked_where(foreground == 0, tag_2d)
    im = axes[3].imshow(tag_vis, cmap="coolwarm")
    axes[3].set_title("1-D tag heatmap\nmasked by foreground")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if hist.size:
        axes[4].bar(centers, hist, width=(centers[1] - centers[0]) if len(centers) > 1 else 1)
        axes[4].plot(centers, smooth, color="black", lw=1.5, label="smoothed")
        if peaks.size:
            axes[4].scatter(centers[peaks], smooth[peaks], color="red", zorder=3,
                            label=f"peaks={len(peaks)}")
        axes[4].legend(fontsize=8)
    axes[4].set_title(
        f"foreground tag histogram\nbins={hist_bins}, prom={peak_prominence}"
    )
    axes[4].set_xlabel("tag value")
    axes[4].set_ylabel("foreground pixels")

    axes[5].imshow(overlay_instances(img, masks))
    diff = n_pred - n_gt
    sign = "+" if diff > 0 else ""
    axes[5].set_title(f"Histogram/NMS masks\npred={n_pred} ({sign}{diff} vs GT)")

    for idx, ax in enumerate(axes):
        if idx != 4:
            ax.axis("off")
    fig.suptitle(
        f"{architecture} | {stage_of(file_name) or 'unknown'} | "
        f"hist peaks={n_peaks}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return n_gt, n_pred, n_peaks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise paper-like 1-D tag heatmaps and histogram peaks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco", default="annotations/instances_test_set.json")
    parser.add_argument("--images-dir", default="crops_full/images")
    parser.add_argument("--plant-mask-dir", default="crops_full/masks")
    parser.add_argument("--out", default="results/paperlike_tag_viz")
    parser.add_argument("--architecture",
                        choices=["p2", "decoder", "decoder_h2", "fpn_h2"],
                        default="p2")
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--foreground-source",
                        choices=["semantic", "plant", "intersection"],
                        default="semantic")
    parser.add_argument("--detection-activation", choices=["raw", "sigmoid"],
                        default="raw")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--hist-bins", type=int, default=128)
    parser.add_argument("--hist-smooth-sigma", type=float, default=2.0)
    parser.add_argument("--peak-prominence", type=float, default=0.10)
    parser.add_argument("--peak-distance", type=int, default=5)
    parser.add_argument("--tag-clip-percentile", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--stage", choices=["early", "mid", "late", "canopy"],
                        default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

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
    if meta.get("epoch") is not None:
        print(f"loaded epoch {meta['epoch'] + 1}")

    coco = COCO(args.coco)
    images_dir = Path(args.images_dir)
    plant_mask_dir = Path(args.plant_mask_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_ids = sorted(coco.imgs.keys())
    if args.stage:
        img_ids = [
            img_id for img_id in img_ids
            if stage_of(coco.imgs[img_id]["file_name"]) == args.stage
        ]
    if args.limit:
        img_ids = img_ids[:args.limit]

    rows = []
    for idx, img_id in enumerate(img_ids, start=1):
        file_name = coco.imgs[img_id]["file_name"]
        out_path = out_dir / f"{Path(file_name).stem}_tag_debug.png"
        n_gt, n_pred, n_peaks = render_one(
            model=model,
            coco=coco,
            img_id=img_id,
            images_dir=images_dir,
            plant_mask_dir=plant_mask_dir,
            out_path=out_path,
            device=device,
            architecture=args.architecture,
            foreground_source=args.foreground_source,
            detection_activation=args.detection_activation,
            detection_threshold=args.detection_threshold,
            hist_bins=args.hist_bins,
            hist_smooth_sigma=args.hist_smooth_sigma,
            peak_prominence=args.peak_prominence,
            peak_distance=args.peak_distance,
            tag_clip_percentile=args.tag_clip_percentile,
            min_pixels=args.min_pixels,
        )
        rows.append((out_path.name, file_name, n_gt, n_pred, n_peaks))
        print(f"{idx}/{len(img_ids)} {file_name}: GT={n_gt} pred={n_pred} peaks={n_peaks}")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Paper-like tag visualisation</title>",
        "<style>body{font-family:system-ui;margin:20px;background:#111;color:#eee}"
        "img{max-width:900px;margin:8px;border:1px solid #444;border-radius:4px}"
        "figure{margin:10px 0}figcaption{color:#ccc}</style></head><body>",
        f"<h1>{html.escape(str(args.checkpoint))}</h1>",
        f"<p>foreground={args.foreground_source}, bins={args.hist_bins}, "
        f"prominence={args.peak_prominence}, stage={args.stage or 'all'}</p>",
    ]
    for out_name, file_name, n_gt, n_pred, n_peaks in rows:
        parts.append(
            f"<figure><img src='{html.escape(out_name)}'>"
            f"<figcaption>{html.escape(file_name)} | "
            f"GT={n_gt}, pred={n_pred}, histogram peaks={n_peaks}</figcaption>"
            "</figure>"
        )
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts))
    print(f"wrote {len(rows)} visualisations → {out_dir}")


if __name__ == "__main__":
    main()
