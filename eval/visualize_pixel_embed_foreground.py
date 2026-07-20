"""Visualise foreground predictions from a pixel-embedding checkpoint.

This is intended for failure-case analysis. For each selected image it
renders:

  original image | GT foreground | predicted foreground score | semantic mask |
  plant/intersection foreground | predicted instances

Examples
--------
    python -m final_project.eval.visualize_pixel_embed_foreground \
        --checkpoint checkpoints/pixel_embed_16d_decoder_h2_cp_best.pth \
        --architecture decoder_h2 \
        --embedding-dim 16 \
        --coco annotations/instances_test_set.json \
        --images-dir crops_full/images \
        --plant-mask-dir crops_full/masks \
        --foreground-source intersection \
        --bandwidth 0.5 \
        --stage canopy \
        --limit 8 \
        --out results/pixel_foreground_viz
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

from final_project.eval.evaluate_pixel_embed import (
    cluster_tags_kd,
    load_plant_mask,
    load_trained_model,
    pick_device,
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


def overlay_binary(img: np.ndarray, mask: np.ndarray, color=(1.0, 0.0, 0.0), alpha=0.45):
    out = img.astype(np.float32) / 255.0
    mask_bool = mask.astype(bool)
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * np.array(color)
    return np.clip(out, 0, 1)


def overlay_instances(img: np.ndarray, masks: list[np.ndarray], alpha=0.5):
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


def build_foreground(
    semantic_prob: np.ndarray,
    plant_fg: np.ndarray | None,
    source: str,
    threshold: float,
) -> np.ndarray:
    sem = semantic_prob > threshold
    if source == "semantic" or plant_fg is None:
        return sem.astype(np.uint8)
    if source == "plant":
        return plant_fg.astype(np.uint8)
    if source == "intersection":
        return (plant_fg.astype(bool) & sem).astype(np.uint8)
    raise ValueError(f"unknown foreground source {source!r}")


@torch.no_grad()
def render_one(
    model: torch.nn.Module,
    coco: COCO,
    img_id: int,
    images_dir: Path,
    plant_mask_dir: Path,
    out_path: Path,
    device: torch.device,
    architecture_name: str,
    foreground_source: str,
    semantic_threshold: float,
    bandwidth: float,
    min_pixels: int,
    max_fit_points: int,
    simple: bool,
) -> tuple[int, int]:
    info = coco.imgs[img_id]
    file_name = info["file_name"]
    img = np.array(Image.open(images_dir / file_name).convert("RGB"))
    H, W = img.shape[:2]
    tensor = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).to(device)

    output = model([tensor])[0]
    semantic_logits = output["semantic"].detach().cpu().numpy()
    tag_map = output["embedding"].detach().cpu().numpy()
    semantic_prob = 1.0 / (1.0 + np.exp(-semantic_logits))

    gt_fg = gt_foreground(coco, img_id, (H, W))
    plant_fg = load_plant_mask(file_name, plant_mask_dir, target_shape=(H, W))
    sem_fg = (semantic_prob > semantic_threshold).astype(np.uint8)

    n_gt = len(coco.getAnnIds(imgIds=img_id))
    if simple:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(overlay_binary(img, gt_fg, color=(0.0, 1.0, 0.0), alpha=0.45))
        axes[0].set_title(f"GT foreground\nGT leaves: {n_gt}")
        axes[1].imshow(overlay_binary(img, sem_fg, color=(1.0, 0.0, 0.0), alpha=0.45))
        axes[1].set_title(f"Predicted foreground\nthreshold={semantic_threshold}")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(
            f"{file_name} | {architecture_name} | "
            f"stage={stage_of(file_name) or 'unknown'}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return n_gt, -1

    foreground = build_foreground(
        semantic_prob,
        plant_fg,
        foreground_source,
        semantic_threshold,
    )
    cluster_masks, _ = cluster_tags_kd(
        tag_map,
        foreground,
        bandwidth=bandwidth,
        min_pixels=min_pixels,
        max_fit_points=max_fit_points,
        rng=np.random.RandomState(0),
    )

    n_pred = len(cluster_masks)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(img)
    axes[0].set_title(f"image\n{file_name}")

    axes[1].imshow(overlay_binary(img, gt_fg, color=(0.0, 1.0, 0.0), alpha=0.45))
    axes[1].set_title(f"GT foreground\nGT leaves: {n_gt}")

    im = axes[2].imshow(semantic_prob, cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title("pred foreground score")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(overlay_binary(img, sem_fg, color=(1.0, 0.0, 0.0), alpha=0.45))
    axes[3].set_title(f"semantic mask\nthreshold={semantic_threshold}")

    axes[4].imshow(overlay_binary(img, foreground, color=(1.0, 0.5, 0.0), alpha=0.45))
    axes[4].set_title(f"cluster foreground\nsource={foreground_source}")

    axes[5].imshow(overlay_instances(img, cluster_masks, alpha=0.5))
    diff = n_pred - n_gt
    sign = "+" if diff > 0 else ""
    axes[5].set_title(f"MeanShift instances\npred: {n_pred} ({sign}{diff} vs GT)")

    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"{architecture_name} | bandwidth={bandwidth} | "
        f"stage={stage_of(file_name) or 'unknown'}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return n_gt, n_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise pixel-embedding foreground predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--coco", default="annotations/instances_test_set.json")
    parser.add_argument("--images-dir", default="crops_full/images")
    parser.add_argument("--plant-mask-dir", default="crops_full/masks")
    parser.add_argument("--out", default="results/pixel_foreground_viz")
    parser.add_argument("--architecture",
                        choices=["p2", "decoder", "decoder_h2", "fpn_h2"],
                        default="p2")
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--head-channels", type=int, default=256)
    parser.add_argument("--decoder-channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=16)
    parser.add_argument("--foreground-source",
                        choices=["semantic", "plant", "intersection"],
                        default="intersection")
    parser.add_argument("--semantic-threshold", type=float, default=0.5)
    parser.add_argument("--bandwidth", type=float, default=0.5)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--max-fit-points", type=int, default=30000)
    parser.add_argument("--stage", choices=["early", "mid", "late", "canopy"],
                        default=None)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument(
        "--simple",
        action="store_true",
        help="render only GT foreground vs predicted foreground mask",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    model, meta = load_trained_model(
        args.checkpoint,
        device,
        trainable_backbone_layers=args.trainable_backbone_layers,
        head_channels=args.head_channels,
        decoder_channels=args.decoder_channels,
        embedding_dim=args.embedding_dim,
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
    for i, img_id in enumerate(img_ids, start=1):
        file_name = coco.imgs[img_id]["file_name"]
        out_path = out_dir / f"{Path(file_name).stem}_foreground.png"
        n_gt, n_pred = render_one(
            model=model,
            coco=coco,
            img_id=img_id,
            images_dir=images_dir,
            plant_mask_dir=plant_mask_dir,
            out_path=out_path,
            device=device,
            architecture_name=args.architecture,
            foreground_source=args.foreground_source,
            semantic_threshold=args.semantic_threshold,
            bandwidth=args.bandwidth,
            min_pixels=args.min_pixels,
            max_fit_points=args.max_fit_points,
            simple=args.simple,
        )
        rows.append((out_path.name, file_name, n_gt, n_pred))
        print(f"{i}/{len(img_ids)} {file_name}: GT={n_gt} pred={n_pred}")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Pixel foreground visualisation</title>",
        "<style>body{font-family:system-ui;margin:20px;background:#111;color:#eee}"
        "img{max-width:760px;margin:8px;border:1px solid #444;border-radius:4px}"
        "figure{margin:10px 0}figcaption{color:#ccc}</style></head><body>",
        f"<h1>{html.escape(str(args.checkpoint))}</h1>",
        f"<p>foreground={args.foreground_source}, bandwidth={args.bandwidth}, "
        f"stage={args.stage or 'all'}</p>",
    ]
    for out_name, file_name, n_gt, n_pred in rows:
        parts.append(
            f"<figure><img src='{html.escape(out_name)}'>"
            f"<figcaption>{html.escape(file_name)} | GT={n_gt}, pred={n_pred}</figcaption>"
            "</figure>"
        )
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts))
    print(f"wrote {len(rows)} visualisations → {out_dir}")


if __name__ == "__main__":
    main()
