"""
Fine-tune the pixel-embedding model on CVPPP (+ optional PACE canopy + copy-paste).

Mirrors the structure of `final_project/train/train_maskrcnn.py` so that
results are directly comparable: same training data, same val/test split,
same checkpointing pattern, same logging format. The only differences
are the model (`PixelEmbeddingModel`) and the loss (`TaggingLoss`), and
the fact that per-epoch validation tracks the loss components rather
than AP50 (clustering + AP50 happens in the separate
`evaluate_pixel_embed.py` script — see Phase 3).

Checkpointing strategy
----------------------
Identical to `train_maskrcnn.py`:
  - <name>_best.pth     overwritten when val_total loss improves (weights only)
  - <name>_last.pth     overwritten every epoch (weights + optimizer + scheduler)
  - <name>_epoch_NN.pth weights-only snapshot every `--snapshot-every` epochs

Usage
-----
    # default — CVPPP only, no copy-paste, 25 epochs
    python -m final_project.train.train_pixel_embed

    # match the variant C training data (CVPPP + PACE + copy-paste)
    python -m final_project.train.train_pixel_embed \\
        --train-coco-extra annotations/instances_train_set.json \\
        --train-images-extra crops_full/images \\
        --copy-paste --leaf-bank leaf_bank --pots-dir crops_full/pots

    # resume from a 'last' checkpoint
    python -m final_project.train.train_pixel_embed \\
        --resume checkpoints/pixel_embed_last.pth
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader

from final_project.data.dataset import LettuceCOCODataset, collate_fn
from final_project.data.transforms import (
    get_eval_transforms,
    get_train_transforms,
)
from final_project.models.pixel_embed import (
    PixelEmbeddingModel,
    TaggingLoss,
    build_pixel_embed_model,
)
from final_project.models.pixel_embed_decoder import (
    build_pixel_embed_decoder_model,
    build_pixel_embed_fpn_h2_decoder_model,
    build_pixel_embed_h2_decoder_model,
)


# ════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ════════════════════════════════════════════════════════════════════
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    epoch: int | None = None,
    best_val_loss: float | None = None,
    config: dict | None = None,
    weights_only: bool = False,
) -> None:
    state: dict = {"model": model.state_dict()}
    if not weights_only:
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()
    state["epoch"] = epoch
    state["best_val_loss"] = best_val_loss
    state["config"] = config
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    device: torch.device | None = None,
) -> tuple[int | None, float]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return state.get("epoch"), state.get("best_val_loss") or float("inf")


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def move_images(images, device):
    return [img.to(device) for img in images]


def move_targets(targets, device):
    out = []
    for t in targets:
        out.append({
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in t.items()
        })
    return out


# ════════════════════════════════════════════════════════════════════
# Train / eval loops
# ════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: PixelEmbeddingModel,
    loss_fn: TaggingLoss,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 50,
    grad_clip: float = 10.0,
) -> dict:
    """One epoch of training. Returns mean loss components for the epoch."""
    model.train()
    sums = {"loss_pull": 0.0, "loss_push": 0.0, "loss_detection": 0.0, "total": 0.0}
    n_batches = 0
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images = move_images(images, device)
        targets = move_targets(targets, device)

        preds = model(images)
        losses = loss_fn(preds, targets)
        total = losses["total"]

        optimizer.zero_grad()
        total.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        for k in sums:
            sums[k] += float(losses[k].item())
        n_batches += 1

        if (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch}  iter {i + 1:4d}/{len(loader)}  "
                f"pull={losses['loss_pull'].item():.3f}  "
                f"push={losses['loss_push'].item():.3f}  "
                f"sem={losses['loss_detection'].item():.3f}  "
                f"total={losses['total'].item():.3f}  lr={lr:.2e}"
            )

    elapsed = time.time() - t0
    avg = {k: v / max(n_batches, 1) for k, v in sums.items()}
    print(
        f"  epoch {epoch} train  pull={avg['loss_pull']:.3f}  "
        f"push={avg['loss_push']:.3f}  sem={avg['loss_detection']:.3f}  "
        f"total={avg['total']:.3f}  ({elapsed:.1f}s, {n_batches} batches)"
    )
    return avg


@torch.no_grad()
def validate_loss(
    model: PixelEmbeddingModel,
    loss_fn: TaggingLoss,
    dataset: LettuceCOCODataset,
    device: torch.device,
) -> dict:
    """Run the model on a dataset and compute the loss components.

    Used as the per-epoch validation signal for picking best checkpoint.
    The full instance-level evaluation (clustering + AP50 + counting
    metrics) lives in `evaluate_pixel_embed.py`.
    """
    model.eval()
    sums = {"loss_pull": 0.0, "loss_push": 0.0, "loss_detection": 0.0, "total": 0.0}
    n = 0
    for idx in range(len(dataset)):
        image, target = dataset[idx]
        image = image.to(device)
        target = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in target.items()
        }
        preds = model([image])
        losses = loss_fn(preds, [target])
        for k in sums:
            sums[k] += float(losses[k].item())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fine-tune the pixel-embedding model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data — primary
    ap.add_argument("--train-coco", default="annotations/cvppp_coco.json")
    ap.add_argument(
        "--train-images",
        default="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant",
    )
    # Data — optional extra (PACE)
    ap.add_argument("--train-coco-extra", default=None)
    ap.add_argument("--train-images-extra", default=None)
    # Val
    ap.add_argument("--val-coco", default="annotations/instances_validation.json")
    ap.add_argument("--val-images", default="crops_full/images")

    # Copy-paste augmentation (applied to extra/PACE source only)
    ap.add_argument("--copy-paste", action="store_true")
    ap.add_argument("--leaf-bank", default="leaf_bank")
    ap.add_argument("--pots-dir", default="crops_full/pots")
    ap.add_argument("--cp-stage-match", choices=["strict", "off"], default="strict")
    ap.add_argument("--cp-p-apply", type=float, default=0.5)
    ap.add_argument("--cp-n-paste-min", type=int, default=1)
    ap.add_argument("--cp-n-paste-max", type=int, default=4)

    # Model
    ap.add_argument(
        "--architecture",
        choices=["p2", "decoder", "decoder_h2", "fpn_h2"],
        default="p2",
        help=(
            "p2: predict heads at P2/H4 then upsample outputs; "
            "decoder: upsample features to H2 and H before heads; "
            "decoder_h2: upsample P2 features to H2 before heads, then upsample outputs; "
            "fpn_h2: fuse P2-P5 at H4, upsample fused features to H2 before heads"
        ),
    )
    ap.add_argument("--trainable-backbone-layers", type=int, default=3)
    ap.add_argument("--head-channels", type=int, default=256)
    ap.add_argument(
        "--decoder-channels",
        type=int,
        default=64,
        help="channels used after feature upsampling when --architecture decoder",
    )
    ap.add_argument("--embedding-dim", type=int, default=16,
                    help="dimensionality of the per-pixel tag (v3 default)")

    # Loss
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian bandwidth for the push term")
    ap.add_argument("--n-sample", type=int, default=20,
                    help="K pixels sampled per instance for pairwise terms")
    ap.add_argument("--min-pixels", type=int, default=10,
                    help="ignore instances with fewer than this many pixels")
    ap.add_argument("--lambda-pull", type=float, default=1.0)
    ap.add_argument("--lambda-push", type=float, default=1.0)
    ap.add_argument("--lambda-detection", type=float, default=1.0)
    ap.add_argument(
        "--detection-loss", choices=["bce", "mse"], default="bce",
        help="foreground objective; use mse for the paper-like heatmap loss",
    )

    # Training
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--lr-step-size", type=int, default=10)
    ap.add_argument("--lr-gamma", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)

    # Checkpointing
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt-name", default="pixel_embed")
    ap.add_argument("--snapshot-every", type=int, default=5)
    ap.add_argument("--resume", default=None)

    # Logging
    ap.add_argument(
        "--history-path",
        default="results/pixel_embed_history.json",
    )
    ap.add_argument(
        "--config-path",
        default="results/pixel_embed_config.json",
    )
    ap.add_argument("--log-every", type=int, default=50)

    # Device
    ap.add_argument("--device", default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── setup ────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(f"seed:   {args.seed}")

    # Datasets
    primary_ds = LettuceCOCODataset(
        images_dir=args.train_images,
        coco_path=args.train_coco,
        transforms=get_train_transforms(copy_paste=None),
    )
    print(f"train (primary): {len(primary_ds)} images  "
          f"(dropped {primary_ds._dropped} with no anns)")

    datasets: list = [primary_ds]

    if args.train_coco_extra and args.train_images_extra:
        cp_transform = None
        if args.copy_paste:
            from final_project.augment.copy_paste import (
                CopyPaste, make_filename_lookup, make_pot_loader,
            )
            pot_loader = make_pot_loader(args.train_coco_extra, args.pots_dir)
            name_lookup = make_filename_lookup(args.train_coco_extra)
            cp_transform = CopyPaste(
                leaf_bank_dir=args.leaf_bank,
                pot_mask_loader=pot_loader,
                filename_from_image_id=name_lookup,
                stage_match=args.cp_stage_match,
                p_apply=args.cp_p_apply,
                n_paste_range=(args.cp_n_paste_min, args.cp_n_paste_max),
                seed=args.seed,
            )
            print(f"copy-paste: ENABLED on {args.train_coco_extra}")
            print(f"            leaf bank size: {len(cp_transform.bank)}  "
                  f"stages: {cp_transform.bank.stages()}")
        else:
            print(f"copy-paste: disabled (adding extra source as plain data)")

        extra_ds = LettuceCOCODataset(
            images_dir=args.train_images_extra,
            coco_path=args.train_coco_extra,
            transforms=get_train_transforms(copy_paste=cp_transform),
        )
        print(f"train (extra): {len(extra_ds)} images  "
              f"(dropped {extra_ds._dropped} with no anns)")
        datasets.append(extra_ds)

    train_ds = ConcatDataset(datasets) if len(datasets) > 1 else primary_ds
    total_imgs = sum(len(d) for d in datasets)
    print(f"train (combined): {total_imgs} images across {len(datasets)} source(s)")

    val_ds = LettuceCOCODataset(
        images_dir=args.val_images,
        coco_path=args.val_coco,
        transforms=get_eval_transforms(),
    )
    print(f"val: {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    # Model + loss
    if args.architecture == "decoder":
        model = build_pixel_embed_decoder_model(
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers,
            head_channels=args.head_channels,
            decoder_channels=args.decoder_channels,
            embedding_dim=args.embedding_dim,
        ).to(device)
    elif args.architecture == "decoder_h2":
        model = build_pixel_embed_h2_decoder_model(
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers,
            head_channels=args.head_channels,
            decoder_channels=args.decoder_channels,
            embedding_dim=args.embedding_dim,
        ).to(device)
    elif args.architecture == "fpn_h2":
        model = build_pixel_embed_fpn_h2_decoder_model(
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers,
            head_channels=args.head_channels,
            decoder_channels=args.decoder_channels,
            embedding_dim=args.embedding_dim,
        ).to(device)
    else:
        model = build_pixel_embed_model(
            pretrained=True,
            trainable_backbone_layers=args.trainable_backbone_layers,
            head_channels=args.head_channels,
            embedding_dim=args.embedding_dim,
        ).to(device)
    loss_fn = TaggingLoss(
        sigma=args.sigma,
        n_sample=args.n_sample,
        min_pixels=args.min_pixels,
        lambda_pull=args.lambda_pull,
        lambda_push=args.lambda_push,
        lambda_detection=args.lambda_detection,
        detection_loss=args.detection_loss,
    )

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params: total={n_total:,}  trainable={n_train:,}  "
          f"frozen={n_total - n_train:,}")

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
    )

    config = vars(args).copy()
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"config written → {config_path}")

    # ── resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        print(f"resuming from {args.resume}")
        last_epoch, best_val_loss = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, device=device,
        )
        if last_epoch is not None:
            start_epoch = last_epoch + 1
        print(f"  resumed at epoch {start_epoch}, "
              f"best_val_loss so far: {best_val_loss:.4f}")

    # ── train ────────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    if history_path.exists() and args.resume:
        history = json.loads(history_path.read_text())
        print(f"resumed history: {len(history)} entries")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")

        train_losses = train_one_epoch(
            model, loss_fn, train_loader, optimizer, device, epoch + 1,
            log_every=args.log_every, grad_clip=args.grad_clip,
        )
        lr_scheduler.step()

        val_losses = validate_loss(model, loss_fn, val_ds, device)
        print(
            f"  val pull={val_losses['loss_pull']:.3f}  "
            f"push={val_losses['loss_push']:.3f}  "
            f"sem={val_losses['loss_detection']:.3f}  "
            f"total={val_losses['total']:.3f}"
        )

        history.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
        })
        history_path.write_text(json.dumps(history, indent=2))

        # ── save: last (resumable) ──────────────────────────────────
        save_checkpoint(
            ckpt_dir / f"{args.ckpt_name}_last.pth",
            model, optimizer, lr_scheduler,
            epoch=epoch, best_val_loss=best_val_loss, config=config,
        )

        # ── save: best ──────────────────────────────────────────────
        if val_losses["total"] < best_val_loss:
            best_val_loss = val_losses["total"]
            save_checkpoint(
                ckpt_dir / f"{args.ckpt_name}_best.pth",
                model,
                epoch=epoch, best_val_loss=best_val_loss, config=config,
                weights_only=True,
            )
            print(f"  ★ new best val_total = {best_val_loss:.4f} → saved")

        # ── save: periodic snapshot ─────────────────────────────────
        if args.snapshot_every > 0 and (epoch + 1) % args.snapshot_every == 0:
            snap_path = (
                ckpt_dir / f"{args.ckpt_name}_epoch_{epoch + 1:02d}.pth"
            )
            save_checkpoint(
                snap_path, model,
                epoch=epoch, best_val_loss=best_val_loss, config=config,
                weights_only=True,
            )
            print(f"  snapshot → {snap_path.name}")

    print(f"\ntraining complete. best val_total: {best_val_loss:.4f}")
    print(f"history → {history_path}")


if __name__ == "__main__":
    main()
