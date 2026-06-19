"""
Fine-tune Mask R-CNN on CVPPP for lettuce leaf instance segmentation.

Starts from COCO V2 pretrained weights (via `build_maskrcnn`), trains on
the CVPPP COCO-converted dataset (3,799 leaf instances across 347 images),
and validates on the PACE val split every epoch.

Checkpointing strategy
----------------------
For each training run we maintain only a small set of files in `checkpoints/`:

  - `<name>_best.pth`        overwritten when val AP50 improves (weights only)
  - `<name>_last.pth`        overwritten every epoch (weights + optimizer +
                             scheduler — resumable)
  - `<name>_epoch_NN.pth`    snapshot every `--snapshot-every` epochs
                             (weights only) — for training-dynamics analysis

Per-epoch metrics (train losses + val AP50/MAE/etc.) are appended to a JSON
history file so training curves can be reconstructed without re-running.

Usage
-----
    # default run (25 epochs, SGD lr=0.005, snapshots every 5 epochs)
    python -m final_project.train.train_maskrcnn

    # adjust epochs / lr / device
    python -m final_project.train.train_maskrcnn \\
        --epochs 30 --lr 0.005 --batch-size 4 --device cuda

    # resume from last checkpoint
    python -m final_project.train.train_maskrcnn \\
        --resume checkpoints/maskrcnn_cvppp_last.pth

    # disable snapshots (only keep best + last)
    python -m final_project.train.train_maskrcnn --snapshot-every 0
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import ConcatDataset, DataLoader

from final_project.data.dataset import LettuceCOCODataset, collate_fn
from final_project.data.transforms import (
    get_eval_transforms,
    get_train_transforms,
)
from final_project.eval.metrics import (
    compute_counting_metrics,
    compute_segmentation_ap,
)
from final_project.models.mask_rcnn import (
    build_maskrcnn,
    count_parameters,
    pick_device,
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


# ════════════════════════════════════════════════════════════════════
# Checkpoint I/O
# ════════════════════════════════════════════════════════════════════
def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    epoch: int | None = None,
    best_ap50: float | None = None,
    config: dict | None = None,
    weights_only: bool = False,
) -> None:
    """Save model (and optionally optimizer/scheduler) to disk.

    weights_only=True drops the optimizer + scheduler state, ~halving the
    file size. Use this for `best` and periodic snapshots; keep the full
    state for `last` so training can resume.
    """
    state: dict = {"model": model.state_dict()}
    if not weights_only:
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        if lr_scheduler is not None:
            state["lr_scheduler"] = lr_scheduler.state_dict()
    state["epoch"] = epoch
    state["best_ap50"] = best_ap50
    state["config"] = config
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler=None,
    device: torch.device | None = None,
) -> tuple[int | None, float]:
    """Restore weights (and optionally optimizer/scheduler) from a checkpoint.

    Returns (last_epoch, best_ap50_so_far).
    """
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if lr_scheduler is not None and "lr_scheduler" in state:
        lr_scheduler.load_state_dict(state["lr_scheduler"])
    return state.get("epoch"), state.get("best_ap50") or 0.0


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════
def move_to_device(images, targets, device):
    images = [img.to(device) for img in images]
    targets = [
        {k: v.to(device) if isinstance(v, torch.Tensor) else v
         for k, v in t.items()}
        for t in targets
    ]
    return images, targets


# ════════════════════════════════════════════════════════════════════
# Train / eval loops
# ════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 50,
    grad_clip: float = 10.0,
) -> dict:
    """Run one training epoch. Returns mean per-loss values."""
    model.train()
    losses_sum: dict = {}
    n_batches = 0
    t0 = time.time()

    for i, (images, targets) in enumerate(loader):
        images, targets = move_to_device(images, targets, device)
        loss_dict = model(images, targets)
        total = sum(loss_dict.values())

        optimizer.zero_grad()
        total.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                grad_clip,
            )
        optimizer.step()

        for k, v in loss_dict.items():
            losses_sum[k] = losses_sum.get(k, 0.0) + v.item()
        losses_sum["total"] = losses_sum.get("total", 0.0) + total.item()
        n_batches += 1

        if (i + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  epoch {epoch}  iter {i + 1:4d}/{len(loader)}  "
                f"loss={total.item():.4f}  lr={lr:.2e}"
            )

    elapsed = time.time() - t0
    avg = {k: v / max(n_batches, 1) for k, v in losses_sum.items()}
    print(
        f"  epoch {epoch} train loss: {avg.get('total', 0.0):.4f}  "
        f"({elapsed:.1f}s, {n_batches} batches)"
    )
    return avg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: LettuceCOCODataset,
    val_coco_path: str | Path,
    device: torch.device,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    min_mask_area: int = 30,
) -> dict:
    """Run inference on a dataset and compute AP50 + counting metrics."""
    model.eval()
    predictions: list[dict] = []

    for idx in range(len(dataset)):
        image, target = dataset[idx]
        image = image.to(device)
        out = model([image])[0]

        scores = out["scores"].cpu().numpy()
        masks_prob = out["masks"].cpu().numpy().squeeze(1)  # (N, H, W)
        img_id = int(target["image_id"].item())

        keep = scores >= score_threshold
        for k in np.where(keep)[0]:
            binary = (masks_prob[k] > mask_threshold).astype(np.uint8)
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
                "score": float(scores[k]),
            })

    if not predictions:
        return {
            "AP50": 0.0, "AP": 0.0, "MAE": 0.0,
            "RMSE": 0.0, "DiC_mean": 0.0, "pred_total": 0,
        }

    ap = compute_segmentation_ap(val_coco_path, predictions)
    ct = compute_counting_metrics(val_coco_path, predictions)
    return {
        "AP50": float(ap["AP50"]),
        "AP": float(ap["AP"]),
        "MAE": float(ct["MAE"]),
        "RMSE": float(ct["RMSE"]),
        "DiC_mean": float(ct["DiC_mean"]),
        "pred_total": int(ct["pred_total"]),
    }


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fine-tune Mask R-CNN on CVPPP, validate on PACE val.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    ap.add_argument("--train-coco", default="annotations/cvppp_coco.json")
    ap.add_argument(
        "--train-images",
        default="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant",
    )
    ap.add_argument(
        "--train-coco-extra", default=None,
        help="optional 2nd training COCO (e.g. PACE canopy annotations)",
    )
    ap.add_argument(
        "--train-images-extra", default=None,
        help="image dir for --train-coco-extra (e.g. crops_full/images)",
    )
    ap.add_argument("--val-coco", default="annotations/instances_validation.json")
    ap.add_argument("--val-images", default="crops_full/images")

    # Copy-paste augmentation (applied to the extra/PACE source only)
    ap.add_argument("--copy-paste", action="store_true",
                    help="enable copy-paste augmentation on --train-coco-extra")
    ap.add_argument("--leaf-bank", default="leaf_bank")
    ap.add_argument("--pots-dir", default="crops_full/pots")
    ap.add_argument("--cp-stage-match", choices=["strict", "off"], default="strict")
    ap.add_argument("--cp-p-apply", type=float, default=0.5,
                    help="probability of applying copy-paste per image")
    ap.add_argument("--cp-n-paste-min", type=int, default=1)
    ap.add_argument("--cp-n-paste-max", type=int, default=4)

    # Model
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--trainable-backbone-layers", type=int, default=3,
                    help="how many of the last 5 ResNet stages to fine-tune")

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

    # Evaluation
    ap.add_argument("--val-score-threshold", type=float, default=0.5)
    ap.add_argument("--val-mask-threshold", type=float, default=0.5)

    # Checkpointing
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ckpt-name", default="maskrcnn_cvppp")
    ap.add_argument("--snapshot-every", type=int, default=5,
                    help="save weights-only snapshot every N epochs (0=off)")
    ap.add_argument("--resume", default=None, help="path to a 'last' ckpt")

    # Logging
    ap.add_argument(
        "--history-path",
        default="results/maskrcnn_cvppp_history.json",
    )
    ap.add_argument(
        "--config-path",
        default="results/maskrcnn_cvppp_config.json",
    )
    ap.add_argument("--log-every", type=int, default=50)

    # Device
    ap.add_argument("--device", default=None,
                    help="cuda / mps / cpu (auto-pick if omitted)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── setup ────────────────────────────────────────────────────────
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    print(f"device: {device}")
    print(f"seed:   {args.seed}")

    # Datasets + loaders
    # Primary source (CVPPP by default) — no copy-paste, just basic aug.
    primary_ds = LettuceCOCODataset(
        images_dir=args.train_images,
        coco_path=args.train_coco,
        transforms=get_train_transforms(copy_paste=None),
    )
    print(f"train (primary): {len(primary_ds)} images  "
          f"(dropped {primary_ds._dropped} with no anns)")

    datasets: list = [primary_ds]

    # Optional 2nd source (PACE canopy) — gets copy-paste if --copy-paste.
    if args.train_coco_extra and args.train_images_extra:
        cp_transform = None
        if args.copy_paste:
            from final_project.augment.copy_paste import (
                CopyPaste, make_filename_lookup, make_pot_loader,
            )
            pot_loader = make_pot_loader(
                args.train_coco_extra, args.pots_dir,
            )
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
            print(f"copy-paste: disabled (still adding extra source as plain training data)")

        extra_ds = LettuceCOCODataset(
            images_dir=args.train_images_extra,
            coco_path=args.train_coco_extra,
            transforms=get_train_transforms(copy_paste=cp_transform),
        )
        print(f"train (extra):   {len(extra_ds)} images  "
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
    print(f"val:   {len(val_ds)} images  "
          f"(dropped {val_ds._dropped} with no anns)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    # Model
    model = build_maskrcnn(
        num_classes=args.num_classes,
        pretrained=True,
        trainable_backbone_layers=args.trainable_backbone_layers,
    ).to(device)

    p = count_parameters(model)
    print(f"params: total={p['total']:,}  trainable={p['trainable']:,}  "
          f"frozen={p['frozen']:,}")

    # Optimizer + LR scheduler
    trainable_params = [p_ for p_ in model.parameters() if p_.requires_grad]
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

    # Save config alongside checkpoints for reproducibility
    config = vars(args).copy()
    config_path = Path(args.config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2))
    print(f"config written → {config_path}")

    # ── resume ───────────────────────────────────────────────────────
    start_epoch = 0
    best_ap50 = 0.0
    if args.resume:
        print(f"resuming from {args.resume}")
        last_epoch, best_ap50 = load_checkpoint(
            args.resume, model, optimizer, lr_scheduler, device=device,
        )
        if last_epoch is not None:
            start_epoch = last_epoch + 1
        print(f"  resumed at epoch {start_epoch}, "
              f"best AP50 so far: {best_ap50:.4f}")

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
            model, train_loader, optimizer, device, epoch + 1,
            log_every=args.log_every, grad_clip=args.grad_clip,
        )
        lr_scheduler.step()

        val_metrics = evaluate(
            model, val_ds, args.val_coco, device,
            score_threshold=args.val_score_threshold,
            mask_threshold=args.val_mask_threshold,
        )
        print(
            f"  val AP50={val_metrics['AP50']:.4f}  "
            f"AP={val_metrics['AP']:.4f}  "
            f"MAE={val_metrics['MAE']:.2f}  "
            f"DiC={val_metrics['DiC_mean']:+.2f}  "
            f"pred={val_metrics['pred_total']}"
        )

        # Append to history JSON
        history.append({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        history_path.write_text(json.dumps(history, indent=2))

        # ── save: last (resumable), best, periodic snapshots ─────────
        save_checkpoint(
            ckpt_dir / f"{args.ckpt_name}_last.pth",
            model, optimizer, lr_scheduler,
            epoch=epoch, best_ap50=best_ap50, config=config,
        )

        if val_metrics["AP50"] > best_ap50:
            best_ap50 = val_metrics["AP50"]
            save_checkpoint(
                ckpt_dir / f"{args.ckpt_name}_best.pth",
                model,
                epoch=epoch, best_ap50=best_ap50, config=config,
                weights_only=True,
            )
            print(f"  ★ new best val AP50 = {best_ap50:.4f} → saved")

        if args.snapshot_every > 0 and (epoch + 1) % args.snapshot_every == 0:
            snap_path = (
                ckpt_dir / f"{args.ckpt_name}_epoch_{epoch + 1:02d}.pth"
            )
            save_checkpoint(
                snap_path, model,
                epoch=epoch, best_ap50=best_ap50, config=config,
                weights_only=True,
            )
            print(f"  snapshot → {snap_path.name}")

    print(f"\ntraining complete. best val AP50: {best_ap50:.4f}")
    print(f"history → {history_path}")


if __name__ == "__main__":
    main()
