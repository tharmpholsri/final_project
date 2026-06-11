"""
Mask R-CNN model factory for leaf instance segmentation.

Wraps torchvision's `maskrcnn_resnet50_fpn_v2` (ResNet-50 + FPN backbone,
COCO V2 weights) with the classifier and mask predictor heads replaced for
our 2-class task (background + leaf).

Why MaskRCNN-V2?
  - V2 weights (2022) are stronger than V1
  - Same API as V1 — drop-in replacement
  - Native handling of variable-sized lists (no manual stacking)

Typical usage:
    from final_project.models.mask_rcnn import build_maskrcnn

    model = build_maskrcnn(num_classes=2, pretrained=True)
    model.to(device)
    model.train()
    losses = model(images, targets)        # training: returns loss dict
    model.eval()
    preds = model(images)                  # inference: list of dicts
"""

from __future__ import annotations

import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Default constants for our task
NUM_CLASSES = 2          # background + leaf
CATEGORY_NAMES = ["__background__", "leaf"]


def build_maskrcnn(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> torch.nn.Module:
    """
    Build a Mask R-CNN with the classifier + mask heads adapted for
    `num_classes` (including background).

    Args:
        num_classes: total classes including background. For our project: 2.
        pretrained: load COCO V2 weights (recommended).
        trainable_backbone_layers: how many of the last 5 ResNet blocks to
            allow gradient updates on. 0 = freeze all (linear probe),
            5 = train everything. Default 3 = a sensible middle ground.

    Returns:
        torchvision MaskRCNN model with our heads swapped in.
    """
    weights = (
        MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    )
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    # ── Replace classifier head (Fast R-CNN box predictor) ─────────
    # COCO weights gave us a 91-class classifier; we want num_classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ── Replace mask predictor head ────────────────────────────────
    # Same idea on the mask side — the final conv outputs num_classes
    # channels.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze ResNet backbone (linear-probe baseline). Heads remain trainable."""
    for p in model.backbone.parameters():
        p.requires_grad = False


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Return total / trainable / frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def pick_device() -> torch.device:
    """Auto-pick the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# =====================================================================
# CLI sanity check — run `python -m final_project.models.mask_rcnn`
# =====================================================================
if __name__ == "__main__":
    print("=== building Mask R-CNN ===")
    model = build_maskrcnn(num_classes=NUM_CLASSES, pretrained=True)
    device = pick_device()
    print(f"device: {device}")

    params = count_parameters(model)
    print(f"total params:     {params['total']:>13,}")
    print(f"trainable params: {params['trainable']:>13,}")
    print(f"frozen params:    {params['frozen']:>13,}")

    print("\n=== head shapes (verify class count) ===")
    box_pred = model.roi_heads.box_predictor
    mask_pred = model.roi_heads.mask_predictor
    print(f"  box  cls_score: {box_pred.cls_score}")
    print(f"  box  bbox_pred: {box_pred.bbox_pred}")
    print(f"  mask predictor output channels = "
          f"{mask_pred.mask_fcn_logits.out_channels}")
    assert box_pred.cls_score.out_features == NUM_CLASSES
    assert mask_pred.mask_fcn_logits.out_channels == NUM_CLASSES

    print("\n=== forward pass dry run ===")
    model.to(device).eval()
    dummy = [torch.rand(3, 320, 320, device=device)]
    with torch.no_grad():
        out = model(dummy)
    pred = out[0]
    print(f"  output keys: {list(pred.keys())}")
    for k, v in pred.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k:>7}: shape={tuple(v.shape)}  dtype={v.dtype}")

    print("\nshape checks passed.")
