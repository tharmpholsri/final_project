from __future__ import annotations

import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor



NUM_CLASSES = 2          
CATEGORY_NAMES = ["__background__", "leaf"]


def build_maskrcnn(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
) -> torch.nn.Module:
    """Build Mask R-CNN for background and leaf classes."""
    weights = (
        MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    )
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def freeze_backbone(model: torch.nn.Module) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")





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
