"""
Augmentation pipelines for the leaf instance segmentation Dataset.

Built on top of `torchvision.transforms.v2`, which natively understands
detection / segmentation targets (boxes + masks get geometric transforms
applied automatically when wrapped in `tv_tensors`).

The Dataset (`final_project/data/dataset.py`) returns plain Python dicts
of regular tensors. The first transform in each pipeline (`WrapAsTVTensors`)
wraps them into `tv_tensors.Image / BoundingBoxes / Mask` so subsequent
augmentations can transform them correctly. The last transform unwraps
them back to plain tensors so model code stays agnostic.

Note on normalization: torchvision's `MaskRCNN` has its own image
normalization step (ImageNet stats) inside `model.transform`. We do NOT
normalize in this pipeline — we only need to provide float images in
`[0, 1]`.

Usage:
    from final_project.data.dataset import LettuceCOCODataset
    from final_project.data.transforms import (
        get_train_transforms,
        get_eval_transforms,
    )

    train_ds = LettuceCOCODataset(
        images_dir="...",
        coco_path="...",
        transforms=get_train_transforms(),
    )
    val_ds = LettuceCOCODataset(
        images_dir="...",
        coco_path="...",
        transforms=get_eval_transforms(),
    )
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


# =====================================================================
# Wrappers — bridge plain dict targets ↔ tv_tensors
# =====================================================================


class WrapAsTVTensors:
    """
    Convert the plain `(image_tensor, target_dict)` produced by
    `LettuceCOCODataset` into v2-compatible types so geometric transforms
    in the pipeline propagate to boxes and masks.
    """

    def __call__(self, image: torch.Tensor, target: dict) -> tuple:
        H, W = image.shape[-2:]
        new_target = dict(target)
        new_target["boxes"] = tv_tensors.BoundingBoxes(
            target["boxes"], format="XYXY", canvas_size=(H, W)
        )
        new_target["masks"] = tv_tensors.Mask(target["masks"])
        return tv_tensors.Image(image), new_target


class UnwrapFromTVTensors:
    """
    Final step that converts `tv_tensors` back to plain `torch.Tensor`,
    so downstream code can treat the dict as a normal dict of tensors.
    """

    def __call__(self, image, target: dict) -> tuple:
        plain_target = {}
        for k, v in target.items():
            if isinstance(v, tv_tensors.TVTensor):
                plain_target[k] = v.as_subclass(torch.Tensor)
            else:
                plain_target[k] = v
        if isinstance(image, tv_tensors.TVTensor):
            image = image.as_subclass(torch.Tensor)
        return image, plain_target


# =====================================================================
# Pipelines
# =====================================================================


def get_train_transforms(
    flip_h: float = 0.5,
    flip_v: float = 0.5,
    photometric: bool = True,
    copy_paste: Optional[Callable] = None,
) -> Callable:
    """
    Training augmentation pipeline.

    Args:
        flip_h: probability of horizontal flip.
        flip_v: probability of vertical flip (lettuce is photographed
            top-down so vertical flip is biologically plausible).
        photometric: if True, apply random brightness / contrast / hue
            shifts to simulate lighting variation.
        copy_paste: optional callable for copy-paste augmentation. It
            takes `(image_tvt, target_dict)` and returns the same after
            pasting extra leaves. Will be implemented in
            `final_project/augment/copy_paste.py`.

    Order of operations:
        WrapAsTVTensors
        → (geometric: flip)
        → (optional: copy-paste — pasted leaves should go through any
           subsequent photometric augmentation but not be flipped again)
        → (photometric)
        → SanitizeBoundingBoxes (drop boxes that became invalid)
        → UnwrapFromTVTensors
    """
    steps: list = [WrapAsTVTensors()]

    if flip_h > 0:
        steps.append(T.RandomHorizontalFlip(p=flip_h))
    if flip_v > 0:
        steps.append(T.RandomVerticalFlip(p=flip_v))

    if copy_paste is not None:
        steps.append(copy_paste)  # see final_project/augment/copy_paste.py

    if photometric:
        # RandomPhotometricDistort is a single transform that randomly
        # applies brightness, contrast, saturation, hue shifts.
        steps.append(T.RandomPhotometricDistort(p=0.5))

    steps.append(T.SanitizeBoundingBoxes())
    steps.append(UnwrapFromTVTensors())

    return T.Compose(steps)


def get_eval_transforms() -> Callable:
    """
    No-augmentation pipeline used for validation and test. The wrap /
    unwrap pair is still applied so the dataset output shape and dtypes
    match the training pipeline exactly.
    """
    return T.Compose([WrapAsTVTensors(), UnwrapFromTVTensors()])


# Alias for clarity at call sites
get_test_transforms = get_eval_transforms


# =====================================================================
# CLI sanity check — `python final_project/data/transforms.py`
# =====================================================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from final_project.data.dataset import LettuceCOCODataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="annotations/instances_validation.json")
    ap.add_argument("--images-dir", default="crops_full/images")
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    # Without augmentation
    ds_eval = LettuceCOCODataset(
        images_dir=args.images_dir,
        coco_path=args.coco,
        transforms=get_eval_transforms(),
    )
    img, target = ds_eval[args.sample_idx]
    print("=== EVAL transform (no augmentation) ===")
    print(f"  image:     shape={tuple(img.shape)} dtype={img.dtype} "
          f"range=[{img.min():.3f}, {img.max():.3f}]")
    print(f"  boxes:     shape={tuple(target['boxes'].shape)} dtype={target['boxes'].dtype}")
    print(f"  masks:     shape={tuple(target['masks'].shape)} dtype={target['masks'].dtype}")
    print(f"  labels:    shape={tuple(target['labels'].shape)}")

    # Same sample with training augmentations (random — run twice to see variation)
    ds_train = LettuceCOCODataset(
        images_dir=args.images_dir,
        coco_path=args.coco,
        transforms=get_train_transforms(),
    )

    print("\n=== TRAIN transform (with augmentation) — sample 3 times ===")
    torch.manual_seed(0)
    for i in range(3):
        img_t, tgt_t = ds_train[args.sample_idx]
        n_boxes = tgt_t["boxes"].shape[0]
        print(f"  trial {i}: image={tuple(img_t.shape)}  "
              f"boxes={n_boxes}  "
              f"masks={tuple(tgt_t['masks'].shape)}  "
              f"img range=[{img_t.min():.3f}, {img_t.max():.3f}]")
        # shape consistency
        assert tgt_t["masks"].shape[0] == tgt_t["boxes"].shape[0] == tgt_t["labels"].shape[0]
        assert tgt_t["masks"].shape[1:] == img_t.shape[1:]

    print("\nshape checks passed.")
