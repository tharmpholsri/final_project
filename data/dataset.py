"""
PyTorch Dataset for COCO-format leaf instance segmentation.

Works uniformly for all three label sources we have:
  - PACE176 test:     annotations/instances_test_set.json   + crops_full/images/
  - PACE176 val:      annotations/instances_validation.json + crops_full/images/
  - CVPPP train pool: annotations/cvppp_coco.json           + Plant_Phenotyping_Datasets/.../Plant/

The dataset returns `(image, target)` tuples in the format expected by
torchvision's Mask R-CNN:

    image:  torch.Tensor (3, H, W), float, in [0, 1]
    target: dict with
        boxes:    Tensor (N, 4) [x1, y1, x2, y2]
        labels:   Tensor (N,)   int64, all = 1 (leaf)
        masks:    Tensor (N, H, W) uint8 binary
        image_id: Tensor (1,)
        area:     Tensor (N,)
        iscrowd:  Tensor (N,)

Usage:
    from final_project.data.dataset import LettuceCOCODataset

    train = LettuceCOCODataset(
        images_dir="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant",
        coco_path="annotations/cvppp_coco.json",
    )
    img, target = train[0]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO


class LettuceCOCODataset(torch.utils.data.Dataset):
    """COCO-style dataset for leaf instance segmentation."""

    def __init__(
        self,
        images_dir: str | Path,
        coco_path: str | Path,
        transforms: Optional[Callable] = None,
        min_anns_per_image: int = 1,
    ):
        """
        Args:
            images_dir: directory that `file_name` in the COCO JSON is
                relative to. For PACE this is `crops_full/images/`. For
                CVPPP it is `Plant_Phenotyping_Datasets/.../Plant/`.
            coco_path: path to the COCO-format JSON.
            transforms: optional callable taking (image, target) and
                returning (image, target). Applied after the base
                tensor conversion. If None, no augmentation is applied.
            min_anns_per_image: skip images with fewer than this many
                annotations. Default 1 — Mask R-CNN crashes on empty
                target sets during training. Pass 0 for evaluation if
                you want to include unlabelled images.
        """
        self.images_dir = Path(images_dir)
        self.coco_path = Path(coco_path)
        self.transforms = transforms

        self.coco = COCO(str(self.coco_path))

        # Keep only images that have enough annotations.
        all_ids = sorted(self.coco.imgs.keys())
        self.ids: list[int] = []
        for img_id in all_ids:
            n = len(self.coco.getAnnIds(imgIds=img_id))
            if n >= min_anns_per_image:
                self.ids.append(img_id)

        self._dropped = len(all_ids) - len(self.ids)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self.images_dir / img_info["file_name"]

        # PIL → numpy(H, W, 3) → tensor(3, H, W) float[0,1]
        pil_img = Image.open(img_path).convert("RGB")
        np_img = np.array(pil_img, dtype=np.uint8)
        image = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        masks: list[np.ndarray] = []
        areas: list[float] = []
        iscrowd: list[int] = []

        for ann in anns:
            # COCO bbox is [x, y, w, h]; torchvision wants [x1, y1, x2, y2].
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            # Handles polygon / RLE / compressed RLE uniformly.
            masks.append(self.coco.annToMask(ann))
            areas.append(ann.get("area", w * h))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if not boxes:
            # Should not happen with min_anns_per_image >= 1, but guard anyway.
            H, W = np_img.shape[:2]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((len(boxes),), dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
                "image_id": torch.tensor([img_id]),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    """
    Mask R-CNN does NOT collate into a single batched tensor — instead
    it expects a list of images and a list of targets, because each
    image has a different number of instances and possibly different
    sizes after augmentation.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# =====================================================================
# CLI sanity check — `python final_project/data/dataset.py`
# =====================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="annotations/cvppp_coco.json")
    ap.add_argument(
        "--images-dir",
        default="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant",
    )
    ap.add_argument("--sample-idx", type=int, default=0)
    args = ap.parse_args()

    ds = LettuceCOCODataset(images_dir=args.images_dir, coco_path=args.coco)
    print(f"dataset: {len(ds)} images (dropped {ds._dropped} with no anns)")

    img, target = ds[args.sample_idx]
    print(f"\nsample [{args.sample_idx}]:")
    print(f"  image:   shape={tuple(img.shape)}  dtype={img.dtype}  "
          f"range=[{img.min():.3f}, {img.max():.3f}]")
    for k, v in target.items():
        if hasattr(v, "shape"):
            print(f"  {k:8}: shape={tuple(v.shape)}  dtype={v.dtype}")
        else:
            print(f"  {k:8}: {v}")

    # Verify shape consistency
    assert img.dim() == 3 and img.shape[0] == 3, "image must be 3xHxW"
    assert target["masks"].shape[0] == target["boxes"].shape[0]
    assert target["masks"].shape[1:] == img.shape[1:]
    print("\nshape checks passed.")

    # Test the collate_fn quickly
    loader = torch.utils.data.DataLoader(
        ds, batch_size=2, shuffle=False, collate_fn=collate_fn
    )
    images, targets = next(iter(loader))
    print(f"\ncollate test: batch size {len(images)}, "
          f"image shapes {[tuple(i.shape) for i in images]}")
