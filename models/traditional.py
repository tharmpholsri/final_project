from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, segmentation




@dataclass
class TraditionalParams:
    sigma_dist: float = 3.0          
    min_distance: int = 30           
    sigma_gradient: float = 2.0      
    min_inst_size: int = 300         

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def segment_leaves(
    image_rgb: np.ndarray,
    plant_mask: np.ndarray,
    params: Optional[TraditionalParams] = None,
) -> np.ndarray:
    if params is None:
        params = TraditionalParams()

    plant_bool = plant_mask > 0
    if not plant_bool.any():
        return np.zeros(plant_mask.shape, dtype=np.int32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    dist = ndimage.distance_transform_edt(plant_bool)
    dist_smooth = ndimage.gaussian_filter(dist, sigma=params.sigma_dist)
    coords = feature.peak_local_max(
        dist_smooth,
        min_distance=int(params.min_distance),
        labels=plant_bool,
    )
    if len(coords) == 0:
        return np.zeros(plant_mask.shape, dtype=np.int32)

    markers = np.zeros(plant_bool.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    grad = ndimage.gaussian_gradient_magnitude(gray, sigma=params.sigma_gradient)
    instances = segmentation.watershed(grad, markers, mask=plant_bool).astype(np.int32)

    if params.min_inst_size > 0:
        for inst_id in np.unique(instances):
            if inst_id == 0:
                continue
            if (instances == inst_id).sum() < params.min_inst_size:
                instances[instances == inst_id] = 0
    if instances.max() > 0:
        kept_ids = sorted(int(v) for v in np.unique(instances) if v != 0)
        remap = {old: new for new, old in enumerate(kept_ids, start=1)}
        out = np.zeros_like(instances)
        for old, new in remap.items():
            out[instances == old] = new
        instances = out
    return instances


def instances_to_predictions(
    instances: np.ndarray,
    image_id: int,
    category_id: int = 1,
    score_mode: str = "area",
) -> list[dict]:
    from pycocotools import mask as mask_utils

    inst_ids = [int(v) for v in np.unique(instances) if v != 0]
    if not inst_ids:
        return []

    areas = {i: int((instances == i).sum()) for i in inst_ids}
    if score_mode == "area":
        max_area = max(areas.values()) or 1
    elif score_mode == "constant":
        max_area = None
    else:
        raise ValueError(f"unknown score_mode={score_mode}")
    preds = []
    for inst_id in inst_ids:
        binary = (instances == inst_id).astype(np.uint8)
        ys, xs = np.where(binary)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
        bbox = [float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1)]
        rle = mask_utils.encode(np.asfortranarray(binary))
        rle["counts"] = rle["counts"].decode("utf-8")  
        score = areas[inst_id] / max_area if score_mode == "area" else 1.0
        preds.append({
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": rle,
            "bbox": bbox,
            "area": areas[inst_id],
            "score": float(score),
        })
    return preds


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    import matplotlib.pyplot as plt
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="crops_full/images/PS_Tray_080_19_p4.png")
    ap.add_argument("--mask",  default="crops_full/masks/PS_Tray_080_19_p4.png")
    ap.add_argument("--out",   default="annotations/traditional_demo.png")
    args = ap.parse_args()

    img = np.array(Image.open(args.image).convert("RGB"))
    plant_mask = np.array(Image.open(args.mask).convert("L"))
    instances = segment_leaves(img, plant_mask)
    n_leaves = int(instances.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title("image"); axes[0].axis("off")
    axes[1].imshow(plant_mask, cmap="gray"); axes[1].set_title("plant mask"); axes[1].axis("off")
    axes[2].imshow(img)
    rng = np.random.default_rng(0)
    overlay = np.zeros((*instances.shape, 4))
    for i in range(1, n_leaves + 1):
        color = rng.uniform(0.4, 1.0, 3)
        overlay[instances == i] = (*color, 0.5)
    axes[2].imshow(overlay)
    axes[2].set_title(f"watershed instances: {n_leaves}")
    axes[2].axis("off")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
    print(f"detected {n_leaves} leaves")
