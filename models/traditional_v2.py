from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage
from skimage import feature, segmentation


@dataclass
class EdgeAwareParams:
    sigma_dist: float = 3.0          
    min_distance: int = 30           
    canny_low:  int = 30             
    canny_high: int = 100            
    shadow_percentile: float = 30.0  
    sigma_gradient: float = 1.0      
    w_gradient: float = 0.4
    w_canny:    float = 0.3
    w_shadow:   float = 0.3
    min_inst_size: int = 100

    def as_dict(self) -> dict:
        return asdict(self)


def segment_leaves_edge_aware(
    image_rgb: np.ndarray,
    plant_mask: np.ndarray,
    params: Optional[EdgeAwareParams] = None,
    return_intermediates: bool = False,
):
    if params is None:
        params = EdgeAwareParams()

    plant_bool = plant_mask > 0
    if not plant_bool.any():
        empty = np.zeros(plant_mask.shape, dtype=np.int32)
        return (empty, {}) if return_intermediates else empty
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad = ndimage.gaussian_gradient_magnitude(
        gray.astype(float), sigma=params.sigma_gradient
    )
    grad_norm = _normalize_01(grad)
    canny = cv2.Canny(gray, params.canny_low, params.canny_high)
    canny = canny.astype(bool) & plant_bool
    canny_f = canny.astype(float)
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    v_channel = hsv[..., 2]
    if plant_bool.any():
        v_in_plant = v_channel[plant_bool]
        shadow_thr = np.percentile(v_in_plant, params.shadow_percentile)
    else:
        shadow_thr = 0
    shadows = (v_channel < shadow_thr) & plant_bool
    shadows_f = shadows.astype(float)

    barrier = (
        params.w_gradient * grad_norm
        + params.w_canny    * canny_f
        + params.w_shadow   * shadows_f
    )
    
    barrier_with_bg = barrier.copy()
    barrier_with_bg[~plant_bool] = 1.0
    free_space = (barrier_with_bg < 0.15) & plant_bool
    dist = ndimage.distance_transform_edt(free_space)
    dist_smooth = ndimage.gaussian_filter(dist, sigma=params.sigma_dist)
    coords = feature.peak_local_max(
        dist_smooth, min_distance=int(params.min_distance), labels=plant_bool
    )
    markers = np.zeros(plant_bool.shape, dtype=np.int32)
    for i, (y, x) in enumerate(coords, start=1):
        markers[y, x] = i
    if markers.max() == 0:
        instances = np.zeros(plant_bool.shape, dtype=np.int32)
    else:
        instances = segmentation.watershed(
            barrier_with_bg, markers, mask=plant_bool
        ).astype(np.int32)
    if params.min_inst_size > 0:
        for inst_id in np.unique(instances):
            if inst_id == 0:
                continue
            if (instances == inst_id).sum() < params.min_inst_size:
                instances[instances == inst_id] = 0

    if instances.max() > 0:
        kept = sorted(int(v) for v in np.unique(instances) if v != 0)
        remap = {old: new for new, old in enumerate(kept, start=1)}
        out = np.zeros_like(instances)
        for old, new in remap.items():
            out[instances == old] = new
        instances = out

    if not return_intermediates:
        return instances

    steps = {
        "1_input_image":   image_rgb,
        "2_plant_mask":    plant_mask,
        "3_grayscale":     gray,
        "4_gradient":      grad_norm,
        "5_canny":         canny_f,
        "6_shadows":       shadows_f,
        "7_barrier":       barrier_with_bg,
        "8_free_space":    free_space.astype(float),
        "9_distance":      dist_smooth,
        "10_markers":      markers,
        "11_instances":    instances,
    }
    return instances, steps


def _normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(float)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def instances_to_predictions(*args, **kwargs):
    from final_project.models.traditional import instances_to_predictions as _f
    return _f(*args, **kwargs)
