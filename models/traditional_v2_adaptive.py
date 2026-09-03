from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from final_project.models.traditional_v2 import (
    EdgeAwareParams,
    segment_leaves_edge_aware,
)

DEFAULT_AREA_THRESHOLD = 150_000


@dataclass
class AdaptiveParams:
    small_params: EdgeAwareParams = field(default_factory=lambda: EdgeAwareParams(
        sigma_dist=3.0, min_distance=20, shadow_percentile=25.0,
        w_gradient=0.42, w_canny=0.28, w_shadow=0.30,
        min_inst_size=100,
    ))
    large_params: EdgeAwareParams = field(default_factory=lambda: EdgeAwareParams(
        sigma_dist=5.0, min_distance=60, shadow_percentile=10.0,
        w_gradient=0.55, w_canny=0.30, w_shadow=0.15,
        min_inst_size=500,
    ))
    area_threshold: int = DEFAULT_AREA_THRESHOLD

def classify_tier(plant_mask: np.ndarray, area_threshold: int = DEFAULT_AREA_THRESHOLD) -> str:
    plant_area = int((plant_mask > 0).sum())
    return "small" if plant_area < area_threshold else "large"


def segment_leaves_adaptive(
    image_rgb: np.ndarray,
    plant_mask: np.ndarray,
    params: AdaptiveParams | None = None,
    return_tier: bool = False,
):
    if params is None:
        params = AdaptiveParams()
    tier = classify_tier(plant_mask, params.area_threshold)
    chosen = params.small_params if tier == "small" else params.large_params
    instances = segment_leaves_edge_aware(image_rgb, plant_mask, params=chosen)
    if return_tier:
        return instances, tier
    return instances
