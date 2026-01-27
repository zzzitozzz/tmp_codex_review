from __future__ import annotations

from .common_config import MapConfig, Rect, TargetParams

GOALS = {
    "normal": Rect(150.0, 156.0, 38.0, 44.0),
    "forceful": Rect(150.0, 156.0, 38.0, 44.0),
}

SPAWN = {
    "normal": Rect(2.0, 156.0, 2.0, 100.0),
    "forceful": Rect(16.0, 22.0, 29.5, 35.5),
}

TARGET_PARAMS = TargetParams(
    r_offset=0.6,
    corner_margin=0.6,
    push_delta=0.8,
    push_max_k=5,
    crowd_push_delta=0.5,
    clip_margin=0.2,
    congestion_on=8,
    congestion_off=5,
    speed_scale=0.5,
    speed_alpha=0.2,
)

INTERSECTION_MAP_CONFIG = MapConfig(
    goals=GOALS,
    spawn=SPAWN,
    target_params=TARGET_PARAMS,
    road_width_regions=(),
    default_road_width=6.0,
)
