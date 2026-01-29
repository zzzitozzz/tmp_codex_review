from __future__ import annotations

from .common_config import MapConfig, Rect, RoadWidthRegion, TargetParams

import numpy as np

GOALS = {
    "normal": Rect(52.5, 54.0, 26.0, 40.0),
    "forceful": Rect(16.0, 22.0, 4.0, 5.5),
}

SPAWN = {
    "normal": Rect(4.0, 34.0, 26.0, 40.0),
    # "normal": Rect(16.0, 22.0, 4.0, 26.0),
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

ROAD_WIDTH_REGIONS = [
    RoadWidthRegion(Rect(2.0, 54.0, 26.0, 40.0), road_width=14.0),
    RoadWidthRegion(Rect(16.0, 22.0, 4.0, 26.0), road_width=6.0),
]

CONVEX_MAP_CONFIG = MapConfig(
    goals=GOALS,
    spawn=SPAWN,
    target_params=TARGET_PARAMS,
    road_width_regions=ROAD_WIDTH_REGIONS,
    default_road_width=6.0,
    wall_arr=np.array([[[2., 40.], [54., 40.]],
                       [[2., 26.], [16., 26.]],
                       [[22., 26.], [54., 26.]],
                       [[16., 4.], [16., 26.]],
                       [[22., 4.], [22., 26.]]]),
    dead_wall_arr=np.array([[]]),
    dests=[[9, 33], [19, 28],  [19, 4], [54, 33]],
    edges={0: [1], 1: [0, 2, 3], 2:[1], 3: [1]},
    dead_edges=[],
    goal_arr=[3, 2],
)
