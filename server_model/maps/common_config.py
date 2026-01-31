from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


DEFAULT_ROAD_WIDTH = 6.0


@dataclass(frozen=True)
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains(self, pos) -> bool:
        x = float(pos[0])
        y = float(pos[1])
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class RoadWidthRegion:
    rect: Rect
    road_width: float


@dataclass(frozen=True)
class TargetParams:
    r_offset: float = 0.6
    corner_margin: float = 0.6
    push_delta: float = 0.8
    push_max_k: int = 5
    crowd_push_delta: float = 0.5
    clip_margin: float = 0.2
    congestion_on: int = 8
    congestion_off: int = 5
    speed_scale: float = 0.5
    speed_alpha: float = 0.2


@dataclass(frozen=True)
class MapConfig:
    goals: dict[str, Rect] = field(default_factory=dict)
    spawn: dict[str, Rect] = field(default_factory=dict)
    target_params: TargetParams = field(default_factory=TargetParams)
    road_width_regions: Iterable[RoadWidthRegion] = field(default_factory=tuple)
    default_road_width: float = DEFAULT_ROAD_WIDTH
    visual_follow_enabled: bool = True
    visual_follow_R_max: float = 5.0
    visual_follow_fov_deg: float = 120.0
    visual_follow_d0: float = 1.2
    visual_follow_p: float = 2.0
    visual_follow_alpha_max: float = 0.25
    visual_follow_update_every_steps: int = 2
    visual_follow_ema_beta: float = 0.2
    visual_follow_vis_block: str = "simple"
    visual_follow_same_dir_deg: float = 60.0
    wall_arr: object | None = None
    dead_wall_arr: object | None = None
    dests: list = field(default_factory=list)
    edges: dict = field(default_factory=dict)
    dead_edges: list = field(default_factory=list)
    goal_arr: list = field(default_factory=list)

    def get_road_width(self, pos) -> float:
        for region in self.road_width_regions:
            if region.rect.contains(pos):
                return region.road_width
        return self.default_road_width
