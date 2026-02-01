import os
import copy

import mesa
import numpy as np
import pandas as pd
import math
from enum import Enum, IntEnum, auto

from params import Trait
from maps.common_config import DEFAULT_ROAD_WIDTH, TargetParams
import nav_targets

class RouteState(Enum):
    NORMAL = auto() #通常：停滞したら再探索してよい
    BLOCKED_WAIT = auto() #不通道路を発見：再探索しない
    KNOWN = auto() #　不通道路をすでに知っている：再探索を許可
    INFORM = auto() #　不通道路情報を他の人に与える


class InfoShareMode(Enum):
    NO_SHARE = auto()
    SHARE_BLOCKED_ROAD = auto()


class BlockInfoState(IntEnum):
    UNKNOWN = 0
    PENDING = 1
    KNOWN = 2

class Turn(Enum):
    STRAIGHT = auto()
    RIGHT = auto()
    LEFT = auto()

DEFAULT_TARGET_PARAMS = TargetParams()

def axis_dir(a, b):
    delta = np.array(b) - np.array(a)
    dx, dy = float(delta[0]), float(delta[1])
    if abs(dx) >= abs(dy) and abs(dx) > 0.0:
        return np.array([math.copysign(1.0, dx), 0.0])
    if abs(dy) > 0.0:
        return np.array([0.0, math.copysign(1.0, dy)])
    return np.array([0.0, 0.0])

def get_turn(dir_in, dir_out):
    if np.allclose(dir_in, 0.0) or np.allclose(dir_out, 0.0):
        return Turn.STRAIGHT
    if np.allclose(dir_in, dir_out) or np.allclose(dir_in, -dir_out):
        return Turn.STRAIGHT
    cross = dir_in[0] * dir_out[1] - dir_in[1] * dir_out[0]
    if cross > 0:
        return Turn.RIGHT
    if cross < 0:
        return Turn.LEFT
    return Turn.STRAIGHT

def choose_inner_corner(cur, dir_in, dir_out, h):
    x_sign = dir_out[0] if abs(dir_out[0]) > 0.0 else -dir_in[0]
    y_sign = dir_out[1] if abs(dir_out[1]) > 0.0 else -dir_in[1]
    x = cur[0] + x_sign * h
    y = cur[1] + y_sign * h
    if x_sign > 0 and y_sign < 0:
        name = "NE"
    elif x_sign < 0 and y_sign < 0:
        name = "NW"
    elif x_sign > 0 and y_sign > 0:
        name = "SE"
    else:
        name = "SW"
    return name, np.array([x, y], dtype=float)

def slope_from_corner(corner_name):
    if corner_name in {"NE", "SW"}:
        return -1.0
    return 1.0

def midline_s(pos, cur, slope):
    dx = float(pos[0]) - float(cur[0])
    dy = float(pos[1]) - float(cur[1])
    if slope > 0:
        return dy - dx
    return dy + dx

def crossed_midline(pos, cur, dir_in, h, slope):
    p_entry = np.array(cur) - np.array(dir_in) * h
    s_entry = midline_s(p_entry, cur, slope)
    s_pos = midline_s(pos, cur, slope)
    return s_pos * s_entry <= 0.0

def crossed_entry_line(pos, cur, dir_in, h):
    p_entry = np.array(cur, dtype=float) - np.array(dir_in, dtype=float) * h
    return np.dot(np.array(pos, dtype=float) - p_entry, np.array(dir_in, dtype=float)) >= 0.0

def is_in_corner_area(pos, cur, half_width, margin):
    dx = abs(float(pos[0]) - float(cur[0]))
    dy = abs(float(pos[1]) - float(cur[1]))
    limit = half_width + margin
    return dx <= limit and dy <= limit

def midline_unit_from_corner(cur, corner, slope):
    sqrt2_inv = 1.0 / math.sqrt(2.0)
    if slope > 0:
        candidates = [np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
    else:
        candidates = [np.array([1.0, -1.0]), np.array([-1.0, 1.0])]
    to_center = np.array(cur, dtype=float) - np.array(corner, dtype=float)
    best = candidates[0]
    if np.dot(candidates[1], to_center) > np.dot(candidates[0], to_center):
        best = candidates[1]
    return best * sqrt2_inv

def clip_to_bounds(pos, space, margin):
    x_min = getattr(space, "x_min", 0.0)
    y_min = getattr(space, "y_min", 0.0)
    x_max = getattr(space, "x_max", None)
    y_max = getattr(space, "y_max", None)
    if x_max is None:
        x_max = space.width
    if y_max is None:
        y_max = space.height
    clipped = np.array(pos, dtype=float)
    clipped[0] = np.clip(clipped[0], x_min + margin, x_max - margin)
    clipped[1] = np.clip(clipped[1], y_min + margin, y_max - margin)
    return clipped

def push_target_to_far_side(target, cur, dir_in, h, slope, delta, max_steps, space, clip_margin):
    p_entry = np.array(cur) - np.array(dir_in) * h
    s_entry = midline_s(p_entry, cur, slope)
    s_target = midline_s(target, cur, slope)
    if s_target * s_entry <= 0.0:
        return clip_to_bounds(target, space, clip_margin)
    candidate = np.array(target, dtype=float)
    step = delta
    for _ in range(max_steps):
        candidate = candidate + step * np.array(dir_in)
        if midline_s(candidate, cur, slope) * s_entry <= 0.0:
            break
        step *= 1.5
    return clip_to_bounds(candidate, space, clip_margin)

class SharedParams:
    "_shared: Common human-related parameters shared between Human and ForcefulHuman instances."
    def __init__(self, in_dest_d, vision, dt):
        self.in_dest_d = in_dest_d
        self.vision = vision
        self.dt = dt


class Human(mesa.Agent):
    STUCK_WINDOW = 10
    STUCK_DIST = 0.2
    REROUTE_COOLDOWN = 30  # 再探索後、30ステップは再探索しない
    NODE_NEAR = 1.0

    def __init__(self, unique_id, model,
                 pos, velocity,
                 tmp_div, shared,
                 space, add_file_name,
                 re_route_state=RouteState.NORMAL,
                 route_idx=0, 
                 tmp_pos=(0., 0.), pos_array=[],
                 in_goal=False,elapsed_time=0.,  # 経過時間
                 forceful_initial=False,
                 can_become_forceful=False,
                 is_forceful=None,
                 has_phone=False,
                 ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.velocity = velocity
        self._shared = shared
        self.init_pos = self.pos.copy()
        self.forceful_initial = forceful_initial
        self.can_become_forceful = can_become_forceful
        self.forceful_trait = forceful_initial
        self.is_forceful = forceful_initial if is_forceful is None else is_forceful
        self.role = "forceful" if self.is_forceful else "normal"
        self.tmp_div = tmp_div #特定の人同士の反発力の大きさを除算もしくは乗算する値
        self.space = space #エージェントが動き回る空間を管理するモジュール
        self.add_file_name = add_file_name #保存するファイル名(の基礎.最終的には絶対パスまたは相対パスができる)
        #####################
        self.re_route_state = re_route_state #再探索の状態を保存する変数(selected_first_subgoalで呼び出すので先に定義しておく必要がある)
        self.rng = model.make_agent_rng(unique_id) #(将来的に)ランダムな要素を入れるためかもしれないため設定
        self.block_info_state = BlockInfoState.UNKNOWN
        self.set_up_initial_route() #最初の目的地と経路を選択
        self.route_idx = route_idx #経路のインデックス
        self.tmp_pos = np.array((0., 0.)) #一時的に計算した結果の位置を保存する値(将来的には壁を乗り越えるなどのありえない挙動をした時に元の位置に戻すために一旦計算した位置を保存している)
        self.pos_array = [] #自分の位置をステップごとに記録する配列
        self.in_goal = in_goal #目的地に到着したか判定するboolean型の変数
        self.pos_array.append(self.pos)
        self.elapsed_time = elapsed_time #経過時間
        self.last_reroute_time = -10**9  # 最後に再探索した時間を保存する変数
        self.target_pos = None
        self._entered_gate = False
        self._needs_reroute_from_share = False
        self._corner_node_id = None
        self._corner_target = None
        self._corner_in_area = False
        self._corner_congested = False
        self._corner_mode = None
        self.speed_scale = 1.0
        self.congested_state = False
        self.has_phone = bool(has_phone)
        self.known_dead_edges = set()
        self.last_advice_version = -1
        self.pending_route = None
        ######################

    @property
    def hspecs(self):
        return self.model.agent_params_by_trait[self.current_trait]

    @property
    def current_trait(self):
        return Trait.FORCEFUL if self.is_forceful else Trait.NORMAL
    
    @property
    def cur_dest(self):
        return self.model.dests[self.route[self.route_idx]]

    def get_target_pos(self):
        return self.aim_pos if self.aim_pos is not None else self.cur_dest

    def _target_params(self) -> TargetParams:
        config = getattr(self.model, "config", None)
        if config is None or getattr(config, "target_params", None) is None:
            return DEFAULT_TARGET_PARAMS
        return config.target_params

    def _road_width_at(self, pos) -> float:
        config = getattr(self.model, "config", None)
        if config is None or not hasattr(config, "get_road_width"):
            return DEFAULT_ROAD_WIDTH
        return config.get_road_width(pos)

    def _half_width_at(self, pos) -> float:
        return self._road_width_at(pos) / 2.0

    # scatter-dest
    def update_aim_pos_from_route(self):
        node_pos = self.cur_dest
        self.update_target_pos_from_route()
        if getattr(self.model, "debug_scatter_dest", False):
            print(f"[scatter-dest] id={self.unique_id} idx={self.route_idx} "
                  f"node={self.route[self.route_idx]} node_pos={node_pos} "
                  f"aim={self.aim_pos} state={self.re_route_state.name}")
        turn_context = self._get_turn_context(self.route_idx)
        if turn_context is not None:
            cur, dir_in, slope, _ = turn_context
            target_params = self._target_params()
            half_width = self._half_width_at(cur)
            self.aim_pos = push_target_to_far_side(
                self.aim_pos, cur, dir_in,
                half_width, slope, target_params.push_delta, target_params.push_max_k,
                self.space, target_params.clip_margin)
        return self.aim_pos

    def update_target_pos_from_route(self):
        if len(self.route) == 1 or len(self.route) == self.route_idx + 1:
            cur_pos = self.cur_dest
            if self._dir_in0 is None:
                self._dir_in0 = axis_dir(self.pos, cur_pos)
            elif len(self.route) == self.route_idx + 1:
                self._dir_in0 = axis_dir(self.pos, cur_pos)
            road_width = self._road_width_at(cur_pos)
            self.aim_pos = nav_targets.compute_straight_target(
                self.pos, cur_pos, self._dir_in0, road_width=road_width)
            return self.aim_pos
        prev_pos = None
        if self.route_idx > 0:
            prev_pos = self.model.dests[self.route[self.route_idx - 1]]
        cur_pos = self.cur_dest
        next_pos = None
        if self.route_idx + 1 < len(self.route):
            next_pos = self.model.dests[self.route[self.route_idx + 1]]
        road_width = self._road_width_at(cur_pos)
        self.aim_pos = nav_targets.compute_target_pos(
            self.pos, prev_pos, cur_pos, next_pos,
            self._shared.in_dest_d, road_width=road_width)
        turn_context = self._get_turn_context(self.route_idx)
        if turn_context is not None:
            cur, dir_in, slope, _ = turn_context
            target_params = self._target_params()
            half_width = self._half_width_at(cur)
            self.aim_pos = push_target_to_far_side(
                self.aim_pos, cur, dir_in,
                half_width, slope, target_params.push_delta, target_params.push_max_k,
                self.space, target_params.clip_margin)
        return self.aim_pos
    
    def _get_turn_context(self, route_idx):
        detail = self._get_turn_detail(route_idx)
        if detail is None:
            return None
        _, cur, _, dir_in, _, slope, corner_name, _ = detail
        return cur, dir_in, slope, corner_name

    def _get_turn_detail(self, route_idx):
        if route_idx + 1 >= len(self.route):
            return None
        cur = np.array(self.model.dests[self.route[route_idx]], dtype=float)
        nxt = np.array(self.model.dests[self.route[route_idx + 1]], dtype=float)
        if route_idx <= 0:
            if self._dir_in0 is None:
                dir_in = axis_dir(self.init_pos, cur)
                if np.allclose(dir_in, 0.0):
                    dir_in = axis_dir(cur, nxt)
                self._dir_in0 = dir_in
            dir_in = self._dir_in0
            prev = np.array(cur, dtype=float)
        else:
            self._dir_in0 = None
            prev = np.array(self.model.dests[self.route[route_idx - 1]], dtype=float)
            dir_in = axis_dir(prev, cur)
        dir_out = axis_dir(cur, nxt)
        turn = get_turn(dir_in, dir_out)
        if turn == Turn.STRAIGHT:
            return None
        half_width = self._half_width_at(cur)
        corner_name, corner_pos = choose_inner_corner(cur, dir_in, dir_out, half_width)
        slope = slope_from_corner(corner_name)
        return prev, cur, nxt, dir_in, dir_out, slope, corner_name, corner_pos

    def _get_dir_in(self, route_idx):
        if route_idx + 1 >= len(self.route):
            return None
        cur = np.array(self.model.dests[self.route[route_idx]], dtype=float)
        if route_idx <= 0:
            if self._dir_in0 is None:
                dir_in = axis_dir(self.init_pos, cur)
                if np.allclose(dir_in, 0.0):
                    nxt = np.array(self.model.dests[self.route[route_idx + 1]], dtype=float)
                    dir_in = axis_dir(cur, nxt)
                self._dir_in0 = dir_in
            return self._dir_in0
        self._dir_in0 = None
        prev = np.array(self.model.dests[self.route[route_idx - 1]], dtype=float)
        return axis_dir(prev, cur)

    def _reset_corner_state(self):
        self._corner_node_id = None
        self._corner_target = None
        self._corner_in_area = False
        self._corner_congested = False
        self._corner_mode = None
        self.congested_state = False

    def _update_corner_congestion(self, count):
        target_params = self._target_params()
        if self.congested_state:
            if count <= target_params.congestion_off:
                self.congested_state = False
        else:
            if count >= target_params.congestion_on:
                self.congested_state = True
        self._corner_congested = self.congested_state
        return self.congested_state

    def _build_crowded_corner_target(self, cur, corner_pos, dir_in, slope):
        target_params = self._target_params()
        u_mid = midline_unit_from_corner(cur, corner_pos, slope)
        base = np.array(corner_pos, dtype=float) + target_params.r_offset * u_mid
        half_width = self._half_width_at(cur)
        return push_target_to_far_side(
            base, cur, dir_in,
            half_width, slope, target_params.crowd_push_delta, target_params.push_max_k,
            self.space, target_params.clip_margin)

    def _update_corner_target_if_needed(self):
        detail = self._get_turn_detail(self.route_idx)
        if detail is None:
            self._reset_corner_state()
            return None
        prev, cur, nxt, dir_in, _, slope, _, corner_pos = detail
        cur_node_id = self.route[self.route_idx]
        if self._corner_node_id != cur_node_id:
            self._corner_congested = False
            self._corner_in_area = False
            self._corner_target = None
            self._corner_node_id = cur_node_id
        target_params = self._target_params()
        half_width = self._half_width_at(cur)
        in_corner = is_in_corner_area(self.pos, cur, half_width, target_params.corner_margin)
        if not in_corner:
            self._corner_in_area = False
            return None
        if self._corner_in_area and self._corner_target is not None:
            return None
        self._corner_in_area = True
        count = self.model.corner_counts.get(cur_node_id, 0)
        congested = self._update_corner_congestion(count)
        if congested:
            target = self._build_crowded_corner_target(cur, corner_pos, dir_in, slope)
            self._corner_mode = "crowded"
        else:
            road_width = self._road_width_at(cur)
            target = nav_targets.compute_target_pos(
                self.pos, prev, cur, nxt,
                self._shared.in_dest_d, road_width=road_width)
            target = push_target_to_far_side(
                target, cur, dir_in,
                half_width, slope, target_params.push_delta, target_params.push_max_k,
                self.space, target_params.clip_margin)
            self._corner_mode = "normal"
        self._corner_target = target
        self.aim_pos = target
        return None

    def _update_corner_speed_scale(self):
        detail = self._get_turn_detail(self.route_idx)
        target_params = self._target_params()
        if detail is None:
            self.congested_state = False
            target = 1.0
        else:
            _, cur, _, _, _, _, _, _ = detail
            cur_node_id = self.route[self.route_idx]
            half_width = self._half_width_at(cur)
            in_corner = is_in_corner_area(self.pos, cur, half_width, target_params.corner_margin)
            if in_corner:
                count = self.model.corner_counts.get(cur_node_id, 0)
                congested = self._update_corner_congestion(count)
                target = target_params.speed_scale if congested else 1.0
            else:
                self.congested_state = False
                self._corner_congested = False
                target = 1.0
        self.speed_scale = (1.0 - target_params.speed_alpha) * self.speed_scale + target_params.speed_alpha * target
        return None

    def set_up_initial_route(self):
        self.route, self.dest = self.model.select_first_subgoal(self)
        self.route_idx = 0
        self.init_pos = self.pos.copy()
        self._dir_in0 = None
        self.update_target_pos_from_route()
        return None
    
    def step(self):  # 次の位置を特定するための計算式を書く
        if self.pending_route is not None:
            self.route = self.pending_route
            self.dest = self.route[0]
            self.route_idx = 0
            self.init_pos = self.pos.copy()
            self._dir_in0 = None
            self.update_target_pos_from_route()
            self.pending_route = None
        if self._needs_reroute_from_share:
            self.route, self.dest = self.model.select_first_subgoal(self)
            self.route_idx = 0
            self.init_pos = self.pos.copy()
            self._dir_in0 = None
            self.update_target_pos_from_route()
            self.last_reroute_time = self.elapsed_time
            self._needs_reroute_from_share = False
        self._update_corner_target_if_needed()
        self._update_corner_speed_scale()
        self._calculate()
        if self.is_forceful:
            dest_dis = self.space.get_distance(self.pos, self.get_target_pos())
        else:
            if self.route[self.route_idx] == self.model.goal_arr[0]: #最終goalなら
                dest_dis = self.space.get_distance(self.pos, self.cur_dest)
            else:
                dest_dis = self.space.get_distance(self.pos, self.get_target_pos())
        self.goal_check(dest_dis)
        self.tmp_pos[0] = self.pos[0] + \
            self.velocity[0] * self._shared.dt  # 仮の位置を計算
        self.tmp_pos[1] = self.pos[1] + self.velocity[1] * self._shared.dt
        return None

    def queue_route_advice(self, route, advice_version):
        if route is None:
            return False
        if advice_version == self.last_advice_version:
            return False
        self.pending_route = route
        self.last_advice_version = advice_version
        return True

    def advance(self):
        self.pos = copy.deepcopy(self.tmp_pos)
        self.pos_array.append(self.pos)
        self.elapsed_time += self._shared.dt
        self.re_route()
        if (self.in_goal):  # goalした場合
            self.model.mark_goal_reached(self)
            path = self.add_file_name
            self.make_dir(path)
            self.write_record(path)
            self.model.space.remove_agent(self)
            self.model.schedule.remove(self)
            return None
        else:
            self.model.space.move_agent(self, self.pos)  # goalしていない場合
        return None

    def _goal_region_contains(self):
        goals = getattr(self.model, "goals", None)
        if not goals:
            return False
        goal = goals.get(self.role)
        if goal is None:
            return False
        return goal.contains(self.pos)
    
    def goal_check(self, dest_dis):
        if self._goal_region_contains():
            self.in_goal = True
            self.velocity = [0.0, 0.0]
            return None
        if len(self.route) == 1:
            return None
        turn_context = self._get_turn_context(self.route_idx)
        if turn_context is not None:
            cur, dir_in, slope, _ = turn_context
            half_width = self._half_width_at(cur)
            if crossed_midline(self.pos, cur, dir_in, half_width, slope):
                if len(self.route) == self.route_idx + 1:
                    self.in_goal = True
                    self.velocity = [0.0, 0.0]
                else:
                    self.route_idx += 1
                    self._reset_corner_state()
                    self.update_aim_pos_from_route() # scatter-dest               
                return None
        elif len(self.route) > 1:
            cur = self.cur_dest
            dir_in = self._get_dir_in(self.route_idx)
            if dir_in is not None:
                half_width = self._half_width_at(cur)
                if crossed_entry_line(self.pos, cur, dir_in, half_width):
                    if len(self.route) == self.route_idx + 1:
                        self.in_goal = True
                        self.velocity = [0.0, 0.0]
                    else:
                        self.route_idx += 1
                        self._reset_corner_state()
                        self.update_target_pos_from_route()
                    return None
        if turn_context is not None and dest_dis < 1.5: #以前の処理
            if len(self.route) == self.route_idx + 1:
                self.in_goal = True
                self.velocity = [0.0, 0.0]
            else:
                self.route_idx += 1
                self._reset_corner_state()
                self.update_target_pos_from_route()
            return None

    def re_route(self):
        WINDOW = self.STUCK_WINDOW
        D_MIN = self.STUCK_DIST
        COOLDOWN = self.REROUTE_COOLDOWN

        if self.re_route_state == RouteState.NORMAL:
            learned = self.maybe_learn_blocked()
            if learned:
                self.last_reroute_time = self.elapsed_time
                return None

        # クールダウン中なら何もしない
        if self.elapsed_time - self.last_reroute_time < COOLDOWN * self._shared.dt:
            return None
        
        pos_num = len(self.pos_array)
        if pos_num < WINDOW + 1:
            return None
        
        cur_pos = self.pos_array[-1]
        past_pos = self.pos_array[-(WINDOW + 1)]
        moved = math.dist(cur_pos, past_pos)
        if moved < D_MIN:
            self.route, self.dest = self.model.select_first_subgoal(self)
            self.route_idx = 0
            self.init_pos = self.pos.copy()
            self._dir_in0 = None
            self.update_target_pos_from_route()
            self.last_reroute_time = self.elapsed_time
        return None


    def maybe_learn_blocked(self):
        DETECT_R = 2.0  # 不通壁から何m以内で「気づいた」とみなすか（要調整）

        for i in range(len(self.model.dead_wall_ab)):
            dist, _ = self.dead_distance_point_to_segment(i)
            if dist < DETECT_R:
                dead_edge_id = self.model.get_dead_edge_id(i)
                if dead_edge_id is not None:
                    self.known_dead_edges.add(int(dead_edge_id))
                self.re_route_state = RouteState.KNOWN
                self.block_info_state = BlockInfoState.KNOWN
                # 不通を考慮した距離木で再ルート
                self.route, self.dest = self.model.select_first_subgoal(self)
                self.route_idx = 0
                self.init_pos = self.pos.copy()
                self._dir_in0 = None
                self.update_target_pos_from_route()
                return True
        return False

    def apply_shared_block_info(self, shared_info):
        if not shared_info:
            return False
        before = len(self.known_dead_edges)
        self.known_dead_edges.update(shared_info)
        if len(self.known_dead_edges) == before:
            return False
        self.block_info_state = BlockInfoState.KNOWN
        if self.re_route_state == RouteState.NORMAL:
            self.re_route_state = RouteState.KNOWN
        self._needs_reroute_from_share = True
        return True

    def share_block_info(self):
        if self.model.info_share_mode != InfoShareMode.SHARE_BLOCKED_ROAD:
            return None
        if self.block_info_state != BlockInfoState.KNOWN:
            return None
        neighbors = self.model.space.get_neighbors(self.pos, 1.5, False)
        for neighbor in neighbors:
            if self.unique_id == neighbor.unique_id:
                continue
            if isinstance(neighbor, Human):
                if neighbor.block_info_state == BlockInfoState.UNKNOWN:
                    neighbor.block_info_state = BlockInfoState.PENDING
        return None

    def update_block_info_state(self):
        if self.model.info_share_mode != InfoShareMode.SHARE_BLOCKED_ROAD:
            return None
        if self.block_info_state == BlockInfoState.PENDING:
            self.block_info_state = BlockInfoState.KNOWN
            if self.re_route_state == RouteState.NORMAL:
                self.re_route_state = RouteState.KNOWN
            self._needs_reroute_from_share = True

    def make_dir(self, path):
        os.makedirs(f"{path}/Data", exist_ok=True)
        if self.model.csv_plot:
            os.makedirs(f"{path}/csv", exist_ok=True)
        return None

    def write_record(self, path):
        label = "forceful" if self.is_forceful else "normal"
        if self.model.csv_plot:
            np.savetxt(f"{path}/csv/id{self.unique_id}_{label}"
                       f".csv", self.pos_array, delimiter=",")
        if self.in_goal:
            with open(f"{self.add_file_name}/Data/"
                      f"{label}.dat", "a") as f:
                f.write(f"{self.elapsed_time} \n")
            if self.is_forceful:
                path = path.replace(f"/seed_{self.model.seed}", "")
                df = pd.DataFrame({"m": [self.hspecs.m], "nol_pop": [self.model.population], "seed": [
                    self.model.seed], "id": [self.unique_id], "elapsed_time": [self.elapsed_time]})
                df.to_csv(f"{path}/forceful_time.csv",
                          mode="a", header=False, index = False)

    def _sincos(self, x2):
        r_0 = np.sqrt((x2[0] - self.pos[0]) ** 2 + (x2[1] - self.pos[1]) ** 2)
        sin = (x2[1] - self.pos[1]) / r_0
        cos = (x2[0] - self.pos[0]) / r_0
        return cos, sin

    def _force(self, dest):
        fx, fy = 0., 0.
        theta = self._sincos(dest)
        neighbors = self.model.space.get_neighbors(
            self.pos, self._shared.vision, False)
        fx, fy = self.force_from_goal(theta)
        for neighbor in neighbors:
            if self.unique_id == neighbor.unique_id:
                continue
            if isinstance(neighbor, Human):
                tmp_fx, tmp_fy = self.force_from_human(neighbor)
                fx += tmp_fx
                fy += tmp_fy
            elif type(neighbor) is Wall:
                None
        tmp_fx, tmp_fy = 0., 0.
        tmp_fx, tmp_fy = self.force_from_wall()
        fx += tmp_fx
        fy += tmp_fy
        fx /= self.hspecs.m
        fy /= self.hspecs.m
        return fx, fy

    def force_from_goal(self, theta):
        v0_eff = self.hspecs.v0 * self.speed_scale
        fx = self.hspecs.m * (v0_eff * theta[0] - self.velocity[0]) / self.hspecs.tau
        fy = self.hspecs.m * (v0_eff * theta[1] - self.velocity[1]) / self.hspecs.tau
        return fx, fy

    def force_from_human(self, neighbor):
        fx, fy = 0., 0.
        n_ij = (self.pos - neighbor.pos) / \
            self.space.get_distance(self.pos, neighbor.pos)
        t_ij = [-n_ij[1], n_ij[0]]
        pair_params = self.model.get_pair_params(self.current_trait, neighbor.current_trait)
        dis = (self.hspecs.r + neighbor.hspecs.r) * pair_params.r_scale - \
            self.space.get_distance(self.pos, neighbor.pos)
        if dis >= 0:
            fx += (pair_params.a * (math.e ** (dis / pair_params.b)) + pair_params.k * dis) * \
                n_ij[0] + pair_params.kappa * dis * \
                np.dot(
                (neighbor.velocity - self.velocity), t_ij)*t_ij[0]
            fy += (pair_params.a * (math.e ** (dis / pair_params.b)) + pair_params.k * dis) * \
                n_ij[1] + pair_params.kappa * dis * \
                np.dot(
                    (neighbor.velocity - self.velocity), t_ij)*t_ij[1]
        else:
            fx += pair_params.a * (math.e **
                                     (dis / pair_params.b)) * n_ij[0]
            fy += pair_params.a * (math.e **
                                     (dis / pair_params.b)) * n_ij[1]
        return fx, fy

    def force_from_wall(self):
        fx, fy = 0., 0.
        for i in range(len(self.model.wall_ab)):
            dis, n_iw = self.distance_point_to_segment(i)
            if dis < self._shared.vision:
                t_iw = np.array([-n_iw[1], n_iw[0]])
                tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
                fx += tmp_fx
                fy += tmp_fy
        if (self.re_route_state == RouteState.BLOCKED_WAIT or
            self.re_route_state == RouteState.KNOWN):
            for j in range(len(self.model.dead_wall_ab)):
                dis, n_iw = self.dead_distance_point_to_segment(j)
                if dis < self._shared.vision:
                    t_iw = np.array([-n_iw[1], n_iw[0]])
                    tmp_fx, tmp_fy = self.wall_force_core(dis, n_iw, t_iw)
                    fx += tmp_fx
                    fy += tmp_fy
        return fx, fy

    def distance_point_to_segment(self, i):
        a = self.model.wall_a[i][:2]
        ab = self.model.wall_ab[i][:2]
        ap = self.pos - a
        ab_len2 = self.model.wall_ab_len2[i]

        if ab_len2 == 0: #壁の両端の座標が同じ場合
            vec = self.pos - a
            dis = np.linalg.norm(vec)
            n_iw = vec / dis if dis > 1e-8 else np.array([0., 0.])
            return dis, n_iw
        
        t = np.dot(ap, ab) / ab_len2 
        if t < 0.0:
            closest = self.model.wall_a[i][:2]
        elif t > 1.0:
            closest = self.model.wall_b[i][:2]
        else:
            closest = self.model.wall_a[i][:2] + t * ab

        vec = self.pos - closest
        dis = np.linalg.norm(vec)
        if dis > 1e-8:
            n_iw = vec / dis
        else:
            n_iw = np.array([0., 0.])
        return dis, n_iw

    def dead_distance_point_to_segment(self, j):
        a = self.model.dead_wall_a[j][:2]
        ab = self.model.dead_wall_ab[j][:2]
        ap = self.pos - a
        ab_len2 = self.model.dead_wall_ab_len2[j]

        if ab_len2 == 0: #壁の両端の座標が同じ場合
            vec = self.pos - a
            dis = np.linalg.norm(vec)
            n_iw = vec / dis if dis > 1e-8 else np.array([0., 0.])
            return dis, n_iw
        
        t = np.dot(ap, ab) / ab_len2 
        if t < 0.0:
            closest = self.model.dead_wall_a[j][:2]
        elif t > 1.0:
            closest = self.model.dead_wall_b[j][:2]
        else:
            closest = self.model.dead_wall_a[j][:2] + t * ab

        vec = self.pos - closest
        dis = np.linalg.norm(vec)
        if dis > 1e-8:
            n_iw = vec / dis
        else:
            n_iw = np.array([0., 0.])
        return dis, n_iw
    
    def wall_force_core(self, dis, n_iw, t_iw):
        fx, fy = 0., 0.
        if dis >= 0:
            fx += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[0] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e ** (dis / self.hspecs.repul_m[1])) + self.hspecs.k *
                    dis) * n_iw[1] - self.hspecs.kappa * dis * np.dot(self.velocity, t_iw) * t_iw[1]
        else:
            fx += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[0]
            fy += (self.hspecs.repul_m[0] * (math.e **
                    (dis / self.hspecs.repul_m[1]))) * n_iw[1]
        return fx, fy
    
    def _calculate(self):
        fx, fy = self._force(self.get_target_pos())
        self.velocity[0] += fx * self._shared.dt
        self.velocity[1] += fy * self._shared.dt
        if (np.linalg.norm(self.velocity, 2) > 1.):  # review
            v = copy.deepcopy(self.velocity)
            vn = np.linalg.norm(v)
            self.velocity = v / vn
        return None
    
class Obstacle(mesa.Agent):
    def __init__(self, unique_id, model, pos, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.dir = dir

    def step(self):
        return None


class Wall(mesa.Agent):
    def __init__(self, unique_id, model, pos, wall_r, dir):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.wall_r = wall_r
        self.dir = dir

    def step(self):
        return None


class Goal(mesa.Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)

    def step(self):
        return None
