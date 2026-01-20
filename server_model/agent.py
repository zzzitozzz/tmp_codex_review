import os
import copy

import mesa
import numpy as np
import pandas as pd
import math
from enum import Enum, IntEnum, auto

from params import Trait
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

ROAD_WIDTH = 6.0
ROAD_HALF_WIDTH = ROAD_WIDTH / 2.0
DELTA_GATE = 0.8
MAX_GATE_PUSH = 5
CLIP_MARGIN = 0.2

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

def push_target_to_far_side(target, cur, dir_in, h, slope, delta, max_steps, space):
    p_entry = np.array(cur) - np.array(dir_in) * h
    s_entry = midline_s(p_entry, cur, slope)
    s_target = midline_s(target, cur, slope)
    if s_target * s_entry <= 0.0:
        return clip_to_bounds(target, space, CLIP_MARGIN)
    candidate = np.array(target, dtype=float)
    step = delta
    for _ in range(max_steps):
        candidate = candidate + step * np.array(dir_in)
        if midline_s(candidate, cur, slope) * s_entry <= 0.0:
            break
        step *= 1.5
    return clip_to_bounds(candidate, space, CLIP_MARGIN)

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
                 ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.velocity = velocity
        self._shared = shared
        self.forceful_initial = forceful_initial
        self.can_become_forceful = can_become_forceful
        self.forceful_trait = forceful_initial
        self.is_forceful = forceful_initial if is_forceful is None else is_forceful
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

    def get_desired_direction(self):
        target = self.get_target_pos()
        if target is None or not np.all(np.isfinite(target)):
            target = self.cur_dest
        if target is None or not np.all(np.isfinite(target)):
            return np.array([0., 0.])
        vec = np.array(target, dtype=float) - self.pos
        norm = np.linalg.norm(vec)
        if norm < 1e-8:
            return np.array([0., 0.])
        return vec / norm

    # scatter-dest
    def update_aim_pos_from_route(self):
        node_pos = self.cur_dest
        walls_for_los = self.model.get_walls_for_los(self.re_route_state)
        self.aim_pos = self.model.generate_scatter_destination(
            self.pos, node_pos, walls_for_los, rng=self.rng)
        if getattr(self.model, "debug_scatter_dest", False):
            print(f"[scatter-dest] id={self.unique_id} idx={self.route_idx} "
                  f"node={self.route[self.route_idx]} node_pos={node_pos} "
                  f"aim={self.aim_pos} state={self.re_route_state.name}")
        turn_context = self._get_turn_context(self.route_idx)
        if turn_context is not None:
            cur, dir_in, slope, _ = turn_context
            self.aim_pos = push_target_to_far_side(
                self.aim_pos, cur, dir_in,
                ROAD_HALF_WIDTH, slope, DELTA_GATE, MAX_GATE_PUSH, self.space)
        return self.aim_pos
    
    def _get_turn_context(self, route_idx):
        if route_idx <= 0 or route_idx + 1 >= len(self.route):
            return None
        prev = self.model.dests[self.route[route_idx - 1]]
        cur = self.model.dests[self.route[route_idx]]
        nxt = self.model.dests[self.route[route_idx + 1]]
        dir_in = axis_dir(prev, cur)
        dir_out = axis_dir(cur, nxt)
        turn = get_turn(dir_in, dir_out)
        if turn == Turn.STRAIGHT:
            return None
        corner_name, _ = choose_inner_corner(np.array(cur, dtype=float), dir_in, dir_out, ROAD_HALF_WIDTH)
        slope = slope_from_corner(corner_name)
        return np.array(cur, dtype=float), dir_in, slope, corner_name

    def set_up_initial_route(self):
        self.route, self.dest = self.model.select_first_subgoal(self)
        self.route_idx = 0
        self.update_target_pos_from_route()
        return None
    
    def step(self):  # 次の位置を特定するための計算式を書く
        if self._needs_reroute_from_share:
            self.route, self.dest = self.model.select_first_subgoal(self)
            self.route_idx = 0
            self.update_target_pos_from_route()
            self.last_reroute_time = self.elapsed_time
            self._needs_reroute_from_share = False
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
    
    def goal_check(self, dest_dis):
        turn_context = self._get_turn_context(self.route_idx)
        if turn_context is not None:
            cur, dir_in, slope, _ = turn_context
            if crossed_midline(self.pos, cur, dir_in, ROAD_HALF_WIDTH, slope):
                if len(self.route) == self.route_idx + 1:
                    self.in_goal = True
                    self.velocity = [0.0, 0.0]
                else:
                    self.route_idx += 1
                    self.update_aim_pos_from_route() # scatter-dest
                return None
        if dest_dis < 1.5:
            if len(self.route) == self.route_idx + 1:
                self.in_goal = True
                self.velocity = [0.0, 0.0]
            else:
                self.route_idx += 1
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
            self.update_target_pos_from_route()
            self.last_reroute_time = self.elapsed_time
        return None


    def maybe_learn_blocked(self):
        DETECT_R = 2.0  # 不通壁から何m以内で「気づいた」とみなすか（要調整）

        for i in range(len(self.model.dead_wall_ab)):
            dist, _ = self.dead_distance_point_to_segment(i)
            if dist < DETECT_R:
                self.re_route_state = RouteState.KNOWN
                self.block_info_state = BlockInfoState.KNOWN
                # 不通を考慮した距離木で再ルート
                self.route, self.dest = self.model.select_first_subgoal(self)
                self.route_idx = 0
                self.update_target_pos_from_route()
                return True
        return False

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
        fx = self.hspecs.m * (self.hspecs.v0 * theta[0] - self.velocity[0]) / self.hspecs.tau
        fy = self.hspecs.m * (self.hspecs.v0 * theta[1] - self.velocity[1]) / self.hspecs.tau
        return fx, fy

    def force_from_human(self, neighbor):
        fx, fy = 0., 0.
        dist = self.space.get_distance(self.pos, neighbor.pos)
        dist_safe = max(dist, 1e-8)
        n_ij = (self.pos - neighbor.pos) / dist_safe
        t_ij = [-n_ij[1], n_ij[0]]
        pair_params = self.model.get_pair_params(self.current_trait, neighbor.current_trait)
        dis = (self.hspecs.r + neighbor.hspecs.r) * pair_params.r_scale - dist
        desired_dir = self.get_desired_direction()
        if np.linalg.norm(desired_dir) > 0.0:
            cos_phi = float(np.clip(np.dot(desired_dir, -n_ij), -1.0, 1.0))
            lam = self.model.strategy.asfm_lambda
            # ASFM: weaken psychological repulsion from behind.
            weight = lam + (1.0 - lam) * (1.0 + cos_phi) / 2.0
        else:
            weight = 1.0
        repulsion = pair_params.a * (math.e ** (dis / pair_params.b))
        if dis >= 0:
            fx += (weight * repulsion + pair_params.k * dis) * \
                n_ij[0] + pair_params.kappa * dis * \
                np.dot(
                (neighbor.velocity - self.velocity), t_ij)*t_ij[0]
            fy += (weight * repulsion + pair_params.k * dis) * \
                n_ij[1] + pair_params.kappa * dis * \
                np.dot(
                    (neighbor.velocity - self.velocity), t_ij)*t_ij[1]
        else:
            fx += weight * repulsion * n_ij[0]
            fy += weight * repulsion * n_ij[1]
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
