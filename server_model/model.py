import mesa
import os
import sys
import warnings
import copy
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
import heapq
import math

from agent import (SharedParams, Human, Wall, RouteState, InfoShareMode,
                   BlockInfoState, is_in_corner_area)
from maps.common_config import DEFAULT_ROAD_WIDTH, TargetParams
from params import Trait, StrategyConfig, build_sfm_params
warnings.simplefilter('ignore', UserWarning)

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


class MoveAgent(mesa.Model):

    def __init__(
            self, population=100, for_population=1, dests=[], edges=[], dead_edges=[], goal_arr=[], v_arg=[], wall_arr=[[]],
            dead_wall_arr=None,seed=1, r=0.5,
            wall_r=0.5, human_var={}, forceful_human_var={},
            width=100, height=100, dt=0.1,
            in_dest_d=3, vision=3, time_step=0,
            add_file_name="", add_file_name_arr=[],
            len_sq=3., f_r=0.,pos_func= {},
            csv_plot=False,
            info_share_mode=InfoShareMode.NO_SHARE,
            phone_ratio=0.0,
            share_interval_sec=0.3,
            R_short=6.0,
            share_block_info=False,
            R_face=1.5,
            forceful_preset="baseline",
            strategy: StrategyConfig | None = None,
            goals=None,
            config=None):
        super().__init__()
        self.population = population
        self.for_population = for_population
        self.dests = dests
        self.edges = edges
        self.dead_edges = dead_edges
        self.goal_arr = goal_arr
        self.goals = goals or {}
        self.config = config
        self.v_arg = v_arg
        self.wall_arr = wall_arr
        self.dead_wall_arr = np.array([[]]) if dead_wall_arr is None else dead_wall_arr
        ####
        self.wall_a, self.wall_b, self.wall_ab, self.wall_ab_len2 = self.pre_wall_arr(self.wall_arr)
        self.dead_wall_a, self.dead_wall_b, self.dead_wall_ab, self.dead_wall_ab_len2 = self.pre_wall_arr(self.dead_wall_arr)
        ####
        self.seed = seed
        self.r = r
        self.wall_r = wall_r

        self.human_var = human_var
        self.forceful_human_var = forceful_human_var
        self.width = width
        self.height = height
        self.dt = dt
        self.in_dest_d = in_dest_d
        self.vision = vision
        self.time_step = time_step
        self.add_file_name_arr = add_file_name_arr
        self.len_sq = len_sq
        self.f_r = f_r
        ###
        self.pos_func = pos_func
        ###
        self.csv_plot = csv_plot
        self.info_share_mode = info_share_mode
        self.phone_ratio = float(phone_ratio)
        self.share_interval_sec = float(share_interval_sec)
        self.R_short = float(R_short)
        self.share_block_info = bool(share_block_info)
        self.R_face = float(R_face)
        self.share_every_steps = max(
            1, int(math.ceil(self.share_interval_sec / self.dt)))
        self.forceful_preset = forceful_preset
        self.strategy = strategy or StrategyConfig()
        self.max_steps = 1500
        self.log_capacity = self.max_steps + 1
        self.num_agents = self.population + self.for_population
        self.pos_log = np.zeros(
            (self.log_capacity, self.num_agents, 2), dtype=np.float32)
        self.state_log = np.zeros(
            (self.log_capacity, self.num_agents), dtype=np.int8)
        self.has_block_info_log = np.zeros(
            (self.log_capacity, self.num_agents), dtype=np.int8)
        self.known_blocks_log = np.zeros(
            (self.log_capacity, self.num_agents), dtype=np.int16)
        self.goal_reached_step = np.full(self.num_agents, -1, dtype=np.int32)
        self.corner_counts = {}
        self.congested_state_by_node = {}
        self.phase_exit_by_node = {}
        self.phase_timer_by_node = {}
        self.exit_counts_by_node = {}
        shared = SharedParams(self.in_dest_d, self.vision, self.dt)
        self.agent_params_by_trait, self.pair_params_table = build_sfm_params(
            self.human_var,
            self.forceful_human_var,
            self.r,
            self.f_r,
            self.forceful_preset,
            self.strategy,
        )
        self.dist_to_goal_normal = [] # dist_to_goal[i]: ノード i から避難所までの最短距離 (A* の g(n) に相当)
        self.next_to_goal_normal = [] # next_to_goal[i]: ノード i から避難所までの最短経路 (A* の f(n) に相当)
        self.dist_to_goal_blocked = [] # 不通道路版
        self.next_to_goal_blocked = [] # 不通道路版
        self.dir_parts()
        # self.schedule = mesa.time.RandomActivation(self) #すべてのエージェントをランダムに呼び出し、各エージェントでstep()を一回呼ぶ。step()だけで変更を適用する
        # すべてのエージェントを順番に呼び出し、すべてのエージェントで順番にstep()を一回読んだ後、すべてのエージェントで順番にadvance()を一回呼ぶ。step()で変更を準備し、advance()で変更を適用する
        self.schedule = mesa.time.SimultaneousActivation(self)
        self.space = mesa.space.ContinuousSpace(width, height, True)
        self.rng = np.random.default_rng(self.seed)
        self.make_agents(shared)
        self.log_initial_state()
        self.running = True
        print(f"change para: {self.check_f_parameter()}")
        self.make_basic_dir()
        self.save_specs_to_file(shared)

    def get_target_params(self) -> TargetParams:
        if self.config is None:
            return TargetParams()
        return self.config.target_params

    def get_road_width(self, pos) -> float:
        if self.config is None:
            return DEFAULT_ROAD_WIDTH
        return self.config.get_road_width(pos)

    def dir_parts(self):
        basic_file_name = f"{self.add_file_name_arr[0]}/nol_pop_{self.population}"
        self.add_file_name = f"{basic_file_name}/"
        self.add_file_name = self.add_file_name + "seed_" + str(self.seed)
        return None

    def get_pair_params(self, trait_i, trait_j):
        return self.pair_params_table.get(trait_i, trait_j)

    def make_basic_dir(self):
        path = f"{self.add_file_name}/Data/"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}normal.dat", "w") as f:
            f.write("evacuation_time\n")
        os.makedirs(path, exist_ok=True)
        with open(f"{path}forceful.dat", "w") as f:
            f.write("evacuation_time\n")
        print(f"{self.add_file_name=}")
        self.ini_force_dataframe()

    def save_specs_to_file(self, shared):
        path = f"{self.add_file_name}/../human_specs.yaml"
        if not os.path.exists(path):
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "run_time" : run_time,
                "shared_val" : vars(shared),
                "dt": self.dt,
                "phone_ratio": self.phone_ratio,
                "share_interval_sec": self.share_interval_sec,
                "share_every_steps": self.share_every_steps,
                "R_short": self.R_short,
                "share_block_info": self.share_block_info,
                "R_face": self.R_face,
                "forceful_preset": self.forceful_preset,
                "strategy": self.strategy.to_dict(),
                "agent_params": {
                    trait.name: vars(self.agent_params_by_trait[trait])
                    for trait in Trait
                },
                "pair_params": self.pair_params_table.as_dict(),
            }
            with open(f"{path}", "w") as f:
                yaml.dump(data, f, sort_keys=False)

    def ini_force_dataframe(self):
        tmp_path = self.add_file_name.replace(f"/seed_{self.seed}", "")
        if os.path.isfile(f"{tmp_path}/forceful_time.csv"):
            None
        else:
            df = pd.DataFrame(
                columns=["m", "nol_pop", "seed", "id", "evacuation_time"])
            df.to_csv(f"{tmp_path}/forceful_time.csv", index=False)

    def make_agents(self, shared):
        tmp_id = 0
        tmp_id = self.generate_human(tmp_id, shared)

    def generate_human(self, tmp_id, shared):
        tmp_div = 1.
        pos_array = []
        human_array = []
        tmp_forceful_num = self.for_population
        ###　(逆)ダイクストラ法 ###
        self.dist_to_goal_normal, self.next_to_goal_normal = self.dijkstra_backward(self.goal_arr[0])
        self.dist_to_goal_blocked, self.next_to_goal_blocked = self.dijkstra_backward_avoid_nodes(self.goal_arr[0], self.dead_edges)
        print(f"{self.next_to_goal_normal=}\n{self.next_to_goal_blocked=}")
        ###　(逆)ダイクストラ法 ###
        for i in range(tmp_id, tmp_id + self.population + self.for_population):  # 1人多く作成(強引な人)
            pos = []
            velocity = []
            if tmp_forceful_num:  # 強引な人(強引な人の位置が先に決まったのち通常の避難者の位置が決まる)
                velocity = self.decide_vel()
                try:
                    pos = self.pos_func.decide_forceful_position(self.rng, self.r, self.f_r, human_array)
                except TypeError:
                    pos = self.pos_func.decide_forceful_position(self.r, self.f_r, human_array)
                if not np.all(np.isfinite(pos)):
                    raise ValueError(f"Non-finite forceful position generated for id {i}: {pos}")
                has_phone = False
                if self.phone_ratio > 0.0:
                    has_phone = self.rng.random() < self.phone_ratio
                human = Human(i, self, pos, velocity,
                              tmp_div, shared,
                              self.space, self.add_file_name,
                              forceful_initial=True,
                              is_forceful=True,
                              has_phone=has_phone,
                              )
                self.space.place_agent(human, pos)
                self.schedule.add(human)
                human_array.append(human)
                tmp_forceful_num -= 1
            else:  # 通常の人
                pos = self.pos_func.decide_position(self.rng, self.r, self.f_r, human_array) #tmp
                if not np.all(np.isfinite(pos)):
                    raise ValueError(f"Non-finite initial position generated for id {i}: {pos}")
                velocity = self.decide_vel()
                has_phone = False
                if self.phone_ratio > 0.0:
                    has_phone = self.rng.random() < self.phone_ratio
                human = Human(i, self, pos, velocity,
                              tmp_div, shared,
                              self.space,
                              self.add_file_name,
                              has_phone=has_phone,)
                self.space.place_agent(human, pos)
                self.schedule.add(human)
                human_array.append(human)
        tmp_id += self.population + self.for_population
        self.all_agents = human_array
        return tmp_id

    def decide_vel(self):
        while 1:
            velocity = self.rng.normal(
                loc=self.v_arg[0], scale=self.v_arg[1], size=2)
            # 初期速度(および希望速さのx,y成分)は0.5以上1以下
            if 0.5 <= np.linalg.norm(velocity, 2) <= 1.:
                break
        return velocity
    
    def decide_dest(self):
        tmp_dest = [] #tmp
        tmp_dest = [0, 1, 2, 1] #tmp
        tmp_dest.append(self.goal_arr[0]) #tmp
        return tmp_dest

    def dijkstra_backward(self, goal_idx):
        return self._dijkstra_backward_core(goal_idx)

    def dijkstra_backward_avoid_nodes(self, goal_idx, blocked_nodes):
        return self._dijkstra_backward_core(goal_idx, blocked_nodes=blocked_nodes)
    
    def set_dijkstra_backward_with_penalty(self, goal_idx, node_penalty):
        return self._dijkstra_backward_core(goal_idx, node_penalty=node_penalty)

    def _dijkstra_backward_core(self, goal_idx, *, blocked_nodes=None, node_penalty=None):
        """
        goal_idx      : ゴールノードのインデックス
        blocked_nodes : 通りたくないノードの集合 (iterable) 例: {3, 5, 7}
        node_penalty  : ノードごとのペナルティ
                        - None          : すべて 0
                        - list / tuple  : node_penalty[v] で参照
                        - dict          : node_penalty.get(v, 0.0) で参照
        戻り値:
            dist[v] : ノード v から goal_idx までの最短距離
            next_to_goal[v] : v からゴール方向へ一歩進む“次ノード”（以前の prev と同じ意味）
        """
        N = len(self.dests)
        INF = 10**15

        # blocked_nodes を集合に
        blocked = set(blocked_nodes) if blocked_nodes is not None else set()

        # ペナルティの取り出し関数
        if node_penalty is None:
            def penalty(v: int) -> float:
                return 0.0
        elif isinstance(node_penalty, (list, tuple)):
            def penalty(v: int) -> float:
                return node_penalty[v]
        else:
            # dict を想定
            def penalty(v: int) -> float:
                return node_penalty.get(v, 0.0)

        dist = [INF] * N
        next_to_goal = [-1] * N   # 以前の prev: v から見た「次のノード」

        # ゴール自体が blocked なら何もできないのでそのまま返す
        if goal_idx in blocked:
            return dist, next_to_goal

        dist[goal_idx] = 0
        pq = [(0, goal_idx)]

        while pq:
            cost, u = heapq.heappop(pq)
            if cost > dist[u]:
                continue

            # u 自体が blocked なら、ここから先は展開しない
            if u in blocked:
                continue

            # すべての辺 u→v を走査する（グラフは無向想定）
            for v in self.edges[u]:
                # v が blocked ならそのノードへは遷移しない
                if v in blocked:
                    continue

                # 辺の長さ
                w = self.dist(self.dests[u], self.dests[v])
                new_cost = cost + w + penalty(v)

                if new_cost < dist[v]:
                    dist[v] = new_cost
                    next_to_goal[v] = u   # v から見た「次のノード」は u（ゴール方向）
                    heapq.heappush(pq, (new_cost, v))

        return dist, next_to_goal

    def get_path(self, start_idx, prev):
        path = []
        cur = start_idx
        while cur != -1:
            path.append(cur)
            if prev[cur] == -1:  # goal 到達
                break
            cur = prev[cur]
        return path
    
    def dist(self, x, y):
        """2点のユークリッド距離"""
        return math.hypot(x[0] - y[0], x[1] - y[1])

    def ccw(self, A, B, C): #"""点 A, B, C が反時計回りかを判定"""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(self, A, B, C, D):
        """線分 AB と CD が交差するかを返す"""
        return (self.ccw(A, C, D) != self.ccw(B, C, D)) and (self.ccw(A, B, C) != self.ccw(A, B, D))

    def has_line_of_sight(self, pos, node_pos, walls):
        """pos -> node_pos の直線が壁と交差しないかを判定"""
        for wall in walls:
            if self.intersect(pos, node_pos, wall[0], wall[1]):
                return False
        return True

    # scatter-dest
    def get_walls_for_los(self, state):
        if state == RouteState.NORMAL:
            return self.wall_arr
        elif state in (RouteState.BLOCKED_WAIT, RouteState.KNOWN):
            if not self._has_dead_walls():
                return self.wall_arr
            return np.concatenate([self.wall_arr, self.dead_wall_arr], axis=0)
        return self.wall_arr

    # scatter-dest
    def generate_scatter_destination(self, pos, node_pos, walls_for_los, rng=None,
                                     base_delta=1.0, base_radius=1.5, max_trials=4): ########
        rng = rng or np.random.default_rng()
        dir_vec = np.array(node_pos) - np.array(pos)
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-8:
            return np.array(node_pos)
        dir_vec = dir_vec / norm
        perp = np.array([-dir_vec[1], dir_vec[0]])
        delta = base_delta
        radius = base_radius
        for _ in range(max_trials):
            eps = rng.uniform(-radius, radius)
            cand = np.array(node_pos) + delta * dir_vec + eps * perp
            if self.has_line_of_sight(pos, cand, walls_for_los):
                return cand
            delta *= 0.5
            radius *= 0.5
        return np.array(node_pos)

    def select_first_subgoal_with_dist(self, agent, dist_to_goal, next_to_goal):
        walls_for_los = self.get_walls_for_los(agent.re_route_state)
        tmp_cost = 999999
        tmp_idx = 0
        for idx, dis in enumerate(dist_to_goal):
            if not self.has_line_of_sight(agent.pos, self.dests[idx], walls_for_los):
                continue
            cost_to_goal = dis + self.space.get_distance(agent.pos, self.dests[idx])
            # if tmp_cost > cost_to_goal or (self.time_step == 0 and idx == 12 and tmp_idx == 7 and cost_to_goal - tmp_cost < 0.5):
            if tmp_cost > cost_to_goal or (agent.re_route_state == RouteState.NORMAL and idx == 12 and tmp_idx == 7 and cost_to_goal - tmp_cost < 0.5):
                tmp_cost = cost_to_goal
                tmp_idx = idx
        tmp_path_arr = copy.copy(next_to_goal)
        tmp_path_arr2 = copy.copy(self.get_path(tmp_idx, tmp_path_arr))  # tmp
        route = tmp_path_arr2
        dest = route[0]
        return route, dest

    def select_first_subgoal(self, agent):
        if self.info_share_mode == InfoShareMode.SHARE_BLOCKED_ROAD and agent.block_info_state == BlockInfoState.KNOWN:
            dist_to_goal = self.dist_to_goal_blocked
            next_to_goal = self.next_to_goal_blocked
        elif agent.re_route_state == RouteState.NORMAL:
            dist_to_goal =  self.dist_to_goal_normal
            next_to_goal = self.next_to_goal_normal
        elif agent.re_route_state == RouteState.BLOCKED_WAIT:
            dist_to_goal =  self.dist_to_goal_blocked
            next_to_goal = self.next_to_goal_blocked
        elif agent.re_route_state == RouteState.KNOWN:
            dist_to_goal =  self.dist_to_goal_blocked
            next_to_goal = self.next_to_goal_blocked
        else:
            dist_to_goal = self.dist_to_goal_normal
            next_to_goal = self.next_to_goal_normal
        return self.select_first_subgoal_with_dist(agent, dist_to_goal, next_to_goal)

    def pre_wall_arr(self, wall_arr):
        if wall_arr is None:
            empty = np.empty((0, 2))
            return empty, empty, empty, np.array([])
        wall_arr = np.asarray(wall_arr)
        if wall_arr.size == 0:
            empty = np.empty((0, 2))
            return empty, empty, empty, np.array([])
        wall_a = wall_arr[:, 0]           # 各壁の始点 (N_wall, 2)
        wall_b = wall_arr[:, 1]           # 各壁の終点 (N_wall, 2)
        wall_ab = wall_b - wall_a         # ベクトル (N_wall, 2)
        wall_ab_len2 = np.array([])
        for ab in wall_ab:
            wall_ab_len2 = np.append(wall_ab_len2, np.dot(ab, ab))
        return wall_a, wall_b, wall_ab, wall_ab_len2

    def _has_dead_walls(self):
        if self.dead_wall_arr is None:
            return False
        if hasattr(self.dead_wall_arr, "size") and self.dead_wall_arr.size == 0:
            return False
        if hasattr(self.dead_wall_arr, "shape") and self.dead_wall_arr.shape[0] == 0:
            return False
        return True

    def make_agent_rng(self, agent_id):
        return np.random.default_rng(self.seed + agent_id)
    
    def check_f_parameter(self):
        count = 0
        normal = self.agent_params_by_trait[Trait.NORMAL]
        forceful = self.agent_params_by_trait[Trait.FORCEFUL]
        if normal.m != forceful.m:
            print(f"self.m change normal={normal.m} forceful={forceful.m}")
            count += 1
        if normal.tau != forceful.tau:
            print("tau change")
            count += 1
        if normal.repul_m[0] != forceful.repul_m[0]:
            print("repul_m[0] change")
            count += 1
        if normal.repul_m[1] != forceful.repul_m[1]:
            print("repul_m[1] change")
            count += 1
        if normal.k != forceful.k:
            print("k change")
            count += 1
        if normal.kappa != forceful.kappa:
            print("kappa change")
            count += 1
        if normal.r != forceful.r:
            print("r change")
            count += 1
        return count

    def _should_log_agent(self, agent, step_idx):
        goal_step = self.goal_reached_step[agent.unique_id]
        return goal_step == -1

    def validate_initial_positions(self):
        for agent in self.all_agents:
            pos_arr = np.asarray(agent.pos, dtype=float)
            if not np.all(np.isfinite(pos_arr)):
                raise ValueError(f"Non-finite initial position detected for id {agent.unique_id}: {agent.pos}")
            if not (0 <= agent.unique_id < self.num_agents):
                raise ValueError(f"Agent id {agent.unique_id} is out of logging range (num_agents={self.num_agents})")

    def log_initial_state(self):
        if not self.csv_plot:
            return None
        self.validate_initial_positions()
        self.log_positions(step_idx=0)
        self.log_states(step_idx=0)
        self.log_block_info(step_idx=0)

    def log_positions(self, step_idx=None):
        idx = self.time_step if step_idx is None else step_idx
        if not (0 <= idx < self.log_capacity):
            return None
        for agent in self.all_agents:
            if not self._should_log_agent(agent, idx): #goal_stepが記録されているならskip
                continue
            pos_arr = np.asarray(agent.pos, dtype=float)
            if not np.all(np.isfinite(pos_arr)):
                raise ValueError(f"Non-finite position logged for id {agent.unique_id} at step {idx}: {agent.pos}")
            self.pos_log[idx, agent.unique_id, :] = pos_arr
        return None

    def log_states(self, step_idx=None):
        idx = self.time_step if step_idx is None else step_idx
        if not (0 <= idx < self.log_capacity):
            return None
        for agent in self.all_agents:
            if not self._should_log_agent(agent, idx):
                continue
            self.state_log[idx, agent.unique_id] = int(agent.block_info_state)
        return None

    def log_block_info(self, step_idx=None):
        idx = self.time_step if step_idx is None else step_idx
        if not (0 <= idx < self.log_capacity):
            return None
        for agent in self.all_agents:
            if not self._should_log_agent(agent, idx):
                continue
            has_info = 1 if agent.known_dead_edges else 0
            self.has_block_info_log[idx, agent.unique_id] = has_info
            self.known_blocks_log[idx, agent.unique_id] = len(agent.known_dead_edges)
        return None

    def mark_goal_reached(self, agent, step_idx=None):
        idx = self.time_step + 1 if step_idx is None else step_idx
        idx = min(idx, self.log_capacity - 1)
        if self.goal_reached_step[agent.unique_id] == -1:
            self.goal_reached_step[agent.unique_id] = idx
            pos_arr = np.asarray(agent.pos, dtype=float)
            self.pos_log[idx, agent.unique_id, :] = pos_arr
            self.state_log[idx, agent.unique_id] = int(agent.block_info_state)
            self.has_block_info_log[idx, agent.unique_id] = 1 if agent.known_dead_edges else 0
            self.known_blocks_log[idx, agent.unique_id] = len(agent.known_dead_edges)



    def step(self):
        self.update_corner_counts()
        # Phase 1: 行動
        self.schedule.step()
        if self.time_step % self.share_every_steps == 0:
            self.communication_step()
        next_step_idx = self.time_step + 1
        if self.csv_plot:
            self.log_positions(step_idx=next_step_idx) #各避難者の位置情報を保存
            self.log_states(step_idx=next_step_idx)
            self.log_block_info(step_idx=next_step_idx)

        self.time_step = next_step_idx
        if self.time_step % 100 == 0:
            if self.all_agent_evacuate():
                self.running = False
        if self.time_step >= self.max_steps: 
            self.timeout_check()
            self.running = False

    def update_corner_counts(self):
        counts = {}
        exit_counts_by_node = {}
        for agent in list(self.schedule.agents):
            if not isinstance(agent, Human):
                continue
            detail = agent._get_turn_detail(agent.route_idx)
            if detail is None:
                continue
            _, cur, _, _, _, _, _, _ = detail
            target_params = self.get_target_params()
            half_width = self.get_road_width(agent.pos) / 2.0
            if not is_in_corner_area(agent.pos, cur, half_width, target_params.corner_margin):
                continue
            cur_node_id = agent.route[agent.route_idx]
            counts[cur_node_id] = counts.get(cur_node_id, 0) + 1
            next_exit_id = None
            if agent.route_idx + 1 < len(agent.route):
                next_exit_id = agent.route[agent.route_idx + 1]
            if next_exit_id is not None:
                node_counts = exit_counts_by_node.setdefault(cur_node_id, {})
                node_counts[next_exit_id] = node_counts.get(next_exit_id, 0) + 1
        self.corner_counts = counts
        self.exit_counts_by_node = exit_counts_by_node
        self._update_congested_states(counts)
        self._update_phase_exits(exit_counts_by_node)

    def _update_congested_states(self, counts):
        target_params = self.get_target_params()
        all_nodes = set(self.congested_state_by_node.keys()) | set(counts.keys())
        updated = {}
        for node_id in all_nodes:
            count = counts.get(node_id, 0)
            congested = self.congested_state_by_node.get(node_id, False)
            if congested:
                if count <= target_params.congestion_off:
                    congested = False
            else:
                if count >= target_params.congestion_on:
                    congested = True
            if congested:
                updated[node_id] = True
        self.congested_state_by_node = updated
        return None

    def _update_phase_exits(self, exit_counts_by_node):
        config = self.config
        if config is None or not getattr(config, "phase_enabled", True):
            return None
        min_green = int(getattr(config, "min_green_steps", 0))
        switch_margin = int(getattr(config, "phase_switch_margin", 0))
        only_when_congested = bool(getattr(config, "phase_only_when_congested", True))
        nodes = (
            set(self.phase_exit_by_node.keys())
            | set(exit_counts_by_node.keys())
            | set(self.congested_state_by_node.keys())
        )
        for node_id in nodes:
            if only_when_congested and not self.congested_state_by_node.get(node_id, False):
                self.phase_timer_by_node[node_id] = 0
                continue
            exit_counts = exit_counts_by_node.get(node_id, {})
            if not exit_counts:
                self.phase_exit_by_node[node_id] = None
                self.phase_timer_by_node[node_id] = 0
                continue
            candidate_exit, candidate_count = max(
                exit_counts.items(), key=lambda item: (item[1], item[0])
            )
            current_exit = self.phase_exit_by_node.get(node_id)
            timer = self.phase_timer_by_node.get(node_id, 0)
            if current_exit is None:
                self.phase_exit_by_node[node_id] = candidate_exit
                self.phase_timer_by_node[node_id] = 0
                continue
            if timer < min_green:
                self.phase_timer_by_node[node_id] = timer + 1
                continue
            current_count = exit_counts.get(current_exit, 0)
            if candidate_exit != current_exit and candidate_count >= current_count + switch_margin:
                self.phase_exit_by_node[node_id] = candidate_exit
                self.phase_timer_by_node[node_id] = 0
            else:
                self.phase_timer_by_node[node_id] = timer + 1
        return None

    def all_agent_evacuate(self):
        return len(self.schedule.agents) == 0

    def timeout_check(self):
        if self.csv_plot:
            for obj in self.schedule.agents:
                if isinstance(obj, Human):
                    path = obj.add_file_name
                    obj.make_dir(path)
                    obj.write_record(path)
        self.write_interrupt()

    def write_interrupt(self):
        normal_num = len(
            open(f"{self.add_file_name}/Data/normal.dat").readlines())
        forceful_num = len(
            open(f"{self.add_file_name}/Data/forceful.dat").readlines())
        if normal_num < self.population + 1:
            if forceful_num < self.for_population + 1:
                with open(f"{self.add_file_name}/Data/normal.dat", "a") as f:
                    f.write(f"interrupt\n")
                with open(f"{self.add_file_name}/Data/forceful.dat", "a") as f:
                    f.write(f"interrupt\n")
            else:
                with open(f"{self.add_file_name}/Data/normal.dat", "a") as f:
                    f.write(f"interrupt\n")
        else:
            with open(f"{self.add_file_name}/Data/forceful.dat", "a") as f:
                f.write(f"interrupt\n")

    def dump_logs_format_A(self, out_dir):
        if not self.csv_plot:
            return None
        os.makedirs(out_dir, exist_ok=True)
        end_step = min(self.time_step, self.max_steps)
        for agent in self.all_agents:
            goal_step = self.goal_reached_step[agent.unique_id]
            last_step = goal_step if goal_step != -1 else end_step
            last_step = min(last_step, self.log_capacity - 1)
            if last_step < 0:
                continue
            pos = self.pos_log[: last_step + 1, agent.unique_id, :]
            state = self.state_log[: last_step + 1, agent.unique_id]
            state_to_use = state if state is not None else np.zeros(len(pos), dtype=np.int8)
            valid_len = 0
            for idx, (p, s) in enumerate(zip(pos, state_to_use)):
                if goal_step != -1 and idx > last_step:
                    break  # goal以降は出力しない
                if not (np.all(np.isfinite(p)) and np.isfinite(s)):
                    break
                # 未記録の初期値が 0,0 ならスキップしたい場合
                if idx > 0 and np.allclose(p, 0.0) and s == 0:
                    break
                valid_len = idx + 1
            if valid_len == 0:
                continue
            has_phone = 1 if agent.has_phone else 0
            has_block = self.has_block_info_log[:valid_len, agent.unique_id]
            known_blocks = self.known_blocks_log[:valid_len, agent.unique_id]
            phone_col = np.full(valid_len, has_phone, dtype=np.int8)
            data = np.column_stack([
                pos[:valid_len],
                state_to_use[:valid_len],
                phone_col,
                has_block,
                known_blocks,
            ])
            np.savetxt(
                os.path.join(out_dir, f"id{agent.unique_id}_normal.csv"),
                data,
                delimiter=",",
            )
        return None

    def communication_step(self):
        if self.phone_ratio > 0.0:
            phone_agents = [
                agent for agent in self.schedule.agents
                if isinstance(agent, Human) and agent.has_phone
            ]
            self._share_block_info_within_agents(phone_agents, self.R_short)
        if self.share_block_info:
            all_agents = [
                agent for agent in self.schedule.agents
                if isinstance(agent, Human)
            ]
            self._share_block_info_within_agents(all_agents, self.R_face)
        return None

    def _share_block_info_within_agents(self, agents, radius):
        if not agents:
            return None
        agent_set = set(agents)
        visited = set()
        for agent in agents:
            if agent in visited:
                continue
            component = []
            stack = [agent]
            visited.add(agent)
            while stack:
                current = stack.pop()
                component.append(current)
                neighbors = self.space.get_neighbors(
                    current.pos, radius, False)
                for neighbor in neighbors:
                    if neighbor in agent_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            if len(component) <= 1:
                continue
            union_info = set()
            for member in component:
                union_info.update(member.known_dead_edges)
            if not union_info:
                continue
            for member in component:
                member.apply_shared_block_info(union_info)
        return None

    def get_dead_edge_id(self, dead_wall_index):
        if not self.dead_edges:
            return None
        if len(self.dead_edges) == 1:
            return self.dead_edges[0]
        if dead_wall_index < len(self.dead_edges):
            return self.dead_edges[dead_wall_index]
        return dead_wall_index
