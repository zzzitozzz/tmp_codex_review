import mesa
import os
import sys
import warnings
import copy
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import heapq
import math

from agent import SharedParams, Human, HumanSpecs, ForcefulHuman, ForcefulHumanSpecs, Wall, RouteState, InfoShareMode, BlockInfoState
warnings.simplefilter('ignore', UserWarning)


class MoveAgent(mesa.Model):

    def __init__(
            self, population=100, for_population=1, dests=[], edges=[], dead_edges=[], goal_arr=[], v_arg=[], wall_arr=[[]],
            dead_wall_arr=[[]],seed=1, r=0.5,
            wall_r=0.5, human_var={}, forceful_human_var={},
            width=100, height=100, dt=0.1,
            in_dest_d=3, vision=3, time_step=0,
            add_file_name="", add_file_name_arr=[],
            len_sq=3., f_r=0.,pos_func= {},
            csv_plot=False,
            info_share_mode=InfoShareMode.NO_SHARE):
        super().__init__()
        self.population = population
        self.for_population = for_population
        self.dests = dests
        self.edges = edges
        self.dead_edges = dead_edges
        self.goal_arr = goal_arr
        self.v_arg = v_arg
        self.wall_arr = wall_arr
        self.dead_wall_arr = dead_wall_arr
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
        self.max_steps = 1500
        self.num_agents = self.population + self.for_population
        self.pos_log = np.zeros(
            (self.max_steps, self.num_agents, 2), dtype=np.float32)
        self.state_log = np.zeros(
            (self.max_steps, self.num_agents), dtype=np.int8)
        shared, human_var_inst, forceful_human_var_inst = self.assign_ini_human_and_forceful_human_var()
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
        self.make_agents(shared, human_var_inst, forceful_human_var_inst)
        self.running = True
        print(f"change para: {self.check_f_parameter()}")
        self.make_basic_dir()
        self.save_specs_to_file(shared, human_var_inst, forceful_human_var_inst)

    def dir_parts(self):
        basic_file_name = f"{self.add_file_name_arr[0]}/nol_pop_{self.population}"
        self.add_file_name = f"{basic_file_name}/"
        self.add_file_name = self.add_file_name + "seed_" + str(self.seed)
        return None

    def assign_ini_human_and_forceful_human_var(self):
        self.m = self.human_var["m"]
        self.tau = self.human_var["tau"]
        self.k = self.human_var["k"]
        self.kappa = self.human_var["kappa"]
        self.repul_h = self.human_var["repul_h"]
        self.repul_m = self.human_var["repul_m"]
        self.f_m = self.forceful_human_var["f_m"]
        self.f_tau = self.forceful_human_var["f_tau"]
        self.f_k = self.forceful_human_var["f_k"]
        self.f_kappa = self.forceful_human_var["f_kappa"]
        self.f_repul_h = self.forceful_human_var["f_repul_h"]
        self.f_repul_m = self.forceful_human_var["f_repul_m"]
        shared = SharedParams(self.in_dest_d, self.vision, self.dt)
        human_var_inst = HumanSpecs(self.r, self.m, self.tau, self.k, self.kappa, self.repul_h, self.repul_m)
        forceful_human_var_inst = ForcefulHumanSpecs(self.f_r, self.f_m, self.f_tau, self.f_k, self.f_kappa, self.f_repul_h, self.f_repul_m)
        return shared, human_var_inst, forceful_human_var_inst

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

    def save_specs_to_file(self, shared, human_specs, forceful_human_specs):
        path = f"{self.add_file_name}/../human_specs.yaml"
        if not os.path.exists(path):
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data = {
                "run_time" : run_time,
                "shared_val" : vars(shared),
                "human": vars(human_specs),
                "forcefulhuman": vars(forceful_human_specs)
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

    def make_agents(self, shared, human_var_inst, forceful_human_var_inst):
        tmp_id = 0
        tmp_id = self.generate_human(tmp_id, shared, human_var_inst, forceful_human_var_inst)

    def generate_human(self, tmp_id, shared, human_var_inst, forceful_human_var_inst):
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
                pos = self.pos_func.decide_forceful_position(self.r, self.f_r, human_array)
                route = copy.copy(self.decide_dest())
                dest = route[0]
                human = ForcefulHuman(i, self, pos, velocity,
                                      dest, route,
                                      tmp_div, shared,
                                      human_var_inst,
                                      self.space, self.add_file_name,
                                      forceful_human_var_inst,
                                      )
                self.space.place_agent(human, pos)
                self.schedule.add(human)
                human_array.append(human)
                tmp_forceful_num -= 1
            else:  # 通常の人
                pos = self.pos_func.decide_position(self.rng, self.r, self.f_r, human_array) #tmp
                velocity = self.decide_vel()
                human = Human(i, self, pos, velocity,
                              tmp_div, shared,
                              human_var_inst, self.space,
                              self.add_file_name,)
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
        wall_a = wall_arr[:, 0]           # 各壁の始点 (N_wall, 2)
        wall_b = wall_arr[:, 1]           # 各壁の終点 (N_wall, 2)
        wall_ab = wall_b - wall_a         # ベクトル (N_wall, 2)
        wall_ab_len2 = np.array([])
        for ab in wall_ab:
            wall_ab_len2 = np.append(wall_ab_len2, np.dot(ab, ab))
        return wall_a, wall_b, wall_ab, wall_ab_len2

    def make_agent_rng(self, agent_id):
        return np.random.default_rng(self.seed + agent_id)
    
    def check_f_parameter(self):
        count = 0
        if self.m != self.f_m:
            print(f"self.m change {self.m=} {self.f_m=}")
            count += 1
        if self.tau != self.f_tau:
            print("tau change")
            count += 1
        if self.repul_h[0] != self.f_repul_h[0]:
            print("repul_h[0] change")
            count += 1
        if self.repul_h[1] != self.f_repul_h[1]:
            print("repul_h[1] change")
            count += 1
        if self.repul_m[0] != self.f_repul_m[0]:
            print("repul_m[0] change")
            count += 1
        if self.repul_m[1] != self.f_repul_m[1]:
            print("repul_m[1] change")
            count += 1
        if self.k != self.f_k:
            print("k change")
            count += 1
        if self.kappa != self.f_kappa:
            print("kappa change")
            count += 1
        if self.f_r != self.r:
            print("f_r change")
            count += 1
        return count

    def log_positions(self):
        if not (0 <= self.time_step < self.max_steps):
            return None
        for agent in self.all_agents:
            self.pos_log[self.time_step, agent.unique_id, :] = agent.pos
        return None

    def log_states(self):
        if not (0 <= self.time_step < self.max_steps):
            return None
        for agent in self.all_agents:
            self.state_log[self.time_step, agent.unique_id] = int(agent.block_info_state)
        return None

    def step(self):
        # Phase 1: 行動
        self.schedule.step()
        if self.csv_plot:
            self.log_positions()

        # Phase 2: 共有
        if self.info_share_mode == InfoShareMode.SHARE_BLOCKED_ROAD:
            for agent in list(self.schedule.agents):
                if isinstance(agent, (Human, ForcefulHuman)):
                    agent.share_block_info()
        if self.csv_plot:
            self.log_states()

        # Phase 3: 状態更新
        if self.info_share_mode == InfoShareMode.SHARE_BLOCKED_ROAD:
            for agent in list(self.schedule.agents):
                if isinstance(agent, (Human, ForcefulHuman)):
                    agent.update_block_info_state()

        self.time_step += 1
        if self.time_step % 100 == 0:
            if self.all_agent_evacuate():
                self.running = False
        if self.time_step >= self.max_steps: 
            self.timeout_check()
            self.running = False

    def all_agent_evacuate(self):
        cur_pop_num = (
            len(open(f"{self.add_file_name}/Data/normal.dat").readlines()))
        if cur_pop_num == self.population + 1:
            if (len(open(f"{self.add_file_name}/Data/forceful.dat").readlines())) + cur_pop_num == self.population + self.for_population + 2:
                return True
        return False

    def timeout_check(self):
        if self.csv_plot:
            for obj in self.schedule.agents:
                if type(obj) is Human or type(obj) is ForcefulHuman:
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
            pos = self.pos_log[:end_step, agent.unique_id, :]
            state = self.state_log[:end_step, agent.unique_id]
            data = np.column_stack([pos, state])
            np.savetxt(
                os.path.join(out_dir, f"id{agent.unique_id}_normal.csv"),
                data,
                delimiter=",",
            )
        return None
