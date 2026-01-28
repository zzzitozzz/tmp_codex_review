import sys
from model import MoveAgent

import numpy as np
from dataclasses import dataclass
from agent import SharedParams, Human, Wall, InfoShareMode
from params import StrategyConfig
from maps.convex_config import CONVEX_MAP_CONFIG

@dataclass
class InitPosFuncs:
    config: object
    f_r_use: bool = False # True: 常に強引な避難者の半径を使用 False: 普段は強引な避難者も通常の避難者と同じ半径を使用

    def decide_position(self, rng, r, f_r, human_array):
        rect = self.config.spawn["normal"]
        while 1:
            x = rng.uniform(rect.x_min, rect.x_max)
            y = rng.uniform(rect.y_min, rect.y_max) #初期配置(未確定)
            if rect.x_min + r * 2 <= x <= rect.x_max - r * 2 and rect.y_min + r * 2 <= y <= rect.y_max - r * 2:
                tmp_pos = np.array((x, y))
                if self.human_pos_check(r, f_r, tmp_pos, human_array): #ボジションチェック(既存のエージェントの位置と被っていないか)
                    pos = tmp_pos
                    break
        return pos

    def decide_forceful_position(self, rng, r, f_r, human_array):
        rect = self.config.spawn["forceful"]
        while 1:
            x = rng.uniform(rect.x_min, rect.x_max)
            y = rng.uniform(rect.y_min, rect.y_max)
            if rect.x_min + r <= x <= rect.x_max - r and rect.y_min + r <= y <= rect.y_max - r:
                tmp_pos = np.array((x, y))
                if self.forceful_human_pos_check(r, f_r, tmp_pos, human_array):
                    pos = tmp_pos
                    break
        return pos

    def human_pos_check(self, r, f_r, tmp_pos, human_array):
        for hu in human_array:
            dis = self.get_distance(tmp_pos, hu.pos)
            if self.f_r_use and hu.is_forceful:
                if dis < r + f_r:
                    return False
            elif dis < r + r:
                return False
        return True

    def forceful_human_pos_check(self, r, f_r, tmp_pos, human_array):
        for hu in human_array:
            dis = self.get_distance(tmp_pos, hu.pos)
            if self.f_r_use:
                radius_sum = f_r + (f_r if hu.is_forceful else r)
            else:
                radius_sum = r + r
            if dis < radius_sum:
                return False
                    
        return True
    
    def get_distance(self, pos1, pos2):
        return np.linalg.norm(pos1 - pos2)

def _resolve_forceful(val, fallback):
    return fallback if val is None else val


def build_sfm_vars(args, f_tau):
    base = {
        "m": args.m,
        "tau": args.tau,
        "k": args.k,
        "kappa": args.kappa,
        "repul_h": [args.repul_h_a, args.repul_h_b],
        "repul_m": [args.repul_m_a, args.repul_m_b],
        "v0": args.v0,
    }
    forceful = {
        "f_m": _resolve_forceful(args.f_m, args.m),
        "f_tau": f_tau,
        "f_k": _resolve_forceful(args.f_k, args.k),
        "f_kappa": _resolve_forceful(args.f_kappa, args.kappa),
        "f_repul_h": [
            _resolve_forceful(args.f_repul_h_a, args.repul_h_a),
            _resolve_forceful(args.f_repul_h_b, args.repul_h_b),
        ],
        "f_repul_m": [
            _resolve_forceful(args.f_repul_m_a, args.repul_m_a),
            _resolve_forceful(args.f_repul_m_b, args.repul_m_b),
        ],
        "f_v0": _resolve_forceful(args.f_v0, args.v0),
    }
    return base, forceful


def build_strategy_config(args, f_tau):
    preset = args.forceful_preset
    auto_goal_strong = args.forceful_mass is not None or args.forceful_tau is not None
    auto_ff_weak = any(val is not None for val in [args.a_ff, args.b_ff, args.k_ff, args.kappa_ff, args.r_ff_scale])
    asym_flags = [args.asym_a, args.asym_b, args.asym_k, args.asym_r, args.asym_kappa]
    has_asym_flags = any(flag is True for flag in asym_flags)
    auto_asym_fn = args.alpha is not None or has_asym_flags
    goal_strong = args.goal_strong or preset in {"goal_strong", "combo"} or auto_goal_strong
    ff_weak = args.ff_weak or preset in {"ff_weak", "combo"} or auto_ff_weak
    asym_fn = args.asym_fn or preset in {"asym_fn", "combo"} or auto_asym_fn
    if asym_fn and not has_asym_flags:
        raise ValueError("--asym_fn requires at least one asym_* flag")
    if asym_fn and args.alpha is None:
        raise ValueError("--asym_fn requires --alpha")
    if args.alpha is not None and not (0.0 < args.alpha < 1.0):
        raise ValueError("--alpha must satisfy 0 < alpha < 1")
    forceful_tau = args.forceful_tau if args.forceful_tau is not None else f_tau
    forceful_mass = args.forceful_mass if args.forceful_mass is not None else args.f_m
    return StrategyConfig(
        goal_strong=goal_strong,
        ff_weak=ff_weak,
        asym_fn=asym_fn,
        asym_a=args.asym_a,
        asym_b=args.asym_b,
        asym_k=args.asym_k,
        asym_r=args.asym_r,
        asym_kappa=args.asym_kappa,
        forceful_mass=forceful_mass,
        forceful_tau=forceful_tau,
        a_ff=args.a_ff,
        b_ff=args.b_ff,
        k_ff=args.k_ff,
        kappa_ff=args.kappa_ff,
        r_ff_scale=args.r_ff_scale,
        alpha=args.alpha,
    )


def apply_strategy_shorthand(args):
    defaults = {
        "forceful_mass": None,
        "forceful_tau": None,
        "a_ff": None,
        "b_ff": None,
        "k_ff": None,
        "kappa_ff": None,
        "r_ff_scale": None,
        "alpha": None,
        "asym_a": None,
        "asym_b": None,
        "asym_k": None,
        "asym_r": None,
        "asym_kappa": None,
        "goal_strong": False,
        "ff_weak": False,
        "asym_fn": False,
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not args.strategy_args:
        return None
    tokens = list(args.strategy_args)
    presets = {"baseline", "goal_strong", "ff_weak", "asym_fn", "combo"}
    if tokens and tokens[0] in presets:
        args.forceful_preset = tokens.pop(0)
    bool_flags = {"goal_strong", "ff_weak", "asym_fn", "asym_a", "asym_b", "asym_k", "asym_r", "asym_kappa"}
    key_map = {
        "f_m": "forceful_mass",
        "f_tau": "forceful_tau",
        "a_ff": "a_ff",
        "b_ff": "b_ff",
        "k_ff": "k_ff",
        "kappa_ff": "kappa_ff",
        "r_ff_scale": "r_ff_scale",
        "alpha": "alpha",
    }
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token in bool_flags:
            setattr(args, token, True)
            idx += 1
            continue
        if token not in key_map:
            raise ValueError(f"Unknown strategy token: {token}")
        if idx + 1 >= len(tokens):
            raise ValueError(f"Missing value for strategy token: {token}")
        value = float(tokens[idx + 1])
        setattr(args, key_map[token], value)
        idx += 2
    return None


def make_new_model_instance(human_var, forceful_human_var, wall_arr, dead_wall_arr, pop_num, for_pop, dests, edges,
                            goal_arr, goals, tmp_seed,len_sq, f_r, f_tau, pos_func, csv_plot,
                            share_block_info=False, forceful_preset="baseline", strategy=None, config=None):
    ex_num = 1 # force_tau
    # if csv_plot:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}_csv/tau_{int(f_tau*100)}/"]
    # else:
    #     file_name_array = [
    #         f"/local_home/keito/simple_convex_map/agst_dir/goal_up_forceful_tau/ex{ex_num}_for_{for_pop}_len_{int(len_sq)}/tau_{int(f_tau*100)}/"]
    if csv_plot:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]
    else:
        file_name_array = [f"./tmp_data/tau_{int(f_tau*100)}/"]

    m = MoveAgent(
        population=pop_num,
        for_population=for_pop,
        dests=dests,
        edges=edges,
        goal_arr=goal_arr,
        goals=goals,
        v_arg=[1., 1.],
        wall_arr=wall_arr,
        dead_wall_arr=dead_wall_arr,
        seed=tmp_seed,  # 乱数生成用
        r=0.5,  # 避難者の大きさ
        wall_r=1.0,  # うそ壁の大きさß
        human_var=human_var,
        forceful_human_var=forceful_human_var,
        width=60,  # 見かけの大きさ(マップ)
        height=60,  # 見かけの大きさ(マップ)
        dt=0.3,
        in_dest_d=3,
        vision=1.5,  # 10
        time_step=0,
        add_file_name="",
        add_file_name_arr=file_name_array,
        len_sq=len_sq,
        f_r=f_r,
        pos_func=pos_func,
        csv_plot=csv_plot,
        info_share_mode=InfoShareMode.SHARE_BLOCKED_ROAD if share_block_info else InfoShareMode.NO_SHARE,
        forceful_preset=forceful_preset,
        strategy=strategy,
        config=config)
    return m


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pop_num", type=int, help="通常の人数")
    parser.add_argument("f_tau", type=float, help="変更する変数の値")
    parser.add_argument("tmp_seed", type=int, help="seed値")
    parser.add_argument("--m", type=float, default=80.0, help="通常避難者の質量")
    parser.add_argument("--f_m", type=float, help="強引避難者の質量")
    parser.add_argument("--tau", type=float, default=0.5, help="通常避難者のtau")
    parser.add_argument("--k", type=float, default=120000.0, help="通常避難者のk")
    parser.add_argument("--kappa", type=float, default=240000.0, help="通常避難者のkappa")
    parser.add_argument("--repul_h_a", type=float, default=2000.0, help="通常避難者のA_ij")
    parser.add_argument("--repul_h_b", type=float, default=0.08, help="通常避難者のB_ij")
    parser.add_argument("--repul_m_a", type=float, default=2000.0, help="壁反発のA")
    parser.add_argument("--repul_m_b", type=float, default=0.08, help="壁反発のB")
    parser.add_argument("--f_k", type=float, help="強引避難者のk")
    parser.add_argument("--f_kappa", type=float, help="強引避難者のkappa")
    parser.add_argument("--f_repul_h_a", type=float, help="強引避難者のA_ij")
    parser.add_argument("--f_repul_h_b", type=float, help="強引避難者のB_ij")
    parser.add_argument("--f_repul_m_a", type=float, help="強引避難者の壁反発A")
    parser.add_argument("--f_repul_m_b", type=float, help="強引避難者の壁反発B")
    parser.add_argument("--v0", type=float, default=0.8, help="通常避難者の希望速度係数")
    parser.add_argument("--f_v0", type=float, help="強引避難者の希望速度係数")
    parser.add_argument("--share_block_info", action="store_true", help="不通道路情報を共有する")
    parser.add_argument("--forceful_preset", default="baseline",
                        choices=["baseline", "goal_strong", "ff_weak", "asym_fn", "combo"],
                        help="SFM係数プリセット")
    parser.add_argument("strategy_args", nargs="*", help="preset/strategy shorthand")
    args = parser.parse_args()
    apply_strategy_shorthand(args)

    pop_num = args.pop_num
    f_tau = args.f_tau
    tmp_seed = args.tmp_seed
    share_block_info = args.share_block_info
    f_r = 0.5 # 強引な避難者の大きさ
    for_pop = 0  # 強引な避難者の人数 #tmp
    csv_plot = True  # csvファイル(各エージェントの動きの軌跡)を出力するかどうか
    len_sq = 3  # 長方形の一辺の長さはlen_sq*2
    # max_f_r = 1.01
    human_var, forceful_human_var = build_sfm_vars(args, f_tau)
    strategy = build_strategy_config(args, f_tau)
    config = CONVEX_MAP_CONFIG
    pos_func = InitPosFuncs(config=config)
    wall_arr = config.wall_arr
    dead_wall_arr = config.dead_wall_arr

    dests = config.dests
    edges = config.edges
    goal_arr = config.goal_arr
    goals = config.goals
    while 1:
        m = make_new_model_instance(
            human_var, forceful_human_var, wall_arr, dead_wall_arr, pop_num, for_pop, dests, edges, goal_arr, goals, tmp_seed, len_sq, f_r, f_tau, pos_func, csv_plot,
            share_block_info=share_block_info, forceful_preset=args.forceful_preset, strategy=strategy, config=config)
        m.running = True
        while m.running:
            m.step()
        if m.csv_plot:
            m.dump_logs_format_A(f"{m.add_file_name}/csv")
        sys.exit()
