from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Tuple


class Trait(Enum):
    NORMAL = auto()
    FORCEFUL = auto()


@dataclass(frozen=True)
class AgentParams:
    r: float
    m: float
    tau: float
    v0: float
    k: float
    kappa: float
    repul_m: list


@dataclass(frozen=True)
class PairParams:
    a: float
    b: float
    k: float
    kappa: float
    r_scale: float = 1.0


class PairParamsTable:
    def __init__(self, table: Dict[Tuple[Trait, Trait], PairParams]):
        self._table = dict(table)

    def get(self, trait_i: Trait, trait_j: Trait) -> PairParams:
        return self._table[(trait_i, trait_j)]

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        formatted = {}
        for (trait_i, trait_j), params in self._table.items():
            key = f"{trait_i.name}->{trait_j.name}"
            formatted[key] = {
                "a": params.a,
                "b": params.b,
                "k": params.k,
                "kappa": params.kappa,
                "r_scale": params.r_scale,
            }
        return formatted


GOAL_STRONG_FACTORS = {"m": 1.2, "v0": 1.2, "tau": 0.8}
FF_WEAK_FACTORS = {"a": 0.7, "b": 1.3, "k": 0.7, "kappa": 0.7, "r_scale": 0.9}
ASYM_FORCEFUL_FACTORS = {"a": 0.7, "b": 1.3, "k": 0.7, "kappa": 0.7, "r_scale": 0.9}
ASYM_NORMAL_FACTORS = {"a": 1.3, "b": 0.7, "k": 1.3, "kappa": 1.3, "r_scale": 1.1}


def _apply_pair_factors(params: PairParams, factors: Dict[str, float]) -> PairParams:
    return PairParams(
        a=params.a * factors.get("a", 1.0),
        b=params.b * factors.get("b", 1.0),
        k=params.k * factors.get("k", 1.0),
        kappa=params.kappa * factors.get("kappa", 1.0),
        r_scale=params.r_scale * factors.get("r_scale", 1.0),
    )


def build_sfm_params(human_var, forceful_human_var, r, f_r, preset_name: str):
    v0 = human_var.get("v0", 0.8)
    f_v0 = forceful_human_var.get("f_v0", v0)
    normal_agent = AgentParams(
        r=r,
        m=human_var["m"],
        tau=human_var["tau"],
        v0=v0,
        k=human_var["k"],
        kappa=human_var["kappa"],
        repul_m=human_var["repul_m"],
    )
    forceful_agent = AgentParams(
        r=f_r,
        m=forceful_human_var["f_m"],
        tau=forceful_human_var["f_tau"],
        v0=f_v0,
        k=forceful_human_var["f_k"],
        kappa=forceful_human_var["f_kappa"],
        repul_m=forceful_human_var["f_repul_m"],
    )
    agent_params = {
        Trait.NORMAL: normal_agent,
        Trait.FORCEFUL: forceful_agent,
    }
    baseline_pair = {
        (Trait.NORMAL, Trait.NORMAL): PairParams(
            a=human_var["repul_h"][0],
            b=human_var["repul_h"][1],
            k=human_var["k"],
            kappa=human_var["kappa"],
        ),
        (Trait.NORMAL, Trait.FORCEFUL): PairParams(
            a=human_var["repul_h"][0],
            b=human_var["repul_h"][1],
            k=human_var["k"],
            kappa=human_var["kappa"],
        ),
        (Trait.FORCEFUL, Trait.NORMAL): PairParams(
            a=forceful_human_var["f_repul_h"][0],
            b=forceful_human_var["f_repul_h"][1],
            k=forceful_human_var["f_k"],
            kappa=forceful_human_var["f_kappa"],
        ),
        (Trait.FORCEFUL, Trait.FORCEFUL): PairParams(
            a=forceful_human_var["f_repul_h"][0],
            b=forceful_human_var["f_repul_h"][1],
            k=forceful_human_var["f_k"],
            kappa=forceful_human_var["f_kappa"],
        ),
    }
    preset = preset_name.lower()
    if preset in {"goal_strong", "combo"}:
        agent_params[Trait.FORCEFUL] = AgentParams(
            r=forceful_agent.r,
            m=forceful_agent.m * GOAL_STRONG_FACTORS["m"],
            tau=forceful_agent.tau * GOAL_STRONG_FACTORS["tau"],
            v0=forceful_agent.v0 * GOAL_STRONG_FACTORS["v0"],
            k=forceful_agent.k,
            kappa=forceful_agent.kappa,
            repul_m=forceful_agent.repul_m,
        )
    if preset in {"ff_weak", "combo"}:
        baseline_pair[(Trait.FORCEFUL, Trait.FORCEFUL)] = _apply_pair_factors(
            baseline_pair[(Trait.FORCEFUL, Trait.FORCEFUL)], FF_WEAK_FACTORS
        )
    if preset in {"asym_fn", "combo"}:
        baseline_pair[(Trait.FORCEFUL, Trait.NORMAL)] = _apply_pair_factors(
            baseline_pair[(Trait.FORCEFUL, Trait.NORMAL)], ASYM_FORCEFUL_FACTORS
        )
        baseline_pair[(Trait.NORMAL, Trait.FORCEFUL)] = _apply_pair_factors(
            baseline_pair[(Trait.NORMAL, Trait.FORCEFUL)], ASYM_NORMAL_FACTORS
        )
    return agent_params, PairParamsTable(baseline_pair)
