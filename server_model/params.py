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


def build_sfm_params(human_var, forceful_human_var, r, f_r, preset_name="baseline"):
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
    return agent_params, PairParamsTable(baseline_pair)
