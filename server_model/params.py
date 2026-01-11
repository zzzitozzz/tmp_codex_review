from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class StrategyConfig:
    goal_strong: bool = False
    ff_weak: bool = False
    asym_fn: bool = False
    asym_a: bool | None = None
    asym_b: bool | None = None
    asym_k: bool | None = None
    asym_r: bool | None = None
    asym_kappa: bool | None = None
    forceful_mass: float | None = None
    forceful_tau: float | None = None
    a_ff: float | None = None
    b_ff: float | None = None
    k_ff: float | None = None
    kappa_ff: float | None = None
    r_ff_scale: float | None = None
    alpha: float | None = None

    def to_dict(self):
        return {
            "goal_strong": self.goal_strong,
            "ff_weak": self.ff_weak,
            "asym_fn": self.asym_fn,
            "asym_a": self.asym_a,
            "asym_b": self.asym_b,
            "asym_k": self.asym_k,
            "asym_r": self.asym_r,
            "asym_kappa": self.asym_kappa,
            "forceful_mass": self.forceful_mass,
            "forceful_tau": self.forceful_tau,
            "a_ff": self.a_ff,
            "b_ff": self.b_ff,
            "k_ff": self.k_ff,
            "kappa_ff": self.kappa_ff,
            "r_ff_scale": self.r_ff_scale,
            "alpha": self.alpha,
        }


def _scaled_pair(params: PairParams, factor: float) -> PairParams:
    return PairParams(
        a=params.a * factor,
        b=params.b * factor,
        k=params.k * factor,
        kappa=params.kappa * factor,
        r_scale=params.r_scale,
    )


def build_sfm_params(human_var, forceful_human_var, r, f_r, preset_name="baseline",
                     strategy: StrategyConfig | None = None):
    strategy = strategy or StrategyConfig()
    preset = preset_name.lower()
    if preset in {"goal_strong", "combo"}:
        strategy = StrategyConfig(**{**strategy.to_dict(), "goal_strong": True})
    if preset in {"ff_weak", "combo"}:
        strategy = StrategyConfig(**{**strategy.to_dict(), "ff_weak": True})
    if preset in {"asym_fn", "combo"}:
        strategy = StrategyConfig(**{**strategy.to_dict(), "asym_fn": True})
    v0 = human_var.get("v0", 0.8)
    base_pair = PairParams(
        a=human_var["repul_h"][0],
        b=human_var["repul_h"][1],
        k=human_var["k"],
        kappa=human_var["kappa"],
    )
    normal_agent = AgentParams(
        r=r,
        m=human_var["m"],
        tau=human_var["tau"],
        v0=v0,
        k=human_var["k"],
        kappa=human_var["kappa"],
        repul_m=human_var["repul_m"],
    )
    if strategy.goal_strong:
        forceful_mass = strategy.forceful_mass if strategy.forceful_mass is not None else human_var["m"]
        forceful_tau = strategy.forceful_tau if strategy.forceful_tau is not None else human_var["tau"]
    else:
        forceful_mass = human_var["m"]
        forceful_tau = human_var["tau"]
    forceful_agent = AgentParams(
        r=f_r,
        m=forceful_mass,
        tau=forceful_tau,
        v0=v0,
        k=human_var["k"],
        kappa=human_var["kappa"],
        repul_m=human_var["repul_m"],
    )
    agent_params = {
        Trait.NORMAL: normal_agent,
        Trait.FORCEFUL: forceful_agent,
    }
    baseline_pair = {
        (Trait.NORMAL, Trait.NORMAL): base_pair,
        (Trait.NORMAL, Trait.FORCEFUL): base_pair,
        (Trait.FORCEFUL, Trait.NORMAL): base_pair,
        (Trait.FORCEFUL, Trait.FORCEFUL): base_pair,
    }
    if strategy.ff_weak:
        baseline_pair[(Trait.FORCEFUL, Trait.FORCEFUL)] = PairParams(
            a=base_pair.a if strategy.a_ff is None else strategy.a_ff,
            b=base_pair.b if strategy.b_ff is None else strategy.b_ff,
            k=base_pair.k if strategy.k_ff is None else strategy.k_ff,
            kappa=base_pair.kappa if strategy.kappa_ff is None else strategy.kappa_ff,
            r_scale=1.0 if strategy.r_ff_scale is None else strategy.r_ff_scale,
        )
    if strategy.asym_fn and strategy.alpha is not None:
        asym_flags = {
            "a": strategy.asym_a,
            "b": strategy.asym_b,
            "k": strategy.asym_k,
            "r": strategy.asym_r,
            "kappa": strategy.asym_kappa,
        }
        apply_all = not any(flag is True for flag in asym_flags.values())
        apply_a = apply_all or asym_flags["a"] is True
        apply_b = apply_all or asym_flags["b"] is True
        apply_k = apply_all or asym_flags["k"] is True
        apply_r = apply_all or asym_flags["r"] is True
        apply_kappa = apply_all or asym_flags["kappa"] is True
        baseline_pair[(Trait.FORCEFUL, Trait.NORMAL)] = PairParams(
            a=base_pair.a * (strategy.alpha if apply_a else 1.0),
            b=base_pair.b * ((1.0 - strategy.alpha) if apply_b else 1.0),
            k=base_pair.k * (strategy.alpha if apply_k else 1.0),
            kappa=base_pair.kappa * (strategy.alpha if apply_kappa else 1.0),
            r_scale=base_pair.r_scale * (strategy.alpha if apply_r else 1.0),
        )
        baseline_pair[(Trait.NORMAL, Trait.FORCEFUL)] = PairParams(
            a=base_pair.a * ((1.0 - strategy.alpha) if apply_a else 1.0),
            b=base_pair.b * (strategy.alpha if apply_b else 1.0),
            k=base_pair.k * ((1.0 - strategy.alpha) if apply_k else 1.0),
            kappa=base_pair.kappa * ((1.0 - strategy.alpha) if apply_kappa else 1.0),
            r_scale=base_pair.r_scale * ((1.0 - strategy.alpha) if apply_r else 1.0),
        )
    return agent_params, PairParamsTable(baseline_pair)
