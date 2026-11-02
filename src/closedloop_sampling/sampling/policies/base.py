"""Policy interface and registry for the five sampling strategies.

Implements Sec. IV (Strategies): every policy answers one question per cycle —
which reachable cell does each vehicle sample next. Policies never touch the
analysis update; they only output observation cells and the vehicles' next
positions. Uncertainty-driven policies read the forecast ensemble; blind
policies must not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import numpy as np


@dataclass
class ChainContext:
    """Chain-constant geometry and settings a policy may need at start-up."""
    wet: np.ndarray                 # (n_wet, 2) int wet cells, row-major order
    wet_lookup: dict                # (row, col) -> wet index
    mask: np.ndarray                # (H, W) bool
    n_fleet: int
    dx_col_m: float
    dy_row_m: float
    v_auv_ms: float
    reach_window_s: float
    n_cycles: int
    rseed: int
    initial_fleet: np.ndarray       # (n_fleet, 2) float — the shared coverage grid


@dataclass
class CycleContext:
    """Per-cycle inputs. `member_target_fields` is populated only for
    uncertainty-driven policies (blind policies must not read it)."""
    cycle: int
    fleet: np.ndarray               # (n_fleet, 2) float — cycle-start positions
    cand_rc: np.ndarray             # (M, 2) reachable cells (empty for static)
    cand_field: np.ndarray          # (M,) wet indices of cand_rc
    feas: list                      # feas[a] = positions into cand_rc for vehicle a
    R: float                        # observation-noise variance
    member_target_fields: Optional[np.ndarray] = None   # (N, H, W), NaN on land
    extras: dict = field(default_factory=dict)          # per-cycle policy diagnostics


class Policy:
    """Base policy. Subclasses set `moving` and implement plan()."""

    moving: bool = True
    uses_ensemble: bool = False     # True only for the uncertainty-driven pair

    def __init__(self, chain: ChainContext):
        self.chain = chain

    def start_fleet(self) -> np.ndarray:
        """Initial vehicle positions; default = the shared coverage grid."""
        return self.chain.initial_fleet.copy()

    def plan(self, ctx: CycleContext) -> Tuple[List[int], np.ndarray]:
        """Return (chosen_field, new_fleet): the wet indices to observe this
        cycle and the vehicles' next positions."""
        raise NotImplementedError


_REGISTRY: Dict[str, Type[Policy]] = {}
# Historical strategy names used by the research records.
ALIASES = {"trace": "a_optimal", "dopt": "d_optimal", "lawn": "lawnmower"}


def register(name: str):
    def deco(cls: Type[Policy]):
        _REGISTRY[name] = cls
        cls.name = name
        return cls
    return deco


def policy_names() -> List[str]:
    return sorted(_REGISTRY)


def make_policy(name: str, chain: ChainContext) -> Policy:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in _REGISTRY:
        raise ValueError(f"unknown policy {name!r}; known: {policy_names()} "
                         f"(aliases: {sorted(ALIASES)})")
    return _REGISTRY[key](chain)
