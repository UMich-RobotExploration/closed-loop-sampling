"""Static policy: fixed sensors on the shared coverage grid.

Implements Sec. IV (Strategies): the vehicles never move; every cycle observes
the same cells — the even coverage grid that also initializes every other
policy's fleet. Deterministic; reads no ensemble.
"""
from __future__ import annotations

from closedloop_sampling.sampling.policies.base import Policy, register


@register("static")
class StaticPolicy(Policy):
    moving = False
    uses_ensemble = False

    def __init__(self, chain):
        super().__init__(chain)
        self._static_field = [
            chain.wet_lookup[(int(r), int(c))] for r, c in chain.initial_fleet
        ]

    def plan(self, ctx):
        return list(self._static_field), ctx.fleet
