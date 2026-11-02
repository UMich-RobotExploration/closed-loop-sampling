"""Random policy: uniform choice over each vehicle's reachable set.

Implements Sec. IV (Strategies): per cycle, each vehicle moves to a uniformly
random feasible cell. The placement generator is seeded
numpy.random.default_rng(rseed*9991 + cycle) — the production formula — so the
draw is paired by (chain seed, cycle) across compared configurations. Reads no
ensemble.
"""
from __future__ import annotations

import numpy as np

from closedloop_sampling.sampling.policies.base import Policy, register

PLACEMENT_SEED_STRIDE = 9991


@register("random")
class RandomPolicy(Policy):
    moving = True
    uses_ensemble = False

    def plan(self, ctx):
        rng = np.random.default_rng(self.chain.rseed * PLACEMENT_SEED_STRIDE
                                    + ctx.cycle)
        wet = self.chain.wet
        new_fleet = ctx.fleet.copy()
        chosen_field = []
        for a in range(self.chain.n_fleet):
            if ctx.feas[a]:
                jf = int(ctx.cand_field[int(rng.choice(ctx.feas[a]))])
                new_fleet[a] = wet[jf]
                if jf not in chosen_field:
                    chosen_field.append(jf)
        return chosen_field, new_fleet
