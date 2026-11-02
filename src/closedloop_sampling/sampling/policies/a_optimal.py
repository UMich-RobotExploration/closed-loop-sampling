"""A-optimal policy: greedy trace-reduction placement on the forecast ensemble.

Implements Sec. III (Sampling), Eq. (2) + GreedyAssign: acquisition score =
per-candidate reduction of the posterior target-field covariance trace,
evaluated on the decoded forecast-ensemble anomalies, with the serial
square-root planning update between placements.
"""
from __future__ import annotations

from closedloop_sampling.sampling.policies.base import Policy, register
from closedloop_sampling.sampling.selection import select_locations
from closedloop_sampling.sampling.uncertainty import anomalies


class _GreedyPolicy(Policy):
    moving = True
    uses_ensemble = True
    criterion: str

    def plan(self, ctx):
        chain = self.chain
        wet = chain.wet
        # Anomalies of the member target fields at the wet cells (production
        # form: m_T[:, wet_rows, wet_cols], mean removed, transposed).
        A = ctx.member_target_fields[:, wet[:, 0], wet[:, 1]]
        Xp = anomalies(A)
        res = select_locations(Xp, ctx.cand_field, ctx.feas, ctx.R,
                               chain.n_fleet, criterion=self.criterion)
        new_fleet = ctx.fleet.copy()
        for a, jf in res["chosen"]:
            new_fleet[a] = wet[jf]
        ctx.extras["score_at_place"] = res["score_at_place"]
        return list(res["chosen_field"]), new_fleet


@register("a_optimal")
class AOptimalPolicy(_GreedyPolicy):
    criterion = "a_optimal"
