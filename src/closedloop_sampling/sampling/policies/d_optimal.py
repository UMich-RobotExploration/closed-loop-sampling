"""D-optimal policy: greedy maximum-variance placement on the forecast
ensemble.

Implements Sec. III (Sampling), Eq. (1) via its as-run variance surrogate
(argmax-equivalent for homoscedastic R; see sampling.acquisition), with the
same greedy assignment and serial square-root planning update as A-optimal.
"""
from __future__ import annotations

from closedloop_sampling.sampling.policies.a_optimal import _GreedyPolicy
from closedloop_sampling.sampling.policies.base import register


@register("d_optimal")
class DOptimalPolicy(_GreedyPolicy):
    criterion = "d_optimal"
