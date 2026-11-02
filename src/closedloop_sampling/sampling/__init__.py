"""Spatial sampling: uncertainty primitives, acquisition scores, reachability,
greedy selection, and the five policies. numpy-only (no torch).

"Sampling" in this package always means choosing observation locations;
diffusion generation lives in closedloop_sampling.forecast.
"""
from closedloop_sampling.sampling.acquisition import (
    a_optimal_score,
    d_optimal_score,
    u_d_information_gain,
)
from closedloop_sampling.sampling.reachability import advected_reachable, cell_metres
from closedloop_sampling.sampling.selection import select_locations
from closedloop_sampling.sampling.uncertainty import anomalies, gram, marginal_variance

__all__ = [
    "a_optimal_score", "d_optimal_score", "u_d_information_gain",
    "advected_reachable", "cell_metres", "select_locations",
    "anomalies", "gram", "marginal_variance",
]
