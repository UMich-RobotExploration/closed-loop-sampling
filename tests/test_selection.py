"""SELECTLOCATIONS invariants: assignment validity, feasibility, and the
strict variance reduction of the serial square-root planning update."""
from __future__ import annotations

import numpy as np

from closedloop_sampling.sampling.selection import select_locations
from closedloop_sampling.sampling.uncertainty import anomalies, marginal_variance


def make_instance(seed=0, n_field=300, N=25, n_fleet=12, m=80):
    rng = np.random.default_rng(seed)
    Xp = anomalies(rng.standard_normal((N, n_field)))
    cand_field = rng.choice(n_field, size=m, replace=False)
    feas = [sorted(rng.choice(m, size=int(rng.integers(3, m // 2)),
                              replace=False).tolist())
            for _ in range(n_fleet)]
    return Xp, cand_field, feas


def test_assignment_invariants():
    for criterion in ("a_optimal", "d_optimal"):
        Xp, cand_field, feas = make_instance(seed=10)
        res = select_locations(Xp, cand_field, feas, R=0.04, n_fleet=12,
                               criterion=criterion)
        vehicles = [a for a, _ in res["chosen"]]
        cells = [j for _, j in res["chosen"]]
        assert len(set(vehicles)) == len(vehicles)      # each vehicle one cell
        assert len(set(cells)) == len(cells)            # each cell one vehicle
        # feasibility respected: every chosen cell is in its vehicle's set
        pos_of = {int(j): p for p, j in enumerate(cand_field)}
        for a, j in res["chosen"]:
            assert pos_of[j] in feas[a]
        assert len(res["chosen"]) == len(res["score_at_place"])


def test_downdate_strictly_reduces_conditioned_variance():
    Xp, cand_field, feas = make_instance(seed=11)
    res = select_locations(Xp, cand_field, feas, R=0.04, n_fleet=12,
                           criterion="a_optimal", return_anomalies=True)
    Xw = res["conditioned_anomalies"]
    var_before = marginal_variance(Xp)
    var_after = marginal_variance(Xw)
    chosen = res["chosen_field"]
    # the chosen cells' variances strictly decrease
    assert (var_after[chosen] < var_before[chosen]).all()
    # the total conditioned trace strictly decreases, and nothing inflates
    assert var_after.sum() < var_before.sum()
    assert (var_after <= var_before + 1e-12).all()


def test_scores_nonincreasing_under_a_optimal_conditioning():
    """Each placement conditions the planning copy, so the credited A-optimal
    score cannot increase when the same cell neighborhood is re-scored; the
    recorded sequence is produced by argmax over a shrinking objective."""
    Xp, cand_field, feas = make_instance(seed=12, n_fleet=8)
    # full feasibility: every vehicle can take every candidate — the score
    # sequence is then globally greedy and nonincreasing
    feas = [list(range(len(cand_field)))] * 8
    res = select_locations(Xp, cand_field, feas, R=0.04, n_fleet=8,
                           criterion="a_optimal")
    s = res["score_at_place"]
    assert all(s[i] >= s[i + 1] - 1e-12 for i in range(len(s) - 1))


def test_empty_feasibility_vehicle_is_skipped():
    Xp, cand_field, feas = make_instance(seed=13, n_fleet=6)
    feas[2] = []
    res = select_locations(Xp, cand_field, feas, R=0.04, n_fleet=6)
    assert all(a != 2 for a, _ in res["chosen"])
    assert len(res["chosen"]) == 5
