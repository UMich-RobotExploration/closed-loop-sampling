"""Port-parity proof for SELECTLOCATIONS.

The reference implementation below is a verbatim copy of the production
greedy (`field_variance_greedy`, the author's own research code, with cluster
paths and context removed), covering BOTH criteria exactly as the campaigns
ran them: "trace" (Eq. 2, the A-optimal arm) and "var" (the variance
surrogate, the D-optimal arm — the research driver invoked the same function
with criterion="var"). One marked change: the final conditioned planning
anomalies Xw are also returned, so the port's conditioning can be compared
state-for-state.

The test runs ~50 randomized instances per criterion (N = 25 members, fleets
{5, 50}, varying candidate counts and feasibility structures, plus degenerate
tie instances with duplicated candidate columns) and requires identical chosen
(vehicle, cell) sequences, per-placement scores equal to 1e-12, and final
conditioned anomalies equal to 1e-12.
"""
from __future__ import annotations

import numpy as np
import pytest

from closedloop_sampling.sampling.selection import select_locations


# ---------------------------------------------------------------------------
# Reference: production field_variance_greedy, verbatim (one marked change:
# Xw is returned alongside the original dict).
# ---------------------------------------------------------------------------
def field_variance_greedy_reference(Xp, cand_field, feas, R, n_fleet,
                                    criterion="trace"):
    N = Xp.shape[1]
    Xw = Xp.copy()
    cand_field = np.asarray(cand_field)
    taken = set(); placed = set(); chosen = []; score_at_place = []

    def dtrp_over(cols):                              # cols = field indices (m,)
        Xc = Xw[cols]                                 # (m, N)
        if criterion == "var":                        # D-opt-equivalent: marginal variance only
            return (Xc ** 2).sum(1) / (N - 1)
        S = Xw.T @ Xw                                 # (N, N)
        num = np.einsum("ij,jk,ik->i", Xc, S, Xc)     # x'_c^T S x'_c for each c  (m,)
        var_c = (Xc ** 2).sum(1) / (N - 1)
        return num / ((N - 1) ** 2 * (var_c + R))     # (m,)

    for _ in range(n_fleet):
        avail_pos = [p for p in range(len(cand_field)) if cand_field[p] not in taken]
        if not avail_pos:
            break
        avail_field = cand_field[avail_pos]
        s = dtrp_over(avail_field)
        score_by_pos = {p: s[i] for i, p in enumerate(avail_pos)}
        best = None                                   # (score, auv, pos)
        for a in range(n_fleet):
            if a in placed:
                continue
            cand = [(score_by_pos[p], a, p) for p in feas[a] if p in score_by_pos]
            if not cand:
                continue
            top = max(cand)
            if best is None or top[0] > best[0]:
                best = top
        if best is None:
            break
        sc, a, p = best
        j = int(cand_field[p]); chosen.append((a, j)); score_at_place.append(float(sc))
        taken.add(j); placed.add(a)
        xj = Xw[j]; var_j = (xj @ xj) / (N - 1); d = var_j + R
        K = (Xw @ xj) / ((N - 1) * d); alpha = 1.0 / (1.0 + np.sqrt(R / d))
        Xw -= alpha * np.outer(K, xj)

    return dict(chosen=chosen, chosen_field=np.array([j for _, j in chosen], int),
                score_at_place=score_at_place,
                conditioned_anomalies=Xw)             # MARKED CHANGE: Xw returned


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def make_instance(rng, n_fleet, n_field=400, n_members=25, ties=False):
    Xp = rng.standard_normal((n_field, n_members))
    m = int(rng.integers(max(2 * n_fleet, 10), min(n_field, 8 * n_fleet + 20)))
    cand_field = rng.choice(n_field, size=m, replace=False)
    if ties:
        # duplicate anomaly rows across candidate pairs -> exactly tied scores
        for i in range(0, min(10, m - 1), 2):
            Xp[cand_field[i + 1]] = Xp[cand_field[i]]
    feas = []
    for _ in range(n_fleet):
        k = int(rng.integers(1, max(2, m // 2)))
        feas.append(sorted(rng.choice(m, size=k, replace=False).tolist()))
    R = float(rng.choice([0.04, 0.0203717216028506, 0.5]))
    return Xp, cand_field, feas, R


CRITERIA = [("a_optimal", "trace"), ("d_optimal", "var")]


@pytest.mark.parametrize("port_name,ref_name", CRITERIA)
@pytest.mark.parametrize("n_fleet", [5, 50])
@pytest.mark.parametrize("ties", [False, True])
def test_parity(port_name, ref_name, n_fleet, ties):
    rng = np.random.default_rng(12345 + n_fleet + (1 if ties else 0)
                                + (100 if port_name == "d_optimal" else 0))
    n_instances = 13 if not ties else 12          # 2*2*2*(13|12) = 100 per criterion pair
    for i in range(n_instances):
        Xp, cand, feas, R = make_instance(rng, n_fleet, ties=ties)
        ref = field_variance_greedy_reference(Xp.copy(), cand, feas, R, n_fleet,
                                              criterion=ref_name)
        got = select_locations(Xp.copy(), cand, feas, R, n_fleet,
                               criterion=port_name, return_anomalies=True)
        assert got["chosen"] == ref["chosen"], (
            f"instance {i}: chosen sequences differ "
            f"(port {got['chosen'][:5]}... vs ref {ref['chosen'][:5]}...)")
        assert np.allclose(got["score_at_place"], ref["score_at_place"],
                           atol=1e-12, rtol=0), f"instance {i}: scores differ"
        assert np.allclose(got["conditioned_anomalies"],
                           ref["conditioned_anomalies"], atol=1e-12, rtol=0), \
            f"instance {i}: conditioned anomalies differ"


def test_parity_degenerate_feasibility():
    """All vehicles share one feasible candidate; empty feasibility lists."""
    rng = np.random.default_rng(7)
    Xp = rng.standard_normal((50, 25))
    cand = np.arange(20)
    feas = [[3]] * 4 + [[]] * 2
    for port_name, ref_name in CRITERIA:
        ref = field_variance_greedy_reference(Xp.copy(), cand, feas, 0.04, 6,
                                              criterion=ref_name)
        got = select_locations(Xp.copy(), cand, feas, 0.04, 6,
                               criterion=port_name, return_anomalies=True)
        assert got["chosen"] == ref["chosen"]
        assert np.allclose(got["conditioned_anomalies"],
                           ref["conditioned_anomalies"], atol=1e-12, rtol=0)
