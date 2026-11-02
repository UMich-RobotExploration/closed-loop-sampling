"""Greedy placement with the serial square-root planning update.

Implements Sec. III (Sampling), SELECTLOCATIONS (Alg. GreedyAssign): place
n_fleet vehicles greedily over feasible (vehicle, candidate-cell) pairs,
scoring with Eq. (2) (A-optimal) or the variance surrogate of Eq. (1)
(D criterion); after every placement the PLANNING copy of the anomalies is
conditioned with a serial square-root rank-one downdate (it conditions the
acquisition scores only — the state ensemble is updated by observation-guided
diffusion, never here):

    d = C-hat_jj + R;  K = Xw x_j / ((N-1) d);  alpha = 1 / (1 + sqrt(R/d));
    Xw <- Xw - alpha K x_j^T.

Port notes (behavior-preserving, verified by tests/test_selection_parity.py):
  - candidate order is the production order — wet cells in row-major order
    from numpy.where(mask), with `cand_cells` indexing that array;
  - tie-breaking is the production tuple rule: per vehicle, max over
    (score, vehicle, position) tuples; across vehicles, strict > keeps the
    earlier vehicle on exact ties;
  - selection is fully deterministic (no RNG anywhere).
numpy-only.
"""
from __future__ import annotations

import numpy as np

from closedloop_sampling.sampling.acquisition import a_optimal_score, d_optimal_score

_SCORERS = {"a_optimal": a_optimal_score, "d_optimal": d_optimal_score}


def select_locations(Xp, cand_field, feas, R, n_fleet, criterion="a_optimal"):
    """Greedy assignment of n_fleet vehicles to candidate cells.

    Xp         : (n_field, N) target-field anomalies over the field of
                 interest (wet cells, row-major).
    cand_field : (M,) field-indices of the reachable candidate cells.
    feas       : list[n_fleet]; feas[a] = positions (into cand_field)
                 reachable by vehicle a.
    R          : observation-noise variance.
    criterion  : "a_optimal" (Eq. 2) | "d_optimal" (variance surrogate of
                 Eq. 1; conditioning/downdate/assignment identical).

    Returns dict: chosen (list of (vehicle, field_index)), chosen_field
    (array of field indices), score_at_place (marginal score credited at each
    placement — a u_A value for a_optimal, a variance for d_optimal).
    """
    score_fn = _SCORERS[criterion]
    N = Xp.shape[1]
    Xw = Xp.copy()
    cand_field = np.asarray(cand_field)
    taken = set()
    placed = set()
    chosen = []
    score_at_place = []

    for _ in range(n_fleet):
        # score every still-available candidate once this round
        avail_pos = [p for p in range(len(cand_field)) if cand_field[p] not in taken]
        if not avail_pos:
            break
        avail_field = cand_field[avail_pos]
        s = score_fn(Xw, avail_field, R)
        score_by_pos = {p: s[i] for i, p in enumerate(avail_pos)}
        # global best (vehicle, candidate): each unplaced vehicle's best
        # feasible-available candidate
        best = None                                   # (score, vehicle, pos)
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
        j = int(cand_field[p])
        chosen.append((a, j))
        score_at_place.append(float(sc))
        taken.add(j)
        placed.add(a)
        # serial square-root rank-one conditioning on cell j (planning update)
        xj = Xw[j]
        var_j = (xj @ xj) / (N - 1)
        d = var_j + R
        K = (Xw @ xj) / ((N - 1) * d)
        alpha = 1.0 / (1.0 + np.sqrt(R / d))
        Xw -= alpha * np.outer(K, xj)

    return dict(chosen=chosen,
                chosen_field=np.array([j for _, j in chosen], int),
                score_at_place=score_at_place)
