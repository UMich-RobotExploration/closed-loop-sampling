"""Acquisition scores for candidate observation cells.

Implements Sec. III (Sampling), Eqs. (1)-(2), exactly as the production runs
evaluated them on the RUNNING conditioned anomalies Xw (see selection.py for
the planning update between placements). numpy-only.

A-optimal (Eq. 2, Gram form — as run):
    u_A(j) = ||C-hat_{:,j}||^2 / (C-hat_jj + R)
           = x_j^T G x_j / ((N-1)^2 (C-hat_jj + R)),   G = Xw^T Xw.
The Gram evaluation is mathematically identical to the explicit-covariance
form and O(m N^2) instead of O(n_cells m N).

D criterion (as run): score(j) = C-hat_jj, the marginal variance. This is a
monotone surrogate of Eq. (1), u_D(j) = 0.5 * log(1 + C-hat_jj / R): for
homoscedastic R the argmax is identical, so greedy selection under the
surrogate reproduces greedy selection under Eq. (1) exactly. The recorded
scores are variances, not information gains.
"""
from __future__ import annotations

import numpy as np


def a_optimal_score(Xw: np.ndarray, cand_cells: np.ndarray, R: float) -> np.ndarray:
    """Eq. (2) over the candidate cells, Gram form, on the (conditioned)
    anomalies Xw (n_cells, N). Returns (m,) scores."""
    Xc = Xw[np.asarray(cand_cells)]              # (m, N)
    N = Xw.shape[1]
    S = Xw.T @ Xw                                # (N, N) Gram
    num = np.einsum("ij,jk,ik->i", Xc, S, Xc)    # x_j^T G x_j per candidate
    var_c = (Xc ** 2).sum(1) / (N - 1)
    return num / ((N - 1) ** 2 * (var_c + R))


def d_optimal_score(Xw: np.ndarray, cand_cells: np.ndarray, R: float) -> np.ndarray:
    """The as-run D criterion: marginal variance C-hat_jj (argmax-equivalent
    to Eq. (1) for homoscedastic R; see module docstring). R is accepted for
    signature symmetry and does not enter the score."""
    Xc = Xw[np.asarray(cand_cells)]
    N = Xw.shape[1]
    return (Xc ** 2).sum(1) / (N - 1)


def u_d_information_gain(variance: np.ndarray, R: float) -> np.ndarray:
    """Eq. (1) proper, u_D = 0.5 * log(1 + C-hat_jj / R) — provided for
    reporting; selection uses the variance surrogate above."""
    return 0.5 * np.log1p(np.asarray(variance) / R)


SCORES = {"a_optimal": a_optimal_score, "d_optimal": d_optimal_score}
