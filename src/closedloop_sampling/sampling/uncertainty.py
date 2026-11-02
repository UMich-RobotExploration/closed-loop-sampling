"""Ensemble uncertainty primitives for the acquisition scores.

Implements Sec. III (Sampling), the ensemble-anomaly quantities behind
Eqs. (1)-(2): the anomaly matrix X, the member Gram matrix G = X^T X, and
per-cell marginal variances. The sample covariance C-hat = X X^T / (N-1) is
NEVER formed — every score is evaluated through X or G (n_cells x N and
N x N objects; the n_cells x n_cells covariance would be prohibitive and is
unnecessary). numpy-only.
"""
from __future__ import annotations

import numpy as np


def anomalies(member_fields: np.ndarray) -> np.ndarray:
    """Member anomalies over the field of interest.

    member_fields : (N, n_cells) — each member's target field at the scored
                    cells (wet cells, in wet row-major order).
    Returns X of shape (n_cells, N): column m is member m's anomaly from the
    ensemble mean. C-hat = X X^T / (N-1) implicitly; never materialized.
    """
    A = np.asarray(member_fields)
    return (A - A.mean(0)).T.copy()


def gram(X: np.ndarray) -> np.ndarray:
    """Member Gram matrix G = X^T X, shape (N, N).

    The identity behind the A-optimal score's fast form:
    sum_k cov(k, j)^2 = x_j^T G x_j / (N-1)^2.
    """
    return X.T @ X


def marginal_variance(X: np.ndarray, cells=None) -> np.ndarray:
    """Per-cell marginal variance C-hat_jj = ||x_j||^2 / (N-1).

    `cells` optionally restricts to a subset of row indices.
    """
    Xc = X if cells is None else X[np.asarray(cells)]
    N = X.shape[1]
    return (Xc ** 2).sum(1) / (N - 1)
