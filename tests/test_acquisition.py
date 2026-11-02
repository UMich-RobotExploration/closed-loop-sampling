"""Acquisition scores: the Gram forms equal the explicit-covariance forms.

On small synthetic ensembles, builds the sample covariance C-hat explicitly
and checks that a_optimal_score reproduces u_A(j) = ||C-hat_{:,j}||^2 /
(C-hat_jj + R) (Eq. 2) and that d_optimal_score reproduces the marginal
variance C-hat_jj (the surrogate whose argmax equals Eq. 1's).
"""
from __future__ import annotations

import numpy as np

from closedloop_sampling.sampling.acquisition import (
    a_optimal_score,
    d_optimal_score,
    u_d_information_gain,
)
from closedloop_sampling.sampling.uncertainty import anomalies, gram, marginal_variance


def explicit_forms(X, cand, R):
    N = X.shape[1]
    C = X @ X.T / (N - 1)                           # explicit sample covariance
    u_a = np.array([(C[:, j] ** 2).sum() / (C[j, j] + R) for j in cand])
    var = np.array([C[j, j] for j in cand])
    return u_a, var


def test_gram_form_equals_explicit():
    rng = np.random.default_rng(0)
    for trial in range(10):
        n_cells = int(rng.integers(20, 120))
        N = int(rng.integers(5, 40))
        A = rng.standard_normal((N, n_cells))
        X = anomalies(A)
        cand = rng.choice(n_cells, size=min(15, n_cells), replace=False)
        R = float(rng.uniform(0.01, 1.0))
        u_a_ref, var_ref = explicit_forms(X, cand, R)
        assert np.allclose(a_optimal_score(X, cand, R), u_a_ref, rtol=1e-10)
        assert np.allclose(d_optimal_score(X, cand, R), var_ref, rtol=1e-10)
        assert np.allclose(marginal_variance(X, cand), var_ref, rtol=1e-10)


def test_gram_matrix_identity():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((25, 60))
    X = anomalies(A)
    G = gram(X)
    assert G.shape == (25, 25)
    j = 7
    N = 25
    lhs = (X[j] @ G @ X[j]) / (N - 1) ** 2
    C_col = X @ X[j] / (N - 1)                      # C-hat_{:, j}
    assert np.isclose(lhs, (C_col ** 2).sum(), rtol=1e-10)


def test_d_surrogate_argmax_equals_information_gain():
    rng = np.random.default_rng(2)
    A = rng.standard_normal((25, 200))
    X = anomalies(A)
    cand = np.arange(200)
    R = 0.04
    var = d_optimal_score(X, cand, R)
    gain = u_d_information_gain(var, R)
    assert int(np.argmax(var)) == int(np.argmax(gain))
    order_v = np.argsort(var)
    order_g = np.argsort(gain)
    assert (order_v == order_g).all()               # monotone transform


def test_anomalies_zero_mean():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((12, 30)) + 5.0
    X = anomalies(A)
    assert np.allclose(X.sum(1), 0.0, atol=1e-10)
