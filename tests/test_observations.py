"""Observation draw: bit-exact reproduction of the production formula."""
from __future__ import annotations

import numpy as np

from closedloop_sampling.observations import (
    IC_KEY_STRIDE,
    OBS_SEED_BASE,
    ic_key,
    noise_variance,
    obs_rng,
    observe,
    sigma_for,
)


def test_sigma_literals():
    assert sigma_for("temp") == 0.2
    assert sigma_for("sst") == 0.2
    assert sigma_for("salt") == 0.2 * 3.6912 / 5.1723      # exact executed expression
    assert sigma_for("salt") == 0.14272954004988112        # its full float repr
    assert sigma_for("sss") == sigma_for("salt")
    assert noise_variance("salt") == (0.2 * 3.6912 / 5.1723) ** 2
    assert noise_variance("temp") == 0.2 ** 2      # the executed R (float square)


def test_ic_key_days_since_2000():
    assert ic_key("2022-08-02T12:00") == 8249
    assert ic_key("2000-01-01T12:00") == 0
    assert ic_key("2000-01-02T00:00") == 1
    assert ic_key("2016-07-27T12:00") == int(
        (np.datetime64("2016-07-27") - np.datetime64("2000-01-01")).astype(int))


def test_draw_bit_exact_vs_production_formula():
    """y = truth[cells] + default_rng(70000 + ic_key*1009 + c + rseed)
    .normal(0, sigma, n) — exactly."""
    rng_cases = [("2022-08-02T12:00", 0, 0), ("2022-08-02T12:00", 17, 2),
                 ("2016-07-27T12:00", 39, 1), ("2023-09-22T12:00", 5, 0)]
    truth = np.arange(400, dtype=float).reshape(20, 20) / 7.0
    cells = np.array([[2, 3], [11, 7], [19, 19], [0, 0], [5, 14]])
    for ic, c, rseed in rng_cases:
        for target in ("temp", "salt"):
            sigma = sigma_for(target)
            key = ic_key(ic)
            got = observe(truth, cells, sigma, obs_rng(key, c, rseed))
            ref_rng = np.random.default_rng(
                OBS_SEED_BASE + key * IC_KEY_STRIDE + c + rseed)
            ref = truth[cells[:, 0], cells[:, 1]] + ref_rng.normal(0, sigma, 5)
            assert np.array_equal(got, ref), (ic, c, rseed, target)


def test_one_draw_is_member_shared():
    """The draw has no member dimension: the API returns one y per cycle."""
    truth = np.zeros((4, 4))
    cells = np.array([[1, 1], [2, 2]])
    y = observe(truth, cells, 0.2, obs_rng(100, 0, 0))
    assert y.shape == (2,)
