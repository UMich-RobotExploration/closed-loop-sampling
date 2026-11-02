"""Observations: noise model, observation-noise RNG scheme, and the single
draw site for synthetic point observations.

Implements Sec. III-C / Alg. 1 line 16: robots observe the TARGET VARIABLE
ONLY, as noisy point values of the true surface field at their arrival cells,
    y = H_p(truth) + eta,   eta ~ N(0, R I),   R = sigma^2.
Noise is drawn exactly once per cycle, shared by all ensemble members; the
same R feeds the acquisition scores, the planning update, and the guidance
misfit — one consistent noise model. numpy-only.

Observation-noise standard deviations (the executed production literals; both
regions): sigma was set once for a temperature sensor on the MAB decoded
reference fields and the other variables hold equal RELATIVE precision,
sigma_var = 0.2 * std_var / std_temp with the 4-decimal field-std literals
(std_temp 5.1723 degC, std_salt 3.6912 psu), giving
    SST  0.2 degC          (R = 0.04)
    SSS  0.14272954004988112 psu (R = 0.020371721602850616)

RNG scheme (production formula, verbatim): the per-cycle observation draw uses
    numpy.random.default_rng(70000 + ic_key*1009 + cycle + rseed)
where ic_key is the IC date's days-since-2000-01-01 integer (unique per IC
across years; the factor 1009 is a prime larger than the cycle count, keeping
(IC, cycle) keys disjoint) and rseed is the chain's observation seed. The
forecast-roll seeds are rseed-INDEPENDENT (see forecast.ensemble), so the
prior is common random numbers across policies and seeds.
"""
from __future__ import annotations

import numpy as np

# Executed production literals (equal relative precision anchored at SST).
STD_TEMP_REF = 5.1723          # MAB decoded-reference surface temperature std (degC)
STD_SALT_REF = 3.6912          # same, surface salinity (psu)
SIGMA_SST = 0.2                                        # degC
SIGMA_SSS = SIGMA_SST * STD_SALT_REF / STD_TEMP_REF    # = 0.142729540049881 psu

SIGMA_OBS = {"temp": SIGMA_SST, "sst": SIGMA_SST,
             "salt": SIGMA_SSS, "sss": SIGMA_SSS}

OBS_SEED_BASE = 70000
IC_KEY_STRIDE = 1009


def sigma_for(target: str) -> float:
    """Observation-noise standard deviation for a target variable."""
    try:
        return SIGMA_OBS[target.lower()]
    except KeyError:
        raise ValueError(f"no observation sigma for target {target!r}; "
                         f"known: {sorted(set(SIGMA_OBS))}") from None


def noise_variance(target: str) -> float:
    """R = sigma^2 for a target variable (feeds planner and misfit alike)."""
    return sigma_for(target) ** 2


def ic_key(ic_datetime: str) -> int:
    """IC key = days since 2000-01-01 of the IC date (production formula).

    `ic_datetime` is ISO8601; only the date part enters the key.
    """
    return int((np.datetime64(ic_datetime[:10]) - np.datetime64("2000-01-01"))
               .astype(int))


def obs_rng(ic_key_value: int, cycle: int, rseed: int) -> np.random.Generator:
    """The per-cycle observation-noise generator (production formula)."""
    return np.random.default_rng(
        OBS_SEED_BASE + ic_key_value * IC_KEY_STRIDE + cycle + rseed)


def observe(truth_field: np.ndarray, cells_rc: np.ndarray, sigma: float,
            rng: np.random.Generator) -> np.ndarray:
    """THE draw site: y = truth at the chosen cells + N(0, sigma^2).

    truth_field : (H, W) physical surface field of the target variable.
    cells_rc    : (K, 2) integer (row, col) observation cells.
    Returns y (K,). One call per cycle; every member's guided update receives
    this same y.
    """
    rows = np.asarray(cells_rc)[:, 0]
    cols = np.asarray(cells_rc)[:, 1]
    return truth_field[rows, cols] + rng.normal(0.0, sigma, len(rows))
