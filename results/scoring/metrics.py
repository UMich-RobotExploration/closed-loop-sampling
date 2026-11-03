"""Final-window campaign scores and the skill fraction, in the paper's
conventions, computed from the released score tables only.

Conventions (Sec. V scoring):
  score(run, W)    = mean posterior (analysis-ensemble mean) RMSE over the last
                     W of 40 cycles; the paper window is W = 8, W = 6 is carried
                     alongside.
  per-IC value     = mean over the three paired seeds of the same-seed score.
  per-IC delta     = mean over seeds of the SAME-SEED difference (the seeds are
                     paired across policies by common random numbers).
  config mean      = mean of the per-IC values over the 10 sealed ICs.
  skill fraction   F = (S_free - S) / (S_free - S_recon), formed PER IC with
                     that IC's free-running score S_free and autoencoder-floor
                     score S_recon (both window means of the per-cycle
                     references), then averaged; the map is affine only per IC.

All loaders read results/scores/; nothing here touches production trees.
"""
from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np

REGIONS = ("mab", "ccs")
VARIABLES = ("sst", "sss")
POLICIES = ("static", "random", "lawnmower", "d_optimal", "a_optimal")
FLEETS_MAIN = (20, 50, 200)
FLEETS_SWEEP = (10, 20, 35, 50, 100, 200)
SEEDS = (0, 1, 2)
WINDOWS = {"last8": 8, "last6": 6}


def scores_dir(explicit: str | None = None) -> Path:
    return Path(explicit) if explicit else Path(__file__).resolve().parents[1] / "scores"


def load_runs(root: Path) -> dict:
    """runs.csv -> {(region, variable, fleet, ic, policy, seed): row dict}."""
    out = {}
    with open(root / "runs.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["region"], r["variable"], int(r["fleet"]), r["ic"],
                   r["policy"], int(r["seed"]))
            out[key] = dict(rmse_last8=float(r["rmse_last8"]),
                            rmse_last6=float(r["rmse_last6"]),
                            scale_eff=float(r["scale_eff"]),
                            wall_s=(int(r["wall_s"]) if r["wall_s"] else None))
    return out


def _load_series(path: Path, value_col: str) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault((r["region"], r["variable"], r["ic"]),
                           [None] * 40)[int(r["cycle"])] = float(r[value_col])
    return {k: np.asarray(v, float) for k, v in out.items()}


def load_free_running(root: Path) -> dict:
    """free_running.csv -> {(region, variable, ic): (40,) per-cycle RMSE}."""
    return _load_series(root / "references/free_running.csv", "rmse")


def load_ae_floor(root: Path) -> dict:
    """ae_floor.csv -> {(region, variable, ic): (40,) per-cycle floor}."""
    return _load_series(root / "references/ae_floor.csv", "floor")


def load_per_cycle(root: Path, region: str, variable: str) -> dict:
    """runs_per_cycle_<region>_<variable>.csv.gz ->
    {(fleet, ic, policy, seed): dict of (40,) arrays prior/post/floor/n_obs}."""
    out = {}
    with gzip.open(root / f"runs_per_cycle_{region}_{variable}.csv.gz", "rt",
                   newline="") as fh:
        for r in csv.DictReader(fh):
            key = (int(r["fleet"]), r["ic"], r["policy"], int(r["seed"]))
            d = out.setdefault(key, dict(prior=[None] * 40, post=[None] * 40,
                                         floor=[None] * 40, n_obs=[None] * 40))
            c = int(r["cycle"])
            d["prior"][c] = float(r["prior_rmse"])
            d["post"][c] = float(r["post_rmse"])
            d["floor"][c] = float(r["floor"])
            d["n_obs"][c] = int(r["n_obs"])
    return {k: {n: np.asarray(v, float) for n, v in d.items()}
            for k, d in out.items()}


def ics_for(root: Path, region: str) -> list[str]:
    """The sealed test ICs present in the release, sorted."""
    ics = set()
    with open(root / "references/spread_error_r.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["region"] == region:
                ics.add(r["ic"])
    return sorted(ics)


def window_mean(series: np.ndarray, window: str) -> float:
    return float(np.mean(series[-WINDOWS[window]:]))


def run_score(runs: dict, region, variable, fleet, ic, policy, seed,
              window: str = "last8") -> float:
    return runs[(region, variable, fleet, ic, policy, seed)][f"rmse_{window}"]


def per_ic_score(runs, region, variable, fleet, ic, policy, window="last8"):
    """Mean over the three paired seeds of the same-seed window score."""
    return float(np.mean([run_score(runs, region, variable, fleet, ic, policy, s,
                                    window) for s in SEEDS]))


def per_ic_delta(runs, region, variable, fleet, ic, a, b, window="last8"):
    """Mean over seeds of the same-seed score difference a - b (CRN pairing)."""
    return float(np.mean([run_score(runs, region, variable, fleet, ic, a, s, window)
                          - run_score(runs, region, variable, fleet, ic, b, s, window)
                          for s in SEEDS]))


def config_mean(runs, root, region, variable, fleet, policy, window="last8"):
    """Mean of the per-IC values over the sealed ICs."""
    return float(np.mean([per_ic_score(runs, region, variable, fleet, ic, policy,
                                       window) for ic in ics_for(root, region)]))


def skill_fraction(runs, free, floor, region, variable, fleet, policy,
                   window="last8") -> dict:
    """F = (S_free - S) / (S_free - S_recon), per IC; see the module docstring."""
    out = {}
    for ic in sorted({k[2] for k in free if k[:2] == (region, variable)}):
        s = per_ic_score(runs, region, variable, fleet, ic, policy, window)
        s_free = window_mean(free[(region, variable, ic)], window)
        s_recon = window_mean(floor[(region, variable, ic)], window)
        out[ic] = (s_free - s) / (s_free - s_recon)
    return out
