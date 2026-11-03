#!/usr/bin/env python
"""Truth access for the closed loop and the scoring.

Interchangeable backends, all serving per-chain callables:
  - TruthBundle: the released per-region npz written by
    datasets/export_truth_bundle.py (10 sealed ICs x 41 six-hour states x
    {SST, SSS, surface u, v} + wet mask + metric cell sizes, model crop).
    The release form.
  - LatentBundle: the released sealed-IC raw-latent bundle written by
    datasets/export_latent_bundle.py (fp16; the IC latent and the per-cycle
    autoencoder floor read it, so the loop runs without the latents store).
  - StoreTruth / latent_source: read the same states directly from the
    preprocessed and latents stores (what the production loop did). Need the
    multi-terabyte stores.

Bundle resolution (find_bundle): $CLOSEDLOOP_ARTIFACTS/<kind>_bundles/
<region>.npz first (the downloaded dataset repository), then the
$CLOSEDLOOP_DATA_ROOT locations the exporters write to.

State indexing: state s corresponds to lead s*6 h from the IC (s = 0 is the
IC state; cycle c of the loop scores state c+1).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

STEP_HOURS = 6


def find_bundle(kind: str, region: str) -> Path | None:
    """Locate a released bundle: artifacts repository first, data root second.

    kind is "truth" or "latent". Returns None when neither location has it.
    """
    assert kind in ("truth", "latent")
    candidates = []
    art = os.environ.get("CLOSEDLOOP_ARTIFACTS")
    if art:
        candidates.append(Path(art) / f"{kind}_bundles" / f"{region}.npz")
    data = os.environ.get("CLOSEDLOOP_DATA_ROOT")
    if data:
        sub = {"truth": "truth", "latent": "latents"}[kind]
        candidates.append(Path(data) / sub / f"{region}_{kind}_bundle.npz")
    for p in candidates:
        if p.exists():
            return p
    return None


class TruthBundle:
    """Reader for <region>_truth_bundle.npz."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.mask = z["mask"]
        self.ic_datetimes = [str(s) for s in z["ic_datetimes"]]
        self.lead_hours = z["lead_hours"]
        self._fields = {k: z[k] for k in ("sst", "sss", "u", "v")}
        # metric cell sizes (newer bundles; lets the loop skip the store)
        self.dx_col_m = float(z["dx_col_m"]) if "dx_col_m" in z.files else None
        self.dy_row_m = float(z["dy_row_m"]) if "dy_row_m" in z.files else None

    def chain_truth(self, ic_datetime: str, target: str):
        """Return state_index -> (H, W) physical target field for one IC."""
        key = {"temp": "sst", "sst": "sst", "salt": "sss", "sss": "sss"}[target.lower()]
        tag = ic_datetime[:13]
        hits = [i for i, s in enumerate(self.ic_datetimes) if s[:13] == tag]
        if len(hits) != 1:
            raise ValueError(f"IC {ic_datetime} matched {len(hits)} bundle entries")
        block = self._fields[key][hits[0]]                 # (41, H, W)

        def field(state: int) -> np.ndarray:
            return block[state].astype(float)
        return field


class LatentBundle:
    """Reader for <region>_latent_bundle.npz (fp16 raw truth latents)."""

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)
        self.ic_datetimes = [str(s) for s in z["ic_datetimes"]]
        self.lead_hours = z["lead_hours"]
        self._latents = z["latents"]                  # (I, 41, C, h, w) fp16

    def chain_latents(self, ic_datetime: str):
        """Return state_index -> RAW (C, h, w) float32 latent for one IC."""
        tag = ic_datetime[:13]
        hits = [i for i, s in enumerate(self.ic_datetimes) if s[:13] == tag]
        if len(hits) != 1:
            raise ValueError(f"IC {ic_datetime} matched {len(hits)} bundle entries")
        block = self._latents[hits[0]]

        def latent(state: int) -> np.ndarray:
            return np.asarray(block[state], dtype=np.float32)
        return latent


class StoreTruth:
    """Reads truth states from the preprocessed store (production path)."""

    def __init__(self, region_cfg, ds):
        self.cfg = region_cfg
        self.ds = ds
        self.crop = region_cfg.metric_crop
        self.mask = ds["mask_rho"].values[self.crop] > 0.5
        self.surface = region_cfg.channel_indices()["surface_s_rho_index"]
        self.cadence = region_cfg.timesplit.cadence_hours

    def chain_truth(self, ic_store_index: int, target: str):
        var = {"temp": "temp", "sst": "temp", "salt": "salt", "sss": "salt"}[target.lower()]

        def field(state: int) -> np.ndarray:
            lead_h = state * STEP_HOURS
            assert lead_h % self.cadence == 0
            tn = ic_store_index + lead_h // self.cadence
            f = self.ds[var].isel(ocean_time=tn, s_rho=self.surface)\
                .values[self.crop].astype(float)
            f[~self.mask] = np.nan
            return f
        return field


def latent_source(latents_arr, ocean_time, ic_datetime: str, cadence_hours: int):
    """Return (ic_store_index, state_index -> RAW (C, h, w) latent) for one IC
    from the latents store (shared ocean_time axis with the preprocessed
    store). Used for the IC latent and the per-cycle AE floor."""
    times = np.asarray(ocean_time).astype("datetime64[h]")
    hits = np.where(times == np.datetime64(ic_datetime[:13], "h"))[0]
    if len(hits) != 1:
        raise ValueError(f"IC {ic_datetime} matched {len(hits)} latent steps")
    ic_idx = int(hits[0])

    def latent(state: int) -> np.ndarray:
        lead_h = state * STEP_HOURS
        assert lead_h % cadence_hours == 0
        return np.asarray(latents_arr[ic_idx + lead_h // cadence_hours],
                          dtype=np.float32)
    return ic_idx, latent
