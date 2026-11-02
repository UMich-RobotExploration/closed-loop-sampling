"""Region-config gate: latent shapes, crops, channel indices, metric_crop.

The wet-cell counts of the released masks (MAB 19,150; CCS 21,240 on the
model crops) are verified in test_wet_cell_counts when the released wet-mask
files are reachable; the pure config assertions always run.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from closedloop_sampling.regions import load

EXPECTED_WET = {"mab": 19150, "ccs": 21240}


def test_latent_shapes():
    m, c = load("mab"), load("ccs")
    assert (m.latent.latent_channels, m.latent.latent_h, m.latent.latent_w) == (84, 13, 30)
    assert (c.latent.latent_channels, c.latent.latent_h, c.latent.latent_w) == (84, 22, 23)
    assert tuple(m.latent.rope_spatial_grid_end_pos) == (12, 29)
    assert tuple(c.latent.rope_spatial_grid_end_pos) == (21, 22)


def test_crops_and_metric_crop():
    m, c = load("mab"), load("ccs")
    assert m.metric_crop == (slice(1, 105), slice(1, 241))
    assert c.metric_crop == (slice(2, 178), slice(1, 185))
    assert (m.grid.cropped_h, m.grid.cropped_w) == (104, 240)
    assert (c.grid.cropped_h, c.grid.cropped_w) == (176, 184)
    for cfg in (m, c):
        e0, e1 = cfg.grid.crop_eta_slice
        x0, x1 = cfg.grid.crop_xi_slice
        assert (e1 - e0, x1 - x0) == (cfg.grid.cropped_h, cfg.grid.cropped_w)
        assert cfg.latent.latent_h == cfg.grid.cropped_h // cfg.latent.compression_per_dim
        assert cfg.latent.latent_w == cfg.grid.cropped_w // cfg.latent.compression_per_dim


def test_channel_indices():
    for name in ("mab", "ccs"):
        idx = load(name).channel_indices()
        assert idx["sst"] == 59
        assert idx["sss"] == 79
        assert idx["u_surface"] == 19
        assert idx["v_surface"] == 39
        assert idx["surface_s_rho_index"] == 19
        assert (idx["zeta"], idx["ubar"], idx["vbar"]) == (80, 81, 82)


def test_alias_and_cache_identity():
    assert load("doppio") is load("mab")


def test_cadence_and_splits():
    m, c = load("mab"), load("ccs")
    assert (m.timesplit.cadence_hours, m.timesplit.internal_stride) == (1, 6)
    assert (c.timesplit.cadence_hours, c.timesplit.internal_stride) == (6, 1)
    assert m.timesplit.test_year_range == (2022, 2024)
    assert c.timesplit.test_year_range == (2016, 2019)


@pytest.mark.weights
def test_wet_cell_counts():
    weights = os.environ.get("CLOSEDLOOP_WEIGHTS")
    if not weights:
        pytest.skip("CLOSEDLOOP_WEIGHTS not set")
    for name, expected in EXPECTED_WET.items():
        p = Path(weights) / name / "wet_mask.npy"
        if not p.exists():
            pytest.skip(f"no wet mask at {p}")
        mask = np.load(p)
        cfg = load(name)
        assert mask.shape == (cfg.grid.cropped_h, cfg.grid.cropped_w)
        assert int(mask.astype(bool).sum()) == expected
