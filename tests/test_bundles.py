"""Bundle readers (experiments/truth.py) against the tiny committed fixtures.

The fixtures mirror the released bundles' key schema exactly: arange-pattern
fields with +1000 per variable (truth) and arange/7 fp16 latents, 2 ICs x 3
states on a 4 x 5 grid with two land cells.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from truth import LatentBundle, TruthBundle, find_bundle  # noqa: E402

FIX = REPO / "tests/fixtures"


def test_truth_bundle_reader():
    b = TruthBundle(FIX / "truth_bundle_tiny.npz")
    assert b.mask.shape == (4, 5) and b.mask.dtype == bool
    assert not b.mask[0, 0] and b.mask[3, 4]
    assert b.ic_datetimes == ["2022-01-01T12", "2022-02-02T12"]
    assert list(b.lead_hours) == [0, 6, 12]
    assert b.dx_col_m == 7004.124 and b.dy_row_m == 6984.775
    # second IC, sss (+1000 offset), state 2: exact arange values at wet cells
    f = b.chain_truth("2022-02-02T12:00", "salt")(2)
    assert f.dtype == np.float64
    i, s, fld = 1, 2, 1          # ic index, state, field index of sss
    expect = (np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)[i, s]
              + 1000 * fld)
    assert np.array_equal(f[b.mask], expect[b.mask].astype(float))
    assert np.isnan(f[~b.mask]).all()
    # target aliases resolve to the same block
    assert np.array_equal(b.chain_truth("2022-02-02T12", "sss")(2)[b.mask],
                          f[b.mask])


def test_truth_bundle_rejects_unknown_ic():
    b = TruthBundle(FIX / "truth_bundle_tiny.npz")
    with pytest.raises(ValueError, match="matched 0"):
        b.chain_truth("2021-12-31T12", "sst")


def test_latent_bundle_reader():
    b = LatentBundle(FIX / "latent_bundle_tiny.npz")
    assert b.ic_datetimes == ["2022-01-01T12", "2022-02-02T12"]
    lat = b.chain_latents("2022-01-01T12:00")
    x = lat(1)
    assert x.dtype == np.float32 and x.shape == (2, 3, 4)
    # exact fp16 round-trip of the stored pattern
    expect = (np.arange(2 * 3 * 2 * 3 * 4, dtype=np.float32)
              .reshape(2, 3, 2, 3, 4)[0, 1] / 7.0).astype(np.float16)
    assert np.array_equal(x, expect.astype(np.float32))
    with pytest.raises(ValueError, match="matched 0"):
        b.chain_latents("2019-01-01T12")


def test_find_bundle_resolution_order(tmp_path, monkeypatch):
    art = tmp_path / "artifacts/truth_bundles"
    art.mkdir(parents=True)
    data = tmp_path / "data/truth"
    data.mkdir(parents=True)
    (art / "mab.npz").write_bytes(b"a")
    (data / "mab_truth_bundle.npz").write_bytes(b"d")
    monkeypatch.setenv("CLOSEDLOOP_ARTIFACTS", str(tmp_path / "artifacts"))
    monkeypatch.setenv("CLOSEDLOOP_DATA_ROOT", str(tmp_path / "data"))
    assert find_bundle("truth", "mab") == art / "mab.npz"
    (art / "mab.npz").unlink()
    assert find_bundle("truth", "mab") == data / "mab_truth_bundle.npz"
    monkeypatch.delenv("CLOSEDLOOP_ARTIFACTS")
    monkeypatch.delenv("CLOSEDLOOP_DATA_ROOT")
    assert find_bundle("truth", "ccs") is None
