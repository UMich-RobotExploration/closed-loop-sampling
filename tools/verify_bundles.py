#!/usr/bin/env python
"""Spot-check the exported truth and latent bundles against the stores.

For each region: the wet mask must equal the store's cropped mask exactly; a
deterministic sample of (IC, state, field) truth slices must equal the store
values bitwise (both are float32 re-reads of the same array); the metric cell
sizes must match cell_metres; and sampled latent states must equal the store's
float32 values after the bundle's documented fp16 quantization.

Usage: python tools/verify_bundles.py --region mab --bundles <dir> [--n 3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from truth import LatentBundle, TruthBundle  # noqa: E402

from closedloop_sampling import paths  # noqa: E402
from closedloop_sampling.regions import load as load_region  # noqa: E402
from closedloop_sampling.sampling.reachability import cell_metres  # noqa: E402

VAR_OF = {"sst": "temp", "sss": "salt", "u": "u", "v": "v"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--bundles", required=True,
                    help="directory holding truth_bundles/ and latent_bundles/")
    ap.add_argument("--n", type=int, default=3, help="slices to sample per bundle")
    args = ap.parse_args()
    cfg = load_region(args.region)
    crop = cfg.metric_crop
    surface = cfg.channel_indices()["surface_s_rho_index"]
    stride = cfg.timesplit.internal_stride
    root = Path(args.bundles)

    tb = TruthBundle(root / "truth_bundles" / f"{args.region}.npz")
    lb = LatentBundle(root / "latent_bundles" / f"{args.region}.npz")
    sealed = json.load(open(REPO / "configs/experiments/ics"
                            / f"test_{args.region}.json"))["canonical_order"]
    assert tb.ic_datetimes == sealed and lb.ic_datetimes == sealed

    ds = xr.open_zarr(paths.resolve(cfg.paths.preprocessed_zarr), consolidated=False)
    mask = ds["mask_rho"].values[crop] > 0.5
    assert np.array_equal(tb.mask, mask), "wet mask differs from the store"
    dx, dy = cell_metres(ds, mask, crop)
    assert tb.dx_col_m == float(dx) and tb.dy_row_m == float(dy)
    times = np.asarray(ds["ocean_time"].values).astype("datetime64[h]")

    rng = np.random.default_rng(20261103)
    for k in range(args.n):
        ic = sealed[int(rng.integers(len(sealed)))]
        state = int(rng.integers(41))
        field = ("sst", "sss", "u", "v")[int(rng.integers(4))]
        got = tb.chain_truth(ic, {"sst": "sst", "sss": "sss"}.get(field, field))(
            state) if field in ("sst", "sss") else None
        ic_idx = int(np.where(times == np.datetime64(ic[:13], "h"))[0][0])
        want = ds[VAR_OF[field]].isel(
            ocean_time=ic_idx + state * stride, s_rho=surface,
        ).values[crop].astype(np.float32)
        want[~mask] = np.nan
        if got is None:                     # u/v: read the raw block directly
            z = np.load(root / "truth_bundles" / f"{args.region}.npz")
            got = z[field][sealed.index(ic), state].astype(float)
        both_nan = np.isnan(got) & np.isnan(want)
        assert np.array_equal(np.nan_to_num(got.astype(np.float32)),
                              np.nan_to_num(want)) and bool(
            (np.isnan(got) == np.isnan(want)).all()), (ic, state, field)
        print(f"  truth slice {k + 1}: {ic} state {state} {field} -- bitwise "
              f"equal on {int(both_nan.size - both_nan.sum())} cells")

    lat_root = zarr.open(paths.resolve(cfg.paths.latents_zarr), mode="r")
    latz, ltimes = lat_root["latents"], np.asarray(
        lat_root["ocean_time"]).astype("datetime64[h]")
    for k in range(args.n):
        ic = sealed[int(rng.integers(len(sealed)))]
        state = int(rng.integers(41))
        got = lb.chain_latents(ic)(state)
        ic_idx = int(np.where(ltimes == np.datetime64(ic[:13], "h"))[0][0])
        want = np.asarray(latz[ic_idx + state * stride], np.float32)
        assert np.array_equal(got, want.astype(np.float16).astype(np.float32)), (
            ic, state)
        q = float(np.max(np.abs(got - want)))
        print(f"  latent slice {k + 1}: {ic} state {state} -- equals the store "
              f"after fp16 quantization (max |q-err| {q:.2e})")

    print(f"VERIFY_BUNDLES {args.region} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
