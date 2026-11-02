#!/usr/bin/env python
"""Export the test-window truth bundle for one region.

Dense surface-only re-export from the preprocessed store: for each sealed test
IC, the 41 six-hour states (leads 0, 6, ..., 240 h) of the four surface fields
the closed loop needs — SST, SSS, surface u, surface v — plus the wet mask,
all on the model crop, written as one compressed npz of roughly 0.1 GB per
region. The closed loop and the scoring read this bundle instead of the
multi-terabyte store.

Chunk note: the store's chunks span all 20 sigma-levels and 72 time steps, so
chunk-aligned extraction of the same windows touches orders of magnitude more
bytes than this dense surface re-export; the bundle is the release form.

Usage:
  python datasets/export_truth_bundle.py --region mab \
      [--ics configs/experiments/ics/test_mab.json] [--out <path.npz>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from closedloop_sampling import paths
from closedloop_sampling.regions import load as load_region

N_CYCLES = 40
STEP_HOURS = 6
N_STATES = N_CYCLES + 1          # leads 0..240 h inclusive


def export_truth_bundle(region_name: str, ic_datetimes, out_path: Path) -> Path:
    cfg = load_region(region_name)
    crop = cfg.metric_crop
    idx = cfg.channel_indices()
    surface = idx["surface_s_rho_index"]
    stride = cfg.timesplit.internal_stride          # index steps per 6 h

    ds = xr.open_zarr(paths.resolve(cfg.paths.preprocessed_zarr), consolidated=False)
    mask = (ds["mask_rho"].values[crop] > 0.5)
    times = np.asarray(ds["ocean_time"].values).astype("datetime64[h]")

    fields = {k: np.empty((len(ic_datetimes), N_STATES, mask.shape[0], mask.shape[1]),
                          dtype=np.float32)
              for k in ("sst", "sss", "u", "v")}
    var_of = {"sst": "temp", "sss": "salt", "u": "u", "v": "v"}

    for i, ic in enumerate(ic_datetimes):
        hits = np.where(times == np.datetime64(ic[:13], "h"))[0]
        if len(hits) != 1:
            raise ValueError(f"IC {ic} matched {len(hits)} store steps")
        ic_idx = int(hits[0])
        t_indices = [ic_idx + t * stride for t in range(N_STATES)]
        if t_indices[-1] >= times.shape[0]:
            raise ValueError(f"IC {ic}: 240 h window runs past the store end")
        for key, var in var_of.items():
            block = ds[var].isel(
                ocean_time=t_indices, s_rho=surface,
            ).values[:, crop[0], crop[1]].astype(np.float32)
            block[:, ~mask] = np.nan
            fields[key][i] = block
        print(f"  [{region_name}] {ic}: store index {ic_idx}, "
              f"{N_STATES} states extracted", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        mask=mask,
        ic_datetimes=np.array(ic_datetimes),
        lead_hours=np.arange(N_STATES, dtype=np.int64) * STEP_HOURS,
        region=region_name,
        **fields,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path} ({size_mb:.0f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--ics", default=None,
                    help="sealed-IC JSON (default: configs/experiments/ics/test_<region>.json)")
    ap.add_argument("--out", default=None,
                    help="output npz (default: <data_root>/truth/<region>_truth_bundle.npz)")
    args = ap.parse_args()

    ics_path = Path(args.ics) if args.ics else (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/ics" / f"test_{args.region}.json")
    sealed = json.load(open(ics_path))
    ic_datetimes = sealed["canonical_order"]
    out = Path(args.out) if args.out else (
        paths.data_root() / "truth" / f"{args.region}_truth_bundle.npz")
    export_truth_bundle(args.region, ic_datetimes, out)


if __name__ == "__main__":
    main()
