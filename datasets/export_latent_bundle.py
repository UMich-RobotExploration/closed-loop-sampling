#!/usr/bin/env python
"""Export the sealed-IC latent bundle for one region.

For each sealed test IC, the 41 six-hour states (leads 0, 6, ..., 240 h) of the
RAW truth latents, stored fp16, as one compressed npz. The closed loop reads
this bundle for the IC latent (state 0, normalized inside the loop) and the
per-cycle autoencoder floor (raw latent decode), so it can run without the full
latents store (MAB 23 GB, CCS 11 GB -> roughly 30 MB per region).

Precision note: fp16 quantization perturbs the stored latents by about 5e-4
relative. The autoencoder floor recomputed from the bundle therefore differs
from the production records in roughly the fourth decimal, and the tiled-IC
initial ensemble starts from the quantized latent; bf16 generation is in any
case not bit-stable across GPU models, so bundle-backed chains reproduce the
production records statistically, not bitwise.

Usage:
  python datasets/export_latent_bundle.py --region mab \
      [--ics configs/experiments/ics/test_mab.json] [--out <path.npz>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from closedloop_sampling import paths
from closedloop_sampling.regions import load as load_region

N_CYCLES = 40
STEP_HOURS = 6
N_STATES = N_CYCLES + 1


def export_latent_bundle(region_name: str, ic_datetimes, out_path: Path) -> Path:
    cfg = load_region(region_name)
    stride = cfg.timesplit.internal_stride          # index steps per 6 h

    root = zarr.open(paths.resolve(cfg.paths.latents_zarr), mode="r")
    latents = root["latents"]
    times = np.asarray(root["ocean_time"]).astype("datetime64[h]")

    out = None
    for i, ic in enumerate(ic_datetimes):
        hits = np.where(times == np.datetime64(ic[:13], "h"))[0]
        if len(hits) != 1:
            raise ValueError(f"IC {ic} matched {len(hits)} latent steps")
        ic_idx = int(hits[0])
        t_indices = [ic_idx + t * stride for t in range(N_STATES)]
        if t_indices[-1] >= times.shape[0]:
            raise ValueError(f"IC {ic}: 240 h window runs past the store end")
        block = np.asarray(latents.oindex[t_indices], dtype=np.float32)
        if out is None:
            out = np.empty((len(ic_datetimes), N_STATES) + block.shape[1:],
                           dtype=np.float16)
        out[i] = block.astype(np.float16)
        print(f"  [{region_name}] {ic}: store index {ic_idx}, "
              f"{N_STATES} latent states ({block.shape[1:]})", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        latents=out,
        ic_datetimes=np.array(ic_datetimes),
        lead_hours=np.arange(N_STATES, dtype=np.int64) * STEP_HOURS,
        region=region_name,
        dtype_note="RAW truth latents, fp16; see the module docstring",
    )
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--ics", default=None,
                    help="sealed-IC JSON (default: configs/experiments/ics/test_<region>.json)")
    ap.add_argument("--out", default=None,
                    help="output npz (default: <data_root>/latents/<region>_latent_bundle.npz)")
    args = ap.parse_args()

    ics_path = Path(args.ics) if args.ics else (
        Path(__file__).resolve().parents[1]
        / "configs/experiments/ics" / f"test_{args.region}.json")
    ic_datetimes = json.load(open(ics_path))["canonical_order"]
    out = Path(args.out) if args.out else (
        paths.data_root() / "latents" / f"{args.region}_latent_bundle.npz")
    export_latent_bundle(args.region, ic_datetimes, out)


if __name__ == "__main__":
    main()
