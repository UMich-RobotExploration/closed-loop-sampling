#!/usr/bin/env python
"""Export a region's wet mask as a standalone .npy (e.g. for a weights release).

Reads `mask_rho` from the region's preprocessed zarr (read-only), binarizes
(> 0.5), applies the RegionConfig crop so the mask matches model I/O
(MAB 104x240, CCS 176x184), and writes `wet_mask.npy` (float32 0/1) plus a
`wet_mask_meta.json` with the crop, shapes, and wet-cell count.

Usage:
    python scripts/export_wet_mask.py --region mab --out <dir>
    python scripts/export_wet_mask.py --region ccs --out <dir> \
        [--preprocessed-zarr <path>]      # override the config-resolved store
"""
import argparse
import json
from pathlib import Path

import numpy as np
import zarr

from closedloop_sampling import paths
from closedloop_sampling.regions import get_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--preprocessed-zarr", default=None,
                    help="override: path to the preprocessed store holding mask_rho")
    args = ap.parse_args()

    cfg = get_config(args.region)
    store = args.preprocessed_zarr or paths.resolve(cfg.paths.preprocessed_zarr)
    root = zarr.open(store, mode="r")
    mask_native = np.asarray(root["mask_rho"])
    crop = cfg.metric_crop
    if mask_native.shape == (cfg.grid.cropped_h, cfg.grid.cropped_w):
        mask_cropped = mask_native                       # store already cropped
        cropped_in_store = True
    else:
        mask_cropped = mask_native[crop]
        cropped_in_store = False
    wet = (mask_cropped > 0.5).astype(np.float32)
    expected = (cfg.grid.cropped_h, cfg.grid.cropped_w)
    assert wet.shape == expected, f"mask shape {wet.shape} != expected {expected}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "wet_mask.npy", wet)
    meta = {
        "region": args.region,
        "source_variable": "mask_rho (preprocessed store), binarized > 0.5",
        "native_grid": [cfg.grid.native_eta_rho, cfg.grid.native_xi_rho],
        "crop_eta": list(cfg.grid.crop_eta_slice),
        "crop_xi": list(cfg.grid.crop_xi_slice),
        "crop_applied_at_export": not cropped_in_store,
        "shape": list(wet.shape),
        "wet_cells": int(wet.sum()),
        "dtype": "float32 (0/1)",
    }
    with open(out / "wet_mask_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[export_wet_mask] {args.region}: shape {wet.shape}, "
          f"wet cells {int(wet.sum())} -> {out/'wet_mask.npy'}")


if __name__ == "__main__":
    main()
