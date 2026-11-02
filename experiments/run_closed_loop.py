#!/usr/bin/env python
"""Run ONE closed-loop chain (Alg. 1): one (region, IC, target, fleet,
strategy, seed) for 40 cycles. GPU.

CLI mirrors the as-run production flags:
  python experiments/run_closed_loop.py --region mab \
      --ic_datetime 2022-08-02T12:00 --target salt --nfleet 50 \
      --strategy a_optimal --rseed 0 --ncyc 40 --start_lead 0 \
      [--scale 0.00022] [--members 25] [--dump-latents]

--scale defaults to the frozen per-(region, variable, fleet) guidance scale in
configs/experiments/guidance_scales.yaml. --start_lead exists for record
compatibility and must be 0 (the published convention; the chain starts at the
IC). Historical strategy aliases resolve (trace, dopt, lawn, nosample).

Output: <CLOSEDLOOP_RUNS>/chains/<region>_<ICTAG>_<target>_n<fleet>_<strategy>_s<rseed>.json
with the production record schema plus chosen-cell logging and per-cycle
wall-clock. With --dump-latents, the per-cycle analysis latents (fp16) go to a
sibling npz.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from truth import TruthBundle, StoreTruth, latent_source  # noqa: E402

from closedloop_sampling import paths  # noqa: E402
from closedloop_sampling.closed_loop import ChainInputs, LoopModels, run_chain  # noqa: E402
from closedloop_sampling.forecast.checkpoints import load_autoencoder, load_denoiser  # noqa: E402
from closedloop_sampling.forecast.normalization import (  # noqa: E402
    load_latent_normalizer,
    load_pixel_normalizer,
)
from closedloop_sampling.regions import load as load_region  # noqa: E402
from closedloop_sampling.sampling.reachability import cell_metres  # noqa: E402


def default_scale(region: str, target: str, nfleet: int) -> float:
    table = yaml.safe_load(open(REPO / "configs/experiments/guidance_scales.yaml"))
    var = {"temp": "sst", "sst": "sst", "salt": "sss", "sss": "sss"}[target.lower()]
    try:
        return float(table["scales"][region][var][f"n{nfleet}"])
    except KeyError:
        raise SystemExit(
            f"no frozen guidance scale for ({region}, {var}, n{nfleet}); "
            f"pass --scale explicitly") from None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["mab", "ccs", "doppio"])
    ap.add_argument("--ic_datetime", required=True, help="e.g. 2022-08-02T12:00")
    ap.add_argument("--target", default="temp", choices=["temp", "salt", "sst", "sss"])
    ap.add_argument("--nfleet", type=int, required=True)
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--rseed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=None,
                    help="guidance scale (default: guidance_scales.yaml lookup)")
    ap.add_argument("--ncyc", type=int, default=40)
    ap.add_argument("--start_lead", type=int, default=0,
                    help="must be 0 (the published convention)")
    ap.add_argument("--members", type=int, default=25)
    ap.add_argument("--dump-latents", action="store_true",
                    help="save per-cycle analysis latents (fp16 npz)")
    ap.add_argument("--truth", choices=["bundle", "store"], default="bundle")
    ap.add_argument("--denoiser-checkpoint", default=None,
                    help="explicit dir with ema.pt (default: release layout)")
    ap.add_argument("--dcae-checkpoint", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.start_lead != 0:
        raise SystemExit("start_lead must be 0: the chain starts at the IC "
                         "(the published convention).")

    region_name = "mab" if args.region == "doppio" else args.region
    cfg = load_region(region_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("the closed loop requires CUDA")
    free_running = args.strategy.lower() in ("free_running", "nosample")
    scale = 0.0 if free_running else (
        args.scale if args.scale is not None
        else default_scale(region_name, args.target, args.nfleet))

    # ---- models ----
    model = load_denoiser(args.denoiser_checkpoint, region=region_name,
                          seed=0, device=device, region_config=cfg)
    dcae = load_autoencoder(args.dcae_checkpoint, region=region_name,
                            device=device, region_config=cfg)
    lnorm = load_latent_normalizer(paths.resolve(cfg.paths.latent_norm_stats))
    pnorm = load_pixel_normalizer(paths.resolve(cfg.paths.pixel_norm_stats))
    models = LoopModels(model=model, dcae=dcae, lnorm=lnorm, pnorm=pnorm,
                        device=device)

    # ---- data ----
    lat_root = zarr.open(paths.resolve(cfg.paths.latents_zarr), mode="r")
    ic_idx, truth_latent = latent_source(
        lat_root["latents"], lat_root["ocean_time"], args.ic_datetime,
        cfg.timesplit.cadence_hours)
    bundle_path = paths.data_root() / "truth" / f"{region_name}_truth_bundle.npz"
    if args.truth == "bundle":
        if not bundle_path.exists():
            raise SystemExit(
                f"truth bundle not found at {bundle_path}; create it with "
                f"datasets/export_truth_bundle.py or pass --truth store")
        bundle = TruthBundle(bundle_path)
        mask = bundle.mask
        truth_field = bundle.chain_truth(args.ic_datetime, args.target)
        ds = xr.open_zarr(paths.resolve(cfg.paths.preprocessed_zarr),
                          consolidated=False)      # grid metrics only
    else:
        ds = xr.open_zarr(paths.resolve(cfg.paths.preprocessed_zarr),
                          consolidated=False)
        st = StoreTruth(cfg, ds)
        mask = st.mask
        truth_field = st.chain_truth(ic_idx, args.target)
    dx_col_m, dy_row_m = cell_metres(ds, mask, cfg.metric_crop)

    inputs = ChainInputs(
        region=cfg, ic_datetime=args.ic_datetime,
        ic_latent_raw=truth_latent(0), truth_latent=truth_latent,
        truth_field=truth_field, mask=mask,
        dx_col_m=dx_col_m, dy_row_m=dy_row_m)

    tag = args.ic_datetime.replace("-", "").replace(":", "")[:11]
    target = {"sst": "temp", "sss": "salt"}.get(args.target, args.target)
    print(f"=== chain {region_name} ic{tag} {target} n{args.nfleet} "
          f"{args.strategy} s{args.rseed} scale={scale} ncyc={args.ncyc} ===",
          flush=True)
    t0 = time.time()
    res = run_chain(models, inputs, target=target, strategy=args.strategy,
                    rseed=args.rseed, guidance_scale=scale,
                    n_fleet=args.nfleet, n_cycles=args.ncyc,
                    n_members=args.members, dump_latents=args.dump_latents)

    out_dir = Path(args.out_dir) if args.out_dir else paths.runs_root() / "chains"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{region_name}_ic{tag}_{target}_n{args.nfleet}_{args.strategy}_s{args.rseed}"
    out = dict(region=region_name, ic_datetime=args.ic_datetime, target=target,
               nfleet=args.nfleet, strategy=args.strategy, rseed=args.rseed,
               start_lead=0, ncyc=args.ncyc, n_members=args.members,
               guidance_scale=scale, records=res.records,
               chosen_rc=res.chosen_log, fleet_rc=res.fleet_log,
               stopped_early=res.stopped_early,
               wall_s_total=round(time.time() - t0, 1))
    fn = out_dir / f"{stem}.json"
    json.dump(out, open(fn, "w"), indent=2)
    print(f"wrote {fn} ({len(res.records)} cycles)")
    if args.dump_latents and res.prior_latents_fp16:
        lfn = out_dir / f"{stem}_latents.npz"
        np.savez_compressed(lfn, prior_latents=np.stack(res.prior_latents_fp16),
                            region=region_name, ic=args.ic_datetime)
        print(f"wrote {lfn}")
    print("CHAIN OK")


if __name__ == "__main__":
    main()
