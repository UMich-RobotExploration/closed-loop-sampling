#!/usr/bin/env python
"""Guidance-scale selection sweep (the frozen pre-registered protocol).

Selects lambda_g per (region, variable, fleet size) on the 15 tuning-year ICs
(configs/experiments/ics/validation_<region>.json) and freezes it before any
test-IC run. TUNING PROTOCOL NOTE: the sweep evaluates a SINGLE-SHOT 144 h
assimilation — placement and guided update on the stored +144 h prior
ensemble, obs frame 3 — not the 6 h closed loop; the selected scales are then
applied unchanged in the 6 h-cadence loop. This is the as-run protocol.

Stage 1: the 7-point log grid {1e-4, 2.2e-4, 4.6e-4, 1e-3, 2.2e-3, 4.6e-3,
1e-2}, one guided ensemble per (IC, scale) with common random numbers (one
x_start per (IC, member), reused across ALL scales; y fixed across scales).
Stability screens:
  S1 member-finite fallback-to-prior count (must be 0 at the selected scale)
  S2 posterior stack absmax < 60 on wet cells
  S3 spread-vs-prior report (collapse or blow-up visible in the landscape)
  S4 noise-floor repeat cell: first roster IC re-run at the stage-1 winner
     with an offset start seed (+999); |repeat - base| is the tie tolerance.
Selection: among scales with S1 = 0 and S2 true at every IC, argmin of the
mean posterior RMSE over the roster; stage 2 probes the winner's log-grid
midpoints (x0.68, x1.47), or extends one grid step when the winner sits at an
endpoint (--extend-edge iterates the extension, at most 2 further steps,
re-selecting each time); ties within the repeat-cell tolerance resolve to the
SMALLER scale.

Placement at tuning time is the fast variance-greedy below (trace_greedy_K,
the calibration lineage's placement); the loop's own selection lives in
closedloop_sampling.sampling.selection.

Prior pools: per roster IC, a stored block of 25 member rollouts to +144 h
(retain-all chaining, one generator per member seeded member_seed(0,
ic_draw_order, m)). --make-pool generates them; --pool-dir points at them.

Usage:
  python experiments/select_guidance_scale.py --region mab --target sst --nfleet 50 \
      --pool-dir <dir> [--make-pool] [--extend-edge]
Writes <CLOSEDLOOP_RUNS>/guidance_sweep/s_star_<region>_<var>_n<K>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import zarr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from truth import latent_source  # noqa: E402

from closedloop_sampling import paths  # noqa: E402
from closedloop_sampling.assimilation.guidance import guided_update  # noqa: E402
from closedloop_sampling.forecast.checkpoints import load_autoencoder, load_denoiser  # noqa: E402
from closedloop_sampling.forecast.edm import edm_sample  # noqa: E402
from closedloop_sampling.forecast.ensemble import (  # noqa: E402
    N_SAMPLING_STEPS,
    SIGMA_DATA,
    SIGMA_MAX,
    SIGMA_MIN,
    member_seed,
)
from closedloop_sampling.forecast.normalization import (  # noqa: E402
    load_latent_normalizer,
    load_pixel_normalizer,
)
from closedloop_sampling.observations import sigma_for  # noqa: E402
from closedloop_sampling.regions import load as load_region  # noqa: E402

SCALES = [1e-4, 2.2e-4, 4.6e-4, 1e-3, 2.2e-3, 4.6e-3, 1e-2]
LEAD_H, N_MEMBERS, POOL_STEPS, COND_STEP, OBS_FRAME = 144, 25, 24, 19, 3
ABSMAX_BOUND = 60.0
REPEAT_SEED_OFFSET = 999


def trace_greedy_K(Xp, R, K):
    """Tuning-time variance-greedy placement (calibration lineage, verbatim)."""
    P = (Xp * Xp).sum(1) / (Xp.shape[1] - 1)
    chosen = []
    for _ in range(K):
        j = int(np.argmax(P))
        chosen.append(j)
        xj = Xp[j]
        gain = (Xp @ xj) ** 2 / ((Xp.shape[1] - 1) ** 2 * (P[j] + R))
        P = P - gain / max(P[j], 1e-12) * P[j]
        P[j] = -np.inf
    return chosen


def make_pool(cfg, model, lnorm, device, ic_dt, k_ic, out_dir):
    """25-member retain-all rollout to +144 h from one tuning IC (one
    generator per member, chained on the last predicted frame)."""
    lat_root = zarr.open(paths.resolve(cfg.paths.latents_zarr), mode="r")
    ic_idx, latent = latent_source(lat_root["latents"], lat_root["ocean_time"],
                                   ic_dt, cfg.timesplit.cadence_hours)
    ic_norm = lnorm.normalize_dynamic_tensor(
        torch.from_numpy(latent(0)[None].astype(np.float32)).to(device))
    C, h, w = latent(0).shape
    members = []
    ac = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"))
    for m in range(N_MEMBERS):
        gen = torch.Generator(device=device)
        gen.manual_seed(member_seed(0, k_ic, m))
        cond = ic_norm.unsqueeze(2)
        steps = []
        with torch.no_grad(), ac:
            for _ in range(POOL_STEPS // 4):
                x = edm_sample(model, conditioning=cond,
                               pred_shape=(1, C, 4, h, w),
                               sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX,
                               sigma_data=SIGMA_DATA, n_steps=N_SAMPLING_STEPS,
                               generator=gen, device=device)
                steps.append(x)
                cond = x[:, :, -1:]
        members.append(torch.cat(steps, 2)[0].float().cpu().numpy())
    tag = ic_dt.replace("-", "")[:8]
    pdir = out_dir / f"ic_{tag}" / "seed_000"
    pdir.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(pdir / "forecasts.zarr"), mode="w")
    root.create_array("forecasts", shape=(1, N_MEMBERS, C, POOL_STEPS, h, w),
                      chunks=(1, N_MEMBERS, C, POOL_STEPS, h, w),
                      dtype=np.float32)[:] = np.stack(members)[None]
    root.create_array("ic_time_indices", shape=(1,), dtype=np.int64)[:] = [ic_idx]
    print(f"  pool written: {pdir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--target", required=True, choices=["sst", "sss"])
    ap.add_argument("--nfleet", type=int, required=True)
    ap.add_argument("--pool-dir", default=None,
                    help="dir of per-IC prior pools (ic_<tag>/seed_000/forecasts.zarr)")
    ap.add_argument("--make-pool", action="store_true",
                    help="generate the prior pools for the roster, then exit")
    ap.add_argument("--extend-edge", action="store_true",
                    help="iterate the endpoint extension (max 2 further steps)")
    args = ap.parse_args()

    cfg = load_region(args.region)
    var = "temp" if args.target == "sst" else "salt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("the sweep requires CUDA")
    roster = json.load(open(REPO / "configs/experiments/ics" /
                            f"validation_{args.region}.json"))["roster_draw_order"]
    pool_dir = Path(args.pool_dir) if args.pool_dir else (
        paths.runs_root() / "guidance_pools" / args.region)

    model = load_denoiser(region=args.region, seed=0, device=device,
                          region_config=cfg)
    dcae = load_autoencoder(region=args.region, device=device, region_config=cfg)
    lnorm = load_latent_normalizer(paths.resolve(cfg.paths.latent_norm_stats))
    pnorm = load_pixel_normalizer(paths.resolve(cfg.paths.pixel_norm_stats))

    if args.make_pool:
        for k_ic, dt in enumerate(roster):
            make_pool(cfg, model, lnorm, device, dt, k_ic, pool_dir)
        print("POOLS OK")
        return

    ds = xr.open_zarr(paths.resolve(cfg.paths.preprocessed_zarr), consolidated=False)
    crop = cfg.metric_crop
    mask = ds["mask_rho"].values[crop] > 0.5
    wet = np.stack(np.where(mask), 1)
    idx = cfg.channel_indices()
    CH = idx["sst"] if var == "temp" else idx["sss"]
    SIG = sigma_for(var)
    R = SIG ** 2
    K = args.nfleet
    C, h, w = cfg.latent.latent_channels, cfg.latent.latent_h, cfg.latent.latent_w

    def dec(lat_np):
        with torch.no_grad():
            t = torch.from_numpy(lat_np.astype(np.float32)).to(device)
            pix = pnorm.denormalize_dynamic_tensor(
                dcae.decode(lnorm.denormalize_dynamic_tensor(t)))
        return pix[:, CH].float().cpu().numpy()

    def crn(seed):
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return torch.randn((1, C, 4, h, w), generator=g, device=device) * SIGMA_MAX

    def dps(cond_np, xs, rows, cols, y, sc):
        cond = torch.from_numpy(cond_np[None, :, None].astype(np.float32)).to(device)
        x0 = guided_update(model, dcae, lnorm, pnorm, cond, xs, rows, cols, y,
                           SIG, CH, guidance_scale=sc, obs_frame=OBS_FRAME,
                           device=device)
        return x0[0, :, OBS_FRAME].float().cpu().numpy()

    def evaluate_scale(ctx, sc):
        rows, cols, y, truth, ens, xs_all = ctx
        post, fb = [], 0
        for m in range(N_MEMBERS):
            fr = dps(ens[m, :, COND_STEP], xs_all[m], rows, cols, y, sc)
            if np.isfinite(fr).all():
                post.append(fr)
            else:
                post.append(ens[m, :, POOL_STEPS - 1])
                fb += 1                                           # S1 fallback
        ps = dec(np.stack(post))
        ps[:, ~mask] = np.nan
        phys = bool(np.isfinite(ps[:, mask]).all()
                    and np.nanmax(np.abs(ps[:, mask])) < ABSMAX_BOUND)  # S2
        rmse = float(np.sqrt(np.nanmean((ps.mean(0)[mask] - truth[mask]) ** 2)))
        spread = float(np.nanmean(np.nanstd(ps, 0)[mask]))              # S3
        return dict(rmse=round(rmse, 4), spread=round(spread, 4),
                    fallbacks=fb, phys=phys,
                    absmax=round(float(np.nanmax(np.abs(ps[:, mask]))), 2))

    out = dict(config=f"{args.region}_{args.target}_n{K}", scales=SCALES,
               K=K, sig=SIG, roster=roster, per_ic={}, screens={})
    contexts = {}
    for k_ic, dt in enumerate(roster):
        tag = dt.replace("-", "")[:8]
        zb = zarr.open(str(pool_dir / f"ic_{tag}/seed_000/forecasts.zarr"), mode="r")
        ens = np.asarray(zb["forecasts"][0], np.float32)
        ic_idx = int(zb["ic_time_indices"][0])
        A = dec(ens[:, :, POOL_STEPS - 1])
        A[:, ~mask] = np.nan
        Xp = np.nan_to_num((A - A.mean(0))[:, mask].T)
        ch = trace_greedy_K(Xp, R, K)
        rows, cols = wet[ch, 0], wet[ch, 1]
        truth = ds[var].isel(
            ocean_time=ic_idx + LEAD_H // cfg.timesplit.cadence_hours,
            s_rho=idx["surface_s_rho_index"]).values[crop].astype(float)
        doy = int((np.datetime64(dt[:10]) - np.datetime64(dt[:4] + "-01-01"))
                  .astype(int)) + 1
        y = truth[rows, cols] + np.random.default_rng(70000 + doy + LEAD_H)\
            .normal(0, SIG, len(rows))
        xs_all = [crn(member_seed(0, k_ic, m)) for m in range(N_MEMBERS)]
        contexts[dt] = (rows, cols, y, truth, ens, xs_all)
        row = dict(scales={})
        for sc in SCALES:
            cell = evaluate_scale(contexts[dt], sc)
            row["scales"][f"{sc:.1e}"] = cell
            print(f"  [{out['config']}] {dt} sc={sc:.1e}: rmse {cell['rmse']} "
                  f"fb={cell['fallbacks']} phys={cell['phys']}", flush=True)
        out["per_ic"][dt] = row

    def eligible_from(keys):
        el = []
        for keyname, scv in keys:
            cells = [out["per_ic"][dt]["scales"].get(keyname) for dt in roster]
            if all(c and c["phys"] and c["fallbacks"] == 0 for c in cells):
                el.append((float(np.mean([c["rmse"] for c in cells])), scv))
        return sorted(el)

    stage1 = eligible_from([(f"{sc:.1e}", sc) for sc in SCALES])
    if not stage1:
        raise SystemExit("no eligible scale on the stage-1 grid (pathology hold)")
    w_mu, w_sc = stage1[0]

    # S4: repeat the first roster IC at the stage-1 winner with offset seeds.
    dt0 = roster[0]
    rows0, cols0, y0, truth0, ens0, _ = contexts[dt0]
    xs_rep = [crn(member_seed(0, 0, m) + REPEAT_SEED_OFFSET) for m in range(N_MEMBERS)]
    rep = evaluate_scale((rows0, cols0, y0, truth0, ens0, xs_rep), w_sc)
    base = out["per_ic"][dt0]["scales"][f"{w_sc:.1e}"]["rmse"]
    nf = round(abs(rep["rmse"] - base), 4)
    out["screens"]["noise_floor"] = dict(cell=f"{dt0}@{w_sc:.1e} (stage-1 winner)",
                                         repeat=rep["rmse"], base=base, delta_abs=nf)

    # Stage 2: midpoints, or endpoint extension (iterated with --extend-edge).
    grid_step = SCALES[-1] / SCALES[-2]
    if w_sc == min(SCALES):
        probes, flag = [w_sc / (SCALES[1] / SCALES[0])], "endpoint-low: extended one grid step"
    elif w_sc == max(SCALES):
        probes, flag = [w_sc * grid_step], "endpoint-high: extended one grid step"
    else:
        probes, flag = [w_sc * 0.68, w_sc * 1.47], None
    probed = {}
    extra_keys = []
    iters = 0
    while True:
        for sc in probes:
            keyname = f"{sc:.3g}"
            for dt in roster:
                out["per_ic"][dt]["scales"][keyname] = evaluate_scale(contexts[dt], sc)
            probed[keyname] = float(np.mean(
                [out["per_ic"][dt]["scales"][keyname]["rmse"] for dt in roster]))
            extra_keys.append((keyname, sc))
            print(f"  [stage2] probed {keyname}: mean rmse {probed[keyname]:.4f}",
                  flush=True)
        elig = eligible_from([(f"{sc:.1e}", sc) for sc in SCALES] + extra_keys)
        best_mu = elig[0][0]
        tie = [sc for mu, sc in elig if abs(mu - best_mu) <= nf]
        s_star = min(tie)
        outer = max(sc for _, sc in elig)
        at_edge = abs(s_star - outer) / outer < 1e-2
        if not (args.extend_edge and at_edge and iters < 2):
            break
        iters += 1
        probes = [outer * grid_step]
        flag = f"endpoint-high: extension iteration {iters}"
    out["stage2"] = dict(probed={k: round(v, 4) for k, v in probed.items()}, flag=flag)
    out["selection"] = dict(
        s_star=s_star, stage1_winner=w_sc, tie_set=tie, noise_floor=nf,
        edge_limited=bool(at_edge), extension_iterations=iters,
        rule="argmin mean posterior RMSE over screened {stage-1 union stage-2}; "
             "ties within the repeat-cell tolerance resolve to the smaller scale")
    out_dir = paths.runs_root() / "guidance_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = out_dir / f"s_star_{out['config']}.json"
    json.dump(out, open(fn, "w"), indent=1)
    print(f"S_STAR[{out['config']}] = {s_star:.3g}  (edge_limited={at_edge})")
    print(f"wrote {fn}")
    print("SWEEP OK")


if __name__ == "__main__":
    main()
