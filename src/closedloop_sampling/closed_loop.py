"""The closed loop — Algorithm 1: forecast, reachability, selection,
observation, guided analysis update; 40 six-hour cycles per chain.

Implements Sec. III and Alg. 1 end to end, exactly as the production campaigns
ran (start_lead = 0 form):

  - The chain starts at the IC itself: the initial analysis ensemble is N
    identical copies of the normalized IC latent — member spread is created by
    the first generative roll, not by perturbing the IC.
  - Cycle c (0-based): roll every member one generative step (4 predicted
    states, retain the FIRST, +6 h; bf16; CRN seeds, see forecast.ensemble);
    compute current-advected reachable sets from the ensemble-mean surface
    current (mean of the prior and forecast decodes); the policy picks each
    vehicle's cell; observe the target variable with noise (one draw, shared
    by all members); regenerate every member by observation-guided diffusion
    at the fixed guidance scale. Cycle c scores lead (c+1)*6 h.
  - Per-cycle AE floor: the RMSE of the decoded RAW truth latent against
    truth on the target channel — the autoencoder's reconstruction limit,
    recomputed every cycle (the skill-fraction denominator).
  - Scoring: prior_rmse / post_rmse are ensemble-mean RMSE of the TARGET
    variable over wet cells.

Per-cycle record schema: the production keys (cycle, prior_rmse, post_rmse,
floor, n_obs) PLUS chosen-cell logging (chosen_rc, fleet_rc) and wall-clock
(wall_s), both new in this release; lawnmower chains also carry
stations_completed / n_skipped as in production.

Seed constants are recorded in configs/experiments/closed_loop.yaml; the code
constants here and in forecast.ensemble / assimilation.guidance /
observations are the same values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from closedloop_sampling.assimilation.guidance import (
    GUIDED_RSEED_STRIDE,
    GUIDED_SEED_OFFSET,
    guided_start_noise,
    guided_update,
)
from closedloop_sampling.forecast.ensemble import (
    CYCLE_SEED_STRIDE,
    HOURS_PER_STEP,
    PRED_STEPS_PER_CALL,
    initial_ensemble,
    member_seed,
    roll_ensemble,
)
from closedloop_sampling.observations import ic_key, obs_rng, observe, sigma_for
from closedloop_sampling.sampling.reachability import advected_reachable
from closedloop_sampling.sampling.policies import ChainContext, CycleContext, make_policy

N_MEMBERS = 25
N_CYCLES = 40
V_AUV_MS = 1.5
REACH_WINDOW_S = 6 * 3600
DENOISER_SEED = 0            # the production loop ran the seed-0 denoiser
FREE_RUNNING = ("free_running", "nosample")


@dataclass
class LoopModels:
    """Trained components (see forecast.checkpoints for the loaders)."""
    model: torch.nn.Module       # denoiser (EMA weights)
    dcae: torch.nn.Module        # autoencoder (EMA weights)
    lnorm: object                # latent normalizer
    pnorm: object                # pixel normalizer
    device: torch.device


@dataclass
class ChainInputs:
    """Per-chain data handles (built by experiments/run_closed_loop.py)."""
    region: object               # RegionConfig
    ic_datetime: str             # ISO8601
    ic_latent_raw: np.ndarray    # (C, h, w) RAW latent of the IC state
    truth_latent: object         # state_index -> (C, h, w) RAW truth latent
    truth_field: object          # state_index -> (H, W) physical target field (NaN land)
    mask: np.ndarray             # (H, W) bool wet mask (model crop)
    dx_col_m: float
    dy_row_m: float


@dataclass
class ChainResult:
    records: list
    chosen_log: list             # per cycle: chosen cells [[r, c], ...]
    fleet_log: list              # per cycle: vehicle start positions
    prior_latents_fp16: list = field(default_factory=list)   # if dump_latents
    stopped_early: bool = False


def coverage_fleet(mask: np.ndarray, wet: np.ndarray, n: int) -> np.ndarray:
    """The shared even coverage grid: lattice points snapped to the nearest
    wet cell, deduplicated; on narrow masks that dedupe below n, top up by
    farthest-point sampling over wet cells (fires only on shortfall)."""
    H, W = mask.shape
    nr = max(1, int(round(np.sqrt(n * H / W))))
    nc = int(np.ceil(n / nr))
    rows = np.linspace(0.1 * H, 0.9 * H, nr)
    cols = np.linspace(0.08 * W, 0.92 * W, nc)
    seen = set()
    pts = []
    for r in rows:
        for c in cols:
            d = (wet[:, 0] - r) ** 2 + (wet[:, 1] - c) ** 2
            p = wet[d.argmin()]
            k = (int(p[0]), int(p[1]))
            if k not in seen:
                seen.add(k)
                pts.append(p)
    pts = np.array(pts[:n], float)
    while len(pts) < n:
        d2 = ((wet[None, :, :] - pts[:, None, :]) ** 2).sum(-1).min(0)
        pts = np.vstack([pts, wet[int(d2.argmax())].astype(float)])
    return pts


def run_chain(
    models: LoopModels,
    inputs: ChainInputs,
    target: str,
    strategy: str,
    rseed: int,
    guidance_scale: float,
    n_fleet: int,
    n_cycles: int = N_CYCLES,
    n_members: int = N_MEMBERS,
    dump_latents: bool = False,
    verbose: bool = True,
) -> ChainResult:
    """One chain: `n_cycles` cycles of Alg. 1 at one (IC, target, fleet,
    policy, seed)."""
    device = models.device
    cfg = inputs.region
    idx = cfg.channel_indices()
    tgt_ch = idx["sst"] if target in ("temp", "sst") else idx["sss"]
    u_ch, v_ch = idx["u_surface"], idx["v_surface"]
    sigma = sigma_for(target)
    R = sigma ** 2
    mask = inputs.mask
    wet = np.stack(np.where(mask), 1)
    wet_lookup = {(int(r), int(c)): i for i, (r, c) in enumerate(wet)}
    key = ic_key(inputs.ic_datetime)
    C, h_lat, w_lat = inputs.ic_latent_raw.shape
    pred_shape = (1, C, PRED_STEPS_PER_CALL, h_lat, w_lat)
    free_running = strategy.lower() in FREE_RUNNING

    def dec(frames, chans, normalized=True):
        """Batched decode of latent frames to physical channels (production
        batching of 25)."""
        out = []
        for b in range(0, len(frames), 25):
            x = torch.from_numpy(frames[b:b + 25].astype(np.float32)).to(device)
            z = models.lnorm.denormalize_dynamic_tensor(x) if normalized else x
            with torch.no_grad():
                pix = models.pnorm.denormalize_dynamic_tensor(models.dcae.decode(z))
            out.append(pix[:, chans].float().cpu().numpy())
        return np.concatenate(out, 0)

    def ens_rmse(fields, truth):
        mm = np.nanmean(np.where(np.isfinite(fields), fields, np.nan), 0)
        return float(np.sqrt(np.nanmean((mm[mask] - truth[mask]) ** 2)))

    # Initial analysis ensemble: N copies of the normalized IC latent.
    ic_norm = models.lnorm.normalize_dynamic_tensor(
        torch.from_numpy(inputs.ic_latent_raw[None].astype(np.float32)).to(device)
    )[0].cpu().numpy()
    prior_lat = initial_ensemble(ic_norm, n_members)

    policy = None
    fleet = coverage_fleet(mask, wet, n_fleet)
    if not free_running:
        chain_ctx = ChainContext(
            wet=wet, wet_lookup=wet_lookup, mask=mask, n_fleet=n_fleet,
            dx_col_m=inputs.dx_col_m, dy_row_m=inputs.dy_row_m,
            v_auv_ms=V_AUV_MS, reach_window_s=REACH_WINDOW_S,
            n_cycles=n_cycles, rseed=rseed, initial_fleet=fleet,
        )
        policy = make_policy(strategy, chain_ctx)
        fleet = policy.start_fleet()

    result = ChainResult(records=[], chosen_log=[], fleet_log=[])
    empty_rc = np.empty((0, 2), np.int16)

    for c in range(n_cycles):
        t0 = time.time()
        state = c + 1                                      # lead (c+1)*6 h
        truth = inputs.truth_field(state)
        floor_f = dec(inputs.truth_latent(state)[None], [tgt_ch],
                      normalized=False)[0, 0]
        floor_f[~mask] = np.nan
        floor = float(np.sqrt(np.nanmean((floor_f[mask] - truth[mask]) ** 2)))

        # Forecast roll (CRN seeds; retain the first predicted state).
        fb = roll_ensemble(models.model, prior_lat, DENOISER_SEED, key, c, device)
        arr = fb[:, :, 0]
        m_T = dec(arr, [tgt_ch])[:, 0]
        m_T[:, ~mask] = np.nan
        prior_rmse = ens_rmse(m_T, truth)

        chosen_field: list = []
        cand_rc = empty_rc
        obs_rc = empty_rc
        fleet_dump = np.empty((0, 2), np.float32)
        extras: dict = {}
        n_fallback = 0

        if free_running:
            post = arr
            m_T_post = m_T
            post_rmse = prior_rmse
        else:
            ctx = None
            if policy.moving:
                uv_fc = dec(arr, [u_ch, v_ch]).astype(float)
                uv_pr = dec(prior_lat, [u_ch, v_ch]).astype(float)
                for a2 in (uv_fc, uv_pr):
                    a2[:, :, ~mask] = np.nan
                c_field = 0.5 * (np.nanmean(uv_pr, 0) + np.nanmean(uv_fc, 0))
                cand_rc, feas, _ = advected_reachable(
                    fleet, wet, c_field, inputs.dx_col_m, inputs.dy_row_m,
                    V_AUV_MS, REACH_WINDOW_S, n_fleet)
                if len(cand_rc) == 0:
                    result.stopped_early = True
                    break
                cand_field = np.array(
                    [wet_lookup[(int(r), int(cc))] for r, cc in cand_rc])
                ctx = CycleContext(cycle=c, fleet=fleet, cand_rc=cand_rc,
                                   cand_field=cand_field, feas=feas, R=R)
                if policy.uses_ensemble:
                    ctx.member_target_fields = m_T
                fleet_dump = fleet.copy().astype(np.float32)   # start positions
            else:
                ctx = CycleContext(cycle=c, fleet=fleet, cand_rc=empty_rc,
                                   cand_field=np.empty(0, int), feas=[], R=R)
                fleet_dump = None                              # set below from cells
            chosen_field, new_fleet = policy.plan(ctx)
            extras = ctx.extras
            if not policy.moving:
                fleet_dump = wet[np.array(chosen_field)].astype(np.float32)
            if not chosen_field:
                result.stopped_early = True
                break
            obs_rc = wet[np.array(chosen_field)]
            rows, cols = obs_rc[:, 0], obs_rc[:, 1]
            y = observe(truth, obs_rc, sigma, obs_rng(key, c, rseed))

            post_members = []
            for m in range(n_members):
                seed = (member_seed(DENOISER_SEED, key, m)
                        + c * CYCLE_SEED_STRIDE + GUIDED_SEED_OFFSET
                        + rseed * GUIDED_RSEED_STRIDE)
                cond = torch.from_numpy(
                    prior_lat[m][None, :, None].astype(np.float32)).to(device)
                x0 = guided_update(
                    models.model, models.dcae, models.lnorm, models.pnorm,
                    cond, guided_start_noise(seed, pred_shape, device),
                    rows, cols, y, sigma, tgt_ch,
                    guidance_scale=guidance_scale, device=device)
                if torch.isfinite(x0).all():
                    post_members.append(x0[0, :, 0].float().cpu().numpy())
                else:
                    post_members.append(arr[m])                # fallback to prior
                    n_fallback += 1
            post = np.stack(post_members).astype(np.float32)
            m_T_post = dec(post, [tgt_ch])[:, 0]
            m_T_post[:, ~mask] = np.nan
            post_rmse = ens_rmse(m_T_post, truth)
            if policy.moving:
                fleet = new_fleet

        prior_lat = post
        if dump_latents:
            result.prior_latents_fp16.append(prior_lat.astype(np.float16))

        rec = dict(cycle=c, prior_rmse=prior_rmse, post_rmse=post_rmse,
                   floor=floor, n_obs=len(chosen_field),
                   wall_s=round(time.time() - t0, 2))
        if n_fallback:
            rec["n_fallback"] = n_fallback
        for k in ("stations_completed", "n_skipped"):
            if k in extras:
                rec[k] = extras[k]
        result.records.append(rec)
        result.chosen_log.append(np.asarray(obs_rc, np.int16).tolist())
        result.fleet_log.append(
            np.asarray(fleet_dump if fleet_dump is not None else empty_rc,
                       np.float32).tolist())
        if verbose:
            print(f"  [{strategy}{rseed if strategy == 'random' else ''}] "
                  f"cyc{c:2d}: prior={prior_rmse:.3f} post={post_rmse:.3f} "
                  f"floor={floor:.3f} nobs={len(chosen_field)} "
                  f"({rec['wall_s']:.0f}s)", flush=True)
    return result
