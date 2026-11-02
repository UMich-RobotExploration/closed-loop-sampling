"""Ensemble generation — the paper's closed-loop convention, and only it.

Implements Sec. III (Forecast) and Alg. 1 lines 3-5: each cycle, every member
of the analysis ensemble conditions one diffusion generation that predicts 4
successive states at 6 h spacing in a single pass; the FIRST (+6 h) predicted
state is retained and becomes that member's forecast state. bf16 autocast on
CUDA. Member noise is seeded by the common-random-numbers (CRN) scheme, so
compared configurations at the same (IC, seed) see the same forecast noise:

    member_seed(denoiser_seed, roster_pos, m) = denoiser_seed*1_000_000
                                              + roster_pos*1_000 + m
    forecast roll at cycle c uses  member_seed + c*100003 + 777777

`roster_pos` is the IC key: for date-addressed ICs it is the IC's
days-since-2000 integer (see closedloop_sampling.observations.ic_key).

The research code's standalone generation path (50 ICs x 50 members, fp32,
retain-all) is deliberately not ported: no paper artifact uses it.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from closedloop_sampling.forecast.edm import edm_sample

# EDM constants as trained/run (Karras 2022 schedule).
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
SIGMA_DATA = 0.5
N_SAMPLING_STEPS = 20
PRED_STEPS_PER_CALL = 4
HOURS_PER_STEP = 6

# CRN seed constants (production values; also recorded in
# configs/experiments/closed_loop.yaml).
CYCLE_SEED_STRIDE = 100003
FORECAST_SEED_OFFSET = 777777


def member_seed(denoiser_seed: int, roster_pos: int, member: int) -> int:
    """Base CRN member seed (production formula, verbatim)."""
    return denoiser_seed * 1_000_000 + roster_pos * 1_000 + member


def forecast_roll_seed(denoiser_seed: int, roster_pos: int, member: int,
                       cycle: int) -> int:
    """Seed for the forecast roll of `member` at `cycle`. Independent of the
    chain's observation seed (rseed), so the forecast noise is common across
    policies and observation seeds — the pairing behind the paper's
    per-(IC, seed) comparisons."""
    return (member_seed(denoiser_seed, roster_pos, member)
            + cycle * CYCLE_SEED_STRIDE + FORECAST_SEED_OFFSET)


def initial_ensemble(ic_latent_normalized: np.ndarray, n_members: int) -> np.ndarray:
    """N identical copies of the normalized IC latent, (N, C, H, W) float32.

    The chain starts from the IC itself (start_lead = 0): member spread is
    created by the first generative roll, not by perturbing the IC.
    """
    return np.tile(ic_latent_normalized[None], (n_members, 1, 1, 1)).astype(np.float32)


@torch.no_grad()
def roll_member(
    model,
    cond_latent: np.ndarray,          # (C, H, W) normalized analysis latent
    seed: int,
    device: torch.device,
    n_sampling_steps: int = N_SAMPLING_STEPS,
    sigma_min: float = SIGMA_MIN,
    sigma_max: float = SIGMA_MAX,
    sigma_data: float = SIGMA_DATA,
    pred_steps: int = PRED_STEPS_PER_CALL,
) -> np.ndarray:
    """One member's forecast block: (C, pred_steps, H, W) float32.

    One diffusion generation under bf16 autocast (CUDA), conditioned on the
    member's analysis latent, with a dedicated generator seeded by the CRN
    scheme. All `pred_steps` predicted states are returned; the closed loop
    retains index 0 (+6 h).
    """
    cond = torch.from_numpy(cond_latent[None, :, None].astype(np.float32)).to(device)
    C, H, W = cond_latent.shape
    pred_shape = (1, C, pred_steps, H, W)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    ac = torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"))
    with ac:
        x = edm_sample(
            model, conditioning=cond, pred_shape=pred_shape,
            sigma_min=sigma_min, sigma_max=sigma_max, sigma_data=sigma_data,
            n_steps=n_sampling_steps, generator=gen, device=device,
        )
    return x[0].float().cpu().numpy()


def roll_ensemble(
    model,
    prior_latents: np.ndarray,        # (N, C, H, W) normalized analysis latents
    denoiser_seed: int,
    roster_pos: int,
    cycle: int,
    device: torch.device,
    seeds: Optional[Sequence[int]] = None,
    **roll_kwargs,
) -> np.ndarray:
    """Forecast blocks for the whole ensemble: (N, C, pred_steps, H, W).

    Default seeding is the production CRN scheme per member; pass `seeds`
    (length N) to override. retain the FIRST predicted state downstream with
    `blocks[:, :, 0]` — the loop convention.
    """
    n = prior_latents.shape[0]
    if seeds is None:
        seeds = [forecast_roll_seed(denoiser_seed, roster_pos, m, cycle)
                 for m in range(n)]
    blocks = [roll_member(model, prior_latents[m], seeds[m], device, **roll_kwargs)
              for m in range(n)]
    return np.stack(blocks)
