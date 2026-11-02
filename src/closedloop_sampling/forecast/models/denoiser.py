#!/usr/bin/env python
"""Regional EDM denoiser: LaDCast 375M transformer with regional RoPE config.

Subclass of `LaDCastTransformer3DModel` with two changes vs upstream:
  - Regional RoPE config: INTEGER grid indices on the latent grid
    (`spatial_deg2rad=False`), no spherical mapping.
  - in_channels = out_channels = conditioning_tensor_in_channels = 84
    (matches the DC-AE latent dim).

No architectural changes beyond config. mask_rho information enters the
denoiser via the DC-AE latents (baked in by the trained encoder); the
denoiser itself sees no static channel.

NOTE on `incl_time_elapsed=True`: it allocates LaDCast's date (seasonal)
embedding module, but `time_elapsed` is NEVER passed at train or inference
time in this pipeline, so the branch is dead weight. It MUST stay True for
state-dict compatibility with the released checkpoints (the embedding's
parameters are part of the 374,938,452-param count).

Smoke test: python -m closedloop_sampling.forecast.models.denoiser
"""
from __future__ import annotations

import sys
from typing import Dict, Optional

from closedloop_sampling.forecast._upstream import ensure_ladcast

ensure_ladcast()

import torch

from ladcast.models.LaDCast_3D_model import LaDCastTransformer3DModel


# ---------------------------------------------------------------------------
# Default 375M config (LaDCast 375M with the regional RoPE overrides; the
# defaults are the MAB latent grid 13x30)
# ---------------------------------------------------------------------------
DEFAULT_375M_CONFIG: Dict[str, object] = dict(
    in_channels=84,
    out_channels=84,
    num_attention_heads=12,
    attention_head_dim=128,
    num_layers=2,
    num_single_layers=4,
    num_refiner_layers=1,
    mlp_ratio=4,
    patch_size=1,
    patch_size_t=1,
    qk_norm="rms_norm",
    rope_theta=256.0,
    rope_axes_dim=(16, 56, 56),
    # Integer-indexed RoPE on the latent grid (MAB default 13x30).
    rope_spatial_grid_start_pos=[0, 0],
    rope_spatial_grid_end_pos=[12, 29],  # eta_lat-1, xi_lat-1
    spatial_deg2rad=False,
    conditioning_tensor_in_channels=84,
    conditioning_tensor_rope_axes_dim=(16, 56, 56),
    incl_time_elapsed=True,   # dead branch; kept True for checkpoint compat (see module docstring)
)


class RegionalDenoiser(LaDCastTransformer3DModel):
    """LaDCast 375M denoiser with regional RoPE config. No bathymetry."""

    @staticmethod
    def default_375M_config() -> Dict[str, object]:
        """Return a fresh copy of the default (MAB-grid) 375M config dict."""
        return dict(DEFAULT_375M_CONFIG)


# Historical name (research code).
UnconditionedDenoiser = RegionalDenoiser


def make_denoiser(
    config: Optional[Dict[str, object]] = None,
    *,
    region_config=None,
) -> RegionalDenoiser:
    """Factory: instantiate RegionalDenoiser with the default 375M config
    (or a caller-supplied override).

    `region_config` (keyword-only): a RegionConfig whose latent grid drives
    rope_spatial_grid_{start,end}_pos. The RoPE grid is NOT auto-derived by the
    model, so a wrong end_pos is a SILENT positional-encoding error:
    MAB latent 13x30 -> end [12,29]; CCS latent 22x23 -> end [21,22].
    """
    if config is None:
        config = RegionalDenoiser.default_375M_config()
    else:
        config = dict(config)
    if region_config is not None:
        lat = region_config.latent
        config["rope_spatial_grid_start_pos"] = list(lat.rope_spatial_grid_start_pos)
        config["rope_spatial_grid_end_pos"] = list(lat.rope_spatial_grid_end_pos)
        config["in_channels"] = lat.latent_channels
        config["out_channels"] = lat.latent_channels
        config["conditioning_tensor_in_channels"] = lat.latent_channels
    return RegionalDenoiser(**config)


# Historical factory name (research code called the production arm "unconditioned").
make_unconditioned_denoiser = make_denoiser


# ---------------------------------------------------------------------------
# Smoke test (CPU, MAB grid)
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    torch.manual_seed(0)
    failed = []

    print("Instantiating RegionalDenoiser (CPU)...", flush=True)
    model = make_denoiser().eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params/1e6:.1f}M  ({n_params})")
    # The released checkpoints carry exactly 374,938,452 params for this config.
    if n_params != 374_938_452:
        failed.append(f"param count {n_params} != 374,938,452 (375M reference)")

    hidden = torch.randn(1, 84, 4, 13, 30)
    cond = torch.randn(1, 84, 1, 13, 30)
    timestep = torch.tensor([500], dtype=torch.long)
    with torch.no_grad():
        out = model(hidden_states=hidden, timestep=timestep,
                    conditioning_tensors=cond, return_dict=True).sample
    print(f"  forward out shape: {tuple(out.shape)}  (expect (1, 84, 4, 13, 30))")
    if tuple(out.shape) != (1, 84, 4, 13, 30):
        failed.append(f"out shape {tuple(out.shape)}")
    if not torch.isfinite(out).all():
        failed.append("output not all-finite")

    if failed:
        print("SMOKE TEST FAIL:", failed)
        return 1
    print("SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
