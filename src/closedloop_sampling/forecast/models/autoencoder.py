#!/usr/bin/env python
"""Regional DC-AE wrapper around LaDCast's `AutoencoderDC`.

Adaptations vs upstream LaDCast (global model) for bounded regional domains:
  1. `SphereConv2d` -> `nn.Conv2d(padding_mode='replicate')` at all 10 sites.
     LaDCast's `SphereConv2d` assumes global-longitude periodicity and pole-
     fold; a regional domain is bounded, so spherical padding is wrong.
  2. `in_channels = out_channels = 84` (83 dynamic + 1 static `mask_rho`).
     Bathymetry `h` is NOT a static autoencoder channel.

The upstream `ladcast` package is untouched: we subclass + monkey-patch at
construction time via a one-shot module-tree walk.

Historical name: this class was `DoppioDCAE` in the research code (after the
MAB dataset); the compat alias is kept because released checkpoints and
downstream code reference the loading pattern. CCS weights load into the same
class — the network is fully convolutional, so the region only changes the
input grid.

Run smoke test:
  python -m closedloop_sampling.forecast.models.autoencoder
"""
from __future__ import annotations

import copy
import sys
from typing import Dict, List, Optional

from closedloop_sampling.forecast._upstream import ensure_ladcast

ensure_ladcast()

import torch
import torch.nn as nn

from ladcast.models.DCAE import AutoencoderDC, SphereConv2d


# ---------------------------------------------------------------------------
# Default LaDCast DC-AE config (DC_AE_84_pretrain.yaml `encdec:` block)
# with the three regional overrides applied. All architectural fields (block
# types, channel widths, attention head dim, qkv multiscales, etc.) match
# LaDCast verbatim. spatial_compression_ratio = 2^(4-1) = 8x.
# ---------------------------------------------------------------------------
DEFAULT_DCAE_CONFIG: Dict[str, object] = dict(
    in_channels=84,             # 83 dynamic + 1 static  (LaDCast: 89)
    out_channels=84,            # mirror in_channels      (LaDCast: 89)
    latent_channels=84,         # latent rep dim          (LaDCast: 84)
    static_channels=1,          # mask_rho only           (LaDCast: 5)
    attention_head_dim=32,
    encoder_block_types=("ResBlock", "ResBlock",
                         "EfficientViTBlock", "EfficientViTBlock"),
    decoder_block_types=("ResBlock", "ResBlock",
                         "EfficientViTBlock", "EfficientViTBlock"),
    encoder_block_out_channels=(252, 504, 504, 1008),
    decoder_block_out_channels=(252, 504, 504, 1008),
    encoder_layers_per_block=(4, 4, 4, 4),
    decoder_layers_per_block=(4, 4, 4, 4),
    encoder_qkv_multiscales=((), (), (5,), (5,)),
    decoder_qkv_multiscales=((), (), (5,), (5,)),
    upsample_block_type="pixel_shuffle",
    downsample_block_type="pixel_unshuffle",
)


# ---------------------------------------------------------------------------
# SphereConv2d -> Conv2d(padding_mode='replicate') swap
# ---------------------------------------------------------------------------

def replace_sphere_with_replicate(module: nn.Module) -> nn.Module:
    """Recursively replace every `SphereConv2d` with a plain `nn.Conv2d`
    using `padding_mode='replicate'`.

    Preserves `in_channels`, `out_channels`, `kernel_size`, `stride`,
    `padding`, `dilation`, `groups`, and the presence/absence of `bias`.
    Weight and bias tensors are copied from the original so the swap is a
    no-op at construction time (the original `SphereConv2d` was Kaiming-
    initialised by its `nn.Conv2d` super-`__init__`; replication-padding
    Conv2d would do the same, but copying makes the swap fully deterministic
    regardless of any future change to either init scheme).

    Returns the same `module` for chaining; mutation is in place.
    """
    # Materialise the (name, child) list first — we mutate during the walk.
    sphere_sites: List[tuple] = [
        (name, child) for name, child in module.named_modules()
        if isinstance(child, SphereConv2d)
    ]

    for full_name, old in sphere_sites:
        replacement = nn.Conv2d(
            in_channels=old.in_channels,
            out_channels=old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=(old.bias is not None),
            padding_mode="replicate",
        )
        # Preserve the original tensors so the wrapper is deterministic w.r.t.
        # whatever init the SphereConv2d already had (Kaiming via its super).
        with torch.no_grad():
            replacement.weight.copy_(old.weight)
            if old.bias is not None:
                replacement.bias.copy_(old.bias)

        # Walk to the parent module via attribute path and setattr.
        parent_path, _, child_name = full_name.rpartition(".")
        parent = module
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        setattr(parent, child_name, replacement)

    # Belt-and-braces: confirm none remain.
    leftover = [n for n, m in module.named_modules()
                if isinstance(m, SphereConv2d)]
    if leftover:
        raise RuntimeError(
            f"replace_sphere_with_replicate left {len(leftover)} "
            f"SphereConv2d(s) behind: {leftover[:5]}"
        )
    return module


# ---------------------------------------------------------------------------
# RegionalDCAE
# ---------------------------------------------------------------------------

class RegionalDCAE(nn.Module):
    """Thin regional wrapper around LaDCast's `AutoencoderDC`.

    Args:
      config: optional dict of overrides applied on top of
        `DEFAULT_DCAE_CONFIG`. Anything left unspecified uses the LaDCast
        default. `**kwargs` override `config`, which overrides the default.

    Forward contract (all tensors batch-first; MAB: H=104, W=240):
      encode(x, static):
        x       : (B, 83, H, W)  normalized dynamic state.
        static  : (B,  1, H, W)  `mask_rho` (binary 0/1, NOT normalized).
        returns : (B, 84, H/8, W/8) latent — MAB (B, 84, 13, 30).
      decode(latent):
        latent  : (B, 84, H/8, W/8)
        returns : (B, 83, H, W) reconstructed dynamic state
                  (the static-channel reconstruction is stripped by
                  AutoencoderDC.decode via `return_static=False`).
      forward(x, static):
        Convenience: encode-then-decode, returns
        {'latent': ..., 'reconstructed': ...}.
    """

    def __init__(
        self,
        config: Optional[Dict[str, object]] = None,
        *,
        region_config=None,
        **kwargs,
    ):
        super().__init__()

        resolved = copy.deepcopy(DEFAULT_DCAE_CONFIG)
        if region_config is not None:
            # Derive channel counts from a RegionConfig. Architecture fields
            # (block types/widths/qkv/compression) stay LaDCast-fixed — only
            # the channel counts are region-dependent. For BOTH MAB and CCS
            # these resolve to 84/84/84/1 (identical 4x20+3 dynamic + 1 static,
            # latent 84), so the model is byte-identical across the region
            # swap; this hook only matters for a future region with a
            # different channel count. `region_config` is keyword-only so the
            # existing positional `config` (arch-override dict) and the
            # zero-arg call `RegionalDCAE()` are both unaffected.
            n_in = (region_config.channels.n_dynamic_channels
                    + region_config.channels.static_channels)
            resolved.update(
                in_channels=n_in,
                out_channels=n_in,
                latent_channels=region_config.latent.latent_channels,
                static_channels=region_config.channels.static_channels,
            )
        if config is not None:
            resolved.update(config)
        resolved.update(kwargs)

        self.dcae = AutoencoderDC(**resolved)
        self.dcae = replace_sphere_with_replicate(self.dcae)

        # Stash the effective config for downstream introspection.
        self._effective_config = resolved

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if static.shape[1] != self._effective_config["static_channels"]:
            raise ValueError(
                f"static has {static.shape[1]} channels; expected "
                f"{self._effective_config['static_channels']} (mask_rho only)."
            )
        out = self.dcae.encode(
            x, static_conditioning_tensor=static, return_dict=True
        )
        return out.latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        out = self.dcae.decode(latent, return_dict=True, return_static=False)
        return out.sample

    def forward(
        self, x: torch.Tensor, static: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        latent = self.encode(x, static)
        reconstructed = self.decode(latent)
        return {"latent": latent, "reconstructed": reconstructed}


# Historical name (research code + released checkpoint loaders).
DoppioDCAE = RegionalDCAE


# ---------------------------------------------------------------------------
# Smoke test (CPU; fresh init, MAB grid)
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    torch.manual_seed(0)
    print("Instantiating RegionalDCAE() (CPU)...", flush=True)
    model = RegionalDCAE().eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  param count: {n_params/1e6:.1f}M  ({n_params})")

    n_sphere = sum(1 for m in model.modules() if isinstance(m, SphereConv2d))
    n_replicate = sum(
        1 for m in model.modules()
        if isinstance(m, nn.Conv2d) and getattr(m, "padding_mode", None) == "replicate"
    )
    print(f"  SphereConv2d remaining: {n_sphere}  (must be 0)")
    print(f"  Conv2d(padding_mode='replicate'): {n_replicate}")

    x = torch.randn(1, 83, 104, 240)
    static = torch.ones(1, 1, 104, 240)
    with torch.no_grad():
        latent = model.encode(x, static)
        recon = model.decode(latent)
    print(f"  encode out shape: {tuple(latent.shape)}  (expect (1, 84, 13, 30))")
    print(f"  decode out shape: {tuple(recon.shape)}   (expect (1, 83, 104, 240))")

    failed = []
    if n_sphere != 0:
        failed.append(f"SphereConv2d remaining = {n_sphere}")
    if tuple(latent.shape) != (1, 84, 13, 30):
        failed.append(f"latent shape {tuple(latent.shape)}")
    if tuple(recon.shape) != (1, 83, 104, 240):
        failed.append(f"recon shape {tuple(recon.shape)}")
    if not torch.isfinite(latent).all() or not torch.isfinite(recon).all():
        failed.append("non-finite outputs")
    if failed:
        print("SMOKE TEST FAIL:", failed)
        return 1
    print("SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(_smoke_test())
