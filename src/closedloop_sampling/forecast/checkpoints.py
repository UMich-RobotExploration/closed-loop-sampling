"""Checkpoint loading for the released forecaster weights.

Implements the inference convention of the released checkpoints: EMA shadow
parameters (ema.pt), best-on-validation selection. Loads resolve against
$CLOSEDLOOP_WEIGHTS using the release layout

    <weights>/<region>/dcae/{model.pt, ema.pt}
    <weights>/<region>/denoiser/seed<N>/{model.pt, ema.pt}

or an explicit checkpoint directory. Loading asserts the exact released
parameter counts: autoencoder 256,388,460 and denoiser 374,938,452.
"""
from __future__ import annotations

import gc
from pathlib import Path
from typing import Optional

import torch

from closedloop_sampling import paths
from closedloop_sampling.forecast.models import RegionalDCAE, make_denoiser

AUTOENCODER_PARAMS = 256_388_460
DENOISER_PARAMS = 374_938_452


def _load_ema_into(model: torch.nn.Module, ema_path: Path) -> torch.nn.Module:
    """Copy EMA shadow parameters into `model` in place (the convention every
    published number was produced with)."""
    ema_path = Path(ema_path)
    if not ema_path.exists():
        raise FileNotFoundError(f"No EMA checkpoint at {ema_path}")
    state = torch.load(ema_path, map_location="cpu", weights_only=False, mmap=True)
    if "shadow_params" not in state:
        raise ValueError(f"{ema_path} has no 'shadow_params' key — unexpected format")
    shadow = state["shadow_params"]
    params = list(model.parameters())
    if len(shadow) != len(params):
        raise ValueError(
            f"Parameter-count mismatch loading EMA: ema {len(shadow)} tensors, "
            f"model {len(params)}."
        )
    with torch.no_grad():
        for mp, sp in zip(params, shadow):
            mp.data.copy_(sp.data)
    del state, shadow
    gc.collect()
    return model


def _assert_params(model: torch.nn.Module, expected: Optional[int], what: str):
    if expected is None:
        return
    n = sum(p.numel() for p in model.parameters())
    if n != expected:
        raise RuntimeError(
            f"{what}: parameter count {n:,} != released {expected:,}. "
            f"The model config does not match the released checkpoints."
        )


def autoencoder_dir(region: str) -> Path:
    return paths.weights_root() / region / "dcae"


def denoiser_dir(region: str, seed: int) -> Path:
    return paths.weights_root() / region / "denoiser" / f"seed{seed}"


def load_autoencoder(
    checkpoint_dir: Optional[Path] = None,
    *,
    region: Optional[str] = None,
    use_ema: bool = True,
    device=None,
    region_config=None,
    expect_params: Optional[int] = AUTOENCODER_PARAMS,
) -> RegionalDCAE:
    """Load the RegionalDCAE from a checkpoint directory (EMA by default).

    Pass either `checkpoint_dir` (a directory containing ema.pt / model.pt) or
    `region` ("mab" | "ccs") to resolve the release layout under
    $CLOSEDLOOP_WEIGHTS. Both regions' weights load into the same class — the
    network is fully convolutional, so the region only changes the input grid.
    """
    if checkpoint_dir is None:
        if region is None:
            raise ValueError("pass checkpoint_dir or region")
        checkpoint_dir = autoencoder_dir(region)
    checkpoint_dir = Path(checkpoint_dir)
    model = RegionalDCAE(region_config=region_config)
    _assert_params(model, expect_params, "autoencoder")
    if use_ema:
        _load_ema_into(model, checkpoint_dir / "ema.pt")
    else:
        state = torch.load(checkpoint_dir / "model.pt", map_location="cpu",
                           weights_only=False)
        model.load_state_dict(state)
    if device is not None:
        model.to(device)
    model.eval()
    return model


def load_denoiser(
    checkpoint_dir: Optional[Path] = None,
    *,
    region: Optional[str] = None,
    seed: int = 0,
    use_ema: bool = True,
    device=None,
    region_config=None,
    expect_params: Optional[int] = DENOISER_PARAMS,
):
    """Load the RegionalDenoiser from a checkpoint directory (EMA by default).

    Pass either `checkpoint_dir` or (`region`, `seed`) for the release layout.
    `region_config` drives the RoPE grid; passing the wrong region's config is
    a SILENT positional-encoding error, so when `region` is given and
    `region_config` is not, the config is loaded from the region registry.
    """
    if checkpoint_dir is None:
        if region is None:
            raise ValueError("pass checkpoint_dir or region")
        checkpoint_dir = denoiser_dir(region, seed)
    if region_config is None and region is not None:
        from closedloop_sampling.regions import load as load_region
        region_config = load_region(region)
    checkpoint_dir = Path(checkpoint_dir)
    model = make_denoiser(region_config=region_config)
    _assert_params(model, expect_params, "denoiser")
    if use_ema:
        _load_ema_into(model, checkpoint_dir / "ema.pt")
    else:
        state = torch.load(checkpoint_dir / "model.pt", map_location="cpu",
                           weights_only=False)
        model.load_state_dict(state)
    if device is not None:
        model.to(device)
    model.eval()
    return model
