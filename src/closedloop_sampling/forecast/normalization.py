#!/usr/bin/env python
"""Per-channel z-score normalization for pixel and latent datasets.

Per-channel normalization at dataloader time; two normalization sets
(pixel + latent); the static mask_rho passes through unnormalized
(handled in pixel_dataset, not here).

Smoke test: python -m closedloop_sampling.forecast.normalization <pixel_stats.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch


class Normalizer:
    """Per-channel z-score normalizer with two modes.

    Pixel mode: dynamic vars by name; 3D vars store per-sigma-level
    arrays of means/stds; 2D vars store scalars. The flat channel
    layout produced by `normalize_dynamic_tensor` follows `var_order`
    with each 3D var contributing n_levels channels in level-order
    (must match the dataset's stacking order).

    Latent mode: single 'latent' entry with an 84-element vector.
    """

    def __init__(
        self,
        means: Dict[str, np.ndarray],
        stds: Dict[str, np.ndarray],
        var_order: List[str] = None,
        mode: str = "pixel",
    ):
        self.means = means
        self.stds = stds
        self.var_order = var_order or list(means.keys())
        self.mode = mode

    def _flat_mean_std(self) -> tuple:
        """Flatten (var_order x per-level/scalar) into two 1-D arrays
        matching the dataset's channel-stacking order."""
        means_list: List[float] = []
        stds_list: List[float] = []
        for var in self.var_order:
            m = self.means[var]
            s = self.stds[var]
            if isinstance(m, np.ndarray) and m.ndim > 0:
                means_list.extend(m.tolist())
                stds_list.extend(s.tolist())
            else:
                means_list.append(float(m))
                stds_list.append(float(s))
        return np.asarray(means_list, dtype=np.float32), np.asarray(stds_list, dtype=np.float32)

    def normalize_dynamic_tensor(self, dynamic: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, C, H, W) with per-channel z-score."""
        means_np, stds_np = self._flat_mean_std()
        if means_np.shape[0] != dynamic.shape[1]:
            raise ValueError(
                f"Normalizer channel count {means_np.shape[0]} != "
                f"tensor C={dynamic.shape[1]}"
            )
        means_t = torch.from_numpy(means_np).to(dynamic.device, dynamic.dtype).view(1, -1, 1, 1)
        stds_t = torch.from_numpy(stds_np).to(dynamic.device, dynamic.dtype).view(1, -1, 1, 1)
        return (dynamic - means_t) / (stds_t + 1e-8)

    def denormalize_dynamic_tensor(self, dynamic_norm: torch.Tensor) -> torch.Tensor:
        """Inverse z-score: (x_norm * std) + mean."""
        means_np, stds_np = self._flat_mean_std()
        means_t = torch.from_numpy(means_np).to(dynamic_norm.device, dynamic_norm.dtype).view(1, -1, 1, 1)
        stds_t = torch.from_numpy(stds_np).to(dynamic_norm.device, dynamic_norm.dtype).view(1, -1, 1, 1)
        return dynamic_norm * (stds_t + 1e-8) + means_t


def load_pixel_normalizer(stats_json_path: str) -> Normalizer:
    """Load the pixel-space normalizer from a region's norm-stats JSON.

    Dynamic var order (matches RegionalPixelDataset's channel layout):
      u (20 levels) -> v (20 levels) -> temp (20 levels) -> salt (20 levels)
      -> zeta -> ubar -> vbar  = 83 channels total.

    An 'h' key, if present in the stats file, is NOT part of the dynamic
    var_order; it is available to callers via the raw JSON.
    """
    with open(stats_json_path) as f:
        stats = json.load(f)

    var_order = ["u", "v", "temp", "salt", "zeta", "ubar", "vbar"]
    means: Dict[str, np.ndarray] = {}
    stds: Dict[str, np.ndarray] = {}
    for var in var_order:
        if var not in stats:
            raise ValueError(f"Var {var!r} missing from {stats_json_path}")
        entry = stats[var]
        if "levels" in entry:
            means[var] = np.asarray(entry["mean"], dtype=np.float32)
            stds[var] = np.asarray(entry["std"], dtype=np.float32)
        else:
            means[var] = np.float32(entry["mean"])
            stds[var] = np.float32(entry["std"])
    return Normalizer(means, stds, var_order=var_order, mode="pixel")


def load_latent_normalizer(stats_json_path: str) -> Normalizer:
    """Load the latent-space normalizer from a region's latent-stats JSON.

    Expected schema (produced by the latent-encoding step):
      {"mean": [..84 floats..], "std": [..84 floats..]}
    """
    if not Path(stats_json_path).exists():
        raise FileNotFoundError(
            f"Latent stats not found at {stats_json_path}. "
            f"This file is produced by the latent-encoding step "
            f"(training/encode_latents.py); run that first."
        )
    with open(stats_json_path) as f:
        stats = json.load(f)
    means = np.asarray(stats["mean"], dtype=np.float32)
    stds = np.asarray(stats["std"], dtype=np.float32)
    if means.shape != (84,):
        raise ValueError(f"Expected (84,) mean, got {means.shape}")
    if stds.shape != (84,):
        raise ValueError(f"Expected (84,) std, got {stds.shape}")
    return Normalizer(
        means={"latent": means},
        stds={"latent": stds},
        var_order=["latent"],
        mode="latent",
    )


# ---------------------------------------------------------------------------
# Smoke test (takes the pixel-stats JSON path as argv[1])
# ---------------------------------------------------------------------------

def _smoke_test(stats_path: str) -> int:
    failed: List[str] = []

    norm = load_pixel_normalizer(stats_path)
    means_flat, stds_flat = norm._flat_mean_std()
    expected_channels = 4 * 20 + 3  # u,v,temp,salt x 20 levels + zeta,ubar,vbar
    if means_flat.shape[0] != expected_channels:
        failed.append(f"flat channel count {means_flat.shape[0]} != {expected_channels}")
    print(f"  pixel normalizer: {means_flat.shape[0]} channels "
          f"(expected {expected_channels})")

    # x = mean + 2*std -> after normalization ~ 2 per channel.
    B, C, H, W = 2, expected_channels, 8, 8
    means_b = torch.from_numpy(means_flat).view(1, C, 1, 1)
    stds_b = torch.from_numpy(stds_flat).view(1, C, 1, 1)
    x_phys = means_b + 2.0 * stds_b + torch.zeros(B, C, H, W)
    x_norm = norm.normalize_dynamic_tensor(x_phys)
    err = (x_norm - 2.0).abs().max().item()
    if err > 1e-3:
        failed.append(f"normalize round-trip: max|err - 2| = {err:.4e}")
    x_back = norm.denormalize_dynamic_tensor(x_norm)
    err2 = (x_back - x_phys).abs().max().item()
    if err2 > 1e-2:
        failed.append(f"denormalize round-trip: max|err| = {err2:.4e}")
    print(f"  forward max|val - 2| = {err:.2e}; round-trip max|err| = {err2:.2e}")

    if failed:
        print("NORMALIZER SMOKE TEST FAIL:", failed)
        return 1
    print("NORMALIZER SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m closedloop_sampling.forecast.normalization <pixel_stats.json>")
    sys.exit(_smoke_test(sys.argv[1]))
