"""Training losses for the forecaster (wet-masked reconstruction error).

Implements the wet-masked objective of the autoencoder training recipe.
"""
from __future__ import annotations

import torch


def mask_aware_mse(
    pred: torch.Tensor,    # (B, 83, H, W)
    target: torch.Tensor,  # (B, 83, H, W)
    mask: torch.Tensor,    # (B,  1, H, W) — binary 0/1
) -> torch.Tensor:
    """Mean squared error over (sample, channel, wet-cell) triples.

      loss = sum((pred-target)^2 * mask) / (n_wet * n_channels)

    where n_wet is the total wet-cell count in the batch and n_channels is the
    number of dynamic channels. Static-channel reconstruction is stripped by
    RegionalDCAE.decode before this loss runs.
    """
    sq_err = (pred - target) ** 2 * mask          # (B, 83, H, W)
    n_wet = mask.sum()                            # scalar
    n_channels = pred.shape[1]
    return sq_err.sum() / (n_wet * n_channels + 1e-8)
