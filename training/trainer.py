"""Shared training utilities: the LR schedule used by both training recipes.

The two entry scripts (train_autoencoder.py, train_denoiser.py) keep their own
checkpoint save/prune helpers — the recipes checkpoint differently (the
autoencoder tracks a two-phase schedule) and the as-run behavior is preserved
verbatim per script.
"""
from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
) -> LambdaLR:
    """Linear warmup, then cosine decay toward zero (as-run schedule for both
    the autoencoder and the denoiser)."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)
