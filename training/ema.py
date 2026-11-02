"""EMA construction with the as-run constants.

Both recipes track an exponential moving average of the weights
(decay 0.9999 with warmup: update_after_step 1000, inv_gamma 1.0, power
0.667) and the released checkpoints' ema.pt files hold these shadow
parameters — inference always uses them.
"""
from __future__ import annotations

from diffusers.training_utils import EMAModel


def build_ema(parameters, config: dict) -> EMAModel:
    """EMAModel over `parameters` using the recipe's ema_* config keys."""
    return EMAModel(
        parameters,
        decay=config["ema_decay"],
        use_ema_warmup=config["ema_use_warmup"],
        update_after_step=config["ema_update_after_step"],
        inv_gamma=config["ema_inv_gamma"],
        power=config["ema_power"],
    )
