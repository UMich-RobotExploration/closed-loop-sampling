"""Regional models: DC-AE (pixel <-> latent) and the EDM denoiser.

Historical aliases (`DoppioDCAE`, `make_unconditioned_denoiser`) are kept so
code written against the research pipeline keeps working.
"""
from closedloop_sampling.forecast.models.autoencoder import DoppioDCAE, RegionalDCAE
from closedloop_sampling.forecast.models.denoiser import (
    RegionalDenoiser,
    UnconditionedDenoiser,
    make_denoiser,
    make_unconditioned_denoiser,
)

__all__ = [
    "RegionalDCAE", "DoppioDCAE",
    "RegionalDenoiser", "UnconditionedDenoiser",
    "make_denoiser", "make_unconditioned_denoiser",
]
