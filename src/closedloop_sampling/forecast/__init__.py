"""Forecaster package: regional latent-diffusion models, EDM machinery,
checkpoint loading, ensemble generation (loop convention), decode, losses.

This is the only subpackage that imports torch and the upstream `ladcast`
dependency (via closedloop_sampling.forecast._upstream).
"""
from closedloop_sampling.forecast.edm import (
    edm_denoise,
    edm_preconditioning,
    edm_sample,
    sigmas_karras,
)

__all__ = ["edm_sample", "edm_denoise", "edm_preconditioning", "sigmas_karras"]
