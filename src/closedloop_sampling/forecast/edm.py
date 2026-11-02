#!/usr/bin/env python
"""Heun EDM sampler (Karras 2022).

Same EDM math as `training.train_denoiser.compute_edm_loss` but in inference
direction: starts at sigma_max, iteratively denoises to sigma_min ~ 0.
Default is deterministic (`s_churn = 0`); set `s_churn > 0` for stochastic
sampling a la EDM Algorithm 2.

Reference: Karras, Aittala, Aila, Laine — "Elucidating the Design Space of
Diffusion-Based Generative Models" (NeurIPS 2022).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch


def sigmas_karras(
    n_steps: int,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    device: torch.device = None,
) -> torch.Tensor:
    """Karras 2022 rho-spaced sigma schedule.

    Returns (n_steps + 1,) tensor decreasing from sigma_max to sigma_min
    with a trailing 0.0 (the "clean" target).
    """
    ramp = torch.linspace(0, 1, n_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def edm_preconditioning(sigma: torch.Tensor, sigma_data: float):
    """Karras 2022 preconditioning factors. Repeated here so this
    module is self-contained (training script has its own copy)."""
    s2 = sigma ** 2
    sd2 = sigma_data ** 2
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * sigma_data / torch.sqrt(s2 + sd2)
    c_in = 1.0 / torch.sqrt(s2 + sd2)
    # `+1e-12` guards log(0) if sigma_next ever bottoms out at exactly 0.
    c_noise = 0.25 * torch.log(sigma + 1e-12)
    return c_skip, c_out, c_in, c_noise


def edm_denoise(
    model,
    noisy: torch.Tensor,         # (B, C, T_pred, H, W)
    sigma: torch.Tensor,         # scalar or (B,)
    conditioning: torch.Tensor,  # (B, C, T_cond, H, W)
    sigma_data: float,
    model_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    """One denoising step -> returns the x0 estimate."""
    if model_kwargs is None:
        model_kwargs = {}
    B = noisy.shape[0]
    sigma_b = sigma.expand(B) if sigma.dim() == 0 else sigma

    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma_b, sigma_data)
    c_skip_b = c_skip.view(B, 1, 1, 1, 1)
    c_out_b = c_out.view(B, 1, 1, 1, 1)
    c_in_b = c_in.view(B, 1, 1, 1, 1)

    result = model(
        hidden_states=c_in_b * noisy,
        timestep=c_noise,
        conditioning_tensors=conditioning,
        **model_kwargs,
    )
    if hasattr(result, "sample"):
        pred = result.sample
    elif isinstance(result, dict):
        pred = result["sample"]
    else:
        pred = result
    return c_skip_b * noisy + c_out_b * pred


@torch.no_grad()
def edm_sample(
    model,
    conditioning: torch.Tensor,   # (B, C, T_cond, H, W) — normalized latent
    pred_shape: Tuple[int, ...],  # (B, C, T_pred, H, W)
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    n_steps: int = 20,
    rho: float = 7.0,
    s_churn: float = 0.0,
    s_min: float = 0.0,
    s_max: float = float("inf"),
    s_noise: float = 1.0,
    model_kwargs: Optional[dict] = None,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Heun 2nd-order EDM sampler.

    Args:
      model: trained denoiser.
      conditioning: clean conditioning latent.
      pred_shape: shape of the prediction window to generate.
      n_steps: denoising steps (LaDCast uses 20 in val).
      s_churn/s_min/s_max/s_noise: stochasticity controls;
        defaults give deterministic sampling.
      model_kwargs: optional extras threaded straight into `model(...)`.
      generator: optional torch.Generator for reproducible noise.

    Returns: predicted clean latent of shape `pred_shape`.
    """
    if device is None:
        device = next(model.parameters()).device
    if model_kwargs is None:
        model_kwargs = {}

    sigmas = sigmas_karras(n_steps, sigma_min, sigma_max, rho, device=device)

    if generator is None:
        x = torch.randn(pred_shape, device=device) * sigmas[0]
    else:
        x = torch.randn(pred_shape, generator=generator, device=device) * sigmas[0]

    for i in range(n_steps):
        sigma_curr = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Optional churn (Algorithm 2 of Karras 2022).
        gamma = (
            min(s_churn / n_steps, 2 ** 0.5 - 1)
            if (s_churn > 0 and s_min <= sigma_curr.item() <= s_max)
            else 0.0
        )
        sigma_hat = sigma_curr * (gamma + 1)
        if gamma > 0:
            if generator is None:
                eps = torch.randn_like(x) * s_noise
            else:
                eps = torch.randn(
                    x.shape, generator=generator, device=device, dtype=x.dtype,
                ) * s_noise
            x = x + eps * (sigma_hat ** 2 - sigma_curr ** 2).sqrt()

        # Euler step.
        x0_hat = edm_denoise(
            model, x, sigma_hat, conditioning, sigma_data, model_kwargs,
        )
        d = (x - x0_hat) / sigma_hat
        x_next = x + d * (sigma_next - sigma_hat)

        # Heun correction (skip on the final step where sigma_next = 0).
        if sigma_next.item() > 0:
            x0_hat_next = edm_denoise(
                model, x_next, sigma_next, conditioning, sigma_data, model_kwargs,
            )
            d_next = (x_next - x0_hat_next) / sigma_next
            x = x + (d + d_next) * 0.5 * (sigma_next - sigma_hat)
        else:
            x = x_next

    return x
