"""Decode latent forecasts back to pixel space (for physical fields/metrics).

Decoding cost: one ~256M-parameter autoencoder forward per (ic, member, step);
cache per run so re-running a metric pipeline does not re-decode. Checkpoint
loading lives in closedloop_sampling.forecast.checkpoints.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
import zarr
import zarr.codecs

from closedloop_sampling.forecast.models import RegionalDCAE


@torch.no_grad()
def decode_latent_forecasts(
    latents: np.ndarray,        # (n_ic, n_members, C, n_steps, H_lat, W_lat) normalized
    dcae: RegionalDCAE,
    device: torch.device,
    latent_normalizer=None,
    batch_size: int = 8,
    precision: str = "bf16",
) -> np.ndarray:
    """Decode latent forecasts -> normalized pixel space.

    Iterates one timestep at a time, batching across (ic, member) pairs.
    Output pixel shape is taken from the decoder itself (region-agnostic).
    Returns `(n_ic, n_members, C_pix, n_steps, H, W)`. Still normalized in
    pixel space — call `denormalize_pixel_forecasts` to convert to physical
    units.
    """
    n_ic, n_members, C, n_steps, H_lat, W_lat = latents.shape
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    out = None                                   # allocated after the first decode

    for step in range(n_steps):
        slice_np = latents[:, :, :, step, :, :].reshape(
            n_ic * n_members, C, H_lat, W_lat,
        )
        slice_t = torch.from_numpy(slice_np.astype(np.float32)).to(device)
        if latent_normalizer is not None:
            slice_t = latent_normalizer.denormalize_dynamic_tensor(slice_t)

        decoded_chunks = []
        for b in range(0, n_ic * n_members, batch_size):
            chunk = slice_t[b : b + batch_size]
            with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype):
                out_chunk = dcae.decode(chunk)
            decoded_chunks.append(out_chunk.detach().float().cpu())
        decoded_step = torch.cat(decoded_chunks, dim=0)

        if out is None:
            C_pix, H, W = decoded_step.shape[1:]
            out = np.zeros((n_ic, n_members, C_pix, n_steps, H, W), dtype=np.float32)
        out[:, :, :, step, :, :] = decoded_step.numpy().reshape(
            n_ic, n_members, *decoded_step.shape[1:],
        )
    return out


def cache_decoded_forecasts(
    decoded: np.ndarray,           # (n_ic, n_members, C_pix, n_steps, H, W)
    output_zarr_path: Path,
    overwrite: bool = False,
):
    """Write the decoded forecasts to a Zarr (zarr-3.x API)."""
    if output_zarr_path.exists():
        if overwrite:
            shutil.rmtree(output_zarr_path)
        else:
            raise FileExistsError(
                f"{output_zarr_path} exists; pass overwrite=True"
            )
    output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(str(output_zarr_path), mode="w")
    n_ic, n_mem, C_pix, n_steps, H, W = decoded.shape
    root.create_array(
        "decoded",
        shape=decoded.shape,
        # One (ic, member, step) slice per chunk.
        chunks=(1, 1, C_pix, 1, H, W),
        dtype=np.float32,
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
        dimension_names=["n_ic", "n_members", "C", "n_steps", "H", "W"],
    )[:] = decoded


def load_cached_decoded(zarr_path: Path) -> np.ndarray:
    root = zarr.open(str(zarr_path), mode="r")
    return root["decoded"][:]


def denormalize_pixel_forecasts(
    decoded_normalized: np.ndarray,
    pixel_normalizer,
    channel_axis: int = -4,
) -> np.ndarray:
    """Normalized pixel-space -> physical units (m/s, degC, PSU, m).

    `channel_axis` default = -4 matches both canonical layouts:
      - 6D forecasts `(n_ic, n_mem, C, n_steps, H, W)` -> C at axis 2
      - 5D truth    `(n_ic, C, n_steps, H, W)`        -> C at axis 1
    """
    if channel_axis < 0:
        channel_axis += decoded_normalized.ndim
    moved = np.moveaxis(decoded_normalized, channel_axis, 0)
    C = moved.shape[0]
    H, W = moved.shape[-2], moved.shape[-1]
    rest_shape = moved.shape[1:-2]
    B = int(np.prod(rest_shape)) if rest_shape else 1
    flat = moved.reshape(C, B, H, W).transpose(1, 0, 2, 3)
    flat_t = torch.from_numpy(flat.astype(np.float32))
    flat_t = pixel_normalizer.denormalize_dynamic_tensor(flat_t)
    out = flat_t.numpy().transpose(1, 0, 2, 3).reshape(C, *rest_shape, H, W)
    return np.moveaxis(out, 0, channel_axis)
