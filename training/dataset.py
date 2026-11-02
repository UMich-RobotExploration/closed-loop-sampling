"""Datasets and samplers for training (pixel store, latent store, sliding-window sampler).

Merged from the forecaster package's data modules; class and function bodies are
unchanged. Normalizers live in closedloop_sampling.forecast.normalization.
"""
from __future__ import annotations


# ======================================================================
# ---- merged from data/sampler.py ----
# ======================================================================
#!/usr/bin/env python
"""Sliding-window index generation and year->time-index mapping.

Sliding window stride 1, shuffled per epoch, sequence = 1 conditioning +
4 prediction frames at 6-hour physical spacing (internal_stride converts
physical spacing to index steps per the region's store cadence).

Both utilities are fully region-agnostic — they take the time array / index
bounds / window params as arguments; a RegionConfig's timesplit is threaded
in by the caller.

Smoke test: python -m closedloop_sampling.data.sampler
"""

import sys
from typing import List, Tuple

import numpy as np


def year_to_index_range(
    year_start: int,
    year_end: int,
    time_array: np.ndarray,
) -> Tuple[int, int]:
    """Map an inclusive year range to (start_idx, end_idx_exclusive)
    on `time_array` (numpy datetime64).

    `time_array[start_idx:end_idx]` is the slice covering all timesteps
    whose calendar year lies in [year_start, year_end].
    """
    start_target = np.datetime64(f"{year_start}-01-01T00:00:00")
    end_target = np.datetime64(f"{year_end + 1}-01-01T00:00:00")
    start_idx = int(np.searchsorted(time_array, start_target, side="left"))
    end_idx = int(np.searchsorted(time_array, end_target, side="left"))
    return start_idx, end_idx


class SlidingWindowSampler:
    """Generates sliding-window starts for a 1-cond + 4-pred sequence
    with a configurable internal stride.

    For a window of `sequence_length` snapshots with `internal_stride`,
    a window starting at index `s` consumes the indices
    `[s, s + stride, s + 2*stride, ...]`. The last in-bounds start is
    `end_idx - (sequence_length - 1) * internal_stride - 1`.
    """

    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        sequence_length: int = 5,
        internal_stride: int = 6,
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.sequence_length = sequence_length
        self.internal_stride = internal_stride
        self.window_span = (sequence_length - 1) * internal_stride + 1
        self.n_windows = max(0, end_idx - start_idx - self.window_span + 1)

    def get_window_indices(self, window_idx: int) -> List[int]:
        if window_idx < 0 or window_idx >= self.n_windows:
            raise IndexError(
                f"Window index {window_idx} out of range [0, {self.n_windows})"
            )
        start = self.start_idx + window_idx
        return [start + i * self.internal_stride for i in range(self.sequence_length)]

    def __len__(self):
        return self.n_windows

    def __iter__(self):
        """In-order iteration; shuffling is the DataLoader's job."""
        for w in range(self.n_windows):
            yield self.get_window_indices(w)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test() -> int:
    failed = []

    n_hours = 365 * 24 * 4
    times = np.arange(n_hours, dtype="timedelta64[h]") + np.datetime64("2007-01-02T01:00:00")

    s, e = year_to_index_range(2007, 2007, times)
    expect_end = int(np.searchsorted(times, np.datetime64("2008-01-01T00:00:00"), side="left"))
    if s != 0 or e != expect_end:
        failed.append(f"year_to_index_range(2007,2007) returned ({s},{e}); "
                      f"expected (0,{expect_end})")
    print(f"  year_to_index_range(2007,2007) = ({s},{e})  span {e-s} hours")

    samp = SlidingWindowSampler(start_idx=0, end_idx=100,
                                sequence_length=5, internal_stride=6)
    if samp.window_span != 25:
        failed.append(f"window_span {samp.window_span} != 25")
    if len(samp) != 76:
        failed.append(f"n_windows {len(samp)} != 76")
    w0 = samp.get_window_indices(0)
    if w0 != [0, 6, 12, 18, 24]:
        failed.append(f"window 0 = {w0}")
    w_last = samp.get_window_indices(len(samp) - 1)
    if w_last != [75, 81, 87, 93, 99]:
        failed.append(f"last window = {w_last}")
    print(f"  SlidingWindowSampler(0,100,5,6): span={samp.window_span}, n={len(samp)}, "
          f"w0={w0}, wlast={w_last}")

    try:
        samp.get_window_indices(len(samp))
        failed.append("out-of-range index did not raise")
    except IndexError:
        pass

    empty = SlidingWindowSampler(start_idx=0, end_idx=10,
                                 sequence_length=5, internal_stride=6)
    if len(empty) != 0:
        failed.append(f"empty sampler has n_windows={len(empty)}")

    if failed:
        print("SAMPLER SMOKE TEST FAIL:", failed)
        return 1
    print("SAMPLER SMOKE TEST PASS")
    return 0




# ======================================================================
# ---- merged from data/pixel_dataset.py ----
# ======================================================================
#!/usr/bin/env python
"""Pixel-space dataset for DC-AE training — config-driven.

Returns one timestep per sample: dynamic channels (per-level 3D + 2D) +
1 static (mask_rho), cropped at sample time from the full-grid store.
Grid crop, variable layout, kept-level count, paths and the train-split
year range all come from a RegionConfig (MAB_CONFIG / CCS_CONFIG);
`config` defaults to MAB_CONFIG so existing MAB callers are unchanged.

Per-channel normalization is applied at dataloader time. Static
(mask_rho) passes through unnormalized.
The DC-AE loss is wet-cell-only MSE; mask_rho is returned as a third
tensor for the loss mask.

This is the PER-FRAME (DC-AE) path — no temporal windowing; INTERNAL_STRIDE
is a denoiser-stage concern and is not used here.

Smoke test (MAB default, or --region ccs; needs the preprocessed store and
norm stats under $CLOSEDLOOP_DATA_ROOT):
  python -m closedloop_sampling.data.pixel_dataset [--region mab|ccs]
"""

import os
import sys
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from closedloop_sampling.regions import MAB_CONFIG
from closedloop_sampling.forecast.normalization import Normalizer, load_pixel_normalizer

# Hint xarray to avoid HDF5 file-lock attempts on network filesystems.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# mask_rho is the ROMS wet/dry mask — the same variable name in every ROMS
# store (MAB and CCS), so it is a convention, not a region constant.
STATIC_MASK_VAR = "mask_rho"

# Backward-compat module constants: the MAB crop offsets, kept as public
# names for external importers that crop MAB fields directly. Derived from
# MAB_CONFIG so they are not re-hardcoded; for MAB these are 1/1/1/1
# (crop (1,105)/(1,241) on the native 106×242 grid).
CROP_TOP = MAB_CONFIG.grid.crop_eta_slice[0]
CROP_BOTTOM = MAB_CONFIG.grid.native_eta_rho - MAB_CONFIG.grid.crop_eta_slice[1]
CROP_LEFT = MAB_CONFIG.grid.crop_xi_slice[0]
CROP_RIGHT = MAB_CONFIG.grid.native_xi_rho - MAB_CONFIG.grid.crop_xi_slice[1]


class RegionalPixelDataset(Dataset):
    """One timestep per sample (region set by `config`; default MAB).

    Returns three tensors per `__getitem__`:
      dynamic   : (n_dyn, H, W) — normalized
      static    : (1, H, W)     — mask_rho, NOT normalized (binary 0/1)
      mask_loss : (1, H, W)     — same as static for now; separate handle
                  so a future loss-mask change touches only the third return.

    Channel order in `dynamic` (must stay in lockstep with the pixel
    `Normalizer`'s var_order): each 3D var per-level, then each 2D var —
    e.g. u[0..K-1], v[..], temp[..], salt[..], zeta, ubar, vbar.
    MAB/CCS both: 4×20 + 3 = 83 dynamic channels.
    """

    def __init__(
        self,
        zarr_path: str,
        normalizer: Normalizer,
        time_idx_range: Tuple[int, int],
        config=MAB_CONFIG,
    ):
        self.zarr_path = zarr_path
        self.normalizer = normalizer
        self.start_idx, self.end_idx = time_idx_range
        if self.end_idx <= self.start_idx:
            raise ValueError(
                f"Empty time range [{self.start_idx}, {self.end_idx})"
            )
        # Region-driven channel layout + crop (the MAB default reproduces the
        # prior hardcoded constants exactly: crop (1,105)/(1,241) == [1:-1,1:-1]).
        # cfg.metric_crop is THE crop — the single source of truth.
        self.dyn3d = list(config.channels.dyn3d_vars)
        self.dyn2d = list(config.channels.dyn2d_vars)
        self.crop_eta, self.crop_xi = config.metric_crop
        self.cropped_h = config.grid.cropped_h
        self.cropped_w = config.grid.cropped_w
        # Per-worker Zarr handle is lazily opened in `_get_ds` so worker
        # subprocesses don't share an xarray Dataset across forks.
        self._ds = None
        # mask_rho is time-invariant: load once at __init__ (cheap; small).
        ds0 = xr.open_zarr(zarr_path, consolidated=False)
        mask_full = ds0[STATIC_MASK_VAR].values  # (eta, xi)
        self.mask_cropped = self._crop_spatial(mask_full).astype(np.float32)
        ds0.close()

    def _get_ds(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_zarr(self.zarr_path, consolidated=False)
        return self._ds

    def __len__(self) -> int:
        return self.end_idx - self.start_idx

    def _crop_spatial(self, x: np.ndarray) -> np.ndarray:
        # Crop the trailing (eta, xi) dims; works for 2D (eta,xi) and 3D
        # (level,eta,xi) arrays alike.
        return x[..., self.crop_eta, self.crop_xi]

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_idx = self.start_idx + idx
        ds = self._get_ds()

        chunks = []
        for var in self.dyn3d:
            arr = ds[var].isel(ocean_time=t_idx).values   # (K, eta, xi)
            chunks.append(self._crop_spatial(arr))         # (K, H, W)
        for var in self.dyn2d:
            arr = ds[var].isel(ocean_time=t_idx).values   # (eta, xi)
            cropped = self._crop_spatial(arr)              # (H, W)
            chunks.append(cropped[np.newaxis, ...])        # (1, H, W)

        dynamic_np = np.concatenate(chunks, axis=0)        # (n_dyn, H, W)
        dynamic = torch.from_numpy(dynamic_np.astype(np.float32))
        # Normalize: prepend batch dim, normalize, drop the batch dim.
        dynamic = self.normalizer.normalize_dynamic_tensor(dynamic.unsqueeze(0)).squeeze(0)

        static = torch.from_numpy(self.mask_cropped).unsqueeze(0)  # (1, H, W)
        mask_loss = static.clone()
        return dynamic, static, mask_loss


# Backward-compat alias: the historical dataset-prefixed name still resolves.
DoppioPixelDataset = RegionalPixelDataset


def make_pixel_dataloader(
    zarr_path: str,
    stats_json_path: str,
    year_range: Tuple[int, int],
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    config=MAB_CONFIG,
) -> DataLoader:
    """Build a pixel-space (per-frame) DataLoader.

    Args:
      year_range: (year_start, year_end) inclusive — the train/val/test split
                  selected by a year predicate on decoded ocean_time.
      shuffle:    True for train; False for val/test (deterministic order).
      config:     RegionConfig (default MAB_CONFIG) — drives crop + channel
                  layout. Pass get_config("ccs") (with CCS zarr/stats/
                  year_range) for CCS.
    """
    normalizer = load_pixel_normalizer(stats_json_path)

    ds0 = xr.open_zarr(zarr_path, consolidated=False)
    time_array = ds0.ocean_time.values
    start_idx, end_idx = year_to_index_range(year_range[0], year_range[1], time_array)
    ds0.close()

    dataset = RegionalPixelDataset(
        zarr_path=zarr_path,
        normalizer=normalizer,
        time_idx_range=(start_idx, end_idx),
        config=config,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,     # match spec; cost is negligible.
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test(region: str = "mab") -> int:
    from closedloop_sampling.regions import get_config
    from closedloop_sampling.paths import resolve
    cfg = get_config(region)
    yr = cfg.timesplit.train_year_range[0]
    failed = []

    dl = make_pixel_dataloader(
        zarr_path=resolve(cfg.paths.preprocessed_zarr),
        stats_json_path=resolve(cfg.paths.pixel_norm_stats),
        year_range=(yr, yr),          # first train year only, to keep it fast
        batch_size=2,
        num_workers=0,
        shuffle=False,
        config=cfg,
    )
    n_dyn = cfg.channels.n_dynamic_channels
    H, W = cfg.grid.cropped_h, cfg.grid.cropped_w
    print(f"  region={region}  dataset len = {len(dl.dataset)}  ({len(dl)} batches @ batch=2)")

    dynamic, static, mask_loss = next(iter(dl))
    print(f"  dynamic shape: {tuple(dynamic.shape)}  static: {tuple(static.shape)}")
    if tuple(dynamic.shape) != (2, n_dyn, H, W):
        failed.append(f"dynamic shape {tuple(dynamic.shape)} != (2,{n_dyn},{H},{W})")
    if tuple(static.shape) != (2, 1, H, W):
        failed.append(f"static shape {tuple(static.shape)} != (2,1,{H},{W})")
    if not torch.isfinite(dynamic).all():
        failed.append("dynamic has non-finite values")
    if not torch.equal(static, mask_loss):
        failed.append("mask_loss differs from static")

    uniq = torch.unique(static)
    ok_bin = torch.allclose(uniq.float(), torch.tensor([0.0, 1.0])) or torch.allclose(
        uniq.float(), torch.tensor([1.0]))
    if not ok_bin:
        failed.append(f"static not binary; unique {uniq.tolist()}")

    chan_mean = dynamic.mean(dim=(0, 2, 3))
    chan_std = dynamic.std(dim=(0, 2, 3))
    print(f"  per-channel mean median {chan_mean.abs().median():.3f}  "
          f"std median {chan_std.median():.3f}  (normalized: expect ~0 / ~1)")
    if chan_mean.abs().median().item() > 5:
        failed.append(f"median |mean| {chan_mean.abs().median():.3f} too large")
    if not (0.01 < chan_std.median().item() < 10):
        failed.append(f"median std {chan_std.median():.3f} outside [0.01,10]")

    if failed:
        print("\nPIXEL DATALOADER SMOKE TEST FAIL:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nPIXEL DATALOADER SMOKE TEST PASS")
    return 0




# ======================================================================
# ---- merged from data/latent_dataset.py ----
# ======================================================================
#!/usr/bin/env python
"""Latent-space dataset for denoiser training.

`LatentDataset` returns (cond, pred, timestamps) windows from a latent Zarr:
1 conditioning + 4 prediction frames at 6-hour physical spacing.

Output axis order — IMPORTANT — this dataset returns:
  cond   : (84, 13, 30)
  pred   : (4, 84, 13, 30)        # T-first
But the denoiser models expect:
  hidden_states         : (B, 84, T_pred=4, 13, 30)   # C-first, T after C
  conditioning_tensors  : (B, 84, T_cond=1, 13, 30)
So the training script must `permute(0, 2, 1, 3, 4)` and `unsqueeze(2)`
to convert. Documented here.

The latent Zarr (config.paths.latents_zarr) and latent stats JSON
(config.paths.latent_norm_stats) are produced by the latent-encoding
step. The smoke test below constructs a synthetic latent Zarr (and
synthetic stats JSON) in a temporary directory to exercise the
dataloader without depending on real encoded latents; cleaned up after
the test.

Smoke test: `python -m closedloop_sampling.data.latent_dataset`
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, DataLoader

from closedloop_sampling.forecast.normalization import Normalizer, load_latent_normalizer
from closedloop_sampling.regions import MAB_CONFIG


os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


class LatentDataset(Dataset):
    """Returns (cond, pred, timestamps) from a latent Zarr."""

    # LEGACY MAB-reference constants — NOT used by the logic any more (the
    # instance reads sequence_length / internal_stride / latent dims from the
    # RegionConfig in __init__). Do NOT reintroduce them into the sampler:
    # INTERNAL_STRIDE is in INDEX steps, so MAB's 6 on CCS's 6-hourly store
    # would silently produce 36 h windows. Kept only for backward reference.
    SEQUENCE_LENGTH = 5     # 1 cond + 4 pred          (MAB default)
    INTERNAL_STRIDE = 6     # INDEX steps (hourly store) (MAB default)
    LATENT_CHANNELS = 84
    LATENT_H = 13           # MAB latent (CCS: 22)
    LATENT_W = 30           # MAB latent (CCS: 23)
    LATENT_VAR = "latents"  # variable name in the Zarr

    def __init__(
        self,
        latent_zarr_path: str,
        normalizer: Normalizer,
        time_idx_range: Tuple[int, int],
        config=MAB_CONFIG,
    ):
        self.latent_zarr_path = latent_zarr_path
        self.normalizer = normalizer
        if not Path(latent_zarr_path).exists():
            raise FileNotFoundError(
                f"Latent Zarr not found at {latent_zarr_path}. "
                f"This file is produced by the latent-encoding step."
            )
        # Region-driven window + latent geometry. CRITICAL: internal_stride is in
        # INDEX steps, not hours — the MAB store is hourly (stride 6 = 6 h) but the
        # CCS store is 6-hourly (stride 1 = 6 h). Inheriting MAB's 6 on CCS would
        # silently make windows 36 h apart. Both regions must yield 6 h PHYSICAL
        # spacing.
        self.sequence_length = config.timesplit.sequence_length
        self.internal_stride = config.timesplit.internal_stride
        self.latent_channels = config.latent.latent_channels
        self.latent_h = config.latent.latent_h
        self.latent_w = config.latent.latent_w
        self._ds = None  # opened lazily per-worker
        start_idx, end_idx = time_idx_range
        self.sampler = SlidingWindowSampler(
            start_idx=start_idx,
            end_idx=end_idx,
            sequence_length=self.sequence_length,
            internal_stride=self.internal_stride,
        )

    def _get_ds(self) -> xr.Dataset:
        if self._ds is None:
            self._ds = xr.open_zarr(self.latent_zarr_path, consolidated=False)
        return self._ds

    def __len__(self) -> int:
        return len(self.sampler)

    def _load_latent_sequence(self, idx: int) -> torch.Tensor:
        """Load a 5-timestep latent sequence and normalize per channel.

        Returns (5, 84, 13, 30) float32.
        """
        window = self.sampler.get_window_indices(idx)
        ds = self._get_ds()
        # Vectorized read of the 5 requested timesteps.
        latents_np = ds[self.LATENT_VAR].isel(ocean_time=window).values  # (T, C, H, W)
        expected = (self.sequence_length, self.latent_channels,
                    self.latent_h, self.latent_w)
        if latents_np.shape != expected:
            raise ValueError(
                f"Unexpected latent shape {latents_np.shape}; expected "
                f"{expected} — verify the encoded Zarr schema."
            )
        tensor = torch.from_numpy(latents_np.astype(np.float32))
        # Normalize: the 5 timesteps act as batch dim for the per-channel
        # normalizer (which expects (B, C, H, W)).
        return self.normalizer.normalize_dynamic_tensor(tensor)

    def __getitem__(self, idx: int):
        sequence = self._load_latent_sequence(idx)         # (5, 84, 13, 30)
        cond = sequence[0]                                  # (84, 13, 30)
        pred = sequence[1:]                                 # (4, 84, 13, 30)
        timestamps = torch.arange(self.sequence_length, dtype=torch.long)
        return cond, pred, timestamps


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_latent_dataloader(
    latent_zarr_path: str,
    latent_stats_path: str,
    year_range: Tuple[int, int],
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    config=MAB_CONFIG,
) -> DataLoader:
    """Build a latent-space DataLoader.

    Args:
      latent_zarr_path / latent_stats_path: produced by the encode step.
      config: RegionConfig (default MAB_CONFIG) — drives internal_stride
        (INDEX steps: MAB 6 on hourly, CCS 1 on 6-hourly — both = 6 h
        physical), sequence_length and latent dims (13x30 vs 22x23).
    """
    normalizer = load_latent_normalizer(latent_stats_path)

    ds0 = xr.open_zarr(latent_zarr_path, consolidated=False)
    time_array = ds0.ocean_time.values
    start_idx, end_idx = year_to_index_range(year_range[0], year_range[1], time_array)
    ds0.close()

    dataset = LatentDataset(
        latent_zarr_path=latent_zarr_path,
        normalizer=normalizer,
        time_idx_range=(start_idx, end_idx),
        config=config,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )


# ---------------------------------------------------------------------------
# Smoke test (synthetic data — real latents come from the encoding step)
# ---------------------------------------------------------------------------

def _make_synthetic_latents(
    out_zarr: str, out_stats_json: str, n_timesteps: int = 200,
) -> Tuple[int, int]:
    """Write a synthetic latent Zarr + stats JSON for the smoke test.

    Schema mirrors the encoding step's expected output:
      Zarr variable `latents` of shape (T, 84, 13, 30); ocean_time coord
      hourly from 2007-01-02T01:00:00.
      Stats JSON: {"mean": [..84..], "std": [..84..]}.
    """
    rng = np.random.default_rng(0)
    latents = rng.standard_normal(
        (n_timesteps, 84, 13, 30), dtype=np.float32
    ) * 0.5  # σ ~ 0.5 to match the EDM σ_data anchor
    times = (
        np.arange(n_timesteps, dtype="timedelta64[h]")
        + np.datetime64("2007-01-02T01:00:00")
    )
    ds = xr.Dataset(
        data_vars={
            "latents": (("ocean_time", "C", "H", "W"), latents),
        },
        coords={"ocean_time": times},
    )
    # Clean any prior store.
    p = Path(out_zarr)
    if p.exists():
        shutil.rmtree(p)
    ds.to_zarr(out_zarr, mode="w", consolidated=False)

    # Stats JSON: per-channel mean/std over the synthetic batch.
    mean84 = latents.reshape(-1, 84, *latents.shape[2:]).mean(axis=(0, 2, 3)).tolist()
    std84 = latents.reshape(-1, 84, *latents.shape[2:]).std(axis=(0, 2, 3)).tolist()
    with open(out_stats_json, "w") as f:
        json.dump({"mean": mean84, "std": std84}, f)

    return 0, n_timesteps


def _smoke_test() -> int:
    failed = []
    tmpdir = tempfile.mkdtemp(prefix="closedloop_sampling_latent_")
    zarr_path = os.path.join(tmpdir, "synthetic_latents.zarr")
    stats_path = os.path.join(tmpdir, "synthetic_latent_stats.json")
    try:
        N_T = 200
        _make_synthetic_latents(zarr_path, stats_path, n_timesteps=N_T)
        print(f"  synthetic Zarr written at {zarr_path} ({N_T} timesteps)")
        # year_range for the synthetic times (2007-01-02T01:00 + 0..199h)
        # All lie in 2007 → year_range=(2007, 2007).
        year_range = (2007, 2007)

        dl = make_latent_dataloader(
            latent_zarr_path=zarr_path,
            latent_stats_path=stats_path,
            year_range=year_range,
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )
        print(f"  dataset len = {len(dl.dataset)}  "
              f"(expected {N_T - 25 + 1 if N_T >= 25 else 0} = N − window_span + 1)")
        cond, pred, ts = next(iter(dl))
        print(f"  cond {tuple(cond.shape)}  "
              f"pred {tuple(pred.shape)}  ts {tuple(ts.shape)}")
        if tuple(cond.shape) != (2, 84, 13, 30):
            failed.append(f"cond shape {tuple(cond.shape)} != (2,84,13,30)")
        if tuple(pred.shape) != (2, 4, 84, 13, 30):
            failed.append(f"pred shape {tuple(pred.shape)} != (2,4,84,13,30)")
        if tuple(ts.shape) != (2, 5):
            failed.append(f"ts shape {tuple(ts.shape)} != (2,5)")
        if not torch.isfinite(cond).all() or not torch.isfinite(pred).all():
            failed.append("non-finite values in cond or pred")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if failed:
        print("\nLATENT DATALOADER SMOKE TEST FAIL:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\nLATENT DATALOADER SMOKE TEST PASS")
    return 0


