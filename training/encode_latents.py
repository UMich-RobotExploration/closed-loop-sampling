#!/usr/bin/env python
"""Encode the full preprocessed pixel Zarr into latent space using
the trained DC-AE encoder (best checkpoint of the production DC-AE run).

Produces (filenames from the RegionConfig; MAB shown):
  - `mab_latents.zarr`           : (T, 84, 13, 30) latents, chunked
                                    72 × 84 × 13 × 30, Blosc-LZ4.
  - `mab_latent_norm_stats.json` : per-channel mean/std over the
                                    2007–2020 train split.

One-shot pass: no training, no optimizer, no checkpointing inside.
Streaming write so we never materialize the 22 GB latent tensor in RAM.

Run (real):
  python -m closedloop_sampling.training.encode_latents

Smoke (no real checkpoint yet — uses freshly-initialised RegionalDCAE):
  python -m closedloop_sampling.training.encode_latents \\
      --smoke-test --skip-checkpoint
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
import zarr
import zarr.codecs
from tqdm import tqdm

from closedloop_sampling import paths
from closedloop_sampling.forecast.models import RegionalDCAE
from dataset import make_pixel_dataloader
from dataset import year_to_index_range
from closedloop_sampling.regions import MAB_CONFIG, CCS_CONFIG

REGION_CONFIGS = {"mab": MAB_CONFIG, "ccs": CCS_CONFIG}


def _resolve_path(value: str, root: str = "data") -> str:
    """Resolve a config path. RegionConfig paths are RELATIVE (data paths
    against $CLOSEDLOOP_DATA_ROOT, run dirs against
    $CLOSEDLOOP_RUNS); absolute values (CLI overrides) pass
    through untouched."""
    if Path(value).is_absolute():
        return str(value)
    return paths.resolve(value, root=root)


# ============================================================
# Defaults
# ============================================================
_MAB = MAB_CONFIG

DEFAULTS = {
    # Paths — RELATIVE to the env roots (resolved at use via _resolve_path);
    # taken from the MAB RegionConfig so the defaults equal `--region mab`.
    "zarr_path": _MAB.paths.preprocessed_zarr,
    "stats_json_path": _MAB.paths.pixel_norm_stats,
    # checkpoint-best of the DC-AE run, under $CLOSEDLOOP_RUNS.
    "checkpoint_dir": str(Path(_MAB.paths.dcae_run_dir) / "checkpoint-best"),
    "output_latent_zarr": _MAB.paths.latents_zarr,
    "output_stats_json": _MAB.paths.latent_norm_stats,
    "batch_size": 8,
    # num_workers=4 with spawn start method (set in main()) — sidesteps
    # a kernel 4.18.0 fork-hang seen on the training cluster. persistent_workers
    # in the dataloader factory keeps spawn-overhead at first-epoch only.
    "num_workers": 4,
    "precision": "bf16",
    "train_year_range": (2007, 2020),
    "full_year_range": (2007, 2024),
}

# MAB latent dims — fallback defaults only; the region overlay in main()
# supplies the actual dims from the RegionConfig.
LATENT_CHANNELS = 84
LATENT_H = 13
LATENT_W = 30


# ============================================================
# Checkpoint loading
# ============================================================
def load_dcae_from_checkpoint(
    checkpoint_dir: Path,
    use_ema: bool = True,
    device: torch.device = None,
) -> RegionalDCAE:
    """Build RegionalDCAE and copy weights from the checkpoint.

    If `use_ema=True` (the LaDCast convention), reads `ema.pt`
    (which is `EMAModel.state_dict()` containing `shadow_params`
    — the averaged tensors) and copies each shadow tensor into the
    matching model parameter in declaration order.
    """
    print("[encode_latents] Building RegionalDCAE...")
    model = RegionalDCAE()

    if use_ema:
        ckpt_path = checkpoint_dir / "ema.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"EMA checkpoint not found: {ckpt_path}")
        print(f"[encode_latents] Loading EMA weights from {ckpt_path}")
        ema_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "shadow_params" not in ema_state:
            raise ValueError(
                "ema.pt has no 'shadow_params' key — file format unexpected. "
                f"Top-level keys: {list(ema_state)[:8]}"
            )
        shadow = ema_state["shadow_params"]
        model_params = list(model.parameters())
        if len(shadow) != len(model_params):
            raise ValueError(
                f"Param count mismatch loading EMA: ema has {len(shadow)} "
                f"shadow tensors, model has {len(model_params)} params."
            )
        for mp, sp in zip(model_params, shadow):
            mp.data.copy_(sp.data)
    else:
        ckpt_path = checkpoint_dir / "model.pt"
        print(f"[encode_latents] Loading MAIN weights from {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state)

    meta_path = checkpoint_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(
            f"[encode_latents] Checkpoint epoch={meta.get('epoch')}, "
            f"val_loss={meta.get('val_loss')}"
        )

    if device is not None:
        model.to(device)
    model.eval()
    return model


# ============================================================
# Output Zarr layout (zarr 3.x)
# ============================================================
def create_output_zarr(
    output_path: Path, n_timesteps: int, time_array: np.ndarray,
    latent_channels: int = LATENT_CHANNELS,
    latent_h: int = LATENT_H, latent_w: int = LATENT_W,
) -> "zarr.Array":
    """Create the output Zarr group with `latents` + `ocean_time`.

    Both arrays carry `dimension_names` so `xarray.open_zarr` reads
    them as named-dimension variables. Latent dims default to MAB
    (84,13,30); CCS passes (84,22,23) from its RegionConfig. Blosc-LZ4 clevel 5.
    """
    if output_path.exists():
        raise FileExistsError(
            f"Output zarr already exists at {output_path}. "
            f"Remove or use a different path before re-running."
        )
    root = zarr.open(str(output_path), mode="w")
    latents_arr = root.create_array(
        "latents",
        shape=(n_timesteps, latent_channels, latent_h, latent_w),
        chunks=(72, latent_channels, latent_h, latent_w),
        dtype=np.float32,
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
        dimension_names=["ocean_time", "C", "H", "W"],
    )
    time_arr = root.create_array(
        "ocean_time",
        shape=time_array.shape,
        chunks=time_array.shape,
        dtype=time_array.dtype,
        dimension_names=["ocean_time"],
    )
    time_arr[:] = time_array
    return latents_arr


# ============================================================
# Encoding pass
# ============================================================
@torch.no_grad()
def encode_all_timesteps(
    model: RegionalDCAE, device: torch.device, config: dict,
) -> Tuple[Path, np.ndarray]:
    """Encode every timestep in the input Zarr; write to the output."""
    ds_in = xr.open_zarr(_resolve_path(config["zarr_path"]), consolidated=False)
    time_array = ds_in.ocean_time.values
    n_timesteps = len(time_array)
    print(f"[encode_latents] Encoding {n_timesteps:,} timesteps")

    output_path = Path(_resolve_path(config["output_latent_zarr"]))
    latents_arr = create_output_zarr(
        output_path, n_timesteps, time_array,
        config.get("latent_channels", LATENT_CHANNELS),
        config.get("latent_h", LATENT_H), config.get("latent_w", LATENT_W))

    loader = make_pixel_dataloader(
        zarr_path=_resolve_path(config["zarr_path"]),
        stats_json_path=_resolve_path(config["stats_json_path"]),
        year_range=tuple(config["full_year_range"]),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        config=REGION_CONFIGS[config.get("region", "mab")],   # region crop/channels
    )

    autocast_dtype = (
        torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    )
    idx = 0
    for dynamic, static, _ in tqdm(loader, total=len(loader), desc="Encoding"):
        dynamic = dynamic.to(device, non_blocking=True)
        static = static.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
            latent = model.encode(dynamic, static)
        block = latent.detach().float().cpu().numpy()
        b = block.shape[0]
        latents_arr[idx:idx + b] = block
        idx += b

    if idx != n_timesteps:
        raise RuntimeError(
            f"Encoded {idx} timesteps but input has {n_timesteps}. "
            f"DataLoader length mismatch."
        )
    print(f"[encode_latents] Wrote {idx:,} latents to {output_path}")
    return output_path, time_array


# ============================================================
# Streaming per-channel stats over the train split
# ============================================================
def compute_latent_stats(
    latent_zarr_path: Path,
    time_array: np.ndarray,
    train_year_range: Tuple[int, int],
    output_json_path: Path,
    latent_channels: int = LATENT_CHANNELS,
    latent_h: int = LATENT_H, latent_w: int = LATENT_W,
):
    """Streaming Welford-style first/second moments per channel.

    All sums in float64; ddof=0 (population variance, matching the pixel
    norm-stats JSON `_meta.std_ddof = 0`). Latent dims default to
    MAB (84,13,30); CCS passes (84,22,23).
    """
    print(
        f"[encode_latents] Computing latent stats over train split "
        f"({train_year_range[0]}–{train_year_range[1]})"
    )
    root = zarr.open(str(latent_zarr_path), mode="r")
    latents = root["latents"]

    train_start, train_end = year_to_index_range(
        train_year_range[0], train_year_range[1], time_array
    )
    print(
        f"[encode_latents] Train split: indices [{train_start}, {train_end})"
    )

    sum_x = np.zeros(latent_channels, dtype=np.float64)
    sum_x2 = np.zeros(latent_channels, dtype=np.float64)
    n_per_channel = 0

    chunk = 72
    for cstart in tqdm(range(train_start, train_end, chunk), desc="Stats"):
        cend = min(cstart + chunk, train_end)
        block = latents[cstart:cend].astype(np.float64)   # (T, 84, 13, 30)
        sum_x += block.sum(axis=(0, 2, 3))
        sum_x2 += (block ** 2).sum(axis=(0, 2, 3))
        n_per_channel += block.shape[0] * latent_h * latent_w

    means = sum_x / n_per_channel
    variances = np.maximum(sum_x2 / n_per_channel - means ** 2, 0)
    stds = np.sqrt(variances)

    stats = {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "_meta": {
            "n_channels": int(latent_channels),
            "n_samples_per_channel": int(n_per_channel),
            "train_year_range": list(train_year_range),
            "computed_with": "streaming_welford_float64_ddof0",
        },
    }
    with open(output_json_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[encode_latents] Wrote stats to {output_json_path}")
    print(
        f"[encode_latents] Mean range: [{means.min():.4f}, {means.max():.4f}], "
        f"median {float(np.median(means)):.4f}"
    )
    print(
        f"[encode_latents] Std  range: [{stds.min():.4f}, {stds.max():.4f}], "
        f"median {float(np.median(stds)):.4f}"
    )
    return means, stds


# ============================================================
# Smoke test (no real checkpoint available yet)
# ============================================================
def _smoke_pass(model: RegionalDCAE, device: torch.device, config: dict, n_timesteps: int = 24) -> int:
    """Encode `n_timesteps` from year 2007 into a synthetic output
    Zarr; verify shapes/finiteness; also run stats over the synthetic
    pass to exercise that code path. Cleans up the synthetic zarr."""
    smoke_zarr = Path(_resolve_path(config["output_latent_zarr"]))
    smoke_json = Path(_resolve_path(config["output_stats_json"]))
    if smoke_zarr.exists():
        shutil.rmtree(smoke_zarr)

    loader = make_pixel_dataloader(
        zarr_path=_resolve_path(config["zarr_path"]),
        stats_json_path=_resolve_path(config["stats_json_path"]),
        year_range=(2007, 2007),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    print(f"[smoke] dataset len = {len(loader.dataset)}; "
          f"encoding first {n_timesteps} timesteps")

    # Time array from year 2007 only.
    ds_in = xr.open_zarr(_resolve_path(config["zarr_path"]), consolidated=False)
    full_time = ds_in.ocean_time.values
    yr_start, yr_end = year_to_index_range(2007, 2007, full_time)
    smoke_time = full_time[yr_start:yr_start + n_timesteps]

    latents_arr = create_output_zarr(smoke_zarr, n_timesteps, smoke_time)
    autocast_dtype = torch.bfloat16
    idx = 0
    with torch.no_grad():
        for dynamic, static, _ in loader:
            dynamic = dynamic.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                latent = model.encode(dynamic, static)
            block = latent.detach().float().cpu().numpy()
            b = block.shape[0]
            take = min(b, n_timesteps - idx)
            latents_arr[idx:idx + take] = block[:take]
            idx += take
            if idx >= n_timesteps:
                break
    print(f"[smoke] wrote {idx} timesteps to {smoke_zarr}")

    # Verify via xarray round-trip
    ds_out = xr.open_zarr(str(smoke_zarr), consolidated=False)
    lat = ds_out["latents"].values
    assert lat.shape == (n_timesteps, LATENT_CHANNELS, LATENT_H, LATENT_W), (
        f"shape mismatch: {lat.shape}"
    )
    assert np.isfinite(lat).all(), "non-finite latents"
    print(
        f"[smoke] xr round-trip OK: latents shape {tuple(ds_out['latents'].shape)}, "
        f"ocean_time {len(ds_out['ocean_time'])} steps"
    )
    print(
        f"[smoke] latent value range: [{lat.min():.3f}, {lat.max():.3f}]  "
        f"std {lat.std():.3f}  mean {lat.mean():.3f}"
    )

    # Exercise stats path on the synthetic encoded zarr (n_per_channel
    # = 24 × 13 × 30 = 9,360 — small but real).
    compute_latent_stats(
        latent_zarr_path=smoke_zarr,
        time_array=smoke_time,
        train_year_range=(2007, 2007),
        output_json_path=smoke_json,
    )

    # Clean up smoke artifacts.
    shutil.rmtree(smoke_zarr)
    smoke_json.unlink(missing_ok=True)
    print("[smoke] artifacts cleaned up")
    return 0


# ============================================================
# Main
# ============================================================
def main():
    # MUST be first: switch DataLoader workers to spawn IPC, sidestepping
    # the cluster's 4.18.0 fork-hang. See train_dcae.py for context.
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)

    p = argparse.ArgumentParser()
    p.add_argument("--region", choices=list(REGION_CONFIGS), default="mab",
                   help="RegionConfig (default mab). Drives "
                        "store/checkpoint/output paths, latent dims, train split.")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument(
        "--use-main", action="store_true",
        help="Use main weights instead of EMA",
    )
    p.add_argument(
        "--dump-config", action="store_true",
        help="Print the fully-resolved config as JSON and exit (no side effects). "
             "Useful for diffing the per-region resolved configs before submit.",
    )
    p.add_argument(
        "--skip-checkpoint", action="store_true",
        help="Skip loading: use freshly-initialized RegionalDCAE. "
             "Only meaningful for the synthetic smoke test before "
             "the production DC-AE run has produced a real checkpoint.",
    )
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument(
        "--skip-stats", action="store_true",
        help="Skip the latent normalization stats computation",
    )
    p.add_argument(
        "--smoke-test", action="store_true",
        help="Synthetic smoke pass: encode 24 timesteps of 2007 to "
             "a smoketest zarr (cleaned up at the end).",
    )
    args = p.parse_args()

    config = dict(DEFAULTS)
    # Region overlay: for --region mab these equal DEFAULTS exactly (DEFAULTS
    # are built from the MAB config), so default behavior is unchanged;
    # --region ccs swaps in the CCS store/checkpoint/output/latent-dims/split.
    # CLI flags below still override.
    rc = REGION_CONFIGS[args.region]
    config["region"] = args.region
    config["zarr_path"] = rc.paths.preprocessed_zarr
    config["stats_json_path"] = rc.paths.pixel_norm_stats
    config["output_latent_zarr"] = rc.paths.latents_zarr
    config["output_stats_json"] = rc.paths.latent_norm_stats
    config["train_year_range"] = tuple(rc.timesplit.train_year_range)
    # encode ALL frames (train+val+test) — full store year span, index-aligned 1:1
    config["full_year_range"] = (int(rc.timesplit.start_time[:4]),
                                 int(rc.timesplit.end_time[:4]))
    config["latent_channels"] = rc.latent.latent_channels
    config["latent_h"] = rc.latent.latent_h
    config["latent_w"] = rc.latent.latent_w
    # checkpoint-best of the region's DC-AE run (runs-root-relative, from the
    # RegionConfig; resolved against $CLOSEDLOOP_RUNS at use)
    config["checkpoint_dir"] = str(Path(rc.paths.dcae_run_dir) / "checkpoint-best")

    if args.checkpoint:
        config["checkpoint_dir"] = args.checkpoint
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.output_dir:
        # Region-generic: keep the RegionConfig's output filenames, redirect the dir.
        d = Path(args.output_dir)
        config["output_latent_zarr"] = str(d / Path(config["output_latent_zarr"]).name)
        config["output_stats_json"] = str(d / Path(config["output_stats_json"]).name)

    if args.smoke_test:
        # Redirect outputs to *_smoketest paths so we never collide with
        # the real artifacts. Names derive from the region's configured outputs.
        _z = Path(config["output_latent_zarr"])
        config["output_latent_zarr"] = str(_z.parent / f"{_z.stem}_smoketest.zarr")
        _s = Path(config["output_stats_json"])
        config["output_stats_json"] = str(_s.parent / f"{_s.stem}_smoketest.json")

    # Dump the FULLY-RESOLVED config and exit, before any side effect (incl. the
    # CUDA check — config diffs run on a login node). This is what region-config
    # diffs compare across regions; it must see exactly what the encode pass
    # would see.
    if args.dump_config:
        print(json.dumps(config, indent=2, default=str))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("encode_latents requires CUDA.")
    device = torch.device("cuda")
    print(f"[encode_latents] device: {torch.cuda.get_device_name(0)}")

    if args.skip_checkpoint:
        if not args.smoke_test:
            raise RuntimeError(
                "--skip-checkpoint is only valid with --smoke-test; the real "
                "encoding pass requires a trained checkpoint."
            )
        print("[encode_latents] --skip-checkpoint: using freshly-init RegionalDCAE")
        model = RegionalDCAE().to(device).eval()
    else:
        model = load_dcae_from_checkpoint(
            Path(_resolve_path(config["checkpoint_dir"], root="runs")),
            use_ema=not args.use_main,
            device=device,
        )

    if args.smoke_test:
        ret = _smoke_pass(model, device, config)
        print("[encode_latents] SMOKE TEST PASS" if ret == 0 else "[encode_latents] SMOKE TEST FAIL")
        sys.exit(ret)

    output_path, time_array = encode_all_timesteps(model, device, config)
    if not args.skip_stats:
        compute_latent_stats(
            output_path, time_array,
            tuple(config["train_year_range"]),
            Path(_resolve_path(config["output_stats_json"])),
            config.get("latent_channels", LATENT_CHANNELS),
            config.get("latent_h", LATENT_H), config.get("latent_w", LATENT_W),
        )
    print("[encode_latents] Done.")


if __name__ == "__main__":
    main()
