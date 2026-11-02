#!/usr/bin/env python
"""Denoiser training script.

Implements LaDCast's EDM-style diffusion training recipe adapted for
our latent-space, mask-free setting. Reference:
ladcast/ladcast/training/train_AR.py.

Run a smoke (creates a synthetic latent Zarr at /tmp if needed):

  python -m closedloop_sampling.training.train_denoiser \\
      --smoke-test --batch-size 2

The smoke automatically redirects all paths to `/tmp/denoiser_smoke_*`,
auto-creates a synthetic latent Zarr there if missing, and reuses
it across multiple runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import EDMDPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel  # noqa: F401  (checkpoint resume uses the class)

from ema import build_ema
from trainer import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

try:
    import wandb
except ImportError:
    wandb = None

from closedloop_sampling import paths
from closedloop_sampling.forecast.models import make_denoiser
from dataset import make_latent_dataloader
from closedloop_sampling.regions import MAB_CONFIG, CCS_CONFIG

REGION_CONFIGS = {"mab": MAB_CONFIG, "ccs": CCS_CONFIG}


def _resolve_path(value: str, root: str = "data") -> str:
    """Resolve a config path. RegionConfig paths are RELATIVE (data paths
    against $CLOSEDLOOP_DATA_ROOT, run dirs against
    $CLOSEDLOOP_RUNS); absolute values (CLI overrides, /tmp smoke
    paths) pass through untouched."""
    if Path(value).is_absolute():
        return str(value)
    return paths.resolve(value, root=root)


# ============================================================
# Defaults (LaDCast-matched)
# ============================================================
_MAB = MAB_CONFIG

DEFAULT_CONFIG = {
    # Paths — RELATIVE to the env roots (resolved at use via _resolve_path);
    # taken from the MAB RegionConfig so the defaults equal `--region mab`.
    "latent_zarr_path": _MAB.paths.latents_zarr,
    "latent_stats_path": _MAB.paths.latent_norm_stats,
    # denoiser_<region>/seed_* run dirs live under $CLOSEDLOOP_RUNS.
    "output_base_dir": _MAB.paths.denoiser_runs_dir,
    # Schedule / optimizer
    "epochs": 15,
    "lr": 1e-4,
    "warmup_steps": 1000,
    "lr_cosine_cycles": 0.5,
    "optimizer_betas": (0.9, 0.999),
    "optimizer_eps": 1e-8,
    "weight_decay": 1e-2,
    "gradient_clip_norm": 1.0,
    # Batch
    "batch_size_per_device": 4,
    "gradient_accumulation_steps": 1,
    # num_workers=4 with spawn start method (set in main()) — sidesteps
    # a kernel 4.18.0 fork-hang seen on the training cluster. persistent_workers
    # in the dataloader factory keeps spawn-overhead at first-epoch only.
    "num_workers": 4,
    # EDM noise schedule
    "num_train_timesteps": 1000,
    "sigma_min": 0.002,
    "sigma_max": 80.0,
    "sigma_data": 0.5,
    "p_mean": -1.2,
    "p_std": 1.2,
    # Push-forward OFF — single-step training
    "num_push_forward_steps": 1,
    # Train/val split (MAB defaults; the region overlay in main() supplies
    # these from the RegionConfig)
    "train_year_range": (2007, 2020),
    "val_year_range": (2021, 2021),
    # EMA
    "ema_decay": 0.9999,
    "ema_use_warmup": True,
    "ema_update_after_step": 1000,
    "ema_inv_gamma": 1.0,
    "ema_power": 0.667,
    # Mixed precision
    "mixed_precision": "bf16",
    # Checkpointing
    "checkpoint_every_n_epochs": 2,
    "keep_last_n_checkpoints": 3,
    "save_best_val": True,
    # Validation (EDM loss only; AR rollout happens at evaluation time)
    "val_every_n_epochs": 2,
    "val_max_batches": 100,
    "log_every_n_steps": 50,
    # Seed sweeps are parameterized via --seed.
    "seed": 42,
    # Smoke knobs (only set in --smoke-test).
    "max_train_batches_per_epoch": None,
    # WandB.
    "wandb_project": "ladcast-regional-denoiser",
    "wandb_run_name": None,
    "wandb_tags": ["denoiser"],
    "no_wandb": False,
    "log_images": None,                    # None → default by context (see main)
    "dcae_checkpoint_for_images": str(
        Path(_MAB.paths.dcae_run_dir) / "checkpoint-best"
    ),
}


# ============================================================
# Factories
# ============================================================
def build_model(region_config=None) -> nn.Module:
    # region_config drives rope_spatial_grid_end_pos (= [H_lat-1, W_lat-1]); it is
    # NOT auto-derived, so a wrong grid is a SILENT positional-encoding error —
    # always pass the region's config rather than silently using the MAB RoPE
    # grid on another region.
    if region_config is None:
        return make_denoiser()                 # MAB default grid
    return make_denoiser(region_config=region_config)


def build_dataloader(year_range, config, shuffle):
    # CRITICAL: pass the RegionConfig so the window sampler uses the region's
    # internal_stride (INDEX steps: MAB 6 on an hourly store, CCS 1 on a
    # 6-hourly store — both = 6 h PHYSICAL). Inheriting MAB's 6 on CCS would
    # silently make training windows 36 h apart.
    return make_latent_dataloader(
        latent_zarr_path=_resolve_path(config["latent_zarr_path"]),
        latent_stats_path=_resolve_path(config["latent_stats_path"]),
        year_range=tuple(year_range),
        batch_size=config["batch_size_per_device"],
        num_workers=config["num_workers"],
        shuffle=shuffle,
        config=REGION_CONFIGS[config.get("region", "mab")],
    )


# ============================================================
# EDM noise sampling
# ============================================================
def sample_edm_sigmas(
    batch_size: int,
    p_mean: float,
    p_std: float,
    discrete_sigmas: torch.Tensor,    # (1000,) — the non-zero scheduler sigmas
    device: torch.device,
) -> torch.Tensor:
    """Karras lognormal sampling, snapped to nearest discrete sigma.

    Port of ladcast.models.utils.Karras_sigmas_lognormal.__call__,
    returning the snapped σ values directly. Equivalent (within
    float precision) to LaDCast's "snap to index → look up
    scheduler.timesteps[indices]" since timesteps = 0.25·log(σ).
    """
    rnd = torch.randn(batch_size, device=device)
    sigmas_cont = torch.exp(rnd * p_std + p_mean)
    diffs = torch.abs(sigmas_cont.unsqueeze(1) - discrete_sigmas.unsqueeze(0))
    idx = torch.argmin(diffs, dim=1)
    return discrete_sigmas[idx]


# ============================================================
# EDM loss
# ============================================================
def edm_preconditioning(sigma: torch.Tensor, sigma_data: float):
    """Karras 2022 EDM preconditioning factors. All return (B,)."""
    s2 = sigma ** 2
    sd2 = sigma_data ** 2
    c_skip = sd2 / (s2 + sd2)
    c_out = sigma * sigma_data / torch.sqrt(s2 + sd2)
    c_in = 1.0 / torch.sqrt(s2 + sd2)
    c_noise = 0.25 * torch.log(sigma)
    return c_skip, c_out, c_in, c_noise


def compute_edm_loss(
    model: nn.Module,
    model_kwargs: dict,
    clean_latent: torch.Tensor,         # (B, C, T, H, W)
    conditioning_latent: torch.Tensor,  # (B, C, 1, H, W)
    sigma: torch.Tensor,                # (B,)
    sigma_data: float,
    time_elapsed: Optional[torch.Tensor] = None,
    return_per_sample: bool = False,
):
    """One EDM training step's loss (Karras 2022, LaDCast convention).

    Weight = (σ² + σ_data²) / (σ · σ_data)²  per LaDCast
    train_AR.py:975-977.
    """
    B = clean_latent.shape[0]
    noise = torch.randn_like(clean_latent)
    sigma_b = sigma.view(B, 1, 1, 1, 1)
    noisy = clean_latent + sigma_b * noise

    c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, sigma_data)
    c_skip_b = c_skip.view(B, 1, 1, 1, 1)
    c_out_b = c_out.view(B, 1, 1, 1, 1)
    c_in_b = c_in.view(B, 1, 1, 1, 1)

    model_input = c_in_b * noisy

    result = model(
        hidden_states=model_input,
        timestep=c_noise,
        conditioning_tensors=conditioning_latent,
        time_elapsed=time_elapsed,
        **model_kwargs,
    )
    if isinstance(result, dict):
        pred = result["sample"]
    elif hasattr(result, "sample"):
        pred = result.sample
    else:
        pred = result

    x0_hat = c_skip_b * noisy + c_out_b * pred

    weight = (sigma ** 2 + sigma_data ** 2) / (sigma * sigma_data) ** 2
    weight_b = weight.view(B, 1, 1, 1, 1)
    weighted_sq_err = (x0_hat - clean_latent) ** 2 * weight_b
    loss = weighted_sq_err.mean()
    if return_per_sample:
        per_sample = weighted_sq_err.detach().mean(dim=(1, 2, 3, 4))
        return loss, per_sample
    return loss


# ============================================================
# Axis-order adapter (dataset T-first → model C-first)
# ============================================================
def reshape_batch_for_model(
    cond: torch.Tensor,  # (B, 84, 13, 30)
    pred: torch.Tensor,  # (B, 4, 84, 13, 30)
) -> Tuple[torch.Tensor, torch.Tensor]:
    cond_out = cond.unsqueeze(2)                  # (B, 84, 1, 13, 30)
    pred_out = pred.permute(0, 2, 1, 3, 4)        # (B, 84, 4, 13, 30)
    return cond_out, pred_out


def extract_inputs(batch: tuple, device: torch.device):
    """Return (cond, pred, model_kwargs) for a latent batch.

    `time_elapsed` is never passed in this pipeline (stays None
    downstream).
    """
    batch = tuple(
        t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        for t in batch
    )
    cond, pred, _ts = batch
    c, p = reshape_batch_for_model(cond, pred)
    return c, p, {}


# ============================================================
# LR scheduler (cosine + warmup)
# ============================================================
# get_cosine_schedule_with_warmup lives in trainer.py (imported above).


# ============================================================
# Validation: EDM loss only (no AR rollout — that's evaluation)
# ============================================================
@torch.no_grad()
def validate(
    model, ema_model, val_loader, accelerator, config,
    discrete_sigmas, max_batches=100,
):
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    model.eval()
    total, n = 0.0, 0
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        c, p, kwargs = extract_inputs(batch, accelerator.device)
        sigma = sample_edm_sigmas(
            p.shape[0], config["p_mean"], config["p_std"],
            discrete_sigmas, accelerator.device,
        )
        loss = compute_edm_loss(
            model, kwargs, p, c, sigma, sigma_data=config["sigma_data"],
        )
        g = accelerator.gather(loss.unsqueeze(0))
        total += g.mean().item()
        n += 1
    if ema_model is not None:
        ema_model.restore(model.parameters())
    model.train()
    return total / max(1, n)


# ============================================================
# Checkpoint save / prune
# ============================================================
def save_checkpoint(
    accelerator, epoch, model, ema_model, optimizer, scheduler,
    val_loss, output_dir: Path, is_best=False, global_step: int = 0,
):
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / f"checkpoint-epoch-{epoch:03d}"
    # ATOMIC: write into a hidden .tmp (meta.json LAST = commit marker), then
    # rename. A walltime-kill mid-write leaves only the .tmp (skipped by the
    # checkpoint-epoch-* glob), so --resume never sees a half-written checkpoint.
    tmp_dir = output_dir / f".checkpoint-epoch-{epoch:03d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    unwrapped = accelerator.unwrap_model(model)
    torch.save(unwrapped.state_dict(), tmp_dir / "model.pt")
    if ema_model is not None:
        torch.save(ema_model.state_dict(), tmp_dir / "ema.pt")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
    wandb_run_id = (wandb.run.id if (wandb is not None
                    and getattr(wandb, "run", None) is not None) else None)
    meta = {"epoch": int(epoch), "global_step": int(global_step),
            "ema_optimization_step": (int(ema_model.optimization_step)
                                      if ema_model is not None else None),
            "wandb_run_id": wandb_run_id,
            "val_loss": float(val_loss), "is_best": bool(is_best)}
    with open(tmp_dir / "meta.json", "w") as f:      # LAST = commit marker
        json.dump(meta, f, indent=2)
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    os.rename(tmp_dir, ckpt_dir)                     # atomic publish
    if is_best:
        best_link = output_dir / "checkpoint-best"
        if best_link.is_symlink():
            best_link.unlink()
        best_link.symlink_to(ckpt_dir.name)
    accelerator.print(
        f"Saved checkpoint to {ckpt_dir}" + (" (BEST)" if is_best else "")
    )


def prune_old_checkpoints(output_dir: Path, keep_last_n: int):
    cs = sorted(
        [p for p in output_dir.glob("checkpoint-epoch-*") if p.is_dir()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if len(cs) > keep_last_n:
        best_link = output_dir / "checkpoint-best"
        best_target = (
            best_link.resolve() if best_link.is_symlink() else None
        )
        for old in cs[:-keep_last_n]:
            if best_target is not None and old.resolve() == best_target:
                continue
            shutil.rmtree(old)


# ============================================================
# Main training
# ============================================================
def train(config: dict):
    set_seed(config["seed"])
    use_wandb = (not config.get("no_wandb", False)) and (wandb is not None)
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"],
        log_with="wandb" if use_wandb else None,
    )

    if use_wandb:
        _region = config.get("region", "mab")
        run_name = (
            config.get("wandb_run_name")
            or os.environ.get("WANDB_RUN_NAME")
            or f"{_region}_seed{config['seed']:03d}_"
               f"{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        tags_env = os.environ.get("WANDB_TAGS", "")
        env_tags = [t for t in tags_env.split(",") if t] if tags_env else []
        tags = list(
            config.get("wandb_tags") or env_tags
            or ["denoiser"]
        )
        for tag in (f"seed:{config['seed']}", _region):
            if tag not in tags:
                tags.append(tag)
        wandb_safe = {
            k: v for k, v in config.items()
            if isinstance(v, (int, float, str, bool, list, tuple))
        }
        wandb_kwargs = {"name": run_name, "tags": tags}
        # Resume the SAME wandb run across a --resume relaunch (one continuous
        # curve across walltime-kills); the run id is persisted in the ckpt meta.
        _rid = None
        _rf = config.get("resume_from")
        if _rf and (Path(_rf) / "meta.json").exists():
            with open(Path(_rf) / "meta.json") as _f:
                _rid = json.load(_f).get("wandb_run_id")
        if _rid:
            wandb_kwargs.update(id=_rid, resume="allow")
            accelerator.print(f"  wandb: RESUMING run id={_rid} (allow)")
        accelerator.init_trackers(
            project_name=config["wandb_project"],
            config=wandb_safe,
            init_kwargs={"wandb": wandb_kwargs},
        )
        accelerator.print(
            f"  wandb: project={config['wandb_project']} run={run_name} tags={tags}"
        )

    accelerator.print(
        f"=== Denoiser training: region={config.get('region', 'mab')}, "
        f"seed={config['seed']} ===")
    accelerator.print(f"  num_processes: {accelerator.num_processes}")
    accelerator.print(f"  mixed_precision: {config['mixed_precision']}")

    # Region-aware run dir (denoiser_mab / denoiser_ccs) so checkpoints
    # never collide across regions.
    _region = config.get("region", "mab")
    _rundir = f"denoiser_{_region}"
    output_dir = (
        Path(_resolve_path(config["output_base_dir"], root="runs"))
        / _rundir
        / f"seed_{config['seed']:03d}"
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(
                {k: v for k, v in config.items() if not isinstance(v, tuple)},
                f, indent=2, default=str,
            )

    accelerator.print("Building denoiser...", flush=True)
    model = build_model(region_config=REGION_CONFIGS[config.get('region', 'mab')])
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"  param count: {n_params:,}")

    accelerator.print("Building dataloaders...", flush=True)
    train_loader = build_dataloader(
        config["train_year_range"], config, shuffle=True
    )
    val_loader = build_dataloader(
        config["val_year_range"], config, shuffle=False
    )
    accelerator.print(f"  train batches/epoch: {len(train_loader)}")
    accelerator.print(f"  val batches:         {len(val_loader)}")

    # EDM sigma grid.
    edm = EDMDPMSolverMultistepScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        sigma_min=config["sigma_min"],
        sigma_max=config["sigma_max"],
        sigma_data=config["sigma_data"],
    )
    # sched.sigmas is length 1001 with a trailing 0.0; drop it so we
    # never snap to σ=0 (would explode c_noise=0.25·log(σ)).
    discrete_sigmas = edm.sigmas[:-1].to(accelerator.device)
    accelerator.print(
        f"  EDM sigma grid: {len(discrete_sigmas)} steps, "
        f"[{discrete_sigmas.min().item():.4f}, "
        f"{discrete_sigmas.max().item():.4f}]"
    )

    accelerator.print("Building optimizer + scheduler...", flush=True)
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=tuple(config["optimizer_betas"]),
        eps=config["optimizer_eps"],
        weight_decay=config["weight_decay"],
    )
    total_steps = (
        len(train_loader) * config["epochs"]
    ) // config["gradient_accumulation_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
        num_cycles=config["lr_cosine_cycles"],
    )

    accelerator.print("Building EMA model...", flush=True)
    ema_model = build_ema(model.parameters(), config)

    accelerator.print("accelerator.prepare()...", flush=True)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    accelerator.print("  prepare done.", flush=True)
    ema_model.to(accelerator.device)
    accelerator.print("  EMA on device.", flush=True)

    # ---- RESUME (additive): restore model + EMA + optimizer + scheduler +
    # epoch/global_step so a walltime-kill continues instead of restarting. Each
    # rank loads the SAME checkpoint after prepare(), so DDP stays consistent.
    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    resume_from = config.get("resume_from")
    if resume_from:
        ckpt = Path(resume_from)
        if not (ckpt / "meta.json").exists():
            raise FileNotFoundError(f"--resume: incomplete checkpoint at {ckpt} (no meta.json)")
        dev = accelerator.device
        accelerator.print(f"=== RESUME from {ckpt} ===", flush=True)
        accelerator.unwrap_model(model).load_state_dict(
            torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False))
        optimizer.load_state_dict(
            torch.load(ckpt / "optimizer.pt", map_location=dev, weights_only=False))
        scheduler.load_state_dict(
            torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=False))
        if ema_model is not None and (ckpt / "ema.pt").exists():
            ema_model.load_state_dict(
                torch.load(ckpt / "ema.pt", map_location="cpu", weights_only=False))
            ema_model.to(dev)
        with open(ckpt / "meta.json") as f:
            _meta = json.load(f)
        start_epoch = int(_meta["epoch"])
        global_step = int(_meta.get("global_step", 0))
        best_val = float(_meta.get("val_loss", float("inf")))
        accelerator.print(
            f"  resumed: start_epoch={start_epoch} global_step={global_step} "
            f"best_val={best_val:.5f}", flush=True)
        if ema_model is not None:
            _es = ema_model.optimization_step
            accelerator.print(
                f"  EMA restored: optimization_step={_es} "
                f"shadow_params={len(ema_model.shadow_params)} "
                f"(pre-kill meta ema_step={_meta.get('ema_optimization_step')})", flush=True)
            if _es != global_step:
                accelerator.print(
                    f"  WARN: EMA step {_es} != global_step {global_step} — "
                    f"EMA may not have resumed cleanly.", flush=True)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss, n_steps_epoch = 0.0, 0
        max_batches = config.get("max_train_batches_per_epoch")
        for batch_idx, batch in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch_idx == 0:
                accelerator.print(f"  first batch landed (epoch {epoch})", flush=True)
            with accelerator.accumulate(model):
                c, p, kwargs = extract_inputs(batch, accelerator.device)
                B = p.shape[0]
                sigma = sample_edm_sigmas(
                    B, config["p_mean"], config["p_std"],
                    discrete_sigmas, accelerator.device,
                )
                loss, per_sample_loss = compute_edm_loss(
                    model, kwargs, p, c, sigma,
                    sigma_data=config["sigma_data"],
                    return_per_sample=True,
                )
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_norm"]
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    ema_model.step(model.parameters())
                    global_step += 1

            epoch_loss += loss.item()
            n_steps_epoch += 1
            if (
                global_step % config["log_every_n_steps"] == 0
                and accelerator.is_main_process
            ):
                lr_now = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"  epoch {epoch:3d} | step {global_step:7d} | "
                    f"loss {loss.item():.5f} | lr {lr_now:.2e}", flush=True,
                )
                if use_wandb:
                    log_payload = {
                        "train/loss": float(loss.detach().item()),
                        "train/lr": float(lr_now),
                        "train/grad_norm": (
                            float(grad_norm) if grad_norm is not None else 0.0
                        ),
                        "train/epoch": int(epoch),
                        "train/sigma_mean": float(sigma.mean().item()),
                    }
                    # σ-binned loss diagnostic — per-batch fixed-edge
                    # bins across the discrete σ grid range
                    # (log-space quartiles).
                    s_min = float(config["sigma_min"])
                    s_max = float(config["sigma_max"])
                    bin_edges = np.exp(np.linspace(
                        math.log(s_min), math.log(s_max), 5,
                    ))
                    sigma_cpu = sigma.detach().float().cpu().numpy()
                    psl_cpu = per_sample_loss.float().cpu().numpy()
                    for q in range(4):
                        lo, hi = bin_edges[q], bin_edges[q + 1]
                        mask = (sigma_cpu >= lo) & (
                            sigma_cpu < hi if q < 3 else sigma_cpu <= hi
                        )
                        if mask.any():
                            log_payload[f"train/loss_sigma_q{q+1}"] = float(
                                psl_cpu[mask].mean()
                            )
                    accelerator.log(log_payload, step=global_step)

        avg = epoch_loss / max(1, n_steps_epoch)
        accelerator.print(f"=== Epoch {epoch} done | avg train loss {avg:.5f} ===")

        if (epoch + 1) % config["val_every_n_epochs"] == 0:
            v = validate(
                model, ema_model, val_loader, accelerator, config,
                discrete_sigmas, max_batches=config["val_max_batches"],
            )
            accelerator.print(f"  val_loss (EMA) = {v:.5f}")
            is_best = v < best_val
            if is_best:
                best_val = v
                accelerator.print(f"  NEW BEST val_loss = {v:.5f}")
            if use_wandb:
                accelerator.log(
                    {"val/loss": float(v), "val/epoch": int(epoch + 1)},
                    step=global_step,
                )
                if (
                    config.get("log_images", False)
                    and accelerator.is_main_process
                ):
                    log_forecast_image(
                        model, ema_model, val_loader, accelerator,
                        epoch + 1, global_step,
                        config["dcae_checkpoint_for_images"],
                        config["sigma_data"], discrete_sigmas,
                    )
        else:
            v = float("inf")
            is_best = False

        if (epoch + 1) % config["checkpoint_every_n_epochs"] == 0 or is_best:
            save_checkpoint(
                accelerator, epoch + 1, model, ema_model,
                optimizer, scheduler, v, output_dir, is_best=is_best,
                global_step=global_step,
            )
            if accelerator.is_main_process:
                prune_old_checkpoints(
                    output_dir, config["keep_last_n_checkpoints"]
                )

    accelerator.print(f"=== Training complete. Best val_loss = {best_val:.5f} ===")
    if use_wandb:
        accelerator.end_training()


# ============================================================
# WandB forecast-image logging (denoiser)
# ============================================================
@torch.no_grad()
def log_forecast_image(
    model, ema_model, val_loader, accelerator, epoch, global_step,
    dcae_checkpoint, sigma_data, discrete_sigmas,
):
    """Log one decoded forecast slice at t=24 h to WandB.

    Single-step denoise (not full EDM sampling) at a moderate σ for
    quick visual signal — full AR-rollout images are too slow for
    every val pass. Adds ~10 s wallclock per val pass.

    Gracefully skips if the DC-AE checkpoint is missing (typical
    before the DC-AE run has finished, or for the smoke).
    """
    if wandb is None:
        return
    ckpt = Path(_resolve_path(dcae_checkpoint, root="runs"))
    if not (ckpt / "ema.pt").exists() and not (ckpt / "model.pt").exists():
        accelerator.print(
            f"  [wandb] no DC-AE checkpoint at {ckpt} — skipping image log"
        )
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Lazy-load DC-AE (cached on the function object across val passes).
    if not hasattr(log_forecast_image, "_dcae"):
        from closedloop_sampling.forecast.decode import load_dcae_for_decode
        log_forecast_image._dcae = load_dcae_for_decode(
            ckpt, use_ema=True, device=accelerator.device,
        )
    dcae = log_forecast_image._dcae

    # EMA-swap so the denoiser image reflects what's saved.
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    model.eval()

    try:
        batch = next(iter(val_loader))
        c_cond, p_clean, kwargs = extract_inputs(
            batch, accelerator.device,
        )
        B = c_cond.shape[0]
        # Pick a moderate σ from the upper-middle of the grid.
        sigma_val = discrete_sigmas[len(discrete_sigmas) // 4]
        sigma = sigma_val.expand(B).to(accelerator.device)
        noise = torch.randn_like(p_clean)
        noisy = p_clean + sigma.view(B, 1, 1, 1, 1) * noise
        c_skip, c_out, c_in, c_noise = edm_preconditioning(sigma, sigma_data)
        model_input = c_in.view(B, 1, 1, 1, 1) * noisy
        with torch.amp.autocast(
            device_type=accelerator.device.type, dtype=torch.bfloat16,
        ):
            result = model(
                hidden_states=model_input,
                timestep=c_noise,
                conditioning_tensors=c_cond,
                **kwargs,
            )
        pred = result.sample if hasattr(result, "sample") else result
        x0_hat = (
            c_skip.view(B, 1, 1, 1, 1) * noisy + c_out.view(B, 1, 1, 1, 1) * pred
        )
        latent_at_24h = x0_hat[0, :, -1:, :, :].permute(1, 0, 2, 3)
        # → (1, 84, 13, 30) — squeeze T_pred=1, add a batch dim of 1 for decode.
        latent_to_decode = latent_at_24h
        with torch.amp.autocast(
            device_type=accelerator.device.type, dtype=torch.bfloat16,
        ):
            decoded = dcae.decode(latent_to_decode).float().cpu().numpy()
        # decoded shape: (1, 83, 104, 240)

        variables = {"u_surface": 19, "sst": 59, "zeta": 80}
        image_logs = {}
        for var_name, ch in variables.items():
            field = decoded[0, ch]
            vmax = float(np.abs(field).max() + 1e-9)
            fig, ax = plt.subplots(1, 1, figsize=(5, 3))
            im = ax.imshow(
                field, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax,
            )
            ax.set_title(f"{var_name} @ t=24h, ep {epoch}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)
            image_logs[f"forecast/{var_name}"] = wandb.Image(
                fig, caption=f"epoch {epoch}",
            )
            plt.close(fig)
        accelerator.log(image_logs, step=global_step)
    finally:
        if ema_model is not None:
            ema_model.restore(model.parameters())
        model.train()


# ============================================================
# Synthetic data setup for smoke tests
# ============================================================
SMOKE_ZARR = "/tmp/denoiser_smoke_latents.zarr"
SMOKE_STATS = "/tmp/denoiser_smoke_latent_stats.json"
SMOKE_RUNS = "/tmp/denoiser_smoke_runs"


def _ensure_smoke_data(n_timesteps: int = 200):
    """Create a synthetic latent Zarr + stats JSON at /tmp if missing.

    Schema matches encode_latents.py's output: variable `latents` of
    shape (T, 84, 13, 30) with `dimension_names`, and `ocean_time`
    coord; stats JSON is the per-channel mean/std schema the latent
    normalizer expects.
    """
    import zarr
    import zarr.codecs
    if Path(SMOKE_ZARR).exists() and Path(SMOKE_STATS).exists():
        return
    if Path(SMOKE_ZARR).exists():
        shutil.rmtree(SMOKE_ZARR)
    rng = np.random.default_rng(0)
    latents = rng.standard_normal(
        (n_timesteps, 84, 13, 30), dtype=np.float32
    ) * 0.5
    times = (
        np.arange(n_timesteps, dtype="timedelta64[h]")
        + np.datetime64("2007-01-02T01:00:00")
    )
    root = zarr.open(SMOKE_ZARR, mode="w")
    root.create_array(
        "latents",
        shape=latents.shape,
        chunks=(72, 84, 13, 30),
        dtype=np.float32,
        compressors=[zarr.codecs.BloscCodec(cname="lz4", clevel=5)],
        dimension_names=["ocean_time", "C", "H", "W"],
    )[:] = latents
    root.create_array(
        "ocean_time",
        shape=times.shape, chunks=times.shape, dtype=times.dtype,
        dimension_names=["ocean_time"],
    )[:] = times
    mean = latents.mean(axis=(0, 2, 3)).tolist()
    std = latents.std(axis=(0, 2, 3)).tolist()
    with open(SMOKE_STATS, "w") as f:
        json.dump({"mean": mean, "std": std,
                   "_meta": {"n_channels": 84, "source": "synthetic"}}, f)
    print(f"[smoke] created synthetic latents at {SMOKE_ZARR} "
          f"({n_timesteps} timesteps)")


# ============================================================
# CLI
# ============================================================
def main():
    # MUST be first: switch DataLoader workers to spawn IPC, sidestepping
    # the cluster's 4.18.0 fork-hang. See train_dcae.py for context.
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)

    p = argparse.ArgumentParser()
    p.add_argument("--region", choices=list(REGION_CONFIGS), default="mab",
                   help="RegionConfig (default mab): drives latent paths, "
                        "split, internal_stride (window spacing) and the RoPE latent grid.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a checkpoint dir (model+ema+optimizer+scheduler+epoch/"
             "step restored) — continues instead of restarting.",
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-base-dir", type=str, default=None)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging entirely.",
    )
    p.add_argument(
        "--log-images", dest="log_images", action="store_true", default=None,
        help="Log decoded forecast images each val pass (default: ON for "
             "single jobs, OFF inside SLURM array jobs).",
    )
    p.add_argument(
        "--no-log-images", dest="log_images", action="store_false",
    )
    p.add_argument(
        "--dump-config", action="store_true",
        help="Print the fully-resolved config as JSON and exit (no side effects). "
             "Useful for diffing the per-region resolved configs before submit.",
    )
    p.add_argument(
        "--dcae-checkpoint-for-images", type=str, default=None,
        help="DC-AE checkpoint dir used for log_forecast_image's decode.",
    )
    args = p.parse_args()

    config = dict(DEFAULT_CONFIG)
    # Region overlay: for --region mab these equal DEFAULT_CONFIG exactly
    # (DEFAULT_CONFIG is built from the MAB config), so default behavior is
    # unchanged; --region ccs swaps in the CCS latents/stats/split.
    _rc = REGION_CONFIGS[args.region]
    config["region"] = args.region
    config["latent_zarr_path"] = _rc.paths.latents_zarr
    config["latent_stats_path"] = _rc.paths.latent_norm_stats
    config["output_base_dir"] = _rc.paths.denoiser_runs_dir
    config["train_year_range"] = tuple(_rc.timesplit.train_year_range)
    config["val_year_range"] = tuple(_rc.timesplit.val_year_range)
    # The forecast-image decode MUST use the region's OWN DC-AE. The DC-AE is fully
    # convolutional, so MAB weights would happily decode a CCS 22x23 latent to a
    # 176x184 field and log a plausible-looking, meaningless image — no shape error to
    # catch it. Same run-dir rule as encode_latents.py (the RegionConfig's
    # dcae_run_dir, resolved against the runs root at use).
    config["dcae_checkpoint_for_images"] = str(
        Path(_rc.paths.dcae_run_dir) / "checkpoint-best"
    )
    if args.config:
        with open(args.config) as f:
            if args.config.endswith((".yaml", ".yml")):
                import yaml
                config.update(yaml.safe_load(f))
            else:
                config.update(json.load(f))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.lr is not None:
        config["lr"] = args.lr
    if args.batch_size is not None:
        config["batch_size_per_device"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.output_base_dir is not None:
        config["output_base_dir"] = args.output_base_dir
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        config["wandb_run_name"] = args.wandb_run_name
    if args.wandb_tags is not None:
        config["wandb_tags"] = args.wandb_tags
    if args.no_wandb:
        config["no_wandb"] = True
    if args.dcae_checkpoint_for_images is not None:
        config["dcae_checkpoint_for_images"] = args.dcae_checkpoint_for_images
    if args.log_images is None:
        # Default ON for single jobs, OFF inside SLURM array jobs (seed
        # sweeps: image logging × N runs is wasteful).
        config["log_images"] = "SLURM_ARRAY_TASK_ID" not in os.environ
    else:
        config["log_images"] = bool(args.log_images)

    if args.smoke_test:
        _ensure_smoke_data(n_timesteps=200)
        config["latent_zarr_path"] = SMOKE_ZARR
        config["latent_stats_path"] = SMOKE_STATS
        config["output_base_dir"] = SMOKE_RUNS
        config["epochs"] = 2
        config["warmup_steps"] = 10
        config["train_year_range"] = (2007, 2007)
        config["val_year_range"] = (2007, 2007)
        config["checkpoint_every_n_epochs"] = 1
        config["val_every_n_epochs"] = 1
        config["val_max_batches"] = 5
        config["max_train_batches_per_epoch"] = 20
        config["log_every_n_steps"] = 5

    if args.resume is not None:
        config["resume_from"] = args.resume

    # Dump the FULLY-RESOLVED config and exit, before any side effect. This is
    # what region-config diffs compare across regions — it must see exactly what
    # train() would see, never a reimplementation of the overlay.
    if args.dump_config:
        print(json.dumps(config, indent=2, default=str))
        return

    # Fresh-start guard: a run WITHOUT --resume must not silently start at epoch 0
    # in a dir that already holds checkpoints (it would overwrite them).
    if not config.get("resume_from"):
        _rg = config.get("region", "mab")
        _rd = f"denoiser_{_rg}"
        _out = (Path(_resolve_path(config["output_base_dir"], root="runs")) / _rd
                / f"seed_{config['seed']:03d}")
        _existing = sorted(_out.glob("checkpoint-epoch-*")) if _out.exists() else []
        if _existing:
            raise SystemExit(
                f"output_dir {_out} already has {len(_existing)} checkpoint(s) and "
                f"--resume was not passed. Pass --resume <checkpoint-dir> to continue, "
                f"or clear/rename the dir to start fresh.")

    train(config)


if __name__ == "__main__":
    main()
