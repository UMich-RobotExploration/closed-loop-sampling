#!/usr/bin/env python
"""DC-AE training script.

Trains RegionalDCAE on the preprocessed pixel Zarr (the Doppio v3 ROMS
reanalysis for the MAB region), following the production recipe:

  - 30 pretrain epochs (decoder-finetune LOCKED to 0 — see DEFAULT_CONFIG)
  - AdamW (0.9, 0.999, eps 1e-8, wd 1e-2), batch=4/device, grad_accum=1
  - LR 1e-4 (1e-5 finetune), cosine + 1000-step warmup
  - bf16, grad-clip 1.0, EMA 0.9999 (warmup, update_after 1000)
  - Mask-aware MSE on dynamic channels; static mask_rho NOT
    normalized; KL disabled.
  - Checkpoint every 5 epochs + best-val; keep last 3 + best.

Usage:
  python -m closedloop_sampling.training.train_dcae \\
      [--region mab|ccs] [--config CONFIG_PATH] [--epochs N] [--lr LR] \\
      [--batch-size BS] [--num-workers N] [--output-dir DIR] \\
      [--smoke-test]

Smoke-test mode runs 1 epoch on the first train year only with a short
warmup; intended for a quick (≲30 min) end-to-end plumbing check
on a single A40.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.training_utils import EMAModel  # noqa: F401  (checkpoint resume uses the class)

from closedloop_sampling.forecast.losses import mask_aware_mse
from ema import build_ema
from trainer import get_cosine_schedule_with_warmup

# wandb is optional — graceful fallback if absent / --no-wandb passed.
try:
    import wandb
except ImportError:
    wandb = None

from closedloop_sampling import paths
from closedloop_sampling.forecast.models import RegionalDCAE
from dataset import make_pixel_dataloader
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
    "zarr_path": _MAB.paths.preprocessed_zarr,
    "stats_json_path": _MAB.paths.pixel_norm_stats,
    "output_dir": _MAB.paths.dcae_run_dir,
    # Schedule
    "pretrain_epochs": 30,
    # LOCKED PRETRAIN-ONLY (was 10). The decoder-finetune/refine phase is
    # NON-FUNCTIONAL (see the landmine comment at the refine transition) and
    # models are trained pretrain-only — apples-to-apples with the released
    # checkpoints, whose refine likewise never ran.
    # 0 => the refine phase is skipped entirely and the buggy DDP path is
    # never reached.
    "decoder_finetune_epochs": 0,
    "lr_pretrain": 1e-4,
    "lr_finetune": 1e-5,
    "warmup_steps": 1000,
    "lr_cosine_cycles": 0.5,
    # Optimizer
    "optimizer_betas": (0.9, 0.999),
    "optimizer_eps": 1e-8,
    "weight_decay": 1e-2,
    "gradient_clip_norm": 1.0,
    # Split (MAB defaults; the region overlay in main() supplies these
    # from the RegionConfig)
    "train_year_range": (2007, 2020),
    "val_year_range": (2021, 2021),
    # Batch
    "batch_size_per_device": 4,
    "gradient_accumulation_steps": 1,
    # num_workers=4 with mp.set_start_method('spawn') at top of main()
    # — sidesteps a kernel 4.18.0 fork-hang seen on the training cluster
    # (an early production attempt at num_workers=0 ran at ~7.5 s/step on
    # 4 A40s, infeasible at full scale). Spawn workers re-import the
    # module per worker but persistent_workers=True (set in the
    # dataloader factory) keeps them alive across epochs.
    "num_workers": 4,
    # EMA
    "ema_decay": 0.9999,
    "ema_use_warmup": True,
    "ema_update_after_step": 1000,
    "ema_inv_gamma": 1.0,
    "ema_power": 0.667,
    # Precision
    "mixed_precision": "bf16",
    # Checkpointing
    "checkpoint_every_n_epochs": 5,
    "keep_last_n_checkpoints": 3,
    "save_best_val": True,
    # Validation
    "val_every_n_epochs": 2,
    "val_max_batches": 200,
    # Per-epoch caps (None = unlimited). Smoke mode caps these
    # to a small number so the plumbing check finishes in minutes
    # rather than hours regardless of NFS first-read latency.
    "max_train_batches_per_epoch": None,
    # Logging
    "log_every_n_steps": 50,
    "seed": 42,
    # WandB (set in main() from CLI / env vars / SLURM context).
    "wandb_project": "ladcast-regional-dcae",
    "wandb_run_name": None,
    "wandb_tags": ["dcae"],
    "log_images_every_n_val": 1,
    "no_wandb": False,
}


# ============================================================
# Loss (mask-aware MSE)
# ============================================================
# mask_aware_mse lives in closedloop_sampling.forecast.losses (imported above).


# ============================================================
# LR scheduler: linear warmup → cosine to ~0
# ============================================================
# get_cosine_schedule_with_warmup lives in trainer.py (imported above).


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(
    accelerator: Accelerator,
    epoch: int,
    model: nn.Module,
    ema_model: Optional[EMAModel],
    optimizer,
    scheduler,
    val_loss: float,
    output_dir: Path,
    is_best: bool = False,
    global_step: int = 0,
):
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / f"checkpoint-epoch-{epoch:03d}"

    # ATOMIC checkpoint: write everything into a hidden .tmp dir (meta.json LAST,
    # as the commit marker), then atomically rename .tmp -> checkpoint-epoch-NNN.
    # A walltime-kill mid-write only ever leaves the .tmp behind (skipped by the
    # `checkpoint-epoch-*` glob), so --resume never sees a half-written checkpoint
    # (the smoke found the non-atomic version could leave a partial dir missing
    # optimizer.pt/meta.json, breaking resume).
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
    # Persist the active wandb run id so --resume reuses the SAME run (continuous
    # curve). wandb.run exists only on the main process, which is the only rank here.
    wandb_run_id = (wandb.run.id if (wandb is not None
                    and getattr(wandb, "run", None) is not None) else None)
    with open(tmp_dir / "meta.json", "w") as f:          # written LAST = commit marker
        json.dump(
            {"epoch": epoch, "global_step": int(global_step),
             "ema_optimization_step": (int(ema_model.optimization_step)
                                       if ema_model is not None else None),
             "wandb_run_id": wandb_run_id,
             "val_loss": float(val_loss), "is_best": bool(is_best)},
            f, indent=2,
        )
    if ckpt_dir.exists():                                 # only on same-epoch re-save
        shutil.rmtree(ckpt_dir)
    os.rename(tmp_dir, ckpt_dir)                          # atomic publish (same FS)

    if is_best:
        best_link = output_dir / "checkpoint-best"
        if best_link.is_symlink() or best_link.exists():
            best_link.unlink()
        best_link.symlink_to(ckpt_dir.name)
    accelerator.print(
        f"Saved checkpoint to {ckpt_dir}" + (" (BEST)" if is_best else "")
    )


def prune_old_checkpoints(output_dir: Path, keep_last_n: int):
    if not output_dir.exists():
        return
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-epoch-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if len(ckpts) <= keep_last_n:
        return
    best_link = output_dir / "checkpoint-best"
    best_target = best_link.resolve() if best_link.is_symlink() else None
    for old in ckpts[:-keep_last_n]:
        if best_target is not None and old.resolve() == best_target:
            continue
        shutil.rmtree(old)


# ============================================================
# Validation
# ============================================================
@torch.no_grad()
def validate(
    model: nn.Module,
    ema_model: Optional[EMAModel],
    val_loader,
    accelerator: Accelerator,
    max_batches: int = 200,
    use_ema: bool = True,
) -> float:
    """Run validation; returns mean masked-MSE across (capped) val set.

    When `use_ema=True`, swaps EMA weights into the live model for the
    eval pass and restores them afterwards.
    """
    if use_ema and ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    model.eval()
    total, n = 0.0, 0
    for i, (dynamic, static, mask) in enumerate(val_loader):
        if i >= max_batches:
            break
        result = model(dynamic, static)
        loss = mask_aware_mse(result["reconstructed"], dynamic, mask)
        gathered = accelerator.gather(loss.detach().unsqueeze(0))
        total += gathered.mean().item()
        n += 1
    if use_ema and ema_model is not None:
        ema_model.restore(model.parameters())
    model.train()
    return total / max(1, n)


# ============================================================
# WandB reconstruction-image logging
# ============================================================
@torch.no_grad()
def log_reconstruction_images(
    model, ema_model, val_loader, accelerator, epoch, global_step,
):
    """Log 3-panel (target | pred | diff) images for SST, surface u, ζ
    using the EMA-swapped weights. Only fires on main process.

    Adds ~2-3 s wallclock per val pass. matplotlib uses Agg backend
    (set lazily on first call) so this works on headless GPU nodes.
    """
    if wandb is None:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Swap EMA in so logged images reflect what we'd save.
    if ema_model is not None:
        ema_model.store(model.parameters())
        ema_model.copy_to(model.parameters())
    model.eval()

    try:
        # One batch suffices.
        batch = next(iter(val_loader))
        dynamic, static, _ = batch
        dynamic = dynamic.to(accelerator.device)
        static = static.to(accelerator.device)
        with torch.amp.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16):
            out = model(dynamic, static)
        pred = out["reconstructed"].float().cpu().numpy()
        target = dynamic.float().cpu().numpy()

        # Surface fields by channel layout: surface u = top σ idx 19,
        # SST = channel 40+19 = 59, ζ = 80.
        variables = {"u_surface": 19, "sst": 59, "zeta": 80}
        image_logs = {}
        for var_name, ch in variables.items():
            t_field = target[0, ch]
            p_field = pred[0, ch]
            d_field = p_field - t_field
            fig, axes = plt.subplots(1, 3, figsize=(12, 3))
            vmax = max(abs(t_field).max(), abs(p_field).max())
            for ax, field, title, vlim_sym in zip(
                axes,
                [t_field, p_field, d_field],
                [f"target_{var_name}", f"pred_{var_name}", f"diff_{var_name}"],
                [vmax, vmax, abs(d_field).max() + 1e-9],
            ):
                im = ax.imshow(field, cmap="RdBu_r", origin="lower",
                               vmin=-vlim_sym, vmax=vlim_sym)
                ax.set_title(title)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046)
            plt.tight_layout()
            image_logs[f"images/{var_name}"] = wandb.Image(
                fig, caption=f"epoch {epoch}",
            )
            plt.close(fig)
        accelerator.log(image_logs, step=global_step)
    finally:
        if ema_model is not None:
            ema_model.restore(model.parameters())
        model.train()


# ============================================================
# Training loop
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
        region = config.get("region", "mab")
        run_name = (
            config.get("wandb_run_name")
            or os.environ.get("WANDB_RUN_NAME")
            # region-aware default so CCS runs are named/filterable as ccs, not
            # mingled with the MAB runs in the shared project.
            or f"dcae_{region}_seed{config['seed']}_{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        tags_env = os.environ.get("WANDB_TAGS", "")
        tags = (
            config.get("wandb_tags")
            or ([t for t in tags_env.split(",") if t] if tags_env else [])
        )
        tags = list(tags)
        if region not in tags:
            tags.append(region)                          # region tag (mab / ccs)
        wandb_safe_config = {
            k: v for k, v in config.items()
            if isinstance(v, (int, float, str, bool, list, tuple))
        }
        wandb_kwargs = {"name": run_name, "tags": tags}
        # Resume the SAME wandb run across a --resume relaunch (one continuous
        # curve across walltime-kills). The run id was persisted to the
        # checkpoint meta on save; combined with the restored global_step the
        # logged steps continue seamlessly.
        resume_run_id = None
        _rf = config.get("resume_from")
        if _rf and (Path(_rf) / "meta.json").exists():
            with open(Path(_rf) / "meta.json") as _f:
                resume_run_id = json.load(_f).get("wandb_run_id")
        if resume_run_id:
            wandb_kwargs.update(id=resume_run_id, resume="allow")
            accelerator.print(f"  wandb: RESUMING run id={resume_run_id} (allow)")
        accelerator.init_trackers(
            project_name=config["wandb_project"],
            config=wandb_safe_config,
            init_kwargs={"wandb": wandb_kwargs},
        )
        accelerator.print(
            f"  wandb: project={config['wandb_project']} run={run_name} tags={tags}")

    accelerator.print("=== DC-AE Training ===")
    accelerator.print(f"num_processes: {accelerator.num_processes}")
    accelerator.print(f"mixed_precision: {config['mixed_precision']}")
    accelerator.print(f"per-device batch: {config['batch_size_per_device']}")
    effective = (
        config["batch_size_per_device"]
        * accelerator.num_processes
        * config["gradient_accumulation_steps"]
    )
    accelerator.print(f"effective batch: {effective}")

    output_dir = Path(_resolve_path(config["output_dir"], root="runs"))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)

    # Model
    region_cfg = REGION_CONFIGS[config.get("region", "mab")]
    accelerator.print(f"Building RegionalDCAE (region={config.get('region', 'mab')})...")
    model = RegionalDCAE(region_config=region_cfg)
    n_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"  param count: {n_params:,}")

    # Dataloaders
    accelerator.print("Building dataloaders...")
    train_loader = make_pixel_dataloader(
        zarr_path=_resolve_path(config["zarr_path"]),
        stats_json_path=_resolve_path(config["stats_json_path"]),
        year_range=tuple(config["train_year_range"]),
        batch_size=config["batch_size_per_device"],
        num_workers=config["num_workers"],
        shuffle=True,
        config=region_cfg,
    )
    val_loader = make_pixel_dataloader(
        zarr_path=_resolve_path(config["zarr_path"]),
        stats_json_path=_resolve_path(config["stats_json_path"]),
        year_range=tuple(config["val_year_range"]),
        batch_size=config["batch_size_per_device"],
        num_workers=config["num_workers"],
        shuffle=False,
        config=region_cfg,
    )
    accelerator.print(f"  train batches/epoch: {len(train_loader)}")
    accelerator.print(f"  val batches      : {len(val_loader)}")

    accelerator.print("Building optimizer + scheduler...", flush=True)
    # Optimizer + scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr_pretrain"],
        betas=tuple(config["optimizer_betas"]),
        eps=config["optimizer_eps"],
        weight_decay=config["weight_decay"],
    )
    total_steps = (
        (len(train_loader) * config["pretrain_epochs"])
        + (len(train_loader) * config["decoder_finetune_epochs"])
    ) // config["gradient_accumulation_steps"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=total_steps,
        num_cycles=config["lr_cosine_cycles"],
    )

    accelerator.print("Building EMA model...", flush=True)
    # EMA (no model_cls — we use raw state_dict() rather than save_pretrained).
    ema_model = build_ema(model.parameters(), config)

    if config.get("use_torch_compile", False):
        accelerator.print("torch.compile enabled (mode=default)…", flush=True)
        model = torch.compile(model, mode="default")

    accelerator.print("accelerator.prepare()...", flush=True)
    # Prepare
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    accelerator.print("  prepare done.", flush=True)
    accelerator.print("Moving EMA to device...", flush=True)
    ema_model.to(accelerator.device)
    accelerator.print("  EMA on device.", flush=True)

    # ---- RESUME from a checkpoint dir (additive; pretrain path) ----------------
    # Restores model + EMA + optimizer + scheduler + epoch/global_step so a
    # walltime-kill/crash continues instead of restarting from 0. Each rank loads
    # the SAME checkpoint after prepare(), so DDP stays consistent. The checkpoint
    # dir is the save_checkpoint() layout: model.pt / ema.pt / optimizer.pt /
    # scheduler.pt / meta.json{epoch, global_step, val_loss}.
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    resume_from = config.get("resume_from")
    if resume_from:
        ckpt = Path(resume_from)
        if not (ckpt / "meta.json").exists():
            # meta.json is written last (commit marker); its absence means an
            # incomplete/partial checkpoint. Atomic saves make this unreachable
            # for a checkpoint-epoch-NNN dir, but guard anyway.
            raise FileNotFoundError(
                f"--resume: incomplete checkpoint at {ckpt} (no meta.json)")
        dev = accelerator.device
        accelerator.print(f"=== RESUME from {ckpt} ===", flush=True)
        accelerator.unwrap_model(model).load_state_dict(
            torch.load(ckpt / "model.pt", map_location="cpu", weights_only=False))
        optimizer.load_state_dict(
            torch.load(ckpt / "optimizer.pt", map_location=dev, weights_only=False))
        scheduler.load_state_dict(
            torch.load(ckpt / "scheduler.pt", map_location="cpu", weights_only=False))
        if ema_model is not None and (ckpt / "ema.pt").exists():
            # Restores EMA shadow_params (the ship-weights the downstream
            # encode/decode pipeline uses) AND optimization_step — diffusers
            # EMAModel.load_state_dict sets both; .to(dev) then moves the loaded
            # shadow tensors onto the GPU.
            ema_model.load_state_dict(
                torch.load(ckpt / "ema.pt", map_location="cpu", weights_only=False))
            ema_model.to(dev)
        with open(ckpt / "meta.json") as f:
            meta = json.load(f)
        start_epoch = int(meta["epoch"])            # completed-epoch count == next loop index
        global_step = int(meta.get("global_step", start_epoch * len(train_loader)))
        best_val_loss = float(meta.get("val_loss", float("inf")))
        ema_step = (ema_model.optimization_step if ema_model is not None else None)
        accelerator.print(
            f"  resumed: start_epoch={start_epoch} global_step={global_step} "
            f"best_val_loss={best_val_loss:.5f}", flush=True)
        if ema_model is not None:
            accelerator.print(
                f"  EMA restored: optimization_step={ema_step} "
                f"shadow_params={len(ema_model.shadow_params)} "
                f"(pre-kill meta ema_step={meta.get('ema_optimization_step')})",
                flush=True)
            # EMA and global_step increment together (one ema.step per sync step),
            # so a correct resume has them equal — flag corruption loudly.
            if ema_step != global_step:
                accelerator.print(
                    f"  WARN: EMA optimization_step {ema_step} != global_step "
                    f"{global_step} — EMA may not have resumed cleanly.", flush=True)
    # Wallclock probe: time per training step (logged alongside loss/lr).
    _t_last_log = None
    _step_at_last_log = 0
    import time as _time
    _step_times = []  # list of (global_step, sec_per_step) — benchmark-mode summary
    _vram_at_step_50 = None
    total_epochs = config["pretrain_epochs"] + config["decoder_finetune_epochs"]

    for epoch in range(start_epoch, total_epochs):
        # ===================== LANDMINE (do not enable blindly) =====================
        # Decoder-finetune/refine phase is currently NON-FUNCTIONAL: freezing the
        # encoder (below) + vanilla DDP hits a `find_unused_parameters` crash on the
        # first refine batch (Accelerator is built without DDP kwargs). Models are
        # trained PRETRAIN-ONLY (decoder_finetune_epochs=0) — apples-to-apples with
        # the released checkpoints, whose refine likewise never ran. With
        # finetune_epochs=0 this branch is UNREACHABLE (`epoch` never equals
        # pretrain_epochs before the loop ends), so the buggy frozen-encoder DDP path
        # is never entered — the crash is dormant. Do NOT enable finetune without
        # applying the DDP fix
        # (DistributedDataParallelKwargs(find_unused_parameters=True)) and smoke-
        # testing the frozen-encoder step first.
        # ===========================================================================
        # Transition into decoder-finetune at epoch == pretrain_epochs.
        if epoch == config["pretrain_epochs"]:
            accelerator.print(
                f"=== Entering decoder-finetune phase at epoch {epoch} ==="
            )
            unwrapped = accelerator.unwrap_model(model)
            for p in unwrapped.dcae.encoder.parameters():
                p.requires_grad_(False)
            # LambdaLR caches base_lrs at __init__ from g["lr"]; it
            # does NOT re-read g["initial_lr"] on subsequent step()s.
            # To lower the LR mid-training we update both the
            # optimizer's lr/initial_lr fields AND the scheduler's
            # base_lrs list so the next scheduler.step() computes
            # lr_lambda(step) × lr_finetune.
            inner_sched = (
                scheduler.scheduler
                if hasattr(scheduler, "scheduler")
                else scheduler
            )
            for i, g in enumerate(optimizer.param_groups):
                g["initial_lr"] = config["lr_finetune"]
                g["lr"] = config["lr_finetune"]
            inner_sched.base_lrs = [
                config["lr_finetune"] for _ in inner_sched.base_lrs
            ]
            accelerator.print(
                f"  encoder frozen; base LR set to {config['lr_finetune']:.2e}"
            )

        model.train()
        epoch_loss, n_steps_epoch = 0.0, 0
        max_batches = config.get("max_train_batches_per_epoch")
        for batch_idx, (dynamic, static, mask) in enumerate(train_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if batch_idx == 0:
                accelerator.print(f"  first batch landed (epoch {epoch})")
            with accelerator.accumulate(model):
                result = model(dynamic, static)
                loss = mask_aware_mse(result["reconstructed"], dynamic, mask)
                accelerator.backward(loss)
                grad_norm = None
                if accelerator.sync_gradients:
                    # clip_grad_norm_ returns the total pre-clip norm
                    # — capture for WandB logging.
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(), config["gradient_clip_norm"]
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    ema_model.step(model.parameters())
                    global_step += 1

            epoch_loss += loss.detach().item()
            n_steps_epoch += 1

            if (
                global_step > 0
                and global_step % config["log_every_n_steps"] == 0
                and accelerator.sync_gradients
                and accelerator.is_main_process
            ):
                current_lr = scheduler.get_last_lr()[0]
                # Steady-state per-step wallclock since previous log.
                _now = _time.time()
                if _t_last_log is None:
                    sec_per_step_str = "  ?  s/step"
                else:
                    n = max(1, global_step - _step_at_last_log)
                    sec_per_step = (_now - _t_last_log) / n
                    sec_per_step_str = f"{sec_per_step:.2f}s/step"
                    _step_times.append((global_step, sec_per_step))
                _t_last_log = _now
                _step_at_last_log = global_step
                # VRAM snapshot once we're past warmup.
                if (
                    _vram_at_step_50 is None and global_step >= 50
                    and torch.cuda.is_available()
                ):
                    _vram_at_step_50 = (
                        torch.cuda.max_memory_allocated() / (1024 ** 2)
                    )
                accelerator.print(
                    f"  epoch {epoch:3d} | step {global_step:7d} | "
                    f"loss {loss.item():.5f} | lr {current_lr:.2e} | "
                    f"{sec_per_step_str}"
                )
                if use_wandb:
                    accelerator.log({
                        "train/loss": float(loss.detach().item()),
                        "train/lr": float(current_lr),
                        "train/grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
                        "train/epoch": int(epoch),
                    }, step=global_step)

        avg_loss = epoch_loss / max(1, n_steps_epoch)
        accelerator.print(
            f"=== Epoch {epoch} done | avg train loss {avg_loss:.5f} ==="
        )

        # Validation
        is_best = False
        val_loss = float("inf")
        if (epoch + 1) % config["val_every_n_epochs"] == 0:
            val_loss = validate(
                model, ema_model, val_loader, accelerator,
                max_batches=config["val_max_batches"],
            )
            accelerator.print(f"  val_loss (EMA) = {val_loss:.5f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
                accelerator.print(f"  NEW BEST val_loss = {val_loss:.5f}")
            if use_wandb:
                accelerator.log({
                    "val/loss": float(val_loss),
                    "val/epoch": int(epoch + 1),
                }, step=global_step)
                if (
                    accelerator.is_main_process
                    and (epoch + 1) % config["log_images_every_n_val"] == 0
                ):
                    log_reconstruction_images(
                        model, ema_model, val_loader, accelerator,
                        epoch + 1, global_step,
                    )

        if (epoch + 1) % config["checkpoint_every_n_epochs"] == 0 or is_best:
            save_checkpoint(
                accelerator, epoch + 1, model, ema_model,
                optimizer, scheduler, val_loss,
                output_dir, is_best=is_best, global_step=global_step,
            )
            if accelerator.is_main_process:
                prune_old_checkpoints(output_dir, config["keep_last_n_checkpoints"])

    accelerator.print(
        f"=== Training complete. Best val_loss = {best_val_loss:.5f} ==="
    )
    if config.get("bench_mode", False) and accelerator.is_main_process:
        # Steady-state mean: skip per-step values from before global_step 20.
        ss = [t for (s, t) in _step_times if s >= 20]
        mean_ss = sum(ss) / len(ss) if ss else float("nan")
        vram = _vram_at_step_50 if _vram_at_step_50 is not None else float("nan")
        accelerator.print(
            "BENCHMARK_SUMMARY: "
            f"label={config.get('bench_label', '?')} "
            f"gpus={accelerator.num_processes} "
            f"batch={config['batch_size_per_device']} "
            f"compile={int(bool(config.get('use_torch_compile', False)))} "
            f"mean_step_time={mean_ss:.3f}s "
            f"n_steady_steps={len(ss)} "
            f"vram_max_MiB={vram:.0f}"
        )
    if use_wandb:
        accelerator.end_training()


# ============================================================
# CLI
# ============================================================
def main():
    # MUST be the first executable line: switch DataLoader workers to
    # spawn IPC, sidestepping the cluster's 4.18.0 fork-hang. Called
    # with force=True because some upstream import may have already
    # set a default. Idempotent: only invoked if not already 'spawn'.
    import multiprocessing as _mp
    if _mp.get_start_method(allow_none=True) != "spawn":
        _mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", choices=list(REGION_CONFIGS), default="mab",
        help="RegionConfig to train on (default mab). "
             "Drives store/stats paths, train/val split, crop, channels.",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume from a checkpoint dir (model+ema+optimizer+scheduler+epoch/"
             "step restored) — continues instead of restarting. Protects the "
             "multi-day run against walltime kills.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override total PRETRAIN epochs (pretrain-only; finetune stays 0)",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--zarr-path", type=str, default=None,
        help="Override the input Zarr path (e.g. for GPFS-staged copies).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="1+1 epochs on the first train year only; ~20 min wall on a single A40.",
    )
    parser.add_argument(
        "--bench-mode", action="store_true",
        help="Benchmark mode: 1 epoch × N steps, no val, no checkpoint; "
             "prints `BENCHMARK_SUMMARY` line at end. Use with --smoke-test.",
    )
    parser.add_argument(
        "--bench-steps", type=int, default=100,
        help="Number of training steps in --bench-mode (default 100).",
    )
    parser.add_argument(
        "--bench-label", type=str, default="bench",
        help="Short identifier added to BENCHMARK_SUMMARY for grepping.",
    )
    parser.add_argument(
        "--use-torch-compile", action="store_true",
        help="Wrap model with torch.compile(mode='default') before "
             "accelerator.prepare(). Adds ~1-2 min compile overhead.",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    parser.add_argument(
        "--dump-config", action="store_true",
        help="Print the fully-resolved config as JSON and exit (no side effects). "
             "Useful for diffing the per-region resolved configs before submit.",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Disable WandB logging entirely.",
    )
    args = parser.parse_args()

    config = dict(DEFAULT_CONFIG)
    # Region overlay: route store/stats paths + train/val split from the
    # RegionConfig. For --region mab these equal DEFAULT_CONFIG exactly
    # (DEFAULT_CONFIG is built from the MAB config), so default behavior is
    # unchanged; --region ccs swaps in the CCS store/stats/split. `region` is
    # stored as a string (JSON-safe for the config.json dump); the RegionConfig
    # object is re-looked-up in train().
    region_cfg = REGION_CONFIGS[args.region]
    config["region"] = args.region
    config["zarr_path"] = region_cfg.paths.preprocessed_zarr
    config["stats_json_path"] = region_cfg.paths.pixel_norm_stats
    config["train_year_range"] = tuple(region_cfg.timesplit.train_year_range)
    config["val_year_range"] = tuple(region_cfg.timesplit.val_year_range)
    # Per-region run dir (runs-root-relative, from the RegionConfig) so
    # checkpoints never collide across regions.
    config["output_dir"] = region_cfg.paths.dcae_run_dir
    if args.config:
        with open(args.config) as f:
            if args.config.endswith((".yaml", ".yml")):
                import yaml
                config.update(yaml.safe_load(f))
            else:
                config.update(json.load(f))
    if args.epochs is not None:
        # Pretrain-only lock: --epochs sets total PRETRAIN epochs; refine stays 0.
        config["pretrain_epochs"] = int(args.epochs)
        config["decoder_finetune_epochs"] = 0
    if args.lr is not None:
        config["lr_pretrain"] = args.lr
    if args.batch_size is not None:
        config["batch_size_per_device"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    if args.resume is not None:
        config["resume_from"] = args.resume
    if args.seed is not None:
        config["seed"] = args.seed
    if args.zarr_path is not None:
        config["zarr_path"] = args.zarr_path
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.wandb_run_name is not None:
        config["wandb_run_name"] = args.wandb_run_name
    if args.wandb_tags is not None:
        config["wandb_tags"] = args.wandb_tags
    if args.no_wandb:
        config["no_wandb"] = True
    if args.use_torch_compile:
        config["use_torch_compile"] = True

    if args.smoke_test:
        config["pretrain_epochs"] = 1
        config["decoder_finetune_epochs"] = 0   # pretrain-only lock (was 1)
        # first train/val year of the selected region (region-agnostic smoke)
        _y = config["train_year_range"][0]
        config["train_year_range"] = (_y, _y)
        config["val_year_range"] = (_y, _y)
        config["warmup_steps"] = 10
        config["checkpoint_every_n_epochs"] = 1
        config["val_every_n_epochs"] = 1
        config["val_max_batches"] = 5
        config["max_train_batches_per_epoch"] = 30
        config["log_every_n_steps"] = 5
        # Keep DEFAULT_CONFIG's num_workers=4 (with spawn start method)
        # so the smoke also exercises the spawn-based workers. Previous
        # smoke used num_workers=0 as a forks-hazard workaround; that's
        # no longer needed once spawn is in place.
        config["output_dir"] = str(Path(config["output_dir"]).parent / "dcae_smoketest")

    if args.bench_mode:
        # Single-epoch streamlined benchmark: no val, no checkpoint, no
        # finetune-phase transition. Stays on year 2007 like --smoke-test.
        config["pretrain_epochs"] = 1
        config["decoder_finetune_epochs"] = 0
        config["val_every_n_epochs"] = 99_999
        config["checkpoint_every_n_epochs"] = 99_999
        config["val_max_batches"] = 0
        config["warmup_steps"] = 10
        config["max_train_batches_per_epoch"] = args.bench_steps
        config["log_every_n_steps"] = 5
        config["train_year_range"] = (2007, 2007)
        config["val_year_range"] = (2007, 2007)
        config["bench_mode"] = True
        config["bench_label"] = args.bench_label
        config["output_dir"] = str(
            Path(config["output_dir"]).parent / f"bench_{args.bench_label}"
        )

    # Dump the FULLY-RESOLVED config and exit, before any side effect. This is
    # what region-config diffs compare across regions — it must see exactly what
    # train() would see, never a reimplementation of the overlay.
    if args.dump_config:
        print(json.dumps(config, indent=2, default=str))
        return

    # Fresh-start guard: a run WITHOUT --resume must not silently start at epoch 0
    # in a dir that already holds checkpoints (it would overwrite them / look like a
    # resume). Require an explicit --resume to continue, or a clean/absent output dir
    # to start fresh. (Resume runs set resume_from and skip this.)
    if not config.get("resume_from"):
        _out = Path(_resolve_path(config["output_dir"], root="runs"))
        _existing = sorted(_out.glob("checkpoint-epoch-*")) if _out.exists() else []
        if _existing:
            raise SystemExit(
                f"output_dir {_out} already has {len(_existing)} checkpoint(s) and "
                f"--resume was not passed. Pass --resume <checkpoint-dir> to continue "
                f"an interrupted run, or clear/rename the dir to start fresh.")

    train(config)


if __name__ == "__main__":
    main()
