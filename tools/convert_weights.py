#!/usr/bin/env python
"""Convert the six production EMA checkpoints to safetensors for release.

For each released checkpoint the EMA shadow parameters (ema.pt, EMAModel state,
tensors in parameter order) are re-keyed by the model's parameter names and
written as ema.safetensors, with a meta.json beside it (epoch, validation loss,
parameter count, source run) and the as-run training configuration copied from
configs/training/. The region configuration rides at <region>/region.yaml.

Output layout (the model repository):
  <out>/mab/dcae/{ema.safetensors, meta.json, training_config.yaml}
  <out>/mab/denoiser/seed0/{...}
  <out>/mab/region.yaml
  <out>/ccs/{dcae, denoiser/seed0, region.yaml}
  <out>/extras/denoiser_seed1_{mab,ccs}/{...}   (trained identically with
        seed 1; not used in any paper result)

Verification: every tensor is checked bitwise against the shadow parameters
after the round trip, and the equivalence suite run with the env checkpoint
dirs pointed here reproduces the reference probe hashes through the
safetensors path.

Usage:
  python tools/convert_weights.py --runs-root <production runs dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from closedloop_sampling.forecast.models import RegionalDCAE, make_denoiser  # noqa: E402
from closedloop_sampling.regions import load as load_region  # noqa: E402

# (release subdir, kind, region, production run dir, training yaml, note)
CHECKPOINTS = [
    ("mab/dcae", "dcae", "mab", "dcae_pilot0", "mab/autoencoder.yaml", ""),
    ("mab/denoiser/seed0", "denoiser", "mab", "denoiser_unconditioned/seed_000",
     "mab/denoiser.yaml", ""),
    ("ccs/dcae", "dcae", "ccs", "dcae_ccs_pilot0", "ccs/autoencoder.yaml", ""),
    ("ccs/denoiser/seed0", "denoiser", "ccs",
     "denoiser_ccs_unconditioned/seed_000", "ccs/denoiser.yaml", ""),
    ("extras/denoiser_seed1_mab", "denoiser", "mab",
     "denoiser_unconditioned/seed_001", "mab/denoiser.yaml",
     "trained identically with seed 1; not used in any paper result"),
    ("extras/denoiser_seed1_ccs", "denoiser", "ccs",
     "denoiser_ccs_unconditioned/seed_001", "ccs/denoiser.yaml",
     "trained identically with seed 1; not used in any paper result"),
]


def build_model(kind: str, region: str):
    cfg = load_region(region)
    return RegionalDCAE(region_config=cfg) if kind == "dcae" \
        else make_denoiser(region_config=cfg)


def convert_one(run_dir: Path, kind: str, region: str, dst: Path,
                training_yaml: Path, note: str):
    ckpt = run_dir / "checkpoint-best"
    state = torch.load(ckpt / "ema.pt", map_location="cpu", weights_only=False)
    shadow = state["shadow_params"]
    model = build_model(kind, region)
    names = [n for n, _ in model.named_parameters()]
    assert len(names) == len(shadow), (
        f"{ckpt}: {len(shadow)} shadow tensors vs {len(names)} model parameters")
    tensors = {n: sp.data.clone().contiguous() for n, sp in zip(names, shadow)}
    n_params = sum(t.numel() for t in tensors.values())

    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "ema.safetensors"
    save_file(tensors, str(out),
              metadata={"format": "pt",
                        "contents": "EMA weights keyed by parameter name",
                        "source_run": f"{run_dir.name}/checkpoint-best"})
    back = load_file(str(out))
    for n in names:
        assert torch.equal(back[n], tensors[n]), f"round-trip mismatch at {n}"

    best = json.load(open(ckpt / "meta.json"))
    meta = dict(kind={"dcae": "autoencoder", "denoiser": "denoiser"}[kind],
                region=region, epoch=best["epoch"], val_loss=best["val_loss"],
                param_count=n_params,
                source_run=f"{run_dir.name}/checkpoint-best",
                ema=True, format="safetensors")
    if note:
        meta["note"] = note
    json.dump(meta, open(dst / "meta.json", "w"), indent=1)
    shutil.copy2(training_yaml, dst / "training_config.yaml")
    print(f"  {dst}: {n_params:,} parameters, epoch {best['epoch']}, "
          f"val_loss {best['val_loss']:.6g}, "
          f"{out.stat().st_size / 1e6:.0f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-root", required=True,
                    help="production training-runs directory (read-only)")
    ap.add_argument("--out", required=True, help="model repository root")
    args = ap.parse_args()
    runs = Path(args.runs_root)
    out = Path(args.out)

    for rel, kind, region, run, yml, note in CHECKPOINTS:
        convert_one(runs / run, kind, region, out / rel,
                    REPO / "configs/training" / yml, note)
    for region in ("mab", "ccs"):
        (out / region).mkdir(exist_ok=True)
        shutil.copy2(REPO / f"configs/regions/{region}.yaml",
                     out / region / "region.yaml")
    (out / "extras" / "README.md").write_text(
        "# Extras\n\nSeed-1 denoisers for both regions, trained identically to "
        "the released seed-0\ncheckpoints; not used in any paper result.\n")
    print("CONVERT_WEIGHTS OK")


if __name__ == "__main__":
    main()
