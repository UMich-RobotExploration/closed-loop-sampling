"""Equivalence gate: this package reproduces the production forecaster
numerics against the released EMA checkpoints.

Probes (references in tests/references/{mab,ccs}.json):
  A        — MAB denoiser: one edm_denoise step at sigma 1.0 on seed-0 inputs;
             exact param count 374,938,452; stats within atol/rtol 1e-5.
  B        — MAB autoencoder: decode a seed-0 latent, encode a seed-1 field;
             exact param count 256,388,460; stats within tolerance.
  ccs      — CCS denoiser: one edm_denoise step on a (1,84,4,22,23) latent.

Checkpoint locations (dirs containing ema.safetensors or ema.pt), via environment:
  CLOSEDLOOP_EQ_DCAE          MAB autoencoder checkpoint dir
  CLOSEDLOOP_EQ_DENOISER_MAB  MAB denoiser (seed 0) checkpoint dir
  CLOSEDLOOP_EQ_DENOISER_CCS  CCS denoiser (seed 0) checkpoint dir
or the release layout under $CLOSEDLOOP_WEIGHTS ({mab,ccs}/dcae,
{mab,ccs}/denoiser/seed0). Tests skip when neither is available.

Run with 2 torch threads for byte-similar numbers (the tolerance comparison is
the contract; the recorded rounded-array hashes are BLAS-threading sensitive).
Marker: weights. Standalone: python tests/test_equivalence.py
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path

import numpy as np

try:
    import pytest
    pytestmark = pytest.mark.weights

    def _skip(msg: str):
        pytest.skip(msg)
except ImportError:                      # standalone runner without pytest
    pytest = None

    class _SkipProbe(Exception):
        pass

    def _skip(msg: str):
        raise _SkipProbe(msg)

try:
    import torch
except ImportError:
    if pytest is not None:               # collection in a torch-free env
        pytest.skip("torch not installed", allow_module_level=True)
    raise

torch.set_num_threads(2)

REFS_DIR = Path(__file__).parent / "references"
REF_MAB = json.load(open(REFS_DIR / "mab.json"))
REF_CCS = json.load(open(REFS_DIR / "ccs.json"))
ATOL = RTOL = 1e-5


def _ckpt_dir(env_name: str, release_rel: str):
    explicit = os.environ.get(env_name)
    if explicit:
        return Path(explicit)
    weights = os.environ.get("CLOSEDLOOP_WEIGHTS")
    if weights and any((Path(weights) / release_rel / f).exists()
                       for f in ("ema.safetensors", "ema.pt")):
        return Path(weights) / release_rel
    _skip(f"{env_name} not set and no release layout under CLOSEDLOOP_WEIGHTS")


def _stats(a: np.ndarray) -> dict:
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min": float(a.min()), "max": float(a.max()),
            "first5": [float(v) for v in a.flat[:5]],
            "sha256_round_1e4": hashlib.sha256(np.round(a, 4).tobytes()).hexdigest()}


def _assert_close(got: dict, ref: dict, label: str):
    for key in ("mean", "std", "min", "max", "first5"):
        if key not in ref:
            continue
        rv, gv = ref[key], got[key]
        if isinstance(rv, list):
            ok = np.allclose(gv, rv, atol=ATOL, rtol=RTOL)
        else:
            ok = np.isclose(gv, rv, atol=ATOL, rtol=RTOL)
        assert ok, f"{label}.{key}: got {gv} vs reference {rv}"
    if got["sha256_round_1e4"] == ref["sha256_round_1e4"]:
        print(f"  {label}: stats within 1e-5 and rounded-array hash identical")
    else:
        print(f"  {label}: stats within 1e-5; rounded-array hash differs "
              f"(BLAS-threading sensitive, informational only)")


def _load_ema(model, ckpt_dir: Path):
    """Directory-level EMA load: exercises whichever format the checkpoint dir
    holds (ema.safetensors preferred, ema.pt otherwise), so pointing the env
    dirs at converted safetensors checkpoints runs the safetensors mode of
    this same suite."""
    from closedloop_sampling.forecast.checkpoints import load_ema_weights
    used = load_ema_weights(model, Path(ckpt_dir))
    print(f"  loaded {used} from {ckpt_dir}")
    return model


def test_probe_a_denoiser_mab():
    from closedloop_sampling.forecast.edm import edm_denoise
    from closedloop_sampling.forecast.models import make_denoiser

    ckpt = _ckpt_dir("CLOSEDLOOP_EQ_DENOISER_MAB", "mab/denoiser/seed0")
    model = make_denoiser().eval()
    n = sum(p.numel() for p in model.parameters())
    assert n == REF_MAB["denoiser_params"], f"denoiser params {n:,}"
    _load_ema(model, ckpt)
    torch.manual_seed(0)
    noisy = torch.randn(1, 84, 4, 13, 30)
    cond = torch.randn(1, 84, 1, 13, 30)
    with torch.no_grad():
        x0 = edm_denoise(model, noisy, torch.tensor(1.0), cond, sigma_data=0.5)
    _assert_close(_stats(x0.numpy().astype(np.float32)),
                  REF_MAB["probe_A_denoiser"], "A")
    del model
    gc.collect()


def test_probe_b_autoencoder_mab():
    from closedloop_sampling.forecast.models import RegionalDCAE

    ckpt = _ckpt_dir("CLOSEDLOOP_EQ_DCAE", "mab/dcae")
    dcae = RegionalDCAE().eval()
    n = sum(p.numel() for p in dcae.parameters())
    assert n == REF_MAB["autoencoder_params"], f"autoencoder params {n:,}"
    _load_ema(dcae, ckpt)

    torch.manual_seed(0)
    lat = torch.randn(1, 84, 13, 30)
    with torch.no_grad():
        rec = dcae.decode(lat)
    _assert_close(_stats(rec.numpy().astype(np.float32)),
                  REF_MAB["probe_B_decode"], "B.decode")

    torch.manual_seed(1)
    x = torch.randn(1, 83, 104, 240)
    static = torch.ones(1, 1, 104, 240)
    with torch.no_grad():
        z = dcae.encode(x, static)
    got = _stats(z.numpy().astype(np.float32))
    _assert_close({k: got[k] for k in ("mean", "std", "sha256_round_1e4")},
                  REF_MAB["probe_B_encode"], "B.encode")
    del dcae
    gc.collect()


def test_probe_ccs_denoiser():
    from closedloop_sampling.forecast.edm import edm_denoise
    from closedloop_sampling.forecast.models import make_denoiser
    from closedloop_sampling.regions import load as load_region

    ckpt = _ckpt_dir("CLOSEDLOOP_EQ_DENOISER_CCS", "ccs/denoiser/seed0")
    model = make_denoiser(region_config=load_region("ccs")).eval()
    n = sum(p.numel() for p in model.parameters())
    assert n == REF_CCS["denoiser_params"], f"ccs denoiser params {n:,}"
    _load_ema(model, ckpt)
    torch.manual_seed(0)
    noisy = torch.randn(1, 84, 4, 22, 23)
    cond = torch.randn(1, 84, 1, 22, 23)
    with torch.no_grad():
        x0 = edm_denoise(model, noisy, torch.tensor(1.0), cond, sigma_data=0.5)
    _assert_close(_stats(x0.numpy().astype(np.float32)),
                  REF_CCS["probe_ccs_denoiser"], "ccs")
    del model
    gc.collect()


if __name__ == "__main__":
    # Standalone runner (works without pytest): run the three probes in order.
    failures = []
    skipped = []
    for fn in (test_probe_a_denoiser_mab, test_probe_b_autoencoder_mab,
               test_probe_ccs_denoiser):
        try:
            fn()
            print(f"{fn.__name__}: PASS")
        except AssertionError as e:
            print(f"{fn.__name__}: FAIL ({e})")
            failures.append(fn.__name__)
        except Exception as e:                        # skip signal from _skip()
            if type(e).__name__ in ("_SkipProbe", "Skipped"):
                print(f"{fn.__name__}: SKIP ({e})")
                skipped.append(fn.__name__)
            else:
                raise
    print("EQUIVALENCE:", "FAIL" if failures else
          ("PASS" if not skipped else f"PASS (skipped: {skipped})"))
    raise SystemExit(1 if failures else 0)
