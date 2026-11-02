"""Filesystem roots resolved from environment variables — the single place
site-specific paths enter the package.

  CLOSEDLOOP_DATA_ROOT : datasets (raw staging, preprocessed stores, norm
                         stats, latents, truth bundles) — see README
  CLOSEDLOOP_WEIGHTS   : released checkpoints (autoencoder + denoiser per
                         region, EMA files)
  CLOSEDLOOP_RUNS      : run outputs (per-chain records, dumps, training runs)
  CLOSEDLOOP_REGIONS   : optional override for the region-config directory;
                         defaults to configs/regions/ in this repository

Region-config path fields are RELATIVE; resolve them with :func:`resolve`.
"""
import os
from pathlib import Path

_HELP = {
    "CLOSEDLOOP_DATA_ROOT": "directory holding per-region data (see README 'Data layout')",
    "CLOSEDLOOP_WEIGHTS": "directory holding the released checkpoints",
    "CLOSEDLOOP_RUNS": "directory where chain records and training outputs are written",
}


def _env(name):
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"{name} is not set. Export it before running, e.g.\n"
            f"    export {name}=/path/to/dir    # {_HELP[name]}"
        )
    return Path(val)


def data_root() -> Path:
    return _env("CLOSEDLOOP_DATA_ROOT")


def weights_root() -> Path:
    return _env("CLOSEDLOOP_WEIGHTS")


def runs_root() -> Path:
    return _env("CLOSEDLOOP_RUNS")


def regions_dir() -> Path:
    """Directory holding the region YAMLs (mab.yaml, ccs.yaml)."""
    override = os.environ.get("CLOSEDLOOP_REGIONS")
    if override:
        p = Path(override)
        if not p.is_dir():
            raise RuntimeError(
                f"CLOSEDLOOP_REGIONS={override} is not a directory"
            )
        return p
    repo_default = Path(__file__).resolve().parents[2] / "configs" / "regions"
    if repo_default.is_dir():
        return repo_default
    raise RuntimeError(
        "region-config directory not found. Either run from a source checkout "
        "(configs/regions/) or export CLOSEDLOOP_REGIONS=/path/to/region/yamls"
    )


def resolve(relative: str, root: str = "data") -> str:
    """Resolve a region-config relative path against a filesystem root.

    root: "data" (default), "weights", or "runs".
    """
    base = {"data": data_root, "weights": weights_root, "runs": runs_root}[root]()
    return str(base / relative)
