#!/usr/bin/env python
"""Download the released model weights into $CLOSEDLOOP_WEIGHTS.

Pulls the model repository named in configs/release.yaml (safetensors EMA
checkpoints, meta.json and training configs for both regions, plus the seed-1
extras) with huggingface_hub. The download lands in the release layout the
loaders resolve:
    <weights>/<region>/dcae/ema.safetensors
    <weights>/<region>/denoiser/seed0/ema.safetensors

Usage: python tools/download_weights.py [--dest <dir>] [--revision <rev>]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=None,
                    help="target directory (default: $CLOSEDLOOP_WEIGHTS)")
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()
    dest = args.dest or os.environ.get("CLOSEDLOOP_WEIGHTS")
    if not dest:
        raise SystemExit("set CLOSEDLOOP_WEIGHTS or pass --dest")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is not installed "
                         "(pip install huggingface_hub)") from None
    repo_id = yaml.safe_load(open(REPO / "configs/release.yaml"))[
        "huggingface"]["model_repo"]
    path = snapshot_download(repo_id=repo_id, revision=args.revision,
                             local_dir=dest)
    print(f"downloaded {repo_id} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
