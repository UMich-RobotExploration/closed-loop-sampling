#!/usr/bin/env python
"""Download the released artifacts dataset into $CLOSEDLOOP_ARTIFACTS.

Pulls the dataset repository named in configs/release.yaml: the truth and
latent bundles (the closed loop runs from these alone), the decoded figure
fields, the render dumps, the free-running latents and the raw run records.
Pass --bundles-only for the two-bundle subset the loop needs (about 320 MB
instead of 2.3 GB).

Usage: python tools/download_artifacts.py [--dest <dir>] [--bundles-only]
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
                    help="target directory (default: $CLOSEDLOOP_ARTIFACTS)")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--bundles-only", action="store_true",
                    help="only truth_bundles/ and latent_bundles/")
    args = ap.parse_args()
    dest = args.dest or os.environ.get("CLOSEDLOOP_ARTIFACTS")
    if not dest:
        raise SystemExit("set CLOSEDLOOP_ARTIFACTS or pass --dest")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is not installed "
                         "(pip install huggingface_hub)") from None
    repo_id = yaml.safe_load(open(REPO / "configs/release.yaml"))[
        "huggingface"]["dataset_repo"]
    patterns = (["truth_bundles/*", "latent_bundles/*"]
                if args.bundles_only else None)
    path = snapshot_download(repo_id=repo_id, repo_type="dataset",
                             revision=args.revision, local_dir=dest,
                             allow_patterns=patterns)
    print(f"downloaded {repo_id} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
