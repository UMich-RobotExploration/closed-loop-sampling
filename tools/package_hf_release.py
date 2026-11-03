#!/usr/bin/env python
"""Assemble the two release repositories from the staging tree, with an upload
manifest. LOCAL ONLY: nothing here uploads anywhere.

  <staging>/model_repo/    converted safetensors checkpoints
                           (tools/convert_weights.py) + the model card
  <staging>/dataset_repo/  the artifacts layout (<staging>/artifacts/, staged
                           by datasets/export_*_bundle.py and
                           tools/stage_artifacts.py) + the dataset card

Files are hard-linked where the filesystem allows it (both repositories live
beside their sources), copied otherwise. upload_manifest.csv at the staging
root records every file of both repositories: repository, path, size, and
sha256 for files under 100 MB.

Usage:
  python tools/package_hf_release.py --staging <dir> \
      [--model-card release/model_card.md] [--dataset-card release/dataset_card.md]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HASH_CAP = 100 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_tree(src: Path, dst: Path):
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if q.exists():
            q.unlink()
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--staging", required=True)
    ap.add_argument("--model-card", default=str(REPO / "release/model_card.md"))
    ap.add_argument("--dataset-card", default=str(REPO / "release/dataset_card.md"))
    args = ap.parse_args()
    staging = Path(args.staging)
    model_src = staging / "model_repo"
    dataset_dst = staging / "dataset_repo"

    # model_repo is already in its release layout; add the card
    shutil.copy2(args.model_card, model_src / "README.md")
    # dataset_repo = artifacts layout + the card
    link_tree(staging / "artifacts", dataset_dst)
    shutil.copy2(args.dataset_card, dataset_dst / "README.md")

    rows, totals = [], {}
    for name, root in (("model", model_src), ("dataset", dataset_dst)):
        total = 0
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            size = p.stat().st_size
            total += size
            rows.append([name, str(p.relative_to(root)), size,
                         sha256(p) if size < HASH_CAP else ""])
        totals[name] = total
    out = staging / "upload_manifest.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["repository", "path", "size_bytes", "sha256"])
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} files; model {totals['model'] / 1e9:.2f} GB, "
          f"dataset {totals['dataset'] / 1e9:.2f} GB")
    print("PACKAGE_HF_RELEASE OK (local only; nothing uploaded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
