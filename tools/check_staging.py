#!/usr/bin/env python
"""Identifier gate over a staged release tree.

Applies the same banned patterns as tools/check_identifiers.py to everything a
staged repository exposes: file PATHS, full text files (json/md/yaml/txt/csv),
npz member names and string-typed values, and safetensors metadata headers.
Array payloads are numeric and are not scanned.

Usage: python tools/check_staging.py --root <staged tree> [--root ...]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from check_identifiers import PATTERNS  # noqa: E402  (base64-decoded at import)

TEXT_SUFFIXES = {".json", ".md", ".yaml", ".yml", ".txt", ".csv", ".cff", ".py"}


def hits(text: str, where: str, found: list):
    for pat in PATTERNS:
        if pat.search(text):
            found.append(f"{where}: matches banned pattern {pat.pattern!r}")


def scan_npz(path: Path, found: list):
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
    except (zipfile.BadZipFile, OSError):
        found.append(f"{path}: unreadable npz")
        return
    hits(" ".join(names), f"{path} (member names)", found)
    import numpy as np
    z = np.load(path, allow_pickle=True)
    for k in z.files:
        a = z[k]
        if a.dtype.kind in ("U", "S", "O"):
            hits(str(a.tolist()), f"{path}:{k}", found)


def scan_safetensors(path: Path, found: list):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = fh.read(min(n, 10 * 1024 * 1024)).decode("utf-8", "replace")
    hits(header, f"{path} (safetensors header)", found)
    json.loads(header)          # the header must be valid JSON


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", action="append", required=True)
    args = ap.parse_args()
    found: list = []
    n_files = 0
    for root in args.root:
        root = Path(root)
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            n_files += 1
            rel = str(p.relative_to(root.parent))
            hits(rel, f"path {rel}", found)
            if p.suffix in TEXT_SUFFIXES:
                hits(p.read_text(errors="replace"), rel, found)
            elif p.suffix == ".npz":
                scan_npz(p, found)
            elif p.suffix == ".safetensors":
                scan_safetensors(p, found)
    for f in found:
        print(f"  HIT: {f}")
    print(f"STAGING IDENTIFIER GATE: {n_files} files scanned, "
          + ("clean" if not found else f"{len(found)} hits"))
    return 0 if not found else 1


if __name__ == "__main__":
    sys.exit(main())
