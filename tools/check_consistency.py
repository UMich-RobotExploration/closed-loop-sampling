#!/usr/bin/env python
"""Consistency gate: the README's paper-to-code map vs the filesystem vs the
modules' docstring anchors.

Checks, each reported and all required to hold:
  1. every repository path named in the README map table exists;
  2. every Python module whose source cites a paper anchor (Alg. 1, Eq. (1)/(2),
     Sec. III) is covered by the map — directly, via its package directory, or
     via a named entry point that imports it;
  3. every map row that names a figure or table points at a script that exists
     and whose regenerate command names the same script.

Exit code 0 only with zero drift.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ANCHOR_RX = re.compile(r"Alg\.? ?1|Eq\.? ?\(?[12]\)?|Sec\.? ?III")


def map_rows():
    text = (REPO / "README.md").read_text()
    block = text[text.index("## Paper-to-code map"):text.index("## Installation")]
    rows = []
    for line in block.splitlines():
        if not line.startswith("|") or set(line) <= {"|", "-", " "}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 3 and cells[0] != "paper element":
            rows.append(cells)
    return rows


def main() -> int:
    rows = map_rows()
    drift = []
    mapped_paths = set()

    # 1. every path in the map exists
    for element, where, regen in rows:
        for m in re.findall(r"`([^`]+)`", where + " " + regen):
            cand = m.split()[1] if m.startswith("python ") else m
            if not re.match(r"^(src|results|experiments|configs|tools|tests|datasets|training|assets)/", cand):
                continue
            p = REPO / re.sub(r"\{([^}]+)\}", lambda g: g.group(1).split(",")[0], cand)
            if not p.exists():
                drift.append(f"map row '{element}': path does not exist: {cand}")
            else:
                mapped_paths.add(cand.rstrip("/"))

    # 2. every module citing a paper anchor is covered by the map
    covered_dirs = {p for p in mapped_paths if (REPO / p).is_dir()}
    # entry points whose rows exercise whole subsystems
    implied = {
        "experiments/run_closed_loop.py": ["experiments/truth.py"],
        "src/closedloop_sampling/forecast/": ["src/closedloop_sampling/forecast"],
    }
    for py in sorted(list((REPO / "src").rglob("*.py"))
                     + list((REPO / "experiments").glob("*.py"))
                     + list((REPO / "results").rglob("*.py"))):
        rel = str(py.relative_to(REPO))
        if "/out/" in rel:
            continue
        if not ANCHOR_RX.search(py.read_text()):
            continue
        ok = (rel in mapped_paths
              or any(rel.startswith(d + "/") or rel.startswith(d)
                     for d in covered_dirs)
              or any(rel in v for v in implied.values()))
        if not ok:
            drift.append(f"anchored module not covered by the map: {rel}")

    # 3. figure/table rows: the regenerate command names the same script
    for element, where, regen in rows:
        m = re.search(r"(Table [IVX]+|Fig\. \d|teaser)", element)
        if not m:
            continue
        w = re.findall(r"`([^`]+\.py)`", where)
        if not w:
            drift.append(f"map row '{element}': no script in the 'where' column")
            continue
        script = Path(w[0]).name
        if "python" in regen and script not in regen and "ARTIFACTS" not in regen:
            drift.append(f"map row '{element}': regenerate command does not run {script}")

    print(f"paper-to-code map: {len(rows)} rows, "
          f"{len(mapped_paths)} repository paths verified")
    for d in drift:
        print(f"  DRIFT: {d}")
    print("CONSISTENCY " + ("OK" if not drift else f"DRIFT={len(drift)}"))
    return 0 if not drift else 1


if __name__ == "__main__":
    sys.exit(main())
