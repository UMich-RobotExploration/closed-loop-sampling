"""Shared plumbing for the table scripts: path resolution and formatting.

Every table script reads ONLY results/scores/ and writes a CSV (full-precision
values plus the printed strings) and the paper's LaTeX table body. The
committed goldens live in results/tables/expected/; regeneration is tested
against them byte for byte.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1]
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

REGION_LABEL = {"mab": "MAB", "ccs": "CCS"}
POLICY_LABEL = {"static": "Static", "random": "Random", "lawnmower": "Lawnmower",
                "d_optimal": "D-optimal", "a_optimal": "A-optimal"}


def parse_args(name: str):
    ap = argparse.ArgumentParser(description=f"Regenerate {name} from results/scores/.")
    ap.add_argument("--scores", default=None, help="scores directory "
                    "(default: <repo>/results/scores)")
    ap.add_argument("--out", default=None, help="output directory "
                    "(default: <repo>/results/out/tables)")
    a = ap.parse_args()
    scores = Path(a.scores) if a.scores else RESULTS / "scores"
    out = Path(a.out) if a.out else RESULTS / "out/tables"
    out.mkdir(parents=True, exist_ok=True)
    return scores, out


def write_csv(path: Path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def write_tex(path: Path, lines):
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def fnum(x) -> str:
    """Full-precision decimal text for the CSV columns."""
    return repr(float(x))


def paper_delta(x: float) -> str:
    """Table II's printed form: three decimals, no leading zero, $-$ sign."""
    s = f"{abs(x):.3f}"[1:]
    return f"$-{s}$" if x < 0 else f"${s}$"


def paper_r(x: float) -> str:
    """Table II's printed r: two decimals, always signed."""
    return f"$+{x:.2f}$" if x >= 0 else f"$-{abs(x):.2f}$"
