"""Run-to-run tolerance floors of the released campaigns.

floor(region, variable, fleet, window) is the maximum, over all replicate
chains and sealed ICs of that slice, of |score(replicate) - score(registered
A-optimal seed-0 chain)| on the window mean. The floors were measured on 144
replicate chains (three early replicate sets of 8 A-optimal seed-0 chains each,
plus a full replicate grid of 120 chains covering every region, variable and
main fleet size on all 10 sealed ICs), recomputed from those chains and
asserted against the production ledger by tools/extract_scores.py, and released
in references/tolerance.csv. No replicate chains exist at the sweep-only fleet
sizes (n = 10/35/100); claims there carry paired tests only.
"""
from __future__ import annotations

import csv
from pathlib import Path


def load_floors(root: Path) -> dict:
    """tolerance.csv -> {(region, variable, fleet): {"last8": f, "last6": f}}."""
    out = {}
    with open(Path(root) / "references/tolerance.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["region"], r["variable"], int(r["fleet"]))] = {
                "last8": float(r["floor_last8"]), "last6": float(r["floor_last6"])}
    return out


def floor_for(floors: dict, region: str, variable: str, fleet: int,
              window: str = "last8") -> float:
    return floors[(region, variable, fleet)][window]
