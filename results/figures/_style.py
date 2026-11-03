"""Shared style and input resolution for the paper figure scripts.

One source of truth for the policy order, labels, markers, colour ramps and
rcParams the figures share, plus the two input roots:
  scores_dir()     the committed scores release (results/scores/)
  artifacts_dir()  $CLOSEDLOOP_ARTIFACTS, the downloaded dataset repository
                   (decoded figure fields, render dumps, member caches)
Figures 3-5 resolve everything from scores_dir(); Figs. 1, 2, 6 and the teaser
video need artifacts_dir().
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1]
if str(RESULTS) not in sys.path:
    sys.path.insert(0, str(RESULTS))

POLICIES = ["static", "random", "lawnmower", "d_optimal", "a_optimal"]
LABEL = {"static": "Static", "random": "Random", "lawnmower": "Lawnmower",
         "d_optimal": "D-optimal", "a_optimal": "A-optimal"}
MARK = {"static": "s", "random": "o", "lawnmower": "^", "d_optimal": "D",
        "a_optimal": "v"}
REGION_LABEL = {"mab": "MAB", "ccs": "CCS"}
VARIABLE_LABEL = {"sst": "SST", "sss": "SSS"}

CLIP = (0.12, 0.82)                 # colormap span for the five-policy ramp
FREE, FLOOR_C, SHADE = "0.25", "0.05", "0.88"

BASE_RC = {"pdf.fonttype": 42, "font.family": "STIXGeneral",
           "mathtext.fontset": "stix", "axes.linewidth": 0.7,
           "axes.edgecolor": "0.45", "xtick.color": "0.45", "ytick.color": "0.45",
           "xtick.labelcolor": "black", "ytick.labelcolor": "black",
           "legend.frameon": False}


def variable_cmap(variable: str):
    import cmocean
    return {"sst": cmocean.cm.thermal, "sss": cmocean.cm.haline}[variable]


def policy_colors(cmap):
    return {p: cmap(np.linspace(CLIP[0], CLIP[1], len(POLICIES))[i])
            for i, p in enumerate(POLICIES)}


def scores_dir() -> Path:
    return RESULTS / "scores"


def out_dir(explicit: str | None = None) -> Path:
    out = Path(explicit) if explicit else RESULTS / "out"
    out.mkdir(parents=True, exist_ok=True)
    return out


def artifacts_dir() -> Path:
    root = os.environ.get("CLOSEDLOOP_ARTIFACTS")
    if not root:
        raise SystemExit(
            "CLOSEDLOOP_ARTIFACTS is not set. This figure draws decoded fields "
            "from the artifacts dataset: download it with "
            "tools/download_artifacts.py and export "
            "CLOSEDLOOP_ARTIFACTS=/path/to/artifacts.")
    p = Path(root)
    if not p.is_dir():
        raise SystemExit(f"CLOSEDLOOP_ARTIFACTS={root} is not a directory")
    return p


def argval(flag: str, default: str) -> str:
    return ([a.split("=", 1)[1] for a in sys.argv if a.startswith(flag + "=")]
            or [default])[0]
