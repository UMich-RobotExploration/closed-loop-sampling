#!/usr/bin/env python
"""Fig. 3: fraction of recoverable error removed, 2 x 2 (SST row over SSS row,
MAB and CCS columns), five policies per fleet size with mean +- 1 sd across the
10 sealed ICs.

The skill fraction is formed per IC with that IC's free-running and
autoencoder-floor window means (results/scoring/metrics.skill_fraction), then
averaged; the final-eight window is the paper window. Inputs are the committed
scores release only (results/scores/); no decoded fields are needed.

Layout: the panel SHAPE is held fixed and the width is the free parameter;
--figw=3.5 (default) is the IEEE single column. --legend in
{single, per-row, right}.
"""
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _style import (BASE_RC, CLIP, FLOOR_C, FREE, LABEL, MARK, POLICIES,
                    REGION_LABEL, argval, out_dir, scores_dir, variable_cmap)

from scoring import metrics as M

SCORES = scores_dir()
OUT = out_dir(argval("--out", ""))
ARMS = POLICIES
REGIONS = ["mab", "ccs"]
FLEETS = [20, 50, 200]
GRID = [("SST", REGIONS), ("SSS", REGIONS)]        # temperature on top
MS, MEW = 3.6, 0.5                 # one source of truth for plotted AND legend markers
FIG_W = float(argval("--figw", "3.5"))
LEGEND = argval("--legend", "single")
assert LEGEND in ("single", "per-row", "right")
plt.rcParams.update({**BASE_RC, "font.size": 7, "axes.titlesize": 8})

# ---- the quantity: per-IC skill fraction, mean +- sd across ICs --------------------
runs = M.load_runs(SCORES)
free = M.load_free_running(SCORES)
floor = M.load_ae_floor(SCORES)
FR = {}
for region in REGIONS:
    FR[region] = {}
    for vn, variable in (("SST", "sst"), ("SSS", "sss")):
        for k in FLEETS:
            for a in ARMS:
                s = M.skill_fraction(runs, free, floor, region, variable, k, a)
                v = np.array(list(s.values()))
                assert len(v) == 10
                FR[region][f"{vn} n={k}"] = FR[region].get(f"{vn} n={k}", {})
                FR[region][f"{vn} n={k}"][a] = dict(mean=float(v.mean()),
                                                    sd=float(v.std(ddof=1)))

# ---- geometry: hold the sibling panel SHAPE, let the width set the size ------------
BASE_LEFT_IN, RIGHT_IN, WSPACE = 0.058 * 7.16, (1 - 0.995) * 7.16, 0.20
YLAB_FS, ROWLAB_FS, GAP_IN = 7, 8, 0.055
YTEXT = "recoverable error removed (final eight cycles)"
LINE_IN = lambda fs: fs * 1.25 / 72
SIB_W = (0.995 - 0.058) * 7.16 / (4 + 3 * WSPACE)
SIB_ASPECT = SIB_W / ((0.915 - 0.315) * 2.35)
LEG_FS = 6.2 if FIG_W > 5 else 5.6
HANDLE_L, HANDLE_PAD = 1.1, 0.5
# Legend handle dashes solved so the entry is exactly two dashes with one gap.
_HANDLE_PT = HANDLE_L * LEG_FS
LEG_DASH = (0.36 * _HANDLE_PT, 0.28 * _HANDLE_PT)

_fm = plt.figure(figsize=(FIG_W, 2.0))
_rm = _fm.canvas.get_renderer()


def _tw(t, fs):
    o = _fm.text(0.5, 0.5, t, fontsize=fs)
    w = o.get_window_extent(renderer=_rm).width / _fm.dpi
    o.remove()
    return w


_LEG_TEXT = ["free-running", "autoencoder floor"] + [LABEL[a] for a in ARMS]
LEGCOL_IN = (max(_tw(t, LEG_FS) for t in _LEG_TEXT)
             + (HANDLE_L + HANDLE_PAD) * LEG_FS / 72 + 0.115) if LEGEND == "right" else 0.0
TITLE_H, XLAB_H, LEG_H, PAD = 0.22, 0.30, 0.34, 0.04
ROW_TITLE_H = [TITLE_H, 0.05]
ROW_XLAB_H = [0.05, XLAB_H]
N_LEG_BANDS = {"per-row": 2, "single": 1, "right": 0}[LEGEND]


def _layout(nlines):
    left_in = BASE_LEFT_IN + GAP_IN + nlines * LINE_IN(YLAB_FS) + GAP_IN + LINE_IN(ROWLAB_FS)
    ax_w = (FIG_W - left_in - RIGHT_IN) / (2 + 1 * WSPACE)
    fig_wd = FIG_W + LEGCOL_IN
    return (left_in, left_in / fig_wd, (left_in + (2 + 1 * WSPACE) * ax_w) / fig_wd,
            ax_w, ax_w / SIB_ASPECT, fig_wd)


LEFT_IN, LEFT, RIGHT, AX_W, AX_H, FIG_WD = _layout(2)
GRID_H = 2 * AX_H + ROW_XLAB_H[0] + ROW_TITLE_H[1]
NLINES = 1 if _tw(YTEXT, YLAB_FS) <= GRID_H - 0.08 else 2
if NLINES == 1:
    LEFT_IN, LEFT, RIGHT, AX_W, AX_H, FIG_WD = _layout(1)
plt.close(_fm)
YLABEL = YTEXT if NLINES == 1 else YTEXT.replace(" (final", "\n(final")
FIG_H = sum(ROW_TITLE_H) + 2 * AX_H + sum(ROW_XLAB_H) + N_LEG_BANDS * LEG_H + 2 * PAD
print(f"figure {FIG_WD:.3f} x {FIG_H:.3f} in; panel {AX_W:.3f} x {AX_H:.3f} in "
      f"(aspect {AX_W / AX_H:.3f}; sibling aspect {SIB_ASPECT:.3f})")

fig = plt.figure(figsize=(FIG_WD, FIG_H))
ROWLAB = []
y = FIG_H - PAD
for ri, (vn, regs) in enumerate(GRID):
    cmap = variable_cmap(vn.lower())
    cols = {a: cmap(np.linspace(CLIP[0], CLIP[1], len(ARMS))[i])
            for i, a in enumerate(ARMS)}
    top = y - ROW_TITLE_H[ri]
    gs = fig.add_gridspec(1, 2, left=LEFT, right=RIGHT, wspace=WSPACE,
                          top=top / FIG_H, bottom=(top - AX_H) / FIG_H)
    for ci, reg in enumerate(regs):
        ax = fig.add_subplot(gs[0, ci])
        xs = np.arange(len(FLEETS))
        off = np.linspace(-0.26, 0.26, len(ARMS))
        for i, a in enumerate(ARMS):
            m = np.array([FR[reg][f"{vn} n={k}"][a]["mean"] for k in FLEETS])
            s = np.array([FR[reg][f"{vn} n={k}"][a]["sd"] for k in FLEETS])
            ax.errorbar(xs + off[i], m, yerr=s, fmt=MARK[a], ms=MS, color=cols[a],
                        ecolor=cols[a], elinewidth=0.9, capsize=1.8, lw=0, mew=MEW,
                        mfc=cols[a], zorder=3)
        ax.axhline(1.0, color=FLOOR_C, ls="--", dashes=(3.4, 2.2), lw=1.05, zorder=4)
        ax.axhline(0.0, color=FREE, ls="-", lw=1.25, zorder=4)
        if ci == 0 and ri == 0:        # identical lines in all four: name them once
            ax.text(len(FLEETS) - 0.55, 0.975, "autoencoder floor", fontsize=5.8,
                    color="0.30", ha="right", va="top", zorder=5)
            ax.text(len(FLEETS) - 0.55, 0.025, "free-running", fontsize=5.8,
                    color="0.30", ha="right", va="bottom", zorder=5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"$n{{=}}{k}$" for k in FLEETS]
                           if ri == len(GRID) - 1 else [])
        ax.set_xlim(-0.5, len(FLEETS) - 0.5)
        ax.set_ylim(-0.07, 1.07)
        if ri == 0:
            ax.set_title(REGION_LABEL[reg], pad=4)
        ax.grid(axis="y", alpha=0.45, lw=0.4)
        ax.tick_params(labelleft=(ci == 0))
        if ci == 0:
            ROWLAB.append((ax, vn))
        ax.tick_params(axis="x", labelsize=7 if AX_W >= 1.10 else 6.0)

    han = {"free": Line2D([], [], color=FREE, lw=1.25, label="free-running"),
           "ae": Line2D([], [], color=FLOOR_C, lw=1.05, ls="--", dashes=LEG_DASH,
                        label="autoencoder floor")}
    han.update({a: Line2D([], [], color=cols[a], lw=0, ls="none", marker=MARK[a],
                          ms=MS, mew=MEW, label=LABEL[a]) for a in ARMS})
    LEG_COLS = [["free", "ae"], ["static", "d_optimal"], ["random", "a_optimal"],
                ["lawnmower"]]
    y = top - AX_H - ROW_XLAB_H[ri]
    if LEGEND == "per-row":
        fig.legend(handles=[han[k] for col in LEG_COLS for k in col],
                   loc="upper center", ncol=4, fontsize=LEG_FS, frameon=False,
                   bbox_to_anchor=((LEFT + RIGHT) / 2, y / FIG_H),
                   handlelength=1.4, columnspacing=0.9)
        y -= LEG_H

NEUTRAL = "0.30"
if LEGEND in ("right", "single"):
    han = {"free": Line2D([], [], color=FREE, lw=1.25, label="free-running"),
           "ae": Line2D([], [], color=FLOOR_C, lw=1.05, ls="--", dashes=LEG_DASH,
                        label="autoencoder floor")}
    han.update({a: Line2D([], [], color=NEUTRAL, lw=0, ls="none", marker=MARK[a],
                          ms=MS, mew=MEW, mfc=NEUTRAL, label=LABEL[a]) for a in ARMS})
if LEGEND == "right":
    ORDER = ["ae", "a_optimal", "d_optimal", "lawnmower", "random", "static", "free"]
    _mid = 1 - (PAD + ROW_TITLE_H[0] + AX_H + ROW_XLAB_H[0] * 0.5) / FIG_H
    fig.legend(handles=[han[k] for k in ORDER], loc="center left", ncol=1,
               fontsize=LEG_FS, frameon=False, bbox_to_anchor=(RIGHT + 0.018, _mid),
               handlelength=HANDLE_L, handletextpad=HANDLE_PAD, labelspacing=0.9,
               borderpad=0.0, borderaxespad=0.0)
if LEGEND == "single":
    LEG_COLS = [["free", "ae"], ["static", "d_optimal"], ["random", "a_optimal"],
                ["lawnmower"]]
    fig.legend(handles=[han[k] for col in LEG_COLS for k in col], loc="upper center",
               ncol=4, fontsize=LEG_FS, frameon=False,
               bbox_to_anchor=((LEFT + RIGHT) / 2, y / FIG_H),
               handlelength=1.4, columnspacing=0.9)
    y -= LEG_H

# Left-hand labels, placed after the draw because their x depends on the tick widths.
fig.canvas.draw()
_inv = fig.transFigure.inverted()
_axes = [a for a, _ in ROWLAB]
_tick_x0 = min(_inv.transform(t.get_window_extent().get_points())[0][0]
               for a in _axes for t in a.get_yticklabels() if t.get_text())
_ylab_w = NLINES * LINE_IN(YLAB_FS) / FIG_WD
_row_w = LINE_IN(ROWLAB_FS) / FIG_WD
_ylab_x = _tick_x0 - GAP_IN / FIG_WD - _ylab_w / 2
_row_x = _ylab_x - _ylab_w / 2 - GAP_IN / FIG_WD - _row_w / 2
assert _row_x - _row_w / 2 > 0.002, "left margin too narrow for the row label"
_grid = [_inv.transform(a.get_window_extent().get_points()) for a in _axes]
fig.text(_ylab_x, 0.5 * (min(g[0][1] for g in _grid) + max(g[1][1] for g in _grid)),
         YLABEL, rotation=90, ha="center", va="center", fontsize=YLAB_FS)
for (_ax, _vn), _g in zip(ROWLAB, _grid):
    fig.text(_row_x, 0.5 * (_g[0][1] + _g[1][1]), _vn, rotation=90, ha="center",
             va="center", fontsize=ROWLAB_FS)

stem = "fig3_skill" + ("_legright" if LEGEND == "right" else "")
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
json.dump(dict(layout="2x2; row 1 = SST (MAB, CCS), row 2 = SSS (MAB, CCS)",
               figsize_in=[round(FIG_WD, 3), round(FIG_H, 3)],
               panel_in=[round(AX_W, 4), round(AX_H, 4)],
               window="final eight cycles (paper window)",
               quantity="fraction of recoverable error removed, mean +- 1 sd "
                        "across 10 sealed ICs",
               values={r: {k: {a: [round(v[a]["mean"], 5), round(v[a]["sd"], 5)]
                               for a in ARMS}
                           for k, v in FR[r].items()} for r in REGIONS},
               inputs="results/scores/ only"),
          open(OUT / f"{stem}_meta.json", "w"), indent=1)
print(f"wrote {stem}.pdf/.png under {OUT}")
print("FIG3_SKILL OK")
