#!/usr/bin/env python
"""Fig. 4: MAB SSS campaign trajectories at the six fleet sizes, plus the
fleet-size sweep.

One row, seven panels: n = 10, 20, 35, 50, 100, 200 as campaign trajectories
(per-cycle analysis-ensemble RMSE, seed-mean then IC-mean over the 10 sealed
ICs), a spacer, then the final-eight window mean against fleet size for the
same five policies, all on ONE shared y-axis (same quantity, same units).
Free-running reference solid neutral, autoencoder floor dashed, final-eight
window shaded.

Inputs are the committed scores release only (results/scores/): the per-cycle
tables, the per-run window scores (asserted consistent with the per-cycle
recomputation), and the free-running/floor references. The submitted variant
is --legend=right with square panels (the default here); --legend=below
reproduces the banded form.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _style import (BASE_RC, CLIP, FLOOR_C, FREE, LABEL, MARK, POLICIES,
                    SHADE, argval, out_dir, scores_dir, variable_cmap)

from scoring import metrics as M

SCORES = scores_dir()
OUT = out_dir(argval("--out", ""))
FIG_W = float(argval("--figw", "7.16"))
SWEEP = argval("--sweep", "rmse")
LEGEND = argval("--legend", "right")
SQUARE = argval("--square", "1") != "0"
assert SWEEP in ("rmse", "rmse_zoom", "skill") and LEGEND in ("below", "right")
WIN = 8                                    # paper window
NS = [10, 20, 35, 50, 100, 200]
ARMS = POLICIES
CMAP = variable_cmap("sss")
COLS = {a: CMAP(np.linspace(CLIP[0], CLIP[1], len(ARMS))[i]) for i, a in enumerate(ARMS)}
LEAD = (np.arange(40) + 1) * 6
plt.rcParams.update(BASE_RC)

# ---- load from the scores release, asserting completeness and consistency ----------
ICS = M.ics_for(SCORES, "mab")
assert len(ICS) == 10
per_cycle = M.load_per_cycle(SCORES, "mab", "sss")
runs = M.load_runs(SCORES)
CURVE, SCORE = {}, {}
for n in NS:
    for arm in ARMS:
        per_ic = []
        for ic in ICS:
            rows = [per_cycle[(n, ic, arm, s)]["post"] for s in M.SEEDS]
            per_ic.append(np.mean(rows, 0))
        per_ic = np.array(per_ic)                       # (10 ICs, 40 cycles)
        CURVE[(n, arm)] = per_ic.mean(0)
        SCORE[(n, arm)] = per_ic[:, -WIN:].mean(1)      # per-IC window score
        # the per-cycle table and the per-run window scores must agree exactly
        ref = np.array([M.per_ic_score(runs, "mab", "sss", n, ic, arm) for ic in ICS])
        assert np.max(np.abs(SCORE[(n, arm)] - ref)) < 1e-12
FRE_IC = np.array([M.load_free_running(SCORES)[("mab", "sss", ic)] for ic in ICS])
FLO_IC = np.array([M.load_ae_floor(SCORES)[("mab", "sss", ic)] for ic in ICS])
FRE, FLO = FRE_IC.mean(0), FLO_IC.mean(0)
# Skill fraction formed PER IC (the map is affine only per IC), then averaged.
_fw, _lw = FRE_IC[:, -WIN:].mean(1), FLO_IC[:, -WIN:].mean(1)
SKILL = {k: (_fw - v) / (_fw - _lw) for k, v in SCORE.items()}
print(f"loaded {len(NS)} fleet sizes x {len(ARMS)} policies x 10 ICs x 3 seeds; "
      "window scores agree with runs.csv to 1e-12")

# ---- layout: 6 trajectory panels + spacer + sweep, one shared y --------------------
LEFT, RIGHT, WSPACE = 0.058, 0.995, 0.155
WR = [1, 1, 1, 1, 1, 1, 0.22, 1.30]


def panel_in(wr, figw):
    k = len(wr)
    mean = sum(wr) / k
    return (RIGHT - LEFT) * figw / (sum(wr) + (k - 1) * WSPACE * mean)


AX_H_PARENT = (0.915 - 0.315) * 2.35
ASPECT = panel_in([1, 1, 1, 0.20, 1.22], 7.16) / AX_H_PARENT
AX_H = panel_in(WR, FIG_W) / ASPECT
TITLE_H, XLAB_H, LEG_H, PAD = 0.22, 0.34, 0.34, 0.04

# Type is specified at the size it should have ON THE PAGE and divided by the
# scale LaTeX applies at 0.99*textwidth.
TEX_W = 0.99 * 7.0
ON_PAGE = dict(base=7.3, title=8.3, ylab=7.3, leg=6.5, nticks=6.6, rylab=6.6, note=5.8)
HANDLE_L, HANDLE_PAD = 1.5, 0.5


def _legend_handles():
    return ([Line2D([], [], color=FREE, lw=1.25, label="free-running"),
             Line2D([], [], color=FLOOR_C, lw=1.05, ls="--", dashes=(3.4, 2.2),
                    label="autoencoder floor")]
            + [Line2D([], [], color=COLS[a], lw=0.95, marker=MARK[a], ms=3.4,
                      label=LABEL[a]) for a in ARMS]
            + [Patch(facecolor=SHADE, edgecolor="none", label="final eight cycles")])


LEG_TEXT = (["free-running", "autoencoder floor"] + [LABEL[a] for a in ARMS]
            + ["final eight cycles"])
_fm = plt.figure(figsize=(FIG_W, 2.0))
_rm = _fm.canvas.get_renderer()


def _tw(t, fs):
    o = _fm.text(0.5, 0.5, t, fontsize=fs)
    w = o.get_window_extent(renderer=_rm).width / _fm.dpi
    o.remove()
    return w


# legend column width and type scale iterate to a fixed point (settles in two)
LEGCOL_IN, FIG_WD = 0.0, FIG_W
for _ in range(8):
    _sc = TEX_W / FIG_WD
    LEG_FS = ON_PAGE["leg"] / _sc
    LEGCOL_IN = (max(_tw(t, LEG_FS) for t in LEG_TEXT)
                 + (HANDLE_L + HANDLE_PAD) * LEG_FS / 72 + 0.115) if LEGEND == "right" else 0.0
    _new = FIG_W + LEGCOL_IN
    if abs(_new - FIG_WD) < 1e-6:
        FIG_WD = _new
        break
    FIG_WD = _new
plt.close(_fm)
TEX_SCALE = TEX_W / FIG_WD
PT = lambda on_page: on_page / TEX_SCALE
LEG_FS = PT(ON_PAGE["leg"])
plt.rcParams.update({"font.size": PT(ON_PAGE["base"]),
                     "axes.titlesize": PT(ON_PAGE["title"])})

if LEGEND == "right":
    # the panel height is set to the legend's measured height, flush by construction
    _fl = plt.figure(figsize=(FIG_WD, 4.0))
    _lg = _fl.legend(handles=_legend_handles(), loc="center left", ncol=1,
                     fontsize=LEG_FS, frameon=False, handlelength=HANDLE_L,
                     handletextpad=HANDLE_PAD, labelspacing=0.9, borderpad=0.0,
                     borderaxespad=0.0)
    _fl.canvas.draw()
    AX_H = _lg.get_window_extent().height / _fl.dpi
    plt.close(_fl)
    if SQUARE:
        # widen the canvas until the panels are square; everything else holds
        FIG_W = AX_H / panel_in(WR, 1.0)
        FIG_WD = FIG_W + LEGCOL_IN
LEFT_D, RIGHT_D = LEFT * FIG_W / FIG_WD, RIGHT * FIG_W / FIG_WD
FIG_H = TITLE_H + AX_H + XLAB_H + (0.0 if LEGEND == "right" else LEG_H) + 2 * PAD
lo = min(FLO.min(), min(CURVE[k].min() for k in CURVE))
hi = max(FRE.max(), max(CURVE[k].max() for k in CURVE))
pad = 0.06 * (hi - lo)
print(f"figure {FIG_WD:.3f} x {FIG_H:.3f} in; panel "
      f"{panel_in(WR, FIG_W):.3f} x {AX_H:.3f} in; legend={LEGEND}; "
      f"shared y [{lo - pad:.4f}, {hi + pad:.4f}] psu")

fig = plt.figure(figsize=(FIG_WD, FIG_H))
top = FIG_H - PAD - TITLE_H
gs = fig.add_gridspec(1, len(WR), left=LEFT_D, right=RIGHT_D, width_ratios=WR,
                      wspace=WSPACE, top=top / FIG_H, bottom=(top - AX_H) / FIG_H)
for ci, n in enumerate(NS):
    ax = fig.add_subplot(gs[0, ci])
    ax.axvspan(198, 240, color=SHADE, zorder=0, lw=0)
    ax.plot(LEAD, FRE, "-", color=FREE, lw=1.25, zorder=5)
    ax.plot(LEAD, FLO, "--", color=FLOOR_C, lw=1.05, dashes=(3.4, 2.2), zorder=5)
    for arm in ARMS:
        ax.plot(LEAD, CURVE[(n, arm)], "-", color=COLS[arm], lw=0.95, zorder=3)
    ax.set_xlim(6, 240)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xticks([100, 200])
    ax.set_xlabel("lead [h]")
    ax.set_title(f"$n = {n}$", pad=4)
    ax.grid(alpha=0.45, lw=0.4)
    ax.tick_params(labelleft=(ci == 0))
    if ci == 0:
        ax.set_ylabel("MAB SSS\nensemble-mean RMSE [psu]", fontsize=PT(ON_PAGE["ylab"]))
fig.add_subplot(gs[0, 6]).axis("off")

ax = fig.add_subplot(gs[0, 7])                       # the sweep, against fleet size
_skill, _zoom = SWEEP == "skill", SWEEP == "rmse_zoom"
# Both references are fleet-invariant: free-running observes nothing and the
# autoencoder floor is identical across chains, so they are the sweep's asymptotes.
ax.axhline(0.0 if _skill else float(FRE[-WIN:].mean()), color=FREE, lw=1.25, zorder=4)
if not _zoom:
    ax.axhline(1.0 if _skill else float(FLO[-WIN:].mean()), color=FLOOR_C, ls="--",
               dashes=(3.4, 2.2), lw=1.05, zorder=4)
# Mean lines only: a +-1 sd band ACROSS ICs would bury arm separations that are
# significant on the paired tests (the policies share ICs and seeds).
src = SKILL if _skill else SCORE
for arm in ARMS:
    m = np.array([src[(n, arm)].mean() for n in NS])
    ax.plot(NS, m, "-", color=COLS[arm], lw=0.95, marker=MARK[arm], ms=3.4,
            mew=0.5, zorder=3)
ax.set_xscale("log")
ax.set_xticks(NS)
ax.set_xticklabels([str(n) for n in NS], fontsize=PT(ON_PAGE["nticks"]))
ax.minorticks_off()
ax.set_xlim(8.5, 235)
if _zoom:
    _v = [src[(n, a)].mean() for n in NS for a in ARMS] + [float(FRE[-WIN:].mean())]
    _m = 0.10 * (max(_v) - min(_v))
    ax.set_ylim(min(_v) - _m, max(_v) + _m)
ax.set_ylim(*(((-0.07, 1.07) if _skill else (lo - pad, hi + pad))
              if not _zoom else ax.get_ylim()))
ax.tick_params(labelleft=True, labelsize=6.0)
ax.set_ylabel("recoverable error removed" if _skill else "window-mean RMSE [psu]",
              fontsize=PT(ON_PAGE["rylab"]), labelpad=2)
ax.set_xlabel("fleet size $n$")
ax.set_title("final eight cycles", pad=4)
ax.grid(alpha=0.45, lw=0.4)

H = _legend_handles()
if LEGEND == "right":
    fig.legend(handles=H, loc="center left", ncol=1, fontsize=LEG_FS, frameon=False,
               bbox_to_anchor=(RIGHT_D + 0.010, (top - AX_H / 2) / FIG_H),
               handlelength=HANDLE_L, handletextpad=HANDLE_PAD, labelspacing=0.9,
               borderpad=0.0, borderaxespad=0.0)
else:
    fig.legend(handles=H, loc="upper center", ncol=8, fontsize=LEG_FS, frameon=False,
               bbox_to_anchor=(0.5, (top - AX_H - XLAB_H) / FIG_H),
               handlelength=HANDLE_L, columnspacing=1.1)

stem = ("fig4_fleet_size" + ({"skill": "_skill", "rmse_zoom": "_zoom"}.get(SWEEP, ""))
        + ("_legright" if LEGEND == "right" else ""))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
plt.close(fig)
json.dump(dict(fleets=NS, window=f"last{WIN} (paper window)",
               aggregation="per IC: mean over the three paired seeds; config: "
                           "mean over the 10 sealed ICs",
               shared_y=[round(lo - pad, 5), round(hi + pad, 5)],
               figsize_in=[round(FIG_WD, 3), round(FIG_H, 3)],
               panel_in=[round(panel_in(WR, FIG_W), 4), round(AX_H, 4)],
               sweep_panel=SWEEP,
               sweep_values={f"n{n}": {a: round(float(SCORE[(n, a)].mean()), 5)
                                       for a in ARMS} for n in NS},
               skill_values={f"n{n}": {a: round(float(SKILL[(n, a)].mean()), 5)
                                       for a in ARMS} for n in NS},
               freerun_window=round(float(FRE[-WIN:].mean()), 5),
               ae_floor_window=round(float(FLO[-WIN:].mean()), 5),
               inputs="results/scores/ only"),
          open(OUT / f"{stem}_meta.json", "w"), indent=1)
print(f"wrote {stem}.pdf/.png under {OUT}")
print("FIG4_FLEET_SIZE OK")
