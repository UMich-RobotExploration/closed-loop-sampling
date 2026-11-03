#!/usr/bin/env python
"""Fig. 1: the method overview -- four vertical step boxes, real renders inside.

Four self-titled rounded boxes read top to bottom and key 1:1 to the caption's
(i)-(v). Each box shows before -> after with the METHOD on the arrow. Every map
panel is the same real render (MAB, IC 2022-04-09T12, cycle 39): forecast
members, ensemble spread, the dumped reachable set with robots and selections.

The submitted variant is the "shell_soft" palette (four hues at equal
perceptual chroma, fill L*92 C*9 / edge L*62 C*32) with the step number on a
disc lifted halfway to the box edge colour ("live").

Input: $CLOSEDLOOP_ARTIFACTS/figure_fields/fig1_method_mab_ic20220409T12.npz
(decoded member fields, the A-optimal chain's cycle-39 spread, fleet, reachable
set and selections, grid geometry).
"""
import json
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap

from _shore import Shore
from _style import argval, artifacts_dir, out_dir, variable_cmap

OUT = out_dir(argval("--out", ""))
SRC = artifacts_dir() / "figure_fields/fig1_method_mab_ic20220409T12.npz"
Z = np.load(SRC, allow_pickle=True)

mask = Z["mask"]
SH = Shore.from_grid("mab", mask, Z["lon"], Z["lat"],
                     float(Z["dx_col_m"]), float(Z["dy_row_m"]))
ASPECT = SH.H / SH.W * SH.dy / SH.dx
dxm, dym = float(Z["dx_col_m"]), float(Z["dy_row_m"])
REACH_M = float(Z["v_auv_ms"]) * float(Z["reach_window_s"])

members = np.asarray(Z["members"], float)
IDX = [int(v) for v in Z["figure_member_indices"]]
spread = np.asarray(Z["spread"], float)
fleet = np.atleast_2d(np.asarray(Z["fleet"], float))
chosen = np.atleast_2d(np.asarray(Z["chosen_rc"], float)).astype(int)
cand = np.atleast_2d(np.asarray(Z["cand_rc"], float)).astype(int)
fw = members[:, mask]
fw = fw[np.isfinite(fw)]
V0, V1 = float(np.percentile(fw, 1)), float(np.percentile(fw, 99))
VS = float(np.nanpercentile(spread[mask], 99))

# ---- recover each robot's advected disk centre from the dumped reachable set -------
U = np.zeros(mask.shape, bool)
U[cand[:, 0], cand[:, 1]] = True
lab = ndimage.label(U)[0]
own = lab[np.round(fleet[:, 0]).astype(int), np.round(fleet[:, 1]).astype(int)]
solo = {}
for i, c in enumerate(own):
    solo.setdefault(int(c), []).append(i)
FULL = np.pi * (REACH_M / dxm) * (REACH_M / dym) * 0.92
cands = []
for c, ids in solo.items():
    if c == 0 or len(ids) != 1:
        continue
    cells = np.argwhere(lab == c).astype(float)
    if len(cells) < FULL:
        continue
    cen = cells.mean(0)
    rmax = np.sqrt(((cells[:, 0] - cen[0]) * dym) ** 2
                   + ((cells[:, 1] - cen[1]) * dxm) ** 2).max()
    cands.append((ids[0], cen, rmax))
assert len(cands) >= 4, f"only {len(cands)} unclipped single-robot disks"
rr = [c[2] / 1000 for c in cands]
assert abs(np.median(rr) - REACH_M / 1000) < 2.0
# The disk must be VISIBLY offset from the robot; drift is a filter, not a tiebreak.
MIN_DRIFT_M = 9000.0
drift = {c[0]: float(np.hypot((c[1][0] - fleet[c[0], 0]) * dym,
                              (c[1][1] - fleet[c[0], 1]) * dxm)) for c in cands}
movers = [c for c in cands if drift[c[0]] >= MIN_DRIFT_M]
assert len(movers) >= 4
picked = [max(movers, key=lambda c: drift[c[0]])]
while len(picked) < 4:
    picked.append(max((c for c in movers if c not in picked),
                      key=lambda c: min(np.hypot((c[1][0] - p[1][0]) * dym,
                                                 (c[1][1] - p[1][1]) * dxm)
                                        for p in picked)))

# ============================== STYLE ===========================================
INK, GREY, LAND = "0.22", "0.45", "#e8e8e8"
PICK = "0.30"
# the submitted palette: four hues held at equal perceptual chroma
# (fill L*92 C*9, edge L*62 C*32, number L*36 C*34)
BOXC = [("#FBE3DF", "#CC837A"), ("#DAEAF8", "#559DCC"),
        ("#F3E6D7", "#B2915E"), ("#DEECDD", "#73A172")]
NUMC = ["#85403A", "#005B81", "#6C501D", "#2F5F32"]
TAG_C = "#666666"
AR_SC, AR_LW = 5.6, 0.6


def mix(c1, c2, t):
    q, w = matplotlib.colors.to_rgb(c1), matplotlib.colors.to_rgb(c2)
    return tuple(u + (v - u) * t for u, v in zip(q, w))


CM_F = variable_cmap("sst").copy()
CM_F.set_bad(LAND)
CM_S = matplotlib.colormaps["Blues"].copy()
CM_S.set_bad(LAND)
FS_TITLE, FS_SYM, FS_SMALL, FS_TAG = 7.5, 8.5, 6.0, 6.0
FS_QUAL = FS_SMALL - 0.7
plt.rcParams.update({"font.size": FS_SMALL, "pdf.fonttype": 42,
                     "font.family": "Nimbus Roman", "mathtext.fontset": "stix",
                     "image.interpolation": "nearest"})

# ============================== GEOMETRY (inches, y measured DOWN) ==============
FIG_W = float(argval("--figw", "3.54"))
MARG = 0.040
WIRE_W = 0.280
TOP_WIRE = WIRE_W
BOX_X0, BOX_X1 = MARG, FIG_W - MARG - WIRE_W
BOX_W = BOX_X1 - BOX_X0
PAD_I = 0.050
TITLE_H = 0.235
NUM_T, NUM_D = 0.038, 0.125
SUB_LINE, SUB_PAD = 0.082, 0.033
GAP_BOX, ARR_GAP, PAD_B = 0.180, 0.022, 0.035
WP = 1.106
HP = WP * ASPECT
ICON_FW, ICON_FH = 0.360, 0.155
ICON_DX, ICON_DY = 0.050, 0.050
ICON_GX, ICON_GY = -0.036, 0.034
ICON_W = ICON_FW + ICON_DX - 2 * ICON_GX
ICON_H = ICON_FH + ICON_DY + 2 * ICON_GY
MEM_OFF, MEM_TWF = ((0.16, 0.16), (0.08, 0.08), (0.0, 0.0)), 0.84
ROUND = 0.012
LWBOX, LWARR, LWPANEL = 0.5, 0.7, 0.38
LWWIRE = LWARR

TITLES = ["Forecast next latent state",
          "Decode ensemble to physical space",
          "Quantify uncertainty, select sampling locations",
          "Assimilate observations"]
NUMS = ["1", "2", "3", "4"]
MLAB = [(["Predict next state in latent space"],
         [r"conditioned on $\mathcal{Z}_t^{+}$"], "qual"),
        (["Decode with learned model"], [], "plain"),
        (["A- or D-optimality"], ["(greedy assignment)"], "qual"),
        (["Observation-conditioned", "diffusion sampling"],
         [r"conditioned on $\mathcal{Z}_t^{+}$ and $y_{t+1}$"], "qual")]
SUBL = [["Latent analysis", r"$N$ members"], [],
        ["Ensemble spread (uncertainty)"], [r"Collect samples $y_{t+1}$"]]
SUBR = [["Latent forecast", "Discarded after planning"],
        ["Ensemble of ocean states"],
        ["Select samples under", "motion constraints"],
        ["Updated latent state"]]
SUB_IT = {"Discarded after planning"}
SUBN = [max(len(a), len(b)) for a, b in zip(SUBL, SUBR)]
SUBH = [SUB_PAD + SUB_LINE * m for m in SUBN]
TAGS = ["latent", r"latent $\rightarrow$ physical", "physical",
        r"physical $\rightarrow$ latent"]

_f = plt.figure(figsize=(FIG_W, 2.0))
_r = _f.canvas.get_renderer()


def tw(t, fs=FS_SMALL):
    o = _f.text(0.5, 0.5, t, fontsize=fs)
    w = o.get_window_extent(renderer=_r).width / _f.dpi
    o.remove()
    return w


TITLE_W = max(tw(t, FS_TITLE) for t in TITLES)
plt.close(_f)

ITEM_W = WP
GAP_A = BOX_W - 2 * PAD_I - 2 * ITEM_W
assert GAP_A > 0.28, f"arrow gap {GAP_A:.3f} in is too small"
H_I = TITLE_H + ICON_H + SUBH[0] + PAD_B
H_II = TITLE_H + HP + SUBH[1] + PAD_B
H_III = TITLE_H + HP + SUBH[2] + PAD_B
H_V = TITLE_H + HP + SUBH[3] + PAD_B
HS = [H_I, H_II, H_III, H_V]
BY0 = [MARG + TOP_WIRE]
for k in range(1, 4):
    BY0.append(BY0[k - 1] + HS[k - 1] + GAP_BOX)
FIG_H = BY0[3] + HS[3] + MARG
assert FIG_H < 4.60
assert TITLE_H > NUM_T + NUM_D + 0.018
fig = plt.figure(figsize=(FIG_W, FIG_H))
X = lambda v: v / FIG_W
Y = lambda v: 1 - v / FIG_H


def rbox(x0, y0, x1, y1, fc, ec, lw=LWBOX, z=1, ls="-"):
    fig.patches.append(FancyBboxPatch((X(x0), Y(y1)), (x1 - x0) / FIG_W,
                                      (y1 - y0) / FIG_H,
                                      boxstyle=f"round,pad=0,rounding_size={ROUND}",
                                      transform=fig.transFigure, fc=fc, ec=ec,
                                      lw=lw, ls=ls, zorder=z))


def slab_icon(cx, ytop, sym):
    """A shallow slab with two ghost slabs receding up-left: a latent state."""
    fw_, fh_, dx_, dy_ = ICON_FW, ICON_FH, ICON_DX, ICON_DY
    gx_, gy_ = ICON_GX, ICON_GY
    x0 = cx - ICON_W / 2 - gx_ * 2
    y0 = ytop + ICON_H

    def slab(sx, sy, fcs, ec, lw, z):
        for pts, fc in zip(([(0, fh_), (dx_, fh_ + dy_), (fw_ + dx_, fh_ + dy_), (fw_, fh_)],
                            [(fw_, 0), (fw_ + dx_, dy_), (fw_ + dx_, fh_ + dy_), (fw_, fh_)],
                            [(0, 0), (fw_, 0), (fw_, fh_), (0, fh_)]), fcs):
            fig.patches.append(Polygon([(X(sx + a), Y(sy - b)) for a, b in pts],
                                       closed=True, transform=fig.transFigure,
                                       fc=fc, ec=ec, lw=lw, zorder=z))
    for g in (2, 1):
        slab(x0 + g * gx_, y0 - g * gy_, ("white", "white", "white"), "0.72", 0.40,
             3 + (2 - g))
    slab(x0, y0, ("0.84", "0.78", "0.93"), "0.52", 0.55, 6)
    fig.text(X(x0 + fw_ / 2), Y(y0 - fh_ / 2), sym, fontsize=FS_SYM, color=INK,
             ha="center", va="center", zorder=7)


def harrow(x0, x1, y, z=6):
    fig.patches.append(FancyArrowPatch((X(x0), Y(y)), (X(x1), Y(y)),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=AR_SC, lw=AR_LW, color=INK,
                                       shrinkA=0, shrinkB=0, zorder=z))


def arrow_label(k, y, a0, a1):
    above, below, bsty = MLAB[k]
    FS_Q = FS_QUAL if bsty == "qual" else FS_SMALL
    ST_Q = "italic" if bsty in ("qual", "italic") else "normal"
    c0, c1 = a0 - AR_CLR, a1 + AR_CLR
    w = max(max((tw(ln) for ln in above), default=0.0),
            max((tw(ln, FS_Q) for ln in below), default=0.0))
    assert w + 2 * LAB_CLR < c1 - c0, f"box {k + 1}: label does not fit its channel"
    xm = min(max(0.5 * (a0 + a1), c0 + LAB_CLR + w / 2), c1 - LAB_CLR - w / 2)
    n_above = len(above)
    assert y - 0.022 - n_above * SUB_LINE > BY0[k] + NUM_T + NUM_D + 0.012
    for j, ln in enumerate(above):
        fig.text(X(xm), Y(y - 0.022 - (n_above - 1 - j) * SUB_LINE), ln,
                 fontsize=FS_SMALL, color=INK, ha="center", va="bottom", zorder=8)
    for j, ln in enumerate(below):
        fig.text(X(xm), Y(y + 0.024 + j * SUB_LINE), ln, fontsize=FS_Q, color=INK,
                 ha="center", va="top", zorder=8, style=ST_Q)


def subcap(cx, ytop, lines):
    for j, ln in enumerate(lines):
        fig.text(X(cx), Y(ytop + 0.026 + j * SUB_LINE), ln, fontsize=FS_SMALL,
                 color=INK, ha="center", va="top", zorder=8,
                 style="italic" if ln in SUB_IT else "normal")
        assert tw(ln) < ITEM_W + 0.12, f"sub-caption '{ln}' is wider than its item"


def map_axes(x0, ytop, w):
    return fig.add_axes([X(x0), Y(ytop + w * ASPECT), w / FIG_W, (w * ASPECT) / FIG_H])


def draw_map(a, field, cmap, vmin, vmax, z=2):
    a.imshow(SH.fill_to_coast(field), cmap=cmap, vmin=vmin, vmax=vmax,
             origin="upper", zorder=1)
    SH.base(a, coast_lw=0.20, spine_lw=LWPANEL)
    a.set_zorder(z)


# ============================== BOXES ===========================================
LX = BOX_X0 + PAD_I + ITEM_W / 2
RX = BOX_X1 - PAD_I - ITEM_W / 2
AR_CLR = 0.060
LAB_CLR = 0.030


def span(lw, rw):
    return LX + lw + AR_CLR, RX - rw - AR_CLR


for k in range(4):
    fc, ec = BOXC[k]
    rbox(BOX_X0, BY0[k], BOX_X1, BY0[k] + HS[k], fc, ec, LWBOX, z=1)
    _ncx, _ncy = BOX_X0 + 0.085 + NUM_D / 2, BY0[k] + NUM_T + NUM_D / 2
    fig.patches.append(Ellipse((X(_ncx), Y(_ncy)), NUM_D / FIG_W, NUM_D / FIG_H,
                               transform=fig.transFigure, fc=mix(NUMC[k], ec, 0.5),
                               ec="none", lw=0.0, zorder=8))
    fig.text(X(_ncx), Y(_ncy), NUMS[k], fontsize=FS_TITLE - 1.0, color="white",
             ha="center", va="center", zorder=9, fontweight="bold")
    fig.text(X(BOX_X0 + 0.085 + NUM_D + 0.045), Y(BY0[k] + 0.040), TITLES[k],
             fontsize=FS_TITLE, color=INK, ha="left", va="top", zorder=8,
             fontweight="bold")
    fig.text(X(BOX_X1 - 0.085), Y(BY0[k] + 0.044), TAGS[k], fontsize=FS_TAG,
             color=TAG_C, ha="right", va="top", style="italic", zorder=8)

# ---- (i) forecast: two latent slab stacks --------------------------------------
cy = BY0[0] + TITLE_H
slab_icon(LX, cy, r"$\mathcal{Z}_t^{+}$")
slab_icon(RX, cy, r"$\mathcal{Z}_{t+1}^{-}$")
_a0, _a1 = span(ICON_W / 2, ICON_W / 2)
harrow(_a0, _a1, cy + ICON_H / 2)
arrow_label(0, cy + ICON_H / 2, _a0, _a1)
subcap(LX, cy + ICON_H, SUBL[0])
subcap(RX, cy + ICON_H, SUBR[0])

# ---- (ii) decode: latent stack -> offset member-field stack --------------------
cy = BY0[1] + TITLE_H
slab_icon(LX, cy + max(0.0, (HP - ICON_H) / 2), r"$\mathcal{Z}_{t+1}^{-}$")
for j, (ox, oy) in enumerate(MEM_OFF):
    a = map_axes(RX - WP / 2 + ox * WP, cy + HP * (1.0 - oy - MEM_TWF), MEM_TWF * WP)
    a.set_zorder(2 + j)
    draw_map(a, members[IDX[::-1][j]], CM_F, V0, V1, z=2 + j)
_a0, _a1 = span(ICON_W / 2, WP / 2)
harrow(_a0, _a1, cy + HP / 2)
arrow_label(1, cy + HP / 2, _a0, _a1)
subcap(RX, cy + HP, SUBR[1])

# ---- (iii-iv) spread -> sample panel (reachable union + robots + selections) ---
cy = BY0[2] + TITLE_H
aL = map_axes(LX - WP / 2, cy, WP)
draw_map(aL, spread, CM_S, 0.0, VS)
aR = map_axes(RX - WP / 2, cy, WP)
draw_map(aR, spread, CM_S, 0.0, VS)
Ud = SH.apply(U).astype(float)
aR.imshow(Ud, cmap=ListedColormap([(0, 0, 0, 0), (0.35, 0.35, 0.35, 0.40)]),
          vmin=0, vmax=1, origin="upper", zorder=4)
aR.contour(Ud, levels=[0.5], colors=["0.42"], linewidths=0.20, zorder=5)
fx_, fy_ = SH.xy(fleet[:, 0], fleet[:, 1])
aR.scatter(fx_, fy_, s=0.8, facecolors="white", edgecolors=GREY, linewidths=0.18,
           zorder=6)
cx_, cy_ = SH.xy(chosen[:, 0], chosen[:, 1])
aR.scatter(cx_, cy_, s=1.6, c=PICK, linewidths=0, zorder=7)
_a0, _a1 = span(WP / 2, WP / 2)
harrow(_a0, _a1, cy + HP / 2)
arrow_label(2, cy + HP / 2, _a0, _a1)
subcap(LX, cy + HP, SUBL[2])
subcap(RX, cy + HP, SUBR[2])

# ---- (v) assimilate -------------------------------------------------------------
cy = BY0[3] + TITLE_H
aO = map_axes(LX - WP / 2, cy, WP)
draw_map(aO, spread, CM_S, 0.0, VS)
aO.scatter(cx_, cy_, s=1.6, c=PICK, linewidths=0, zorder=7)
subcap(LX, cy + HP, SUBL[3])
_ay = cy + HP / 2
_a0, _a1 = span(WP / 2, ICON_W / 2)
harrow(_a0, _a1, _ay)
arrow_label(3, _ay, _a0, _a1)
slab_icon(RX, cy + max(0.0, (HP - ICON_H) / 2), r"$\mathcal{Z}_{t+1}^{+}$")
subcap(RX, cy + max(0.0, (HP - ICON_H) / 2) + ICON_H, SUBR[3])

# ============================== RETURN WIRE ======================================
_wx = BOX_X1 + WIRE_W / 2
_wy = MARG + WIRE_W / 2
_exit_y = BY0[3] + HS[3] / 2
_cxb = 0.5 * (BOX_X0 + BOX_X1)
for xs_, ys_ in (([X(BOX_X1 + ARR_GAP), X(_wx)], [Y(_exit_y), Y(_exit_y)]),
                 ([X(_wx), X(_wx)], [Y(_exit_y), Y(_wy)]),
                 ([X(_wx), X(_cxb)], [Y(_wy), Y(_wy)])):
    fig.add_artist(Line2D(xs_, ys_, transform=fig.transFigure, color=INK,
                          lw=LWWIRE, zorder=6))
fig.patches.append(FancyArrowPatch((X(_cxb), Y(_wy)), (X(_cxb), Y(BY0[0] - ARR_GAP)),
                                   transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=5.0, lw=LWWIRE, color=INK,
                                   shrinkA=0, shrinkB=0, zorder=6))
fig.text(X(0.5 * (_wx + _cxb)), Y(_wy), "Next cycle", fontsize=FS_SMALL + 0.8,
         color=INK, ha="center", va="center", zorder=7,
         bbox=dict(fc="white", ec="none", pad=0.8))
for k in range(3):
    fig.patches.append(FancyArrowPatch((X(_cxb), Y(BY0[k] + HS[k] + ARR_GAP)),
                                       (X(_cxb), Y(BY0[k + 1] - ARR_GAP)),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=5.0, lw=LWARR, color=INK,
                                       shrinkA=0, shrinkB=0, zorder=6))

MINPT = min(FS_TITLE - 1.0, FS_SYM, FS_SMALL, FS_TAG, FS_QUAL)
assert MINPT >= 5.3
assert TITLE_W + tw(TAGS[2], FS_TAG) + NUM_D + 0.22 < BOX_W
DPI = 600
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig1_overview.{ext}", bbox_inches="tight", pad_inches=0.02,
                dpi=DPI)
plt.rcParams["svg.fonttype"] = "none" if "--svgtext" in sys.argv else "path"
fig.savefig(OUT / "fig1_overview.svg", bbox_inches="tight", pad_inches=0.02, dpi=DPI)
plt.close(fig)
json.dump(dict(description="Vertical step-box method overview, (i)-(v); every "
                           "panel is the same real render.",
               figsize_in=[FIG_W, round(FIG_H, 3)],
               panel_in=[round(WP, 3), round(HP, 3)],
               titles=TITLES, tags=TAGS, min_font_pt=MINPT,
               palette="shell_soft", number_style="live",
               physical=dict(ic=str(Z["ic"]), region="mab", cycle=int(Z["cycle"]),
                             members_indices=IDX)),
          open(OUT / "fig1_overview_meta.json", "w"), indent=1)
print(f"min type {MINPT} pt; wrote fig1_overview.pdf/.svg/.png under {OUT}")
print("FIG1_OVERVIEW OK")
