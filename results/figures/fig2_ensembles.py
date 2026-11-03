#!/usr/bin/env python
"""Fig. 2: the forecast ensemble and its error, both regions in one figure.

4 x 6 grid:
  rows 0-1  MAB, SST then SSS   (IC 2022-04-09T12)
  rows 2-3  CCS, SST then SSS   (IC 2016-07-27T12)
columns: truth | member 1 | member 2 | ensemble mean | ensemble spread |
|mean - truth|. The state shown is the free-running (nosample) analysis
ensemble at cycle 39 = t+240 h.

One panel width for both regions, so the row heights follow each region's own
aspect; column titles on the top row only; the region named once per pair;
colorbars per row (the regions' field ranges do not overlap); per-row colour
limits with the 1st/5th-percentile floors and the shared spread/error rule.
View-from-shore convention throughout.

Inputs: $CLOSEDLOOP_ARTIFACTS/figure_fields/fig2_members_<region>_<ic>.npz
(decoded once from the released free-running latents). Each file carries the
spread-error correlation measured at staging; it is recomputed here and
asserted, so a drifted decode cannot slip through.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from _shore import ANCHORS, COAST, Shore
from _style import BASE_RC, REGION_LABEL, argval, artifacts_dir, out_dir, variable_cmap

FIG_W = float(argval("--figw", "7.16"))
OUT = out_dir(argval("--out", ""))
FF = artifacts_dir() / "figure_fields"
REGIONS = [("mab", "ic20220409T12"), ("ccs", "ic20160727T12")]
VROWS = [("sst", dict(unit=r"$^\circ$C", label="SST")),
         ("sss", dict(unit="psu", label="SSS"))]
CM_SPREAD, CM_ERR = "Blues", "Reds"
SHARE_FACTOR = 2.0
FIG1_PANEL_H = 0.9326          # the panel height the furniture scale is defined on

plt.rcParams.update({**BASE_RC, "font.size": 6.5, "axes.titlesize": 7,
                     "axes.titlepad": 3.0, "axes.linewidth": 0.5,
                     "image.interpolation": "nearest"})

REG = {}
for rk, ictag in REGIONS:
    npz = np.load(FF / f"fig2_members_{rk}_{ictag}.npz", allow_pickle=True)
    mask = npz["mask"]
    sh = Shore.from_grid(rk, mask, npz["lon"], npz["lat"],
                         float(npz["dx_m"]), float(npz["dy_m"]))
    # both regions take MAB's anchors so the rose and bar sit in the same corner
    REG[rk] = dict(label=REGION_LABEL[rk], ic=str(npz["ic"]), npz=npz, mask=mask,
                   SH=sh, aspect=sh.H / sh.W * sh.dy / sh.dx,
                   anchors=dict(ANCHORS["mab"]))
    print(f"{rk:4s} {REG[rk]['ic']}  grid {sh.H}x{sh.W}  "
          f"aspect {REG[rk]['aspect']:.4f}  transform {sh.t}, det {sh.det:+.0f}")

ROWS = [(rk, vk, cfg) for rk, _ in REGIONS for vk, cfg in VROWS]

# ---- geometry -----------------------------------------------------------------------
WR = [1, 1, 1, 1, 0.185, 1, 0.185, 1, 0.185]          # 4 maps, bar, map, bar, map, bar
WSPACE, RIGHT = 0.055, 0.995
VARLAB_FS, REGLAB_FS, GAP_IN = 5.8, 7.0, 0.050
LINE_IN = lambda fs: fs * 1.25 / 72
BASE_LEFT_IN = 0.048 * 7.16
LEFT_IN = BASE_LEFT_IN + GAP_IN + LINE_IN(REGLAB_FS)
LEFT = LEFT_IN / FIG_W
_unit = (RIGHT - LEFT) * FIG_W / (sum(WR) * (1 + (len(WR) - 1) * WSPACE / len(WR)))
RH = [_unit * REG[rk]["aspect"] for rk, _, _ in ROWS]
ROW_GAP_IN = 0.12
_H10 = (_unit * REG["mab"]["aspect"]) * (10 + 9 * 0.11) / (0.955 - 0.008)
TOP_IN, BOT_IN = (1 - 0.955) * _H10, 0.008 * _H10
FIG_H = sum(RH) + ROW_GAP_IN * (len(ROWS) - 1) + TOP_IN + BOT_IN
UNIT_DY = 0.0035 * _H10 / FIG_H

# rose-to-bar spacing matched in INCHES across the two row heights
_ROW_H = {rk: _unit * REG[rk]["aspect"] for rk in REG}
_BAR_Y = ANCHORS["mab"]["bar_y"]
_GAP_IN = (ANCHORS["mab"]["rose"][1] - _BAR_Y) * _ROW_H["mab"]
for rk in REG:
    _rx = REG[rk]["anchors"]["rose"][0]
    REG[rk]["anchors"]["rose"] = (_rx, _BAR_Y + _GAP_IN / _ROW_H[rk])
print(f"panel {_unit:.3f} in wide; row heights {[round(h, 3) for h in RH]} in; "
      f"figure {FIG_W} x {FIG_H:.3f} in")

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(len(ROWS), len(WR), width_ratios=WR, height_ratios=RH,
                       hspace=ROW_GAP_IN / (sum(RH) / len(RH)), wspace=WSPACE,
                       left=LEFT, right=RIGHT,
                       top=1 - TOP_IN / FIG_H, bottom=BOT_IN / FIG_H)

log = {"_meta": {"state": "free-running nosample ensemble, cycle 39 = t+240 h",
                 "rows": "MAB SST, MAB SSS, CCS SST, CCS SSS",
                 "ics": {rk: REG[rk]["ic"] for rk in REG},
                 "convention": "view from shore",
                 "members_shown": [1, 2],
                 "sss_lower_percentile": 5, "sst_lower_percentile": 1,
                 "share_rule": f"spread and error share limits when caps within "
                               f"{SHARE_FACTOR}x",
                 "colorbars": "per row",
                 "figsize_in": [FIG_W, round(FIG_H, 3)],
                 "panel_w_in": round(_unit, 4),
                 "row_h_in": [round(h, 4) for h in RH]},
       "rows": {}}

CB_ROWS, VARLAB_AXES, FURN_R = [], [], [None]
for ri, (rk, vk, cfg) in enumerate(ROWS):
    R = REG[rk]
    SH, mask = R["SH"], R["mask"]
    d = {k: np.asarray(R["npz"][f"{vk}_{k}"], float)
         for k in ("truth", "m1", "m2", "mean", "spread", "err")}
    tw = d["truth"][mask]
    p_lo = 5 if vk == "sss" else 1
    v0, v1 = float(np.percentile(tw, p_lo)), float(np.percentile(tw, 99))
    VS = float(np.percentile(d["spread"][mask], 99))
    VE = float(np.percentile(d["err"][mask], 99))
    shared = max(VS, VE) / max(min(VS, VE), 1e-12) <= SHARE_FACTOR
    sp_max, er_max = (max(VS, VE),) * 2 if shared else (VS, VE)
    sw, ew = d["spread"][mask], d["err"][mask]
    good = np.isfinite(sw) & np.isfinite(ew)
    r = float(np.corrcoef(sw[good], ew[good])[0, 1])
    stored = float(R["npz"][f"expected_r_{vk}"])
    assert abs(r - stored) < 1e-9, (
        f"{rk} {vk}: spread-error r {r:+.6f} drifted from the staged value "
        f"{stored:+.6f}")

    panels = [("truth", d["truth"], variable_cmap(vk), v0, v1, None),
              ("member 1", d["m1"], variable_cmap(vk), v0, v1, None),
              ("member 2", d["m2"], variable_cmap(vk), v0, v1, None),
              ("ensemble mean", d["mean"], variable_cmap(vk), v0, v1, "fd"),
              ("ensemble spread", d["spread"], CM_SPREAD, 0.0, sp_max, "sp"),
              (r"$|$mean $-$ truth$|$", d["err"], CM_ERR, 0.0, er_max, "er")]
    gcol = [0, 1, 2, 3, 5, 7]
    row_ims = {}
    for pi, (title, fld, cm, a0, a1, cbcol) in enumerate(panels):
        ax = fig.add_subplot(gs[ri, gcol[pi]])
        cmap = cm if isinstance(cm, str) else cm
        im = ax.imshow(SH.fill_to_coast(fld), cmap=cmap, vmin=a0, vmax=a1,
                       origin="upper", zorder=1)
        SH.base(ax, coast_lw=0.25, spine_lw=0.35)
        if ri == 0:
            ax.set_title(title, fontsize=6.5)
        if pi == 0:
            ax.set_ylabel(cfg["label"], fontsize=VARLAB_FS, labelpad=2)
            VARLAB_AXES.append((ax, rk))
            if vk == "sst":            # compass and bar once per REGION
                SH.borders(ax, lw=0.25)
                ax.apply_aspect()
                _bb = ax.get_position(original=False)
                # one furniture scale, taken from the first (MAB) panel
                if FURN_R[0] is None:
                    FURN_R[0] = _bb.height * fig.get_figheight() / FIG1_PANEL_H
                _r = FURN_R[0]
                SH.furniture(ax, fig, size=0.48 * _r,
                             tag=f"{rk} {vk} first panel (scale {_r:.3f})",
                             anchors=R["anchors"], letters="NESW",
                             label_fs=3.6 * _r, bar_lw=0.7 * _r)
        if cbcol == "fd":
            row_ims["fd"] = (ax, im, v0, v1, 1)
        elif cbcol == "sp":
            row_ims["sp"] = (ax, im, 0.0, sp_max, 2)
        elif cbcol == "er":
            CB_ROWS.append([row_ims["fd"], row_ims["sp"], (ax, im, 0.0, er_max, 2)])
    log["rows"][f"{rk} {cfg['label']}"] = dict(
        ic=R["ic"], pearson_spread_vs_abserr_240h=round(r, 4),
        field_limits=[round(v0, 3), round(v1, 3)], lower_percentile=p_lo,
        spread_cap=round(VS, 4), error_cap=round(VE, 4),
        shared_spread_error_limits=bool(shared))
    print(f"  {rk:4s} {cfg['label']}  r@240h {r:+.3f}   field [{v0:.2f},{v1:.2f}]  "
          f"caps {VS:.3f}/{VE:.3f} -> {'SHARED' if shared else 'SEPARATE'}")

# ---- colorbars ----------------------------------------------------------------------
fig.canvas.draw()
_inv = fig.transFigure.inverted()
GAP, BW = 0.005, 0.0062 * 7.16 / FIG_W
for _ri, _row in enumerate(CB_ROWS):
    for _ax, _im, _lo, _hi, _nd in _row:
        _pb = _inv.transform(_ax.get_window_extent().get_points())
        _c = fig.add_axes([_pb[1][0] + GAP, _pb[0][1], BW, _pb[1][1] - _pb[0][1]])
        _cb = fig.colorbar(_im, cax=_c)
        _cb.set_ticks([_lo, _hi])
        _c.set_yticklabels([f"{_lo:.{_nd}f}", f"{_hi:.{_nd}f}"])
        _c.tick_params(labelsize=4.3, length=1.2, width=0.3, color=COAST, pad=1.0)
        _cb.outline.set(edgecolor=COAST, linewidth=0.35)
        _p = _c.get_position()
        fig.text(_p.x0 + _p.width / 2, _p.y1 + UNIT_DY, ROWS[_ri][2]["unit"],
                 ha="center", va="bottom", fontsize=4.6, color="0.30")

# ---- the region, once per pair, outboard of the SST/SSS labels -----------------------
_lab_x0 = min(_inv.transform(a.yaxis.label.get_window_extent().get_points())[0][0]
              for a, _ in VARLAB_AXES)
_reg_w = LINE_IN(REGLAB_FS) / FIG_W
_reg_x = _lab_x0 - GAP_IN / FIG_W - _reg_w / 2
assert _reg_x - _reg_w / 2 > 0.001, "left margin too narrow for the region label"
for rk, _ in REGIONS:
    _bx = [_inv.transform(a.get_window_extent().get_points())
           for a, k in VARLAB_AXES if k == rk]
    fig.text(_reg_x, 0.5 * (min(b[0][1] for b in _bx) + max(b[1][1] for b in _bx)),
             REGION_LABEL[rk], rotation=90, ha="center", va="center",
             fontsize=REGLAB_FS)

STEM = "fig2_ensembles"
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{STEM}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
json.dump(log, open(OUT / f"{STEM}_stats.json", "w"), indent=1)
print(f"wrote {STEM}.pdf/.png under {OUT}")
print("FIG2_ENSEMBLES OK")
