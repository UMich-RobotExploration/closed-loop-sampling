#!/usr/bin/env python
"""Fig. 6: the confidently-wrong case -- three diagnostic panels for the MAB
free-running ensemble at t+240 h from IC 2022-08-02.

  (a) absolute SST error of the ensemble mean
  (b) ensemble spread
  (c) normalized displacement eta = |error| / spread (diverging scale centred
      at eta = 1)

Fields come from the staged artifacts dataset
($CLOSEDLOOP_ARTIFACTS/figure_fields/fig6_diagnostic_mab_20220802.npz), decoded
once from the released free-running latents; the display caps (99th-percentile)
ship inside the file so the figure is a pure replot. The panel letter rides on
each colorbar label; type is specified at its on-page size and divided by the
scale LaTeX applies (--texscale, 0.470 for a single-column ieeeconf figure).
"""
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from _shore import COAST, Shore
from _style import BASE_RC, argval, artifacts_dir, out_dir

TAG = argval("--tag", "20220802")
OUT = out_dir(argval("--out", ""))
SRC = artifacts_dir() / f"figure_fields/fig6_diagnostic_mab_{TAG}.npz"

TEX_SCALE = float(argval("--texscale", "0.470"))
PT = lambda on_page: on_page / TEX_SCALE
LABEL_PT, TICK_PT = PT(5.9), PT(5.0)

Z = np.load(SRC, allow_pickle=True)
mask = Z["mask"]
SH = Shore.from_grid("mab", mask, Z["lon"], Z["lat"], float(Z["dx_m"]), float(Z["dy_m"]))
ASPECT = SH.H / SH.W * SH.dy / SH.dx
VE, VS, VZ = float(Z["VE"]), float(Z["VS"]), float(Z["VZ"])
err, sp, eta = (np.array(Z[k], float) for k in ("err", "spread", "eta"))
for arr in (err, sp, eta):
    arr[~mask] = np.nan
print(f"{TAG}: caps err {VE:.2f} degC, spread {VS:.2f} degC, eta {VZ:.1f}; "
      f"grid {SH.H}x{SH.W}")

plt.rcParams.update({**BASE_RC, "font.size": TICK_PT, "axes.linewidth": 0.5,
                     "image.interpolation": "nearest"})

FIG_W = 7.16
LEFT, RIGHT, WSPACE = 0.004, 0.996, 0.05
PAN_W = (RIGHT - LEFT) * FIG_W / (3 + 2 * WSPACE)
PAN_H = PAN_W * ASPECT
CB_H = 0.055 * PAN_H
MAP_GAP = 0.050
LABELPAD = 1.5 / 72
BOT = 0.014 + TICK_PT / 72 + LABELPAD + LABEL_PT / 72 + 0.016
TOP_PAD = 0.020
FIG_H = TOP_PAD + PAN_H + MAP_GAP + CB_H + BOT
print(f"panel {PAN_W:.3f} x {PAN_H:.3f} in (aspect {ASPECT:.3f}); "
      f"figure {FIG_W} x {FIG_H:.3f} in")

fig = plt.figure(figsize=(FIG_W, FIG_H))
gs = gridspec.GridSpec(2, 3, height_ratios=[PAN_H, CB_H],
                       hspace=MAP_GAP / (0.5 * (PAN_H + CB_H)), wspace=WSPACE,
                       left=LEFT, right=RIGHT,
                       top=1 - TOP_PAD / FIG_H, bottom=BOT / FIG_H)
ZNORM = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=VZ)
panels = [(err, "Reds", VE, None, "Absolute SST error [°C]"),
          (sp, "Blues", VS, None, "Ensemble spread [°C]"),
          (eta, "RdBu_r", None, ZNORM, r"Normalized displacement $\eta$")]

# shrink the label type together until the widest label fits its own panel
_f0 = plt.figure(figsize=(FIG_W, 2.0))
_r0 = _f0.canvas.get_renderer()


def _tw(t, fs):
    o = _f0.text(0.5, 0.5, t, fontsize=fs)
    w = o.get_window_extent(renderer=_r0).width / _f0.dpi
    o.remove()
    return w


_labs = [f"({chr(97 + i)}) {q[4]}" for i, q in enumerate(panels)]
while LABEL_PT > 1 and max(_tw(t, LABEL_PT) for t in _labs) > PAN_W - 0.04:
    LABEL_PT -= 0.05
plt.close(_f0)

_AXS, _CAXS = [], []
for i, (f2d, cm, v1, nrm, clab) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    im = (ax.imshow(SH.fill_to_coast(f2d), cmap=cm, norm=nrm, origin="upper", zorder=1)
          if nrm is not None else
          ax.imshow(SH.fill_to_coast(f2d), cmap=cm, vmin=0, vmax=v1, origin="upper",
                    zorder=1))
    SH.base(ax, coast_lw=0.3, spine_lw=0.4)
    if i == 0:
        SH.furniture(ax, fig, size=0.48, tag=TAG)
    cax = fig.add_subplot(gs[1, i])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"({chr(97 + i)}) {clab}", fontsize=LABEL_PT, labelpad=1.5)
    # fewer ticks, none at the very ends of a sequential bar (end labels spill
    # into the neighbouring panel); the diverging bar keeps 0 and 1 explicitly
    _lo, _hi = 0.0, (v1 if nrm is None else VZ)
    if nrm is None:
        _tk = MaxNLocator(nbins=4).tick_values(_lo, _hi)
    else:
        _tk = [0.0, 1.0, round(1 + (VZ - 1) / 2), round(VZ)]
    _pad = 0.03 * (_hi - _lo)
    _min = _lo if nrm is not None else _lo + _pad
    cb.set_ticks(sorted({float(t) for t in _tk if _min <= t <= _hi - _pad}))
    cb.ax.tick_params(labelsize=TICK_PT, length=2, width=0.4, color=COAST,
                      direction="out")
    cb.outline.set(edgecolor=COAST, linewidth=0.4)
    _AXS.append(ax)
    _CAXS.append(cax)

# pull every colorbar back to the width of the map above it
fig.canvas.draw()
_inv = fig.transFigure.inverted()
for _ax, _cax in zip(_AXS, _CAXS):
    _pb = _inv.transform(_ax.get_window_extent().get_points())
    _cb = _cax.get_position()
    _cax.set_position([_pb[0][0], _cb.y0, _pb[1][0] - _pb[0][0], _cb.height])

STEM = f"fig6_confidently_wrong_{TAG}"
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{STEM}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=600)
plt.close(fig)

_d = open(OUT / f"{STEM}.pdf", "rb").read()
_mb = re.search(rb"/MediaBox\s*\[([^\]]*)\]", _d).group(1).split()
_w, _h = float(_mb[2]) / 72, float(_mb[3]) / 72
_scale = 0.99 * 3.4 / _w
print(f"wrote {STEM}.pdf/.png  {_w:.3f} x {_h:.3f} in; single-column scale "
      f"{_scale:.3f}x -> label {LABEL_PT * _scale:.2f} pt on the page")
json.dump(dict(tag=TAG, ic=str(Z["ic"]), source=str(SRC.name),
               figsize_in=[FIG_W, round(FIG_H, 3)], pdf_in=[round(_w, 3), round(_h, 3)],
               tex_scale_assumed=TEX_SCALE, tex_scale_measured=round(_scale, 4),
               caps=dict(error_degC=VE, spread_degC=VS, eta=VZ)),
          open(OUT / f"{STEM}_meta.json", "w"), indent=1)
print("FIG6_CONFIDENTLY_WRONG OK")
