#!/usr/bin/env python
"""Fig. 5: where each policy actually sampled -- forecast spread, then the five
policies as dot maps.

Six panels, 3 x 2:
  (a) free-running ensemble spread at cycle 39 (t+240 h), the policy-independent
      input ("what is uncertain if you never sample"), in the same Blues the
      other spread maps use
  (b)-(f) Static, Random, Lawnmower, D-optimal, A-optimal: every sampled cell
      over the 40 cycles as a dot (all picks, not unique cells, so Static's 50
      fixed sensors show as one overplotted lattice while the moving policies
      spread out)

Chain: MAB, SST, fleet size 50, IC 2022-04-09T12 (a sealed test IC),
observation seed 0. Inputs are the committed scores release only:
results/scores/locations/<policy>_chosen.csv and fig5_context.npz (wet mask,
cycle-39 spread at wet cells, grid scalars, display north/east). Dots are a
single dark navy with no time colour; view-from-shore convention throughout.
"""
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _shore import ANCHORS, Shore
from _style import BASE_RC, LABEL, POLICIES, argval, out_dir, scores_dir

SCORES = scores_dir()
LOC = SCORES / "locations"
OUT = out_dir(argval("--out", ""))

# type is specified at the size it should have ON THE PAGE, then divided by the
# scale LaTeX applies (0.473x for a single-column ieeeconf figure)
TEX_SCALE = float(argval("--texscale", "0.473"))
PT = lambda on_page: on_page / TEX_SCALE
DOT = "#16305e"
DOT_S = float(argval("--dots", "1.1"))
CM_SPREAD = "Blues"
CYCLE = 39

ctx = np.load(LOC / "fig5_context.npz")
shape = tuple(ctx["mask_shape"])
mask = np.unpackbits(ctx["mask_packed"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
spread = np.full(shape, np.nan, np.float32)
spread[mask] = ctx["spread39_wet"]
SH = Shore(str(ctx["region"]), mask, float(ctx["dx_col_m"]), float(ctx["dy_row_m"]),
           north=ctx["north"], east=ctx["east"])
ASPECT = SH.H / SH.W * SH.dy / SH.dx
vmax = float(np.nanpercentile(spread[mask], 99))

picks = {}
for p in POLICIES:
    with open(LOC / f"{p}_chosen.csv", newline="") as fh:
        picks[p] = np.array([[int(r["row"]), int(r["col"])]
                             for r in csv.DictReader(fh)])
    print(f"  {p:10s} picks {len(picks[p]):5d}  "
          f"unique cells {len(set(map(tuple, picks[p]))):5d}")

plt.rcParams.update({**BASE_RC, "font.size": 6.5, "image.interpolation": "nearest"})
FIG_W = 7.16
LEFT, RIGHT, WSPACE = 0.006, 0.994, 0.05
PAN_W = (RIGHT - LEFT) * FIG_W / (3 + 2 * WSPACE)
PAN_H = PAN_W * ASPECT
LABEL_PT = PT(5.9)
ROW_GAP, PAD = 0.055, 0.035
LAB_OFF_FRAC = 0.040
LAB_OFF, TEXT_H = LAB_OFF_FRAC * PAN_H, LABEL_PT / 72
CAP_H = 2 * LAB_OFF + TEXT_H - ROW_GAP
assert CAP_H > TEXT_H, "caption band is too short for its own text"
FIG_H = 2 * PAN_H + 2 * CAP_H + ROW_GAP + 2 * PAD
print(f"panel {PAN_W:.3f} x {PAN_H:.3f} in (aspect {ASPECT:.3f}); "
      f"figure {FIG_W} x {FIG_H:.3f} in")

fig = plt.figure(figsize=(FIG_W, FIG_H))
PANELS = [("spread", "ensemble spread")] + [(p, LABEL[p]) for p in POLICIES]
for i, (key, lab) in enumerate(PANELS):
    r, c = divmod(i, 3)
    top = FIG_H - PAD - r * (PAN_H + CAP_H + ROW_GAP)
    gs = fig.add_gridspec(1, 3, left=LEFT, right=RIGHT, wspace=WSPACE,
                          top=top / FIG_H, bottom=(top - PAN_H) / FIG_H)
    ax = fig.add_subplot(gs[0, c])
    if key == "spread":
        ax.imshow(SH.fill_to_coast(spread), cmap=CM_SPREAD, vmin=0.0, vmax=vmax,
                  origin="upper", zorder=1)
        SH.base(ax, coast_lw=0.25, spine_lw=0.35)
    else:
        SH.base(ax, ocean=True, coast_lw=0.25, spine_lw=0.35)   # pale ocean, grey land
        x, y = SH.xy(picks[key][:, 0], picks[key][:, 1])
        ax.scatter(x, y, s=DOT_S, c=DOT, marker="o", linewidths=0, zorder=5)
    if i == 0:
        SH.borders(ax, lw=0.25)
        ax.apply_aspect()
        _bb = ax.get_position(original=False)
        _r = _bb.height * fig.get_figheight() / 0.9326
        SH.furniture(ax, fig, size=0.48 * _r, tag="panel (a)",
                     anchors=dict(ANCHORS[SH.region]), letters="NESW",
                     label_fs=3.6 * _r, bar_lw=0.7 * _r)
        # no colorbar: panel (a) is qualitative context; the range is in the caption
    ax.text(0.5, -LAB_OFF_FRAC, f"({chr(97 + i)}) {lab}", transform=ax.transAxes,
            ha="center", va="top", fontsize=LABEL_PT)

STEM = "fig5_locations"
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"{STEM}.{ext}", dpi=400, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
json.dump(dict(slice=str(ctx["slice"]), region="mab", variable="SST", nfleet=50,
               ic="2022-04-09T12", is_sealed_test_ic=True, cycles=40,
               panels=[p[1] for p in PANELS],
               spread_panel=f"free-running (nosample) ensemble spread at cycle "
                            f"{CYCLE} (t+240 h); colour 0..{vmax:.4f} degC "
                            "(99th pct over wet cells); no colorbar drawn",
               spread_colour_range_degC=[0.0, round(vmax, 4)],
               dots="every pick over 40 cycles (not unique cells); single dark "
                    "navy, no time colour",
               picks={p: dict(total=int(len(picks[p])),
                              unique=int(len(set(map(tuple, picks[p])))))
                      for p in POLICIES},
               figsize_in=[FIG_W, round(FIG_H, 3)],
               panel_in=[round(PAN_W, 4), round(PAN_H, 4)],
               convention="view from shore; mab fliplr, det +1",
               inputs="results/scores/locations/ only"),
          open(OUT / f"{STEM}_meta.json", "w"), indent=1)
print(f"wrote {STEM}.pdf/.png under {OUT}")
print("FIG5_LOCATIONS OK")
