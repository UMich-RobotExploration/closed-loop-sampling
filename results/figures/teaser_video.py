#!/usr/bin/env python
"""The loop video: one 3 x 5 animation of a full 40-cycle campaign.

    row 1   truth | member 1 | member 2 | member 3 | free-running mean
    row 2   Static | Random | Lawnmower | D-optimal | A-optimal -- each policy's
            robots, reachable set and picks, drawn on that policy's own
            ensemble spread
    row 3   |ensemble mean - truth| for each of those five policies

Style follows the members figure: cmocean thermal/haline for the field, Blues
for spread, Reds for error, view-from-shore maps, compass and 100 km bar on the
first panel, one thin two-tick colorbar per row. Scales are fixed for the whole
run so motion is the only thing that changes. Phases per 2 s cycle: forecast
crossfade -> reachable sets in -> glide -> observe.

Known limits, recorded rather than papered over: the glide is the straight
fleet[t] -> fleet[t+1] path (the advection field is not dumped, so drift and
propulsion are not separable), and the spread shown is the PRIOR ensemble's (no
analysis spread field is dumped).

Inputs: $CLOSEDLOOP_ARTIFACTS/render_dumps/<slice>/<policy>.npz (position and
field dumps of the six chains) and figure_fields/video_members_<slice>.npz
(three decoded members per cycle plus grid geometry). Frames go to
--out/frames_<slice>/; --encode assembles an mp4 and a short GIF with ffmpeg
when it is on PATH, otherwise the commands are printed.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from _shore import ANCHORS, COAST, Shore
from _style import BASE_RC, LABEL, POLICIES, artifacts_dir, out_dir, variable_cmap

ARMS = POLICIES
ROW1 = ["truth", "member 1", "member 2", "member 3", "free-running mean"]
INK, GREY, LAND, PICK = "0.22", "0.45", "#e8e8e8", "0.30"
MK_ROB, MK_PICK, LW_MK = 9.8, 19.7, 0.63
COAST_LW = 0.75
FIG1_PANEL_H = 0.9326

ap = argparse.ArgumentParser()
ap.add_argument("--slice", default="mab_sst_ic20220409T12_n50",
                help="render-dump slice under $CLOSEDLOOP_ARTIFACTS/render_dumps/")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--cycle-sec", type=float, default=2.0)
ap.add_argument("--figw", type=float, default=16.0)
ap.add_argument("--coastlw", type=float, default=COAST_LW)
ap.add_argument("--outline", action="store_true", help="one frame, then stop")
ap.add_argument("--outline-cycle", type=int, default=20)
ap.add_argument("--cycles", default="", help="c0:c1 subrange")
ap.add_argument("--out", default="")
ap.add_argument("--encode", action="store_true",
                help="assemble mp4 + teaser GIF with ffmpeg after rendering")
A = ap.parse_args()
COAST_LW = A.coastlw
region, variable = A.slice.split("_")[:2]
CM_F = variable_cmap(variable).copy()
CM_F.set_bad(LAND)
CM_S = matplotlib.colormaps["Blues"].copy()
CM_S.set_bad(LAND)
CM_E = matplotlib.colormaps["Reds"].copy()
CM_E.set_bad(LAND)
UNIT = {"sst": r"$^\circ$C", "sss": "psu"}[variable]

ART = artifacts_dir()
OUT = out_dir(A.out)
FR = OUT / f"frames_{A.slice}"
FR.mkdir(parents=True, exist_ok=True)

# ---- data ----------------------------------------------------------------------
DUMPS = ART / "render_dumps" / A.slice
Z = {a: np.load(DUMPS / f"{a}.npz", allow_pickle=True)
     for a in ARMS + ["free_running"]}
NC = len(Z["free_running"]["cycle"])

# guard against pairing fleet (robot-indexed) with chosen_rc (planner-indexed)
_z0 = Z["free_running"]
_rad_cells = (float(_z0["v_auv"]) * float(_z0["reach_window_s"])
              / min(float(_z0["dx_col_m"]), float(_z0["dy_row_m"])))
for _a in ARMS:
    _fl = Z[_a]["fleet"]
    _st = max(float(np.linalg.norm(np.asarray(_fl[t + 1], float)
                                   - np.asarray(_fl[t], float), axis=1).max())
              for t in range(NC - 1))
    assert _st <= 5.0 * _rad_cells, (
        f"{_a}: a robot moves {_st:.1f} cells in one cycle against a "
        f"{_rad_cells:.1f}-cell propulsion reach -- fleet and chosen_rc are "
        "probably being paired by index")
mask = Z["free_running"]["mask"]
LEAD = np.asarray(Z["free_running"]["lead_h"])

MEMP = ART / "figure_fields" / f"video_members_{A.slice}.npz"
MEMZ = np.load(MEMP, allow_pickle=True)
MEM = MEMZ["members"]                                  # (3, NC, H, W)
assert MEM.shape[1] == NC
SH = Shore.from_grid(region, mask, MEMZ["lon"], MEMZ["lat"],
                     float(MEMZ["dx_m"]), float(MEMZ["dy_m"]))
TRUTH = np.asarray(Z["free_running"]["truth"], float)
FMEAN = np.asarray(Z["free_running"]["prior_mean"], float)
FSPR = np.asarray(Z["free_running"]["spread"], float)
print(f"{A.slice}: {NC} cycles, grid {SH.H}x{SH.W}, members {MEM.shape}", flush=True)


def ERR(a, which):
    return np.abs(np.asarray(Z[a][which], float) - np.asarray(Z[a]["truth"], float))


def pct(stack, q):
    return float(np.nanpercentile(np.concatenate([s[..., mask].ravel()
                                                  for s in stack]), q))


_F1 = [TRUTH, FMEAN, MEM[0], MEM[1], MEM[2]]
P_LO = 5 if variable == "sss" else 1       # the 1st-pct floor crushes SSS plumes
V_F = (pct(_F1, P_LO), pct(_F1, 99))
V_S = (0.0, pct([np.asarray(Z[a]["spread"], float) for a in ARMS] + [FSPR], 99))
V_E = (0.0, pct([ERR(a, w) for a in ARMS for w in ("prior_mean", "post_mean")], 99))
print(f"  field {V_F[0]:.1f}..{V_F[1]:.1f}  spread 0..{V_S[1]:.2f}  "
      f"error 0..{V_E[1]:.2f}", flush=True)

# ---- geometry --------------------------------------------------------------------
plt.rcParams.update({**BASE_RC, "font.size": 11, "image.interpolation": "nearest"})
RAW = tuple(TRUTH.shape[1:])
assert SH.apply(np.zeros(RAW)).shape == (SH.H, SH.W)
ASPECT = SH.H / SH.W * SH.dy / SH.dx
DPI = 160
LEFT, RIGHT, WSP, HSP = 0.020, 0.955, 0.012, 0.330
TITLE_IN, PAD_IN = 0.62, 0.06
FIG_W = A.figw
PAN_W = (RIGHT - LEFT) * FIG_W / (5 + 4 * WSP)
PAN_H = PAN_W * ASPECT
FIG_H = TITLE_IN + 3 * PAN_H + 2 * (HSP * PAN_H) + 2 * PAD_IN
W_PX, H_PX = int(FIG_W * DPI), int(FIG_H * DPI)
W_PX += W_PX % 2
H_PX += H_PX % 2
fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
X = lambda v: v / FIG_W
Y = lambda v: 1 - v / FIG_H
print(f"  frame {W_PX}x{H_PX} px; panel {PAN_W:.3f} x {PAN_H:.3f} in", flush=True)

AX = []
for r in range(3):
    y0 = TITLE_IN + r * (PAN_H + HSP * PAN_H)
    row = []
    for c in range(5):
        x0 = LEFT * FIG_W + c * (PAN_W + WSP * PAN_W)
        ax = fig.add_axes([X(x0), Y(y0 + PAN_H), X(PAN_W), PAN_H / FIG_H])
        ax.set_xticks([])
        ax.set_yticks([])
        row.append(ax)
    AX.append(row)
TITLE = fig.text(0.5, Y(0.19), "", ha="center", va="center", fontsize=19,
                 color="0.15")
ROWLAB = ["", "spread + sampling", "ensemble-mean error"]

_blank = np.full(RAW, np.nan)
IM = [[None] * 5 for _ in range(3)]
for c in range(5):
    IM[0][c] = AX[0][c].imshow(SH.fill_to_coast(_blank), cmap=CM_F, vmin=V_F[0],
                               vmax=V_F[1], origin="upper", zorder=1)
    IM[1][c] = AX[1][c].imshow(SH.fill_to_coast(_blank), cmap=CM_S, vmin=V_S[0],
                               vmax=V_S[1], origin="upper", zorder=1)
    IM[2][c] = AX[2][c].imshow(SH.fill_to_coast(_blank), cmap=CM_E, vmin=V_E[0],
                               vmax=V_E[1], origin="upper", zorder=1)
OV = []
CM_REACH = ListedColormap([(0, 0, 0, 0), (0.35, 0.35, 0.35, 0.40)])
CM_EDGE = ListedColormap([(0, 0, 0, 0), (0.42, 0.42, 0.42, 1.0)])
for c in range(5):
    ax = AX[1][c]
    OV.append(dict(
        reach=ax.imshow(np.zeros((SH.H, SH.W)), cmap=CM_REACH, vmin=0, vmax=1,
                        origin="upper", alpha=0.0, zorder=4,
                        interpolation="nearest"),
        edge=ax.imshow(np.zeros((SH.H, SH.W)), cmap=CM_EDGE, vmin=0, vmax=1,
                       origin="upper", alpha=0.0, zorder=5,
                       interpolation="nearest"),
        pick=ax.scatter([], [], s=MK_PICK, c=PICK, linewidths=0, zorder=6,
                        alpha=0.0),
        rob=ax.scatter([], [], s=MK_ROB, facecolors="white", edgecolors=GREY,
                       linewidths=LW_MK, zorder=7)))
for r in range(3):
    for c in range(5):
        SH.base(AX[r][c], coast_lw=COAST_LW, spine_lw=round(1.4 * COAST_LW, 3))
SH.borders(AX[0][0], lw=0.3)
AX[0][0].apply_aspect()
_r = AX[0][0].get_position(original=False).height * fig.get_figheight() / FIG1_PANEL_H
SH.furniture(AX[0][0], fig, size=0.48 * _r, tag="first panel",
             anchors=dict(ANCHORS[region]), letters="NESW", label_fs=3.6 * _r,
             bar_lw=0.7 * _r)
for c, t in enumerate(ROW1):
    AX[0][c].set_title(t, fontsize=13, color="0.15", pad=5)
for r, cap in enumerate(ROWLAB):
    if cap:
        AX[r][0].text(-0.022, 0.5, cap, transform=AX[r][0].transAxes, rotation=90,
                      ha="right", va="center", fontsize=12, color="0.15")
for c, a in enumerate(ARMS):
    AX[1][c].set_title(LABEL[a], fontsize=13, color="0.15", pad=5)

fig.canvas.draw()
_inv = fig.transFigure.inverted()
CB_W = 0.011
for r, (im, lo, hi, nd) in enumerate(((IM[0][0], V_F[0], V_F[1], 1),
                                      (IM[1][0], V_S[0], V_S[1], 2),
                                      (IM[2][0], V_E[0], V_E[1], 2))):
    pb = _inv.transform(AX[r][4].get_window_extent().get_points())
    cax = fig.add_axes([pb[1][0] + 0.007, pb[0][1], CB_W, pb[1][1] - pb[0][1]])
    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([lo, hi])
    cax.set_yticklabels([f"{lo:.{nd}f}", f"{hi:.{nd}f}"])
    cax.tick_params(labelsize=9.5, length=1.8, width=0.4, color=COAST, pad=1.6)
    cb.outline.set(edgecolor=COAST, linewidth=0.4)
    p = cax.get_position()
    fig.text(p.x0 + p.width / 2, p.y1 + 0.004, UNIT, ha="center", va="bottom",
             fontsize=9.5, color="0.30")

# ---- phases ----------------------------------------------------------------------
NF = int(round(A.fps * A.cycle_sec))
PH_F = [("forecast", int(0.20 * NF)), ("reach", int(0.13 * NF)),
        ("glide", int(0.47 * NF))]
PH_F.append(("observe", NF - sum(n for _, n in PH_F)))


def reach_mask(a, t):
    m = np.zeros(RAW, bool)
    rc = np.asarray(Z[a]["cand_rc"][t], int)
    if len(rc):
        m[rc[:, 0], rc[:, 1]] = True
    inner = np.ones_like(m)
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return SH.apply(m).astype(np.float32), SH.apply(m & ~inner).astype(np.float32)


def lerp(u, v, s):
    return (1.0 - s) * u + s * v


def draw(t, phase, s):
    IM[0][0].set_data(SH.fill_to_coast(TRUTH[t]))
    for j in range(3):
        IM[0][1 + j].set_data(SH.fill_to_coast(MEM[j, t]))
    prevF = FMEAN[t - 1] if t > 0 else FMEAN[t]
    IM[0][4].set_data(SH.fill_to_coast(lerp(prevF, FMEAN[t], s)
                                       if phase == "forecast" else FMEAN[t]))
    for c, a in enumerate(ARMS):
        tru = np.asarray(Z[a]["truth"][t], float)
        ep = np.abs(np.asarray(Z[a]["prior_mean"][t], float) - tru)
        eq = np.abs(np.asarray(Z[a]["post_mean"][t], float) - tru)
        prev_e = (np.abs(np.asarray(Z[a]["post_mean"][t - 1], float)
                         - np.asarray(Z[a]["truth"][t - 1], float)) if t > 0 else ep)
        spn = np.asarray(Z[a]["spread"][t], float)
        spp = np.asarray(Z[a]["spread"][t - 1], float) if t > 0 else spn
        IM[1][c].set_data(SH.fill_to_coast(lerp(spp, spn, s)
                                           if phase == "forecast" else spn))
        IM[2][c].set_data(SH.fill_to_coast(
            lerp(prev_e, ep, s) if phase == "forecast" else
            lerp(ep, eq, s) if phase == "observe" else ep))
        f0 = np.asarray(Z[a]["fleet"][t], float)
        has_reach = len(np.asarray(Z[a]["cand_rc"][t], int)) > 0
        u, e = reach_mask(a, t - 1 if (phase == "forecast" and t > 0) else t)
        OV[c]["reach"].set_data(u)
        OV[c]["edge"].set_data(e)
        al = (0.0 if not has_reach else
              (1.0 - s) if phase == "forecast" else
              s if phase == "reach" else 1.0)
        OV[c]["reach"].set_alpha(al)
        OV[c]["edge"].set_alpha(al)
        # chosen_rc[t] holds the same SET as fleet[t+1] but in planner order, so
        # it is never a per-robot glide target; the dump has no 40th leg: hold.
        f1 = (np.asarray(Z[a]["fleet"][t + 1], float) if t + 1 < NC else f0)
        g = 0.0 if phase in ("forecast", "reach") else (s if phase == "glide" else 1.0)
        pos = lerp(f0, f1, g)
        OV[c]["rob"].set_offsets(np.c_[SH.xy(pos[:, 0], pos[:, 1])])
        ch = np.asarray(Z[a]["chosen_rc"][t - 1 if (phase == "forecast" and t > 0)
                                          else t], float)
        OV[c]["pick"].set_offsets(np.c_[SH.xy(ch[:, 0], ch[:, 1])])
        OV[c]["pick"].set_alpha(min(1.0, 2.2 * s) if phase == "observe"
                                else (1.0 - s) if phase == "forecast" else 0.0)
    TITLE.set_text(f"t + {int(LEAD[t])} h")


if A.outline:
    draw(A.outline_cycle, "observe", 1.0)
    p = OUT / f"teaser_outline_{A.slice}.png"
    fig.savefig(p, dpi=DPI, facecolor="white")
    print(f"wrote {p} ({W_PX}x{H_PX}), cycle {A.outline_cycle}")
    print("TEASER_OUTLINE OK")
    raise SystemExit(0)

c0, c1 = (0, NC) if not A.cycles else tuple(int(v) for v in A.cycles.split(":"))
CYC = list(range(c0, min(c1, NC)))
t_start, nf = time.time(), 0
for t in CYC:
    for phase, n in PH_F:
        for k in range(n):
            draw(t, phase, (k + 1) / n)
            fig.savefig(FR / f"f{nf:06d}.png", dpi=DPI, facecolor="white")
            nf += 1
    if (t - CYC[0]) % 5 == 0:
        print(f"  cycle {t:2d}  {nf:5d} frames  "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
plt.close(fig)
el = time.time() - t_start
json.dump(dict(slice=A.slice, layout="3 rows x 5 cols", row1=ROW1,
               row2=[LABEL[a] for a in ARMS],
               cycles=[CYC[0], CYC[-1]], frames=nf, fps=A.fps,
               cycle_sec=A.cycle_sec, frame_px=[W_PX, H_PX], phases=dict(PH_F),
               scales=dict(field=list(V_F), spread=list(V_S), error=list(V_E)),
               spread_is="prior (forecast) ensemble sd; no analysis spread "
                         "field is dumped",
               glide="straight fleet[t] -> fleet[t+1]; drift and propulsion "
                     "are not separable from the dump",
               wall_clock_min=round(el / 60, 2)),
          open(OUT / f"teaser_manifest_{A.slice}.json", "w"), indent=1)
print(f"wrote {nf} frames to {FR} in {el / 60:.1f} min ({W_PX}x{H_PX})")

mp4 = OUT / f"loop_3x5_{A.slice}_1920w.mp4"
gif = OUT / f"teaser_{A.slice}.gif"
cmd_mp4 = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(A.fps),
           "-i", str(FR / "f%06d.png"), "-vf", "scale=1920:-2:flags=lanczos",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
           "-movflags", "+faststart", "-map_metadata", "-1", str(mp4)]
cmd_gif = ["ffmpeg", "-y", "-loglevel", "error", "-start_number",
           str(18 * NF), "-framerate", str(A.fps), "-i", str(FR / "f%06d.png"),
           "-frames:v", str(int(15 * A.fps)), "-vf",
           "fps=10,scale=640:-2:flags=lanczos,split[a][b];[a]palettegen[p];"
           "[b][p]paletteuse", "-map_metadata", "-1", str(gif)]
if A.encode and shutil.which("ffmpeg"):
    subprocess.run(cmd_mp4, check=True)
    subprocess.run(cmd_gif, check=True)
    print(f"encoded {mp4} and {gif}")
else:
    print("to encode:")
    print("  " + " ".join(cmd_mp4))
    print("  " + " ".join(cmd_gif))
print("TEASER_VIDEO_FRAMES OK")
