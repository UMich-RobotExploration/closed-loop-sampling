#!/usr/bin/env python
"""Build the scores release (results/scores/) from the production run records.

This is the auditable extraction pipeline behind every number in results/:
it reads the production trees READ-ONLY and writes plain CSV files. Everything
downstream (results/scoring, results/tables, results/figures) reads only the
files this script writes, never the production trees.

Inputs (all under --source-root, the author's production project):
  <src>/phase3_sandbox_ipp/q3_field_uncertainty/results/q3_seed_ic*.json
      the 2,318 registered run records (40 cycles each; the 20 derived
      free-running SSS records carry region/ic keys instead of CLI echoes)
  <src>/phase3_sandbox_ipp/q3_field_uncertainty/results/q3_render_<slice>_<arm>.npz
      the position dumps of the chains Fig. 5 uses
  <src>/phase3_sandbox_ipp/q3_field_uncertainty/_mlxor_workshop/noise_floor/rep{1..4}/results/
      replicate chains (run-to-run floor measurement)
  <src>/phase3_sandbox_ipp/q3_field_uncertainty/_mlxor_workshop/verify/members_fields_doppio.npz
      grid geometry (lon/lat) for the Fig. 5 drawing context
  <src>/phase3_sandbox_ipp/icra_desk/c4/{c4_emitter.json,r240_table.json}
      emitted floors (recomputed here and asserted equal) and spread-error r
  <src>/phase3_sandbox_ipp/q3_field_uncertainty/logs/*.out
      sbatch logs; per-chain wall-clock joined on the first-line command

Outputs (results/scores/):
  manifest.csv              one row per record file, reconciled to 2,318
  runs.csv                  one row per main + fleet-sweep run (2,250)
  runs_per_cycle.csv.gz     one row per (run, cycle) for those runs (90,000)
  references/free_running.csv   per-cycle free-running RMSE (40 chains)
  references/ae_floor.csv       per-cycle autoencoder floor (per region/variable/IC)
  references/tolerance.csv      rerun floors per slice (recomputed + asserted) and
                                the replicate-chain inventory
  references/spread_error_r.csv spread-error correlation r at 144 h and 240 h
  locations/<policy>_{chosen,fleet}.csv  per-cycle positions of the Fig. 5 chains
  locations/fig5_context.npz    drawing context for Fig. 5 (wet mask, cycle-39
                                free-running spread at wet cells, grid scalars)

Naming: regions are released as mab/ccs (production key "doppio" = MAB) and
policies as static/random/lawnmower/d_optimal/a_optimal/free_running; the
source_file column preserves the production filename for auditability.

Wall-clock: the cost ledger holds only per-configuration aggregates, so wall_s
is re-derived here exactly the way that ledger was built: each sbatch log whose
first line is a seedrun command and whose tail prints WALL_S= and the OK
sentinel is parsed, and a run's wall_s is the WALL_S of the most recent such
log matching the run's full parameter tuple. Replicate reruns share the same
command line, so wall_s_n_logs records the multiplicity; wall_s is blank when
no log survives.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

import numpy as np

REGION_OUT = {"doppio": "mab", "ccs": "ccs"}
VARIABLE_OUT = {"temp": "sst", "salt": "sss"}
POLICY_OUT = {"static": "static", "random": "random", "lawn": "lawnmower",
              "dopt": "d_optimal", "trace": "a_optimal", "nosample": "free_running"}
FLEETS_MAIN = (20, 50, 200)
FLEETS_SWEEP = (10, 35, 100)
WINDOWS = {"last6": 6, "last8": 8}

REC_RX = re.compile(
    r"^q3_seed_ic(\d{8}T\d{2})_(doppio|ccs)_L0(_salt)?_n(\d+)_([a-z]+)(_matched)?_s(\d)\.json$")
LOG_RX = re.compile(
    r"--region=(\w+) --ic_datetime=(\S+) --nfleet=(\d+) --scale=(\S+) --ncyc=40 "
    r"--start_lead=0 --strategy=(\w+) --rseed=(\d+) --seedrun(.*)")

SERVES = {
    "main": "Table I; Table II; Table III; Fig. 3",
    "main_sweep_cell": "Table I; Table II; Table III; Fig. 3; Fig. 4",
    "fleet_sweep": "Fig. 4",
    "free_running": ("free-running reference: Figs. 3-4 baselines, Fig. 2 and "
                     "Fig. 6 fields, spread-error r"),
    "sensitivity_matched_sigma": ("matched observation-noise sensitivity slice; "
                                  "no submitted table or figure"),
    "excluded_stray": "none (excluded)",
}


def iso_ic(tag: str) -> str:
    return f"{tag[:4]}-{tag[4:6]}-{tag[6:8]}T{tag[9:11]}"


def fnum(x) -> str:
    """Full-precision decimal text for a float; empty for None."""
    return "" if x is None else repr(float(x))


def window_mean(records, w: int, key: str = "post_rmse") -> float:
    return float(np.mean([r[key] for r in records][-w:]))


def write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", required=True,
                    help="production project root (read-only)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <repo>/results/scores)")
    ap.add_argument("--fig5-slice", default="ic20220409T12_doppio_L0_n50",
                    help="render-dump slice whose positions Fig. 5 draws")
    args = ap.parse_args()

    src = Path(args.source_root)
    p3 = src / "phase3_sandbox_ipp"
    res = p3 / "q3_field_uncertainty/results"
    mw = p3 / "q3_field_uncertainty/_mlxor_workshop"
    c4 = p3 / "icra_desk/c4"
    logs = p3 / "q3_field_uncertainty/logs"
    repo = Path(__file__).resolve().parents[1]
    out = Path(args.out) if args.out else repo / "results/scores"
    out.mkdir(parents=True, exist_ok=True)

    sealed = {}
    for prod, rel in (("doppio", "mab"), ("ccs", "ccs")):
        sealed[prod] = set(json.load(open(
            repo / f"configs/experiments/ics/test_{rel}.json"))["canonical_order"])

    # ---------------- pass 1: parse every record file ----------------
    files = sorted(res.glob("q3_seed_ic*.json"))
    rows_manifest, rows_runs, rows_cycles = [], [], []
    rows_free, rows_floor = [], []
    floor_by = {}          # (region, variable, ic) -> 40-cycle floor list (first seen)
    floor_mismatch = []
    counts = {k: 0 for k in ("main", "fleet_sweep", "free_running",
                             "sensitivity_matched_sigma", "excluded_stray")}

    # wall-clock join (see module docstring)
    wall = {}
    n_logs = 0
    for lf in sorted(logs.glob("*.out")):
        try:
            with open(lf, errors="replace") as fh:
                head = fh.readline()
                if "q3_field_coverage.py" not in head or "--seedrun" not in head:
                    continue
                m = LOG_RX.search(head)
                if not m:
                    continue
                body = fh.read()
        except OSError:
            continue
        w = re.findall(r"WALL_S=(\d+)", body)
        if not w or "Q3_SEEDRUN OK" not in body:
            continue
        reg, ic, n, _sc, arm, rs, rest = m.groups()
        extra = [t for t in rest.split() if t != "--target=salt"]
        if extra:
            continue      # off-protocol flags (e.g. the matched-sigma slice): not a registered run
        tgt = "salt" if "--target=salt" in rest else "temp"
        wall.setdefault((reg, ic[:13], int(n), arm, int(rs), tgt), []).append(int(w[-1]))
        n_logs += 1
    print(f"wall-clock join: {n_logs} OK seedrun logs parsed from {logs.name}/")

    for f in files:
        m = REC_RX.match(f.name)
        assert m, f"unrecognized record filename: {f.name}"
        tag, region, salt, n, arm, matched, seed = m.groups()
        ic = iso_ic(tag)
        variable = "salt" if salt else "temp"
        fleet, seed = int(n), int(seed)
        data = json.load(open(f))
        rec = data["records"]
        assert len(rec) == 40, f"{f.name}: {len(rec)} cycles"
        derived = "derived" in data

        note = ""
        if matched:
            category = "sensitivity_matched_sigma"
        elif arm == "nosample":
            category = "free_running"
        elif ic not in sealed[region]:
            category = "excluded_stray"
            note = ("tuning-year IC (not in the sealed test roster), scale "
                    f"{data.get('scale_eff')}; never read by the paper pipeline")
        elif region == "doppio" and variable == "salt" and fleet in FLEETS_SWEEP:
            category = "fleet_sweep"
        else:
            assert fleet in FLEETS_MAIN, f.name
            category = "main"
        counts[category] += 1

        serves = SERVES[category]
        if category == "main" and region == "doppio" and variable == "salt":
            serves = SERVES["main_sweep_cell"]

        rows_manifest.append([
            f.name, category, REGION_OUT[region], VARIABLE_OUT[variable], fleet,
            ic, POLICY_OUT[arm], "" if derived else seed,
            fnum(data.get("scale_eff")) if arm != "nosample" else "",
            serves, note])

        # per-cycle AE floor: identical across arms, seeds and fleets of one
        # (region, variable, IC) -- asserted here, recorded once
        fl = [r["floor"] for r in rec]
        key3 = (region, variable, ic)
        if key3 not in floor_by:
            floor_by[key3] = fl
            if category != "excluded_stray":
                for r in rec:
                    rows_floor.append([REGION_OUT[region], VARIABLE_OUT[variable],
                                       ic, r["cycle"], fnum(r["floor"])])
        elif fl != floor_by[key3]:
            floor_mismatch.append(f.name)

        if category == "free_running":
            for r in rec:
                assert r["prior_rmse"] == r["post_rmse"], f.name
                rows_free.append([REGION_OUT[region], VARIABLE_OUT[variable],
                                  ic, r["cycle"], fnum(r["prior_rmse"])])
            continue

        if category in ("main", "fleet_sweep"):
            ws = wall.get((region, ic, fleet, arm, seed, variable), [])
            rows_runs.append([
                REGION_OUT[region], VARIABLE_OUT[variable], fleet, ic,
                POLICY_OUT[arm], seed, fnum(data["scale_eff"]),
                fnum(window_mean(rec, 8)), fnum(window_mean(rec, 6)),
                ws[-1] if ws else "", len(ws)])
            for r in rec:
                rows_cycles.append([
                    REGION_OUT[region], VARIABLE_OUT[variable], fleet, ic,
                    POLICY_OUT[arm], seed, r["cycle"], fnum(r["prior_rmse"]),
                    fnum(r["post_rmse"]), fnum(r["floor"]), r["n_obs"]])

    # ---------------- reconcile the census ----------------
    assert len(rows_manifest) == 2318, len(rows_manifest)
    expected = dict(main=1800, fleet_sweep=450, free_running=40,
                    sensitivity_matched_sigma=25, excluded_stray=3)
    assert counts == expected, counts
    assert len(rows_runs) == 2250 and len(rows_cycles) == 2250 * 40
    assert not floor_mismatch, f"AE floor differs across chains: {floor_mismatch[:5]}"
    n_wall = sum(1 for r in rows_runs if r[9] != "")
    print(f"census reconciled: {counts}; wall_s joined for {n_wall}/2250 runs")

    sort_key = lambda r: (r[0], r[1], r[2], r[3], r[4], r[5])
    rows_runs.sort(key=sort_key)
    rows_cycles.sort(key=lambda r: (r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    rows_free.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    rows_floor.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    write_csv(out / "manifest.csv",
              ["source_file", "category", "region", "variable", "fleet", "ic",
               "policy", "seed", "scale_eff", "serves", "note"], rows_manifest)
    write_csv(out / "runs.csv",
              ["region", "variable", "fleet", "ic", "policy", "seed", "scale_eff",
               "rmse_last8", "rmse_last6", "wall_s", "wall_s_n_logs"], rows_runs)
    # one logical table, split per (region, variable) to keep every tracked
    # file under the release's 1 MB cap
    hdr = ["region", "variable", "fleet", "ic", "policy", "seed", "cycle",
           "prior_rmse", "post_rmse", "floor", "n_obs"]
    for region in ("ccs", "mab"):
        for variable in ("sss", "sst"):
            part = [r for r in rows_cycles if r[0] == region and r[1] == variable]
            pc = out / f"runs_per_cycle_{region}_{variable}.csv.gz"
            with gzip.open(pc, "wt", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(hdr)
                wr.writerows(part)
            print(f"  wrote {pc} ({len(part)} rows)")
    write_csv(out / "references/free_running.csv",
              ["region", "variable", "ic", "cycle", "rmse"], rows_free)
    write_csv(out / "references/ae_floor.csv",
              ["region", "variable", "ic", "cycle", "floor"], rows_floor)

    # ---------------- tolerance floors: recompute, assert, release ----------------
    em = json.load(open(c4 / "c4_emitter.json"))["floors"]
    rows_tol = []
    for region in ("doppio", "ccs"):
        for vn, tt in (("SST", ""), ("SSS", "_salt")):
            for k in FLEETS_MAIN:
                devs = {w: [] for w in WINDOWS}
                rep_counts = {rp: 0 for rp in (1, 2, 3, 4)}
                n_cells = 0
                for ic in sorted(sealed[region]):
                    tag = ic.replace("-", "").replace(":", "")[:11]
                    name = f"q3_seed_ic{tag}_{region}_L0{tt}_n{k}_trace_s0.json"
                    reg_f = res / name
                    reps = {}
                    for rp in (1, 2, 3, 4):
                        p = mw / f"noise_floor/rep{rp}/results" / name
                        if p.exists():
                            rr = json.load(open(p))["records"]
                            if len(rr) == 40:
                                reps[rp] = rr
                    if not reps or not reg_f.exists():
                        continue
                    n_cells += 1
                    reg_rec = json.load(open(reg_f))["records"]
                    for rp, rr in reps.items():
                        rep_counts[rp] += 1
                        for w, W in WINDOWS.items():
                            devs[w].append(abs(window_mean(rr, W) - window_mean(reg_rec, W)))
                key = f"{region}/{vn}/n{k}"
                flo = {w: max(devs[w]) for w in WINDOWS}
                for w in WINDOWS:
                    assert abs(flo[w] - em[key]["floor"][w]) < 1e-12, (key, w)
                assert n_cells == em[key]["n_cells"]
                rows_tol.append([
                    REGION_OUT[region], vn.lower(), k,
                    fnum(flo["last8"]), fnum(flo["last6"]),
                    fnum(float(np.median(devs["last8"]))),
                    fnum(float(np.median(devs["last6"]))),
                    n_cells, sum(rep_counts.values()),
                    rep_counts[1], rep_counts[2], rep_counts[3], rep_counts[4],
                    em[key]["argmax"]["last8"][0], em[key]["argmax"]["last8"][1]])
    write_csv(out / "references/tolerance.csv",
              ["region", "variable", "fleet", "floor_last8", "floor_last6",
               "median_abs_dev_last8", "median_abs_dev_last6", "n_ics",
               "n_replicate_chains", "n_rep1", "n_rep2", "n_rep3", "n_rep4",
               "argmax_ic_last8", "argmax_rep_last8"], rows_tol)
    print("tolerance floors recomputed from the replicate chains and asserted "
          "equal to the emitted ledger")

    # ---------------- spread-error r ----------------
    r240 = json.load(open(c4 / "r240_table.json"))["regions"]
    rows_r = []
    for region in ("doppio", "ccs"):
        for ic in sorted(sealed[region]):
            for vn in ("SST", "SSS"):
                e = r240[region]["ics"][ic][vn]
                rows_r.append([REGION_OUT[region], vn.lower(), ic,
                               fnum(e["144h"]), fnum(e["240h"])])
    write_csv(out / "references/spread_error_r.csv",
              ["region", "variable", "ic", "r_144h", "r_240h"], rows_r)

    # ---------------- Fig. 5 locations + drawing context ----------------
    loc = out / "locations"
    loc.mkdir(exist_ok=True)
    slice_ = args.fig5_slice
    arms = ("static", "random", "lawn", "dopt", "trace")
    Z = {a: np.load(res / f"q3_render_{slice_}_{a}.npz", allow_pickle=True)
         for a in arms + ("nosample",)}
    mask = Z["trace"]["mask"]
    for a in arms:
        rows_c, rows_f = [], []
        for c in range(len(Z[a]["cycle"])):
            for rc in np.atleast_2d(np.asarray(Z[a]["chosen_rc"][c], int)):
                rows_c.append([c, int(rc[0]), int(rc[1])])
            for j, rc in enumerate(np.atleast_2d(np.asarray(Z[a]["fleet"][c], float))):
                rows_f.append([c, j, int(round(rc[0])), int(round(rc[1]))])
        write_csv(loc / f"{POLICY_OUT[a]}_chosen.csv", ["cycle", "row", "col"], rows_c)
        write_csv(loc / f"{POLICY_OUT[a]}_fleet.csv",
                  ["cycle", "robot", "row", "col"], rows_f)

    # drawing context: wet mask, free-running spread at cycle 39 (wet cells,
    # f32), metric cell sizes, and the display north/east unit vectors
    # (precomputed from the grid lon/lat so the committed file stays small)
    geo = np.load(mw / "verify/members_fields_doppio.npz")
    assert np.array_equal(geo["mask"], mask)
    lon = np.asarray(geo["lon"], float)[:, ::-1]   # display orientation (fliplr)
    lat = np.asarray(geo["lat"], float)[:, ::-1]
    i, j = lon.shape[0] // 2, lon.shape[1] // 2
    c0 = float(np.cos(np.radians(np.nanmean(lat))))
    r = np.array([(lon[i, j + 1] - lon[i, j - 1]) * c0, lat[i, j + 1] - lat[i, j - 1]])
    u = np.array([(lon[i - 1, j] - lon[i + 1, j]) * c0, lat[i - 1, j] - lat[i + 1, j]])
    r /= np.linalg.norm(r)
    u /= np.linalg.norm(u)
    assert r[0] * u[1] - r[1] * u[0] > 0, "display transform is a mirror image"
    spread39 = np.asarray(Z["nosample"]["spread"], np.float32)[39]
    np.savez_compressed(
        loc / "fig5_context.npz",
        slice=slice_, region="mab", transform="fliplr",
        mask_shape=np.asarray(mask.shape), mask_packed=np.packbits(mask),
        spread39_wet=spread39[mask],
        spread_meta=np.bytes_(b"free-running (nosample) prior ensemble spread, "
                              b"cycle 39 = t+240 h, SST, degC, values at wet "
                              b"cells in row-major order"),
        dx_col_m=float(Z["trace"]["dx_col_m"]), dy_row_m=float(Z["trace"]["dy_row_m"]),
        north=np.array([r[1], u[1]]), east=np.array([r[0], u[0]]))
    print(f"  wrote {loc / 'fig5_context.npz'}")

    (loc / "README.md").write_text(
        "# Fig. 5 source positions\n\n"
        f"Per-cycle sampled cells (`<policy>_chosen.csv`) and cycle-start robot\n"
        f"positions (`<policy>_fleet.csv`) of the five policy chains drawn in\n"
        f"Fig. 5: MAB, SST, fleet size 50, IC 2022-04-09T12 (a sealed test IC),\n"
        f"observation seed 0. Extracted by tools/extract_scores.py from the\n"
        f"production position dumps `q3_render_{slice_}_<policy>.npz`\n"
        "(40 cycles; `chosen` rows are planner-ordered picks, `fleet` rows are\n"
        "robot-indexed cycle-START positions; row/col index the raw model grid).\n\n"
        "`fig5_context.npz` carries the drawing context: the wet mask (packed\n"
        "bits), the free-running ensemble spread at cycle 39 over wet cells\n"
        "(panel (a) of the figure), metric cell sizes, and the precomputed\n"
        "display north/east vectors for the map furniture. The full field dumps\n"
        "are released in the artifacts dataset (`render_dumps/`).\n")

    print("EXTRACT_SCORES OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
