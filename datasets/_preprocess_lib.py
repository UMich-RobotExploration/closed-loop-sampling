#!/usr/bin/env python
"""Preprocessing CLI — staged raw NetCDF -> single chunked Zarr per region.

One entry point for the whole preprocessing stage; every region parameter
comes from the RegionConfig (`closedloop_sampling.regions.get_config`) and every
path is relative to $CLOSEDLOOP_DATA_ROOT (closedloop_sampling.paths).

MAB (Doppio v3 `his`, hourly, non-overlapping 72-step files):
  python datasets/_preprocess_lib.py --region mab                      # run the pipeline
  python datasets/_preprocess_lib.py --region mab --validate           # checks 6a-6e

  Per source NetCDF file: subset to the chosen variables + 20 sigma levels,
  destagger u/v/ubar/vbar to rho-points (Arakawa-C, mask-aware), zero land
  cells (mask_rho==0), cast float32, and write that file's 72-step slice into
  the output Zarr via a chunk-aligned region= write (1 input file == 1
  time-chunk -> race-free parallel writes). Static fields written once in the
  template. A JSON manifest of completed file indices makes the run resumable
  (re-run skips done files).

CCS (UCSC CCSRA31 posterior, 6-hourly, OVERLAPPING 33-step files):
  python datasets/_preprocess_lib.py --region ccs --stage build-timeline
  python datasets/_preprocess_lib.py --region ccs --stage transform [--smoke N | --full]
  python datasets/_preprocess_lib.py --region ccs --stage clip-temp
  python datasets/_preprocess_lib.py --region ccs --validate
  python datasets/_preprocess_lib.py --region ccs                      # all stages in order

  build-timeline: de-overlap the staged files into a canonical timeline
  manifest (metadata only — no data variables loaded). transform: per
  canonical instant, NaN-fill -> destagger -> level-drop -> landmask ->
  float32 on the full native grid, written into an MAB-format zarr with
  statics joined from ccs_static.nc. clip-temp: one-time in-place clip of
  physically-impossible temp artifacts (full pre-clip audit log). validate:
  full-store structural/physical checks.

The destagger helpers are shared between the two regions (they take explicit
`xi_rho`/`eta_rho` kwargs; the defaults are the MAB grid dims so MAB behavior
is byte-unchanged, and the CCS calls pass the CCS grid dims explicitly).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import json
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Must be set BEFORE xarray/netCDF4/HDF5 load. The staged NetCDF data may
# live on a network filesystem; HDF5>=1.10 default file-locking can
# crash/hang there, and netCDF4/HDF5 is not thread-safe (an earlier
# ThreadPoolExecutor variant segfaulted).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # many procs -> no BLAS oversubscription

import numpy as np
import xarray as xr
import netCDF4 as nc
import zarr
import zarr.codecs
from numcodecs import Blosc

from closedloop_sampling.regions import get_config
from closedloop_sampling.paths import resolve

MAB = get_config("mab")
CCS = get_config("ccs")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fail(msg, tag="preprocess"):
    print(f"[{tag}] FAIL: {msg}", flush=True)
    sys.exit(1)


# ============================================================
# MAB constants (config-derived) + lazily-resolved paths
# ============================================================
N_WORKERS = 16                                        # default; --workers overrides
STEPS_PER_FILE = MAB.raw.steps_per_file               # 72 = output time-chunk
S_LEVELS = list(MAB.vertical.kept_level_indices)      # 20 of 40
DYN3D = list(MAB.channels.dyn3d_vars)                 # -> (t, 20, 106, 242) f32
DYN2D = list(MAB.channels.dyn2d_vars)                 # -> (t, 106, 242) f32
DYN_ALL = DYN3D + DYN2D
STATIC_2D = list(MAB.channels.static_vars_2d)
STATIC_1D_SUB = list(MAB.channels.static_vars_1d)     # subset to S_LEVELS
STATIC_1D_FULL = ["s_w", "Cs_w"]                      # full 41
STATIC_SCALAR = ["Vtransform", "Vstretching", "theta_s", "theta_b",
                 "Tcline", "hc"]
COMPRESSOR = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
ETA_RHO = MAB.grid.native_eta_rho
XI_RHO = MAB.grid.native_xi_rho
NLEV = MAB.vertical.n_levels_kept


def mab_raw_dir():
    return resolve(MAB.raw.staging_dir)


def mab_out():
    return resolve(MAB.paths.preprocessed_zarr)


def mab_manifest():
    return mab_out().rstrip("/") + ".manifest.json"


def sorted_files():
    """his_raw/*.nc sorted by the NNNN run index (== chronological)."""
    fs = glob.glob(os.path.join(mab_raw_dir(), MAB.raw.staging_glob_patterns[0]))
    def nnnn(p):
        m = re.search(r"doppio_fwd_(\d+)_plusdye\.nc$", p)
        return int(m.group(1))
    return sorted(fs, key=nnnn)


# ---- destaggering (Arakawa-C u/v -> rho, mask-aware) ----
def _destagger_xi(arr, mask_u, xi_rho=XI_RHO):
    """arr (...,eta,xi_u=241) -> (...,eta,xi_rho=242). mask_u (eta,241).

    `xi_rho` (output width) defaults to the MAB module constant so existing
    callers are unchanged; pass it for a different grid (e.g. CCS xi_rho=186)."""
    left, right = arr[..., :-1], arr[..., 1:]            # (...,eta,240)
    ml = (mask_u[:, :-1] > 0.5)                            # (eta,240)
    mr = (mask_u[:, 1:] > 0.5)
    both = ml & mr
    avg = 0.5 * (left + right)
    interior = np.where(both, avg,
               np.where(ml, left,
               np.where(mr, right, 0.0)))                  # (...,eta,240)
    out = np.empty(arr.shape[:-1] + (xi_rho,), dtype=np.float32)
    out[..., 1:-1] = interior
    out[..., 0] = out[..., 1]        # nearest-interior boundary
    out[..., -1] = out[..., -2]
    return out


def _destagger_eta(arr, mask_v, eta_rho=ETA_RHO):
    """arr (...,eta_v=105,xi) -> (...,eta_rho=106,xi). mask_v (105,xi).

    `eta_rho` (output height) defaults to the MAB module constant so existing
    callers are unchanged; pass it for a different grid (e.g. CCS eta_rho=181)."""
    top, bot = arr[..., :-1, :], arr[..., 1:, :]           # (...,104,xi)
    mt = (mask_v[:-1, :] > 0.5)
    mb = (mask_v[1:, :] > 0.5)
    both = mt & mb
    avg = 0.5 * (top + bot)
    interior = np.where(both, avg,
               np.where(mt, top,
               np.where(mb, bot, 0.0)))
    out = np.empty(arr.shape[:-2] + (eta_rho, arr.shape[-1]),
                   dtype=np.float32)
    out[..., 1:-1, :] = interior
    out[..., 0, :] = out[..., 1, :]
    out[..., -1, :] = out[..., -2, :]
    return out


def _apply_landmask(arr, mask_rho):
    """Zero cells where mask_rho==0 (broadcast static mask over t[,s])."""
    return np.where(mask_rho > 0.5, arr, 0.0).astype(np.float32)


def load_static(ref_path):
    ds = xr.open_dataset(ref_path, engine="netcdf4", decode_times=True)
    out = {}
    for v in STATIC_2D:
        out[v] = ds[v].values.astype(
            np.float32 if v != "mask_rho" else np.float32)
    for v in STATIC_1D_SUB:
        out[v] = ds[v].isel(s_rho=S_LEVELS).values.astype(np.float32)
    for v in STATIC_1D_FULL:
        out[v] = ds[v].values.astype(np.float32)
    for v in STATIC_SCALAR:
        out[v] = np.float32(ds[v].values)
    masks = {m: ds[m].values.astype(np.float32)
             for m in ("mask_rho", "mask_u", "mask_v")}
    ds.close()
    return out, masks


def build_template(files, masks):
    """Read each file's ocean_time, assemble the full coord (validated),
    write the compute=False template (static + ocean_time eager,
    dynamic placeholders chunked 72)."""
    log(f"reading ocean_time from {len(files)} files to build coord ...")

    def _times(p):
        d = xr.open_dataset(p, engine="netcdf4", decode_times=True)
        t = d["ocean_time"].values
        d.close()
        return t

    # SERIAL on purpose: netCDF4/HDF5 is not thread-safe (an earlier
    # ThreadPoolExecutor here segfaulted). ~2191 tiny coord reads,
    # a couple minutes, negligible vs the multi-hour pipeline.
    per_file = []
    for i, p in enumerate(files):
        per_file.append(_times(p))
        if (i + 1) % 250 == 0:
            log(f"  ocean_time read {i + 1}/{len(files)}")
    lens = [len(t) for t in per_file]
    if any(n != STEPS_PER_FILE for n in lens):
        bad = [(files[i], lens[i]) for i, n in enumerate(lens)
               if n != STEPS_PER_FILE]
        raise ValueError(f"non-72-step file(s): {bad[:5]}")
    ocean_time = np.concatenate(per_file)
    nt = ocean_time.size
    d = np.diff(ocean_time).astype("timedelta64[s]").astype(int)
    if not np.all(d == 3600):
        raise ValueError(f"ocean_time not strictly hourly/contiguous; "
                         f"bad gaps at {np.where(d != 3600)[0][:5]}")
    log(f"ocean_time OK: {nt} steps {ocean_time[0]} .. {ocean_time[-1]}")

    static, _ = load_static(files[0])
    coords = {"ocean_time": ("ocean_time", ocean_time),
              "s_rho": ("s_rho", static["s_rho"]),
              "s_w": ("s_w", static["s_w"])}
    dv = {}
    for v in STATIC_2D:
        dv[v] = (("eta_rho", "xi_rho"), static[v])
    dv["Cs_r"] = (("s_rho",), static["Cs_r"])
    dv["Cs_w"] = (("s_w",), static["Cs_w"])
    for v in STATIC_SCALAR:
        dv[v] = ((), static[v])
    import dask.array as da
    for v in DYN3D:
        dv[v] = (("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                 da.zeros((nt, NLEV, ETA_RHO, XI_RHO), dtype="float32",
                          chunks=(STEPS_PER_FILE, NLEV, ETA_RHO, XI_RHO)))
    for v in DYN2D:
        dv[v] = (("ocean_time", "eta_rho", "xi_rho"),
                 da.zeros((nt, ETA_RHO, XI_RHO), dtype="float32",
                          chunks=(STEPS_PER_FILE, ETA_RHO, XI_RHO)))
    ds = xr.Dataset(dv, coords=coords)
    ds.attrs.update(
        source="Doppio DopAnV3R3-ini2007 his (S3 mirror, his_raw/)",
        s_levels=str(S_LEVELS), destagger="Arakawa-C->rho, mask-aware",
        land_value="0 (mask_rho is a separate static channel)")
    enc = {}
    for v in DYN3D:
        enc[v] = {"chunks": (STEPS_PER_FILE, NLEV, ETA_RHO, XI_RHO),
                  "compressor": COMPRESSOR, "dtype": "float32"}
    for v in DYN2D:
        enc[v] = {"chunks": (STEPS_PER_FILE, ETA_RHO, XI_RHO),
                  "compressor": COMPRESSOR, "dtype": "float32"}
    log("writing compute=False template (static eager, dynamic deferred)")
    ds.to_zarr(mab_out(), mode="w", compute=False, consolidated=False,
               encoding=enc)
    return nt


def process_file(args):
    idx, path, t0 = args
    t1 = t0 + STEPS_PER_FILE
    try:
        _, masks = load_static(path)  # masks from this file (grid is static)
        ds = xr.open_dataset(path, engine="netcdf4", decode_times=True)
        out = {}
        u = ds["u"].isel(s_rho=S_LEVELS).values.astype(np.float32)
        out["u"] = _apply_landmask(_destagger_xi(u, masks["mask_u"]),
                                   masks["mask_rho"])
        v = ds["v"].isel(s_rho=S_LEVELS).values.astype(np.float32)
        out["v"] = _apply_landmask(_destagger_eta(v, masks["mask_v"]),
                                   masks["mask_rho"])
        for nm in ("temp", "salt"):
            a = ds[nm].isel(s_rho=S_LEVELS).values.astype(np.float32)
            out[nm] = _apply_landmask(a, masks["mask_rho"])
        out["zeta"] = _apply_landmask(
            ds["zeta"].values.astype(np.float32), masks["mask_rho"])
        out["ubar"] = _apply_landmask(
            _destagger_xi(ds["ubar"].values.astype(np.float32),
                          masks["mask_u"]), masks["mask_rho"])
        out["vbar"] = _apply_landmask(
            _destagger_eta(ds["vbar"].values.astype(np.float32),
                           masks["mask_v"]), masks["mask_rho"])
        ds.close()
        dv = {}
        for nm in DYN3D:
            dv[nm] = (("ocean_time", "s_rho", "eta_rho", "xi_rho"),
                      out[nm])
        for nm in DYN2D:
            dv[nm] = (("ocean_time", "eta_rho", "xi_rho"), out[nm])
        slab = xr.Dataset(dv)
        slab.to_zarr(mab_out(), region={"ocean_time": slice(t0, t1)},
                     consolidated=False)
        return idx, True, None
    except Exception as e:  # noqa: BLE001
        return idx, False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def mab_run(n_workers=N_WORKERS):
    files = sorted_files()
    if not files:
        log(f"ERROR no source files in {mab_raw_dir()}")
        return 1
    log(f"{len(files)} source files")

    out = mab_out()
    manifest = mab_manifest()
    fresh = not os.path.isdir(out)
    if fresh:
        nt = build_template(files, None)
    else:
        log("output store exists -> resuming")
        nt = len(files) * STEPS_PER_FILE

    done = set()
    if os.path.exists(manifest):
        done = set(json.load(open(manifest)).get("done", []))
        log(f"manifest: {len(done)}/{len(files)} files already done")

    tasks = [(i, p, i * STEPS_PER_FILE)
             for i, p in enumerate(files) if i not in done]
    log(f"{len(tasks)} files to process with {n_workers} workers")
    t_start = time.time()
    n_ok = len(done)
    # spawn (not fork): the parent has already touched netCDF4/HDF5
    # (build_template); forking that state into 16 workers can crash.
    # Spawn = fresh interpreter per worker, each re-imports this module
    # (so the HDF5/thread env above is re-applied) -> isolated & safe.
    ctx = multiprocessing.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=n_workers,
                                mp_context=ctx) as ex:
        for idx, ok, err in ex.map(process_file, tasks):
            if ok:
                done.add(idx)
                n_ok += 1
                if n_ok % 25 == 0 or n_ok == len(files):
                    el = time.time() - t_start
                    pct = 100.0 * n_ok / len(files)
                    log(f"progress {n_ok}/{len(files)} ({pct:.1f}%) "
                        f"elapsed {el/60:.1f}m")
                    json.dump({"done": sorted(done)}, open(manifest, "w"))
            else:
                log(f"FILE {idx} FAILED: {err}")
                json.dump({"done": sorted(done)}, open(manifest, "w"))
    json.dump({"done": sorted(done)}, open(manifest, "w"))

    if len(done) != len(files):
        log(f"INCOMPLETE: {len(done)}/{len(files)} done -> rerun to resume")
        return 1
    zarr.consolidate_metadata(out)
    log(f"DONE: {len(files)} files, {nt} timesteps, consolidated.")
    return 0


# ============================================================
# MAB validation (checks 6a-6e)
# ============================================================
def mab_validate():
    """Validation for the preprocessed MAB Zarr (checks 6a-6e).

    Reuses the exact preprocessing functions so destagger/mask checks are
    self-consistent. Writes a report; prints PASS/FAIL; returns nonzero on
    any failure.

    6e note: a full train-split re-derivation costs the same as the stats
    job itself, so 6e is a *targeted exact* re-check (all 2D vars fully +
    each 3D var at sigma-level index 0) compared to the stored JSON within
    float32 tolerance -- a faithful integrity check within the time budget.
    """
    out = mab_out()
    json_stats = resolve(MAB.paths.pixel_norm_stats)
    report = os.path.join(os.path.dirname(out),
                          "mab_preprocessing_validation_report.txt")
    nt_expect = MAB.timesplit.n_timesteps_total
    train_start = MAB.timesplit.start_time[:10]
    train_end = f"{MAB.timesplit.train_year_range[1]}-12-31"
    results = []

    def chk(name, ok, detail=""):
        results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}"
              + (f" :: {detail}" if detail else ""), flush=True)

    t0 = time.time()
    ds = xr.open_zarr(out)
    za = zarr.open(out, mode="r")

    # ---- 6a structural ----
    for v in DYN_ALL + STATIC_2D + ["s_rho", "s_w", "Cs_r", "Cs_w"] \
            + STATIC_SCALAR:
        chk(f"6a present:{v}", v in ds.variables)
    chk("6a u shape", tuple(ds["u"].shape) == (nt_expect, NLEV, ETA_RHO, XI_RHO),
        str(tuple(ds["u"].shape)))
    chk("6a v shape", tuple(ds["v"].shape) == (nt_expect, NLEV, ETA_RHO, XI_RHO),
        str(tuple(ds["v"].shape)))
    chk("6a zeta shape", tuple(ds["zeta"].shape) == (nt_expect, ETA_RHO, XI_RHO),
        str(tuple(ds["zeta"].shape)))
    ot = ds["ocean_time"].values
    chk("6a ntime", ot.size == nt_expect, str(ot.size))
    chk("6a time range",
        str(ot[0])[:16] == MAB.timesplit.start_time[:16]
        and str(ot[-1])[:16] == MAB.timesplit.end_time[:16],
        f"{str(ot[0])[:16]} .. {str(ot[-1])[:16]}")
    chk(f"6a s_rho coord == S_LEVELS-derived (len {NLEV})",
        ds.sizes.get("s_rho") == NLEV, str(ds.sizes.get("s_rho")))
    uchunks = za["u"].chunks
    chk(f"6a time chunk == {STEPS_PER_FILE}", uchunks[0] == STEPS_PER_FILE,
        f"u chunks={uchunks}")
    comp = za["u"].compressor
    chk("6a compressor Blosc/lz4",
        comp is not None and getattr(comp, "cname", "") == "lz4",
        str(comp))
    chk("6a dtype float32", str(za["u"].dtype) == "float32",
        str(za["u"].dtype))

    # ---- 6b content ----
    mask = ds["mask_rho"].values
    land = mask <= 0.5
    sl = ds.isel(ocean_time=slice(0, 4))
    for v in DYN_ALL:
        a = sl[v].values  # (t,[s,]eta_rho,xi_rho); land is (eta_rho,xi_rho)
        # land cells must be exactly 0 (fancy-index last two dims)
        land_max = np.nanmax(np.abs(a[..., land])) if a[..., land].size else 0
        chk(f"6b land==0:{v}", land_max == 0.0, f"max|land|={land_max}")
        wet_vals = a[..., ~land]
        chk(f"6b wet finite:{v}", np.all(np.isfinite(wet_vals)),
            f"finite={np.all(np.isfinite(wet_vals))}")
    # sane bounds (generous; ROMS temp in degC, salt PSU)
    tmin, tmax = float(sl["temp"].min()), float(sl["temp"].max())
    chk("6b temp range sane", -5 < tmax < 45 and -5 <= tmin,
        f"[{tmin:.2f},{tmax:.2f}]")
    smin, smax = float(sl["salt"].min()), float(sl["salt"].max())
    chk("6b salt range sane", 0 <= smin and smax < 45,
        f"[{smin:.2f},{smax:.2f}]")
    # The MAB `his` store is HOURLY INSTANTANEOUS over a domain incl. the
    # Gulf of Maine / Bay of Fundy approach (among Earth's largest
    # tides) -> |zeta| ~6 m near the open boundary is PHYSICAL. A 5 m
    # bound would implicitly be calibrated to tide-AVERAGED SSH. 15 m
    # still catches genuine corruption (NaN-blowup, unit errors,
    # ~order-of-magnitude garbage) while admitting real tidal+surge sea
    # level.
    zmin, zmax = float(sl["zeta"].min()), float(sl["zeta"].max())
    zabs = max(abs(zmin), abs(zmax))
    chk("6b zeta sane (|zeta|<15m, tidal)", zabs < 15.0,
        f"min={zmin:.3f} max={zmax:.3f}")

    # ---- 6c/6d destagger + end-to-end vs raw NetCDF ----
    files = sorted_files()
    k = len(files) // 2  # mid-archive file (not just file 0)
    fpath = files[k]
    _, masks = load_static(fpath)
    rd = xr.open_dataset(fpath, engine="netcdf4", decode_times=True)
    t_a, t_b = k * STEPS_PER_FILE, (k + 1) * STEPS_PER_FILE
    zslab = ds.isel(ocean_time=slice(t_a, t_b))
    # 6c: single field (u, t-local 0) destagger correctness
    u_raw = rd["u"].isel(s_rho=S_LEVELS).values.astype(np.float32)
    u_re = _apply_landmask(_destagger_xi(u_raw, masks["mask_u"]),
                           masks["mask_rho"])
    chk("6c destagger u vs raw (one file)",
        np.allclose(u_re, zslab["u"].values, rtol=1e-5, atol=1e-5,
                    equal_nan=True),
        f"maxdiff={np.max(np.abs(u_re - zslab['u'].values)):.2e}")
    # 6d: full end-to-end recompute of this file's chunk, all vars
    e2e_ok, e2e_det = True, []
    recompute = {
        "v": _apply_landmask(_destagger_eta(
            rd["v"].isel(s_rho=S_LEVELS).values.astype(np.float32),
            masks["mask_v"]), masks["mask_rho"]),
        "temp": _apply_landmask(
            rd["temp"].isel(s_rho=S_LEVELS).values.astype(np.float32),
            masks["mask_rho"]),
        "salt": _apply_landmask(
            rd["salt"].isel(s_rho=S_LEVELS).values.astype(np.float32),
            masks["mask_rho"]),
        "zeta": _apply_landmask(rd["zeta"].values.astype(np.float32),
                                masks["mask_rho"]),
        "ubar": _apply_landmask(_destagger_xi(
            rd["ubar"].values.astype(np.float32), masks["mask_u"]),
            masks["mask_rho"]),
        "vbar": _apply_landmask(_destagger_eta(
            rd["vbar"].values.astype(np.float32), masks["mask_v"]),
            masks["mask_rho"]),
    }
    for v, arr in recompute.items():
        md = float(np.max(np.abs(arr - zslab[v].values)))
        if md > 1e-4:
            e2e_ok = False
        e2e_det.append(f"{v}:{md:.1e}")
    chk("6d end-to-end chunk vs raw recompute", e2e_ok,
        " ".join(e2e_det))
    rd.close()

    # ---- 6e statistics vs stored JSON (targeted exact) ----
    if not os.path.exists(json_stats):
        chk("6e stats JSON exists", False, json_stats)
    else:
        js = json.load(open(json_stats))
        tr = ds.sel(ocean_time=slice(train_start, train_end))
        wet = tr["mask_rho"] > 0.5
        ok6e, det6e = True, []
        for v in DYN2D:  # full, cheap
            a = tr[v].astype("float64").where(wet)
            m = float(a.mean(skipna=True).compute())
            s = float(a.std(skipna=True).compute())
            if not (np.isclose(m, js[v]["mean"], rtol=1e-4, atol=1e-6)
                    and np.isclose(s, js[v]["std"], rtol=1e-4, atol=1e-6)):
                ok6e = False
            det6e.append(f"{v}(m{m:.3g}/{js[v]['mean']:.3g})")
        for v in DYN3D:  # level-0 only (keep within time budget)
            a = tr[v].isel(s_rho=0).astype("float64").where(wet)
            m = float(a.mean(skipna=True).compute())
            if not np.isclose(m, js[v]["mean"][0], rtol=1e-4, atol=1e-6):
                ok6e = False
            det6e.append(f"{v}L0(m{m:.3g}/{js[v]['mean'][0]:.3g})")
        chk("6e stats match JSON (2D full + 3D level0)", ok6e,
            " ".join(det6e))

    # ---- report ----
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    lines = [f"MAB preprocessing validation  ({time.strftime('%F %T')})",
             f"store: {out}",
             f"elapsed: {time.time()-t0:.0f}s",
             f"PASS={n_pass} FAIL={n_fail}", ""]
    for nm, ok, det in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {nm}"
                     + (f" :: {det}" if det else ""))
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n=== VALIDATION {'PASSED' if n_fail == 0 else 'FAILED'} "
          f"(PASS={n_pass} FAIL={n_fail}) -> {report} ===", flush=True)
    return 0 if n_fail == 0 else 1


# ============================================================
# CCS Stage 1: build the de-overlapped canonical timeline
# ============================================================
# The only fully-novel logic in the CCS ingest (MAB had no file overlap), so
# it is built + verified in isolation on FILE METADATA (time axes only; no
# data variables loaded) before any transform is written. Every CCS parameter
# is read from the CCS RegionConfig — NO CCS constant is hardcoded here.
#
# Dedup rule (later-file-wins, with a core-priority seam) — the exact boundary
# convention:
#   - Each collection (core, ext) is de-overlapped INDEPENDENTLY by half-open
#     ownership: sorted chronologically, file_i owns the canonical timestamps
#     in [start_i, start_{i+1})  — i.e. its 16-step (4-day) lead. The
#     overlapping tail (t >= start_{i+1}) is dropped in favor of the LATER
#     file within the same collection. The LAST file of a collection keeps
#     its FULL tail (no successor to hand off to).
#   - The two collections are then MERGED. In the ~2010-12-21..29 core/ext
#     seam (a true duplicate, Float64 core vs Int16 ext), CORE wins: any
#     timestamp present in BOTH the core and ext de-overlapped timelines is
#     taken from the core file.
#   - FENCEPOST: a handoff timestamp (a file's start == the previous file's
#     interior step) is owned by EXACTLY ONE file — the later one within a
#     collection (`t < start_{i+1}` is strict), and the core file at the
#     cross-collection seam (core overwrites ext in the merge).
#
# Writes a JSON timeline manifest beside where the preprocessed zarr will
# live. Self-validates fail-loud (non-zero exit); no magic numbers.

class FileRec:
    __slots__ = ("path", "basename", "collection", "raw", "epochs")

    def __init__(self, path, collection, raw, epochs):
        self.path = path
        self.basename = os.path.basename(path)
        self.collection = collection
        self.raw = raw            # raw ocean_time values (seconds since epoch)
        self.epochs = epochs      # int seconds-since-config-epoch, one per step


def ccs_staging_dir():
    return resolve(CCS.raw.staging_dir)


def ccs_manifest_path():
    return os.path.join(os.path.dirname(resolve(CCS.paths.preprocessed_zarr)),
                        "ccs_timeline_manifest.json")


def collect_files():
    patterns = CCS.raw.staging_glob_patterns            # (core_pat, ext_pat)
    if len(patterns) != 2:
        _fail(f"expected 2 glob patterns (core, ext); got {patterns}",
              tag="build-timeline")
    core_pat, ext_pat = patterns
    tags = {core_pat: "core", ext_pat: "ext"}

    per_pattern = {}
    seen = set()
    files = []
    steps_expected = CCS.raw.steps_per_file
    bad_step_files = []
    for pat in patterns:
        matched = sorted(glob.glob(os.path.join(ccs_staging_dir(), pat)))
        per_pattern[pat] = len(matched)
        for p in matched:
            b = os.path.basename(p)
            # belt-and-braces: a file must match exactly one pattern
            if b in seen:
                _fail(f"file {b} matched two glob patterns", tag="build-timeline")
            seen.add(b)
            coll = tags[pat]
            with nc.Dataset(p) as d:
                ot = d.variables["ocean_time"]
                units = getattr(ot, "units", None)
                cal = getattr(ot, "calendar", "standard")
                if units != CCS.raw.time_units:
                    _fail(f"{b}: ocean_time units {units!r} != config "
                          f"{CCS.raw.time_units!r}", tag="build-timeline")
                raw = np.asarray(ot[:])
            if len(raw) != steps_expected:
                bad_step_files.append((b, len(raw)))
            epochs = [int(round(float(x))) for x in raw]   # seconds since config epoch
            files.append(FileRec(p, coll, raw, epochs))
    return files, per_pattern, bad_step_files


def collection_timeline(coll_files):
    """coll_files sorted by first epoch. Returns {epoch: (basename, step, coll)}.

    file_i owns [start_i, start_{i+1}); last file keeps its full tail.
    """
    coll_files = sorted(coll_files, key=lambda f: f.epochs[0])
    out = {}
    n = len(coll_files)
    for i, f in enumerate(coll_files):
        nxt_start = coll_files[i + 1].epochs[0] if i + 1 < n else None
        for step, e in enumerate(f.epochs):
            if nxt_start is None or e < nxt_start:      # half-open, strict
                # within a collection two files never claim the same epoch here,
                # because the half-open interval of the earlier file excludes it.
                out[e] = (f.basename, step, f.collection)
    return out


def build_canonical(files):
    core = [f for f in files if f.collection == "core"]
    ext = [f for f in files if f.collection == "ext"]
    core_map = collection_timeline(core)
    ext_map = collection_timeline(ext)
    # Merge: ext first, then core OVERWRITES -> core wins the cross-collection seam.
    canonical = dict(ext_map)
    n_seam_core_wins = 0
    for e, rec in core_map.items():
        if e in canonical and canonical[e][2] != "core":
            n_seam_core_wins += 1
        canonical[e] = rec
    return canonical, core_map, ext_map, n_seam_core_wins


def self_validate(canonical, files):
    results = []
    cadence_s = CCS.timesplit.cadence_hours * 3600
    epochs = sorted(canonical.keys())

    # a. total count
    n = len(epochs)
    exp = CCS.timesplit.expected_deoverlapped_steps
    ok = (n == exp)
    results.append((f"total unique == {exp}", ok, f"got {n}"))
    if not ok:
        return results, epochs

    # b. strictly-monotonic contiguous cadence
    diffs = np.diff(np.array(epochs, dtype=np.int64))
    bad = np.where(diffs != cadence_s)[0]
    ok = (len(bad) == 0)
    detail = "all 21600s" if ok else f"{len(bad)} bad steps; first at idx {int(bad[0])}"
    results.append((f"diff == {cadence_s}s, monotonic, no gap/dup", ok, detail))

    # c. span endpoints
    units = CCS.raw.time_units
    cal = CCS.raw.calendar
    first_dt = nc.num2date(epochs[0], units, cal, only_use_cftime_datetimes=False,
                           only_use_python_datetimes=True)
    last_dt = nc.num2date(epochs[-1], units, cal, only_use_cftime_datetimes=False,
                          only_use_python_datetimes=True)
    first_iso = first_dt.isoformat()
    last_iso = last_dt.isoformat()
    ok_first = first_iso.startswith(CCS.timesplit.start_time)
    ok_last = last_iso.startswith(CCS.timesplit.end_time)
    results.append((f"first == {CCS.timesplit.start_time}", ok_first, first_iso))
    results.append((f"last  == {CCS.timesplit.end_time}", ok_last, last_iso))

    # d. every owning (file, step) valid
    by_base = {f.basename: f for f in files}
    steps_pf = CCS.raw.steps_per_file
    bad_owner = 0
    for e in epochs:
        b, step, coll = canonical[e]
        f = by_base.get(b)
        if f is None or not (0 <= step < steps_pf) or f.epochs[step] != e:
            bad_owner += 1
    results.append(("every entry (file, step) valid & epoch-consistent",
                    bad_owner == 0, f"{bad_owner} invalid"))

    # e. no timestamp assigned twice (dict guarantees uniqueness; confirm len)
    results.append(("no timestamp assigned to two entries", n == len(set(epochs)),
                    f"{n} unique keys"))
    return results, epochs


def iso(e):
    dt = nc.num2date(e, CCS.raw.time_units, CCS.raw.calendar,
                     only_use_cftime_datetimes=False, only_use_python_datetimes=True)
    return dt.isoformat()[:16]


def spot_check_interior(files, canonical, n_seams=5):
    core = sorted([f for f in files if f.collection == "core"],
                  key=lambda f: f.epochs[0])
    print("\n--- interior handoff spot-check (core/core; later-file-wins, half-open) ---")
    idxs = np.linspace(len(core) // 5, 4 * len(core) // 5, n_seams).astype(int)
    a_starts = {f.basename: f.epochs[0] for f in core}
    for i in idxs:
        a, b = core[i], core[i + 1]
        overlap = sorted(set(a.epochs) & set(b.epochs))
        owners = {}
        for e in overlap:
            owners.setdefault(canonical[e][0], 0)
            owners[canonical[e][0]] += 1
        print(f"  {a.basename} -> {b.basename}: {len(overlap)} shared "
              f"[{iso(overlap[0])} .. {iso(overlap[-1])}]")
        print(f"     shared-timestamp owners: {owners}")
        # Correct later-file-wins property (half-open): the EARLIER file `a` owns
        # NONE of its overlap tail; every shared instant goes to a strictly-later
        # file. The last shared instant is `b`'s start of ITS successor, so it is
        # owned by that even-later file (the fencepost) — a 16+1 split, not 17.
        earlier_owns_zero = a.basename not in owners
        all_later = all(a_starts.get(fb, a.epochs[0] + 1) > a.epochs[0] for fb in owners)
        print(f"     earlier file '{a.basename}' owns 0 of overlap: {earlier_owns_zero}; "
              f"all owners strictly later than it: {all_later}  (16+1 split = correct fencepost)")


def spot_check_seam(files, canonical):
    print("\n--- core/ext seam spot-check (2010-12-21..29; CORE must win) ---")
    # find core & ext files whose window includes 2010-12-21..29
    def in_2010_seam(f):
        return any(iso(e).startswith("2010-12-2") for e in f.epochs)
    cores = [f for f in files if f.collection == "core" and in_2010_seam(f)]
    exts = [f for f in files if f.collection == "ext" and in_2010_seam(f)]
    print(f"  core files over seam: {[f.basename for f in cores]}")
    print(f"  ext  files over seam: {[f.basename for f in exts]}")
    # the shared window between the last core and first ext (identical 33-step file)
    if cores and exts:
        c = max(cores, key=lambda f: f.epochs[0])   # last core
        e = min(exts, key=lambda f: f.epochs[0])    # first ext
        shared = sorted(set(c.epochs) & set(e.epochs))
        owners = {}
        for ep in shared:
            owners.setdefault((canonical[ep][0], canonical[ep][2]), 0)
            owners[(canonical[ep][0], canonical[ep][2])] += 1
        print(f"  {c.basename} (core) ∩ {e.basename} (ext): {len(shared)} shared "
              f"[{iso(shared[0])} .. {iso(shared[-1])}]")
        print(f"     owners of shared: {owners}")
        all_core = all(canonical[ep][2] == "core" for ep in shared)
        print(f"     ALL shared owned by CORE: {all_core}")


def ccs_build_timeline():
    print("=" * 66)
    print(f"CCS timeline builder — region '{CCS.name}'")
    print("=" * 66)
    print(f"staging_dir: {ccs_staging_dir()}")
    print(f"patterns:    {CCS.raw.staging_glob_patterns}")
    print(f"steps/file:  {CCS.raw.steps_per_file}   cadence: {CCS.timesplit.cadence_hours}h")
    print(f"expected de-overlapped steps: {CCS.timesplit.expected_deoverlapped_steps}")
    print()

    files, per_pattern, bad_step_files = collect_files()
    for pat, cnt in per_pattern.items():
        print(f"  glob {pat!r}: {cnt} files")
    print(f"  total files: {len(files)}")
    if bad_step_files:
        print(f"  !! non-{CCS.raw.steps_per_file}-step files (outliers): {bad_step_files}")
    else:
        print(f"  all files have {CCS.raw.steps_per_file} steps ✓")

    canonical, core_map, ext_map, n_seam = build_canonical(files)
    print(f"\n  core de-overlapped: {len(core_map)}   ext de-overlapped: {len(ext_map)}")
    print(f"  canonical (merged, core-priority): {len(canonical)}")
    print(f"  cross-collection seam timestamps where CORE overrode ext: {n_seam}")

    # ---- self-validate ----
    print("\n" + "=" * 66)
    print("SELF-CHECKS (fail-loud)")
    print("=" * 66)
    results, epochs = self_validate(canonical, files)
    all_ok = True
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}   ({detail})")
        all_ok = all_ok and ok

    # ---- spot checks ----
    spot_check_interior(files, canonical)
    spot_check_seam(files, canonical)

    if not all_ok:
        _fail("one or more self-checks failed — manifest NOT written",
              tag="build-timeline")

    # ---- write manifest ----
    manifest_path = ccs_manifest_path()
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    entries = []
    for e in epochs:
        b, step, coll = canonical[e]
        entries.append({
            "timestamp_iso": iso(e) + ":00",
            "epoch_seconds": e,                 # seconds since config epoch (see meta.time_units)
            "owning_file": b,
            "collection": coll,
            "step_index_within_file": step,
        })
    manifest = {
        "_meta": {
            "region": CCS.name,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "time_units": CCS.raw.time_units,
            "calendar": CCS.raw.calendar,
            "cadence_hours": CCS.timesplit.cadence_hours,
            "expected_deoverlapped_steps": CCS.timesplit.expected_deoverlapped_steps,
            "n_entries": len(entries),
            "dedup_rule": "later-file-wins (half-open per collection) + core-wins at seam",
            "staging_dir": ccs_staging_dir(),
            "n_files_core": sum(1 for f in files if f.collection == "core"),
            "n_files_ext": sum(1 for f in files if f.collection == "ext"),
        },
        "timeline": entries,
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh)
    os.replace(tmp, manifest_path)
    print(f"\n[build-timeline] ALL CHECKS PASS — wrote {len(entries)} entries")
    print(f"  manifest: {manifest_path}  ({os.path.getsize(manifest_path)/1e6:.1f} MB)")
    return 0


# ============================================================
# CCS Stage 2+3: transform + write (manifest-driven)
# ============================================================
# Per canonical instant (from ccs_timeline_manifest.json, Stage 1): open the
# owning staged file, read step_index, NaN-fill -> destagger -> level-drop ->
# landmask -> float32, on the FULL 181x186 grid (no crop), and write into an
# MAB-format zarr (float32, Blosc-LZ4, time-chunked) with statics joined
# from ccs_static.nc.
#
# The destagger helpers are SHARED with the MAB section above (not copied).
# They accept explicit `xi_rho`/`eta_rho` kwargs (defaults = the MAB module
# constants, so MAB behavior is byte-unchanged); the CCS calls pass the CCS
# grid dims from the RegionConfig explicitly — no module-global mutation.

def load_ccs_manifest():
    with open(ccs_manifest_path()) as f:
        return json.load(f)


def kept_level_positions():
    """Positions within the 21-staged s_rho axis to KEEP (drop-deepest -> 20)."""
    staged = list(CCS.vertical.staged_level_indices)
    return [staged.index(k) for k in CCS.vertical.kept_level_indices]


def load_ccs_statics():
    """mask_rho/mask_u/mask_v (for destagger+landmask) + the carried statics."""
    s = {}
    kp = kept_level_positions()
    with nc.Dataset(resolve(CCS.paths.static_source)) as d:
        for v in ("mask_rho", "mask_u", "mask_v"):
            s[v] = np.asarray(d.variables[v][:]).astype(np.float32)
        for v in CCS.channels.static_vars_2d:
            s[v] = np.asarray(d.variables[v][:]).astype(np.float32)
        for v in CCS.channels.static_vars_1d:            # s_rho, Cs_r -> subset to kept levels
            s[v] = np.asarray(d.variables[v][:])[kp].astype(np.float32)
    return s


def transform_instant(entry, ds_cache, statics, kept_pos):
    """Return dict {var: float32 array} for one canonical instant, matching
    the MAB store layout (3D vars (20,eta,xi); 2D vars (eta,xi))."""
    b = entry["owning_file"]
    step = entry["step_index_within_file"]
    if b not in ds_cache:
        ds_cache[b] = nc.Dataset(os.path.join(ccs_staging_dir(), b))
    ds = ds_cache[b]

    mask_rho, mask_u, mask_v = statics["mask_rho"], statics["mask_u"], statics["mask_v"]
    out = {}

    # --- rho-grid 3D: temp, salt --- (level-drop -> landmask; NaN land handled by landmask)
    for v in ("temp", "salt"):
        arr = np.asarray(ds.variables[v][step])          # (21, 181, 186) float64
        arr = arr[kept_pos]                              # (20, 181, 186)
        out[v] = _apply_landmask(arr, mask_rho)          # -> float32, land 0

    # --- staggered 3D: u (u-pts), v (v-pts) --- NaN-fill -> destagger -> level-drop -> landmask
    u = np.asarray(ds.variables["u"][step])              # (21,181,185)
    u = np.where(np.isnan(u), 0.0, u)                    # NaN-fill BEFORE destagger
    u = _destagger_xi(u, mask_u, xi_rho=CCS.grid.native_xi_rho)     # (21,181,186)
    out["u"] = _apply_landmask(u[kept_pos], mask_rho)

    v = np.asarray(ds.variables["v"][step])              # (21,180,186)
    v = np.where(np.isnan(v), 0.0, v)
    v = _destagger_eta(v, mask_v, eta_rho=CCS.grid.native_eta_rho)  # (21,181,186)
    out["v"] = _apply_landmask(v[kept_pos], mask_rho)

    # --- rho-grid 2D: zeta --- landmask (NaN land handled)
    zeta = np.asarray(ds.variables["zeta"][step])        # (181,186)
    out["zeta"] = _apply_landmask(zeta, mask_rho)

    # --- staggered 2D: ubar, vbar --- NaN-fill -> destagger -> landmask
    ubar = np.where(np.isnan(np.asarray(ds.variables["ubar"][step])), 0.0,
                    np.asarray(ds.variables["ubar"][step]))   # (181,185)
    out["ubar"] = _apply_landmask(
        _destagger_xi(ubar, mask_u, xi_rho=CCS.grid.native_xi_rho), mask_rho)
    vbar = np.where(np.isnan(np.asarray(ds.variables["vbar"][step])), 0.0,
                    np.asarray(ds.variables["vbar"][step]))   # (180,186)
    out["vbar"] = _apply_landmask(
        _destagger_eta(vbar, mask_v, eta_rho=CCS.grid.native_eta_rho), mask_rho)

    return out


# ---- Write (MAB-format zarr, resumable scaffold) ----
def build_store(zarr_path, n_time, statics, time_chunk):
    eta, xi = CCS.grid.native_eta_rho, CCS.grid.native_xi_rho
    nlev = CCS.vertical.n_levels_kept
    blosc = [zarr.codecs.BloscCodec(cname="lz4", clevel=5)]
    root = zarr.open(zarr_path, mode="w")
    tchunk = min(time_chunk, n_time)
    # dynamic vars (MAB dim order)
    for v in CCS.channels.dyn3d_vars:                    # (ocean_time, s_rho, eta_rho, xi_rho)
        root.create_array(v, shape=(n_time, nlev, eta, xi),
                          chunks=(tchunk, nlev, eta, xi), dtype=np.float32,
                          compressors=blosc,
                          dimension_names=["ocean_time", "s_rho", "eta_rho", "xi_rho"])
    for v in CCS.channels.dyn2d_vars:                    # (ocean_time, eta_rho, xi_rho)
        root.create_array(v, shape=(n_time, eta, xi),
                          chunks=(tchunk, eta, xi), dtype=np.float32,
                          compressors=blosc,
                          dimension_names=["ocean_time", "eta_rho", "xi_rho"])
    # statics
    for v in CCS.channels.static_vars_2d:
        root.create_array(v, shape=(eta, xi), chunks=(eta, xi), dtype=np.float32,
                          dimension_names=["eta_rho", "xi_rho"])[:] = statics[v]
    for v in CCS.channels.static_vars_1d:
        root.create_array(v, shape=(nlev,), chunks=(nlev,), dtype=np.float32,
                          dimension_names=["s_rho"])[:] = statics[v]
    # ocean_time coord (canonical de-overlapped axis; seconds since config epoch)
    ot = root.create_array("ocean_time", shape=(n_time,), chunks=(n_time,),
                           dtype=np.int64, dimension_names=["ocean_time"])
    ot.attrs["units"] = CCS.raw.time_units
    ot.attrs["calendar"] = CCS.raw.calendar
    return root


def done_marker_path(zarr_path):
    return zarr_path.rstrip("/") + ".done.json"


def load_done(zarr_path):
    p = done_marker_path(zarr_path)
    if os.path.exists(p):
        with open(p) as f:
            return set(json.load(f).get("done", []))
    return set()


def persist_done(zarr_path, done):
    p = done_marker_path(zarr_path)
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"done": sorted(done)}, f)
    os.replace(tmp, p)


# ---- parallel worker scaffold (mirrors the MAB spawn pattern) ----
# WORK-UNIT = one output time-chunk (== zarr time_chunk). Because each canonical
# instant maps (via the manifest) to a UNIQUE output index, and chunks partition
# [0, n) disjointly, no two work-units ever write the same output slice — writes
# are race-free BY CONSTRUCTION (each worker writes one disjoint zarr chunk file).
# A worker opens the ~few owning-files spanning its chunk ONCE (amortizing the
# cold-open cost that made the single-process dry-run ~4 s/instant).
_W = {}


def _worker_init(zarr_path):
    """Per-worker (spawn) init: 1 thread/worker, HDF5 locking off, open store."""
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    _W["root"] = zarr.open(zarr_path, mode="r+")
    _W["statics"] = load_ccs_statics()
    _W["kept"] = kept_level_positions()


def _process_chunk_core(out_start, block, root, statics, kept):
    """Transform `block` (contiguous canonical instants) and write to the
    disjoint output region [out_start, out_start+len(block))."""
    ds_cache = {}
    nblk = len(block)
    try:
        bufs = {}
        for j, e in enumerate(block):
            arrs = transform_instant(e, ds_cache, statics, kept)
            for v, a in arrs.items():
                bufs.setdefault(v, [None] * nblk)[j] = a
        for v, lst in bufs.items():
            root[v][out_start:out_start + nblk] = np.stack(lst)
    finally:
        for d in ds_cache.values():
            try:
                d.close()
            except Exception:
                pass
    return out_start


def _worker_chunk(task):
    out_start, block = task
    return _process_chunk_core(out_start, block, _W["root"], _W["statics"], _W["kept"])


def ccs_run(entries, zarr_path, statics, time_chunk, n_workers=1, persist_every=8):
    """Chunk-parallel, resumable. Work-unit = one output time-chunk (== zarr
    time_chunk). n_workers=1 runs serially in-process (smoke/dry-run path).
    Resumable via a chunk-level done-list (restart skips completed chunks)."""
    n = len(entries)
    T = time_chunk
    # partition [0, n) into disjoint output chunks; assert no overlap
    chunks = [(c * T, entries[c * T:min((c + 1) * T, n)])
              for c in range((n + T - 1) // T)]
    covered = sorted((s, s + len(b)) for s, b in chunks)
    for (s0, e0), (s1, _e1) in zip(covered, covered[1:]):
        assert e0 == s1, f"chunk output ranges overlap/gap: {e0} != {s1}"
    assert covered[0][0] == 0 and covered[-1][1] == n, "chunks miss [0, n)"

    done = load_done(zarr_path)                        # set of completed chunk out_starts
    if os.path.isdir(zarr_path) and done:
        root = zarr.open(zarr_path, mode="r+")
    else:
        root = build_store(zarr_path, n, statics, time_chunk)
        done = set()
    root["ocean_time"][:] = np.array([e["epoch_seconds"] for e in entries], dtype=np.int64)

    todo = [(s, b) for (s, b) in chunks if s not in done]

    if n_workers <= 1:
        kept = kept_level_positions()
        for k, (s, b) in enumerate(todo):
            _process_chunk_core(s, b, root, statics, kept)
            done.add(s)
            if (k + 1) % persist_every == 0:
                persist_done(zarr_path, done)
        persist_done(zarr_path, done)
    else:
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx,
                                    initializer=_worker_init,
                                    initargs=(zarr_path,)) as ex:
            futs = [ex.submit(_worker_chunk, (s, b)) for (s, b) in todo]
            for k, fut in enumerate(cf.as_completed(futs)):
                done.add(fut.result())
                if (k + 1) % persist_every == 0:
                    persist_done(zarr_path, done)
        persist_done(zarr_path, done)
    return root


# ---- Smoke validation (fail-loud) ----
def validate_smoke(zarr_path, entries, statics):
    print("\n" + "=" * 66)
    print("SMOKE VALIDATION (fail-loud)")
    print("=" * 66)
    ds = xr.open_zarr(zarr_path, consolidated=False, decode_times=False)
    n = len(entries)
    eta, xi = CCS.grid.native_eta_rho, CCS.grid.native_xi_rho
    nlev = CCS.vertical.n_levels_kept
    mask_rho = statics["mask_rho"]
    land = mask_rho <= 0.5
    fails = []

    # shapes + dtype
    for v in CCS.channels.dyn3d_vars:
        got = ds[v].shape
        ok = got == (n, nlev, eta, xi) and ds[v].dtype == np.float32
        print(f"  [{'PASS' if ok else 'FAIL'}] {v} shape {got} dtype {ds[v].dtype} (expect ({n},{nlev},{eta},{xi}) f32)")
        if not ok: fails.append(f"{v} shape/dtype")
    for v in CCS.channels.dyn2d_vars:
        got = ds[v].shape
        ok = got == (n, eta, xi) and ds[v].dtype == np.float32
        print(f"  [{'PASS' if ok else 'FAIL'}] {v} shape {got} dtype {ds[v].dtype} (expect ({n},{eta},{xi}) f32)")
        if not ok: fails.append(f"{v} shape/dtype")

    # land == 0 exactly + wet finite
    for v in list(CCS.channels.dyn3d_vars) + list(CCS.channels.dyn2d_vars):
        a = ds[v].values                              # small (smoke)
        is3d = a.ndim == 4
        land_b = land[None, None] if is3d else land[None]
        land_vals = a[np.broadcast_to(land_b, a.shape)]
        land_ok = np.all(land_vals == 0.0)
        wet_b = (~land)[None, None] if is3d else (~land)[None]
        wet_vals = a[np.broadcast_to(wet_b, a.shape)]
        wet_ok = np.all(np.isfinite(wet_vals))
        print(f"  [{'PASS' if land_ok else 'FAIL'}] {v} land==0  |  [{'PASS' if wet_ok else 'FAIL'}] {v} wet finite")
        if not land_ok: fails.append(f"{v} land!=0")
        if not wet_ok: fails.append(f"{v} wet non-finite")

    # value ranges vs config
    def wet_range(v):
        a = ds[v].values
        wb = (~land)[None, None] if a.ndim == 4 else (~land)[None]
        w = a[np.broadcast_to(wb, a.shape)]
        return float(w.min()), float(w.max())
    tmn, tmx = wet_range("temp"); smn, smx = wet_range("salt"); zmn, zmx = wet_range("zeta")
    V = CCS.validation
    print(f"  temp wet [{tmn:.2f},{tmx:.2f}] vs config [{V.temp_min},{V.temp_max}]: "
          f"{'PASS' if V.temp_min <= tmn and tmx <= V.temp_max else 'WARN(surface-only bound)'}")
    print(f"  salt wet [{smn:.2f},{smx:.2f}] vs config [{V.salt_min},{V.salt_max}]: "
          f"{'PASS' if V.salt_min <= smn and smx <= V.salt_max else 'WARN'}")
    print(f"  zeta wet [{zmn:.3f},{zmx:.3f}] vs config |zeta|<{V.zeta_abs_max}: "
          f"{'PASS' if max(abs(zmn),abs(zmx)) < V.zeta_abs_max else 'WARN'}")

    # ocean_time matches manifest first N, 21600s spaced
    ot = ds["ocean_time"].values
    man_ep = np.array([e["epoch_seconds"] for e in entries], dtype=np.int64)
    ot_ok = np.array_equal(ot, man_ep)
    diffs = np.diff(ot)
    spacing_ok = np.all(diffs == CCS.timesplit.cadence_hours * 3600)
    print(f"  [{'PASS' if ot_ok else 'FAIL'}] ocean_time == manifest first {n}  |  "
          f"[{'PASS' if spacing_ok else 'FAIL'}] spacing == {CCS.timesplit.cadence_hours*3600}s")
    if not ot_ok: fails.append("ocean_time mismatch")
    if not spacing_ok: fails.append("ocean_time spacing")

    # statics present + shape
    for v in CCS.channels.static_vars_2d:
        ok = ds[v].shape == (eta, xi)
        if not ok: fails.append(f"static {v} shape")
    for v in CCS.channels.static_vars_1d:
        ok = ds[v].shape == (nlev,)
        if not ok: fails.append(f"static {v} shape")
    print(f"  [{'PASS' if not any('static' in f for f in fails) else 'FAIL'}] statics present + correct shape "
          f"(2d {CCS.channels.static_vars_2d} -> ({eta},{xi}); 1d {CCS.channels.static_vars_1d} -> ({nlev},))")

    # destagger sanity: one coastal wet rho cell, u ≈ mean of flanking u-points
    print("\n  --- destagger sanity (shared _destagger_xi on CCS geometry) ---")
    mask_u = statics["mask_u"]
    kept_pos = kept_level_positions()
    e0 = entries[0]
    with nc.Dataset(os.path.join(ccs_staging_dir(), e0["owning_file"])) as d0:
        u_raw = np.asarray(d0.variables["u"][e0["step_index_within_file"]])   # (21,181,185)
    u_raw = np.where(np.isnan(u_raw), 0.0, u_raw)
    lev = kept_pos[-1]  # surface (of the 21-staged); its position within kept is -1
    # find an interior wet rho column c (1..184) with both flanking u-points wet
    found = None
    for r in range(20, 160):
        for c in range(1, xi - 1):
            if (not land[r, c]) and mask_u[r, c-1] > 0.5 and mask_u[r, c] > 0.5:
                found = (r, c); break
        if found: break
    if found is None:
        print("     (no interior wet cell with both flanking u-points found — skip)")
    else:
        r, c = found
        rho_u_surface = ds["u"].values[0, -1]                 # (181,186) surface of kept
        expect = 0.5 * (u_raw[lev, r, c-1] + u_raw[lev, r, c])
        got = float(rho_u_surface[r, c])
        ok = np.isfinite(got) and abs(got - expect) < 1e-4
        print(f"     rho-cell ({r},{c}) surface u = {got:.5f}; "
              f"0.5*(u[{c-1}]+u[{c}]) = {expect:.5f}; match={ok}")
        if not ok: fails.append("destagger spot-check")

    ds.close()
    print("\n  SMOKE RESULT:", "ALL PASS" if not fails else f"FAIL: {fails}")
    return not fails


def ccs_transform(args):
    man = load_ccs_manifest()
    tl = man["timeline"]

    if args.full:
        # ---- FULL run: whole de-overlapped manifest -> the real store, resumable ----
        out = resolve(CCS.paths.preprocessed_zarr)
        exp = CCS.timesplit.expected_deoverlapped_steps
        if len(tl) != exp:
            _fail(f"manifest has {len(tl)} entries, expected {exp} — refusing full run.",
                  tag="transform")
        entries = tl
        print("=" * 66)
        print(f"CCS preprocess (transform+write) — region '{CCS.name}' — FULL {len(entries)} instants")
        print("=" * 66)
        print(f"manifest: {ccs_manifest_path()}  ({man['_meta']['n_entries']} total entries)")
        print(f"destagger dims: xi_rho={CCS.grid.native_xi_rho} eta_rho={CCS.grid.native_eta_rho}  "
              f"(native {CCS.grid.native_eta_rho}x{CCS.grid.native_xi_rho})")
        print(f"kept level positions (drop-deepest): {kept_level_positions()}")
        print(f"out (REAL store): {out}")
        print(f"resumable: done-marker at {done_marker_path(out)}  "
              f"(existing done chunks are skipped; store is NOT wiped)")
        print(f"workers={args.workers}  time_chunk={args.time_chunk}")
        print(flush=True)
        statics = load_ccs_statics()
        ccs_run(entries, out, statics, args.time_chunk, n_workers=args.workers)
        print(f"\n[transform] FULL write complete -> {out}  ({args.workers} workers)")
        print("Next: validate with datasets/_preprocess_lib.py --region ccs --validate")
        return 0

    # ---- SMOKE run (throwaway) ----
    print("=" * 66)
    print(f"CCS preprocess (transform+write) — region '{CCS.name}' — SMOKE {args.smoke} instants")
    print("=" * 66)
    print(f"manifest: {ccs_manifest_path()}  ({man['_meta']['n_entries']} total entries)")
    print(f"destagger dims (passed explicitly to the shared helpers): "
          f"xi_rho={CCS.grid.native_xi_rho} eta_rho={CCS.grid.native_eta_rho}  "
          f"(native {CCS.grid.native_eta_rho}x{CCS.grid.native_xi_rho})")
    print(f"kept level positions (drop-deepest): {kept_level_positions()}")
    print(f"out (throwaway): {args.out}")
    print()

    entries = tl[:args.smoke]
    if os.path.isdir(args.out):
        shutil.rmtree(args.out)                    # self-cleaning smoke
    for p in (done_marker_path(args.out), done_marker_path(args.out) + ".tmp"):
        if os.path.exists(p):
            os.remove(p)

    statics = load_ccs_statics()
    print(f"statics loaded from {resolve(CCS.paths.static_source)}: "
          f"2d {list(CCS.channels.static_vars_2d)} + masks(u,v) + 1d {list(CCS.channels.static_vars_1d)}")

    ccs_run(entries, args.out, statics, args.time_chunk, n_workers=args.workers)
    print(f"wrote {len(entries)} instants -> {args.out}  ({args.workers} workers)")

    ok = validate_smoke(args.out, entries, statics)
    if not ok:
        _fail("smoke validation failed", tag="transform")
    print("\n[transform] SMOKE PASS")
    return 0


# ============================================================
# CCS clip-temp: one-time in-place clip of impossible temp artifacts
# ============================================================
def ccs_clip_temp():
    """One-time in-place clip of physically-impossible temp artifacts, with a
    full pre-clip audit log so the raw values stay on the record.

    Full-store validation found wet temp cells outside the locked physical
    bound [0, 28] C: one corrupt source column at 2014-10-19 (reading -6.12 /
    +30.04 / +33.58 at cell 47.9N/124.6W) plus a coastal-WA cluster of
    sub-zero (~-1.5 C) cells across ~17 steps. Total ~2.7e-6% of wet values,
    all physically impossible. They must be removed BEFORE normalization
    stats so the +33.58 outlier can't distort the mean/std.

    Two phases:
      PHASE 1 (read-only): enumerate EVERY cell outside [temp_min, temp_max],
        record (t, timestamp, level, eta, xi, lat, lon, raw, clipped) and
        WRITE the audit log (ccs_temp_clip_log.{md,csv}, beside the store)
        to disk BEFORE any store write.
      PHASE 2 (write): clip only the affected zarr time-chunks in place
        (temp only), then verify (range within bound, count,
        ocean_time/salt/shapes unchanged).

    Bounds come from the CCS RegionConfig; nothing hardcoded. Idempotent: a
    second run finds 0 out-of-range cells and writes nothing.
    """
    store = resolve(CCS.paths.preprocessed_zarr)
    log_md = Path(os.path.dirname(store)) / "ccs_temp_clip_log.md"
    log_csv = Path(os.path.dirname(store)) / "ccs_temp_clip_log.csv"
    lo, hi = float(CCS.validation.temp_min), float(CCS.validation.temp_max)
    root = zarr.open(store, mode="r+")
    temp = root["temp"]
    n = temp.shape[0]
    Tc = temp.chunks[0]
    lat = np.asarray(root["lat_rho"][:]); lon = np.asarray(root["lon_rho"][:])
    mask = np.asarray(root["mask_rho"][:]) > 0.5
    tl = load_ccs_manifest()["timeline"]
    print(f"store: {store}")
    print(f"temp shape={temp.shape} chunks={temp.chunks} bound=[{lo},{hi}]")

    # ---------- PHASE 1: enumerate (read-only) ----------
    records = []                                        # dicts
    affected_chunks = set()
    for t0 in range(0, n, Tc):
        t1 = min(t0 + Tc, n)
        blk = temp[t0:t1]
        out = (blk < lo) | (blk > hi)
        if out.any():
            for (tt, k, i, j) in np.argwhere(out):
                t = int(t0 + tt)
                raw = float(blk[tt, k, i, j])
                records.append({
                    "t_index": t,
                    "timestamp": tl[t]["timestamp_iso"],
                    "level": int(k), "eta": int(i), "xi": int(j),
                    "lat": round(float(lat[i, j]), 4), "lon": round(float(lon[i, j]), 4),
                    "wet": bool(mask[i, j]),
                    "raw_value": round(raw, 5),
                    "clipped_value": round(min(max(raw, lo), hi), 5),
                })
                affected_chunks.add(t0)
        if (t0 // Tc) % 100 == 0:
            print(f"  [scan] ...{t1}/{n}  found={len(records)}", flush=True)

    n_out = len(records)
    n_cold = sum(1 for r in records if r["raw_value"] < lo)
    n_hot = sum(1 for r in records if r["raw_value"] > hi)
    steps = sorted({r["t_index"] for r in records})
    all_wet = all(r["wet"] for r in records)
    print(f"\n[PHASE 1] cells outside [{lo},{hi}]: {n_out}  (<{lo}: {n_cold}, >{hi}: {n_hot})  "
          f"steps={len(steps)}  all wet={all_wet}")

    _write_clip_log(records, lo, hi, n_out, n_cold, n_hot, steps,
                    store, log_md, log_csv)
    print(f"[PHASE 1] audit log written: {log_md.name} + {log_csv.name}  ({n_out} rows)")

    # ---------- PHASE 2: clip affected chunks (write) ----------
    ot_before = np.asarray(root["ocean_time"][:]).copy()
    salt_probe_before = np.asarray(root["salt"][n // 2, -1]).copy()
    all_vars = (list(CCS.channels.dyn3d_vars) + list(CCS.channels.dyn2d_vars)
                + list(CCS.channels.static_vars_2d) + list(CCS.channels.static_vars_1d)
                + ["ocean_time"])
    shapes_before = {k: tuple(root[k].shape) for k in all_vars}

    modified = 0
    for t0 in sorted(affected_chunks):
        t1 = min(t0 + Tc, n)
        blk = temp[t0:t1]
        c = int(((blk < lo) | (blk > hi)).sum())
        np.clip(blk, lo, hi, out=blk)
        temp[t0:t1] = blk
        modified += c
    print(f"[PHASE 2] clipped {modified} cells across {len(affected_chunks)} chunks")

    # ---------- verification ----------
    post_min, post_max = np.inf, -np.inf
    remain = 0
    for t0 in sorted(affected_chunks):
        t1 = min(t0 + Tc, n)
        blk = temp[t0:t1]                               # (T,20,eta,xi)
        w = blk[np.broadcast_to(mask, blk.shape)]       # wet cells only
        post_min = min(post_min, float(w.min())); post_max = max(post_max, float(w.max()))
        remain += int(((blk < lo) | (blk > hi)).sum())

    ot_after = np.asarray(root["ocean_time"][:])
    salt_after = np.asarray(root["salt"][n // 2, -1])
    shapes_after = {k: tuple(root[k].shape) for k in all_vars}
    ot_ok = bool(np.array_equal(ot_before, ot_after))
    salt_ok = bool(np.array_equal(salt_probe_before, salt_after))
    shapes_ok = shapes_before == shapes_after

    print("\n=== CLIP VERIFICATION ===")
    print(f"cells modified: {modified}  (expected {n_out})   match={modified == n_out}")
    print(f"affected steps ({len(steps)}): {steps}")
    print(f"post-clip temp wet range over affected chunks: [{post_min:.4f}, {post_max:.4f}]  "
          f"within [{lo},{hi}]={post_min >= lo - 1e-6 and post_max <= hi + 1e-6}")
    print(f"remaining out-of-range cells: {remain}  (expect 0)")
    print(f"ocean_time unchanged: {ot_ok}")
    print(f"salt untouched (probe step {n//2}, surface): {salt_ok}")
    print(f"all array shapes unchanged: {shapes_ok}")
    all_ok = (modified == n_out and remain == 0 and ot_ok and salt_ok and shapes_ok
              and post_min >= lo - 1e-6 and post_max <= hi + 1e-6)
    print("\nCLIP:", "OK" if all_ok else "FAIL")
    return 0 if all_ok else 1


def _write_clip_log(records, lo, hi, n_out, n_cold, n_hot, steps,
                    store, log_md, log_csv):
    with open(log_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else
                           ["t_index", "timestamp", "level", "eta", "xi", "lat", "lon",
                            "wet", "raw_value", "clipped_value"])
        w.writeheader()
        for r in records:
            w.writerow(r)

    corrupt = [r for r in records if r["raw_value"] < -3 or r["raw_value"] > hi]
    etas = {}
    for r in records:
        etas[r["eta"]] = etas.get(r["eta"], 0) + 1
    top_eta = sorted(etas.items(), key=lambda kv: -kv[1])[:6]
    lines = []
    A = lines.append
    A("# CCS temp-artifact clip log (pre-clip raw values on record)\n")
    A(f"**Date:** {time.strftime('%Y-%m-%d')}  **Store:** `{store}`\n")
    A(f"These are physically-impossible source-reanalysis artifacts in `temp`, "
      f"**clipped in place to the locked physical bound [{lo}, {hi}] C** before "
      f"normalization stats so the +33.58 C outlier can't distort mean/std. Bound "
      f"rationale: observed clean 40-yr range (real max 26.44 C SoCal Aug-2018, deep "
      f"floor ~1.5 C) — see the CCS RegionConfig validation ranges. "
      f"Raw values captured here BEFORE the clip; the store now holds the clipped values. "
      f"Full per-cell enumeration in `ccs_temp_clip_log.csv`.\n")
    A(f"## Summary\n")
    A(f"- cells clipped: **{n_out}**  (below {lo}: {n_cold}; above {hi}: {n_hot})")
    A(f"- distinct time steps affected: **{len(steps)}**")
    A(f"- all clipped cells are wet (mask_rho>0.5): "
      f"{all(r['wet'] for r in records)}")
    A(f"- eta-row concentration (row: count): {dict(top_eta)}  "
      f"(coastal-WA cluster ~rows 167-171)\n")
    A(f"## Corrupt-column cells (|raw| extreme — the 2014-10-19 source glitch)\n")
    A("| t | timestamp | level | eta | xi | lat | lon | raw | clipped |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in corrupt:
        A(f"| {r['t_index']} | {r['timestamp']} | {r['level']} | {r['eta']} | {r['xi']} | "
          f"{r['lat']} | {r['lon']} | {r['raw_value']} | {r['clipped_value']} |")
    A(f"\n## Sub-zero coastal cells (sample; full list in CSV)\n")
    A("| t | timestamp | level | eta | xi | lat | lon | raw | clipped |")
    A("|---|---|---|---|---|---|---|---|---|")
    for r in [x for x in records if lo > x["raw_value"] >= -3][:15]:
        A(f"| {r['t_index']} | {r['timestamp']} | {r['level']} | {r['eta']} | {r['xi']} | "
          f"{r['lat']} | {r['lon']} | {r['raw_value']} | {r['clipped_value']} |")
    A("")
    log_md.write_text("\n".join(lines))


# ============================================================
# CCS validation (full store, read-only)
# ============================================================
def ccs_validate():
    """Validate the produced CCS preprocessed store.

    Read-only, run AFTER the full transform completes. Every threshold comes
    from the CCS RegionConfig; NO CCS constant is hardcoded. Emits a report
    file + PASS/FAIL tally, HARD-FAILs (non-zero exit) on any
    structural/physical violation, and SOFT-reports the temp range (the temp
    bound stays provisional until locked after review).
    """
    store = resolve(CCS.paths.preprocessed_zarr)
    report = os.path.join(os.path.dirname(store),
                          "ccs_preprocessing_validation_report.txt")

    lines = []
    n_pass = 0
    n_fail = 0

    def chk(name, ok, detail=""):
        nonlocal n_pass, n_fail
        tag = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        line = f"[{tag}] {name} :: {detail}"
        print(line, flush=True)
        lines.append(line)

    def soft(name, detail=""):
        line = f"[SOFT] {name} :: {detail}"
        print(line, flush=True)
        lines.append(line)

    lines.append(f"CCS preprocessing validation  ({datetime.now(timezone.utc).isoformat()})")
    lines.append(f"store: {store}")

    if not os.path.isdir(store):
        chk("store exists", False, store)
        _emit_ccs_report(lines, n_pass, n_fail, report)
        return 1

    ds = xr.open_zarr(store, consolidated=False, decode_times=False)

    eta, xi = CCS.grid.native_eta_rho, CCS.grid.native_xi_rho
    nlev = CCS.vertical.n_levels_kept
    exp_n = CCS.timesplit.expected_deoverlapped_steps
    cadence_s = CCS.timesplit.cadence_hours * 3600

    # ---- time axis ----
    ot = ds["ocean_time"].values.astype(np.int64)
    chk(f"ocean_time count == {exp_n}", len(ot) == exp_n, f"{len(ot)}")
    diffs = np.diff(ot)
    chk(f"ocean_time diff == {cadence_s}s, monotonic, no gap/dup",
        bool(np.all(diffs == cadence_s)),
        "ok" if np.all(diffs == cadence_s) else f"{int((diffs!=cadence_s).sum())} bad")

    # ---- shapes + dtype ----
    n = len(ot)
    for v in CCS.channels.dyn3d_vars:
        chk(f"{v} shape ({n},{nlev},{eta},{xi}) f32",
            tuple(ds[v].shape) == (n, nlev, eta, xi) and ds[v].dtype == np.float32,
            f"{tuple(ds[v].shape)} {ds[v].dtype}")
    for v in CCS.channels.dyn2d_vars:
        chk(f"{v} shape ({n},{eta},{xi}) f32",
            tuple(ds[v].shape) == (n, eta, xi) and ds[v].dtype == np.float32,
            f"{tuple(ds[v].shape)} {ds[v].dtype}")

    # ---- statics present + shape ----
    for v in CCS.channels.static_vars_2d:
        chk(f"static {v} ({eta},{xi})", tuple(ds[v].shape) == (eta, xi), f"{tuple(ds[v].shape)}")
    for v in CCS.channels.static_vars_1d:
        chk(f"static {v} ({nlev},)", tuple(ds[v].shape) == (nlev,), f"{tuple(ds[v].shape)}")

    # ---- land==0 + wet finite (stream over time to bound memory) ----
    mask_rho = ds["mask_rho"].values
    land = mask_rho <= 0.5
    n_wet = int((~land).sum())
    all_vars = list(CCS.channels.dyn3d_vars) + list(CCS.channels.dyn2d_vars)
    # sample a stride of timesteps for the exhaustive land/finite check + track wet ranges
    stride = max(1, n // 200)                       # ~200 timesteps sampled
    tsel = list(range(0, n, stride))
    ranges = {}
    land_bad = {v: 0 for v in all_vars}
    wet_nonfinite = {v: 0 for v in all_vars}
    for v in all_vars:
        vmin, vmax = np.inf, -np.inf
        arr = ds[v]
        is3d = arr.ndim == 4
        for t in tsel:
            a = arr[t].values
            lb = land[None] if is3d else land
            lvals = a[np.broadcast_to(lb, a.shape)]
            if not np.all(lvals == 0.0):
                land_bad[v] += 1
            wb = (~land)[None] if is3d else (~land)
            w = a[np.broadcast_to(wb, a.shape)]
            if not np.all(np.isfinite(w)):
                wet_nonfinite[v] += 1
            if w.size:
                vmin = min(vmin, float(w.min())); vmax = max(vmax, float(w.max()))
        ranges[v] = (vmin, vmax)
        chk(f"{v} land==0 (sampled {len(tsel)} steps)", land_bad[v] == 0, f"{land_bad[v]} bad")
        chk(f"{v} wet finite (sampled)", wet_nonfinite[v] == 0, f"{wet_nonfinite[v]} bad")

    # ---- physical ranges: salt/zeta HARD, temp SOFT ----
    V = CCS.validation
    smn, smx = ranges["salt"]
    chk(f"salt within [{V.salt_min},{V.salt_max}]", V.salt_min <= smn and smx <= V.salt_max,
        f"[{smn:.2f},{smx:.2f}]")
    zmn, zmx = ranges["zeta"]
    chk(f"|zeta| < {V.zeta_abs_max}", max(abs(zmn), abs(zmx)) < V.zeta_abs_max,
        f"[{zmn:.3f},{zmx:.3f}]")
    tmn, tmx = ranges["temp"]
    within = V.temp_min <= tmn and tmx <= V.temp_max
    # suggested bound = observed range with a small margin, rounded outward
    lo_s = float(np.floor(tmn - 1.0)); hi_s = float(np.ceil(tmx + 1.0))
    soft("temp range (SOFT — provisional bound)",
         f"observed [{tmn:.2f},{tmx:.2f}] vs config [{V.temp_min},{V.temp_max}] "
         f"({'within' if within else 'OUTSIDE'}); SUGGESTED bound [{lo_s},{hi_s}] "
         f"(sampled {len(tsel)} steps — for locking after review)")

    ds.close()
    lines.append(f"\nwet cells per level-slice: {n_wet}  (land: {int(land.sum())})")
    _emit_ccs_report(lines, n_pass, n_fail, report)
    return 0 if n_fail == 0 else 1


def _emit_ccs_report(lines, n_pass, n_fail, report):
    lines.append(f"\nPASS={n_pass} FAIL={n_fail}")
    tmp = report + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, report)
    print(f"\n=== PASS={n_pass} FAIL={n_fail} — report: {report} ===")


# ============================================================
# Dispatcher
# ============================================================
MAB_STAGES = ("run", "validate", "all")
CCS_STAGES = ("build-timeline", "transform", "clip-temp", "validate", "all")


def main():
    exp = CCS.timesplit.expected_deoverlapped_steps
    ap = argparse.ArgumentParser(
        description="Regional preprocessing: staged raw NetCDF -> chunked Zarr "
                    "(see the module docstring for the per-region stages).")
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--stage", default=None,
                    choices=sorted(set(MAB_STAGES) | set(CCS_STAGES)),
                    help="mab: run|validate|all (default run); "
                         "ccs: build-timeline|transform|clip-temp|validate|all "
                         "(default all = the four stages in order, transform full)")
    ap.add_argument("--validate", action="store_true",
                    help="shorthand for --stage validate")
    ap.add_argument("--clip-temp", action="store_true",
                    help="shorthand for --stage clip-temp (ccs only)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (spawn). mab default 16; "
                         "ccs default 1 = serial in-process")
    ap.add_argument("--smoke", type=int, default=96,
                    help="[ccs transform] transform the first N canonical instants "
                         f"(smoke). Full run (all {exp}) needs --full.")
    ap.add_argument("--full", action="store_true",
                    help="[ccs transform] full run into the real store (resumable)")
    ap.add_argument("--out", type=str, default="/tmp/ccs_smoke.zarr",
                    help="[ccs transform smoke] throwaway smoke output "
                         "(NOT the real store)")
    ap.add_argument("--time-chunk", type=int, default=72,
                    help="[ccs transform] output zarr time-chunk "
                         "(default 72, matching the MAB store)")
    args = ap.parse_args()

    stage = args.stage
    if args.validate:
        stage = "validate"
    if args.clip_temp:
        stage = "clip-temp"
    if stage is None:
        stage = "run" if args.region == "mab" else "all"

    if args.region == "mab":
        if stage not in MAB_STAGES:
            ap.error(f"stage {stage!r} is not a mab stage; choose from {MAB_STAGES}")
        if stage == "run":
            return mab_run(args.workers or N_WORKERS)
        if stage == "validate":
            return mab_validate()
        rc = mab_run(args.workers or N_WORKERS)          # all
        if rc:
            return rc
        return mab_validate()

    # ccs
    if stage not in CCS_STAGES:
        ap.error(f"stage {stage!r} is not a ccs stage; choose from {CCS_STAGES}")
    if args.workers is None:
        args.workers = 1
    if stage == "build-timeline":
        return ccs_build_timeline()
    if stage == "transform":
        return ccs_transform(args)
    if stage == "clip-temp":
        return ccs_clip_temp()
    if stage == "validate":
        return ccs_validate()
    # all: build-timeline -> transform (full) -> clip-temp -> validate
    rc = ccs_build_timeline()
    if rc:
        return rc
    args.full = True
    rc = ccs_transform(args)
    if rc:
        return rc
    rc = ccs_clip_temp()
    if rc:
        return rc
    return ccs_validate()


if __name__ == "__main__":
    sys.exit(main())
