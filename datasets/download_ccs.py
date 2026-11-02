#!/usr/bin/env python
"""CCSRA31 production downloader — per-file OPeNDAP fetch, resumable, robust.

Stages the full ~3,590-file CCSRA31 posterior archive from UCSC's THREDDS
server, with the robustness a multi-day run over an old server needs
(manifest, resume-by-skip, exponential backoff, sticky-retry logging,
circuit breaker, per-file staging).

Streaming fetch core
--------------------
`_copy_dims`, `_create_empty_var`, `_copy_var_attrs`, `_write_var_streamed`
and `peak_rss_mb` implement a memory-safe streaming fetch (~110 MB peak RSS):
peak RAM is one (time_block × everything-else) slice per variable. The
streaming pattern is the memory-safety guarantee; do not change it.
Variables in `S_RHO_SUBSET_VARS` (and the 3D dynamic vars) get
`.isel(s_rho=S_LEVELS)` applied to trim 42 -> 21 σ-levels during the
network fetch.

Robustness patterns
-------------------
`is_throttle` error classification, backoff `min(30*2**(attempt-1), 600)`,
`STICKY=50` sticky-retry warnings, `BREAKER=10`
consecutive-failures-with-no-success circuit breaker, atomic tmp-then-rename
manifest persistence, and a main pass followed by a failed-list sweep pass.

Where to run
------------
On a host with unrestricted outbound TCP to `oceanmodeling.ucsc.edu` (HPC
compute/container nodes are often TCP-blocked; a login or data-transfer node
typically works). Run under tmux for a multi-day babysit; the download is
expected to take on the order of days over an old server.

Prerequisites
-------------
1. `pydap` installed (netCDF4 builds often disable DAP support; pydap is
   xarray's fallback via `engine="pydap"`).
2. SSL env vars pointing at the system CA bundle (see check_env).
3. Enough space for staging (~475 GB at the 21-level subset).

Layout
------
Per-file staging: one NetCDF per source file, filename preserved so the
downstream preprocessing step can dedup by timestamp. Static / grid vars
are fetched ONCE from the earliest core file into a separate
`ccs_static.nc`; concatenation + timestamp-dedup + level-drop-to-20 are
deferred to the preprocessing step (datasets/_preprocess_lib.py --region ccs):
stage raw, engineer locally.

Time is staged **raw numeric** (`decode_times=False`) with `units` /
`calendar` attrs preserved — sidesteps the netCDF4 datetime64 buffer
limitation and lets the preprocessing step decode all files consistently.

The staging directory defaults to the CCS RegionConfig `raw.staging_dir`
resolved against $CLOSEDLOOP_DATA_ROOT; override with --out.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import ssl
import sys
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import netCDF4 as nc
import xarray as xr

from closedloop_sampling.regions import get_config
from closedloop_sampling.paths import resolve

CFG = get_config("ccs")


# ============================================================
# Constants — locked
# ============================================================
# Every OTHER level from 1..41 = 21 levels (config staged_level_indices).
# Drop-to-20 happens LATER in preprocessing; the downloader stages 21 so
# the preprocessing step has a choice.
S_LEVELS = list(CFG.vertical.staged_level_indices)

DYN3D = list(CFG.channels.dyn3d_vars)
DYN2D = list(CFG.channels.dyn2d_vars)
DYNAMIC_VARS = DYN3D + DYN2D

# Coordinate axes we need to materialise so the staged NetCDF is
# self-describing. s_rho / Cs_r get subset to S_LEVELS like the MAB
# staging does.
COORD_VARS = ["ocean_time", "s_rho", "s_w", "Cs_r", "Cs_w"]

# Static / grid / s-coordinate metadata (fetched ONCE into ccs_static.nc,
# NOT per-file).
STATIC_VARS = [
    # Bathy + masks
    "h", "mask_rho", "mask_u", "mask_v", "mask_psi",
    # Geometry
    "lat_rho", "lon_rho", "lat_u", "lon_u", "lat_v", "lon_v",
    "lat_psi", "lon_psi", "angle", "pm", "pn", "f",
    # s-coordinate scalars/machinery (s_rho, s_w, Cs_r, Cs_w already in COORD_VARS)
    "Vtransform", "Vstretching", "theta_s", "theta_b", "Tcline", "hc",
]

# Variables that require .isel(s_rho=S_LEVELS) during fetch. Note: DYN3D
# also implicitly need this, handled in the fetch path directly.
S_RHO_SUBSET_VARS = {"s_rho", "Cs_r"}

# Robustness constants.
STICKY = 50                              # sticky-retry-count warning cadence
BREAKER = 10                             # consec fails w/ zero successes → escalate
BACKOFF_BASE = 30                        # seconds
BACKOFF_CAP = 600                        # seconds
DEFAULT_RETRIES = 5

DEFAULT_SLEEP = 3.0                      # pacing between files
DEFAULT_TIME_BLOCK = 8                   # timesteps per DAP fetch (probe-validated)

# THREDDS catalog + dodsC endpoints for both collections.
CATALOGS = {
    "core": {
        "html": ("https://oceanmodeling.ucsc.edu/thredds/catalog/"
                 "wc12.0_ccsra31_01/posterior/catalog.html"),
        "dodsc_base": ("https://oceanmodeling.ucsc.edu/thredds/dodsC/"
                       "wc12.0_ccsra31_01/posterior/"),
        "prefix": "wc12_ccsra31_his_",
        "expected_count": 2827,
    },
    "ext": {
        "html": ("https://oceanmodeling.ucsc.edu/thredds/catalog/"
                 "wc12.0_ccsra31_ext_01/posterior/catalog.html"),
        "dodsc_base": ("https://oceanmodeling.ucsc.edu/thredds/dodsC/"
                       "wc12.0_ccsra31_ext_01/posterior/"),
        "prefix": "fwd_",
        "expected_count": 763,
    },
}

# Filename regexes. NB: extension pattern requires `.nc` immediately
# after the second date-block, so `_derived_vars.nc` companions are
# naturally excluded (there's `_derived_vars` between the date and `.nc`
# in those). Belt-and-braces filter applied on top.
CORE_FILE_RE = re.compile(r"(wc12_ccsra31_his_(\d{8})_(\d{8})\.nc)")
EXT_FILE_RE = re.compile(r"(fwd_\d+_001_(\d{8})_(\d{8})\.nc)")

STATIC_FILENAME = "ccs_static.nc"
MANIFEST_FILENAME = "ccs_download_manifest.json"
LOG_FILENAME = "ccs_download.log"

CATALOG_FETCH_TIMEOUT = 90               # seconds per HTML GET
CATALOG_FETCH_RETRIES = 3


# ============================================================
# Logging
# ============================================================
def log(logf: Path, msg: str) -> None:
    """Timestamped, stdout + file."""
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line, flush=True)
    with open(logf, "a") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ============================================================
# Environment + memory helpers
# ============================================================
def check_env(logf: Path) -> None:
    """Verify SSL env vars; warn (don't abort) if unset."""
    log(logf, "=" * 60)
    log(logf, "Environment check")
    log(logf, "=" * 60)
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        val = os.environ.get(var)
        if val:
            log(logf, f"  {var} = {val}")
        else:
            log(logf, f"  WARNING: {var} not set — expect "
                      f"CERTIFICATE_VERIFY_FAILED on OPeNDAP open.")
            log(logf, f"  Suggested value: /etc/pki/tls/certs/ca-bundle.crt")


def peak_rss_mb() -> Optional[float]:
    """Process max RSS in MB (Linux `ru_maxrss` is KB)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


# ============================================================
# Throttle / error classification
# ============================================================
def is_throttle(e: BaseException) -> bool:
    """Throttle is the "NC_EAUTH" / "authorization failure" family, which
    despite the name is the server's rate-limit signal."""
    s = str(e).lower()
    return ("authorization failure" in s
            or "nc_eauth" in s
            or "too many requests" in s
            or "503" in s
            or "connection reset" in s
            or "read timeout" in s)


# ============================================================
# Catalog enumeration
# ============================================================
def _http_get(url: str, timeout: int = CATALOG_FETCH_TIMEOUT) -> str:
    """Fetch a URL as text via urllib.request. Uses default SSL context
    which respects SSL_CERT_FILE."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(
        url, headers={"User-Agent": "ladcast-regional-ccs-downloader"},
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


def enumerate_collection(name: str, cfg: dict, logf: Path) -> list:
    """Fetch the THREDDS catalog HTML and extract state-file records.

    Returns: list of dicts
        {"collection", "filename", "url", "date_start", "date_end"}
    """
    regex = CORE_FILE_RE if name == "core" else EXT_FILE_RE
    log(logf, f"[enum] fetching catalog for '{name}': {cfg['html']}")

    last_err = None
    for attempt in range(1, CATALOG_FETCH_RETRIES + 1):
        try:
            html = _http_get(cfg["html"])
            break
        except Exception as e:
            last_err = e
            log(logf, f"[enum] '{name}' catalog fetch attempt "
                      f"{attempt}/{CATALOG_FETCH_RETRIES} failed: "
                      f"{type(e).__name__}: {e}")
            if attempt < CATALOG_FETCH_RETRIES:
                time.sleep(min(30 * attempt, 90))
    else:
        raise RuntimeError(
            f"catalog fetch failed for '{name}' after "
            f"{CATALOG_FETCH_RETRIES} attempts: {last_err}"
        )

    seen_names = set()
    records = []
    for match in regex.finditer(html):
        fname = match.group(1)
        # Belt-and-braces: skip anything that ended up matching a
        # derived_vars companion (regex alone should already exclude).
        if "_derived_vars" in fname:
            continue
        if fname in seen_names:
            continue
        seen_names.add(fname)
        date_start = match.group(2)
        date_end = match.group(3)
        records.append({
            "collection": name,
            "filename": fname,
            "url": cfg["dodsc_base"] + fname,
            "date_start": date_start,
            "date_end": date_end,
            "status": "pending",
            "attempts": 0,
            "elapsed_s": None,
            "size_bytes": None,
            "error": None,
            "last_updated_utc": None,
        })

    log(logf, f"[enum] '{name}' found {len(records)} state files "
              f"(expected ~{cfg['expected_count']})")
    if abs(len(records) - cfg["expected_count"]) > 100:
        log(logf, f"[enum] WARNING: '{name}' file count "
                  f"{len(records)} differs from expected "
                  f"{cfg['expected_count']} by > 100 — sanity check the "
                  f"catalog HTML.")

    records.sort(key=lambda r: r["date_start"])
    return records


def enumerate_all(collections: list, logf: Path) -> list:
    all_records = []
    for name in collections:
        cfg = CATALOGS[name]
        all_records.extend(enumerate_collection(name, cfg, logf))
    log(logf, f"[enum] total files across "
              f"{'+'.join(collections)}: {len(all_records)}")
    return all_records


# ============================================================
# Manifest I/O
# ============================================================
def load_or_init_manifest(mpath: Path, collections: list, logf: Path) -> dict:
    """Load existing manifest if present, else enumerate and create."""
    if mpath.exists():
        log(logf, f"[manifest] resuming from {mpath}")
        with open(mpath) as fh:
            m = json.load(fh)
        n_done = sum(1 for f in m["files"] if f["status"] == "done")
        n_failed = sum(1 for f in m["files"] if f["status"] == "failed")
        n_pending = sum(1 for f in m["files"] if f["status"] == "pending")
        log(logf, f"[manifest] {n_done} done, {n_failed} failed, "
                  f"{n_pending} pending "
                  f"(of {len(m['files'])} total)")
        return m

    log(logf, f"[manifest] initialising fresh — enumerating catalogs")
    files = enumerate_all(collections, logf)
    m = {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "collections": collections,
        "s_levels": S_LEVELS,
        "dynamic_vars": DYNAMIC_VARS,
        "static_vars": STATIC_VARS,
        "coord_vars": COORD_VARS,
        "files": files,
    }
    persist_manifest(mpath, m)
    log(logf, f"[manifest] wrote {len(files)} entries -> {mpath}")
    return m


def persist_manifest(mpath: Path, manifest: dict) -> None:
    """Atomic tmp-then-rename."""
    tmp = mpath.with_suffix(mpath.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, mpath)


# ============================================================
# Streaming fetch core (memory-safe; see module docstring) with
# s_rho_subset support for DYN3D + Cs_r + s_rho.
# ============================================================
def _copy_dims(src: xr.Dataset, dst: nc.Dataset, want: set,
               s_rho_len: Optional[int] = None) -> None:
    """Create only the dimensions we'll actually use, matching sizes.
    If `s_rho_len` is set, override s_rho dim to that (post-subset) size.
    """
    for name in want:
        if name in src.dims:
            if name == "s_rho" and s_rho_len is not None:
                dst.createDimension(name, s_rho_len)
            else:
                dst.createDimension(name, src.sizes[name])


def _create_empty_var(dst: nc.Dataset, name: str, dims: tuple,
                      dtype: np.dtype) -> nc.Variable:
    """Create a variable in the destination with no data yet.

    COORD_VARS keep native numeric dtype (float64 or int64 depending on
    file's `units` epoch). Dynamic + static vars are cast to float64.
    """
    if name in COORD_VARS:
        return dst.createVariable(name, dtype, dims, zlib=False)
    return dst.createVariable(name, np.float64, dims, zlib=False)


def _copy_var_attrs(src_da: xr.DataArray, dst_var: nc.Variable) -> None:
    """Preserve CF attrs (units, calendar, long_name, …) verbatim.
    Critical for `ocean_time` — its units/calendar are what the local
    preprocessing step uses to decode times uniformly across files."""
    for k, v in src_da.attrs.items():
        if k.startswith("_"):
            continue
        try:
            dst_var.setncattr(k, v)
        except Exception:
            pass


def _write_var_streamed(
    src: xr.Dataset, dst: nc.Dataset, var: str,
    time_block: int, s_rho_subset: Optional[list] = None,
) -> None:
    """Stream `var` from remote to local in time-blocks.

    Peak RAM = one (time_block × everything-else) slice.
    If `s_rho_subset` is set AND var has `s_rho` dim, subsets that axis
    during the DAP fetch (matches the MAB staging pattern).
    """
    src_da = src[var]
    if s_rho_subset is not None and "s_rho" in src_da.dims:
        src_da = src_da.isel(s_rho=s_rho_subset)
    dims = src_da.dims
    dtype = src_da.dtype

    if var not in dst.variables:
        dst_var = _create_empty_var(dst, var, dims, dtype)
        _copy_var_attrs(src_da, dst_var)
    else:
        dst_var = dst.variables[var]

    if "ocean_time" not in dims:
        dst_var[:] = src_da.values
        gc.collect()
        return

    time_axis = dims.index("ocean_time")
    n_time = src.sizes["ocean_time"]
    for t0 in range(0, n_time, time_block):
        t1 = min(t0 + time_block, n_time)
        chunk = src_da.isel(ocean_time=slice(t0, t1)).values
        if time_axis == 0:
            dst_var[t0:t1] = chunk
        else:
            sel = [slice(None)] * dst_var.ndim
            sel[time_axis] = slice(t0, t1)
            dst_var[tuple(sel)] = chunk
        del chunk
        gc.collect()


# ============================================================
# Per-file OPeNDAP fetch (state file — dynamic + coord vars only)
# ============================================================
def fetch_state_file(url: str, out_path: Path, time_block: int) -> None:
    """Fetch one state file (dynamic + coord vars, s_rho subset applied
    to DYN3D). Writes an atomic tmp file then renames on success so a
    partial write never masquerades as a completed file."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    ds = xr.open_dataset(url, engine="pydap", decode_times=False)

    # Kept var list: coord vars + dynamic vars. No static/grid here.
    wanted = [c for c in COORD_VARS if c in ds.variables]
    wanted += [v for v in DYNAMIC_VARS if v in ds.variables]

    try:
        needed_dims = set()
        for v in wanted:
            src_da = ds[v]
            if v in DYN3D or v in S_RHO_SUBSET_VARS:
                src_da = src_da.isel(s_rho=S_LEVELS)
            needed_dims.update(src_da.dims)

        with nc.Dataset(tmp_path, mode="w", format="NETCDF4") as dst:
            _copy_dims(ds, dst, needed_dims, s_rho_len=len(S_LEVELS))

            # Non-time-varying (coords like s_rho / s_w / Cs_r / Cs_w)
            # first, then time-varying streamed.
            statics_and_coords = [v for v in wanted
                                  if "ocean_time" not in ds[v].dims]
            dynamics = [v for v in wanted if "ocean_time" in ds[v].dims]

            for v in statics_and_coords:
                subset = S_LEVELS if v in S_RHO_SUBSET_VARS else None
                _write_var_streamed(ds, dst, v, time_block, s_rho_subset=subset)
            for v in dynamics:
                subset = S_LEVELS if v in DYN3D else None
                _write_var_streamed(ds, dst, v, time_block, s_rho_subset=subset)

            dst.setncattr("source_url", url)
            dst.setncattr("staged_utc",
                          datetime.now(timezone.utc).isoformat())
            dst.setncattr("staged_by", "download_ccs.py")
            dst.setncattr("s_levels_from_original_42", str(S_LEVELS))
    finally:
        try:
            ds.close()
        except Exception:
            pass

    os.replace(tmp_path, out_path)


def fetch_static_file(url: str, out_path: Path, time_block: int,
                      logf: Path) -> None:
    """Fetch static / grid / s-coordinate vars from `url` into a
    standalone `ccs_static.nc`. Called ONCE, from the earliest core file."""
    log(logf, f"[static] fetching static/grid vars from {url}")
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    ds = xr.open_dataset(url, engine="pydap", decode_times=False)

    wanted = [v for v in STATIC_VARS if v in ds.variables]
    # Also include the s-coord axes so the file is self-describing.
    wanted += [c for c in COORD_VARS
               if c in ds.variables and c != "ocean_time"]
    missing = [v for v in (STATIC_VARS + [c for c in COORD_VARS
                                          if c != "ocean_time"])
               if v not in ds.variables]
    if missing:
        log(logf, f"[static] missing (skipped): {missing}")

    try:
        needed_dims = set()
        for v in wanted:
            src_da = ds[v]
            if v in S_RHO_SUBSET_VARS:
                src_da = src_da.isel(s_rho=S_LEVELS)
            needed_dims.update(src_da.dims)

        with nc.Dataset(tmp_path, mode="w", format="NETCDF4") as dst:
            _copy_dims(ds, dst, needed_dims, s_rho_len=len(S_LEVELS))
            for v in wanted:
                subset = S_LEVELS if v in S_RHO_SUBSET_VARS else None
                _write_var_streamed(ds, dst, v, time_block,
                                    s_rho_subset=subset)
            dst.setncattr("source_url", url)
            dst.setncattr("staged_utc",
                          datetime.now(timezone.utc).isoformat())
            dst.setncattr("staged_by", "download_ccs.py (static file)")
            dst.setncattr("s_levels_from_original_42", str(S_LEVELS))
    finally:
        try:
            ds.close()
        except Exception:
            pass

    os.replace(tmp_path, out_path)
    log(logf, f"[static] wrote {out_path} "
              f"({out_path.stat().st_size / 1e6:.1f} MB, {len(wanted)} vars)")


# ============================================================
# Fetch + retry wrapper (per-file loop)
# ============================================================
def is_staged_ok(path: Path, min_ok_size_bytes: int = 20 * 1024 * 1024) -> bool:
    """Resume detection: file exists, non-trivial size, opens cleanly,
    has ocean_time > 3. If any check fails, the file gets re-fetched."""
    if not path.exists():
        return False
    if path.stat().st_size < min_ok_size_bytes:
        return False
    try:
        with nc.Dataset(str(path), "r") as ds:
            if "ocean_time" not in ds.dimensions:
                return False
            if len(ds.dimensions["ocean_time"]) < 3:
                return False
            # Sanity: at least one dynamic var present.
            if not any(v in ds.variables for v in DYNAMIC_VARS):
                return False
        return True
    except Exception:
        return False


def fetch_with_retries(
    record: dict, out_path: Path, time_block: int, max_retries: int,
    rc: dict, logf: Path,
) -> tuple:
    """Try to fetch a single state file with backoff. Returns
    (success: bool, error_str_or_none: str|None, was_throttle: bool)."""
    key = record["filename"]
    attempt = 0
    was_throttle_last = False
    while True:
        attempt += 1
        record["attempts"] += 1
        try:
            fetch_state_file(record["url"], out_path, time_block)
            return True, None, False
        except Exception as e:
            err_type = type(e).__name__
            err_msg = str(e).splitlines()[0][:200]
            throttle = is_throttle(e)
            was_throttle_last = throttle
            rc[key] = rc.get(key, 0) + 1
            if rc[key] >= STICKY and rc[key] % STICKY == 0:
                log(logf, f">>> STICKY WARNING file {key} "
                          f"cumulative_retries={rc[key]}")

            if throttle:
                # Throttle: log + backoff + retry until max_retries.
                back = min(BACKOFF_BASE * (2 ** min(attempt - 1, 5)),
                           BACKOFF_CAP)
                log(logf, f"RETRY {key} attempt {attempt} class=throttle "
                          f"cum={rc[key]} err={err_type}: {err_msg} "
                          f"backoff={back}s")
                if attempt >= max_retries:
                    return False, f"{err_type}: {err_msg}", True
                time.sleep(back)
            else:
                back = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_CAP)
                log(logf, f"RETRY {key} attempt {attempt} class=other "
                          f"cum={rc[key]} err={err_type}: {err_msg} "
                          f"backoff={back}s")
                if attempt >= max_retries:
                    return False, f"{err_type}: {err_msg}", False
                time.sleep(back)


# ============================================================
# Main loop
# ============================================================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="CCSRA31 production downloader (resumable, robust).",
    )
    ap.add_argument("--out", type=str, default=None,
                    help="staging directory (default: the CCS config "
                         "staging_dir resolved against "
                         "$CLOSEDLOOP_DATA_ROOT)")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help=f"pacing seconds between files "
                         f"(default {DEFAULT_SLEEP})")
    ap.add_argument("--time-block", type=int, default=DEFAULT_TIME_BLOCK,
                    help=f"timesteps per DAP fetch "
                         f"(default {DEFAULT_TIME_BLOCK})")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                    help=f"per-file retry cap (default {DEFAULT_RETRIES})")
    ap.add_argument("--limit", type=int, default=None,
                    help="bound run to first N pending files (test mode)")
    ap.add_argument("--collections", type=str, default="both",
                    choices=["core", "ext", "both"],
                    help="which collection(s) to enumerate + download")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the end-of-run failed-list sweep pass")
    return ap.parse_args()


def _select_collections(arg: str) -> list:
    if arg == "core":
        return ["core"]
    if arg == "ext":
        return ["ext"]
    return ["core", "ext"]


def _print_progress(logf: Path, idx: int, n_total: int, n_done: int,
                    n_failed: int, t_sess_start: float,
                    session_bytes: float, breakdown: str = "") -> None:
    """Aggregate progress line every 20 files."""
    elapsed = time.time() - t_sess_start
    n_processed = n_done + n_failed
    if n_processed == 0:
        rate = 0.0
        eta_h = float("inf")
    else:
        rate = n_processed / elapsed
        remaining = n_total - n_processed
        eta_h = (remaining / rate) / 3600.0 if rate > 0 else float("inf")
    gb = session_bytes / 1e9
    rss = peak_rss_mb()
    rss_str = f"peak_rss={rss:.0f}MB" if rss is not None else "peak_rss=?"
    log(logf, f"[progress] {idx}/{n_total} handled  "
              f"done={n_done} failed={n_failed}  "
              f"rate={3600 * rate:.1f}/h  ETA={eta_h:.1f}h  "
              f"session_GB={gb:.2f}  {rss_str}  {breakdown}")


def _get_pending_and_static_source(manifest: dict, staging_dir: Path,
                                   logf: Path,
                                   status_filter=("pending",)) -> tuple:
    """Walk manifest for files in `status_filter`; also identify the
    static-source URL (earliest core file, whether pending or done).

    Return (pending_records_list, static_source_url_or_none)."""
    static_source = None
    for r in sorted(manifest["files"], key=lambda x: x["date_start"]):
        if r["collection"] == "core":
            static_source = r["url"]
            break
    if static_source is None:
        # Ext-only run — use earliest ext file.
        for r in sorted(manifest["files"], key=lambda x: x["date_start"]):
            static_source = r["url"]
            break

    pending = [r for r in manifest["files"] if r["status"] in status_filter]

    # Resume: for pending records whose staged file already looks OK
    # on disk, mark done immediately (recovers from a crash between the
    # write and the manifest persist).
    n_recovered = 0
    for r in pending[:]:
        out_path = staging_dir / r["filename"]
        if is_staged_ok(out_path):
            r["status"] = "done"
            r["size_bytes"] = out_path.stat().st_size
            r["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
            pending.remove(r)
            n_recovered += 1
    if n_recovered:
        log(logf, f"[resume] recovered {n_recovered} pending records "
                  f"whose staged files exist and look valid")

    return pending, static_source


def _handle_one_file(
    record: dict, staging_dir: Path, time_block: int, retries: int,
    rc: dict, logf: Path,
) -> tuple:
    """Dispatch fetch_with_retries for one record; return
    (success: bool, throttle_flag: bool, elapsed_s: float,
    size_bytes: int|None)."""
    key = record["filename"]
    out_path = staging_dir / key
    t0 = time.time()
    success, err, throttle = fetch_with_retries(
        record, out_path, time_block, retries, rc, logf,
    )
    elapsed = time.time() - t0
    record["elapsed_s"] = elapsed
    record["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    if success:
        record["status"] = "done"
        record["error"] = None
        record["size_bytes"] = out_path.stat().st_size
        rc[key] = 0
        return True, False, elapsed, record["size_bytes"]
    else:
        record["status"] = "failed"
        record["error"] = err
        return False, throttle, elapsed, None


def _run_pass(
    pass_name: str, pending: list, manifest: dict, staging_dir: Path,
    mpath: Path, logf: Path, time_block: int, retries: int, sleep_s: float,
    limit: Optional[int], global_success_start: bool,
) -> tuple:
    """Loop the main download; returns
    (n_new_done, n_new_failed, aborted: bool)."""
    n_new_done = 0
    n_new_failed = 0
    consec_throttle = 0
    global_success = global_success_start
    rc: dict = {}
    t_sess = time.time()
    session_bytes = 0.0

    if limit is not None:
        pending = pending[:limit]

    n_total = len(pending)
    log(logf, f"[{pass_name}] starting: {n_total} files to fetch")

    for idx, record in enumerate(pending, start=1):
        success, throttle, elapsed, size = _handle_one_file(
            record, staging_dir, time_block, retries, {}, logf,
        )
        persist_manifest(mpath, manifest)

        if success:
            n_new_done += 1
            global_success = True
            consec_throttle = 0
            session_bytes += size or 0
            log(logf, f"[{pass_name}] OK   {record['filename']}  "
                      f"{elapsed:6.1f}s  {(size or 0) / 1e6:7.1f}MB  "
                      f"[{idx}/{n_total}]")
        else:
            n_new_failed += 1
            if throttle:
                consec_throttle += 1
                if (not global_success) and consec_throttle >= BREAKER:
                    log(logf, f"!!! ESCALATE: {consec_throttle} consecutive "
                              f"throttle failures with zero successes ever. "
                              f"The server is likely down or has hardened "
                              f"the rate limit past our workaround. Aborting "
                              f"— investigate before resuming.")
                    return n_new_done, n_new_failed, True
            log(logf, f"[{pass_name}] FAIL {record['filename']}  "
                      f"{elapsed:6.1f}s  err={record['error']}  "
                      f"[{idx}/{n_total}]")

        if idx % 20 == 0:
            _print_progress(logf, idx, n_total, n_new_done, n_new_failed,
                            t_sess, session_bytes)

        if idx < n_total:
            time.sleep(sleep_s)

    _print_progress(logf, n_total, n_total, n_new_done, n_new_failed,
                    t_sess, session_bytes,
                    breakdown=f"[{pass_name} DONE]")
    return n_new_done, n_new_failed, False


def main() -> int:
    args = parse_args()
    staging_dir = Path(args.out) if args.out else Path(resolve(CFG.raw.staging_dir))
    staging_dir.mkdir(parents=True, exist_ok=True)
    logf = staging_dir / LOG_FILENAME
    mpath = staging_dir / MANIFEST_FILENAME
    static_path = staging_dir / STATIC_FILENAME

    log(logf, "=" * 60)
    log(logf, "=== CCS downloader START")
    log(logf, "=" * 60)
    log(logf, f"staging_dir  = {staging_dir}")
    log(logf, f"manifest     = {mpath}")
    log(logf, f"static_file  = {static_path}")
    log(logf, f"time_block   = {args.time_block} timesteps per DAP fetch")
    log(logf, f"sleep        = {args.sleep}s pacing between files")
    log(logf, f"retries      = {args.retries} per file")
    log(logf, f"limit        = {args.limit if args.limit else 'none'}")
    log(logf, f"collections  = {args.collections}")
    log(logf, f"s_levels     = {S_LEVELS} ({len(S_LEVELS)} levels)")
    log(logf, f"dynamic_vars = {DYNAMIC_VARS}")

    check_env(logf)

    collections = _select_collections(args.collections)
    try:
        manifest = load_or_init_manifest(mpath, collections, logf)
    except Exception as e:
        log(logf, f"!!! ABORT: could not load/init manifest: "
                  f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 3

    # -- static-vars file (fetched ONCE)
    pending, static_source_url = _get_pending_and_static_source(
        manifest, staging_dir, logf,
    )
    persist_manifest(mpath, manifest)

    if not is_staged_ok(static_path, min_ok_size_bytes=1 * 1024 * 1024):
        if static_source_url is None:
            log(logf, "!!! ABORT: no source URL available for static file "
                      "(empty manifest?)")
            return 3
        try:
            fetch_static_file(static_source_url, static_path,
                              args.time_block, logf)
        except Exception as e:
            log(logf, f"!!! ABORT: static-file fetch failed: "
                      f"{type(e).__name__}: {e}")
            log(logf, "     (static vars are one-time and required — "
                      "will not proceed without them)")
            traceback.print_exc()
            return 3
    else:
        log(logf, f"[static] {static_path} already staged "
                  f"({static_path.stat().st_size / 1e6:.1f} MB) — skipping")

    # -- main pass
    global_success_start = any(f["status"] == "done"
                               for f in manifest["files"])
    n_done, n_failed, aborted = _run_pass(
        "main", pending, manifest, staging_dir, mpath, logf,
        args.time_block, args.retries, args.sleep, args.limit,
        global_success_start,
    )
    if aborted:
        return 2

    # -- sweep pass over anything still failed
    if not args.no_sweep and not args.limit:
        failed_records, _ = _get_pending_and_static_source(
            manifest, staging_dir, logf, status_filter=("failed",),
        )
        if failed_records:
            log(logf, f"[sweep] retrying {len(failed_records)} failed files")
            for r in failed_records:
                r["status"] = "pending"
                r["attempts"] = 0
            persist_manifest(mpath, manifest)
            sweep_done, sweep_failed, sweep_aborted = _run_pass(
                "sweep", failed_records, manifest, staging_dir, mpath, logf,
                args.time_block, args.retries, args.sleep, None,
                global_success_start=True,     # main pass already succeeded
            )
            if sweep_aborted:
                return 2
            n_done += sweep_done
            n_failed = sweep_failed

    # -- final summary
    n_total_done = sum(1 for f in manifest["files"] if f["status"] == "done")
    n_total_failed = sum(1 for f in manifest["files"] if f["status"] == "failed")
    n_total_pending = sum(1 for f in manifest["files"] if f["status"] == "pending")
    log(logf, "=" * 60)
    log(logf, f"=== CCS downloader END")
    log(logf, f"    total done:    {n_total_done}")
    log(logf, f"    total failed:  {n_total_failed}")
    log(logf, f"    total pending: {n_total_pending}")
    log(logf, f"    (manifest: {mpath})")
    log(logf, "=" * 60)

    return 0 if n_total_failed == 0 and n_total_pending == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
