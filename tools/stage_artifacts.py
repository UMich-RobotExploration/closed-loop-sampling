#!/usr/bin/env python
"""Stage the artifacts dataset (figure fields, render dumps, member caches,
free-running latents, raw run records) from the production trees into a staging
directory, re-expressed as plain npz with release naming.

This is the auditable pipeline behind the released dataset repository's
non-bundle content (the truth and latent bundles are written by
datasets/export_{truth,latent}_bundle.py). Reads the production trees
READ-ONLY; writes only under --staging.

Layout written (identical to the published dataset repository, so
$CLOSEDLOOP_ARTIFACTS can point at either):
  figure_fields/fig1_method_mab_ic20220409T12.npz     Fig. 1 render context
  figure_fields/fig2_members_<region>_<ic>.npz        Fig. 2 decoded fields
  figure_fields/fig6_diagnostic_mab_20220802.npz      Fig. 6 diagnostic fields
  figure_fields/video_members_<slice>.npz             loop-video member decodes
  figure_fields/README.md                             provenance note
  render_dumps/<slice>/<policy>.npz                   per-cycle field/position dumps
  free_running_latents/<region>_<ic>_latents.npz      per-cycle member latents
  records/raw_run_records.tar.zst                     the 2,318 run records

Naming: regions mab/ccs, policies static/random/lawnmower/d_optimal/a_optimal/
free_running; slice = <region>_<variable>_<ictag>_n<fleet>.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REGION_OUT = {"doppio": "mab", "ccs": "ccs"}
POLICY_OUT = {"static": "static", "random": "random", "lawn": "lawnmower",
              "dopt": "d_optimal", "trace": "a_optimal",
              "nosample": "free_running"}
# the four dumped slices (region key, variable, production slice stem)
SLICES = [("doppio", "sst", "ic20220409T12_doppio_L0_n50"),
          ("doppio", "sss", "ic20220409T12_doppio_L0_salt_n50"),
          ("ccs", "sst", "ic20160727T12_ccs_L0_n50"),
          ("ccs", "sss", "ic20160727T12_ccs_L0_salt_n50")]
MEMBER_CACHE_SUB = {"doppio": "verify", "ccs": "ccs"}
RENDER_KEYS = ("spread", "prior_mean", "post_mean", "truth", "fleet", "cand_rc",
               "chosen_rc", "cycle", "lead_h", "floor", "prior_rmse", "post_rmse",
               "mask", "nfleet", "scale_eff", "dx_col_m", "dy_row_m", "v_auv",
               "reach_window_s")


def release_slice(region_prod: str, variable: str, stem: str) -> str:
    ictag = stem.split("_")[0]
    n = stem.split("_n")[-1]
    return f"{REGION_OUT[region_prod]}_{variable}_{ictag}_n{n}"


def stage_render_dumps(res: Path, out: Path):
    for region_prod, variable, stem in SLICES:
        dst = out / "render_dumps" / release_slice(region_prod, variable, stem)
        dst.mkdir(parents=True, exist_ok=True)
        for arm, pol in POLICY_OUT.items():
            src = res / f"q3_render_{stem}_{arm}.npz"
            z = np.load(src, allow_pickle=True)
            data = {k: z[k] for k in RENDER_KEYS}
            data["strategy"] = np.array(pol)
            data["region"] = np.array(REGION_OUT[region_prod])
            data["variable"] = np.array(variable)
            np.savez_compressed(dst / f"{pol}.npz", **data)
        print(f"  render_dumps/{dst.name}: 6 policies", flush=True)


def stage_fig1(src_root: Path, out: Path):
    res = src_root / "phase3_sandbox_ipp/q3_field_uncertainty/results"
    mw = src_root / "phase3_sandbox_ipp/q3_field_uncertainty/_mlxor_workshop"
    c4 = src_root / "phase3_sandbox_ipp/icra_desk/c4"
    CYC = 39
    tr = np.load(res / "q3_render_ic20220409T12_doppio_L0_n50_trace.npz",
                 allow_pickle=True)
    mb = np.load(c4 / "sysfig_members_ic20220409T12_doppio_cyc39.npz",
                 allow_pickle=True)
    geo = np.load(mw / "verify/members_fields_doppio.npz")
    assert np.array_equal(geo["mask"], tr["mask"]) and np.array_equal(
        mb["mask"], tr["mask"])
    ff = out / "figure_fields"
    ff.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ff / "fig1_method_mab_ic20220409T12.npz",
        region="mab", ic="2022-04-09T12", cycle=CYC, lead_h=240,
        mask=tr["mask"], lon=geo["lon"], lat=geo["lat"],
        dx_col_m=float(tr["dx_col_m"]), dy_row_m=float(tr["dy_row_m"]),
        v_auv_ms=float(tr["v_auv"]), reach_window_s=float(tr["reach_window_s"]),
        spread=np.asarray(tr["spread"], np.float32)[CYC],
        fleet=np.asarray(tr["fleet"][CYC], np.float32),
        chosen_rc=np.asarray(tr["chosen_rc"][CYC], np.int32),
        cand_rc=np.asarray(tr["cand_rc"][CYC], np.int32),
        members=np.asarray(mb["members"], np.float32),
        figure_member_indices=np.asarray(mb["figure_member_indices"], np.int64),
        chain="a_optimal (A-optimal) chain, observation seed 0; members from "
              "the free-running ensemble at the same cycle")
    print("  figure_fields/fig1_method_mab_ic20220409T12.npz", flush=True)


def stage_fig2(src_root: Path, out: Path):
    mw = src_root / "phase3_sandbox_ipp/q3_field_uncertainty/_mlxor_workshop"
    ff = out / "figure_fields"
    ff.mkdir(parents=True, exist_ok=True)
    for region_prod, ic in (("doppio", "2022-04-09T12"), ("ccs", "2016-07-27T12")):
        npz = np.load(mw / MEMBER_CACHE_SUB[region_prod]
                      / f"members_fields_{region_prod}.npz")
        mask = npz["mask"]
        data = dict(region=REGION_OUT[region_prod], ic=ic, mask=mask,
                    lon=npz["lon"], lat=npz["lat"],
                    dx_m=float(npz["dx_cm"]), dy_m=float(npz["dy_cm"]))
        for vk, VN in (("sst", "SST"), ("sss", "SSS")):
            for k in ("truth", "m1", "m2", "mean", "spread", "err"):
                data[f"{vk}_{k}"] = np.asarray(npz[f"{ic}|{VN}|{k}"], np.float32)
            sw = data[f"{vk}_spread"][mask]
            ew = data[f"{vk}_err"][mask]
            good = np.isfinite(sw) & np.isfinite(ew)
            data[f"expected_r_{vk}"] = float(np.corrcoef(sw[good], ew[good])[0, 1])
        ictag = "ic" + ic.replace("-", "")[:8] + "T" + ic[11:13]
        name = f"fig2_members_{REGION_OUT[region_prod]}_{ictag}.npz"
        np.savez_compressed(ff / name, **data)
        print(f"  figure_fields/{name} "
              f"(r_sst {data['expected_r_sst']:+.3f}, "
              f"r_sss {data['expected_r_sss']:+.3f})", flush=True)


def stage_fig6(src_root: Path, out: Path):
    c4 = src_root / "phase3_sandbox_ipp/icra_desk/c4"
    ff = out / "figure_fields"
    ff.mkdir(parents=True, exist_ok=True)
    z = np.load(c4 / "diag_fields_20220802.npz", allow_pickle=True)
    np.savez_compressed(ff / "fig6_diagnostic_mab_20220802.npz",
                        region="mab",
                        **{k: z[k] for k in ("err", "spread", "eta", "mask",
                                             "lon", "lat", "dx_m", "dy_m", "VE",
                                             "VS", "VZ", "ic", "tag", "cycle",
                                             "lead_h")})
    print("  figure_fields/fig6_diagnostic_mab_20220802.npz", flush=True)


def stage_video_members(src_root: Path, loop_video_root: Path, out: Path):
    mw = src_root / "phase3_sandbox_ipp/q3_field_uncertainty/_mlxor_workshop"
    ff = out / "figure_fields"
    ff.mkdir(parents=True, exist_ok=True)
    for region_prod, variable, stem in SLICES:
        base = stem.replace("_salt", "")          # cache names carry _salt last
        tag = "_salt" if variable == "sss" else ""
        src = loop_video_root / f"loop_members_{base}{tag}.npz"
        geo = np.load(mw / MEMBER_CACHE_SUB[region_prod]
                      / f"members_fields_{region_prod}.npz")
        mem = np.load(src)["members"]
        name = f"video_members_{release_slice(region_prod, variable, stem)}.npz"
        np.savez_compressed(ff / name, members=np.asarray(mem, np.float32),
                            region=REGION_OUT[region_prod], variable=variable,
                            lon=geo["lon"], lat=geo["lat"],
                            dx_m=float(geo["dx_cm"]), dy_m=float(geo["dy_cm"]),
                            source="three free-running members decoded per cycle")
        print(f"  figure_fields/{name} {mem.shape}", flush=True)


def stage_free_running_latents(e2_latents: Path, out: Path):
    dst = out / "free_running_latents"
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(e2_latents.glob("q3_seed_ic*_latents.npz"))
    assert len(files) == 20, f"{len(files)} e2 latent dumps (expected 20)"
    for f in files:
        # q3_seed_ic<tag>_<region>_L0_n50_nosample_s0_latents.npz
        parts = f.name.split("_")
        ictag, region_prod = parts[2], parts[3]
        z = np.load(f, allow_pickle=True)
        name = f"{REGION_OUT[region_prod]}_{ictag}_free_running_latents.npz"
        np.savez_compressed(dst / name,
                            prior_latents=z["prior_latents"],
                            region=REGION_OUT[region_prod],
                            ic=f"{ictag[2:6]}-{ictag[6:8]}-{ictag[8:10]}T{ictag[11:13]}",
                            note="per-cycle prior member latents (fp16) of the "
                                 "free-running chain, loop convention, N=25")
        print(f"  free_running_latents/{name}", flush=True)


def stage_records(res: Path, out: Path):
    dst = out / "records"
    dst.mkdir(parents=True, exist_ok=True)
    tar = dst / "raw_run_records.tar.zst"
    files = sorted(p.name for p in res.glob("q3_seed_ic*.json"))
    assert len(files) == 2318, len(files)
    lst = dst / "_filelist.txt"
    lst.write_text("\n".join(files) + "\n")
    # pipe through zstd: the system tar predates --zstd
    with open(tar, "wb") as out_fh:
        p_tar = subprocess.Popen(["tar", "-cf", "-", "-C", str(res), "-T", str(lst)],
                                 stdout=subprocess.PIPE)
        p_zst = subprocess.Popen(["zstd", "-q", "-T0"], stdin=p_tar.stdout,
                                 stdout=out_fh)
        p_tar.stdout.close()
        assert p_zst.wait() == 0 and p_tar.wait() == 0, "tar | zstd failed"
    lst.unlink()
    print(f"  records/raw_run_records.tar.zst "
          f"({tar.stat().st_size / 1e6:.0f} MB, {len(files)} records)", flush=True)


def write_readme(out: Path):
    (out / "figure_fields" / "README.md").write_text(
        "# Figure fields\n\n"
        "Decoded-field extracts behind the paper figures, re-expressed as plain\n"
        "npz from the production caches (read-only; see tools/stage_artifacts.py\n"
        "in the code repository):\n\n"
        "- `fig1_method_mab_ic20220409T12.npz`: cycle-39 render of the A-optimal\n"
        "  chain (spread, fleet, reachable set, selections) plus three decoded\n"
        "  free-running members and the grid geometry.\n"
        "- `fig2_members_<region>_<ic>.npz`: truth, two members, ensemble mean,\n"
        "  spread and error of the free-running ensemble at cycle 39 (t+240 h),\n"
        "  both variables, plus the spread-error correlation measured at staging.\n"
        "- `fig6_diagnostic_mab_20220802.npz`: absolute error, spread and\n"
        "  normalized displacement eta with their display caps.\n"
        "- `video_members_<slice>.npz`: three free-running members decoded per\n"
        "  cycle for the loop video.\n\n"
        "The per-cycle positions drawn in Fig. 5 ship as CSVs INSIDE the code\n"
        "repository (results/scores/locations/), extracted from the same render\n"
        "dumps released here under `render_dumps/mab_sst_ic20220409T12_n50/`.\n")
    print("  figure_fields/README.md", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--loop-video-root", required=True,
                    help="directory holding the loop_members_<slice>.npz caches")
    ap.add_argument("--e2-latents", required=True,
                    help="directory holding the 20 free-running latent dumps")
    ap.add_argument("--staging", required=True,
                    help="output root (the artifacts dataset layout)")
    ap.add_argument("--only", default="all",
                    choices=["all", "figures", "video", "renders", "latents",
                             "records"])
    args = ap.parse_args()
    src = Path(args.source_root)
    res = src / "phase3_sandbox_ipp/q3_field_uncertainty/results"
    out = Path(args.staging)
    out.mkdir(parents=True, exist_ok=True)

    if args.only in ("all", "figures"):
        stage_fig1(src, out)
        stage_fig2(src, out)
        stage_fig6(src, out)
        write_readme(out)
    if args.only in ("all", "video"):
        stage_video_members(src, Path(args.loop_video_root), out)
    if args.only in ("all", "renders"):
        stage_render_dumps(res, out)
    if args.only in ("all", "latents"):
        stage_free_running_latents(Path(args.e2_latents), out)
    if args.only in ("all", "records"):
        stage_records(res, out)
    print("STAGE_ARTIFACTS OK")


if __name__ == "__main__":
    sys.exit(main())
