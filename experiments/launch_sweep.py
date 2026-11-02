#!/usr/bin/env python
"""Enumerate the campaign runs and emit a job list (no submission).

Produces one CLI line per chain, suitable for a SLURM array wrapper
(slurm/closed_loop_array.sbatch) or any batch system:

  main campaign   2 regions x 2 targets x fleets {20,50,200} x 5 policies
                  x 10 sealed ICs x 3 seeds                     = 1,800 chains
  fleet sweep     per configs/experiments/fleet_sweep.yaml
                  (MAB salinity, fleets {10,35,100})            =   450 chains
  replicates      the rerun-tolerance grid: A-optimal, seed 0,
                  every (region, target, fleet, IC) cell        =   120 chains
  free-running    2 regions x 2 targets x 10 ICs (fleet-invariant) = 40 chains

Guidance scales come from guidance_scales.yaml via run_closed_loop.py's
default lookup, so the job lines carry no scale unless --explicit-scales.

Usage: python experiments/launch_sweep.py [--set main|fleet|replicates|free|all]
       [--out jobs.tsv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
POLICIES = ["static", "random", "lawnmower", "a_optimal", "d_optimal"]
SEEDS = [0, 1, 2]
MAIN_FLEETS = [20, 50, 200]


def sealed(region):
    return json.load(open(REPO / "configs/experiments/ics" /
                          f"test_{region}.json"))["canonical_order"]


def line(region, ic, target, nfleet, strategy, rseed, extra=""):
    base = (f"--region {region} --ic_datetime {ic} --target {target} "
            f"--nfleet {nfleet} --strategy {strategy} --rseed {rseed} "
            f"--ncyc 40 --start_lead 0")
    return (base + (" " + extra if extra else "")).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="all",
                    choices=["main", "fleet", "replicates", "free", "all"])
    ap.add_argument("--out", default=None, help="write the list here (default stdout)")
    args = ap.parse_args()

    jobs = []
    if args.set in ("main", "all"):
        for region in ("mab", "ccs"):
            for target in ("temp", "salt"):
                for n in MAIN_FLEETS:
                    for strategy in POLICIES:
                        for ic in sealed(region):
                            for s in SEEDS:
                                jobs.append(line(region, ic, target, n, strategy, s))
    if args.set in ("fleet", "all"):
        fs = yaml.safe_load(open(REPO / "configs/experiments/fleet_sweep.yaml"))
        for n in fs["fleets"]:
            for strategy in fs["strategies"]:
                for ic in sealed(fs["region"]):
                    for s in fs["seeds"]:
                        jobs.append(line(fs["region"], ic, fs["target"], n,
                                         strategy, s))
    if args.set in ("replicates", "all"):
        # Rerun-tolerance grid: an independent repeat of every A-optimal seed-0
        # chain; the tolerance floor is max |score(repeat) - score(registered)|
        # per (region, target, fleet).
        for region in ("mab", "ccs"):
            for target in ("temp", "salt"):
                for n in MAIN_FLEETS:
                    for ic in sealed(region):
                        jobs.append(line(region, ic, target, n, "a_optimal", 0,
                                         extra="--out-dir replicates"))
    if args.set in ("free", "all"):
        for region in ("mab", "ccs"):
            for target in ("temp", "salt"):
                for ic in sealed(region):
                    jobs.append(line(region, ic, target, 50, "free_running", 0,
                                     extra="--dump-latents"))

    text = "\n".join(jobs) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {len(jobs)} job lines to {args.out}")
    else:
        print(text, end="")
        print(f"# {len(jobs)} job lines", flush=True)


if __name__ == "__main__":
    main()
