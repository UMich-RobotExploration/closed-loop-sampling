#!/usr/bin/env python
"""Run the free-running reference chains (no sampling, no update).

One chain per (region, target, sealed IC): the forecast rolls 40 cycles with
the loop convention and no observations — the free-running error curve that
anchors the skill fraction (0 = free-running, 1 = autoencoder floor). The
production campaign ran these at n_fleet 50 nominally; the chains are
fleet-invariant (no vehicles act). With --dump-latents each chain also saves
its per-cycle member latents (fp16), the inputs behind the spread-error
diagnostics and the ensemble figures.

Usage:
  python experiments/run_free_running.py --region mab --target temp [--dump-latents]
Enumerates the sealed ICs from configs/experiments/ics/test_<region>.json and
invokes run_closed_loop.py's machinery chain by chain.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", required=True, choices=["mab", "ccs"])
    ap.add_argument("--target", default="temp", choices=["temp", "salt"])
    ap.add_argument("--dump-latents", action="store_true")
    ap.add_argument("--truth", choices=["bundle", "store"], default="bundle")
    args = ap.parse_args()

    sealed = json.load(open(REPO / "configs/experiments/ics" /
                            f"test_{args.region}.json"))
    for ic in sealed["canonical_order"]:
        cmd = [sys.executable, str(REPO / "experiments/run_closed_loop.py"),
               "--region", args.region, "--ic_datetime", ic,
               "--target", args.target, "--nfleet", "50",
               "--strategy", "free_running", "--rseed", "0",
               "--truth", args.truth]
        if args.dump_latents:
            cmd.append("--dump-latents")
        print("::", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
