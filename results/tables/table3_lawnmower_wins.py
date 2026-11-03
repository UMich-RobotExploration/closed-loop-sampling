#!/usr/bin/env python
"""Table III: Lawnmower wins over both uncertainty-driven policies on SST.

A win is an (IC, seed) chain, out of 30 per (region, fleet), whose final-eight
window RMSE under Lawnmower is below BOTH A-optimal and D-optimal on the same
seed. Each printed cell gives wins / wins whose margin over the better of the
two exceeds the pooled run-to-run floor of the slice / distinct ICs with at
least one win.

Reads results/scores/ only; emits table3_lawnmower_wins.csv and .tex.
"""
from __future__ import annotations

from _lib import REGION_LABEL, fnum, parse_args, write_csv, write_tex

from scoring import metrics as M
from scoring import tolerance as F


def main():
    scores, out = parse_args("Table III")
    runs = M.load_runs(scores)
    floors = F.load_floors(scores)
    variable = "sst"

    rows_csv, rows_tex = [], []
    for region in M.REGIONS:
        cells = []
        for fleet in M.FLEETS_MAIN:
            floor = F.floor_for(floors, region, variable, fleet, "last8")
            wins = above = 0
            win_ics = set()
            win_rows = []
            for ic in M.ics_for(scores, region):
                for seed in M.SEEDS:
                    s_lawn = M.run_score(runs, region, variable, fleet, ic,
                                         "lawnmower", seed)
                    s_a = M.run_score(runs, region, variable, fleet, ic,
                                      "a_optimal", seed)
                    s_d = M.run_score(runs, region, variable, fleet, ic,
                                      "d_optimal", seed)
                    margin = min(s_a, s_d) - s_lawn
                    if margin > 0:
                        wins += 1
                        win_ics.add(ic)
                        if margin > floor:
                            above += 1
                        win_rows.append((ic, seed, margin))
            rows_csv.append([region, fleet, wins, above, len(win_ics),
                             fnum(floor),
                             ";".join(f"{ic}/s{s}:{m:.4f}" for ic, s, m in win_rows)])
            cells.append(f"{wins} / {above} / {len(win_ics)}")
        rows_tex.append(f"{REGION_LABEL[region]} & " + " & ".join(cells) + " \\\\")

    tex = ["\\begin{tabular}{lccc}", "\\toprule",
           "& $n=20$ & $n=50$ & $n=200$ \\\\", "\\midrule"] + rows_tex + \
          ["\\bottomrule", "\\end{tabular}"]

    write_csv(out / "table3_lawnmower_wins.csv",
              ["region", "fleet", "wins", "wins_above_floor", "ics_with_win",
               "floor_last8", "win_chains"], rows_csv)
    write_tex(out / "table3_lawnmower_wins.tex", tex)
    print("TABLE3 OK")


if __name__ == "__main__":
    main()
