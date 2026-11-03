#!/usr/bin/env python
"""Table II: per-IC spread-error correlation and sampling benefit.

For every sealed IC: r, the spread-error correlation of the free-running
ensemble at 240 h (references/spread_error_r.csv), and Delta_n, the final-eight
window RMSE of Random minus A-optimal at fleet size n (positive favours
A-optimal; per-IC mean over the three paired seeds). A dagger marks
|Delta_n| above the pooled run-to-run floor of the slice.

Reads results/scores/ only; emits table2_per_ic.csv and table2_per_ic.tex.
"""
from __future__ import annotations

import csv

from _lib import (REGION_LABEL, fnum, paper_delta, paper_r, parse_args,
                  write_csv, write_tex)

from scoring import metrics as M
from scoring import tolerance as F


def main():
    scores, out = parse_args("Table II")
    runs = M.load_runs(scores)
    floors = F.load_floors(scores)
    r240 = {}
    with open(scores / "references/spread_error_r.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            r240[(r["region"], r["variable"], r["ic"])] = float(r["r_240h"])

    rows_csv, rows_tex = [], []
    for region in M.REGIONS:
        rows_tex.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{REGION_LABEL[region]}}}}}\\\\")
        for ic in M.ics_for(scores, region):
            cells = []
            for variable in M.VARIABLES:
                rv = r240[(region, variable, ic)]
                cells.append(paper_r(rv))
                for fleet in M.FLEETS_MAIN:
                    # paper sign convention: Random minus A-optimal
                    d = M.per_ic_delta(runs, region, variable, fleet, ic,
                                       "random", "a_optimal")
                    floor = F.floor_for(floors, region, variable, fleet, "last8")
                    dag = abs(d) > floor
                    rows_csv.append([region, ic, variable, fnum(rv), fleet,
                                     fnum(d), int(dag)])
                    s = paper_delta(d)
                    cells.append(s[:-1] + "^\\dagger$" if dag else s)
            rows_tex.append(f"{ic[:10]} & " + " & ".join(cells) + " \\\\")
        if region != M.REGIONS[-1]:
            rows_tex.append("\\midrule")

    tex = ["\\begin{tabular}{l r rrr r rrr}", "\\toprule",
           " & \\multicolumn{4}{c}{SST} & \\multicolumn{4}{c}{SSS} \\\\",
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}",
           "Date & $r$ & $\\Delta_{20}$ & $\\Delta_{50}$ & $\\Delta_{200}$",
           "     & $r$ & $\\Delta_{20}$ & $\\Delta_{50}$ & $\\Delta_{200}$ \\\\",
           "\\midrule"] + rows_tex + ["\\bottomrule", "\\end{tabular}"]

    write_csv(out / "table2_per_ic.csv",
              ["region", "ic", "variable", "r_240h", "fleet",
               "delta_random_minus_a_optimal", "above_floor"], rows_csv)
    write_tex(out / "table2_per_ic.tex", tex)
    print("TABLE2 OK")


if __name__ == "__main__":
    main()
