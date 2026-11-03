#!/usr/bin/env python
"""Table I: campaign score per region, variable and fleet size, all policies.

Score = mean analysis-ensemble RMSE over the final eight cycles, averaged over
the 10 sealed ICs and 3 paired seeds. Bold marks the best policy per row from
the UNROUNDED scores. A dagger marks a policy not reliably separated from the
best: separation requires Wilcoxon p < 0.05 on the 10 per-IC paired deltas AND
a mean gap above the pooled run-to-run floor of the slice.

Reads results/scores/ only; emits table1_rmse.csv and table1_rmse.tex.
"""
from __future__ import annotations

from _lib import (POLICY_LABEL, REGION_LABEL, fnum, parse_args, write_csv,
                  write_tex)

from scoring import metrics as M
from scoring import stats_tests as T
from scoring import tolerance as F


def main():
    scores, out = parse_args("Table I")
    runs = M.load_runs(scores)
    floors = F.load_floors(scores)

    rows_csv, rows_tex = [], []
    for region in M.REGIONS:
        ics = M.ics_for(scores, region)
        for variable in M.VARIABLES:
            for fi, fleet in enumerate(M.FLEETS_MAIN):
                sc = {p: M.config_mean(runs, scores, region, variable, fleet, p)
                      for p in M.POLICIES}
                best = min(M.POLICIES, key=lambda p: sc[p])
                floor = F.floor_for(floors, region, variable, fleet, "last8")
                cells = []
                for p in M.POLICIES:
                    if p == best:
                        rows_csv.append([region, variable, fleet, p, fnum(sc[p]),
                                         f"{sc[p]:.3f}", 1, 0, "", "", fnum(floor)])
                        cells.append(f"\\textbf{{{sc[p]:.3f}}}")
                        continue
                    d = [M.per_ic_delta(runs, region, variable, fleet, ic, p, best)
                         for ic in ics]
                    r = T.separated(d, floor)
                    dagger = not r["separated"]
                    rows_csv.append([region, variable, fleet, p, fnum(sc[p]),
                                     f"{sc[p]:.3f}", 0, int(dagger),
                                     fnum(r["gap"]), fnum(r["wilcoxon_p"]),
                                     fnum(floor)])
                    cells.append(f"{sc[p]:.3f}" + ("$^\\dagger$" if dagger else ""))
                lead = (f"\\multirow{{3}}{{*}}{{\\shortstack[l]"
                        f"{{{REGION_LABEL[region]}\\\\{variable.upper()}}}}}"
                        if fi == 0 else "")
                rows_tex.append(f"{lead}\n & {fleet}  & " + " & ".join(cells) + " \\\\"
                                if fi == 0 else f" & {fleet}  & " + " & ".join(cells) + " \\\\")
            rows_tex.append("%MIDRULE")

    tex = ["\\begin{tabular}{ll ccccc}", "\\toprule",
           " & $n$ & " + " & ".join(f"\\emph{{{POLICY_LABEL[p]}}}" for p in M.POLICIES)
           + " \\\\", "\\midrule"]
    body = []
    for r in rows_tex[:-1]:
        body.append("\\midrule" if r == "%MIDRULE" else r)
    tex += body + ["\\bottomrule", "\\end{tabular}"]

    write_csv(out / "table1_rmse.csv",
              ["region", "variable", "fleet", "policy", "score", "score_3dp",
               "best", "dagger", "gap_vs_best", "wilcoxon_p", "floor_last8"],
              rows_csv)
    write_tex(out / "table1_rmse.tex", tex)
    print("TABLE1 OK")


if __name__ == "__main__":
    main()
