#!/usr/bin/env python
"""Golden verification: compare every printed number in the submitted paper's
Tables I-III against the values regenerated from results/scores/, within the
paper's printed rounding.

--archive points at the submitted source package (kept outside this
repository); --tables points at a directory holding the regenerated
table{1,2,3} CSVs (results/tables/expected by default). The comparison is
cell by cell; any mismatch is reported, never papered over.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REGION_OF = {"MAB": "mab", "CCS": "ccs"}
POLICIES = ("static", "random", "lawnmower", "d_optimal", "a_optimal")
FLEETS = (20, 50, 200)


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def tex_block(text: str, label: str) -> str:
    """The ACTIVE table body for a label: commented lines are dropped first,
    because the sources keep superseded drafts of the same tables commented out."""
    text = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    tag = f"\\label{{{label}}}"
    assert text.count(tag) == 1, f"{label}: {text.count(tag)} active occurrences"
    i = text.index(tag)
    return text[i:text.index("\\end{tabular}", i)]


def parse_table1(tex: str):
    """-> {(region, variable, fleet, policy): (printed, bold, dagger)}."""
    out, region, variable = {}, None, None
    for line in tex_block(tex, "tab:summary").splitlines():
        m = re.search(r"\\shortstack\[l\]\{(MAB|CCS)\\\\(SST|SSS)\}", line)
        if m:
            region, variable = REGION_OF[m.group(1)], m.group(2).lower()
        m = re.match(r"\s*(?:\\multirow.*)?&\s*(\d+)\s*&(.*)\\\\", line)
        if m and region:
            fleet = int(m.group(1))
            cells = [c.strip() for c in m.group(2).split("&")]
            assert len(cells) == 5, line
            for p, c in zip(POLICIES, cells):
                bold = "\\textbf" in c
                dag = "dagger" in c
                v = re.search(r"(\d+\.\d+)", c).group(1)
                out[(region, variable, fleet, p)] = (v, bold, dag)
    return out


def parse_table2(tex: str):
    """-> {(region, ic, variable): r}, {(region, ic, variable, fleet): (delta, dagger)}."""
    rs, ds, region = {}, {}, None
    for line in tex_block(tex, "tab:peric").splitlines():
        m = re.search(r"\\multicolumn\{9\}\{l\}\{\\emph\{(MAB|CCS)\}\}", line)
        if m:
            region = REGION_OF[m.group(1)]
        m = re.match(r"\s*(\d{4}-\d{2}-\d{2})\s*&(.*)\\\\", line)
        if m and region:
            ic = m.group(1) + "T12"
            cells = [c.strip() for c in m.group(2).split("&")]
            assert len(cells) == 8, line
            for vi, variable in enumerate(("sst", "sss")):
                rc = cells[4 * vi]
                rs[(region, ic, variable)] = re.search(r"([+-]\d+\.\d+)", rc).group(1)
                for fi, fleet in enumerate(FLEETS):
                    c = cells[4 * vi + 1 + fi]
                    v = re.search(r"(-?)\.(\d{3})", c.replace("$", ""))
                    ds[(region, ic, variable, fleet)] = (
                        f"{v.group(1)}.{v.group(2)}", "dagger" in c)
    return rs, ds


def parse_table3(tex: str):
    out = {}
    for line in tex_block(tex, "tab:lawnwins").splitlines():
        m = re.match(r"\s*(MAB|CCS)\s*&(.*)\\\\", line)
        if m:
            cells = [c.strip() for c in m.group(2).split("&")]
            for fleet, c in zip(FLEETS, cells):
                w, a, i = (int(x) for x in c.split("/"))
                out[(REGION_OF[m.group(1)], fleet)] = (w, a, i)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--archive", required=True,
                    help="submitted source package directory (read-only)")
    ap.add_argument("--tables", default=None,
                    help="regenerated tables dir (default: results/tables/expected)")
    args = ap.parse_args()
    arch = Path(args.archive)
    repo = Path(__file__).resolve().parents[1]
    tdir = Path(args.tables) if args.tables else repo / "results/tables/expected"

    tex4 = (arch / "sections/04_experimental_setup.tex").read_text()
    tex5 = (arch / "sections/05_results.tex").read_text()

    rows, bad = [], 0

    def check(table, cell, paper, regen, note=""):
        nonlocal bad
        ok = paper == regen
        bad += not ok
        rows.append((table, cell, paper, regen, "MATCH" if ok else "MISMATCH", note))

    # Table I: printed score (3dp), bold placement, dagger placement
    p1 = parse_table1(tex4)
    for r in read_csv(tdir / "table1_rmse.csv"):
        key = (r["region"], r["variable"], int(r["fleet"]), r["policy"])
        paper_v, paper_b, paper_d = p1.pop(key)
        cell = "/".join(map(str, key))
        check("I", cell + " score", paper_v, r["score_3dp"])
        check("I", cell + " bold", paper_b, r["best"] == "1")
        check("I", cell + " dagger", paper_d, r["dagger"] == "1")
    assert not p1, f"paper Table I cells not regenerated: {sorted(p1)}"

    # Table II: r (2dp signed), delta (3dp, paper sign = random - a_optimal), dagger
    pr, pd = parse_table2(tex5)
    seen_r = set()
    for r in read_csv(tdir / "table2_per_ic.csv"):
        key = (r["region"], r["ic"], r["variable"], int(r["fleet"]))
        paper_v, paper_dag = pd.pop(key)
        d = float(r["delta_random_minus_a_optimal"])
        mine = f"{abs(d):.3f}"[1:]
        mine = ("-" + mine) if d < 0 and mine != ".000" else mine
        cell = "/".join(map(str, key))
        check("II", cell + " delta", paper_v, mine)
        check("II", cell + " dagger", paper_dag, r["above_floor"] == "1")
        rkey = key[:3]
        if rkey not in seen_r:
            seen_r.add(rkey)
            rv = float(r["r_240h"])
            check("II", "/".join(rkey) + " r240",
                  pr.pop(rkey), f"{'+' if rv >= 0 else '-'}{abs(rv):.2f}")
    assert not pd and not pr, "paper Table II cells not regenerated"

    # Table III: wins / above-floor wins / ICs with a win
    p3 = parse_table3(tex5)
    for r in read_csv(tdir / "table3_lawnmower_wins.csv"):
        key = (r["region"], int(r["fleet"]))
        check("III", "/".join(map(str, key)),
              p3.pop(key),
              (int(r["wins"]), int(r["wins_above_floor"]), int(r["ics_with_win"])))
    assert not p3

    out = repo / "results/out/golden_table_check.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["table", "cell", "paper", "regenerated", "verdict", "note"])
        w.writerows(rows)
    n = len(rows)
    print(f"{n} comparisons, {n - bad} match, {bad} mismatch -> {out}")
    for t, c, p, g, v, _ in rows:
        if v == "MISMATCH":
            print(f"  MISMATCH Table {t} {c}: paper {p} vs regenerated {g}")
    print("GOLDEN_TABLES " + ("OK" if bad == 0 else f"MISMATCHES={bad}"))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
