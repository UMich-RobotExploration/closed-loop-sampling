"""Replay the recorded planner fixtures: feeding the frozen inputs (member
target fields at wet cells, candidate set, feasibility partition, R) to
select_locations must reproduce the recorded chosen cells exactly.

Fixtures are instrumented GPU chain runs
(experiments/run_closed_loop.py --dump-selection-fixtures K); the planner
itself is numpy-only, so the replay is deterministic on CPU. Fixture files
resolve from $CLOSEDLOOP_ARTIFACTS/selection_fixtures/ or
tests/fixtures/selection_replay/; the test skips when none exist (the
synthetic parity suite, tests/test_selection_parity.py, covers the algorithm
meanwhile).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


def _fixture_files():
    roots = [REPO / "tests/fixtures/selection_replay"]
    art = os.environ.get("CLOSEDLOOP_ARTIFACTS")
    if art:
        roots.append(Path(art) / "selection_fixtures")
    out = []
    for r in roots:
        if r.is_dir():
            out += sorted(r.glob("*_selection_fixtures.npz"))
    return out


FILES = _fixture_files()
if not FILES:
    pytest.skip("no recorded selection fixtures (generate with "
                "run_closed_loop.py --dump-selection-fixtures)",
                allow_module_level=True)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_replay_reproduces_chosen_cells(path):
    from closedloop_sampling.sampling.selection import select_locations
    from closedloop_sampling.sampling.uncertainty import anomalies

    z = np.load(path, allow_pickle=True)
    n_cycles = int(z["n_cycles"])
    assert n_cycles >= 1
    for c in range(40):
        key = f"c{c}_member_fields_wet"
        if key not in z.files:
            continue
        A = np.asarray(z[key], np.float32)
        cand_field = np.asarray(z[f"c{c}_cand_field"], np.int64)
        feas = [np.asarray(f, np.int64) for f in z[f"c{c}_feas"]]
        R = float(z[f"c{c}_R"])
        criterion = str(z[f"c{c}_criterion"])
        n_fleet = int(z[f"c{c}_n_fleet"])
        recorded = [int(j) for j in z[f"c{c}_chosen_field"]]
        res = select_locations(anomalies(A), cand_field, feas, R, n_fleet,
                               criterion=criterion)
        assert list(res["chosen_field"]) == recorded, (
            f"{path.name} cycle {c}: replay chose different cells")
        print(f"  {path.name} cycle {c}: {len(recorded)} placements identical")
