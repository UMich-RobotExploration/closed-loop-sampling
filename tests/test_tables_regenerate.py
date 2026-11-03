"""Regenerating the paper tables from results/scores/ reproduces the committed
goldens byte for byte (CSV and LaTeX body alike)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
TABLES = REPO / "results/tables"
EXPECTED = TABLES / "expected"
SCRIPTS = ("table1_rmse", "table2_per_ic", "table3_lawnmower_wins")


@pytest.mark.parametrize("name", SCRIPTS)
def test_table_regenerates(name, tmp_path):
    r = subprocess.run([sys.executable, str(TABLES / f"{name}.py"),
                        "--out", str(tmp_path)],
                       capture_output=True, text=True, cwd=TABLES)
    assert r.returncode == 0, r.stdout + r.stderr
    for ext in ("csv", "tex"):
        got = (tmp_path / f"{name}.{ext}").read_bytes()
        want = (EXPECTED / f"{name}.{ext}").read_bytes()
        assert got == want, f"{name}.{ext} drifted from the committed golden"
