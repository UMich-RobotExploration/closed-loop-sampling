"""Hygiene gate as a test: tools/check_identifiers.py over the tracked tree."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_tree_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_identifiers.py")],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_history_clean():
    proc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_identifiers.py"), "--log"],
        capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
