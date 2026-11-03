"""Paired significance tests and the separation rule used by the tables.

Separation (Table I dagger rule, Sec. V): two policies are reliably separated
on a slice iff BOTH
  (a) the two-sided Wilcoxon signed-rank test on the 10 per-IC paired deltas
      gives p < 0.05, and
  (b) the mean gap exceeds the pooled run-to-run floor of that slice
      (references/tolerance.csv).
A dagger in Table I marks an arm NOT reliably separated from the row's best.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

ALPHA = 0.05


def wilcoxon_p(deltas) -> float:
    """Two-sided Wilcoxon signed-rank p on paired per-IC deltas."""
    return float(st.wilcoxon(np.asarray(deltas, float)).pvalue)


def sign_test_p(deltas) -> float:
    d = np.asarray(deltas, float)
    return float(st.binomtest(int((d < 0).sum()), len(d), 0.5).pvalue)


def separated(deltas, floor: float, alpha: float = ALPHA) -> dict:
    """The separation rule on a set of per-IC paired deltas."""
    d = np.asarray(deltas, float)
    p = wilcoxon_p(d)
    gap = float(d.mean())
    return dict(gap=gap, wilcoxon_p=p, floor=floor,
                separated=bool(p < alpha and abs(gap) > floor))
