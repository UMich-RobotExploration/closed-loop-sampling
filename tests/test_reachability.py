"""Reachability gate (port of the production reachability verification).

Uses the real per-axis grid metrics recorded by the campaigns
(tests/references/grid_metrics.json) with synthetic open-water masks. Checks:
  (1) metric correctness on the anisotropic CCS grid: a 1.5 m/s vehicle over
      6 h (32.4 km propulsion) reaches ~3.7 cells E-W vs ~2.9 N-S — never a
      uniform cell radius;
  (2) zero current: the advected set equals the bare propulsion disk and its
      metric extent respects the propulsion radius;
  (3) advection sanity and axis pairing: a uniform current shifts the
      reachable-set centroid DOWNSTREAM — u along COLUMNS, v along ROWS —
      with cosine ~ +1 between the centroid shift and the true drift;
  (4) feasibility lists index cand_rc consistently.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from closedloop_sampling.sampling.reachability import advected_reachable

GM = json.load(open(Path(__file__).parent / "references" / "grid_metrics.json"))
V = GM["v_auv_ms"]
WIN = GM["reach_window_s"]
PROP_M = V * WIN                                  # 32,400 m


def open_water(h=60, w=80):
    mask = np.ones((h, w), bool)
    wet = np.stack(np.where(mask), 1)
    return mask, wet


def centroid(cells):
    return np.asarray(cells, float).mean(0)


def test_ccs_metric_correctness():
    dx, dy = GM["ccs"]["dx_col_m"], GM["ccs"]["dy_row_m"]
    reach_ew = PROP_M / dx
    reach_ns = PROP_M / dy
    assert 3.5 < reach_ew < 3.9, reach_ew          # ~3.73 cells E-W
    assert 2.7 < reach_ns < 3.1, reach_ns          # ~2.92 cells N-S
    assert reach_ew > reach_ns                     # anisotropy preserved
    # The historical isotropic 7000 m radius (4.63 cells on every axis) is
    # wrong on this grid on both axes.
    uniform = PROP_M / 7000.0
    assert abs(uniform - reach_ew) > 0.5 and abs(uniform - reach_ns) > 1.0


def test_zero_current_equals_propulsion_disk():
    dx, dy = GM["mab"]["dx_col_m"], GM["mab"]["dy_row_m"]
    mask, wet = open_water()
    fleet = np.array([[30.0, 40.0]])
    c0 = np.zeros((2,) + mask.shape)
    cand_rc, feas, reach = advected_reachable(fleet, wet, c0, dx, dy, V, WIN, 1)
    # metric extent bounded by the propulsion radius
    d_row = (cand_rc[:, 0] - 30.0) * dy
    d_col = (cand_rc[:, 1] - 40.0) * dx
    assert ((d_row ** 2 + d_col ** 2) <= PROP_M ** 2 + 1e-6).all()
    # symmetric about the vehicle (open water, no current)
    assert abs(centroid(cand_rc)[0] - 30.0) < 1e-9
    assert abs(centroid(cand_rc)[1] - 40.0) < 1e-9
    # cell extents match the per-axis radii
    assert cand_rc[:, 0].max() - 30 == int(PROP_M / dy)
    assert cand_rc[:, 1].max() - 40 == int(PROP_M / dx)


def test_advection_axis_pairing_and_direction():
    dx, dy = GM["mab"]["dx_col_m"], GM["mab"]["dy_row_m"]
    mask, wet = open_water()
    fleet = np.array([[30.0, 40.0]])
    # u = xi/COLUMN velocity: columns must shift, rows must not.
    u = 0.6
    cu = np.zeros((2,) + mask.shape)
    cu[0] += u
    cand_u, _, _ = advected_reachable(fleet, wet, cu, dx, dy, V, WIN, 1)
    shift = centroid(cand_u) - np.array([30.0, 40.0])
    expected_cols = u * WIN / dx
    assert abs(shift[0]) < 0.35                    # rows barely move
    assert abs(shift[1] - expected_cols) < 0.35    # columns shift downstream
    # v = eta/ROW velocity: rows shift, columns do not.
    v = -0.4
    cv = np.zeros((2,) + mask.shape)
    cv[1] += v
    cand_v, _, _ = advected_reachable(fleet, wet, cv, dx, dy, V, WIN, 1)
    shift_v = centroid(cand_v) - np.array([30.0, 40.0])
    expected_rows = v * WIN / dy
    assert abs(shift_v[1]) < 0.35
    assert abs(shift_v[0] - expected_rows) < 0.35
    # cosine(centroid shift, true drift) ~ +1 for a diagonal current
    cd = np.zeros((2,) + mask.shape)
    cd[0] += 0.5
    cd[1] += 0.3
    cand_d, _, _ = advected_reachable(fleet, wet, cd, dx, dy, V, WIN, 1)
    shift_m = (centroid(cand_d) - np.array([30.0, 40.0])) * np.array([dy, dx])
    drift_m = np.array([0.3 * WIN, 0.5 * WIN])     # (row metres, col metres)
    cos = (shift_m @ drift_m) / (np.linalg.norm(shift_m) * np.linalg.norm(drift_m))
    assert cos > 0.98, cos


def test_feasibility_indexes_candidates():
    dx, dy = GM["ccs"]["dx_col_m"], GM["ccs"]["dy_row_m"]
    mask, wet = open_water(40, 50)
    fleet = np.array([[10.0, 10.0], [30.0, 40.0]])
    c = np.zeros((2,) + mask.shape)
    c[0] += 0.3
    cand_rc, feas, reach = advected_reachable(fleet, wet, c, dx, dy, V, WIN, 2)
    assert len(feas) == 2
    for a in range(2):
        cells = cand_rc[np.asarray(feas[a], int)]
        assert reach[a].sum() == len(cells)
        lookup = {(int(r), int(cc)) for r, cc in wet[reach[a]]}
        assert {(int(r), int(cc)) for r, cc in cells} == lookup
