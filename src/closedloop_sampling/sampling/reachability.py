"""Current-advected reachability (vehicle physics, target-independent).

Implements Sec. III (Reachable sets) / Alg. 1 lines 8-10: per cycle each
vehicle's reachable set is the wet cells it can propel to in the cycle window
while the depth-mean surface current carries it — metres-based against the
region's anisotropic per-axis cell size, never a uniform cell radius.

Ported verbatim from the production reachability (the metres + corrected
u/v-pairing form the campaigns ran). The research module also carried an ETKF
and a latent-inflation helper; both are dead on the production path and are
not ported. numpy-only.
"""
from __future__ import annotations

import numpy as np


def cell_metres(ds, mask, crop):
    """Domain-mean per-axis cell size (metres) from the region's ROMS grid
    metrics. ROMS `pm` = 1/dx (xi = COLUMN = ~E-W spacing), `pn` = 1/dy
    (eta = ROW = ~N-S spacing). Returns (dx_col_m, dy_row_m). Region-agnostic:
    reads this store's own metrics, so CCS gets its anisotropic
    (~8.7 km col / ~11.1 km row) sizes and MAB ~7004/6985 m automatically.
    """
    pm = ds["pm"].values[crop]
    pn = ds["pn"].values[crop]
    dx_col_m = float(1.0 / np.nanmean(pm[mask]))
    dy_row_m = float(1.0 / np.nanmean(pn[mask]))
    return dx_col_m, dy_row_m


def advected_reachable(fleet, wet_rc, c_field, dx_col_m, dy_row_m, v_prop_ms,
                       t_s, n_fleet):
    """Current-advected reachability in METRES against anisotropic per-axis
    cell size. For each vehicle at `fleet`, the reachable set is the cells it
    can propel to over t_s given the current carries it: reachable when
    ||d - c*t|| <= v_prop*t (metres), d = fleet->cell displacement, so the
    disk is centred DOWNSTREAM at f + c*t.

    Axis convention (native rotated cell frame; NO angle rotation — u, v are
    native xi/eta components): c_field[0] = u = xi/COLUMN velocity;
    c_field[1] = v = eta/ROW velocity. Correct pairing: ROW/eta advected by v,
    COLUMN/xi advected by u.

    Returns (cand_rc (M, 2) reachable cells, feas list[n_fleet] of
    position-indices into cand_rc reachable by each vehicle,
    reach (n_fleet, n_wet) bool).
    """
    u_wet = c_field[0][wet_rc[:, 0], wet_rc[:, 1]]                            # xi/col velocity (m/s)
    v_wet = c_field[1][wet_rc[:, 0], wet_rc[:, 1]]                            # eta/row velocity (m/s)
    dr0 = wet_rc[None, :, :] - fleet[:, None, :]                              # (n_fleet,n_wet,2) cell offsets (drow,dcol)
    d_row_m = dr0[:, :, 0] * dy_row_m                                         # N-S metres
    d_col_m = dr0[:, :, 1] * dx_col_m                                         # E-W metres
    Rm = v_prop_ms * t_s                                                      # propulsion radius (m)
    bare = (d_row_m ** 2 + d_col_m ** 2) <= Rm ** 2                           # propulsion disk (pre-advection)
    u_auv = np.array([np.nanmean(u_wet[bare[a]]) if bare[a].any() else 0.0
                      for a in range(n_fleet)])
    v_auv = np.array([np.nanmean(v_wet[bare[a]]) if bare[a].any() else 0.0
                      for a in range(n_fleet)])
    drift_row_m = np.nan_to_num(v_auv) * t_s                                  # ROW advected by v
    drift_col_m = np.nan_to_num(u_auv) * t_s                                  # COLUMN advected by u
    dra_row = d_row_m - drift_row_m[:, None]
    dra_col = d_col_m - drift_col_m[:, None]
    reach = (dra_row ** 2 + dra_col ** 2) <= Rm ** 2                          # advected disk (downstream)
    cand_any = reach.any(0)
    cand_rc = wet_rc[cand_any]
    look = {(int(r), int(cc)): i for i, (r, cc) in enumerate(cand_rc)}
    feas = [[look[(int(r), int(cc))] for r, cc in wet_rc[reach[a]]]
            for a in range(n_fleet)]
    return cand_rc, feas, reach
