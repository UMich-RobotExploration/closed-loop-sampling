"""Lawnmower policy: systematic boustrophedon survey baseline (pure numpy,
no RNG).

Implements Sec. IV (Strategies), the uncertainty-blind survey baseline. The
routing code below is the production implementation, unchanged.

Partition (component-aware): 4-connected wet components are labeled first;
vehicles are apportioned over components by largest remainder on wet count;
a recursive WET-COUNT bisection (each node split perpendicular to its longer
METRIC extent, dy_row_m/dx_col_m; workload-balanced coverage partitioning in
the DARP / Rekleitis style) then runs within each component that received at
least one vehicle. Components with no vehicle are ORPHANS: attached to the
metre-nearest routed cell's vehicle for partition accounting, but never routed
(a blind bisection across disconnected components produces route jumps far
beyond the per-cycle reach). Deterministic given (wet, n, grid metrics).

Route: per-cell boustrophedon over every k-th column (serpentine;
lane-to-lane hops count toward path length), k = the smallest lane-skip whose
total path fits the campaign travel budget
    budget_m = SLACK * v_auv * window_s * n_stations
(Choset-Pignon lawnmower decomposition; lane spacing set by the platform
travel budget). Stations (multi-leg + wet-transit): the serpentine is
decomposed into LEGS at over-reach boundaries (consecutive route cells further
apart than reach_m); within each leg, stations are placed uniformly in arc
length with snapping restricted to the leg's own cells; each over-reach jump
is sampled along its straight line at <= SLACK*reach_m spacing and samples
snap to wet cells within PASS_CHEB (wet-transit stations; dry samples yield
none — land crossings stay stationless). The schedule budget charges transits
explicitly; leg stations are apportioned to legs by largest remainder on leg
path; zero-station legs are excised (their cells remain observable en route).

Tracking: metre-true nearest-feasible step toward the current station under
the SHARED advected reachability; arrival within PASS_CHEB cells advances the
pointer; no metric progress for STALL_MAX cycles skips the station (counted).
No uncertainty information enters any choice.
"""
from __future__ import annotations

import numpy as np

from closedloop_sampling.sampling.policies.base import Policy, register

SLACK = 0.8            # fraction of the kinematic travel budget a route may claim (design declaration)
PASS_CHEB = 1          # station counts as reached within this chebyshev cell distance
STALL_MAX = 3          # cycles without metric progress before the current station is skipped
PROGRESS_FRAC = 0.25   # required per-cycle approach toward the station, as a fraction of reach_m


def label_wet_components(wet, grid_shape):
    """4-connectivity connected-component labeling of the wet mask.
    Returns (comp (n_wet,) 0-based component id per wet cell, sizes (n_comp,)
    wet-cell counts, method). Deterministic either way."""
    m = np.zeros(grid_shape, bool)
    m[wet[:, 0], wet[:, 1]] = True
    try:
        from scipy.ndimage import label as _label
        lab, _ = _label(m, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        method = "scipy.ndimage.label"
    except ImportError:                                    # pure-numpy BFS fallback
        lab = np.zeros(grid_shape, int); nxt = 0
        for r0, c0 in wet:
            if lab[r0, c0] == 0:
                nxt += 1; lab[r0, c0] = nxt; stack = [(int(r0), int(c0))]
                while stack:
                    r, c = stack.pop()
                    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= rr < grid_shape[0] and 0 <= cc < grid_shape[1] \
                                and m[rr, cc] and lab[rr, cc] == 0:
                            lab[rr, cc] = nxt; stack.append((rr, cc))
        method = "numpy-bfs"
    comp = lab[wet[:, 0], wet[:, 1]].astype(int) - 1
    return comp, np.bincount(comp), method


def build_cells(wet, n, dx_col_m, dy_row_m):
    """Component-aware wet-count partition (see module docstring).
    Component order: size desc, then min wet index (stable). Vehicles per
    component by LARGEST REMAINDER on wet count; the bisection runs inside each
    component with >= 1 vehicle; 0-vehicle components are orphans, each orphan
    cell attached to the vehicle owning the metre-nearest routed cell
    (accounting only — never routed).
    Returns (cells, orphans): cells = list of n ROUTED index arrays into `wet`;
    orphans = list of n attached orphan index arrays into `wet` (possibly empty)."""
    def rec(idx, nt):                                      # bisection
        if nt == 1:
            return [idx]
        n1 = nt // 2; n2 = nt - n1
        rows = wet[idx, 0]; cols = wet[idx, 1]
        row_ext = (rows.max() - rows.min()) * dy_row_m
        col_ext = (cols.max() - cols.min()) * dx_col_m
        key = rows if row_ext > col_ext else cols          # tie -> column coordinate
        order = np.argsort(key, kind="stable")
        cut = int(round(len(idx) * n1 / nt))
        return rec(idx[order[:cut]], n1) + rec(idx[order[cut:]], n2)

    grid_shape = (int(wet[:, 0].max()) + 1, int(wet[:, 1].max()) + 1)
    comp, sizes, _ = label_wet_components(wet, grid_shape)
    ncomp = len(sizes)
    min_idx = np.array([int(np.flatnonzero(comp == c).min()) for c in range(ncomp)])
    corder = sorted(range(ncomp), key=lambda c: (-int(sizes[c]), int(min_idx[c])))
    quota = n * sizes.astype(float) / len(wet)             # largest-remainder apportionment
    alloc = np.floor(quota).astype(int)
    for c in sorted(corder, key=lambda c: -(quota[c] - np.floor(quota[c])))[:n - int(alloc.sum())]:
        alloc[c] += 1                                      # stable sort: remainder tie -> component order
    cells = []
    for c in corder:
        if alloc[c] >= 1:
            cells.extend(rec(np.flatnonzero(comp == c), int(alloc[c])))
    assert len(cells) == n, f"apportionment produced {len(cells)} != {n} cells"
    orphans = [[] for _ in range(n)]
    orphan_comps = [c for c in corder if alloc[c] == 0]
    if orphan_comps:
        routed_all = np.concatenate(cells)
        owner = np.concatenate([np.full(len(ci), a, int) for a, ci in enumerate(cells)])
        rw = wet[routed_all]
        for c in orphan_comps:
            for o in np.flatnonzero(comp == c):
                d2 = ((rw[:, 0] - wet[o, 0]) * dy_row_m) ** 2 + ((rw[:, 1] - wet[o, 1]) * dx_col_m) ** 2
                orphans[int(owner[int(np.argmin(d2))])].append(int(o))   # tie -> first routed cell
    return cells, [np.asarray(sorted(o), int) for o in orphans]


LAST_STATION_DIAG = None   # per-call diagnostics from build_route's station step:
                           # dict(n_interp=#interpolated off-route stations,
                           #      snap_cheb=[chebyshev cell distance interp->snapped wet cell, ...])


def build_route(wet, cell_idx, n_stations, dx_col_m, dy_row_m, v_auv, window_s):
    """Budget-matched boustrophedon route over ONE cell; multi-leg +
    wet-transit stations (see module docstring). `wet` is the FULL wet-cell
    array (transit snapping needs it); leg-station snapping stays restricted to
    the leg's own cells.
    Returns (stations (ns,2) int, k lane-skip, path_m, budget_m, max_pair_m,
    leginfo). Pairs crossing dry stretches may exceed reach (dry_pairs_m);
    within-leg pairs can exceed reach by snap quantization on single-column
    legs (tracked in within_leg_over_reach_m)."""
    budget_m = SLACK * v_auv * window_s * n_stations
    rc = wet[np.asarray(cell_idx, int)]
    cols = np.unique(rc[:, 1])

    def serpentine(kept):                       # alternate row direction per kept column
        out = []
        for i, c in enumerate(kept):
            r = np.sort(rc[rc[:, 1] == c, 0])
            if i % 2 == 1:
                r = r[::-1]
            out.extend((int(rr), int(c)) for rr in r)
        return np.asarray(out, int)

    def seglen(o):                              # consecutive metre distances incl. lane hops
        if len(o) < 2:
            return np.zeros(0)
        d = np.diff(o, axis=0)
        return np.sqrt((d[:, 0] * dy_row_m) ** 2 + (d[:, 1] * dx_col_m) ** 2)

    route = None; k_used = None; path_m = None
    for k in range(1, len(cols) + 1):           # smallest lane-skip that fits the budget
        o = serpentine(cols[::k])
        p = float(seglen(o).sum())
        if p <= budget_m:
            route, k_used, path_m = o, k, p
            break
    if route is None:                           # even the single lane exceeds -> longest prefix
        k_used = len(cols)
        o = serpentine(cols[::k_used])
        cum = np.concatenate([[0.0], np.cumsum(seglen(o))])
        nkeep = max(1, int(np.searchsorted(cum, budget_m, side="right")))
        route, path_m = o[:nkeep], float(cum[nkeep - 1])
    # --- leg decomposition + wet-transit stations (k-selection above unchanged) ---
    global LAST_STATION_DIAG
    reach_m = v_auv * window_s
    seg = seglen(route)
    half_cell = 0.5 * min(dx_col_m, dy_row_m)              # vertex-snap tolerance along the arc

    def met(a, b):
        return float(np.sqrt(((a[0] - b[0]) * dy_row_m) ** 2 + ((a[1] - b[1]) * dx_col_m) ** 2))

    brk = [int(b) for b in np.flatnonzero(seg > reach_m)]  # over-reach boundaries
    legs = []; s0 = 0
    for b in brk:
        legs.append((s0, b)); s0 = b + 1
    legs.append((s0, len(route) - 1))
    jumps = [float(seg[b]) for b in brk]
    L = [float(seg[a:b].sum()) for a, b in legs]
    totL = float(sum(L))
    jst = []; jextra = []; jcls = []; snap_cheb_tr = []
    for b in brk:                                          # wet-transit stations per jump
        p0 = route[b].astype(float); p1 = route[b + 1].astype(float); d = float(seg[b])
        m = int(np.ceil(d / (SLACK * reach_m)))
        pts = []
        for i in range(1, m):                              # interior samples; endpoints are route cells
            p = p0 + (p1 - p0) * (i / m)
            d2 = ((wet[:, 0] - p[0]) * dy_row_m) ** 2 + ((wet[:, 1] - p[1]) * dx_col_m) ** 2
            q = wet[int(np.argmin(d2))]
            cheb = float(max(abs(q[0] - p[0]), abs(q[1] - p[1])))
            if cheb <= PASS_CHEB:                          # dry samples yield NO station
                pts.append((np.asarray(q, int), i, m)); snap_cheb_tr.append(round(cheb, 3))
        chain = [p0] + [q.astype(float) for q, _, _ in pts] + [p1]
        subs = [met(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
        jextra.append(int(sum(int(np.ceil(s / reach_m)) - 1 for s in subs)))
        jcls.append("wet" if len(pts) == m - 1 else ("dry" if not pts else "partial"))
        jst.append(pts)
    transit_extra = int(sum(jextra))
    n_st = max(2, min(n_stations - transit_extra, int(np.ceil(totL / (SLACK * reach_m))) + 1))
    if totL > 0:                                           # leg stations by largest remainder on leg path
        quota = np.array(L) * n_st / totL
        s_leg = np.floor(quota).astype(int)
        for i in sorted(range(len(legs)), key=lambda i: -(quota[i] - np.floor(quota[i])))[:n_st - int(s_leg.sum())]:
            s_leg[i] += 1
    else:
        s_leg = np.zeros(len(legs), int); s_leg[0] = 1
    excised_cells = int(sum((b - a + 1) for (a, b), s in zip(legs, s_leg) if s == 0))
    tagged = []                                            # (cell, ("leg"|"transit", leg_id), sample (i,m)|None)

    def put(q, kind, samp=None):
        if tagged and np.array_equal(q, tagged[-1][0]):
            return                                         # dedupe consecutive identical
        tagged.append((np.asarray(q, int), kind, samp))

    n_interp = 0; snap_cheb = []
    for li, ((a, b), si) in enumerate(zip(legs, s_leg)):
        if si > 0:                                         # arc placement, per leg
            sub = route[a:b + 1]; subseg = seg[a:b]
            cum = np.concatenate([[0.0], np.cumsum(subseg)])
            for s in np.linspace(0.0, cum[-1], si):
                i = min(max(int(np.searchsorted(cum, s, side="right")) - 1, 0), len(sub) - 1)
                if abs(s - cum[i]) <= half_cell:
                    put(sub[i], ("leg", li)); continue
                if i + 1 < len(sub) and abs(cum[i + 1] - s) <= half_cell:
                    put(sub[i + 1], ("leg", li)); continue
                t = (s - cum[i]) / subseg[i]
                p = sub[i] + t * (sub[i + 1] - sub[i])
                d2 = ((sub[:, 0] - p[0]) * dy_row_m) ** 2 + ((sub[:, 1] - p[1]) * dx_col_m) ** 2
                q = sub[int(np.argmin(d2))]                # snap restricted to the LEG's own cells
                n_interp += 1
                snap_cheb.append(float(max(abs(q[0] - p[0]), abs(q[1] - p[1]))))
                put(q, ("leg", li))
        if li < len(jst):                                  # encounter order: leg, then its jump
            for q, i, m in jst[li]:
                put(q, ("transit", li), (i, m))
    stations = np.asarray([t[0] for t in tagged], int)
    dry_pairs_m = []; within_over = []; nondry_over = []
    max_pair_m = 0.0; max_nondry_pair_m = 0.0; max_within_leg_pair_m = 0.0
    for i in range(1, len(tagged)):
        (qa, ka, sa), (qb, kb, sb) = tagged[i - 1], tagged[i]
        d = met(qa.astype(float), qb.astype(float))
        max_pair_m = max(max_pair_m, d)
        if ka[0] == "leg" and kb[0] == "leg":
            if ka[1] == kb[1]:
                cls, dry = "within-leg", False
                max_within_leg_pair_m = max(max_within_leg_pair_m, d)
                if d > reach_m:
                    within_over.append(round(d))           # snap-quantization pairs (see docstring)
            else:
                cls, dry = "leg<->leg-direct", True        # spans a dry/excised gap
        elif ka[0] == "leg":                               # leg -> first kept transit sample
            cls, dry = ("leg-end<->transit", False) if sb[0] == 1 else ("dry-flank", True)
        elif kb[0] == "leg":                               # last kept transit sample -> leg
            cls, dry = ("transit<->leg-start", False) if sa[0] == sa[1] - 1 else ("dry-flank", True)
        else:
            cls, dry = ("transit<->transit", False) if (ka[1] == kb[1] and sb[0] == sa[0] + 1) \
                else ("dry-flank", True)
        if dry:
            dry_pairs_m.append(round(float(d)))
        else:
            max_nondry_pair_m = max(max_nondry_pair_m, d)
            if d > reach_m:
                nondry_over.append((cls, round(float(d))))  # audit-priced
    LAST_STATION_DIAG = dict(n_interp=n_interp, snap_cheb=snap_cheb)
    leginfo = dict(n_legs=len(legs), jumps=[round(j) for j in jumps], jump_class=jcls,
                   n_transit_st=int(sum(len(p) for p in jst)), transit_extra=transit_extra,
                   snap_cheb_transit=snap_cheb_tr, excised_cells=excised_cells,
                   dry_pairs_m=dry_pairs_m, max_nondry_pair_m=round(max_nondry_pair_m),
                   max_within_leg_pair_m=round(max_within_leg_pair_m),
                   within_leg_over_reach_m=within_over, nondry_over_pairs=nondry_over)
    return stations, k_used, path_m, budget_m, max_pair_m, leginfo


def lawn_step(pos_rc, route, ptr, stall, feas_cells_rc, dx_col_m, dy_row_m, reach_m):
    """One survey step under the SHARED feasibility set.
    chosen = feasible cell minimizing metre distance to route[ptr]. Arrival
    within PASS_CHEB advances ptr (consuming any further stations already
    within PASS_CHEB); otherwise the approach must shrink the distance by
    PROGRESS_FRAC*reach_m or stall increments, and STALL_MAX stalls skip the
    station (counted). Pointer saturates at the last station; thereafter the
    vehicle station-keeps on it.
    Returns (chosen (2,) int array, ptr, stall, skipped_increment)."""
    def dist_m(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return np.sqrt(((a[..., 0] - b[0]) * dy_row_m) ** 2 + ((a[..., 1] - b[1]) * dx_col_m) ** 2)

    tgt = route[ptr]
    d = dist_m(feas_cells_rc, tgt)
    chosen = np.asarray(feas_cells_rc[int(np.argmin(d))], int)   # argmin tie -> first (no RNG)
    skipped = 0
    if max(abs(int(chosen[0]) - int(tgt[0])), abs(int(chosen[1]) - int(tgt[1]))) <= PASS_CHEB:
        ptr += 1; stall = 0
        while ptr < len(route) and max(abs(int(chosen[0]) - int(route[ptr][0])),
                                       abs(int(chosen[1]) - int(route[ptr][1]))) <= PASS_CHEB:
            ptr += 1                             # pass-through stations already reached
    elif float(dist_m(chosen, tgt)) < float(dist_m(pos_rc, tgt)) - PROGRESS_FRAC * reach_m:
        stall = 0                                # genuine metric progress toward the station
    else:
        stall += 1
        if stall >= STALL_MAX:
            ptr += 1; skipped = 1; stall = 0     # station judged unreachable: skip (counted)
    ptr = min(ptr, len(route) - 1)               # saturate at last station (station-keeping)
    return chosen, ptr, stall, skipped


@register("lawnmower")
class LawnmowerPolicy(Policy):
    """Policy wrapper: routes built once per chain; per cycle each vehicle
    takes one tracked step toward its current station (the production loop's
    lawn branch, unchanged)."""
    moving = True
    uses_ensemble = False

    def __init__(self, chain):
        super().__init__(chain)
        cells, _ = build_cells(chain.wet, chain.n_fleet,
                               chain.dx_col_m, chain.dy_row_m)
        self.routes = [build_route(chain.wet, cells[a], chain.n_cycles,
                                   chain.dx_col_m, chain.dy_row_m,
                                   chain.v_auv_ms, chain.reach_window_s)[0]
                       for a in range(chain.n_fleet)]
        n = chain.n_fleet
        self.ptr = np.zeros(n, int)
        self.stall = np.zeros(n, int)
        self.done = np.zeros(n, bool)            # final-station first-arrival latch
        self.completed = np.zeros(n, int)
        self.skipped = np.zeros(n, int)

    def start_fleet(self):
        """Lawnmower owns its initialization: first waypoint of each route."""
        return np.array([r[0] for r in self.routes], float)

    def plan(self, ctx):
        chain = self.chain
        wet = chain.wet
        new_fleet = ctx.fleet.copy()
        chosen_field = []
        reach_m = chain.v_auv_ms * chain.reach_window_s
        for a in range(chain.n_fleet):
            if not ctx.feas[a]:
                continue
            fc_rc = ctx.cand_rc[np.asarray(ctx.feas[a], int)]   # this vehicle's feasible wet cells
            p0 = int(self.ptr[a])
            tgt0 = self.routes[a][p0]
            ch, p1, s1, sk = lawn_step(ctx.fleet[a], self.routes[a], p0,
                                       int(self.stall[a]), fc_rc,
                                       chain.dx_col_m, chain.dy_row_m, reach_m)
            self.ptr[a] = p1
            self.stall[a] = s1
            self.skipped[a] += sk
            if max(abs(int(ch[0]) - int(tgt0[0])), abs(int(ch[1]) - int(tgt0[1]))) <= PASS_CHEB:
                if p1 > p0:
                    self.completed[a] += p1 - p0            # stations passed this cycle
                elif not self.done[a]:
                    self.completed[a] += 1                  # final station, first arrival
                    self.done[a] = True
            jf = int(chain.wet_lookup[(int(ch[0]), int(ch[1]))])
            assert jf in {int(ctx.cand_field[p]) for p in ctx.feas[a]}, \
                f"lawnmower: chosen cell {jf} not feasible for vehicle {a}"
            new_fleet[a] = wet[jf]
            if jf not in chosen_field:
                chosen_field.append(jf)
        ctx.extras["stations_completed"] = [int(x) for x in self.completed]
        ctx.extras["n_skipped"] = [int(x) for x in self.skipped]
        return chosen_field, new_fleet
