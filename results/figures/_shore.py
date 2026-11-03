"""View-from-shore map convention for the paper figures.

Every map is drawn as seen from the shore: ocean at the top, coast along the
bottom, and the display is a PROPER rotation of the geography (determinant +1),
never a mirror image. The raw arrays are drawn with imshow(origin="upper"); in
both regions the row index increases landward, so drawing them untransformed
would mirror the geography. The fix per region:
    mab : flip left-right      (north points down-left)
    ccs : transpose            (north points right)
Furniture: a compass rose (red north; N/E/S/W taken from the grid, so a mirror
would show) above a 100 km bar, both on land; optional very light state and
country borders (Natural Earth 50 m, when cartopy and the grid lon/lat are
available). Nothing here touches data: chains, decodes, reachable sets and
routes are all computed in raw grid coordinates; only the drawing is
transformed.

Construction: Shore.from_grid(region, mask, lon, lat, dx, dy) derives the
display north/east vectors from the grid; Shore(region, mask, dx, dy, north,
east) takes them precomputed (the committed Fig. 5 context stores them, so the
figure regenerates without the full lon/lat arrays; borders are skipped then).
"""
from __future__ import annotations

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D, ScaledTranslation
from scipy.ndimage import distance_transform_edt, gaussian_filter

LAND, COAST, OCEAN = "#e8e8e8", "0.45", "#e8f4fb"
BORDER, INK, N_RED = "0.80", "0.25", "#c0392b"
TRANSFORM = {"mab": "fliplr", "ccs": "transpose"}
# bottom-right anchors (axes fraction), on land in both regions
ANCHORS = {"mab": dict(rose=(0.93, 0.30), bar_cx=0.93, bar_y=0.06),
           "ccs": dict(rose=(0.88, 0.31), bar_cx=0.88, bar_y=0.06)}
K, SIG, LEV = 4, 2.2, 0.3          # coast smoothing of the displayed mask


class Shore:
    def __init__(self, region, mask, dx, dy, north, east, lon=None, lat=None):
        """mask (and lon/lat if given) in RAW array orientation; dx, dy =
        metres per raw column / raw row; north/east = display-space unit
        vectors of grid north and east."""
        self.region, self.t = region, TRANSFORM[region]
        self.mask = self.apply(mask)
        self.lon = self.apply(lon) if lon is not None else None
        self.lat = self.apply(lat) if lat is not None else None
        self.H, self.W = self.mask.shape
        self.dx, self.dy = (dy, dx) if self.t == "transpose" else (dx, dy)
        self.SM = self.smooth(mask)
        self.BX = (np.arange(self.SM.shape[1]) + 0.5) / K - 0.5
        self.BY = (np.arange(self.SM.shape[0]) + 0.5) / K - 0.5
        self._FI, self._FJ = distance_transform_edt(~self.mask, return_indices=True)[1]
        self.north = np.asarray(north, float)
        self.east = np.asarray(east, float)
        self.det = float(self.east[0] * self.north[1] - self.east[1] * self.north[0])
        assert self.det > 0, f"{region}: display is a mirror image (det {self.det:+.2f})"
        self._tree = None

    @classmethod
    def from_grid(cls, region, mask, lon, lat, dx, dy):
        """Derive the display north/east vectors from the grid lon/lat."""
        t = TRANSFORM[region]
        lond = np.asarray(lon, float)[:, ::-1] if t == "fliplr" else np.asarray(lon, float).T
        latd = np.asarray(lat, float)[:, ::-1] if t == "fliplr" else np.asarray(lat, float).T
        i, j = lond.shape[0] // 2, lond.shape[1] // 2
        c0 = float(np.cos(np.radians(np.nanmean(latd))))
        r = np.array([(lond[i, j + 1] - lond[i, j - 1]) * c0, latd[i, j + 1] - latd[i, j - 1]])
        u = np.array([(lond[i - 1, j] - lond[i + 1, j]) * c0, latd[i - 1, j] - latd[i + 1, j]])
        r /= np.linalg.norm(r)
        u /= np.linalg.norm(u)
        sh = cls(region, mask, dx, dy, north=np.array([r[1], u[1]]),
                 east=np.array([r[0], u[0]]), lon=lon, lat=lat)
        sh.c0 = c0
        return sh

    # ---- geometry -------------------------------------------------------------------
    def apply(self, a):
        """Raw 2-D array -> displayed orientation."""
        a = np.asarray(a)
        return a[:, ::-1] if self.t == "fliplr" else a.T

    def xy(self, rows, cols):
        """Raw (row, col) positions -> displayed (x, y) data coordinates."""
        rows = np.asarray(rows, float)
        cols = np.asarray(cols, float)
        if self.t == "fliplr":
            return (self.W - 1) - cols, rows
        return rows, cols

    def fill_to_coast(self, f_raw):
        """Display-only nearest-wet fill of a raw field, in displayed orientation."""
        return self.apply(f_raw)[self._FI, self._FJ]

    def smooth(self, m_raw):
        return gaussian_filter(np.kron(self.apply(m_raw).astype(float),
                                       np.ones((K, K))), SIG, mode="nearest")

    def boundary(self, ax, m_raw, color, lw, zorder, alpha=1.0, level=LEV):
        f = self.smooth(m_raw)
        x = (np.arange(f.shape[1]) + 0.5) / K - 0.5
        y = (np.arange(f.shape[0]) + 0.5) / K - 0.5
        ax.contour(x, y, f, levels=[level], colors=color, linewidths=lw,
                   alpha=alpha, zorder=zorder)

    def base(self, ax, ocean=False, coast_lw=0.3, spine_lw=0.4):
        if ocean:
            ax.imshow(np.zeros_like(self.mask, float), cmap=ListedColormap([OCEAN]),
                      origin="upper", zorder=0.5)
        ax.contourf(self.BX, self.BY, self.SM, levels=[-1.0, LEV], colors=[LAND], zorder=3)
        ax.contour(self.BX, self.BY, self.SM, levels=[LEV], colors=COAST,
                   linewidths=coast_lw, zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect(self.dy / self.dx)
        for s in ax.spines.values():
            s.set_linewidth(spine_lw)
            s.set_edgecolor(COAST)

    # ---- borders --------------------------------------------------------------------
    def _to_rc(self, lon, lat):
        from scipy.spatial import cKDTree
        if self._tree is None:
            self._tree = cKDTree(np.column_stack([(self.lon * self.c0).ravel(),
                                                  self.lat.ravel()]))
            self._cell = float(np.nanmedian(np.hypot(
                np.diff(self.lon, axis=1).ravel() * self.c0,
                np.diff(self.lat, axis=1).ravel())))
        dist, idx = self._tree.query(np.column_stack([np.asarray(lon) * self.c0,
                                                      np.asarray(lat)]))
        r, c = np.unravel_index(idx, self.mask.shape)
        return r.astype(float), c.astype(float), dist

    def borders(self, ax, lw=0.3, color=BORDER, zorder=3.5):
        """Very light state/province + country borders (Natural Earth 50 m);
        silent no-op offline or when the grid lon/lat were not provided."""
        if self.lon is None:
            print("  [borders skipped: no grid lon/lat in this context]")
            return 0
        try:
            from cartopy.io import shapereader
            n = 0
            for name in ("admin_1_states_provinces_lines", "admin_0_boundary_lines_land"):
                shp = shapereader.natural_earth(resolution="50m", category="cultural",
                                                name=name)
                for geom in shapereader.Reader(shp).geometries():
                    for ln in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                        s = np.asarray(ln.coords)
                        r, c, dist = self._to_rc(s[:, 0], s[:, 1])
                        ok = dist < 1.5 * self._cell
                        if ok.sum() < 2:
                            continue
                        ax.plot(np.where(ok, c, np.nan), np.where(ok, r, np.nan), "-",
                                color=color, lw=lw, zorder=zorder)
                        n += 1
            return n
        except Exception as e:                        # noqa: BLE001
            print(f"  [borders unavailable: {type(e).__name__}: {e}]")
            return 0

    # ---- furniture ------------------------------------------------------------------
    def rose(self, ax, fig, center, size=1.0, letters="NESW", halo=False,
             style="solid", min_letter_fs=0.0):
        """Compass rose: four solid points, grey with a red north, white centre
        dot. Directions come from the grid (north AND east), so a mirrored map
        would show."""
        from matplotlib.patches import Circle
        tr = (Affine2D().scale(1 / 72.0) + fig.dpi_scale_trans
              + ScaledTranslation(center[0], center[1], ax.transAxes))
        n, e = self.north, self.east
        L = 11.0 * size
        if halo:
            ax.add_patch(Circle((0, 0), L + 1.6 * size, fc="white", ec="none",
                                alpha=0.9, transform=tr, zorder=9.5))
        if style == "solid":
            w = 2.0 * size
            for d, col in ((e, "0.45"), (-n, "0.45"), (-e, "0.45"), (n, N_RED)):
                p = np.array([-d[1], d[0]])
                ax.add_patch(Polygon([tuple(w * p), tuple(L * d), tuple(-w * p)],
                                     closed=True, fc=col, ec="none", transform=tr,
                                     zorder=10))
            ax.add_patch(Circle((0, 0), 1.1 * size, fc="white", ec="none",
                                transform=tr, zorder=11))
        else:
            w = 2.3 * size
            for d, dark in ((e, INK), (-n, INK), (-e, INK), (n, N_RED)):
                p = np.array([-d[1], d[0]])
                ax.add_patch(Polygon([(0, 0), tuple(L * d), tuple(w * p)], closed=True,
                                     fc=dark, ec="none", transform=tr, zorder=10))
                ax.add_patch(Polygon([(0, 0), tuple(L * d), tuple(-w * p)], closed=True,
                                     fc="white", ec=dark, lw=0.3 * max(size, 0.6),
                                     transform=tr, zorder=10))
        for d, lab in ((n, "N"), (e, "E"), (-n, "S"), (-e, "W")):
            if lab not in letters:
                continue
            red = lab == "N"
            ax.annotate(lab, xy=center, xycoords="axes fraction",
                        xytext=tuple((L + 4.2 * size) * d), textcoords="offset points",
                        fontsize=max((6.2 if red else 4.6) * size, min_letter_fs),
                        ha="center", va="center", color=N_RED if red else INK,
                        fontweight="bold" if red else "normal", zorder=12)

    def scale_bar(self, ax, cx, y_frac, size=1.0, km=100.0, label="above", fs=None,
                  halo=False, lw=None):
        """Bar of `km` centred at axes-fraction x = cx at height y_frac; label
        above or below. Returns whether the BAR lies on land."""
        cells = km * 1000.0 / self.dx
        x0 = cx * self.W - cells / 2
        y = (1 - y_frac) * self.H
        fs = fs if fs is not None else 5.0 * max(size, 0.72)
        ax.plot([x0, x0 + cells], [y, y], "-", color=INK,
                lw=(lw if lw is not None else 1.0 * max(size, 0.7)), zorder=7)
        kw = dict(bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5)) if halo else {}
        if label == "left":
            ax.text(x0 - 0.8, y, f"{km:.0f} km", ha="right", va="center", fontsize=fs,
                    color=INK, zorder=7, **kw)
        else:
            off = 2.0 if label == "above" else -2.0
            ax.text(x0 + cells / 2, y - off, f"{km:.0f} km", ha="center",
                    va="bottom" if label == "above" else "top", fontsize=fs,
                    color=INK, zorder=7, **kw)
        xs = np.clip(np.linspace(x0, x0 + cells, 8).astype(int), 0, self.W - 1)
        return bool(all(not self.mask[min(int(y), self.H - 1), x] for x in xs))

    def furniture(self, ax, fig, size=1.0, tag="", anchors=None, letters="NESW",
                  label_fs=None, min_letter_fs=0.0, bar_lw=None):
        """Rose above, bar with its label ABOVE it beneath, both at the
        region's bottom-right anchors (override with `anchors`)."""
        a = anchors or ANCHORS[self.region]
        self.rose(ax, fig, a["rose"], size, letters=letters,
                  min_letter_fs=min_letter_fs)
        on_land = self.scale_bar(ax, a["bar_cx"], a["bar_y"], size, fs=label_fs,
                                 lw=bar_lw)
        print(f"  [{self.region}{(' ' + tag) if tag else ''}] rose+bar at bottom "
              f"right; bar on land: {on_land}; det {self.det:+.0f}; north "
              f"{np.degrees(np.arctan2(self.north[0], self.north[1])):+.0f} deg from up")
        return on_land
