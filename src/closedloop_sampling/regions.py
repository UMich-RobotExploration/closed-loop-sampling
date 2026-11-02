"""Per-region configuration: schema + registry.

Implements Sec. IV (Experimental setup): the two regions, their grids, sigma
levels, channel layout, latent shapes, and temporal splits.

READ-ONLY config objects. `RegionConfig` and its sub-dataclasses hold every
dataset-specific value (grid, levels, crop, latent, cadence, paths, validation
ranges, dedup) so preprocessing, dataloading, forecasting, and the closed loop
read one object instead of scattered constants. The authoritative values live
in configs/regions/{mab,ccs}.yaml; :func:`load` parses them into the schema
below. The registry replaces the research code's arm-to-factory monkey
patching: `make_denoiser(region_config)` is constructed natively per region.

Paths inside a config are RELATIVE to $CLOSEDLOOP_DATA_ROOT (data) or
$CLOSEDLOOP_RUNS / $CLOSEDLOOP_WEIGHTS (runs/weights); resolve with
`closedloop_sampling.paths.resolve`.

Usage:
    from closedloop_sampling.regions import load
    cfg = load("mab")              # or "ccs"; "doppio" is a compat alias for "mab"
    idx = cfg.channel_indices()    # {"sst": 59, "sss": 79, "surface_s_rho_index": 19, ...}
    crop = cfg.metric_crop         # THE grid/metric crop (single source of truth)
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import yaml


# ============================================================
# Schema — same field names for every region (swap the config, not the code)
# ============================================================
@dataclass(frozen=True)
class RawSource:
    staging_glob_patterns: Tuple[str, ...]   # CCS has two (core, ext); MAB one
    staging_dir: str                         # relative to the data root
    steps_per_file: int
    time_units: str                          # CF units string of raw/store ocean_time
    calendar: str
    decode_times_on_read: bool
    dtype_on_disk: str


@dataclass(frozen=True)
class GridConfig:
    native_eta_rho: int
    native_xi_rho: int
    u_point_dims: Tuple[int, int]            # (eta_u, xi_u)
    v_point_dims: Tuple[int, int]            # (eta_v, xi_v)
    crop_eta_slice: Tuple[int, int]          # (start, stop) positive indices into eta_rho
    crop_xi_slice: Tuple[int, int]           # (start, stop) positive indices into xi_rho
    cropped_h: int                           # = eta stop - start
    cropped_w: int                           # = xi  stop - start
    cell_size_m: float                       # scalar mean cell size (m)
    crop_note: str = ""                      # physical-edge descriptor (documentation only)


@dataclass(frozen=True)
class VerticalConfig:
    native_level_count: int
    staged_level_indices: Tuple[int, ...]    # native indices present in the staged/store data
    kept_level_indices: Tuple[int, ...]      # native indices after the level-drop (len == n_levels_kept)
    n_levels_kept: int
    dropped_staged_pos: Optional[int]        # position within staged array that was dropped; None if none
    surface_level_index: int                 # index of the surface level within the kept array (= n_levels_kept - 1)
    cs_r_kept_note: str = ""                 # Cs_r values of the kept levels (documentation only)


@dataclass(frozen=True)
class ChannelConfig:
    dyn3d_vars: Tuple[str, ...]              # per-level 3D vars, in channel-stacking order
    dyn2d_vars: Tuple[str, ...]              # 2D surface vars, in channel-stacking order
    n_dynamic_channels: int                  # = len(dyn3d)*n_levels_kept + len(dyn2d)
    static_channels: int
    total_in_channels: int
    total_out_channels: int
    static_vars_2d: Tuple[str, ...]          # rho-grid 2D statics carried into the preprocessed store
    static_vars_1d: Tuple[str, ...]          # s-coord 1D arrays subset to kept levels (s_rho, Cs_r)
    destagger_masks: Tuple[str, ...]         # transient u/v masks used for destaggering (NOT stored)


@dataclass(frozen=True)
class LatentConfig:
    latent_h: int                            # cropped_h // compression_per_dim
    latent_w: int
    latent_channels: int
    compression_per_dim: int
    rope_spatial_grid_start_pos: Tuple[int, int]
    rope_spatial_grid_end_pos: Tuple[int, int]   # [latent_h - 1, latent_w - 1]
    n_spatial_tokens: int                    # latent_h * latent_w


@dataclass(frozen=True)
class TimeSplitConfig:
    cadence_hours: int                       # native store cadence (MAB hourly=1; CCS 6)
    internal_stride: int                     # index steps per 6h window step (MAB 6; CCS 1)
    interval_between_pred: int               # PHYSICAL hours between predicted frames (6 both)
    sequence_length: int                     # cond + pred frames
    n_cond_frames: int
    n_pred_frames: int
    n_timesteps_total: int
    expected_deoverlapped_steps: Optional[int]   # None if no overlap dedup applies
    start_time: str                          # ISO8601
    end_time: str
    train_year_range: Optional[Tuple[int, int]]
    val_year_range: Optional[Tuple[int, int]]
    test_year_range: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class DedupConfig:                           # only populated when overlapping raw files exist
    rule: str
    slice_len_steps: int
    last_file_keeps_tail: bool
    seam_prefers: str
    half_open: bool


@dataclass(frozen=True)
class ValidationRanges:
    temp_min: float
    temp_max: float
    salt_min: float
    salt_max: float
    zeta_abs_max: float


@dataclass(frozen=True)
class Paths:
    """All RELATIVE: data paths against $CLOSEDLOOP_DATA_ROOT, run dirs against
    $CLOSEDLOOP_RUNS or $CLOSEDLOOP_WEIGHTS (closedloop_sampling.paths.resolve)."""
    preprocessed_zarr: str
    pixel_norm_stats: str
    latents_zarr: str
    latent_norm_stats: str
    static_source: str                       # where mask_rho/h/etc. come from
    dcae_run_dir: str
    denoiser_runs_dir: str


@dataclass(frozen=True)
class RegionConfig:
    name: str
    description: str
    raw: RawSource
    grid: GridConfig
    vertical: VerticalConfig
    channels: ChannelConfig
    latent: LatentConfig
    timesplit: TimeSplitConfig
    dedup: Optional[DedupConfig]             # None for non-overlapping datasets (MAB)
    validation: ValidationRanges
    paths: Paths

    def channel_indices(self) -> dict:
        """Channel-index map, COMPUTED from the var/level layout (never hardcoded).

        Channel-stacking order is dyn3d vars (each contributing n_levels_kept
        channels, surface = last level) then dyn2d vars. Surface fields sit at
        `base_offset(var) + surface_level_index`.
        """
        n = self.vertical.n_levels_kept
        s = self.vertical.surface_level_index          # = n - 1
        v3 = self.channels.dyn3d_vars
        base = {var: i * n for i, var in enumerate(v3)}     # u:0, v:n, temp:2n, salt:3n
        two_d_base = len(v3) * n                            # first 2D channel
        out = {
            "u_surface": base["u"] + s,
            "v_surface": base["v"] + s,
            "sst": base["temp"] + s,
            "sss": base["salt"] + s,
            "surface_s_rho_index": s,
        }
        for j, var in enumerate(self.channels.dyn2d_vars):
            out[var] = two_d_base + j                        # zeta, ubar, vbar
        return out

    @property
    def metric_crop(self) -> Tuple[slice, slice]:
        """THE native-grid crop as (eta_slice, xi_slice) — the single source of
        truth for every consumer (dataloaders, truth reads, metric domains).
        Lives in the region YAML and nowhere else; no caller writes the crop as
        literals."""
        e0, e1 = self.grid.crop_eta_slice
        x0, x1 = self.grid.crop_xi_slice
        return (slice(e0, e1), slice(x0, x1))


# ============================================================
# Registry: YAML -> schema
# ============================================================
_ALIASES = {"doppio": "mab"}                 # historical region name
_KNOWN = ("mab", "ccs")


def _tuplify(value):
    if isinstance(value, list):
        return tuple(_tuplify(v) for v in value)
    return value


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown YAML keys {sorted(unknown)}")
    return cls(**{k: _tuplify(v) for k, v in d.items()})


def load(name: str) -> RegionConfig:
    """Load the RegionConfig for `name` ('mab' | 'ccs'; 'doppio' = 'mab') from
    configs/regions/<name>.yaml (or $CLOSEDLOOP_REGIONS/<name>.yaml). Cached:
    aliases resolve to the same instance."""
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _KNOWN:
        raise ValueError(
            f"Unknown region {name!r}; expected one of {sorted(_KNOWN) + sorted(_ALIASES)}."
        )
    return _load_canonical(key)


@functools.lru_cache(maxsize=None)
def _load_canonical(key: str) -> RegionConfig:
    from closedloop_sampling.paths import regions_dir

    path = regions_dir() / f"{key}.yaml"
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    metric_crop = raw.pop("metric_crop")
    cfg = RegionConfig(
        name=raw["name"],
        description=raw["description"],
        raw=_build(RawSource, raw["raw"]),
        grid=_build(GridConfig, raw["grid"]),
        vertical=_build(VerticalConfig, raw["vertical"]),
        channels=_build(ChannelConfig, raw["channels"]),
        latent=_build(LatentConfig, raw["latent"]),
        timesplit=_build(TimeSplitConfig, raw["timesplit"]),
        dedup=_build(DedupConfig, raw["dedup"]) if raw.get("dedup") else None,
        validation=_build(ValidationRanges, raw["validation"]),
        paths=_build(Paths, raw["paths"]),
    )
    # The YAML's metric_crop block must agree with the grid crop it mirrors.
    if (tuple(metric_crop["eta"]) != cfg.grid.crop_eta_slice
            or tuple(metric_crop["xi"]) != cfg.grid.crop_xi_slice):
        raise ValueError(f"{path}: metric_crop does not match grid crop slices")
    return cfg


def get_config(name: str) -> RegionConfig:
    """Alias of :func:`load`, kept for code carried from the forecaster package."""
    return load(name)


def __getattr__(attr):
    # Lazy constants so `import closedloop_sampling.regions` needs no YAML read.
    lazy = {"MAB_CONFIG": "mab", "CCS_CONFIG": "ccs", "DOPPIO_CONFIG": "mab"}
    if attr in lazy:
        return load(lazy[attr])
    raise AttributeError(attr)


__all__ = [
    "RegionConfig", "RawSource", "GridConfig", "VerticalConfig", "ChannelConfig",
    "LatentConfig", "TimeSplitConfig", "DedupConfig", "ValidationRanges", "Paths",
    "load", "get_config",
]
