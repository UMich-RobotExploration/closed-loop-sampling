# configs/experiments/ics/

Initial-condition lists, shipped as data.

- `test_{mab,ccs}.json` — the SEALED test ICs (10 per region, canonical
  order), verbatim from the campaign's sealing records, including the full
  sealing metadata (selection criteria, candidate list, tercile bounds,
  subset rules). The selection rule was truth-free (ensemble-only spread
  statistics at 144 h) and pre-registered. The quarantined candidate prior
  pools that fed the selection were deleted after sealing, so the selection
  is NOT recomputable: these lists ship as data, not as the output of a
  script in this repository.
- `validation_{mab,ccs}.json` — the 15 tuning-year ICs per region on which
  the guidance scales were selected and frozen (MAB: 2021; CCS: 2015 — each
  region's validation year, strictly before its sealed test window). Draw
  order preserved; the first entry is the sweep's noise-floor repeat cell.

Subset rule used by the campaigns (recorded in the sealed files): the n = 200
temperature slices use the first 5 ICs of the canonical order; salinity
slices use the full 10.
