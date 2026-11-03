# results/

The released evaluation: scores, scoring conventions, tables and figures.
Everything downstream of `scores/` reads only `scores/` (or the artifacts
dataset for decoded fields) — never the production trees.

- `scores/` — the committed scores release, built by `tools/extract_scores.py`:
  `manifest.csv` (one row per raw record, census-reconciled to 2,318),
  `runs.csv` (2,250 main + fleet-sweep runs: final-window scores, frozen scale,
  wall-clock where a log survives), `runs_per_cycle_<region>_<variable>.csv.gz`
  (90,000 rows), `references/` (free-running curves, autoencoder floor,
  rerun-tolerance floors, spread-error r) and `locations/` (the Fig. 5
  positions and drawing context).
- `scoring/` — the paper's scoring conventions as code: final-window means and
  the skill fraction F (`metrics.py`), the replicate-spread floors
  (`tolerance.py`), the paired Wilcoxon test and the separation rule
  (`stats_tests.py`).
- `tables/` — `table{1,2,3}_*.py` regenerate the paper's tables (CSV + LaTeX
  body); `expected/` holds the committed goldens, and
  `tests/test_tables_regenerate.py` asserts regeneration equals them byte for
  byte. Every printed number was verified against the submitted tables
  (466/466 cells).
- `figures/` — `fig{1..6}_*.py` and `teaser_video.py`; Figs. 3-5 regenerate
  from `scores/` alone, Figs. 1, 2, 6 and the video read the artifacts dataset
  ($CLOSEDLOOP_ARTIFACTS). `_shore.py` is the view-from-shore map convention,
  `_style.py` the shared style.
- `out/` — gitignored scratch for generated outputs and job logs.
