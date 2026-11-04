# Closed-Loop Adaptive Sampling on a Generative Ocean Model

Shayesteh Hafezi, Alan Papalia — University of Michigan

[![paper](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org)
[![weights](https://img.shields.io/badge/weights-Hugging_Face-yellow.svg)](https://huggingface.co/shafezi/closed-loop-sampling)
[![artifacts](https://img.shields.io/badge/artifacts-Hugging_Face-yellow.svg)](https://huggingface.co/datasets/shafezi/closed-loop-sampling-artifacts)

## The problem

The ocean is vast, dynamic, and sparsely observed, and a fleet of mobile
robots can only measure a few points of it per cycle. Deciding where those
few measurements should go requires a forecast of the evolving state, a
usable notion of forecast uncertainty, a planner that respects how far each
robot can actually travel through the currents, and a way to fold the
collected measurements back into the forecast. This repository implements and
evaluates a closed loop in which one learned generative ocean model plays all
four roles — forecaster, uncertainty source, planner input, and assimilator —
so that a 25-member analysis ensemble tracks the evolving ocean state from
the samples the fleet collects.

## How it works

<p align="center"><img src="assets/method.png" width="420" alt="Method overview: forecast the next latent state, decode the ensemble, quantify uncertainty and select sampling locations, assimilate observations."/></p>

One six-hour cycle of Alg. 1, orchestrated by
[`src/closedloop_sampling/closed_loop.py`](src/closedloop_sampling/closed_loop.py):

1. **Forecast.** Each of the 25 analysis members is rolled one step in latent
   space by the EDM denoiser, conditioned on its current latent analysis; the
   +6 h prediction is retained
   ([`src/closedloop_sampling/forecast/`](src/closedloop_sampling/forecast/)).
2. **Decode.** The autoencoder decodes the latent ensemble to physical surface
   fields — the target variable for planning and scoring, and the surface
   currents for reachability
   ([`forecast/decode.py`](src/closedloop_sampling/forecast/decode.py)).
3. **Quantify and select.** Ensemble anomalies give the A- or D-optimal
   acquisition score of every reachable wet cell; a greedy assignment places
   the fleet inside the current-advected reachable sets, applying the serial
   square-root planning update between picks
   ([`src/closedloop_sampling/sampling/`](src/closedloop_sampling/sampling/)).
4. **Assimilate.** The noisy point observations at the selected cells are
   pulled into every member by observation-conditioned diffusion sampling,
   and the updated latent analysis starts the next cycle
   ([`src/closedloop_sampling/assimilation/`](src/closedloop_sampling/assimilation/)).

## The two study regions

<p align="center"><img src="assets/usa_overview.png" width="760" alt="The two model domains outlined on a map of the United States: the Mid-Atlantic Bight on the East Coast and the California Current System on the West Coast."/></p>

| | MAB | CCS |
|---|---|---|
| reanalysis | Doppio (Wilkin et al., 2022) | CCSRA31 (Neveu et al., 2016) |
| native grid -> model crop | 106 x 242 -> 104 x 240 | 181 x 186 -> 176 x 184 |
| cell size | ~7.0 km (7004 x 6985 m) | ~8.7 x 11.1 km (anisotropic 0.1 deg grid) |
| store span | hourly, 2007-01 to 2024-12 (157,752 steps) | 6-hourly, 1980-01 to 2019-05 (57,441 steps) |
| test years | 2022-2024 (sealed ICs 2022-2023) | 2016-2019 (sealed ICs 2016-2018) |
| latent shape | 84 x 13 x 30 | 84 x 22 x 23 |

Values from [`configs/regions/mab.yaml`](configs/regions/mab.yaml) and
[`configs/regions/ccs.yaml`](configs/regions/ccs.yaml), the authoritative
region configurations.

## Results

Across 120 test cases (2 regions x 2 variables x 3 fleet sizes x 10 sealed
initial conditions; 5 policies x 3 paired seeds each), uncertainty-driven
sampling generally outperforms static, random and systematic-coverage
baselines while adding little planning overhead: A-optimal removes 38/47/61
percent of the recoverable MAB temperature error at fleet sizes 20/50/200
(final-eight-cycle window); the median full forecast-plan-assimilate cycle
takes 123 s on MAB and 151 s on CCS on a single NVIDIA A40 in bf16, inside a
six-hour observation interval; and across the 1,800-run campaign every
printed number in the paper's three tables regenerates from the released
per-run scores (verified cell by cell, 466/466).

<p align="center"><img src="assets/fig3_skill.png" width="500" alt="Fraction of recoverable error removed per policy, fleet size, region and variable."/></p>

Fraction of recoverable error removed over the final eight cycles (0 =
free-running, 1 = autoencoder floor), mean and one standard deviation across
the 10 sealed ICs — regenerated from the committed scores by
`results/figures/fig3_skill.py`.

<p align="center"><img src="assets/fig6_confidently_wrong.png" width="700" alt="A confidently wrong case: absolute SST error, ensemble spread, and their normalized displacement for the MAB 2022-08-02 window."/></p>

The benefit of uncertainty-driven sampling tracks how well ensemble spread
aligns with forecast error, and the alignment fails on individual dates: in
the MAB 2022-08-02 window above, large errors sit where the ensemble is
confident (normalized displacement well above 1). Table II quantifies this
per IC via the spread-error correlation r; Fig. 6 in the paper is this case.

One full campaign, end to end — all 40 cycles of the MAB SST chain from IC
2022-04-09: the free-running forecast ensemble (top row), each policy's
robots, reachable sets and selected cells on its own ensemble spread (middle),
and the ensemble-mean error each policy leaves behind (bottom).


https://github.com/user-attachments/assets/b940f49d-dea3-40a7-8da8-03e4d5998b55


## Paper-to-code map

| paper element | where | regenerate |
|---|---|---|
| Alg. 1 (the closed loop) | `src/closedloop_sampling/closed_loop.py` | `python experiments/run_closed_loop.py --region mab --ic_datetime 2022-04-09T12:00 --nfleet 50 --strategy a_optimal --rseed 0` |
| Eq. (1)-(2) acquisition scores | `src/closedloop_sampling/sampling/acquisition.py`, `src/closedloop_sampling/sampling/uncertainty.py` | exercised by `tests/test_acquisition.py` |
| greedy assignment + planning update | `src/closedloop_sampling/sampling/selection.py` | parity-tested in `tests/test_selection_parity.py` |
| the five policies | `src/closedloop_sampling/sampling/policies/` | `tests/test_policies.py` |
| reachability under predicted currents | `src/closedloop_sampling/sampling/reachability.py` | `tests/test_reachability.py` |
| observation model (Sec. III-C) | `src/closedloop_sampling/observations.py` | bit-exactness proof in `tests/test_observations.py` |
| observation-conditioned diffusion update | `src/closedloop_sampling/assimilation/guidance.py` | GPU; part of any chain run |
| forecaster (EDM, loop ensemble convention) | `src/closedloop_sampling/forecast/` | equivalence suite `tests/test_equivalence.py` |
| guidance-scale selection (frozen) | `experiments/select_guidance_scale.py`, `configs/experiments/guidance_scales.yaml` | values released; sweep re-runnable on GPU |
| sealed test ICs | `configs/experiments/ics/test_{mab,ccs}.json` | shipped as data (selection inputs were deleted before unsealing) |
| campaign enumeration (2,410 chains) | `experiments/launch_sweep.py` | `python experiments/launch_sweep.py --dry-run` |
| per-run scores and references | `results/scores/` | `python tools/extract_scores.py --source-root <production tree>` (author-side) |
| scoring conventions (window means, skill fraction F, floors, Wilcoxon) | `results/scoring/` | imported by the tables and Fig. 3-4 |
| Table I (campaign score) | `results/tables/table1_rmse.py` | `python results/tables/table1_rmse.py` |
| Table II (per-IC r and Delta) | `results/tables/table2_per_ic.py` | `python results/tables/table2_per_ic.py` |
| Table III (Lawnmower wins) | `results/tables/table3_lawnmower_wins.py` | `python results/tables/table3_lawnmower_wins.py` |
| Fig. 1 (method overview) | `results/figures/fig1_overview.py` | needs `$CLOSEDLOOP_ARTIFACTS` |
| Fig. 2 (ensemble and error) | `results/figures/fig2_ensembles.py` | needs `$CLOSEDLOOP_ARTIFACTS` |
| Fig. 3 (recoverable error removed) | `results/figures/fig3_skill.py` | `python results/figures/fig3_skill.py` (scores only) |
| Fig. 4 (fleet size) | `results/figures/fig4_fleet_size.py` | `python results/figures/fig4_fleet_size.py` (scores only) |
| Fig. 5 (where policies sample) | `results/figures/fig5_locations.py` | `python results/figures/fig5_locations.py` (scores only) |
| Fig. 6 (confidently wrong) | `results/figures/fig6_confidently_wrong.py` | needs `$CLOSEDLOOP_ARTIFACTS` |
| loop video / teaser | `results/figures/teaser_video.py` | needs `$CLOSEDLOOP_ARTIFACTS` + ffmpeg |
| spread-error correlation r | `results/scores/references/spread_error_r.csv` | released values (recompute needs the free-running latents in the artifacts dataset) |
| rerun-tolerance floors | `results/scores/references/tolerance.csv` | recomputed from the replicate chains and asserted by `tools/extract_scores.py` |
| golden verification vs the submitted tables | `tools/verify_tables_vs_tex.py` | author-side (needs the submission archive); result: 466/466 cells match |

## Installation

```
git clone https://github.com/UMich-RobotExploration/closed-loop-sampling.git
cd closed-loop-sampling
pip install -e .[figures,dev]         # production-freeze pins + matplotlib/cmocean + pytest
python tools/fetch_upstream.py        # pins LaDCast into third_party/ (verified)
pip install --no-deps third_party/ladcast_upstream   # makes `import ladcast` resolve
export CLOSEDLOOP_WEIGHTS=...  CLOSEDLOOP_ARTIFACTS=...
python tools/download_weights.py
python tools/download_artifacts.py --bundles-only   # ~320 MB; omit the flag for all 2 GB
pytest                                # CPU subset; weights probes run where weights exist
```

## Data

Both reanalyses are public products; nothing here mirrors them.

- MAB: Doppio reanalysis (Wilkin et al., 2022), ~3.3 TB of raw NetCDF via
  `datasets/download_mab.sh`, preprocessed to a 940 GB store.
- CCS: CCSRA31 (Neveu et al., 2016), ~2.5 TB staged over OPeNDAP via
  `datasets/download_ccs.py`, preprocessed to a 381 GB store.

The small path: the released **truth bundles** (the 10 sealed test windows per
region: 41 six-hour states of SST, SSS and surface currents, ~110 MB/region)
and **latent bundles** (the same windows as raw fp16 latents, ~30 MB/region)
let the closed loop, the scoring and the figures run without any store. Truth
bundles are bitwise equal to the stores; latent bundles carry ~5e-4 relative
fp16 quantization (see AS-RUN NOTES). Redistribution terms are stated in the
dataset card and `datasets/README.md`.

## Weights

[huggingface.co/shafezi/closed-loop-sampling](https://huggingface.co/shafezi/closed-loop-sampling):
safetensors EMA checkpoints for both regions — autoencoder 256,388,460
parameters, denoiser 374,938,452 parameters — each with its as-run training
configuration and provenance metadata, plus seed-1 denoiser extras that no
paper result uses. The equivalence suite must reproduce the recorded probe
statistics (1e-5) and rounded-array hashes with these files.

## Reproduction tiers

**Tier 1 — tables and score figures from the released scores (minutes, CPU).**
```
python results/tables/table1_rmse.py
python results/tables/table2_per_ic.py
python results/tables/table3_lawnmower_wins.py
python results/figures/fig3_skill.py
python results/figures/fig4_fleet_size.py
python results/figures/fig5_locations.py
pytest tests/test_tables_regenerate.py   # regeneration equals the committed goldens
```
No downloads needed: `results/scores/` is committed.

**Tier 2 — run closed-loop chains on the released bundles (one GPU;
1.4 h per MAB chain, 1.7 h per CCS chain).**
```
python tools/download_weights.py && python tools/download_artifacts.py
python experiments/run_closed_loop.py --region mab \
    --ic_datetime 2022-04-09T12:00 --nfleet 50 --strategy a_optimal --rseed 0
python experiments/run_free_running.py --region mab --target temp
python results/figures/fig1_overview.py      # and fig2 / fig6 / teaser_video
```
bf16 generation is not bit-stable across GPU models and the bundles are fp16,
so a rerun reproduces the released records statistically (cycle-level RMSE
magnitudes and gaps), not bitwise.

**Tier 3 — the full campaign, or retraining (cluster scale).**
```
python experiments/launch_sweep.py           # emits all 2,410 chain commands
```
The 1,800-run campaign matrix costs about 2,800 GPU-hours on A40-class
hardware (plus roughly 800 more for the fleet sweep, replicate floors and
free-running references). Retraining from scratch needs the multi-terabyte
raw mirrors and preprocessed stores above (`datasets/README.md`, then
`training/README.md`); retrained weights are new draws of the optimization and
**will not reproduce the reference hashes** — those identify the released
checkpoints only.

## As-run notes

1. The released `guidance_scales.yaml` carries the selected value **0.0217**
   for CCS SSS at n = 20/35 (a second-stage endpoint extension of the sweep
   grid); the paper prose rounds it to 2.2e-2.
2. The guidance scales were tuned under a 144 h single-shot conditioning
   protocol on 15 tuning-year ICs per region, then frozen and applied in the
   6 h closed loop.
3. D-optimal ranks candidates by the marginal-variance surrogate, which is
   argmax-equivalent to the paper's Eq. (1) information gain; per-placement
   scores stored in the records are variances, not log-gains.
4. The executed observation-noise literals are sigma_SST = 0.2 degC and
   sigma_SSS = 0.2 * 3.6912 / 5.1723 = 0.14272954004988112 psu (equal relative
   precision anchored at SST; both regions).
5. The initial ensemble is N = 25 identical copies of the normalized IC
   latent; member spread is created by the first generative roll.
6. The production SST code path carried a numerically inert legacy SSS misfit
   term (y = 0, sigma = 1e3, about 4e-8 relative weight), kept there only for
   bit-exact regression against an earlier pipeline; this release implements
   the target-only misfit and documents the omission in
   `src/closedloop_sampling/assimilation/guidance.py`.
7. `scale_eff` recorded in every run file is the launcher's `--scale` verbatim:
   a fixed guidance scale with a constant schedule, no per-cycle adaptation and
   no observation-count normalization.

**Limitations.** The loop is evaluated against the reanalyses it was trained
on, with synthetic point observations of one surface variable per campaign;
ensemble spread can diverge from forecast error on individual dates (Fig. 6 and
Table II quantify this), and no real-vehicle constraints beyond reachability
are modeled.

## Citations

```bibtex
@inproceedings{hafezi2027closedloop,
  author = {Hafezi, Shayesteh and Papalia, Alan},
  title  = {Closed-Loop Adaptive Sampling on a Generative Ocean Model},
  year   = {2027},
  note   = {Under review},
}
@misc{zhuang2025ladcast,
  author        = {Zhuang, Yilin and Duraisamy, Karthik},
  title         = {{LaDCast}: A Latent Diffusion Model for Medium-Range Ensemble Weather Forecasting},
  year          = {2025},
  howpublished  = {arXiv preprint arXiv:2506.09193},
}
@article{wilkin2022doppio_reanalysis,
  author  = {Wilkin, John and Levin, Julia and Moore, Andrew and Arango, Hernan and L{\'o}pez, Alexander and Hunter, Elias},
  title   = {A Data-Assimilative Model Reanalysis of the {U.S.} {Mid Atlantic Bight} and {Gulf of Maine}},
  journal = {Progress in Oceanography},
  volume  = {209},
  pages   = {102919},
  year    = {2022},
}
@article{neveu2016ccsra,
  author  = {Neveu, Emilie and Moore, Andrew M. and Edwards, Christopher A. and Fiechter, J{\'e}r{\^o}me and Drake, Patrick and Crawford, William J. and Jacox, Michael G. and Nuss, Emma},
  title   = {An Historical Analysis of the {California Current} Circulation Using {ROMS} {4D-Var}: System Configuration and Diagnostics},
  journal = {Ocean Modelling},
  volume  = {99},
  pages   = {133--151},
  year    = {2016},
}
```

## License

MIT (code, released weights, and original released data). The forecaster
architecture derives from LaDCast, which is dual-licensed (Apache-2.0 for code
adapted from third-party sources, MIT otherwise); the upstream code is **not
redistributed** here — `tools/fetch_upstream.py` fetches the pinned commit into
the gitignored `third_party/` and verifies it against `configs/upstream.lock`.
