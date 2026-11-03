---
license: other
license_name: mixed-sources
license_details: >-
  Derived subsets of two public ocean reanalyses, redistributed with
  attribution for reproducibility; see "Sources and terms". Original content
  (run records, decoded model fields, latents) under MIT.
tags:
- ocean-modeling
- ensemble-forecasting
- robotics
- benchmark
---

# Closed-Loop Adaptive Sampling on a Generative Ocean Model — artifacts

The evaluation artifacts behind the paper "Closed-Loop Adaptive Sampling on a
Generative Ocean Model" (Hafezi and Papalia): everything needed to re-run the
closed loop on the sealed test windows and to regenerate the paper's decoded
figures and loop video, without the multi-terabyte reanalysis stores.

## Contents

| path | size | content |
|---|---|---|
| `truth_bundles/{mab,ccs}.npz` | 111 / 113 MB | per sealed test IC (10 per region), the 41 six-hour states of SST, SSS and surface u/v on the model crop, the wet mask and the metric cell sizes; float32, bitwise equal to the preprocessed reanalysis stores |
| `latent_bundles/{mab,ccs}.npz` | 25 / 32 MB | the same 41-state windows of the RAW truth latents, fp16; the closed loop reads the initial-condition latent and the per-cycle autoencoder floor from these |
| `figure_fields/` | 59 MB | decoded-field extracts behind Figs. 1, 2 and 6 and the loop-video member decodes; see its README |
| `render_dumps/<slice>/<policy>.npz` | 298 MB | per-cycle truth, ensemble mean, spread, robot positions, reachable sets and selected cells of the six dumped chains per slice (MAB and CCS showcase ICs, both variables, fleet 50) |
| `free_running_latents/` | 1.6 GB | per-cycle prior member latents (fp16, N = 25) of the 20 free-running chains, the inputs behind the spread-error diagnostics and the ensemble figures |
| `records/raw_run_records.tar.zst` | 3 MB | the 2,318 raw run-record JSONs of the campaigns (40 cycles each: prior/posterior RMSE, autoencoder floor, observation count) |

The per-run scores, the per-cycle score tables, the tolerance floors and the
Fig. 5 sampling positions are committed INSIDE the code repository
(`results/scores/`), extracted from these same records and dumps by an
auditable script.

Precision notes: truth bundles are float32 and bitwise equal to the stores
(spot-checked at export); latent bundles are fp16 (about 5e-4 relative
quantization), so bundle-backed chains reproduce the production records
statistically, not bitwise.

## Sources and terms

- **MAB (Doppio reanalysis).** Wilkin et al. (2022), Progress in Oceanography
  209:102919. Distributed by Rutgers University (THREDDS) and through the NOAA
  Open Data Dissemination program, whose terms make the data open to the public
  with attribution requested and no implied NOAA endorsement. The bundles here
  are small derived subsets (four surface fields at 10 dates plus their 240-hour
  windows, on the model crop), redistributed with attribution under those terms.
- **CCS (CCSRA31 reanalysis).** Neveu et al. (2016), Ocean Modelling 99:133-151;
  see also Moore et al. (2019) for the broader ROMS 4D-Var synthesis.
  Distributed by the UC Santa Cruz ocean modeling group (OPeNDAP/THREDDS). The
  access pages publish no explicit data license; the group asks that the system
  papers be referenced. Accordingly this dataset redistributes only a small
  derived subset (the same four surface fields at 10 dates) for
  reproducibility, with attribution; the reanalysis itself is not mirrored, and
  the code repository's download scripts point at the original servers.
- Decoded fields, latents and run records are model outputs created by the
  authors (MIT).

## Use

```
export CLOSEDLOOP_ARTIFACTS=/path/to/artifacts
python tools/download_artifacts.py            # or --bundles-only (about 320 MB)
python experiments/run_closed_loop.py --region mab \
    --ic_datetime 2022-04-09T12:00 --nfleet 50 --strategy a_optimal --rseed 0
```

## Citation

```bibtex
@inproceedings{hafezi2027closedloop,
  author = {Hafezi, Shayesteh and Papalia, Alan},
  title  = {Closed-Loop Adaptive Sampling on a Generative Ocean Model},
  year   = {2027},
  note   = {Under review},
}
@article{wilkin2022doppio_reanalysis,
  author  = {Wilkin, John and Levin, Julia and Moore, Andrew and Arango, Hernan
             and L{\'o}pez, Alexander and Hunter, Elias},
  title   = {A Data-Assimilative Model Reanalysis of the {U.S.} {Mid Atlantic
             Bight} and {Gulf of Maine}},
  journal = {Progress in Oceanography},
  volume  = {209},
  pages   = {102919},
  year    = {2022},
}
@article{neveu2016ccsra,
  author  = {Neveu, Emilie and Moore, Andrew M. and Edwards, Christopher A. and
             Fiechter, J{\'e}r{\^o}me and Drake, Patrick and Crawford,
             William J. and Jacox, Michael G. and Nuss, Emma},
  title   = {An Historical Analysis of the {California Current} Circulation
             Using {ROMS} {4D-Var}: System Configuration and Diagnostics},
  journal = {Ocean Modelling},
  volume  = {99},
  pages   = {133--151},
  year    = {2016},
}
```
