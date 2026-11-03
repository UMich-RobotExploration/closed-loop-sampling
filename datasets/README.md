# datasets/

Acquisition and preprocessing for the two regions. Order of operations:

MAB (Doppio v3, public S3 mirror, ~3.3 TB raw):
  1. `bash download_mab.sh`                      — mirror the raw NetCDF files
  2. `python preprocess_mab.py`                  — destagger, crop, land-zero -> chunked store
  3. `python validate_store.py --region mab`     — store integrity checks
  4. `python compute_normalization_stats.py --region mab`

CCS (UCSC CCSRA31 over OPeNDAP, ~2.5 TB staged):
  1. `python download_ccs.py`                    — resumable per-file fetch
  2. `python build_ccs_timeline.py`              — canonical de-overlapped timeline (later-file-wins)
  3. `python preprocess_ccs.py`                  — transform onto the timeline
  4. `python clip_ccs_temp_artifacts.py`         — clip impossible temperature cells (audited)
  5. `python validate_store.py --region ccs`
  6. `python compute_normalization_stats.py --region ccs`

Release extracts:
  - `python export_wet_mask.py --region <r>`     — mask_rho on the model crop
  - `python export_truth_bundle.py --region <r>` — sealed-IC surface truth bundle
    (41 six-hour states x {SST, SSS, surface u, v} + mask + metric cell sizes,
    ~0.1 GB/region; the closed loop and scoring read this instead of the
    multi-terabyte store)
  - `python export_latent_bundle.py --region <r>` — sealed-IC RAW truth latents
    (the same 41-state windows, fp16, ~30 MB/region; the IC latent and the
    per-cycle autoencoder floor read this instead of the latents store)

Redistribution of the bundles: both reanalyses are public products. Doppio is
distributed by Rutgers University and through the NOAA Open Data Dissemination
program (open to the public, attribution requested); CCSRA31 is served by the
UC Santa Cruz ocean modeling group, whose access pages publish no explicit data
license. The released bundles are therefore small DERIVED subsets only — four
surface fields on the model crop over the 10 sealed test windows per region —
redistributed with attribution for reproducibility (citations in the dataset
card); neither reanalysis is mirrored, and the download scripts above point at
the original servers.

`_preprocess_lib.py` holds the stage functions (unchanged from the production
pipeline); the entry scripts above are thin region/stage bindings over it.
All paths resolve via $CLOSEDLOOP_DATA_ROOT (see src/closedloop_sampling/paths.py).
