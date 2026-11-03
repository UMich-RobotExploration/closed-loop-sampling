# experiments/

Entry points for running the published experiments. All constants (ensemble
size 25, 40 six-hour cycles, seeds, sigma_obs, guidance scales) are frozen in
`configs/experiments/`; the scripts only select among them.

- `run_closed_loop.py` — ONE chain of Alg. 1: (region, IC, target variable,
  fleet size, policy, seed), 40 cycles on GPU. Truth and latents resolve from
  the released bundles ($CLOSEDLOOP_ARTIFACTS first, then the data root;
  `--truth store` reproduces the store-backed production path). `--scale`
  defaults to the frozen per-(region, variable, fleet) guidance scale;
  `--start_lead` must be 0 (the published convention).
- `run_free_running.py` — the no-sampling reference chains for one
  (region, variable): the free-running error curves that anchor the skill
  fraction; `--dump-latents` saves the per-cycle member latents.
- `launch_sweep.py` — enumerates the full campaign: 1,800 main + 450
  fleet-sweep + 120 replicate + 40 free-running chain commands (2,410 lines).
- `select_guidance_scale.py` — the frozen guidance-scale sweep machinery
  (7-point grid, pre-registered stability screens, two stages); the selected
  values ship in `configs/experiments/guidance_scales.yaml`.
- `truth.py` — truth and latent access: released-bundle readers
  (`TruthBundle`, `LatentBundle`, resolution order in `find_bundle`) and the
  store-backed fallbacks.

Chains write `<CLOSEDLOOP_RUNS>/chains/<region>_<ICTAG>_<target>_n<fleet>_<policy>_s<seed>.json`
with the production record schema (per-cycle prior/posterior RMSE, autoencoder
floor, observation count) plus chosen-cell logging and per-cycle wall-clock.
