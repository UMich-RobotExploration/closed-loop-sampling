# training/

As-run training pipeline for the forecaster (one autoencoder + one denoiser
per region). Recipes are configs/training/<region>/{autoencoder,denoiser}.yaml
— the production runs' configurations verbatim, with region and year ranges
first-class.

  1. `python train_autoencoder.py --region <r> --config ../configs/training/<r>/autoencoder.yaml`
     30 pretrain epochs; the decoder-finetune stage is disabled (non-functional
     under DDP); selection is best-on-validation; inference uses EMA weights.
  2. `python encode_latents.py --region <r>`
     Streams the preprocessed store through the trained encoder; writes the
     latents store and the train-split latent normalization stats.
  3. `python train_denoiser.py --region <r> --seed <0|1> --config ../configs/training/<r>/denoiser.yaml`
     15 epochs, EDM objective (sigma in [0.002, 80], sigma_data 0.5, lognormal
     (-1.2, 1.2)); best-on-validation; EMA at inference.

Shared pieces: `dataset.py` (pixel/latent datasets + sliding-window sampler),
`trainer.py` (LR schedule), `ema.py` (EMA constructor with the as-run
constants). Entry scripts carry their own checkpoint save/prune logic (the two
recipes checkpoint differently); core loops are unchanged from the production
pipeline. SLURM templates: ../slurm/.
