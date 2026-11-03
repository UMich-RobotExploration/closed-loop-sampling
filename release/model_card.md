---
license: mit
tags:
- ocean-modeling
- diffusion-models
- ensemble-forecasting
- data-assimilation
- robotics
library_name: pytorch
---

# Closed-Loop Adaptive Sampling on a Generative Ocean Model — weights

Regional latent-diffusion ocean forecasters for two ROMS reanalysis domains:
the Mid-Atlantic Bight (MAB, from the Doppio reanalysis) and the California
Current System (CCS, from CCSRA31). Each region has a deep compressive
autoencoder (256,388,460 parameters) and an EDM-trained latent denoiser
(374,938,452 parameters). These are the exact weights behind every number in
the paper "Closed-Loop Adaptive Sampling on a Generative Ocean Model"
(Hafezi and Papalia): the closed loop forecasts an analysis ensemble in latent
space, decodes it, plans sampling locations from ensemble statistics, and
assimilates observations by observation-conditioned diffusion sampling.

This is a research model, not an operational forecast. It is trained on
reanalysis fields of two specific regional domains and evaluated against the
same reanalyses; it has no skill guarantee on real-time data and must not be
used for navigation or safety decisions.

## Files

```
mab/dcae/        ema.safetensors, meta.json, training_config.yaml
mab/denoiser/seed0/            same triple
mab/region.yaml  the complete region configuration (grid, crop, channels)
ccs/             same layout
extras/denoiser_seed1_{mab,ccs}/   trained identically with seed 1;
                                   not used in any paper result
```

Every checkpoint is the exponential-moving-average (EMA) weights of the
best-on-validation checkpoint, re-keyed by parameter name and converted to
safetensors; `meta.json` records the epoch, validation loss, parameter count
and source training run. The as-run training configuration sits beside each
checkpoint.

## Training, in brief

Per region: the autoencoder compresses 84 input channels (20 sigma-levels each
of u, v, temperature and salinity, the 2-D dynamic fields zeta/ubar/vbar, and
the binary wet mask) on the model crop
(MAB 104 x 240, CCS 176 x 184) to an 84-channel latent (MAB 13 x 30,
CCS 22 x 23); the denoiser is trained with the EDM objective on four-state
latent sequences and generates with a 20-step Karras schedule (rho 7, Heun,
sigma 0.002-80, sigma_data 0.5). Inference for the paper ran in bf16 on a
single NVIDIA A40. Full hyperparameters are in each `training_config.yaml`.

## Verification

The code repository's equivalence suite loads these files and must reproduce
the recorded probe statistics (tolerance 1e-5) and rounded-array SHA-256 hashes
on CPU:

| probe | sha256 (outputs rounded to 1e-4) |
|---|---|
| MAB denoiser, fixed-noise probe | `974408e42ee9e5247ffab7fba85031cb0bd633c09726b5cfd73c106f1cd08056` |
| MAB autoencoder, decode probe | `ef675e85f337909433474cdb975bfce693693f02703fe645040ee93fb88bc6c9` |
| MAB autoencoder, encode probe | `c2d2311393633d82ed3a88fd3a645a0b8f60f8420daa926b2a57d585501fe5e4` |
| CCS denoiser, fixed-noise probe | `1b19d9d1119d6e398897a63028a2a8d86608ab403f4b124c0355f0c8ba892521` |

Retrained weights will not reproduce these hashes; they identify the released
checkpoints.

## Lineage and license

The forecaster architecture derives from LaDCast (Zhuang and Duraisamy, 2025;
`github.com/tonyzyl/ladcast`), adapted to regional ocean domains. LaDCast is
dual-licensed: code adapted from third-party sources under Apache-2.0, original
content under MIT. The code repository pins the upstream commit and fetches it
at setup rather than redistributing it. These weights, trained from
scratch on the reanalysis data by the authors, are released under MIT.

Training data: MAB from the Doppio reanalysis (Wilkin et al., 2022,
Progress in Oceanography 209:102919; distributed by Rutgers University and via
the NOAA Open Data Dissemination program); CCS from CCSRA31 (Neveu et al.,
2016, Ocean Modelling 99:133-151; distributed by the UC Santa Cruz ocean
modeling group). Neither reanalysis is redistributed here.

## Use

```
pip install -e .            # the code repository
export CLOSEDLOOP_WEIGHTS=/path/to/weights
python tools/download_weights.py
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
@misc{zhuang2025ladcast,
  author        = {Zhuang, Yilin and Duraisamy, Karthik},
  title         = {{LaDCast}: A Latent Diffusion Model for Medium-Range
                   Ensemble Weather Forecasting},
  year          = {2025},
  howpublished  = {arXiv preprint arXiv:2506.09193},
}
```
