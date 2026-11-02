# Closed-Loop Adaptive Sampling on a Generative Ocean Model

Shayesteh Hafezi, Alan Papalia

Code for the paper. A fleet of simulated vehicles repeatedly samples a
regional ocean forecast produced by a latent-diffusion model: each six-hour
cycle rolls the analysis ensemble forward one step, computes current-advected
reachable sets for every vehicle, scores candidate cells with an acquisition
score on the forecast ensemble (A- or D-optimal, with a serial square-root
planning update between placements), takes noisy point observations of the
target variable, and regenerates the analysis ensemble with observation-guided
diffusion at a fixed guidance scale. Two regions (Mid-Atlantic Bight and
California Current System), two target variables (surface temperature and
salinity), five sampling policies, 40 cycles per chain.

Sections below are placeholders; they are filled in the results round.

## Installation
(to be written)

## Weights and data
(to be written)

## Reproducing the paper
(to be written)

## Repository layout
(to be written)

## Citation
See CITATION.cff.
