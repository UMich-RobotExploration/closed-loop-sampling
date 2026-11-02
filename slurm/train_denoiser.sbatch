#!/bin/bash
#SBATCH --job-name=cls_train_den
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=<LOG_DIR>/cls_train_den_%A_%a.out

# Fill the <PLACEHOLDERS> for your site. Export the environment roots first:
#   CLOSEDLOOP_DATA_ROOT, CLOSEDLOOP_WEIGHTS, CLOSEDLOOP_RUNS
set -euo pipefail
cd <REPO_DIR>
python training/train_denoiser.py --region <REGION> --seed <0|1> --config configs/training/<REGION>/denoiser.yaml
