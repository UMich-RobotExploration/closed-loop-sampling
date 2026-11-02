#!/bin/bash
#SBATCH --job-name=cls_train_ae
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=72:00:00
#SBATCH --output=<LOG_DIR>/cls_train_ae_%A_%a.out

# Fill the <PLACEHOLDERS> for your site. Export the environment roots first:
#   CLOSEDLOOP_DATA_ROOT, CLOSEDLOOP_WEIGHTS, CLOSEDLOOP_RUNS
set -euo pipefail
cd <REPO_DIR>
python training/train_autoencoder.py --region <REGION> --config configs/training/<REGION>/autoencoder.yaml
