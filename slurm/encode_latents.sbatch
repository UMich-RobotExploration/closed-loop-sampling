#!/bin/bash
#SBATCH --job-name=cls_encode
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=<LOG_DIR>/cls_encode_%A_%a.out

# Fill the <PLACEHOLDERS> for your site. Export the environment roots first:
#   CLOSEDLOOP_DATA_ROOT, CLOSEDLOOP_WEIGHTS, CLOSEDLOOP_RUNS
set -euo pipefail
cd <REPO_DIR>
python training/encode_latents.py --region <REGION>
