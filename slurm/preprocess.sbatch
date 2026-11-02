#!/bin/bash
#SBATCH --job-name=cls_preproc
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<CPU_PARTITION>
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=<LOG_DIR>/cls_preproc_%j.out
# Fill the <PLACEHOLDERS> for your site. CPU-only. Export CLOSEDLOOP_DATA_ROOT.
set -euo pipefail
cd <REPO_DIR>
python datasets/preprocess_<REGION>.py
python datasets/validate_store.py --region <REGION>
