#!/bin/bash
#SBATCH --job-name=cls_loop
#SBATCH --account=<ACCOUNT>
#SBATCH --partition=<GPU_PARTITION>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=<LOG_DIR>/cls_loop_%A_%a.out
#SBATCH --array=0-<N_JOBS-1>
# Fill the <PLACEHOLDERS> for your site. Export the environment roots first:
#   CLOSEDLOOP_DATA_ROOT, CLOSEDLOOP_WEIGHTS, CLOSEDLOOP_RUNS
set -euo pipefail
cd <REPO_DIR>
# One chain per array task, from a job list made by experiments/launch_sweep.py:
ARGS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" <JOB_LIST_TSV>)
python experiments/run_closed_loop.py $ARGS
