#!/bin/bash
# Optional SLURM header — a placeholder TEMPLATE only. Fill in the
# placeholders before `sbatch datasets/download_mab.sh`, or just run it
# directly: `bash datasets/download_mab.sh` (the #SBATCH lines are then
# ordinary comments). `mkdir -p logs` first if submitting via sbatch.
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_CPU_PARTITION>
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
# 20h: measured wallclock ~11h for the full 3.3 TB at an observed
# ~86 MB/s (S3 egress + NFS-write ceiling; concurrency-saturated)
# -> ~2x margin. `aws s3 sync` is resumable regardless.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>
#SBATCH --output=logs/s3_sync_%j.out
#SBATCH --error=logs/s3_sync_%j.err
#SBATCH --job-name=mab_s3_sync

# Bulk S3 mirror of the full Doppio v3 `his/` archive (the MAB raw staging).
# Source : s3://noaa-ioos-roms-doppio-v3-pds/his/  (public, anon)
# Dest   : $CLOSEDLOOP_DATA_ROOT/mab/his_raw
# 2191 NetCDF (~3.33 TB) + 1 kerchunk .zarr ref (~226 MB; harmless,
# useful locally). `aws s3 sync` is idempotent + resumable: re-running
# this script re-syncs only missing/changed objects (no data loss on
# interrupt / timeout / resubmit).
#
# Requires the AWS CLI on PATH. Only anonymous access is used
# (--no-sign-request); no credentials are needed.

set -euo pipefail

: "${CLOSEDLOOP_DATA_ROOT:?export CLOSEDLOOP_DATA_ROOT=/path/to/data first (see README 'Data layout')}"

TARGET="$CLOSEDLOOP_DATA_ROOT/mab/his_raw"
mkdir -p "$TARGET"

echo "=== START: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "host        : $(hostname)"
echo "aws         : $(command -v aws) ($(aws --version 2>&1))"
echo "Source      : s3://noaa-ioos-roms-doppio-v3-pds/his/"
echo "Target      : $TARGET"

# Parallelism: 16 concurrent requests, no S3 accel. (No max_bandwidth
# line: awscli 1.x REJECTS `max_bandwidth 0` at runtime -- "Invalid
# rate: 0", exit 255 -- so omit it entirely; absent => unlimited, the
# intended behaviour. Also defensively clear any stale value that a
# prior run may have persisted into the shared ~/.aws/config.)
sed -i '/max_bandwidth/d' "$HOME/.aws/config" 2>/dev/null || true
aws configure set default.s3.max_concurrent_requests 16
aws configure set default.s3.use_accelerate_endpoint false

aws s3 sync --no-sign-request \
    --cli-read-timeout 0 \
    --cli-connect-timeout 60 \
    s3://noaa-ioos-roms-doppio-v3-pds/his/ \
    "$TARGET"

echo "=== SYNC COMPLETE: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# ---- Verification ----
FILE_COUNT=$(find "$TARGET" -name "*.nc" | wc -l)
TOTAL_SIZE=$(du -sb "$TARGET" | cut -f1)
TOTAL_GB=$((TOTAL_SIZE / 1024 / 1024 / 1024))

echo "NetCDF files : $FILE_COUNT (expected 2191)"
echo "Total size   : $TOTAL_SIZE bytes (~${TOTAL_GB} GB; expected ~3334 GB"
echo "               incl. the ~226 MB doppio_fwd_plusdye.zarr kerchunk ref)"

if [[ "$FILE_COUNT" -ne 2191 ]]; then
    echo "WARNING: NetCDF file count mismatch (got $FILE_COUNT, expected 2191)"
    echo "         sync may be incomplete (timeout?) -- rerun to resume."
    exit 1
fi

echo "=== VERIFICATION PASSED: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
