#!/bin/bash
#SBATCH --job-name=fb3d_m1_preprocess
#SBATCH --partition=gpuq
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=logs/m1_preprocess_%A_%a.out
#SBATCH --error=logs/m1_preprocess_%A_%a.err

set -euo pipefail

REPO="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
SCRIPT=$REPO/facebench/preprocessing/preprocess.py
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

# 8-way sharding: 4 workers for OFC (arrays 0-3), 4 for syndrome (arrays 4-7)
WORKER_ID=$((SLURM_ARRAY_TASK_ID % 4))
N_WORKERS=4

if [ "$SLURM_ARRAY_TASK_ID" -lt 4 ]; then
    EXP="ofc"
else
    EXP="syndrome"
fi

echo "Starting M1 preprocessing: exp=$EXP worker=$WORKER_ID/4"
echo "Node: $(hostname), Array task: $SLURM_ARRAY_TASK_ID"
date

"${FACEBENCH_PYTHON:-python3}" "$SCRIPT" \
    --exp "$EXP" \
    --worker-id "$WORKER_ID" \
    --n-workers "$N_WORKERS"

echo "Worker $WORKER_ID ($EXP) done at $(date)"
