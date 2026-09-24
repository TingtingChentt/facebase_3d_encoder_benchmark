#!/bin/bash
#SBATCH --job-name=fb3d_m2_controls
#SBATCH --partition=defq
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-7
#SBATCH --output=logs/m2_controls_%A_%a.out
#SBATCH --error=logs/m2_controls_%A_%a.err

set -euo pipefail

REPO="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
SCRIPT=$REPO/facebench/preprocessing/m2_preprocess_controls.py
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

echo "M2 control preprocessing: worker=$SLURM_ARRAY_TASK_ID/8"
echo "Node: $(hostname), Array task: $SLURM_ARRAY_TASK_ID"
date

"${FACEBENCH_PYTHON:-python3}" "$SCRIPT" \
    --dataset all \
    --worker-id "$SLURM_ARRAY_TASK_ID" \
    --n-workers 8

echo "Worker $SLURM_ARRAY_TASK_ID done at $(date)"
