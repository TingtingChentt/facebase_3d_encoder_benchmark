#!/bin/bash
#SBATCH --job-name=fb3d_ply2obj
#SBATCH --partition=defq
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=logs/ply_to_obj_%A_%a.out
#SBATCH --error=logs/ply_to_obj_%A_%a.err

set -euo pipefail

REPO="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
SCRIPT=$REPO/facebench/preprocessing/convert_ply_to_obj.py
LOG_DIR=$REPO/logs
mkdir -p "$LOG_DIR"

WORKER_ID=$SLURM_ARRAY_TASK_ID
N_WORKERS=8

echo "PLY→OBJ conversion: worker $WORKER_ID/$N_WORKERS"
echo "Node: $(hostname), Array task: $SLURM_ARRAY_TASK_ID"
date

"${FACEBENCH_PYTHON:-python3}" "$SCRIPT" \
    --dataset both \
    --worker-id "$WORKER_ID" \
    --n-workers "$N_WORKERS" \
    --n-procs 8

echo "Worker $WORKER_ID done at $(date)"
