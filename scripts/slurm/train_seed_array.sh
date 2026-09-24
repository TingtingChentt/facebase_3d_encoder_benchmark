#!/bin/bash
# ---------------------------------------------------------------------------
# TRAINING-SEED sweep — 5 tasks x 4 encoders x 5 train seeds = 100 runs.
# thread: facebase3d-trainseed-2026-08-31
#
# Regenerate the joblist with facebench/eval/make_trainseed_jobs.py.
# Each array index runs one line of trainseed_joblist.txt.
#
# The data-split seed is pinned to 42 on every line, so all 100 runs are
# scored on the IDENTICAL test subjects and the spread across --train-seed is
# retraining variance alone. Do not "fix" that to vary the split.
# ---------------------------------------------------------------------------
#SBATCH --job-name=fb3d_tseed
#SBATCH --partition=gpu-xe9680q
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1-100%8
#SBATCH --output=runs/_trainseed_logs/tseed_%A_%a.out
#SBATCH --error=runs/_trainseed_logs/tseed_%A_%a.err

set -euo pipefail

JOBLIST="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"/facebench/eval/trainseed_joblist.txt
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")

if [ -z "$CMD" ]; then
    echo "No job at line ${SLURM_ARRAY_TASK_ID} of $JOBLIST" >&2
    exit 1
fi

echo "=== array task ${SLURM_ARRAY_TASK_ID} ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CMD: $CMD"
date

eval "$CMD"

echo "Done at $(date)"
