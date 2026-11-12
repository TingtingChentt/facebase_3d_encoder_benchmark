#!/bin/bash
# Train every reported model: 10 task families x 5 train seeds x 4 encoders = 200 runs.
# Regenerate the joblist first: python facebench/training/make_jobs.py
# The data-split seed is pinned to 42 on every line, so all five seeds of a family
# are scored on IDENTICAL test subjects and their spread is retraining variance alone.
# Early stopping is off (--patience == --epochs == 200); the best-validation-macro-F1
# checkpoint is kept.
#SBATCH --job-name=fb3d_train
#SBATCH --partition=gpu-xe9680q
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-200%16
#SBATCH --output=runs/_logs/train_%A_%a.out
#SBATCH --error=runs/_logs/train_%A_%a.err
set -euo pipefail
JOBLIST="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"/facebench/training/train_joblist.txt
N=$(grep -c . "$JOBLIST"); [ "$N" -eq 200 ] || { echo "joblist has $N lines, expected 200" >&2; exit 1; }
CMD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
echo "task ${SLURM_ARRAY_TASK_ID}/200  Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-none}"; echo "CMD: $CMD"; date
eval "$CMD"
echo "Done at $(date)"
