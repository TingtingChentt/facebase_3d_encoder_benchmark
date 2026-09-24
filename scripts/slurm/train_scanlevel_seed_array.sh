#!/bin/bash
# ---------------------------------------------------------------------------
# LEAKED-ARM training-seed sweep — 2 tasks x PointNet++ x 5 train seeds = 10 runs.
# thread: facebase3d-figs-2026-09-04
#
# Retrains the scan-level (leaky) arm of Fig. 5(a) so both bars on that panel
# are five-training-seed means, matching Table 2's convention. The corrected arm
# already exists from the 2026-08-31 sweep and is NOT retrained here.
#
# Regenerate the joblist with facebench/eval/make_scanlevel_trainseed_jobs.py.
# Each array index runs one line of scanlevel_trainseed_joblist.txt.
#
# --seed is pinned to 42 on every line, so all 10 runs share one leaky split and
# the spread across --train-seed is retraining variance alone. Do not "fix" that.
#
# THE LEAK IS INTENTIONAL AND IS THE POINT OF THE PANEL. It lives in the frozen
# manifests under data/processed/legacy_scanlevel/, not in dataset.py. Nothing
# these runs produce is task performance — read that README before using any of
# it for anything.
# ---------------------------------------------------------------------------
#SBATCH --job-name=fb3d_sltseed
#SBATCH --partition=gpu-xe9680q
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --array=1-10%5
#SBATCH --output=runs/_scanlevel_trainseed_logs/sltseed_%A_%a.out
#SBATCH --error=runs/_scanlevel_trainseed_logs/sltseed_%A_%a.err

set -euo pipefail

JOBLIST="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"/facebench/eval/scanlevel_trainseed_joblist.txt
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
