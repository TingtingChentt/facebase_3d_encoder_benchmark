#!/bin/bash
#SBATCH --job-name=fb3d_xsx
#SBATCH --partition=defq
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/xsite/extract_%A_%a.out
#SBATCH --error=logs/xsite/extract_%A_%a.err

# PART A stage 2 — seeded feature extraction for the cross-site folds.
# thread facebase3d-paper-2026-08-08
#
# TEST SPLIT ONLY. These folds are scored softmax-at-scan-level, and the softmax
# head reads test logits alone — the TRAIN split is a gallery that only proto
# and knn11 consume. score_grid.py loads it lazily for exactly this reason, so
# extracting it here would compute 4,178-6,812 scans per fold x 8 (encoder,
# seed) combinations that nothing would ever read.
#
# CPU, and threads=16 deliberately: that is what every cached feature file in
# results/features/ was produced at, and the 08-09 thread probe
# showed CPU inference is bit-identical only AT A FIXED THREAD COUNT (metrics
# are unaffected either way, but there is no reason to introduce the variation).
#
# 5 inference seeds for PointNet++, 1 for the three deterministic encoders,
# per C.seeds_for(). INFERENCE ONLY — checkpoints are read-only.
#
# Usage: sbatch --array=1-48%12 xsite_extract.sh joblist_xsite_extract.txt

set -euo pipefail

PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
JOBLIST="${1:?usage: sbatch --array=1-N xsite_extract.sh <joblist.txt>}"

mkdir -p "$PROJ/logs/xsite"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
if [ -z "$LINE" ]; then
    echo "No job at array index ${SLURM_ARRAY_TASK_ID} in $JOBLIST" >&2
    exit 1
fi
read -r TASK ENCODER SEED <<< "$LINE"

echo "=== fb3d cross-site feature extraction ==="
echo "Node:      $(hostname)"
echo "Task:      $TASK"
echo "Encoder:   $ENCODER"
echo "Inference seed: $SEED   (data split seed stays 42)"
date

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

$PY -u "$PROJ/facebench/eval/extract_features.py" \
    --task "$TASK" \
    --encoder "$ENCODER" \
    --seed "$SEED" \
    --splits test \
    --threads "${SLURM_CPUS_PER_TASK}"

echo "Done at $(date)"
