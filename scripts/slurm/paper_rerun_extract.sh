#!/bin/bash
#SBATCH --job-name=fb3d_rerun
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/paper_reruns/extract_%A_%a.out
#SBATCH --error=logs/paper_reruns/extract_%A_%a.err

# 5-seed inference re-eval — feature extraction array (blocker B1,
# thread facebase3d-paper-2026-08-04).
#
# CPU deliberately, not GPU: CPU inference is bit-identical by construction,
# so the "same seed => identical output" guarantee needs no cuDNN determinism
# flags. The workload is small enough that CPU is not the bottleneck.
#
# INFERENCE ONLY. Checkpoints are read-only; nothing is retrained.
#
# Usage: sbatch --array=1-26 paper_rerun_extract.sh joblist_priority.txt

set -euo pipefail

PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
JOBLIST="${1:?usage: sbatch --array=1-N paper_rerun_extract.sh <joblist.txt>}"

mkdir -p "$PROJ/logs/paper_reruns"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
if [ -z "$LINE" ]; then
    echo "No job at array index ${SLURM_ARRAY_TASK_ID} in $JOBLIST" >&2
    exit 1
fi
read -r TASK ENCODER SEED <<< "$LINE"

echo "=== fb3d paper re-eval extraction ==="
echo "Node:      $(hostname)"
echo "Array idx: ${SLURM_ARRAY_TASK_ID}"
echo "Task:      $TASK"
echo "Encoder:   $ENCODER"
echo "Inference seed: $SEED   (data split seed stays 42)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
date

# Keep BLAS threading consistent with what we tell torch, so timings are
# comparable across array tasks.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

$PY -u "$PROJ/facebench/eval/extract_features.py" \
    --task "$TASK" \
    --encoder "$ENCODER" \
    --seed "$SEED" \
    --threads "${SLURM_CPUS_PER_TASK}"

echo "Done at $(date)"
