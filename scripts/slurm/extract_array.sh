#!/bin/bash
# Frozen-feature extraction (train + test splits) for every trained model:
# one line of facebench/training/extract_joblist.txt = "<task> <encoder> <inference seed>"
# (400 lines: PointNet++ x 5 inference seeds, the deterministic encoders once).
#
# CPU deliberately, not GPU: CPU inference is bit-identical by construction,
# so the "same seed => identical output" guarantee needs no cuDNN determinism
# flags. INFERENCE ONLY: checkpoints are read-only.
#
# Usage: sbatch --array=1-400%24 scripts/slurm/extract_array.sh
#SBATCH --job-name=fb3d_extract
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/extract/extract_%A_%a.out
#SBATCH --error=logs/extract/extract_%A_%a.err
set -euo pipefail
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
JOBLIST="${1:-$PROJ/facebench/training/extract_joblist.txt}"
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$JOBLIST")
[ -n "$LINE" ] || { echo "No job at array index ${SLURM_ARRAY_TASK_ID} in $JOBLIST" >&2; exit 1; }
read -r TASK ENCODER SEED <<< "$LINE"
echo "Task $TASK  encoder $ENCODER  inference seed $SEED (data split seed stays 42)"; date
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
"$PY" -u "$PROJ/facebench/inference/extract_features.py" \
    --task "$TASK" --encoder "$ENCODER" --seed "$SEED" --threads "${SLURM_CPUS_PER_TASK}"
echo "Done at $(date)"
