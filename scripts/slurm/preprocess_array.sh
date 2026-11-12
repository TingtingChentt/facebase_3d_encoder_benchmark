#!/bin/bash
# Mesh -> 4,096-point cloud for every scan in data/splits/scans.csv (26,990 scans).
# One cleaning rule for all six datasets (weld coincident vertices, grow from the
# largest component); see facebench/preprocessing/preprocess.py. Per-scan RNG is
# seeded from the scan id, so reruns are bit-identical. Existing outputs are skipped.
#SBATCH --job-name=fb3d_prep
#SBATCH --partition=defq
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --array=0-119
#SBATCH --output=logs/preprocess/prep_%A_%a.out
#SBATCH --error=logs/preprocess/prep_%A_%a.err
set -euo pipefail
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
: "${FACEBASE_ROOT:?set FACEBASE_ROOT to the directory holding the FB-*_files downloads}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$PROJ/facebench/preprocessing/preprocess.py" \
  --list "$PROJ/data/splits/scans.csv" --worker-id "$SLURM_ARRAY_TASK_ID" --n-workers 120
