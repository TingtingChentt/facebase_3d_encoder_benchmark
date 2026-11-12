#!/bin/bash
# Region occlusion, one array task per trained CF20_full_v3 PointNet++ model (train seeds 1-5).
# Needs runs/ checkpoints and results/features/ (train split, inference seed 0). Run once per list:
#   LIST=results/interp/pilot_scans.csv LM=results/interp/landmarks     OUTC=results/interp/compute
#   LIST=results/interp/ext_scans.csv   LM=results/interp/landmarks_ext OUTC=results/interp/compute_ext
# then: python analyses/interpretability/occlusion.py
#SBATCH --job-name=fb3d_occ
#SBATCH --partition=defq
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --array=1-5
#SBATCH --output=logs/interp/occ_%A_%a.out
set -euo pipefail
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
: "${LIST:?set LIST}" "${LM:?set LM}" "${OUTC:?set OUTC}"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
"$PY" "$PROJ/analyses/interpretability/interp_compute.py" --ts "$SLURM_ARRAY_TASK_ID" \
  --list "$PROJ/$LIST" --lm-dir "$PROJ/$LM" --out-dir "$PROJ/$OUTC"
