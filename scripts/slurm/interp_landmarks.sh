#!/bin/bash
# MediaPipe FaceMesh landmarks for the interpretability scan lists (made by
# analyses/interpretability/make_scan_lists.py and prepare_cases.py; post hoc only;
# the encoders never see landmarks). Needs a separate env with mediapipe==0.10.14
# (see docs/REPRODUCE.md); MP_PYTHON points at it. Run once per list:
#   LIST=results/interp/pilot_scans.csv  OUT=results/interp/landmarks
#   LIST=results/interp/ext_scans.csv    OUT=results/interp/landmarks_ext
#   LIST=results/interp/nn_scans.csv     OUT=results/interp/landmarks_nn
#   LIST=results/interp/nn_scans_ext.csv OUT=results/interp/landmarks_nn_ext
#SBATCH --job-name=fb3d_lmk
#SBATCH --partition=defq
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=0-3
#SBATCH --output=logs/interp/lmk_%A_%a.out
set -euo pipefail
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
: "${FACEBASE_ROOT:?set FACEBASE_ROOT}" "${LIST:?set LIST}" "${OUT:?set OUT}"
"${MP_PYTHON:?set MP_PYTHON to the mediapipe env python}" "$PROJ/analyses/interpretability/landmark_scans.py" \
  --list "$PROJ/$LIST" --out-dir "$PROJ/$OUT" --qc-dir "$PROJ/$OUT/qc" \
  --worker-id "$SLURM_ARRAY_TASK_ID" --n-workers 4
