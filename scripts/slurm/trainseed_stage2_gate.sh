#!/bin/bash
#SBATCH --job-name=fb3d_ts_gate
#SBATCH --partition=defq
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=runs/_trainseed_logs/gate_%j.out
#SBATCH --error=runs/_trainseed_logs/gate_%j.err
set -euo pipefail
"${FACEBENCH_PYTHON:-python3}" \
  "${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"/facebench/eval/preflight_trainseed.py
