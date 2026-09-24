#!/bin/bash
# Cross-site fix — stage 3: score the cached features and bootstrap over
# held-out-site subjects. thread facebase3d-xsitediag-2026-09-07.
#
# The two arms are scored into SEPARATE suffixes, matching how the published
# folds are held (_xsite / _xsite_bin). They are six folds of the same sites
# under two label collapses, and the paper's claim is the contrast between them,
# so they are never pooled into one file.
#SBATCH --job-name=fb3d_xsfix_an
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/xsite/xsfix_analyze_%j.out
#SBATCH --error=logs/xsite/xsfix_analyze_%j.err
set -euo pipefail

PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
cd "$PROJ/facebench/eval"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

FOUR=$($PY -c "import rerun_config as C; print(','.join(C.XSITE_FIX_TASKS))")
BIN=$($PY -c "import rerun_config as C; print(','.join(C.XSITE_FIX_BIN_TASKS))")

echo "=== gate ==="
$PY preflight_xsite_fix.py

echo "=== four-class arm: $FOUR ==="
$PY score_grid.py --tasks "$FOUR"
$PY analyze_ci.py --tasks "$FOUR" --suffix _xsitefix

echo "=== binary arm: $BIN ==="
$PY score_grid.py --tasks "$BIN"
$PY analyze_ci.py --tasks "$BIN" --suffix _xsitefix_bin

echo "Done at $(date)"
