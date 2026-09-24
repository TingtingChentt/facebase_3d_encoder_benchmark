#!/bin/bash
# No-early-stopping retrain — stage 2: score the cached features, bootstrap,
# fold into the training-seed artefacts.
# threads facebase3d-a2diag-2026-09-04 / facebase3d-a1diag-2026-09-04.
#
# Runs afterok the family's 40-job extraction array.
# Usage: sbatch --dependency=afterok:<extract array> fix_stage2_analyze.sh A2_fix
#SBATCH --job-name=fb3d_fix_an
#SBATCH --partition=defq
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/paper_reruns/fix_analyze_%j.out
#SBATCH --error=logs/paper_reruns/fix_analyze_%j.err
set -euo pipefail

FAMILY="${1:?usage: sbatch fix_stage2_analyze.sh <A1_fix|A2_fix>}"
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
cd "$PROJ/facebench/eval"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

TASKS=$($PY -c "import rerun_config as C; print(','.join(C.fix_family('$FAMILY')['tasks']))")
SUFFIX=$($PY -c "print('_' + '$FAMILY'.lower().replace('_',''))")
echo "family: $FAMILY   suffix: $SUFFIX"
echo "tasks:  $TASKS"

echo "=== gate: checkpoints, patience==epochs, split unchanged ==="
$PY preflight_fix.py --family "$FAMILY"

echo "=== 2a: score_grid ==="
$PY score_grid.py --tasks "$TASKS"

echo "=== 2b: analyze_ci (subject bootstrap) ==="
$PY analyze_ci.py --tasks "$TASKS" --suffix "$SUFFIX"

echo "=== 2c: fold into the _trainseed artefacts ==="
$PY merge_fix_into_trainseed.py --family "$FAMILY"

echo "=== 2d: collapse across training seeds ==="
$PY aggregate_trainseed.py

echo "Done at $(date)"
