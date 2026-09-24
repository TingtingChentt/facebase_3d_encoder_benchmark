#!/bin/bash
# Stage 4 — score the cached features, bootstrap, and collapse across
# training seeds. Runs after the extraction array completes.
#SBATCH --job-name=fb3d_ts_analyze
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=runs/_trainseed_logs/analyze_%j.out
#SBATCH --error=runs/_trainseed_logs/analyze_%j.err
set -euo pipefail

PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
cd "$PROJ/facebench/eval"

TASKS=$($PY -c "import rerun_config as C; print(','.join(C.TRAINSEED_TASKS))")
echo "tasks: $TASKS"

echo "=== stage 4a: score_grid ==="
$PY score_grid.py --tasks "$TASKS"

echo "=== stage 4b: analyze_ci (subject bootstrap + top-k) ==="
$PY analyze_ci.py --tasks "$TASKS" --suffix _trainseed

echo "=== stage 4c: collapse across training seeds ==="
$PY aggregate_trainseed.py

echo "Done at $(date)"
