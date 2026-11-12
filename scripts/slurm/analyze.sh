#!/bin/bash
# Score cached features, subject-level bootstrap, collapse over training seeds.
# One output suffix per family (results/*<suffix>*.csv); families are never pooled.
#   A1_fix_v3 -> _a1fixv3   A2_fix_v3 -> _a2fixv3   C_full_v3 -> _cfullv3
#   CF20_full_v3 -> _cf20fullv3   all seven CF{K}_full_v3 cutoffs together -> _cfv3
#SBATCH --job-name=fb3d_analyze
#SBATCH --partition=defq
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
set -euo pipefail
PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
cd "$PROJ"
export PYTHONPATH="$PROJ${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
fam() { "$PY" -c "from facebench.training import task_registry as C; print(','.join($1))"; }
for F in A1_fix_v3 A2_fix_v3 C_full_v3 CF20_full_v3; do
  SUF=$("$PY" -c "from facebench.training import task_registry as C; print(C.SUFFIX['$F'])")
  T=$(fam "C.FAMILIES['$F']")
  echo "=== $F  suffix $SUF ==="
  "$PY" facebench/inference/score_grid.py --tasks "$T"
  "$PY" facebench/evaluation/analyze_ci.py --tasks "$T" --suffix "$SUF"
  "$PY" facebench/evaluation/aggregate_trainseed.py --suffix "$SUF" --out-prefix "${SUF#_}_trainseed"
done
echo "=== class-size sweep, suffix _cfv3 ==="
CF=$(fam "[t for b in C.CF_FAMILIES for t in C.FAMILIES[b]]")
"$PY" facebench/inference/score_grid.py --tasks "$CF"
"$PY" facebench/evaluation/analyze_ci.py --tasks "$CF" --suffix _cfv3
"$PY" facebench/evaluation/aggregate_trainseed.py --suffix _cfv3 --out-prefix cfv3_trainseed
echo "Done at $(date)"
