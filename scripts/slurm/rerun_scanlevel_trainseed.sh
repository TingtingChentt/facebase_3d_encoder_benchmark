#!/bin/bash
#SBATCH --job-name=fb3d_sl_ts
#SBATCH --partition=defq
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=1-10%10
#SBATCH --output=results/scanlevel_ts_%A_%a.out
#SBATCH --error=results/scanlevel_ts_%A_%a.err

# Score the RETRAINED leaky checkpoints (B{1,2}_scanlevel_ts{1..5}) under the
# corrected arm's protocol. thread facebase3d-figs-2026-09-04.
#
# CPU AT 16 THREADS IS NOT A PREFERENCE. AUDIT_NONDETERMINISM.md line 58: the
# corrected arm was run on CPU to avoid cuDNN non-determinism and every cached
# feature file was produced at 16 BLAS threads. Fig. 5(a) contrasts two SPLITS;
# scoring one arm on a different device would make it a device contrast too.
#
# One task x one training seed per array element, so the five retrainings of a
# task are scored by five independent processes and cannot share state.
set -euo pipefail

TASKS=(B1 B1 B1 B1 B1 B2 B2 B2 B2 B2)
SEEDS=(1  2  3  4  5  1  2  3  4  5)
i=$((SLURM_ARRAY_TASK_ID - 1))
TASK=${TASKS[$i]}
TS=${SEEDS[$i]}

echo "Leaked-arm re-eval — ${TASK}_scanlevel_ts${TS}, 5 inference seeds, CPU/16"
echo "Node: $(hostname)"
date

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

"${FACEBENCH_PYTHON:-python3}" \
    "${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"/facebench/eval/rerun_scanlevel.py \
    --tasks "$TASK" --train-seed "$TS" --device cpu --threads 16

echo "Done at $(date)"
