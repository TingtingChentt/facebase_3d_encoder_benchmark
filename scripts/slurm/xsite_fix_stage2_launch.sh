#!/bin/bash
# Cross-site fix — stage 2 launcher. thread facebase3d-xsitediag-2026-09-07.
#
# Runs afterok the 48-run training array. Gates on the checkpoints, then builds
# the two extraction joblists and submits the extraction arrays and the scoring
# job with the right dependencies. It exists as a JOB rather than a hand-run
# command because the joblist length is not known until training finishes, and
# an array's size has to be fixed at submit time.
#SBATCH --job-name=fb3d_xsfix_l
#SBATCH --partition=defq
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/xsite/xsfix_launch_%j.out
#SBATCH --error=logs/xsite/xsfix_launch_%j.err
set -euo pipefail

PROJ="${FACEBENCH_ROOT:?set FACEBENCH_ROOT to the repository root}"
PY="${FACEBENCH_PYTHON:-python3}"
cd "$PROJ/facebench/eval"

echo "=== gate: 48 checkpoints, patience==epochs, folds and splits unchanged ==="
$PY preflight_xsite_fix.py

echo "=== build extraction joblists (--require-all: refuse a partial sweep) ==="
$PY make_xsite_extract_joblist.py --fix --require-all \
    --out joblist_xsite_fix_extract.txt
$PY make_xsite_extract_joblist.py --fix --binary --require-all \
    --out joblist_xsite_fix_bin_extract.txt

N4=$(grep -c . joblist_xsite_fix_extract.txt)
NB=$(grep -c . joblist_xsite_fix_bin_extract.txt)
echo "four-class: $N4 jobs   binary: $NB jobs"

E4=$(sbatch --parsable --array=1-"$N4"%12 "$PROJ/experiments/slurm/xsite_extract.sh" \
     "$PROJ/facebench/eval/joblist_xsite_fix_extract.txt")
EB=$(sbatch --parsable --array=1-"$NB"%12 "$PROJ/experiments/slurm/xsite_extract.sh" \
     "$PROJ/facebench/eval/joblist_xsite_fix_bin_extract.txt")
echo "extraction arrays: $E4 (four-class), $EB (binary)"

AN=$(sbatch --parsable --dependency=afterok:"$E4":"$EB" \
     "$PROJ/experiments/slurm/xsite_fix_stage3_analyze.sh")
echo "scoring job: $AN"
echo "Done at $(date)"
