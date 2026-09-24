# Reproduction guide

Three levels, from cheapest to most complete. Pick the one that matches what
you want to check.

- **Level 1** — regenerate every published figure and number from the shipped
  result tables. No GPU, no FaceBase access, minutes.
- **Level 2** — re-score the frozen models from cached features. No GPU, needs
  the cached feature files (not shipped; regenerate in Level 3).
- **Level 3** — retrain everything from the FaceBase source meshes. ~130
  GPU-hours for all arms.

Set these once:

```bash
export FACEBENCH_ROOT=$PWD                     # repository root
export FACEBENCH_PYTHON=$(which python)        # used by the SLURM templates
```

---

## Level 1 — numbers and figures from shipped results

The trained models are the expensive part, and everything downstream of them is
in this repository: per-run metrics, subject bootstrap intervals, seed
summaries, contrast tests, and the subject-level prediction arrays the figures
read.

```bash
# LaTeX macro blocks (the manuscript quotes no number inline)
python facebench/eval/emit_numbers_trainseed.py
python facebench/eval/emit_numbers_leak_trainseed.py
python facebench/eval/emit_numbers_xsite.py --fix
python facebench/eval/emit_numbers_xsite.py --fix --binary
python facebench/eval/emit_perclass.py --family trainseed
python facebench/eval/recognizability.py
python facebench/eval/error_concentration.py

# figures
cd facebench/paper_figs
python fig1_tasks_encoders_trainseed.py
python fig2_core.py
python fig3_recognizability.py
python fig4_fooled.py          # writes both fig4_leakage.png and fig5_xsite.png
python figS1_perclass_head.py
python figS2_redistribution.py
python figS3_error_structure.py
```

**Expected outcome.** The seven figures above are byte-identical to the ones in
`figures/`. The macro blocks in `results/paper_macros/` are value-identical to
the submitted manuscript; the only lines that change are the provenance
comments naming the generator's path.

Two artifacts cannot be regenerated at this level and are documented rather than
shipped:

- `fig0_overview.py` needs the FaceBase demographic tables and one exemplar
  scan, neither of which is redistributable.
- `figS4_saliency.py` needs a 28 MB gradient-saliency tensor; regenerate it with
  `python facebench/eval/saliency_full_test.py` once checkpoints exist.

---

## Level 2 — re-score from cached features

The evaluation design here exists because the full head × level decision grid —
softmax / prototype / cosine-kNN, at scan and subject level — is a function of
**one forward pass per scan**. `extract_features.py` runs each encoder once and
caches the trunk feature, the 256-d pre-logit embedding, and the logits; every
head, level, and bootstrap replicate is then derived in NumPy.

```bash
cd facebench/eval
TASKS=$(python -c "import rerun_config as C; print(','.join(C.TRAINSEED_TASKS))")

python score_grid.py  --tasks "$TASKS"                      # per-unit predictions
python analyze_ci.py  --tasks "$TASKS" --suffix _trainseed  # bootstrap CIs + top-k
python aggregate_trainseed.py                               # collapse across seeds
```

`analyze_ci.py` runs a 2,000-replicate subject bootstrap and takes roughly an
hour over the full task set; `--skip-collect` recomputes only the paired
contrasts.

Feature extraction deliberately runs on **CPU**. CPU inference is bit-identical
by construction, so the "same seed ⇒ same output" guarantee needs no cuDNN
determinism flags. `facebench/eval/diff_gpu_cpu.py` quantifies the GPU/CPU
difference rather than assuming it away.

---

## Level 3 — full retraining

### (a) Preprocess

```bash
export FACEBASE_ROOT=/path/to/facebase/download
python facebench/preprocessing/preprocess.py --exp ofc
python facebench/preprocessing/preprocess.py --exp syndrome
python facebench/preprocessing/m2_preprocess_controls.py
python facebench/preprocessing/build_combined_manifest.py
```

Each surface: keep the largest connected component, sample ~3× the target count
(Poisson-disc where available, area-weighted otherwise), reduce to **4,096
points by farthest-point sampling**, center at the centroid, scale so the
maximum radius is 1. No landmarking, no Procrustes, no template fitting. Only
translation and global scale are normalized.

### (b) The 5-training-seed sweep — 100 runs, ~68 GPU-hours

```bash
python facebench/eval/make_trainseed_jobs.py    # -> facebench/eval/trainseed_joblist.txt
sbatch scripts/slurm/train_seed_array.sh        # array 1-100
```

Every emitted command pins `--seed 42` (the data split) and varies only
`--train-seed`. That is the whole design: all 100 models are scored on
byte-identical test subjects, so the spread is retraining variance and nothing
else.

Training: AdamW, lr 1e-3 cosine-annealed to 1e-5 over at most 200 epochs, weight
decay 1e-4, class-weighted cross-entropy, batch 16 (PointNet++) or 32–64. The
checkpoint with the best validation macro-F1 is selected. The two cleft tasks
run the full 200 epochs with early stopping **disabled** — they have a long
initial plateau during which validation macro-F1 is noise, and a 20-epoch
patience can kill a run before it starts learning. This is why the reported
cleft arms are the `A1_fix` / `A2_fix` task keys.

### (c) Gate, then extract

```bash
sbatch scripts/slurm/trainseed_stage2_gate.sh       # or: python facebench/eval/preflight_trainseed.py
python facebench/eval/make_joblist.py --stage trainseed > facebench/eval/joblist_trainseed.txt
sbatch --array=1-$(wc -l < facebench/eval/joblist_trainseed.txt) \
       scripts/slurm/paper_rerun_extract.sh facebench/eval/joblist_trainseed.txt
```

The gate checks all 100 checkpoints exist **and** that each run's `config.json`
records `split_seed == 42` and the train seed its task name claims. It exits
non-zero on any failure, because a silently missing seed would shrink a standard
deviation rather than raise an error.

Only PointNet++ gets five inference seeds. GeomMLP, PointNet, and DGCNN return
bit-identical features across repeat runs — established by measurement
(`facebench/eval/probe_determinism.py`, `results/AUDIT_NONDETERMINISM.md`), not
by reading the code — so their inference-seed SD is exactly 0 and extra seeds
would burn 4× the compute for identical numbers.

### (d) Score and aggregate

```bash
sbatch scripts/slurm/trainseed_stage4_analyze.sh
```

### (e) The other arms

| Arm | Jobs | Command |
|---|---|---|
| Subject leakage (leaked scan-level, 5 seeds) | 10 | `make_scanlevel_trainseed_jobs.py` → `scripts/slurm/train_scanlevel_seed_array.sh` → `aggregate_scanlevel_trainseed.py` |
| Leave-one-site-out, 4-class + binary | 48 | `make_xsite_fix_jobs.py` → `scripts/slurm/xsite_fix_array.sh` → `xsite_fix_stage2_launch.sh` → `xsite_fix_stage3_analyze.sh` |
| Cleft tasks without early stopping | 40 | `make_fix_jobs.py --family A1_fix` / `--family A2_fix` → `scripts/slurm/a{1,2}_fix_array.sh` |

Cross-site folds are one training run per (site, encoder) by design, with
intervals from a subject bootstrap **within** the held-out site. Adding a seed
axis there would change two things at once and make the before/after
uninterpretable.

---

## What is and is not bit-reproducible

| | Reproducible? |
|---|---|
| Scoring, bootstrap, aggregation from cached features | **Yes** — deterministic given the same features |
| CPU feature extraction from the same checkpoint and seed | **Yes** — bit-identical |
| Training from `--train-seed N` on the same hardware/versions | **Yes** in practice; cross-version and cross-GPU drift is expected |
| Regenerating the `.npy` point clouds from FaceBase meshes | **No** — the offline farthest-point sampling was not seeded when the stored clouds were produced, so you get a different point sample of the same surfaces. Expect agreement within the reported seed SD, not exact equality. |

The SLURM templates under `scripts/slurm/` were written for the CHOP HPC cluster
(partitions `defq` / `gpu-xe9680q`). Adapt the partition, account, and resource
directives; the commands they wrap are ordinary shell and need no scheduler.
