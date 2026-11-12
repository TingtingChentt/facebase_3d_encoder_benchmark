# Reproduction guide

Two levels.

- **Level 1 — from the shipped result tables.** Every number the manuscript
  quotes and the figures built from aggregate results. CPU, minutes, no FaceBase
  access.
- **Level 2 — from the FaceBase downloads.** Labelled manifests, preprocessing,
  200 training runs, feature extraction, statistics, the four analyses and the
  face-rendering figures.

```bash
export FACEBENCH_ROOT=$PWD                     # repository root (SLURM templates)
export FACEBENCH_PYTHON=$(which python)        # interpreter the SLURM templates call
```

---

## Level 1 — numbers and figures from shipped results

```bash
# LaTeX macro blocks -> results/paper_macros/
python facebench/evaluation/emit_numbers_corpus.py           # Table 1 counts
python facebench/evaluation/emit_numbers_cf20.py             # 49-class task
python facebench/evaluation/emit_numbers_cfull.py            # combined screening
python facebench/evaluation/emit_numbers_ofc.py              # cleft tasks
python analyses/cohort_sensitivity/write_numbers_cohort.py   # cohort sensitivity
python analyses/scan_aggregation/write_numbers_aggctl.py     # controlled aggregation
python analyses/prioritization/write_numbers_protoret.py     # diagnostic prioritization
python analyses/interpretability/write_numbers_interp.py     # region occlusion

# supplementary tables -> results/paper_tables/
python analyses/cohort_sensitivity/write_supp_table_cohort.py
python analyses/scan_aggregation/write_supp_table_aggctl.py

# figures -> figures/output/
for f in fig1_tasks_encoders_trainseed fig_aggregation fig_prioritization fig3_recognizability \
         figS1_perclass_head figS2_redistribution figS3_error_structure figS5_cutoff; do
  python figures/$f.py
done
```

**Expected outcome (checked before release).** All 307 macros the manuscript
uses are regenerated with values identical to the manuscript's, and the two
supplementary tables are identical to the manuscript's. The eight figures are
byte-identical to the published PNGs when rendered with the same matplotlib
build: `fig3_recognizability` and `figS5_cutoff` with matplotlib 3.9.4
(`environment.yml`), the other six with matplotlib 3.11.0. With another version
they differ only in sub-pixel glyph and marker placement.

The class-level summaries some of these scripts need (per-class F1 and test
size, an averaged confusion matrix, error-concentration statistics, task counts)
are read from `results/derived/`; with the predictions present (Level 2) the
scripts recompute and overwrite them.

---

## Level 2 — from the FaceBase downloads

```bash
export FACEBASE_ROOT=/path/to/facebase/download
export FACEBENCH_DATA_ROOT=$PWD                # or a scratch directory for clouds and checkpoints
```

### (a) Labelled manifests and partitions

```bash
python facebench/preprocessing/build_manifests.py   # data/manifests/*.csv, from the FaceBase metadata
python facebench/preprocessing/check_splits.py      # every partition == data/splits/<task>.csv
```

### (b) Point clouds (26,990 scans, CPU; deterministic per scan)

```bash
sbatch scripts/slurm/preprocess_array.sh
```

After this step, `check_splits.py` passes with the clouds present as well; a
partial run would change the combined and cleft partitions (see `docs/DATA.md`).

### (c) Training: 10 task configurations × 4 encoders × 5 training seeds

```bash
python facebench/training/make_jobs.py        # facebench/training/{train,extract}_joblist.txt (200 / 400 lines)
sbatch scripts/slurm/train_array.sh           # one GPU per run, 200 epochs, best-validation-macro-F1 checkpoint
```

The generated commands are the ones the paper's models were trained with
(checked line for line). Retraining reproduces the reported numbers up to the
reported training-run SDs: the training seed makes each run reproducible on the
same hardware and software stack, but GPU kernels are not bit-stable across
stacks.

### (d) Frozen features (CPU), predictions and statistics

```bash
sbatch --array=1-400%24 scripts/slurm/extract_array.sh   # PointNet++ x 5 inference seeds, other encoders once
sbatch scripts/slurm/analyze.sh                          # predictions, subject bootstrap CIs, seed summaries -> results/
```

Extraction runs on CPU with a seeded farthest-point-sampling start index. With the
paper's checkpoints, re-extraction on a different CPU reproduced the cached
features to float32 rounding (relative difference ≤ 1e-6) with identical
predicted classes; scoring is deterministic given the features.

Then run Level 1 to regenerate the macros and figures from the new results.

### (e) Analyses built on the trained models (no retraining)

```bash
# cohort sensitivity of combined screening (Fig. 3, Tables 3 and S1); needs FACEBASE_ROOT for age/sex
python analyses/cohort_sensitivity/cohort_confound.py
python analyses/cohort_sensitivity/age_adjusted.py
python figures/fig_cohort_confound.py

# controlled same-patient aggregation (Fig. 4, Table S3)
python analyses/scan_aggregation/scan_count_controlled.py

# diagnostic prioritization: top-K candidates, patient retrieval, limited references (Fig. 5)
python analyses/prioritization/proto_retrieval.py
```

Each writes `results/<analysis>/`; then rerun the matching `write_numbers_*` /
`write_supp_table_*` scripts and figures of Level 1. Re-running all three with the
paper's predictions reproduced every shipped table byte for byte.

### (f) Anatomical interpretability (Fig. 7, Figs. S6–S7)

1. **Scan lists and cases** (local; they carry labels):

   ```bash
   python analyses/interpretability/make_scan_lists.py                        # pilot / ext test scans
   python analyses/interpretability/prepare_cases.py                          # cases.csv, nn_scans.csv
   python analyses/interpretability/prepare_cases.py --list ext_scans.csv --tag ext
   ```

2. **Landmarks** (post hoc only; the encoders never see them). Create a small
   environment with MediaPipe 0.10.14, OpenCV, trimesh, plyfile, SciPy, pandas
   and Pillow, and point `MP_PYTHON` at it:

   ```bash
   LIST=results/interp/pilot_scans.csv  OUT=results/interp/landmarks        sbatch scripts/slurm/interp_landmarks.sh
   LIST=results/interp/ext_scans.csv    OUT=results/interp/landmarks_ext    sbatch scripts/slurm/interp_landmarks.sh
   LIST=results/interp/nn_scans.csv     OUT=results/interp/landmarks_nn     sbatch scripts/slurm/interp_landmarks.sh
   LIST=results/interp/nn_scans_ext.csv OUT=results/interp/landmarks_nn_ext sbatch scripts/slurm/interp_landmarks.sh
   ```

3. **Occlusion** with the five 49-class PointNet++ models:

   ```bash
   LIST=results/interp/pilot_scans.csv LM=results/interp/landmarks     OUTC=results/interp/compute     sbatch scripts/slurm/interp_occlusion.sh
   LIST=results/interp/ext_scans.csv   LM=results/interp/landmarks_ext OUTC=results/interp/compute_ext sbatch scripts/slurm/interp_occlusion.sh
   python analyses/interpretability/occlusion.py
   ```

4. **Figures, tables and macros:**

   ```bash
   python analyses/interpretability/main_figure_interp.py     # figures + results/interp/tables/
   python analyses/interpretability/write_numbers_interp.py
   ```

The occlusion scores were computed with 8 CPU threads; another thread count
changes `cos_true` in the 7th decimal and nothing else.

### (g) Figures that render faces

Fig. 1, Fig. 7, Fig. 8 (top row), Supplementary Fig. S1 and Figs. S6–S7 show
real FaceBase scans. Which subjects are shown is read from a local
`data/exemplars.csv` that is not distributed (see `figures/exemplars.py` and
`data/exemplars.example.csv`); no reported number depends on it.

```bash
python figures/fig1_scan_quality.py           # display-quality score per scan (choice of scan to draw)
python figures/fig1_clinical.py               # Fig. 1
python figures/figS_overview_registration.py  # Fig. S1a renders (needs open3d)
python figures/figS_overview.py               # Fig. S1
python figures/fig_encoders_renders.py        # Fig. 8 top-row renders (needs step (f))
python figures/fig_encoders.py                # Fig. 8
```

With the paper's exemplars these reproduce the published figures: Fig. 7,
Figs. S6–S7 and Fig. 8 byte for byte (matplotlib 3.9.4); Fig. 1 to within 2/255
in 286 pixels of the panel-(a) face render (matplotlib 3.11.0; the mesh is read
from the FaceBase file instead of an intermediate OBJ export). Fig. S1 reports
the scan counts actually used per task (24,612 combined, 6,873 cleft) and draws
the exemplar's final point cloud.
