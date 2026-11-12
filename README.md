### From Screening to Differential Diagnosis: Benchmarking 3D Facial Representations Across Craniofacial Conditions

Code, subject-level data partitions and aggregate result tables for the paper
*"From Screening to Differential Diagnosis: Benchmarking 3D Facial
Representations Across Craniofacial Conditions"* (Chen, Ahsan, Cotney, Liao & Wang).

---

## Overview

This repository benchmarks 3D facial point-cloud encoders **without landmark
registration, dense correspondence or template fitting** on four FaceBase tasks
that span the range of diagnostic specificity:

| Task | *K* | Cohorts | Task key |
|---|---|---|---|
| Combined affected-vs-unaffected screening | 2 | all six | `C_full_v3` |
| Cleft screening (affected vs. unaffected) | 2 | FB-5A, FB-56 | `A2_fix_v3` |
| Cleft subtyping (unaffected / CL / CP / CLP) | 4 | FB-5A, FB-56 | `A1_fix_v3` |
| Genetic syndrome diagnosis | 49 | FB-TJ0 | `CF20_full_v3` |

Four encoders are compared under one preprocessing and training protocol:
**GeomMLP** (handcrafted global geometric descriptors), **PointNet**,
**PointNet++** (MSG) and **DGCNN**. Two inference rules are evaluated on the same
frozen encoders — the softmax head and cosine **prototype inference** on the
256-d embedding — at scan level and at patient level (a patient's repeated scans
aggregated). Every number is a mean over **five independently trained models**
(10 task configurations × 4 encoders × 5 training seeds = 200 models) evaluated
on one fixed subject-level split.

On top of the benchmark the paper reports four analyses, all included here:
**cohort sensitivity** of combined screening, controlled **repeated-scan
aggregation**, **diagnostic prioritization** (top-K candidates, patient
retrieval, limited reference sets) and **anatomical interpretability** (region
occlusion).

---

## Repository layout

```
facebench/                  the pipeline, one sub-package per stage
  preprocessing/            FaceBase mesh -> 4,096-point cloud (preprocess.py, mesh_utils.py)
                            labelled manifests from FaceBase metadata (build_manifests.py, labels/)
                            partition check (check_splits.py)
  models/                   GeomMLP, PointNet, PointNet++, DGCNN, prototype head
  data/                     dataset class: manifest -> point clouds, subject-level splits
  training/                 train.py, trainer, seeding, task registry, job lists (make_jobs.py)
  inference/                frozen-feature extraction; softmax / prototype / kNN predictions
  evaluation/               metrics, subject bootstrap CIs, seed summaries, contrasts,
                            per-class analyses, LaTeX macro emitters
analyses/
  cohort_sensitivity/       within-cohort and age/sex-adjusted screening (Fig. 3, Tables 3, S1)
  scan_aggregation/         controlled same-patient aggregation (Fig. 4, Table S3)
  prioritization/           top-K candidates, patient retrieval, limited references (Fig. 5)
  interpretability/         landmark regions, region occlusion, nearest neighbours (Fig. 7, S6, S7)
figures/                    one script per paper figure; rendered figures in figures/output/
scripts/slurm/              SLURM array templates (preprocess, train, extract, analyze, interpretability)
data/splits/                subject-level partitions: scan / subject IDs + train/val/test, NO labels
data/labels/                manual review table used by the label curation
results/                    aggregate result tables, paper macros (results/paper_macros/), tables
docs/                       DATA.md, REPRODUCE.md, RESULTS_MAP.md (paper item -> script)
tests/                      model and seeding tests
```

`docs/RESULTS_MAP.md` lists, for every figure, table and quoted number in the
paper, the script that produces it and the files it reads.

---

## Data and what is (not) distributed

All scans come from [FaceBase](https://www.facebase.org) and must be obtained
from FaceBase under its data-access terms.

| ID | FaceBase accession | Dataset | Scans | Subjects |
|---|---|---|---|---|
| FB-5A | FB00001369 | Pittsburgh Oral Facial Cleft Studies (OFC1) | 3,099 | 3,099 |
| FB-56 | FB00001368 | Oral-Facial Cleft Families: Phenotype and Genetics (OFC2) | 3,839 | 3,838 |
| FB-TJ0 | FB00000861 | Developing 3D Craniofacial Morphometry Data and Tools to Transform Dysmorphology | 13,307 | 5,103 |
| FB-TK0 | FB00000892 | 3D White Light Photogrammetry Images of North American Children | 686 | 686 |
| FB-VWP | FB00000491.01 | 3D Facial Norms | 2,454 | 2,454 |
| FB-TX4 | FB00000667.01 | 3D Facial Images — Tanzania | 3,605 | 3,605 |
| | | **Total processed** | **26,990** | **18,785** |

**Shipped:** the subject-level partitions as **ID lists without labels**
(`data/splits/`: one file per task with `scan_id, split`, plus `scans.csv` with
each scan's cohort, subject ID, FaceBase file path and benchmark eligibility),
the code that rebuilds the labelled manifests from the FaceBase metadata, and
aggregate result tables.

**Not shipped:** scans, meshes, point clouds, landmarks, trained checkpoints,
diagnoses or cleft labels per subject, and per-scan model predictions or
embeddings. With FaceBase access, `facebench/preprocessing/build_manifests.py`
regenerates the labelled manifests exactly (checked row for row), and
`check_splits.py` confirms that they reproduce the shipped partitions. See
`docs/DATA.md`.

---

## Installation

```bash
git clone https://github.com/TingtingChentt/facebase_3d_encoder_benchmark.git
cd facebase_3d_encoder_benchmark
conda env create -f environment.yml && conda activate facebench
pip install -e .            # optional: every script also runs directly as `python path/to/script.py`
```

Training needs one GPU; feature extraction, scoring, statistics and figures run
on CPU. The landmark step of the interpretability analysis needs a second small
environment with MediaPipe, and the template-fitting panel of Fig. S1 needs
open3d (see `docs/REPRODUCE.md`).

| Variable | Default | Meaning |
|---|---|---|
| `FACEBASE_ROOT` | — | your FaceBase download (directory holding `FB-5A_files/`, …) |
| `FACEBENCH_DATA_ROOT` | repository root | where point clouds (`data/processed/`) and checkpoints (`runs/`) are written |
| `FACEBENCH_ROOT` | — | repository root, for the SLURM templates |
| `FACEBENCH_PYTHON` | `python3` | interpreter the SLURM templates call |

---

## Reproducing the paper

Full instructions and expected outcomes are in `docs/REPRODUCE.md`.

**Without FaceBase access** — every macro and the aggregate-based figures from
the shipped result tables (CPU, minutes):

```bash
python facebench/evaluation/emit_numbers_corpus.py     # Table 1 counts
python facebench/evaluation/emit_numbers_cfull.py      # combined screening
python facebench/evaluation/emit_numbers_ofc.py        # cleft tasks
python analyses/cohort_sensitivity/write_numbers_cohort.py
python analyses/scan_aggregation/write_numbers_aggctl.py
python analyses/prioritization/write_numbers_protoret.py
python analyses/interpretability/write_numbers_interp.py
python figures/fig1_tasks_encoders_trainseed.py         # Fig. 2   (also: fig_aggregation, fig_prioritization,
                                                        #           figS1_perclass_head, figS2_redistribution)
```

**With FaceBase access** — the full pipeline:

```bash
export FACEBASE_ROOT=/path/to/facebase FACEBENCH_ROOT=$PWD
python facebench/preprocessing/build_manifests.py      # labelled manifests -> data/manifests/
python facebench/preprocessing/check_splits.py         # partitions == data/splits/
sbatch scripts/slurm/preprocess_array.sh               # 26,990 meshes -> 4,096-point clouds
python facebench/training/make_jobs.py                 # 200 training + 400 extraction commands
sbatch scripts/slurm/train_array.sh                    # 200 models
sbatch --array=1-400%24 scripts/slurm/extract_array.sh # frozen features
sbatch scripts/slurm/analyze.sh                        # predictions, bootstrap CIs, seed summaries
# then the four analyses and the remaining figures (docs/REPRODUCE.md)
```

---

## Headline results

Scan level, softmax head, mean ± SD over five independently trained models
(paper Fig. 2; `facebench/evaluation/readme_tables.py` from
`results/*_trainseed_summary.csv`):

| Task | *K* | Encoder | Accuracy | Macro AUC |
|---|---|---|---|---|
| Combined screening | 2 | GeomMLP | 0.806 ± 0.004 | 0.886 ± 0.003 |
| | | PointNet | 0.867 ± 0.010 | 0.939 ± 0.004 |
| | | DGCNN | 0.874 ± 0.005 | 0.948 ± 0.005 |
| | | PointNet++ | 0.893 ± 0.007 | 0.956 ± 0.002 |
| Cleft screening | 2 | GeomMLP | 0.767 ± 0.016 | 0.726 ± 0.002 |
| | | PointNet | 0.763 ± 0.022 | 0.788 ± 0.007 |
| | | DGCNN | 0.815 ± 0.018 | 0.914 ± 0.019 |
| | | PointNet++ | 0.872 ± 0.015 | 0.914 ± 0.012 |
| Cleft subtyping | 4 | GeomMLP | 0.634 ± 0.036 | 0.669 ± 0.007 |
| | | PointNet | 0.694 ± 0.099 | 0.585 ± 0.052 |
| | | DGCNN | 0.668 ± 0.082 | 0.715 ± 0.058 |
| | | PointNet++ | 0.676 ± 0.073 | 0.683 ± 0.096 |
| Syndrome diagnosis | 49 | GeomMLP | 0.103 ± 0.004 | 0.718 ± 0.004 |
| | | PointNet | 0.119 ± 0.010 | 0.734 ± 0.003 |
| | | DGCNN | 0.211 ± 0.015 | 0.797 ± 0.006 |
| | | PointNet++ | 0.248 ± 0.015 | 0.808 ± 0.005 |

49-class syndrome diagnosis at patient level, PointNet++ (385 held-out
patients; paper Table 2): softmax accuracy 0.321 ± 0.021, top-5 0.615 ± 0.014;
prototype inference accuracy 0.350 ± 0.012, top-5 0.652 ± 0.010.

---

## Citation

See `CITATION.cff`. Code is released under the MIT licence (`LICENSE`); the
FaceBase data remain under FaceBase's terms of use.
