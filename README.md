# From Screening to Differential Diagnosis
### A 3D point cloud benchmark for craniofacial conditions

Code, subject-level split manifests, and result tables for the paper
*"From Screening to Differential Diagnosis: A 3D Point Cloud Benchmark for
Craniofacial Conditions"* (Chen, Ahsan, Cotney, Liao & Wang).

---

## Overview

How far can a 3D facial scan carry a clinical decision, and where does it stop?
This repository benchmarks **registration-free** point-cloud encoders — no
landmark detection, no dense correspondence, no template fitting — across five
craniofacial tasks that span the full range of diagnostic specificity:

| | Task | *K* | Cohorts |
|---|---|---|---|
| **c2** | Cleft screening (affected vs. unaffected) | 2 | FB-5A, FB-56 |
| **b** | Combined craniofacial screening | 2 | all six |
| **c4** | Cleft subtyping (unaffected / CL / CP / CLP) | 4 | FB-5A, FB-56 |
| **s19** | Syndrome category | 19 | FB-TJ0 |
| **d33** | Clinical diagnosis | 33 | FB-TJ0 |

Four encoders are compared under one preprocessing and optimization framework:
**GeomMLP** (26 handcrafted global geometric descriptors, 0.11 M params),
**PointNet** (3.48 M), **PointNet++** MSG (1.74 M), and **DGCNN** (1.81 M).
Two inference rules are evaluated over the same frozen encoders — the trained
softmax head and cosine **prototype retrieval** on the 256-d penultimate
embedding — at both scan and patient level.

**Every headline number is a five-training-seed mean.** The data split is frozen
at seed 42 in all runs, so the reported standard deviation is *retraining*
variance measured on byte-identical test subjects, reported alongside — and
never pooled with — the subject-level bootstrap interval. The main sweep is
5 tasks × 4 encoders × 5 training seeds = **100 independently trained models**.

### Headline result

Scan-level softmax inference, mean ± SD over five independently trained models
(generated from `results/trainseed_summary.csv`):

| Task | *K* | Chance acc. | Encoder | Accuracy | Macro AUC |
|---|---|---|---|---|---|
| Cleft screening | 2 | 0.500 | GeomMLP | 0.722 ± 0.018 | 0.653 ± 0.002 |
| | | | PointNet | 0.736 ± 0.058 | 0.712 ± 0.046 |
| | | | DGCNN | 0.820 ± 0.021 | 0.831 ± 0.033 |
| | | | PointNet++ | 0.795 ± 0.023 | 0.843 ± 0.010 |
| Combined screening | 2 | 0.500 | GeomMLP | 0.850 ± 0.002 | 0.913 ± 0.001 |
| | | | PointNet | 0.875 ± 0.004 | 0.935 ± 0.004 |
| | | | DGCNN | 0.884 ± 0.005 | 0.952 ± 0.005 |
| | | | PointNet++ | 0.905 ± 0.003 | 0.963 ± 0.002 |
| Cleft subtyping | 4 | 0.250 | GeomMLP | 0.582 ± 0.014 | 0.590 ± 0.016 |
| | | | PointNet | 0.737 ± 0.076 | 0.531 ± 0.045 |
| | | | DGCNN | 0.698 ± 0.101 | 0.684 ± 0.024 |
| | | | PointNet++ | 0.633 ± 0.108 | 0.622 ± 0.057 |
| Syndrome category | 19 | 0.053 | GeomMLP | 0.106 ± 0.004 | 0.670 ± 0.009 |
| | | | PointNet | 0.117 ± 0.010 | 0.690 ± 0.015 |
| | | | DGCNN | 0.237 ± 0.033 | 0.794 ± 0.006 |
| | | | PointNet++ | 0.267 ± 0.029 | 0.775 ± 0.009 |
| Clinical diagnosis | 33 | 0.030 | GeomMLP | 0.153 ± 0.017 | 0.763 ± 0.004 |
| | | | PointNet | 0.161 ± 0.027 | 0.801 ± 0.011 |
| | | | DGCNN | 0.280 ± 0.039 | 0.857 ± 0.009 |
| | | | PointNet++ | 0.295 ± 0.029 | 0.845 ± 0.008 |

Accuracy falls steeply with diagnostic granularity while macro AUC stays well
above chance, i.e. the embedding keeps class-wise discrimination that top-1
accuracy hides. The repository also reproduces the three evaluation-design
results: patient-level aggregation, subject leakage, and leave-one-site-out
generalization (see `docs/RESULTS_MAP.md`).

---

## Repository layout

```
facebench/
  models/           encoders, dataset/splitting, trainer, evaluation, training entry point
  preprocessing/    mesh -> 4,096-point cloud pipeline, manifest and label construction
  eval/             the frozen-feature evaluation pipeline, statistics, macro emitters
  paper_figs/       the figure scripts (one per paper figure)
scripts/slurm/      SLURM array templates for the sweeps (CHOP HPC; adapt to your cluster)
data/processed/     subject-level split manifests  <-- the split files, tracked
results/            per-run metrics, bootstrap CIs, seed summaries, LaTeX macro blocks
figures/            the paper figures as published
docs/               reproduction guide, data guide, paper-to-artifact map, references
```

`data/processed/*.csv` are the **split manifests** — the single most important
thing to reuse if you want numbers comparable to ours, because the subject-level
partition determines whether repeated scans of one patient leak across the
train/test boundary. See `docs/DATA.md`.

---

## Datasets

All 3D facial scans come from the [FaceBase](https://www.facebase.org)
consortium and are obtained from FaceBase directly. **No scans, meshes, or
point clouds are redistributed in this repository.**

| ID | FaceBase accession | Dataset | Scans | Subjects | Role |
|---|---|---|---|---|---|
| FB-5A | FB00001369 | Pittsburgh Oral Facial Cleft Studies (OFC1) | 3,099 | 3,099 | cleft, 8 sites |
| FB-56 | FB00001368 | Oral-Facial Cleft Families: Phenotype and Genetics (OFC2) | 3,839 | 3,838 | cleft, 6 sites |
| FB-TJ0 | FB00000861 | Developing 3D Craniofacial Morphometry Data and Tools to Transform Dysmorphology | 13,307 | 5,103 | genetic syndromes |
| FB-TK0 | FB00000892 | 3D White Light Photogrammetry Images of North American Children | 686 | 686 | unaffected |
| FB-VWP | FB00000491.01 | 3D Facial Norms | 2,454 | 2,454 | unaffected |
| FB-TX4 | FB00000667.01 | 3D Facial Images — Tanzania | 3,605 | 3,605 | unaffected |
| | | **Total processed** | **26,990** | **18,785** | |

Task sizes after label harmonization and the minimum class-size thresholds
(≥175 scans for the 19-class task, ≥50 for the 33-class task):
19-class = 12,990 scans / 4,955 subjects · 33-class = 3,623 scans / 1,241
subjects · combined screening = 24,973 scans · cross-site cleft pool = 6,938
scans / 6,937 subjects across 11 acquisition sites.

FB-TJ0 carries heavy repeated imaging (median 2.6, up to 34 scans per subject),
which is why every split in this benchmark is made at the **subject** level.

---

## Installation

```bash
git clone <this repository>
cd facebase_3d_encoder_benchmark

# conda (matches the environment the published results were produced in)
conda env create -f environment.yml
conda activate facebench

# or pip, into a Python 3.9+ environment
pip install -r requirements.txt
```

Published results were produced with **Python 3.9.25, PyTorch 2.8.0 + CUDA
12.8, NumPy 2.0.1, pandas 2.3.3, scikit-learn 1.6.1, SciPy 1.13.1, matplotlib
3.9.4, trimesh 4.11.5**. Training needs one GPU; the entire evaluation and
figure pipeline runs on CPU.

Two environment variables control where things live (both optional):

| Variable | Default | Meaning |
|---|---|---|
| `FACEBENCH_DATA_ROOT` | repository root | where the manifests' relative `out_path` values resolve, i.e. where the processed `.npy` point clouds live |
| `FACEBASE_ROOT` | — | your local copy of the downloaded FaceBase source tree; only needed for preprocessing |

---

## Reproducing the paper

### 1. Result tables and figures, without a GPU

Everything downstream of the trained models is shipped, so the published numbers
and figures regenerate in minutes from this repository alone:

```bash
python facebench/eval/emit_numbers_trainseed.py        # 5-seed macros (Table 2, Sec. 2.2)
python facebench/eval/emit_numbers_leak_trainseed.py   # leakage contrast, both arms 5-seed
python facebench/eval/emit_numbers_xsite.py --fix      # leave-one-site-out, 4-class
python facebench/eval/emit_numbers_xsite.py --fix --binary
python facebench/eval/emit_perclass.py --family trainseed
python facebench/eval/recognizability.py
python facebench/eval/error_concentration.py

cd facebench/paper_figs
python fig1_tasks_encoders_trainseed.py   # Fig. 1
python fig2_core.py                       # Fig. 2
python fig3_recognizability.py            # Fig. 3
python fig4_fooled.py                     # Fig. 4 (leakage) + Fig. 5 (cross-site)
python figS1_perclass_head.py; python figS2_redistribution.py
python figS3_error_structure.py
```

These regenerate the figures in `figures/` byte-identically and the LaTeX macro
blocks in `results/paper_macros/` value-identically to the submitted manuscript.
(`fig0_overview.py` and `figS4_saliency.py` additionally need the FaceBase
source metadata and a saliency tensor that is not redistributed; see
`docs/RESULTS_MAP.md`.)

### 2. Full reproduction, from FaceBase downloads

```bash
export FACEBASE_ROOT=/path/to/your/facebase/download
export FACEBENCH_ROOT=$PWD

# (a) meshes -> normalized 4,096-point clouds + manifests
python facebench/preprocessing/preprocess.py --exp ofc
python facebench/preprocessing/preprocess.py --exp syndrome
python facebench/preprocessing/m2_preprocess_controls.py
python facebench/preprocessing/build_combined_manifest.py

# (b) the 5-training-seed sweep: 100 runs, ~68 GPU-hours
python facebench/eval/make_trainseed_jobs.py        # writes trainseed_joblist.txt
sbatch scripts/slurm/train_seed_array.sh            # or run the lines yourself

# (c) one forward pass per scan, cached; every head and level derives from it
python facebench/eval/preflight_trainseed.py        # refuses to proceed on a missing checkpoint
sbatch scripts/slurm/paper_rerun_extract.sh

# (d) score the head x level grid, bootstrap, collapse across training seeds
sbatch scripts/slurm/trainseed_stage4_analyze.sh
```

A single training run is reproducible from its `--train-seed`; step (d) is
deterministic given the cached features. `docs/REPRODUCE.md` walks through each
stage, including the leakage and cross-site arms, and states what is and is not
bit-reproducible.

### 3. Train one model

```bash
python facebench/models/train.py \
    --model pointnet2 --exp-id my_run \
    --manifest data/processed/syndrome_manifest_b1.csv \
    --exp-type syndrome --num-classes 19 \
    --seed 42 --train-seed 1
```

`--seed` fixes the **data split** and should stay at 42 to remain comparable
with the published results; `--train-seed` is the axis the sweep varies.

---

## Three seeds, deliberately kept apart

A recurring source of confusion in this pipeline, so it is named explicitly
everywhere in the code (`facebench/eval/rerun_config.py`):

| Seed | Values | Controls | Varying it measures |
|---|---|---|---|
| **data split** | 42, frozen | train/val/test membership | *nothing here* — never varied, or test sets stop being comparable |
| **training** | 1–5 | weight init, batch order, augmentation | retraining variance — **the reported ± SD** |
| **inference** | 0–4 | PointNet++ farthest-point-sampling start index, weights frozen | FPS noise; averaged down *within* each training seed before the SD is taken |

The subject-level bootstrap CI is a fourth, orthogonal quantity (cohort
sampling). All are carried side by side in `results/trainseed_summary.csv` and
are never pooled.

One reproducibility limit is stated plainly: the offline farthest-point sampling
used to build the stored point clouds was not seeded, so regenerating the `.npy`
files from the FaceBase meshes yields a different point sample of the same
surfaces. Results should match within the reported seed SD, not exactly. Only
PointNet++ is non-deterministic at inference; this was established empirically
rather than assumed (`facebench/eval/probe_determinism.py`,
`results/AUDIT_NONDETERMINISM.md`).

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{chen2026facebase3d,
  author  = {Chen, Tingting and Ahsan, Mian Umair and Cotney, Justin and
             Liao, Eric and Wang, Kai},
  title   = {From Screening to Differential Diagnosis: A 3D Point Cloud
             Benchmark for Craniofacial Conditions},
  year    = {2026}
}
```

Please also cite the encoder architectures and the data source:

```bibtex
@inproceedings{qi2017pointnet,
  author    = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2017}
}

@inproceedings{qi2017pointnet2,
  author    = {Qi, Charles R. and Yi, Li and Su, Hao and Guibas, Leonidas J.},
  title     = {PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@article{wang2019dgcnn,
  author  = {Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and
             Bronstein, Michael M. and Solomon, Justin M.},
  title   = {Dynamic Graph CNN for Learning on Point Clouds},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2019},
  volume  = {38},
  number  = {5},
  pages   = {1--12}
}
```

The FaceBase datasets must be cited by their accessions (FB00001369,
FB00001368, FB00000861, FB00000892, FB00000491.01, FB00000667.01) under the
FaceBase data-use terms. Related prior work on 3D and 2D facial phenotyping is
collected in `docs/references.bib`.

---

## License

Code and derived result tables: **MIT** (see `LICENSE`). The FaceBase facial
scans are *not* covered by this license and are not redistributed here; access
is through FaceBase under its own terms.

## Contact

Questions about the benchmark: open an issue, or contact the corresponding
author of the paper.
