# How the syndrome labels and splits are built

These scripts turn the FB-TJ0 metadata into the labels and the subject split of
the syndrome-diagnosis tasks. You do not run them by hand:
`facebench/preprocessing/build_manifests.py` writes their input
(`data/labels/build/fbtj0_cohort.csv`: the FB-TJ0 scans of `data/splits/scans.csv`
with their `Syndrome Category`), runs them in order, and checks their output
against the shipped split files.

| Step | Script | Does |
|---|---|---|
| 1 | `build_clinical_filled_manifest.py` | per-subject check of the `Clinical Diagnosis` field against `Syndrome` (agree / alias / conflict) |
| 2 | `build_syndrome_name_manifest.py` | the `Syndrome`-field rules: case variants merged, non-diagnostic values and Rett subtypes excluded |
| 3 | `build_syndrome_final_manifest.py` | + Moebius class, listed conflicts and conflicts from the manual review (`data/labels/alias_pairs_reviewed.csv`) excluded; keeps classes with ≥50 scans and ≥20 subjects |
| 4 | `build_class_filter_manifests.py` | one subject split per class (seed 42 + CRC32 of the class name, 70/15/15) over every class with ≥50 scans and ≥10 subjects, then the seven cutoff manifests; asserts the K=20 manifest equals step 3 |

All intermediate and output files go to `data/labels/build/` (not distributed:
they are subject-level and labelled). Inputs: the FB-TJ0 metadata table
(`FACEBASE_ROOT/FB-TJ0_files/FB00000861_metadata_2020-09-14.csv`) and the
shipped review table. Only scans marked `eligible` in `scans.csv` enter the
label pool.

Run through `build_manifests.py` on the FaceBase release used for the paper, the
chain reproduced all seven syndrome manifests (`scan_id, fbid, syndrome_name,
fixed_split`) used in training, row for row. `pool.py` holds the pool
construction shared by step 4.
