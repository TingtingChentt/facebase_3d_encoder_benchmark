# Data guide

## What is and is not in this repository

**Included:** the split manifests (`data/processed/*.csv`), which define every
train/validation/test partition used in the paper, plus the derived result
tables in `results/`.

**Not included:** any 3D facial scan, mesh, texture, or processed point cloud.
All scans come from the [FaceBase](https://www.facebase.org) consortium and are
obtained from FaceBase under its own access terms. Nothing that could
reconstruct a face is redistributed here.

## Getting the source data

Request/download the six datasets from FaceBase:

| ID | Accession | Format distributed |
|---|---|---|
| FB-5A | FB00001369 | OBJ meshes + participant/image-mapping tables |
| FB-56 | FB00001368 | OBJ meshes + participant/image-mapping tables |
| FB-TJ0 | FB00000861 | PLY surfaces + scan-level metadata CSV |
| FB-TK0 | FB00000892 | PLY |
| FB-VWP | FB00000491.01 | OBJ |
| FB-TX4 | FB00000667.01 | OBJ |

Point `FACEBASE_ROOT` at the root of your download. The manifests' `src_path`
column is stored relative to that root (e.g.
`FB-5A_files/Images and Videos/3DImages/Colorado/CL0001C_Clean.obj`), so
preprocessing can locate every scan without editing a manifest.

## Manifest schema

| Manifest | Task | Columns |
|---|---|---|
| `ofc_manifest.csv` | cleft 4-class / binary | `scan_id, src_path, dataset, study_id, cleft_type, site, out_path` |
| `ofc_manifest_xsite.csv` | leave-one-site-out | as above + `subject_id` (`dataset\|study_id`) |
| `syndrome_manifest_b1.csv` | 19-class | `scan_id, src_path, fbid, syndrome_category, out_path` |
| `syndrome_clinical_manifest.csv` | 33-class | `scan_id, src_path, fbid, clinical_diagnosis, out_path` |
| `combined_manifest.csv` | combined screening | `scan_id, src_path, dataset, binary_label, out_path` |
| `controls_manifest.csv` | control cohorts | `scan_id, src_path, dataset, out_path` |
| `legacy_scanlevel/B{1,2}_scanlevel_manifest.csv` | leakage contrast, leaked arm | the B1/B2 manifests with `fbid` **removed** |

`out_path` is relative to the repository root
(`data/processed/<cohort>/<scan_id>.npy`) and is resolved at load time against
`FACEBENCH_DATA_ROOT`, which defaults to the repository root. Set it to put the
~2 GB of processed point clouds on scratch:

```bash
export FACEBENCH_DATA_ROOT=/scratch/facebench
```

## How the splits are made

`facebench/models/dataset.py` builds the partition at load time from the
manifest — there is no separate split file to drift out of sync.

- **Subject-level, stratified, 70/15/15**, shuffled with a fixed seed (42).
  Every scan of one subject lands in one partition.
- The subject key is the dataset-specific participant id (`fbid`, or
  `dataset|study_id` for cleft), paired with the dataset identifier because
  study ids are reused across FaceBase cohorts.
- Subjects whose repeated scans carry inconsistent labels are dropped before
  splitting.
- When a manifest has no subject-id column (the cleft manifests, and the
  deliberately leaky `legacy_scanlevel/` pair), the split falls back to
  scan-level stratification. For the cleft cohorts this is benign — one subject
  in 6,937 contributes more than one scan, and its two scans fall in validation
  and test. For `legacy_scanlevel/` it is the whole point: that arm exists to
  measure how much scan-level splitting inflates results.
- **Leave-one-site-out** (`--holdout-site`) makes the test set exactly one
  acquisition site and stratifies train/validation over the rest. It refuses to
  run if any subject appears both at the held-out site and in the training pool.

### One caveat worth reading before you compare numbers

`dataset.py` filters the manifest to rows whose `.npy` file exists *before*
splitting. The split is therefore a function of which processed files are
present. The released manifests already list only the scans that processed
successfully, so regenerating all of them reproduces the partition — but a
partial preprocessing run will silently produce a different split.

## Attrition

A small number of scans fail preprocessing (unreadable geometry, or too few
vertices after keeping the largest connected component): 12 scans for the
19-class task, 5 for the 33-class task, 361 for combined screening. All task
sizes quoted in the paper are post-attrition.
