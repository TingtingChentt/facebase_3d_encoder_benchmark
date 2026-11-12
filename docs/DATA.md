# Data guide

## What is and is not in this repository

**Included**

- `data/splits/` — the subject-level partitions, as ID lists **without labels**:
  - `scans.csv` — every processed scan (26,990): `scan_id, dataset, subject_id,
    src_path, out_path, eligible`. `src_path` is relative to your FaceBase
    download; `out_path` is where `preprocess.py` writes the point cloud;
    `eligible` marks the 26,627 scans that are in the benchmark (see below).
  - `<task>.csv` — one file per task family, `scan_id, split`
    (`train`/`val`/`test`), in the row order of the manifest the models were
    trained on: `A1_fix_v3` (cleft subtyping), `A2_fix_v3` (cleft screening),
    `C_full_v3` (combined screening), `CF{10,15,20,30,40,50,75}_full_v3`
    (syndrome diagnosis at each class-size cutoff; `CF20` is the main task).
- `data/labels/alias_pairs_reviewed.csv` — the manual review of every pair of
  FB-TJ0 `Syndrome` and `Clinical Diagnosis` values used by the label curation
  (diagnosis strings and subject counts only, no subject IDs).
- `results/` — aggregate result tables (per-run metrics, bootstrap intervals,
  contrasts, class-level summaries); see `docs/RESULTS_MAP.md`.

**Not included**

- any 3D scan, mesh, texture, point cloud, facial landmark or trained checkpoint;
- per-subject diagnoses or cleft labels — the labelled manifests are rebuilt
  locally from the FaceBase metadata (next section);
- per-scan or per-subject model predictions and embeddings. The few per-patient
  tables that figures need (`results/scan_aggregation/`) carry random patient
  and class codes instead of FaceBase IDs and diagnoses
  (`facebench/evaluation/deidentify.py`);
- the exemplar subjects shown in the face-rendering figures
  (`data/exemplars.csv`; see `figures/exemplars.py`).

All scans come from the [FaceBase](https://www.facebase.org) consortium and are
obtained from FaceBase under its own access terms.

| ID | Accession | Format distributed |
|---|---|---|
| FB-5A | FB00001369 | OBJ meshes + participant / image-mapping tables |
| FB-56 | FB00001368 | OBJ meshes + participant / image-mapping tables |
| FB-TJ0 | FB00000861 | PLY surfaces + subject-level metadata table |
| FB-TK0 | FB00000892 | PLY |
| FB-VWP | FB00000491.01 | OBJ (+ demographic export) |
| FB-TX4 | FB00000667.01 | OBJ |

## Rebuilding the labelled manifests

```bash
export FACEBASE_ROOT=/path/to/facebase/download
python facebench/preprocessing/build_manifests.py   # -> data/manifests/*.csv
python facebench/preprocessing/check_splits.py      # partitions == data/splits/
```

`build_manifests.py` reads the shipped ID files and attaches labels from the
FaceBase metadata:

- **Cleft tasks:** `Cleft_Type` of `FB-5A_files/Data/OFC1_ALLIDs.csv` and
  `FB-56_files/Data/OFC2_ALLIDs.csv`, by StudyID (Unaffected, Cleft Lip, Cleft
  Palate, Cleft Lip and Palate). Binary screening collapses the three cleft types
  to Affected.
- **Combined screening:** cleft types → Affected, Unaffected → Unaffected;
  FB-TJ0 `Syndrome Category` Control → Unaffected, any other category →
  Affected (Uncategorized excluded); FB-TK0, FB-TX4 and FB-VWP → Unaffected.
- **Syndrome diagnosis:** the curated FB-TJ0 `Syndrome` field, built by the
  label chain in `facebench/preprocessing/labels/` (case variants merged;
  non-diagnostic values, Rett subtypes and the Moebius class excluded; subjects
  whose `Clinical Diagnosis` names a different disorder excluded after manual
  review). Classes with ≥50 scans and ≥20 subjects are kept: 49 classes, 7,472
  scans, 2,719 subjects. The same chain assigns the syndrome split.

Every manifest is checked against the shipped split file (same rows, same
order, same stored split). Re-running it on the FaceBase release used for the
paper reproduced all ten manifests used in training cell for cell.

## Preprocessing (`facebench/preprocessing/preprocess.py`)

One rule for all six datasets:

1. load the surface geometry only (no texture or material);
2. weld coincident vertices, so connectivity reflects the surface rather than
   the file's vertex duplication at texture seams;
3. connected components over the welded surface; keep the component with the
   largest area and add any component within 4% of the mesh bounding-box
   diagonal of the kept surface (keeps the adjacent pieces of a multi-camera
   capture, drops detached fragments and distant background);
4. area-weighted surface sampling to 3 × 4,096 points, farthest-point sampling
   to 4,096; centre on the centroid and scale to the unit sphere.

The RNG is seeded per scan from the scan ID, so preprocessing is deterministic.

## How the splits are made

`facebench/data/dataset.py` builds the partition at load time.

- **Syndrome tasks:** the split is stored in the manifest (`fixed_split`). It
  was assigned once per class over subjects (seed 42 + CRC32 of the class name,
  70/15/15, at least one subject per partition), so every scan of a subject is
  in one partition and the seven cutoffs have nested test sets.
- **Combined screening:** subject-level, stratified by label, 70/15/15, seed 42.
- **Cleft tasks:** scan-level, stratified by label, seed 42 (the cleft
  manifests carry no subject column; one subject in 6,872 has two scans).

The data-split seed is never varied; the five training seeds change only weight
initialisation, batch order and augmentation, so all five models of a task are
evaluated on the same test subjects.

## Which scans are in the benchmark (`eligible`)

Of the 26,990 processed scans, **26,627 are in the benchmark**. The other 363:

- 297 scans (FB-TK0 282, FB-TJ0 11, FB-VWP 2, FB-TX4 1, FB-5A 1) for which no
  usable point cloud is produced;
- 66 scans (FB-56 64, FB-TJ0 1, FB-TX4 1) that do yield a point cloud with the
  final preprocessing but were not in the task manifests when the models were
  trained; they are left out so that the partitions are exactly those of the
  paper.

Consequently the combined screening task has 24,612 scans (17,723 subjects) and
the cleft tasks 6,873 scans; the labelled combined-screening manifest has 24,973
rows before scans without a point cloud are removed. `dataset.py` drops manifest
rows whose point cloud is missing *before* drawing the combined and cleft splits,
so a partial preprocessing run would silently change those partitions;
`check_splits.py` and the shipped split files guard against that.
