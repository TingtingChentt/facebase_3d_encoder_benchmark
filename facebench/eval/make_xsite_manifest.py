#!/usr/bin/env python3
"""
Emit ofc_manifest_xsite.csv = ofc_manifest.csv + one added `subject_id` column.
thread facebase3d-paper-2026-08-08, PART A.

WHY A SUBJECT COLUMN AT ALL. The bootstrap unit for every interval in this paper
is the TEST SUBJECT, not the scan. ofc_manifest.csv has no subject column, so
extract_features.py falls back to a synthetic per-scan id and the bootstrap
degenerates to resampling scans. For the OFC cohort that is very nearly exact,
but "very nearly" is the kind of thing this project has spent two weeks
retiring, so the subject is made explicit instead of assumed.

WHY A COMPOSITE KEY. study_id alone is NOT a subject identifier: 48 study_ids
occur in both FB-56 and FB-5A, which are two different cohorts reusing the same
ID scheme, so grouping on study_id would merge distinct people. Keying on
(dataset, study_id) resolves that. Under the composite key exactly ONE subject
in the whole 6,873-scan cohort has two scans (at PH); under bare study_id it
looks like 49. No composite subject spans two sites, so the site split remains
leakage-free by construction.

THE SPLITS ARE UNCHANGED. FaceBaseDataset ignores subject_id for exp="ofc"
(SUBJECT_ID_COL["ofc"] is None), and the cross-site branch splits on `site` and
the label only. This script VERIFIES that rather than asserting it: for every
viable fold it builds both manifests' train/val/test and requires the row sets
to be identical. Training therefore keeps reading ofc_manifest.csv; only feature
extraction reads this file.

Usage: python3 make_xsite_manifest.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
SRC = PROJECT / "data/processed/ofc_manifest.csv"
DST = PROJECT / "data/processed/ofc_manifest_xsite.csv"
AUDIT = PROJECT / "results/xsite_fold_audit.csv"

sys.path.insert(0, str(PROJECT / "facebench/models"))
from dataset import FaceBaseDataset                          # noqa: E402


def main():
    df = pd.read_csv(SRC)
    if "subject_id" in df.columns:
        raise SystemExit(f"{SRC} already has a subject_id column")
    df["subject_id"] = (df["dataset"].astype(str) + "|"
                        + df["study_id"].astype(str))
    df.to_csv(DST, index=False)
    print(f"wrote {DST}  ({len(df)} rows, +1 column)")

    viable = pd.read_csv(AUDIT).query("viable").site.astype(str).tolist()
    print(f"\nverifying splits unchanged over {len(viable)} viable folds...")
    ok = True
    for site in viable:
        for split in ("train", "val", "test"):
            kw = dict(exp="ofc", split=split, seed=42, augment=False,
                      holdout_site=site)
            a = FaceBaseDataset(manifest_csv=str(SRC), **kw)
            b = FaceBaseDataset(manifest_csv=str(DST), **kw)
            same = (len(a) == len(b) and
                    a.df["scan_id"].equals(b.df["scan_id"]))
            ok &= same
            if not same:
                print(f"  MISMATCH {site}/{split}: {len(a)} vs {len(b)}")
        t = FaceBaseDataset(manifest_csv=str(DST), exp="ofc", split="test",
                            seed=42, augment=False, holdout_site=site)
        print(f"  {site:3s} test scans={len(t):5d} "
              f"subjects={t.df['subject_id'].nunique():5d}")
    print(f"\n=> SPLITS IDENTICAL: {ok}")
    if not ok:
        sys.exit("split mismatch — do not use this manifest")


if __name__ == "__main__":
    main()
