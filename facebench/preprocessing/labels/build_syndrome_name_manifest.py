#!/usr/bin/env python3
"""
Build syndrome_name_manifest.csv: FB-TJ0 scans labelled by the metadata
`Syndrome` column.

Why this column: `Syndrome` is filled for 100% of FB-TJ0 subjects and uses
standard names, whereas `Clinical Diagnosis` (the d33 label) is free text,
blank for ~49% of subjects, and splits one condition across many spellings.
The 19-class (Syndrome Category) and 33-class (Clinical Diagnosis) manifests
are NOT touched by this script.

Rules (fixed 2026-09-30, before training):
  1. Pool = syndrome_manifest.csv (all processed FB-TJ0 scans) with an existing .npy.
  2. Case-only spelling variants are merged to the most frequent spelling.
  3. Excluded (not a diagnosis): Relative, Unaffected Unrelated, Unknown,
     Facial Dysmorphia, possibly genetic.
  4. Excluded: Cleft Lip/Palate (non-syndromic; covered by the cleft tasks).
  5. Excluded: Rett subtypes (Rett Syndrome_Other/_FOXG1/_CDKL5); classic
     "Rett Syndrome" is kept.
  6. Keep classes with >= MIN_SCANS scans AND >= MIN_SUBJECTS subjects.
"""
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS = PROJECT_ROOT / "data/labels"          # manual review table (shipped)
PROCESSED = LABELS / "build"                     # intermediate outputs
SRC = PROCESSED / "fbtj0_cohort.csv"   # FB-TJ0 scan list + `Syndrome Category`, written by
                                     # facebench/preprocessing/build_manifests.py
OUT = PROCESSED / "syndrome_name_manifest.csv"
META = (Path(os.environ.get("FACEBASE_ROOT", "FACEBASE_ROOT_not_set"))
        / "FB-TJ0_files/FB00000861_metadata_2020-09-14.csv")

MIN_SCANS = 50
MIN_SUBJECTS = 20

EXCLUDE = {
    "Relative", "Unaffected Unrelated", "Unknown",
    "Facial Dysmorphia, possibly genetic",
    "Cleft Lip/Palate",
    "Rett Syndrome_Other", "Rett Syndrome_FOXG1", "Rett Syndrome_CDKL5",
}


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    df = df[df["eligible"]].reset_index(drop=True)   # scans in the benchmark (data/splits/scans.csv)
    meta = pd.read_csv(META, usecols=["FBID", "Syndrome"])
    df = df.merge(meta, left_on="fbid", right_on="FBID", how="left")
    assert df["Syndrome"].notna().all(), "scan with no Syndrome value"
    df["Syndrome"] = df["Syndrome"].str.strip()
    print(f"pool: {len(df)} scans, {df.fbid.nunique()} subjects")

    # case-only variants -> most frequent spelling (by subject count)
    key = df["Syndrome"].str.lower()
    canon = (df.assign(k=key).groupby(["k", "Syndrome"]).fbid.nunique()
               .reset_index().sort_values("fbid")
               .drop_duplicates("k", keep="last").set_index("k")["Syndrome"])
    df["syndrome_name"] = key.map(canon)
    merged = (df["syndrome_name"] != df["Syndrome"]).sum()
    print(f"case-merged: {merged} scans relabelled")

    df = df[~df["syndrome_name"].isin(EXCLUDE)]
    c = df.groupby("syndrome_name").agg(scans=("scan_id", "size"),
                                        subj=("fbid", "nunique"))
    keep = c[(c.scans >= MIN_SCANS) & (c.subj >= MIN_SUBJECTS)].index
    df = df[df["syndrome_name"].isin(keep)].reset_index(drop=True)

    out = df[["scan_id", "src_path", "fbid", "syndrome_name", "out_path"]]
    out.to_csv(OUT, index=False)
    print(f"kept: {len(keep)} classes, {len(out)} scans, "
          f"{out.fbid.nunique()} subjects -> {OUT}")
    print(c.loc[keep].sort_values("subj", ascending=False).to_string())


if __name__ == "__main__":
    main()
