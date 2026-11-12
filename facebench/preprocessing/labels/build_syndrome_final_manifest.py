#!/usr/bin/env python3
"""
Build syndrome_final_manifest.csv: the `Syndrome` column as the label, checked
against `Clinical Diagnosis`.

Decided 2026-10-07, after a label-source audit of Syndrome vs Clinical Diagnosis: use the Syndrome column directly,
not a "CD filled" label. CD is used only as a check, to drop subjects whose CD
names a different disorder.

Rules = the n50 rules (build_syndrome_name_manifest.py, imported so they
cannot drift) plus three exclusions, applied BEFORE the class-size filter so
the filter sees the final counts:
  1-5. n50: processed pool, case-merge, drop non-diagnostic values, Cleft
       Lip/Palate and Rett subtypes.
  A.   Drop the whole "Moebius Syndrome" class. The value looks like a
       placeholder for "uncategorised chromosomal": most of its subjects have
       a CD that describes a CNV or another condition (see the report).
  B.   Drop subjects on the builder's conflict list (check == "conflict" in
       clinical_filled_subjects.csv; 22 subjects in metadata).
  C.   Drop subjects whose (Syndrome, CD) pair the manual alias review marked
       "conflict" (alias_pairs_reviewed.csv; 3 pairs, 4 subjects).
  6.   Keep classes with >= MIN_SCANS scans AND >= MIN_SUBJECTS subjects.

Writes:
  data/labels/build/syndrome_final_manifest.csv   one row per scan
  data/labels/build/syndrome_final_excluded.csv   one row per subject removed by A-C
No existing manifest is modified.
"""
import importlib.util
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LABELS = PROJECT_ROOT / "data/labels"          # manual review table (shipped)
PROCESSED = LABELS / "build"                     # intermediate outputs
PROCESSED.mkdir(parents=True, exist_ok=True)
OUT = PROCESSED / "syndrome_final_manifest.csv"
OUT_EXCL = PROCESSED / "syndrome_final_excluded.csv"   # subject-level, so local only
SUBJ = PROCESSED / "clinical_filled_subjects.csv"
REVIEW = LABELS / "alias_pairs_reviewed.csv"   # manual review of Syndrome vs Clinical Diagnosis pairs

_spec = importlib.util.spec_from_file_location(
    "n50", Path(__file__).parent / "build_syndrome_name_manifest.py")
N50 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(N50)

EXCLUDE_CLASSES = {"Moebius Syndrome"}


def main():
    df = pd.read_csv(N50.SRC)
    df = df[df["eligible"]].reset_index(drop=True)   # scans in the benchmark (data/splits/scans.csv)
    meta = pd.read_csv(N50.META, usecols=["FBID", "Syndrome"])
    df = df.merge(meta, left_on="fbid", right_on="FBID", how="left")
    assert df["Syndrome"].notna().all(), "scan with no Syndrome value"
    df["Syndrome"] = df["Syndrome"].str.strip()
    key = df["Syndrome"].str.lower()
    canon = (df.assign(k=key).groupby(["k", "Syndrome"]).fbid.nunique()
               .reset_index().sort_values("fbid")
               .drop_duplicates("k", keep="last").set_index("k")["Syndrome"])
    df["syndrome_name"] = key.map(canon)
    df = df[~df["syndrome_name"].isin(N50.EXCLUDE)]
    print(f"after n50 exclusions: {len(df)} scans, {df.fbid.nunique()} subjects")

    subj = pd.read_csv(SUBJ)
    rev = pd.read_csv(REVIEW)
    rconf = rev[rev.verdict == "conflict"][["syndrome", "clinical_diagnosis", "note"]]
    rconf_ids = subj.merge(rconf, on=["syndrome", "clinical_diagnosis"])
    assert len(rconf_ids) == rconf.merge(rev)["n_subjects"].sum(), "review pair lookup"

    reasons = {}
    for f in df.loc[df.syndrome_name.isin(EXCLUDE_CLASSES), "fbid"]:
        reasons[f] = "A_moebius_class"
    for f in subj.loc[subj.check == "conflict", "fbid"]:
        reasons.setdefault(f, "B_listed_conflict")
    for f in rconf_ids.fbid:
        reasons.setdefault(f, "C_review_conflict")

    hit = df[df.fbid.isin(reasons)]
    excl = (hit.groupby("fbid").agg(syndrome_name=("syndrome_name", "first"),
                                    n_scans=("scan_id", "size"))
               .reset_index()
               .merge(subj[["fbid", "clinical_diagnosis"]], on="fbid", how="left"))
    excl["reason"] = excl.fbid.map(reasons)
    excl = excl.sort_values(["reason", "syndrome_name"])
    excl.to_csv(OUT_EXCL, index=False)
    print("excluded (in pool):")
    print(excl.groupby("reason").agg(subjects=("fbid", "size"),
                                     scans=("n_scans", "sum")).to_string())
    df = df[~df.fbid.isin(reasons)]

    c = df.groupby("syndrome_name").agg(scans=("scan_id", "size"),
                                        subj=("fbid", "nunique"))
    keep = c[(c.scans >= N50.MIN_SCANS) & (c.subj >= N50.MIN_SUBJECTS)].index
    df = df[df["syndrome_name"].isin(keep)].reset_index(drop=True)
    out = df[["scan_id", "src_path", "fbid", "syndrome_name", "out_path"]]
    out.to_csv(OUT, index=False)
    print(f"kept: {len(keep)} classes, {len(out)} scans, "
          f"{out.fbid.nunique()} subjects -> {OUT}")
    print(c.loc[keep].sort_values("subj", ascending=False).to_string())


if __name__ == "__main__":
    main()
