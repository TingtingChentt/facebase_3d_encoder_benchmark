#!/usr/bin/env python3
"""Build the labelled task manifests from the FaceBase metadata and the shipped splits.

The repository ships subject-level partitions WITHOUT labels (data/splits/):

  scans.csv         every processed scan: scan_id, dataset, subject_id, src_path,
                    out_path, eligible (in the benchmark or not)
  <task>.csv        scan_id, split  — one file per task family, in the row order
                    of the manifest the models were trained on

Labels are re-derived here from the FaceBase metadata (FACEBASE_ROOT), so a
user with FaceBase access gets exactly the manifests used in the paper:

  data/manifests/ofc_manifest.csv              cleft subtyping / cleft screening
  data/manifests/combined_manifest.csv         combined affected-vs-unaffected screening
  data/manifests/syndrome_k{K}_manifest.csv    syndrome diagnosis (K = 20 is the main task)

Label rules
  cleft    `Cleft_Type` of OFC1_ALLIDs.csv (FB-5A) / OFC2_ALLIDs.csv (FB-56), by StudyID
  combined cleft types -> Affected, Unaffected -> Unaffected; FB-TJ0 `Syndrome
           Category` Control -> Unaffected, any other category -> Affected;
           FB-TK0, FB-TX4, FB-VWP -> Unaffected
  syndrome the label chain in facebench/preprocessing/labels/ (FB-TJ0 `Syndrome`
           field, curated; see the README there), run on the eligible FB-TJ0 scans

Every manifest is checked against the shipped split file: same rows, same order,
and (syndrome tasks) the same stored `fixed_split`. The cleft and combined
partitions are drawn at load time by facebench/data/dataset.py (stratified,
seed 42); `python -m facebench.preprocessing.check_splits` confirms that they
reproduce the shipped files.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import pandas as pd

from facebench.paths import MANIFESTS, ROOT, SPLITS, facebase

LABELS_DIR = Path(__file__).resolve().parent / "labels"
LABEL_BUILD = ROOT / "data/labels/build"
CF_CUTOFFS = [10, 15, 20, 30, 40, 50, 75]
CLEFT_TYPES = {"Unaffected", "Cleft Lip", "Cleft Palate", "Cleft Lip and Palate"}
OFC_SOURCES = {  # dataset -> (FaceBase file, subject-id prefix used in scans.csv)
    "FB-5A": ("FB-5A_files/Data/OFC1_ALLIDs.csv", "fb5a_"),
    "FB-56": ("FB-56_files/Data/OFC2_ALLIDs.csv", "fb56_"),
}
TJ0_META = "FB-TJ0_files/FB00000861_metadata_2020-09-14.csv"


def split_file(task):
    return pd.read_csv(SPLITS / f"{task}.csv")


def in_split_order(df, task):
    """Rows of `df` restricted to, and ordered as, the shipped split file."""
    sp = split_file(task)
    out = sp[["scan_id"]].merge(df, on="scan_id", how="left", validate="1:1")
    missing = out.drop(columns="scan_id").isna().all(axis=1)
    assert not missing.any(), f"{task}: {missing.sum()} split rows have no label"
    return out, sp


def cleft_labels(scans):
    rows = []
    for ds, (rel, prefix) in OFC_SOURCES.items():
        ids = pd.read_csv(facebase(rel), encoding="latin-1", low_memory=False,
                          usecols=["StudyID", "Cleft_Type"]).dropna(subset=["Cleft_Type"])
        lab = dict(zip(ids["StudyID"].astype(str), ids["Cleft_Type"]))
        sub = scans[scans.dataset == ds].copy()
        sub["study_id"] = sub.subject_id.str.removeprefix(prefix)
        sub["cleft_type"] = sub.study_id.map(lab)
        sub["site"] = sub.study_id.map(lambda s: (re.match(r"([A-Z]{2})", s) or [None, "UNK"])[1])
        rows.append(sub)
    return pd.concat(rows)


def ofc_manifest(scans):
    lab = cleft_labels(scans)
    cols = ["scan_id", "src_path", "dataset", "study_id", "cleft_type", "site", "out_path"]
    m, _ = in_split_order(lab[cols], "A1_fix_v3")
    assert m.cleft_type.isin(CLEFT_TYPES).all(), "unexpected cleft type in the benchmark rows"
    m2, _ = in_split_order(lab[cols], "A2_fix_v3")
    assert m.scan_id.equals(m2.scan_id), "the two cleft tasks share one manifest"
    return m[cols]


def combined_manifest(scans, ofc):
    meta = pd.read_csv(facebase(TJ0_META), usecols=["FBID", "Syndrome Category"])
    cat = dict(zip(meta.FBID, meta["Syndrome Category"]))
    lab = scans.copy()
    lab["binary_label"] = None
    o = lab.scan_id.map(dict(zip(ofc.scan_id, ofc.cleft_type)))
    is_ofc = lab.dataset.isin(OFC_SOURCES)
    lab.loc[is_ofc, "binary_label"] = o[is_ofc].map(
        lambda t: None if pd.isna(t) else ("Unaffected" if t == "Unaffected" else "Affected"))
    tj = lab.dataset == "FB-TJ0"
    c = lab.loc[tj, "subject_id"].str.removeprefix("fbtj0_").map(cat)
    lab.loc[tj, "binary_label"] = c.map(
        lambda v: None if pd.isna(v) or v == "Uncategorized" else
        ("Unaffected" if v == "Control" else "Affected"))
    lab.loc[lab.dataset.isin(["FB-TK0", "FB-TX4", "FB-VWP"]), "binary_label"] = "Unaffected"
    cols = ["scan_id", "src_path", "dataset", "binary_label", "out_path", "subject_id"]
    m, _ = in_split_order(lab[cols], "C_full_v3")
    assert m.binary_label.isin(["Affected", "Unaffected"]).all()
    return m[cols]


def syndrome_manifests(scans):
    """Run the label chain on the eligible FB-TJ0 scans, then check it against the splits."""
    LABEL_BUILD.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(facebase(TJ0_META), usecols=["FBID", "Syndrome Category"])
    tj = scans[scans.dataset == "FB-TJ0"].copy()
    tj["fbid"] = tj.subject_id.str.removeprefix("fbtj0_")
    tj["syndrome_category"] = tj.fbid.map(dict(zip(meta.FBID, meta["Syndrome Category"])))
    tj[["scan_id", "src_path", "fbid", "syndrome_category", "out_path", "eligible"]].to_csv(
        LABEL_BUILD / "fbtj0_cohort.csv", index=False)
    for step in ["build_clinical_filled_manifest.py", "build_syndrome_name_manifest.py",
                 "build_syndrome_final_manifest.py", "build_class_filter_manifests.py"]:
        print(f"  label chain: {step}")
        subprocess.run([sys.executable, str(LABELS_DIR / step)], check=True,
                       stdout=subprocess.DEVNULL)
    out = {}
    cols = ["scan_id", "src_path", "fbid", "syndrome_name", "out_path", "fixed_split"]
    for k in CF_CUTOFFS:
        built = pd.read_csv(LABEL_BUILD / f"class_filter/syndrome_k{k}_manifest.csv")
        m, sp = in_split_order(built[cols], f"CF{k}_full_v3")
        assert len(built) == len(sp), f"K={k}: label chain gave {len(built)} rows, split has {len(sp)}"
        assert (m.fixed_split.values == sp.split.values).all(), f"K={k}: fixed_split differs"
        out[k] = m[cols]
    return out


def main():
    scans = pd.read_csv(SPLITS / "scans.csv")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    print("cleft labels ...")
    ofc = ofc_manifest(scans)
    ofc.to_csv(MANIFESTS / "ofc_manifest.csv", index=False)
    print(f"  ofc_manifest.csv  {len(ofc):,} rows")
    print("combined labels ...")
    comb = combined_manifest(scans, cleft_labels(scans))
    comb.to_csv(MANIFESTS / "combined_manifest.csv", index=False)
    print(f"  combined_manifest.csv  {len(comb):,} rows")
    print("syndrome labels ...")
    for k, m in syndrome_manifests(scans).items():
        m.to_csv(MANIFESTS / f"syndrome_k{k}_manifest.csv", index=False)
        print(f"  syndrome_k{k}_manifest.csv  {len(m):,} rows, {m.syndrome_name.nunique()} classes")
    print(f"manifests -> {MANIFESTS}")


if __name__ == "__main__":
    main()
