#!/usr/bin/env python3
"""
Does the cleft cohort's SCAN-level split actually leak?
thread facebase3d-paper-2026-08-08.

WHY THIS EXISTS. Methods claimed "the unit of analysis is the subject, not the
scan. Every split is over subject identifiers." That is true for the syndrome
and combined tasks but NOT as implemented for the cleft tasks: dataset.py sets
SUBJECT_ID_COL["ofc"] = None, which routes A1/A2 and the cross-site folds'
train/val split down the scan-level stratified branch. The draft mitigated this
in prose ("the cleft cohorts are unaffected in practice"), which is an argument
rather than a measurement. This measures it.

WHAT IT CHECKS, using the real Dataset objects rather than a reimplementation of
the split, so it cannot drift from what the models actually trained on:
  1. how many cleft subjects contribute more than one scan at all;
  2. for each such subject, which partitions its scans land in;
  3. the train-test and train-val subject intersections.

Subjects are keyed on the COMPOSITE (dataset, study_id). study_id alone is not a
person: 48 study_ids occur in both FB-56 and FB-5A, which are different cohorts
reusing an ID scheme.

The distinction that matters is train-vs-test. A subject split across VAL and
TEST cannot inflate the test metric by memorisation, because no scan of that
subject was ever trained on; at worst it couples model selection to the test
subject slightly. A subject split across TRAIN and TEST would be real leakage.

Usage: python3 audit_ofc_split.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "facebench/models"))
from dataset import FaceBaseDataset                         # noqa: E402

MANIFEST = PROJ / "data/processed/ofc_manifest.csv"
KEY = ["dataset", "study_id"]


def main():
    raw = pd.read_csv(MANIFEST)
    counts = raw.groupby(KEY).size()
    multi = counts[counts > 1]
    print(f"cleft manifest: {len(raw)} scans, {len(counts)} subjects "
          f"(composite {'+'.join(KEY)})")
    print(f"subjects contributing more than one scan: {len(multi)}")

    splits = {}
    for name in ("train", "val", "test"):
        d = FaceBaseDataset(str(MANIFEST), "ofc", split=name, seed=42)
        splits[name] = set(map(tuple, d.df[KEY].values))
        print(f"  {name:5} {len(d.df):5} scans, {len(splits[name]):5} subjects")

    straddle_train_test = 0
    for key in multi.index:
        where = [s for s in splits if tuple(key) in splits[s]]
        print(f"  subject {tuple(key)}: scans={int(multi[key])} -> {where}")
        if "train" in where and "test" in where:
            straddle_train_test += 1

    tt = splits["train"] & splits["test"]
    tv = splits["train"] & splits["val"]
    print(f"\ntrain-test subject intersection: {len(tt)}")
    print(f"train-val  subject intersection: {len(tv)}")
    print(f"subjects straddling train and test: {straddle_train_test}")

    verdict = ("NO train-test subject leakage: the scan-level split is "
               "equivalent to a subject-level one for this cohort"
               if len(tt) == 0 and straddle_train_test == 0
               else "TRAIN-TEST LEAKAGE PRESENT — the prose claim is false")
    print(f"\nVERDICT: {verdict}")
    return 0 if len(tt) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
