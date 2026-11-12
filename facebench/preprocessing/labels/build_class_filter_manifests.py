#!/usr/bin/env python3
"""
Build the class-size cutoff manifests for the Syndrome label 


2026-10-08: show how performance changes with the class-size cutoff (not
to choose a best cutoff). One run per cutoff x 4 encoders x 5 train seeds, early
stopping off.

ONE SHARED SPLIT. Every subject in the pre-filter pool whose class could pass
any cutoff is assigned train/val/test ONCE, class by class, with its own rng
per class (seed 42 + crc32 of the class name). A class's assignment therefore
does not depend on which other classes are in the task, so a higher cutoff's
splits are exactly the lower cutoff's splits restricted to fewer classes, and
the test sets are nested. Per class: n_test = max(1, int(0.15 n)),
n_val = max(1, int(0.15 n)), the same fractions as dataset.py.
The split is written to a `fixed_split` column, which dataset.py honours.

Pool = pool.build_pool() (the final Syndrome
builder's rules up to, not including, the class-size filter). The 20-subject
cutoff must reproduce data/labels/build/syndrome_final_manifest.csv scan for scan.

Writes data/labels/build/class_filter/:
  split_subjects.csv              one row per subject (class, fixed_split)
  syndrome_k{K}_manifest.csv      one per cutoff (>=50 scans AND >=K subjects)
No existing manifest is modified.
"""
import importlib.util
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
OUT = PROJ / "data/labels/build/class_filter"
SPLIT_SEED = 42
TEST_FRAC = VAL_FRAC = 0.15
MIN_SCANS = 50
CUTOFFS = [10, 15, 20, 30, 40, 50, 75]

_spec = importlib.util.spec_from_file_location(
    "cfr", Path(__file__).parent / "pool.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def assign(subjects, cls):
    rng = np.random.default_rng([SPLIT_SEED, zlib.crc32(cls.encode())])
    s = sorted(subjects)
    rng.shuffle(s)
    n = len(s)
    nt, nv = max(1, int(n * TEST_FRAC)), max(1, int(n * VAL_FRAC))
    return {**{x: "test" for x in s[:nt]}, **{x: "val" for x in s[nt:nt + nv]},
            **{x: "train" for x in s[nt + nv:]}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, _ = R.build_pool()
    c = df.groupby("syndrome_name").agg(scans=("scan_id", "size"), subj=("fbid", "nunique"))
    eligible = c[(c.scans >= MIN_SCANS) & (c.subj >= min(CUTOFFS))].index

    split = {}
    for cls in eligible:
        split.update(assign(df.loc[df.syndrome_name == cls, "fbid"].unique(), cls))
    df = df[df.syndrome_name.isin(eligible)].copy()
    df["fixed_split"] = df.fbid.map(split)
    assert df.fixed_split.notna().all()
    assert (df.groupby("fbid").fixed_split.nunique() == 1).all()
    assert (df.groupby("fbid").syndrome_name.nunique() == 1).all()

    (df.drop_duplicates("fbid")[["fbid", "syndrome_name", "fixed_split"]]
       .sort_values(["syndrome_name", "fbid"])
       .to_csv(OUT / "split_subjects.csv", index=False))

    cols = ["scan_id", "src_path", "fbid", "syndrome_name", "out_path", "fixed_split"]
    prev_test = None
    print(f"{'K':>4} {'classes':>8} {'subj':>6} {'scans':>6} {'train':>6} {'val':>5} "
          f"{'test':>5} {'min test/cls':>12}")
    for k in CUTOFFS:
        keep = c[(c.scans >= MIN_SCANS) & (c.subj >= k)].index
        m = df[df.syndrome_name.isin(keep)][cols].reset_index(drop=True)
        sub = m.drop_duplicates("fbid")
        per = sub.groupby(["syndrome_name", "fixed_split"]).size().unstack(fill_value=0)
        assert (per[["train", "val", "test"]] >= 1).all().all(), f"K={k}: empty split"
        test = set(sub.loc[sub.fixed_split == "test", "fbid"])
        if prev_test is not None:
            assert test <= prev_test, f"K={k}: test set not nested"
        prev_test = test
        if k == 20:
            fin = pd.read_csv(PROJ / "data/labels/build/syndrome_final_manifest.csv")
            assert set(m.scan_id) == set(fin.scan_id) and len(m) == len(fin)
        m.to_csv(OUT / f"syndrome_k{k}_manifest.csv", index=False)
        n = sub.fixed_split.value_counts()
        print(f"{k:>4} {len(keep):>8} {len(sub):>6} {len(m):>6} {n['train']:>6} "
              f"{n['val']:>5} {n['test']:>5} {per['test'].min():>12}")


if __name__ == "__main__":
    main()
