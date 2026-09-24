#!/usr/bin/env python3
"""
Cross-site fold audit + regression guard for the holdout_site branch.
thread facebase3d-paper-2026-08-08, PART A.

Two jobs, both of which must pass before any training is launched:

  1. REGRESSION. holdout_site=None must reproduce the pre-existing random split
     EXACTLY — same row sets, same order, for train/val/test. The cross-site
     work must not perturb any number already in the manuscript. Checked against
     the pre-edit dataset.py kept as dataset_pre_holdout_backup.py, so this
     compares against the actual old code rather than against a remembered
     description of it.

  2. FOLD VIABILITY. For every site, instantiate the real Dataset objects and
     report post-attrition train/val/test sizes and per-class TEST counts. The
     manifest count is PRE-attrition; the gate is defined on what survives.

GATE: any site whose TEST split has a class with fewer than MIN_TEST_PER_CLASS
scans is reported NOT VIABLE and must be dropped from the sweep.

Usage: python3 audit_xsite_folds.py
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
MODELS = PROJECT / "facebench/models"
MANIFEST = PROJECT / "data/processed/ofc_manifest.csv"
EXP_TYPE = "ofc"
SEED = 42
MIN_TEST_PER_CLASS = 5

sys.path.insert(0, str(MODELS))
from dataset import FaceBaseDataset                        # noqa: E402


def load_old_dataset_cls():
    """Import the pre-edit dataset.py under its own module name so both the old
    and new FaceBaseDataset can be held in memory at once."""
    p = MODELS / "dataset_pre_holdout_backup.py"
    spec = importlib.util.spec_from_file_location("dataset_old", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FaceBaseDataset


def regression_check():
    print("=" * 78)
    print("1. REGRESSION — holdout_site=None vs. pre-edit dataset.py")
    print("=" * 78)
    Old = load_old_dataset_cls()
    ok = True
    for split in ("train", "val", "test"):
        kw = dict(manifest_csv=str(MANIFEST), exp=EXP_TYPE, split=split,
                  seed=SEED, augment=False)
        old = Old(**kw)
        new = FaceBaseDataset(**kw)          # holdout_site defaults to None
        same_n = len(old) == len(new)
        same_rows = old.df.equals(new.df)    # values AND order
        same_map = old.label_map == new.label_map
        print(f"  {split:5s} n_old={len(old):5d} n_new={len(new):5d}  "
              f"rows_identical={same_rows}  label_map_identical={same_map}")
        ok &= same_n and same_rows and same_map
    print(f"  => REGRESSION {'PASS' if ok else 'FAIL'}\n")
    return ok


def fold_audit():
    print("=" * 78)
    print("2. FOLD VIABILITY — post-attrition, from the real Dataset objects")
    print("=" * 78)
    df = pd.read_csv(MANIFEST)
    sites = sorted(df["site"].astype(str).unique())
    site_ds = (df.groupby("site")["dataset"]
                 .agg(lambda s: "+".join(sorted(s.unique()))).to_dict())

    rows = []
    for site in sites:
        kw = dict(manifest_csv=str(MANIFEST), exp=EXP_TYPE, seed=SEED,
                  augment=False, holdout_site=site)
        tr = FaceBaseDataset(**kw, split="train")
        va = FaceBaseDataset(**kw, split="val")
        te = FaceBaseDataset(**kw, split="test")
        counts = te.class_counts()
        thin = {c: n for c, n in counts.items() if n < MIN_TEST_PER_CLASS}
        rows.append(dict(
            site=site, datasets=site_ds[site],
            n_train=len(tr), n_val=len(va), n_test=len(te),
            **{f"test_{c}": n for c, n in counts.items()},
            min_test_class=min(counts.values()),
            viable=not thin,
        ))
        # Splits must partition the usable manifest exactly once.
        assert len(tr) + len(va) + len(te) == len(tr.df) + len(va.df) + len(te.df)

    out = pd.DataFrame(rows)
    total = out[["n_train", "n_val", "n_test"]].sum(1).unique()
    print(out.to_string(index=False))
    print(f"\n  train+val+test per fold (must be constant): {total}")

    viable = out[out.viable].site.tolist()
    dropped = out[~out.viable]
    print(f"\n  VIABLE folds ({len(viable)}): {', '.join(viable)}")
    print(f"  NOT VIABLE  ({len(dropped)}):")
    for _, r in dropped.iterrows():
        thin = {c.replace('test_', ''): r[c] for c in out.columns
                if c.startswith('test_') and r[c] < MIN_TEST_PER_CLASS}
        print(f"    {r.site:3s} min_class={r.min_test_class:3d}  thin={thin}")

    # The two fold groups the verdict must be split by.
    both = [s for s in viable if "+" in site_ds[s]]
    single = [s for s in viable if "+" not in site_ds[s]]
    print(f"\n  GROUP 1 — site isolated from dataset (both datasets remain in "
          f"training): {', '.join(both)}")
    print(f"  GROUP 2 — site and dataset confounded: {', '.join(single)}")

    out.to_csv(PROJECT / "results/xsite_fold_audit.csv", index=False)
    print(f"\n  -> results/xsite_fold_audit.csv")
    return viable, both, single


if __name__ == "__main__":
    ok = regression_check()
    fold_audit()
    if not ok:
        sys.exit("REGRESSION FAILED — do not launch training")
