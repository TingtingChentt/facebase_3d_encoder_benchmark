#!/usr/bin/env python3
"""
Fold a no-early-stopping retrain into the training-seed artefacts.
threads facebase3d-a2diag-2026-09-04 / facebase3d-a1diag-2026-09-04.

WHY A MERGE AND NOT A RERUN. aggregate_trainseed.py reads two fixed filenames
(per_seed_results_trainseed.csv, summary_ci_trainseed.csv) and groups by
base_task, so an A1_fix / A2_fix base_task lands beside A1 / A2 with nothing
else disturbed. Regenerating those files by re-running analyze_ci over every
task would cost a full bootstrap pass to reproduce 100 tasks' worth of numbers
byte for byte.  So the 20 new tasks are scored under their own suffix and
appended here.

THE APPEND IS PROVED NON-DESTRUCTIVE. Every row that is not one of this
family's tasks is compared cell by cell against the backup taken before the
write; any difference aborts before anything is written. Re-running is
idempotent: existing rows for the family are dropped first, so a second pass
replaces rather than duplicates them.

The truncated arm's rows stay. These retrains do not delete the evidence that
early stopping cut those runs short — both arms are needed to state why the row
moved.

Usage: python3 merge_fix_into_trainseed.py --family A2_fix
       (after analyze_ci.py --suffix _a2fix)
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

OUT = C.OUT
FILES = [("per_seed_results", True), ("summary_ci", True),
         ("contrast_tests", False)]


def merge(stem, required, suffix, new_tasks):
    dst = OUT / f"{stem}_trainseed.csv"
    src = OUT / f"{stem}{suffix}.csv"
    if not src.exists():
        if required:
            sys.exit(f"missing {src} — run analyze_ci.py --suffix {suffix} first")
        print(f"  {stem}: no {suffix} file, skipped")
        return
    try:
        add = pd.read_csv(src)
    except pd.errors.EmptyDataError:
        # analyze_ci writes contrast_tests<suffix>.csv even when a family has no
        # contrasts to run — A2/A1 are single-head, scan-level tasks, so the C1
        # head comparison and the C2 aggregation comparison have nothing to pair.
        # A zero-byte file is that expected outcome, not a failure.
        add = None
    if add is None or not len(add):
        if required:
            sys.exit(f"{src} is empty but {stem} is required")
        print(f"  {stem}: {suffix} file has no rows, skipped")
        return
    if not dst.exists():
        sys.exit(f"missing {dst}")

    bak = dst.with_suffix(f".csv.bak{suffix}")
    if not bak.exists():
        shutil.copy2(dst, bak)
    base = pd.read_csv(dst)

    keep = base[~base["task"].isin(new_tasks)].reset_index(drop=True)
    n_replaced = len(base) - len(keep)
    out = pd.concat([keep, add], ignore_index=True)

    # Non-destructiveness proof: the untouched rows must survive the round trip
    # unchanged, columns included.
    ref = pd.read_csv(bak)
    ref_keep = ref[~ref["task"].isin(new_tasks)].reset_index(drop=True)
    if list(out.columns) != list(ref.columns):
        sys.exit(f"{stem}: column set differs between {suffix} and _trainseed — "
                 f"{set(out.columns) ^ set(ref.columns)}")
    back = out[~out["task"].isin(new_tasks)].reset_index(drop=True)
    if not back.equals(ref_keep):
        sys.exit(f"{stem}: pre-existing rows changed — refusing to write")

    out.to_csv(dst, index=False)
    print(f"  {stem}: {len(ref_keep)} kept + {len(add)} {new_tasks and 'new'} "
          f"({n_replaced} replaced) -> {len(out)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(C.FIX_FAMILIES))
    a = ap.parse_args()
    new_tasks = set(C.fix_family(a.family)["tasks"])
    suffix = "_" + a.family.lower().replace("_", "")     # A2_fix -> _a2fix

    print(f"merging {len(new_tasks)} {a.family} tasks into the _trainseed artefacts")
    for stem, required in FILES:
        merge(stem, required, suffix, new_tasks)
    print("done — now run aggregate_trainseed.py")


if __name__ == "__main__":
    main()
