#!/usr/bin/env python3
"""
Regression guard for the 2026-08-07 secondary-task extension.

Adding the seven secondary tasks meant re-running analyze_ci.py over all eleven
tasks, which rewrites summary_ci.csv / per_seed_results.csv / contrast_tests.csv
wholesale. The four priority tasks scored on 08-04 MUST come back bit-identical:
their prediction files were not regenerated, the bootstrap draws are seeded on
n_subjects alone, and nothing in the estimator changed. Any drift means the
scoping change leaked into the core claim, so this fails loudly.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

BACKUP = C.OUT / "_backup_20260807"
KEYS = {
    "summary_ci.csv": ["task", "encoder", "head", "level", "metric"],
    "per_seed_results.csv": ["task", "encoder", "head", "level", "seed", "metric"],
    "contrast_tests.csv": ["contrast", "task", "metric"],
}

failed = False
for name, keys in KEYS.items():
    old = pd.read_csv(BACKUP / name)
    new = pd.read_csv(C.OUT / name)
    tasks = sorted(old.task.unique())
    sub = new[new.task.isin(tasks)]
    o = old.set_index(keys).sort_index()
    n = sub.set_index(keys).sort_index()

    print(f"\n{name}: {len(old)} old rows ({len(tasks)} tasks) "
          f"-> {len(new)} new rows, {len(sub)} on the old tasks")
    if len(o) != len(n) or not o.index.equals(n.index):
        print(f"  FAIL: row set changed  (old {len(o)} / new {len(n)})")
        print(f"    dropped: {sorted(set(o.index) - set(n.index))[:5]}")
        print(f"    added:   {sorted(set(n.index) - set(o.index))[:5]}")
        failed = True
        continue

    # bool columns (seed_deterministic, ci_contains_zero, ...) are numeric to
    # pandas but do not subtract; compare them for equality instead.
    num = [c for c in o.columns if pd.api.types.is_numeric_dtype(o[c])
           and not pd.api.types.is_bool_dtype(o[c])]
    nonnum = [c for c in o.columns if c not in num]
    delta = (o[num] - n[num]).abs().max()
    worst = float(delta.max())
    mismatched = [c for c in nonnum if not o[c].equals(n[c])]
    print(f"  rows identical; max abs delta over {len(num)} numeric columns "
          f"= {worst:.3e}; {len(nonnum)} non-numeric columns compared exactly")
    if worst != 0.0 or mismatched:
        print("  FAIL: priority-task numbers moved")
        if worst != 0.0:
            print(delta[delta > 0].to_string())
        if mismatched:
            print(f"  non-numeric columns differing: {mismatched}")
        failed = True
    else:
        print("  OK — exactly reproduced")

print("\nFAIL" if failed else "\nPASS — the four priority tasks are unchanged")
sys.exit(1 if failed else 0)
