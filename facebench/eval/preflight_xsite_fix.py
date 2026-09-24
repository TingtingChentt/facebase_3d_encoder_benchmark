#!/usr/bin/env python3
"""
Gate between the cross-site no-early-stopping retrain and the scoring stages.
thread facebase3d-xsitediag-2026-09-07.

Checks, for all 48 runs (6 folds x 4 encoders x 2 label collapses):
  * the checkpoint exists;
  * patience == epochs, so the early-stopping branch provably never fired;
  * holdout_site is the fold the task name claims, and split_seed is 42;
  * n_train / n_val / n_test match the published XS_*/XSB_* run being replaced,
    so the held-out site is the same population and only the stopping rule
    moved.

Exits non-zero on any failure: a fold missing from a six-fold comparison would
quietly change which sites the conclusion rests on.

Usage: python3 preflight_xsite_fix.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

SPLIT_KEYS = ["n_train", "n_val", "n_test"]

missing, bad, ok = [], [], 0
for task in C.XSITE_FIX_TASKS + C.XSITE_FIX_BIN_TASKS:
    t = C.TASKS[task]
    ref_task = t["legacy_base"]
    for enc in C.task_encoders(task):
        ck = C.checkpoint_path(task, enc)
        if not ck.exists():
            missing.append(f"{task}/{enc}: no checkpoint at {ck}")
            continue
        cfg_p = C.RUNS / t["run_dir"] / enc / "config.json"
        try:
            cfg = json.loads(cfg_p.read_text())
        except Exception as e:                      # noqa: BLE001
            bad.append(f"{task}/{enc}: unreadable config.json ({e})")
            continue
        if cfg.get("patience") != cfg.get("epochs"):
            bad.append(f"{task}/{enc}: patience={cfg.get('patience')} vs "
                       f"epochs={cfg.get('epochs')} — early stopping could "
                       f"still fire, this is not a fixed run")
        if cfg.get("holdout_site") != t["holdout_site"]:
            bad.append(f"{task}/{enc}: holdout_site={cfg.get('holdout_site')} "
                       f"but the task name says {t['holdout_site']}")
        if cfg.get("split_seed") != C.DATA_SPLIT_SEED:
            bad.append(f"{task}/{enc}: split_seed={cfg.get('split_seed')} "
                       f"but must be {C.DATA_SPLIT_SEED}")
        if cfg.get("num_classes") != t["num_classes"]:
            bad.append(f"{task}/{enc}: num_classes={cfg.get('num_classes')} "
                       f"but the task is {t['num_classes']}-class")
        ref_p = C.RUNS / C.TASKS[ref_task]["run_dir"] / enc / "config.json"
        if ref_p.exists():
            ref = json.loads(ref_p.read_text())
            for k in SPLIT_KEYS:
                if cfg.get(k) != ref.get(k):
                    bad.append(f"{task}/{enc}: {k}={cfg.get(k)} but "
                               f"{ref_task}/{enc} has {ref.get(k)} — the "
                               f"partition moved, not just the stopping rule")
        else:
            bad.append(f"{task}/{enc}: no {ref_task} run to compare against")
        ok += 1

n_want = (len(C.XSITE_FIX_TASKS) + len(C.XSITE_FIX_BIN_TASKS)) * len(C.ENCODERS)
print(f"cross-site fix: checkpoints verified {ok}/{n_want}")
for m in missing:
    print("  MISSING:", m)
for b in bad:
    print("  BAD:", b)
if missing or bad:
    sys.exit(f"preflight FAILED: {len(missing)} missing, {len(bad)} bad")
print("preflight OK — safe to extract")
