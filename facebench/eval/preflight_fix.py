#!/usr/bin/env python3
"""
Gate between a no-early-stopping retrain and the scoring stages.
threads facebase3d-a2diag-2026-09-04 / facebase3d-a1diag-2026-09-04.

Same contract as preflight_trainseed.py, plus the two checks specific to these
retrains:

  * patience == epochs on EVERY run, so the early-stopping branch provably
    never fired. A run still carrying patience=20 would be one of the
    checkpoints the sweep exists to replace, scored under the new name.
  * the split is byte-identical to the truncated runs being replaced (n_train,
    n_val, n_test and the test class counts). Only the stopping rule may
    differ; if the partition moved, the comparison is not a comparison of
    protocols any more.

Exits non-zero on any failure so a partial sweep is never scored — a missing
seed would shrink an SD rather than raise an error.

Usage: python3 preflight_fix.py --family A2_fix
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

SPLIT_KEYS = ["n_train", "n_val", "n_test", "test_class_counts"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(C.FIX_FAMILIES))
    a = ap.parse_args()
    fam = C.fix_family(a.family)

    missing, bad_cfg, ok = [], [], 0
    for task in fam["tasks"]:
        want_ts = C.TASKS[task]["train_seed"]
        # the truncated run this one replaces, e.g. A2_fix_ts3 -> A2_ts3
        ref_task = C.trainseed_task(fam["legacy_base"], want_ts)
        for enc in C.task_encoders(task):
            ck = C.checkpoint_path(task, enc)
            if not ck.exists():
                missing.append(f"{task}/{enc}: no checkpoint at {ck}")
                continue
            cfg_p = C.RUNS / C.TASKS[task]["run_dir"] / enc / "config.json"
            try:
                cfg = json.loads(cfg_p.read_text())
            except Exception as e:                      # noqa: BLE001
                bad_cfg.append(f"{task}/{enc}: unreadable config.json ({e})")
                continue
            if cfg.get("split_seed") != C.DATA_SPLIT_SEED:
                bad_cfg.append(f"{task}/{enc}: split_seed={cfg.get('split_seed')} "
                               f"but must be {C.DATA_SPLIT_SEED}")
            if cfg.get("train_seed") != want_ts:
                bad_cfg.append(f"{task}/{enc}: train_seed={cfg.get('train_seed')} "
                               f"but task name says {want_ts}")
            if cfg.get("patience") != cfg.get("epochs"):
                bad_cfg.append(f"{task}/{enc}: patience={cfg.get('patience')} vs "
                               f"epochs={cfg.get('epochs')} — early stopping "
                               f"could still fire, this is not a fixed run")
            ref_p = C.RUNS / C.TASKS[ref_task]["run_dir"] / enc / "config.json"
            if ref_p.exists():
                ref = json.loads(ref_p.read_text())
                for k in SPLIT_KEYS:
                    if cfg.get(k) != ref.get(k):
                        bad_cfg.append(
                            f"{task}/{enc}: {k}={cfg.get(k)} but {ref_task}/{enc} "
                            f"has {ref.get(k)} — the partition moved, not just "
                            f"the stopping rule")
            else:
                bad_cfg.append(f"{task}/{enc}: no {ref_task} run to compare the "
                               f"split against")
            ok += 1

    n_want = len(fam["tasks"]) * len(fam["encoders"])
    print(f"{a.family}: checkpoints verified {ok}/{n_want}")
    for m in missing:
        print("  MISSING:", m)
    for b in bad_cfg:
        print("  BAD:", b)
    if missing or bad_cfg:
        sys.exit(f"preflight FAILED: {len(missing)} missing, {len(bad_cfg)} bad")
    print("preflight OK — safe to extract")


if __name__ == "__main__":
    main()
