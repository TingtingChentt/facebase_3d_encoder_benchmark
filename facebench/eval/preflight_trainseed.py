#!/usr/bin/env python3
"""
Gate between the training-seed sweep and the scoring stages.

Verifies that all 100 (task, encoder, train seed) checkpoints exist and that
each run's config.json records the split seed FROZEN at 42 and the intended
train seed. Exits non-zero on any failure so the dependent extraction array
never runs against a partial sweep — a silently-missing seed would shrink an
SD rather than raise an error, which is the worst possible failure mode here.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

missing, bad_cfg, ok = [], [], 0
for task in C.TRAINSEED_TASKS:
    want_ts = C.TASKS[task]["train_seed"]
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
                           f"but must be {C.DATA_SPLIT_SEED} — this run is NOT "
                           f"comparable and must be discarded, not scored")
        if cfg.get("train_seed") != want_ts:
            bad_cfg.append(f"{task}/{enc}: train_seed={cfg.get('train_seed')} "
                           f"but task name says {want_ts}")
        ok += 1

print(f"checkpoints verified: {ok}/{len(C.TRAINSEED_TASKS) * len(C.ENCODERS)}")
for m in missing:
    print("  MISSING:", m)
for b in bad_cfg:
    print("  BAD:", b)
if missing or bad_cfg:
    sys.exit(f"preflight FAILED: {len(missing)} missing, {len(bad_cfg)} bad")
print("preflight OK — safe to extract")
