#!/usr/bin/env python3
"""
Emit the training joblist for a no-early-stopping retrain.
threads facebase3d-a2diag-2026-09-04 / facebase3d-a1diag-2026-09-04.

See the A1_FIX / A2_FIX blocks in rerun_config.py for why these sweeps exist.
In short: on both cleft tasks the model sits on an initial plateau (train loss
at ln(K)) for tens of epochs with validation macro-F1 pure noise, and patience
counted on that noisy F1 truncated runs inside it. Whether a run survived was a
draw, so the published checkpoints and the five retrainings are not measuring
the same protocol.

--patience equal to --epochs means the early-stopping branch can never fire.
Checkpoint selection is unchanged (best validation macro-F1), so the only
difference from the runs these replace is that a run is allowed to reach the
epoch where its best occurs. Disabling early stopping can only change a result
where a run was previously stopped BEFORE its best checkpoint would have
occurred, so it is a strictly safer protocol, not a more permissive one.

Everything else is pinned to the runs being replaced: same manifest, same
--seed 42 data split, same --train-seed axis, same class count and binary mode.

Usage: python3 make_fix_jobs.py --family A2_fix
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C

PY = os.environ.get("FACEBENCH_PYTHON", sys.executable)
TRAIN = C.PROJECT / "facebench/models/train.py"

# Minutes per run at the full epoch budget, for long-pole-first ordering only.
# A2 from the 2026-09-05 sweep, A1 from the 2026-09-05 patience probes.
COST_MIN = {
    "A1_fix": {"dgcnn": 100, "pointnet2": 95, "pointnet": 20, "geommlp": 8},
    "A2_fix": {"pointnet2": 95, "dgcnn": 100, "pointnet": 17, "geommlp": 7},
}


def command(family, encoder, seed):
    fam = C.fix_family(family)
    task = C.trainseed_task(family, seed)
    t = C.TASKS[task]
    parts = [
        PY, str(TRAIN),
        "--model", encoder,
        "--exp-id", task,
        "--manifest", str(t["manifest"]),
        "--exp-type", t["exp_type"],
        "--num-classes", str(t["num_classes"]),
        "--seed", str(C.DATA_SPLIT_SEED),      # FROZEN — never varied
        "--train-seed", str(seed),             # the swept axis
        "--epochs", str(fam["epochs"]),
        "--patience", str(fam["epochs"]),      # == epochs: never fires
    ]
    if t["binary_mode"]:
        parts.append("--binary-mode")
    if t.get("aux_dim"):
        parts += ["--aux-dim", str(t["aux_dim"])]
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(C.FIX_FAMILIES))
    a = ap.parse_args()
    fam = C.fix_family(a.family)
    cost = COST_MIN[a.family]

    jobs = [(e, s) for e in fam["encoders"] for s in C.TRAIN_SEEDS]
    jobs.sort(key=lambda j: -cost[j[0]])       # long pole first

    out = Path(__file__).parent / f"{a.family.lower()}_joblist.txt"
    with open(out, "w") as f:
        for enc, seed in jobs:
            f.write(command(a.family, enc, seed) + "\n")

    total = sum(cost[e] for e, _ in jobs)
    print(f"{len(jobs)} jobs -> {out}")
    print(f"estimated {total/60:.1f} GPU-hours total, "
          f"longest single job {max(cost.values())} min")


if __name__ == "__main__":
    main()
