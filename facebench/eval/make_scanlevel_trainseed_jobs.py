#!/usr/bin/env python3
"""
Emit the joblist for the LEAKED-ARM training-seed sweep.
thread facebase3d-figs-2026-09-04.

WHY. Fig. 5(a) contrasts a scan-level (leaky) split against the subject-disjoint
split. After the 2026-08-31 sweep the corrected arm is a five-training-seed mean
while the leaked arm is still one unseeded checkpoint per task, so the two bars
are no longer like-for-like — the asymmetry the 2026-08-18 rework removed, back
in a subtler form. This retrains the leaked arm at the same five training seeds.

PointNet++ only: Fig. 5(a) plots that encoder alone. B1 and B2 only: C_scanlevel
is not on the panel.

PROTOCOL IS THE CORRECTED ARM'S. Same epochs, lr, weight decay and batch size as
B{1,2}_corrected_ts*, and --seed pinned to 42, so the only thing that differs
between the red and blue bars is which split the model was trained on. That is
the entire claim of the panel.

THE LEAK IS IN THE MANIFEST, NOT IN THE CODE. See
data/processed/legacy_scanlevel/README.md. Nothing these runs produce is task
performance; it is only ever the inflated arm of the leakage contrast.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C

ENCODER = "pointnet2"          # the only encoder on Fig. 5(a)

# Mean minutes per run, scaled from the corrected arm's measured cost by the
# ratio of training-set sizes. Ordering only; nothing downstream reads these.
COST_MIN = {"B1_scanlevel": 214, "B2_scanlevel": 22}

PY = os.environ.get("FACEBENCH_PYTHON", sys.executable)
TRAIN = C.PROJECT / "facebench/models/train.py"


def command(task, seed):
    t = C.TASKS[C.trainseed_task(task, seed)]
    parts = [
        PY, str(TRAIN),
        "--model", ENCODER,
        "--exp-id", C.trainseed_task(task, seed),
        "--manifest", str(t["manifest"]),
        "--exp-type", t["exp_type"],
        "--num-classes", str(t["num_classes"]),
        "--seed", str(C.DATA_SPLIT_SEED),   # FROZEN — never varied
        "--train-seed", str(seed),          # the swept axis
    ]
    if t["binary_mode"]:
        parts.append("--binary-mode")
    if t.get("aux_dim"):
        parts += ["--aux-dim", str(t["aux_dim"])]
    return " ".join(parts)


def main():
    jobs = [(task, seed)
            for task in C.SCANLEVEL_BASE_TASKS
            for seed in C.TRAIN_SEEDS]
    jobs.sort(key=lambda j: -COST_MIN[j[0]])   # long pole first

    out = Path(__file__).parent / "scanlevel_trainseed_joblist.txt"
    with open(out, "w") as f:
        for task, seed in jobs:
            f.write(command(task, seed) + "\n")

    total = sum(COST_MIN[t] for t, _ in jobs)
    print(f"{len(jobs)} jobs -> {out}")
    print(f"estimated {total/60:.1f} GPU-hours total, "
          f"longest single job {max(COST_MIN.values())} min")


if __name__ == "__main__":
    main()
