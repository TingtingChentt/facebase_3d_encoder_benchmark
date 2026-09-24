#!/usr/bin/env python3
"""
Emit the joblist for the TRAINING-SEED sweep (thread facebase3d-trainseed-2026-08-31).

Rationale (2026-08-31): the published intervals are a bootstrap over test subjects
plus an inference-seed spread with weights frozen. Neither is retraining
variance. This sweep retrains every (task, encoder) at five TRAINING seeds on
the FROZEN split-42 partition, so mean +/- SD is run-to-run spread measured on
an identical test set.

Emits one shell command per line; scripts/slurm/train_seed_array.sh runs
line $SLURM_ARRAY_TASK_ID. Longest jobs first so the array tail is short.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C

TRAIN_SEEDS = [1, 2, 3, 4, 5]
SWEEP_TASKS = ["A1", "A2", "B1_corrected", "B2_corrected", "C_corrected"]

# Mean minutes for one run, read off the existing single-seed train_log.csv
# files (epochs x mean elapsed_s). Used ONLY to order the array so the long
# pole starts first — nothing downstream depends on these numbers.
COST_MIN = {
    ("A1", "geommlp"): 1, ("A1", "pointnet"): 3, ("A1", "dgcnn"): 18, ("A1", "pointnet2"): 19,
    ("A2", "geommlp"): 1, ("A2", "pointnet"): 3, ("A2", "dgcnn"): 44, ("A2", "pointnet2"): 59,
    ("B1_corrected", "geommlp"): 2, ("B1_corrected", "pointnet"): 29,
    ("B1_corrected", "dgcnn"): 77, ("B1_corrected", "pointnet2"): 181,
    ("B2_corrected", "geommlp"): 1, ("B2_corrected", "pointnet"): 4,
    ("B2_corrected", "dgcnn"): 19, ("B2_corrected", "pointnet2"): 18,
    ("C_corrected", "geommlp"): 7, ("C_corrected", "pointnet"): 19,
    ("C_corrected", "dgcnn"): 156, ("C_corrected", "pointnet2"): 152,
}

PY = os.environ.get("FACEBENCH_PYTHON", sys.executable)
TRAIN = C.PROJECT / "facebench/models/train.py"


def seeded_exp_id(task, seed):
    """B1_corrected + seed 3 -> B1_corrected_ts3.

    Keeps every seeded run in its own runs/<exp_id>/<encoder>/ tree, so the
    frozen published checkpoints are never touched and
    rerun_config.checkpoint_path() resolves the seeded ones with no change:
    the trainer names checkpoints {ModelClass}_{exp_id}_best.pt.
    """
    return f"{task}_ts{seed}"


def command(task, encoder, seed):
    t = C.TASKS[task]
    parts = [
        PY, str(TRAIN),
        "--model", encoder,
        "--exp-id", seeded_exp_id(task, seed),
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
    jobs = [(task, enc, seed)
            for task in SWEEP_TASKS
            for enc in C.ENCODERS
            for seed in TRAIN_SEEDS]
    jobs.sort(key=lambda j: -COST_MIN[(j[0], j[1])])

    out = Path(__file__).parent / "trainseed_joblist.txt"
    with open(out, "w") as f:
        for task, enc, seed in jobs:
            f.write(command(task, enc, seed) + "\n")

    total = sum(COST_MIN[(t, e)] for t, e, _ in jobs)
    print(f"{len(jobs)} jobs -> {out}")
    print(f"estimated {total/60:.1f} GPU-hours total, "
          f"longest single job {max(COST_MIN.values())} min")


if __name__ == "__main__":
    main()
