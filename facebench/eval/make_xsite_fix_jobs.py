#!/usr/bin/env python3
"""
Emit the training joblist for the cross-site LOSO retrain with early stopping
disabled. thread facebase3d-xsitediag-2026-09-07.

See the cross-site block in rerun_config.py for why. In short: every XS_*/XSB_*
checkpoint was trained under the patience-20 rule that the A1/A2 diagnosis
showed truncates cleft training inside its initial plateau, and the four-class
folds stopped at 21-46 epochs, so Sec 2.7's cross-site contrast cannot yet be
separated from the stopping rule.

BOTH ARMS. Sec 2.7 reports a contrast between four-class and binary transfer on
identical folds; correcting one arm alone would turn that contrast into a
protocol comparison.

  6 folds x 4 encoders x 2 label collapses = 48 runs.

The training manifest is ofc_manifest.csv, NOT the ofc_manifest_xsite.csv that
rerun_config carries for these tasks. That is deliberate and matches the runs
being replaced: the xsite manifest is the same rows plus a composite
dataset|study_id subject column that exists so the SCORING bootstrap unit is the
test subject. make_xsite_manifest.py verifies the two are row-identical.

Usage: python3 make_xsite_fix_jobs.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C

# The retrains are SEEDED, the runs they replace were not. Two consequences,
# both stated in the manuscript rather than buried here:
#   * --train-seed makes each run reproducible, and also gives each dataloader
#     worker its own augmentation stream. Without it the four forked workers
#     replay one identical stream (the latent bug found 2026-08-31), so seeding
#     is a SECOND protocol change riding alongside the stopping rule. It is the
#     same choice the A1_fix / A2_fix sweeps already made, and the corrected
#     arm — not the difference — is what gets reported.
#   * One seeded draw against one unseeded draw does not separate retraining
#     variance from the stopping rule. On A2 that variance ran 1-10 pp of AUC
#     depending on encoder. If a cross-site conclusion turns on a difference of
#     that size, it needs a seed axis on the affected folds before it is stated.
XSITE_FIX_TRAIN_SEED = 1

PY = os.environ.get("FACEBENCH_PYTHON", sys.executable)
TRAIN = C.PROJECT / "facebench/models/train.py"
TRAIN_MANIFEST = C.DATA / "ofc_manifest.csv"

# Minutes per run at the full 200-epoch budget, from the 2026-09-05 A1 probes
# (same cohort, same scale). Ordering only.
COST_MIN = {"dgcnn": 100, "pointnet2": 95, "pointnet": 20, "geommlp": 8}


def command(task, encoder):
    t = C.TASKS[task]
    parts = [
        PY, "-u", str(TRAIN),
        "--model", encoder,
        "--exp-id", task,
        "--manifest", str(TRAIN_MANIFEST),
        "--exp-type", t["exp_type"],
        "--num-classes", str(t["num_classes"]),
        "--holdout-site", t["holdout_site"],
        "--seed", str(C.DATA_SPLIT_SEED),        # FROZEN — never varied
        "--train-seed", str(XSITE_FIX_TRAIN_SEED),
        "--epochs", str(C.XSITE_FIX_EPOCHS),
        "--patience", str(C.XSITE_FIX_EPOCHS),   # == epochs: never fires
    ]
    if t["binary_mode"]:
        parts.append("--binary-mode")
    return " ".join(parts)


def main():
    tasks = C.XSITE_FIX_TASKS + C.XSITE_FIX_BIN_TASKS
    jobs = [(t, e) for t in tasks for e in C.task_encoders(t)]
    jobs.sort(key=lambda j: -COST_MIN[j[1]])     # long pole first

    out = Path(__file__).parent / "xsite_fix_joblist.txt"
    with open(out, "w") as f:
        for task, enc in jobs:
            f.write(command(task, enc) + "\n")

    total = sum(COST_MIN[e] for _, e in jobs)
    print(f"{len(jobs)} jobs -> {out}")
    print(f"estimated {total/60:.1f} GPU-hours total, "
          f"longest single job {max(COST_MIN.values())} min")


if __name__ == "__main__":
    main()
