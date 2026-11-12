#!/usr/bin/env python3
"""
Write the SLURM job lists for every reported model.

  facebench/training/train_joblist.txt     one train.py command line per (task, encoder):
                                       10 task families x 5 train seeds x 4 encoders = 200
  facebench/training/extract_joblist.txt   "<task> <encoder> <inference seed>" per line,
                                       consumed by scripts/slurm/extract_array.sh

Every train command pins the data-split seed to 42 and disables early stopping
(--patience == --epochs == 200); only --train-seed varies within a family.
The lists embed absolute paths for this checkout, so they are gitignored.

Usage: python3 facebench/training/make_jobs.py [--python /path/to/python3]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse

HERE = Path(__file__).resolve().parent
from facebench.training import task_registry as C                                    # noqa: E402

# Longest jobs first so the array's tail is short.
COST = {"dgcnn": 4, "pointnet2": 3, "pointnet": 2, "geommlp": 1}


def train_cmd(task, enc, py):
    t = C.TASKS[task]
    cmd = [py, str(C.PROJECT / "facebench/training/train.py"),
           "--model", enc, "--exp-id", t["run_dir"], "--manifest", str(t["manifest"]),
           "--exp-type", t["exp_type"], "--num-classes", str(t["num_classes"]),
           "--seed", str(C.DATA_SPLIT_SEED), "--train-seed", str(t["train_seed"]),
           "--epochs", str(C.EPOCHS), "--patience", str(C.EPOCHS)]
    if t["binary_mode"]:
        cmd.append("--binary-mode")
    return " ".join(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    jobs = [(t, e) for t in C.TRAINED_TASKS for e in C.task_encoders(t)]
    jobs.sort(key=lambda x: -COST[x[1]])
    train = [train_cmd(t, e, a.python) for t, e in jobs]
    extract = [f"{t} {e} {s}" for t in C.TRAINED_TASKS for e in C.task_encoders(t)
               for s in C.seeds_for(e)]
    (HERE / "train_joblist.txt").write_text("".join(l + "\n" for l in train))
    (HERE / "extract_joblist.txt").write_text("".join(l + "\n" for l in extract))
    print(f"train_joblist.txt: {len(train)} jobs\nextract_joblist.txt: {len(extract)} jobs")


if __name__ == "__main__":
    main()
