#!/usr/bin/env python3
"""
Emit the feature-extraction joblist for a no-early-stopping retrain.
threads facebase3d-a2diag-2026-09-04 / facebase3d-a1diag-2026-09-04.

One line per (task, encoder, inference seed), the format
scripts/slurm/paper_rerun_extract.sh reads. Inference seeds come from
seeds_for(): only PointNet++ is non-deterministic at inference (FPS start
index), so the other three encoders are extracted once — repeating them would
reproduce identical arrays at 4x the cost.

  5 train seeds x (geommlp 1 + pointnet 1 + dgcnn 1 + pointnet2 5) = 40 jobs.

Usage: python3 make_fix_extract_jobs.py --family A2_fix
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C

# Longest first so the array tail is short. Read off the 2026-09-01 A2_ts
# extraction logs (array 19972727); the A1 cohort is the same manifest.
COST_MIN = {"dgcnn": 20, "pointnet2": 11, "pointnet": 2, "geommlp": 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(C.FIX_FAMILIES))
    a = ap.parse_args()
    fam = C.fix_family(a.family)

    jobs = [(task, enc, seed)
            for task in fam["tasks"]
            for enc in C.task_encoders(task)
            for seed in C.seeds_for(enc)]
    jobs.sort(key=lambda j: -COST_MIN[j[1]])

    out = Path(__file__).parent / f"{a.family.lower()}_extract_joblist.txt"
    with open(out, "w") as f:
        for task, enc, seed in jobs:
            f.write(f"{task} {enc} {seed}\n")
    print(f"{len(jobs)} jobs -> {out}")
    print(f"estimated {sum(COST_MIN[e] for _, e, _ in jobs) / 60:.1f} CPU-hours")


if __name__ == "__main__":
    main()
