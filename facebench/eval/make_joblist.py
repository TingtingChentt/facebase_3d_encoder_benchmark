#!/usr/bin/env python3
"""
Emit the (task, encoder, inference_seed) job list for the 5-seed re-eval,
one job per line, for consumption by a SLURM array.

Deterministic encoders get ONE seed. That is not a shortcut: the determinism
probe (results/audit/determinism_probe.csv) shows GeomMLP,
PointNet and DGCNN return bit-identical features across repeat runs, so extra
seeds would reproduce identical numbers at 4x the compute. Their
inference-seed SD is therefore exactly 0 by construction, and is reported as
such. Only PointNet++ is swept over {0,1,2,3,4}.

Usage:
  python3 make_joblist.py --stage priority > joblist_priority.txt
  python3 make_joblist.py --stage secondary > joblist_secondary.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

# Measured on the login node at 4 threads (determinism probe).
SEC_PER_SCAN = {"geommlp": 0.001, "pointnet": 0.092,
                "dgcnn": 1.102, "pointnet2": 0.604}
# train + test scans per task (from each run's config.json)
SCANS = {
    "A1": 5844, "A1_S1": 5844, "A1_S2": 5844, "A1_S3": 5844, "A2": 5843,
    "B1_corrected": 11014, "B2_corrected": 3136,
    "B1_proto_cetrain": 11014, "B2_proto_cetrain": 3136,
    "B2_r2": 2487, "C_corrected": 20869,
}


STAGES = {
    "priority": lambda: C.PRIORITY_TASKS,
    "secondary": lambda: C.SECONDARY_TASKS,
    # Training-seed sweep: the 25 (base task x train seed) runs. Inference
    # seeds are swept WITHIN each of them exactly as for the frozen
    # checkpoints, so each retrained model's point estimate has its FPS noise
    # averaged down before the across-training-seed SD is taken. Those are
    # different quantities and pooling them would misstate both.
    "trainseed": lambda: C.TRAINSEED_TASKS,
}


def base_of(task):
    """Seeded tasks cost exactly what their base task costs."""
    return C.TASKS[task].get("base_task", task)


def jobs(stage):
    tasks = STAGES[stage]()
    out = []
    for t in tasks:
        for e in C.task_encoders(t):
            for s in C.seeds_for(e):
                out.append((t, e, s))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(STAGES), required=True)
    p.add_argument("--cost", action="store_true",
                   help="print a cost estimate to stderr instead of the list")
    a = p.parse_args()

    js = jobs(a.stage)
    if a.cost:
        tot, longest = 0.0, ("", 0.0)
        for t, e, s in js:
            sec = SCANS[base_of(t)] * SEC_PER_SCAN[e]
            tot += sec
            if sec > longest[1]:
                longest = (f"{t}/{e}/seed{s}", sec)
        print(f"stage={a.stage}  jobs={len(js)}", file=sys.stderr)
        print(f"total ~{tot/3600:.1f} CPU-hours (4 threads)", file=sys.stderr)
        print(f"longest single job: {longest[0]} ~{longest[1]/3600:.1f} h",
              file=sys.stderr)
        return

    for t, e, s in js:
        print(f"{t} {e} {s}")


if __name__ == "__main__":
    main()
