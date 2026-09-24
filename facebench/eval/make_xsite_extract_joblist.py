#!/usr/bin/env python3
"""
Emit the cross-site feature-extraction job list, one line per
(task, encoder, inference seed).
thread facebase3d-paper-2026-08-08, PART A stage 2.

Seed counts come from C.seeds_for(), not from a literal here: 5 inference seeds
for PointNet++ (whose FPS draw is non-deterministic) and 1 for the three
encoders that were shown bit-identical across repeat runs. Duplicating that rule
by hand is how the two drift apart.

Only folds whose checkpoints actually exist are emitted, so this can be run
while the training array is still draining and it will simply pick up whatever
has finished. --require-all fails instead, for the final pass.

Usage:
  python3 make_xsite_extract_joblist.py                 # what is ready now
  python3 make_xsite_extract_joblist.py --require-all   # all 48, or fail
  python3 make_xsite_extract_joblist.py --binary ...    # the XSB_* binary folds
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

HERE = Path(__file__).parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="joblist_xsite_extract.txt")
    p.add_argument("--require-all", action="store_true",
                   help="Exit non-zero if any checkpoint is missing.")
    p.add_argument("--binary", action="store_true",
                   help="Emit the XSB_* binary-task folds instead of XS_*.")
    p.add_argument("--fix", action="store_true",
                   help="Emit the no-early-stopping retrains (XS_fix_* / "
                        "XSB_fix_*) rather than the published folds.")
    a = p.parse_args()
    if a.fix:
        tasks = C.XSITE_FIX_BIN_TASKS if a.binary else C.XSITE_FIX_TASKS
    else:
        tasks = C.XSITE_BIN_TASKS if a.binary else C.XSITE_TASKS

    lines, missing = [], []
    for task in tasks:
        for enc in C.task_encoders(task):
            ckpt = C.checkpoint_path(task, enc)
            if not ckpt.exists():
                missing.append(f"{task}/{enc}: {ckpt.name}")
                continue
            for seed in C.seeds_for(enc):
                lines.append(f"{task} {enc} {seed}\n")

    out = HERE / a.out
    out.write_text("".join(lines))
    print(f"{len(lines)} extraction jobs -> {out}")
    n_expect = sum(len(C.seeds_for(e)) for e in C.ENCODERS) * len(tasks)
    print(f"  (full sweep would be {n_expect})")
    if missing:
        print(f"\n  {len(missing)} checkpoint(s) not yet on disk:")
        for m in missing:
            print(f"    {m}")
        if a.require_all:
            sys.exit("--require-all: refusing to emit a partial job list")


if __name__ == "__main__":
    main()
