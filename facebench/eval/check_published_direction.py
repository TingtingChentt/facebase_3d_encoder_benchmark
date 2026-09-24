#!/usr/bin/env python3
"""
Where do the PUBLISHED PointNet++ point estimates sit relative to the seeded
re-eval range, and IN WHICH DIRECTION?
thread facebase3d-paper-2026-08-08.

My 08-07 reply reported "10 of 13 published PN++ values fall outside the 5-seed
range" and listed four of them, all of which happened to be ABOVE the seeded
mean. In the 08-09 PART B message I generalised that to "the published value is
higher in every case". This script checks that generalisation against all
thirteen instead of four, because the DIRECTION is what discriminates between
candidate explanations: a one-sided discrepancy implies a biased estimator,
whereas a two-sided one implies the seeded range simply understates the spread.

Published values are transcribed from EXPERIMENTAL_FINDINGS.md with the source
line recorded for each, so the comparison is auditable rather than remembered.

Usage: python3 check_published_direction.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

# (task, encoder, head, level, metric, published, EXPERIMENTAL_FINDINGS.md line)
PUBLISHED = [
    ("A1", "softmax", "scan", "accuracy", 0.740, "A1 4-class table"),
    ("A2", "softmax", "scan", "accuracy", 0.841, "A2 binary table"),
    ("A2", "softmax", "scan", "macro_auc", 0.856, "A2 binary table"),
    ("B2_r2", "softmax", "scan", "accuracy", 0.401, "line 221"),
    ("B1_corrected", "softmax", "scan", "accuracy", 0.298, "line 341"),
    ("B1_corrected", "softmax", "scan", "macro_f1", 0.260, "line 341"),
    ("B1_corrected", "softmax", "scan", "macro_auc", 0.771, "line 341"),
    ("B2_corrected", "softmax", "scan", "accuracy", 0.276, "line 343 / 257"),
    ("B2_corrected", "softmax", "scan", "macro_f1", 0.222, "line 343 / 257"),
    ("B2_corrected", "softmax", "scan", "macro_auc", 0.850, "line 343 / 257"),
    ("C_corrected", "softmax", "scan", "accuracy", 0.902, "08-07 reply"),
    ("C_corrected", "softmax", "scan", "macro_f1", 0.899, "08-07 reply"),
    ("C_corrected", "softmax", "scan", "macro_auc", 0.960, "08-07 reply"),
]


def main():
    summ = pd.read_csv(C.OUT / "summary_ci.csv")
    rows = []
    for task, head, level, metric, pub, src in PUBLISHED:
        r = summ[(summ.task == task) & (summ.encoder == "pointnet2") &
                 (summ["head"] == head) & (summ.level == level) &
                 (summ.metric == metric)]
        if len(r) != 1:
            print(f"  SKIP {task}/{metric}: {len(r)} rows matched")
            continue
        r = r.iloc[0]
        lo, hi = r.seed_min, r.seed_max
        if pub < lo:
            where, direction = "OUTSIDE", "below"
        elif pub > hi:
            where, direction = "OUTSIDE", "above"
        else:
            where, direction = "inside", "-"
        rows.append(dict(
            task=task, metric=metric, published=pub,
            seed_mean=round(r["mean"], 4),
            seed_min=round(lo, 4), seed_max=round(hi, 4),
            delta_pp=round(100 * (pub - r["mean"]), 2),
            where=where, direction=direction,
            in_boot_ci=bool(r.boot_ci_lo <= pub <= r.boot_ci_hi),
            src=src))

    df = pd.DataFrame(rows)
    print("=" * 96)
    print("PUBLISHED PointNet++ values vs the 5-seed inference range "
          "(seed_min..seed_max)")
    print("=" * 96)
    print(df.to_string(index=False))

    out = df[df["where"] == "OUTSIDE"]
    above = (out.direction == "above").sum()
    below = (out.direction == "below").sum()
    print(f"\n  outside the seeded range : {len(out)}/{len(df)}")
    print(f"    of those, ABOVE the seeded mean : {above}")
    print(f"    of those, BELOW the seeded mean : {below}")
    print(f"  inside the bootstrap subject CI   : {int(df.in_boot_ci.sum())}/{len(df)}")

    print("\n" + "=" * 96)
    if above and below:
        print("THE DISCREPANCY IS TWO-SIDED.")
        print("  A biased estimator (one device, one batching, one thread count")
        print("  systematically favouring one direction) does NOT fit: published")
        print("  values land on BOTH sides of the seeded range. What fits is that")
        print("  the 5-seed range UNDERSTATES the estimator's true spread, and each")
        print("  published number is a single unseeded draw from that wider spread.")
    else:
        print("THE DISCREPANCY IS ONE-SIDED — consistent with a biased estimator.")
    print("=" * 96)
    df.to_csv(C.OUT / "published_direction_check.csv", index=False)
    print(f"\n-> {C.OUT.name}/published_direction_check.csv")


if __name__ == "__main__":
    main()
