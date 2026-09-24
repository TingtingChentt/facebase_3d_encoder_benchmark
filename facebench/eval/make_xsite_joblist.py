#!/usr/bin/env python3
"""
Emit the leave-one-site-out job list.
thread facebase3d-paper-2026-08-08, PART A.

The fold set is NOT hard-coded here — it is read back from the audit CSV that
audit_xsite_folds.py writes, so a fold that fails the post-attrition viability
gate can never reach the queue by being stale in a list someone forgot to edit.

Usage:
  python3 make_xsite_joblist.py                 # all viable folds x 4 encoders
  python3 make_xsite_joblist.py --only PH:pointnet2 --out joblist_xsite_smoke.txt
"""
import argparse
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
AUDIT = PROJECT / "results/xsite_fold_audit.csv"
HERE = Path(__file__).parent
ENCODERS = ["geommlp", "pointnet", "dgcnn", "pointnet2"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="joblist_xsite.txt")
    p.add_argument("--only", default=None,
                   help="Comma-separated SITE:ENCODER pairs (smoke run).")
    p.add_argument("--exclude", default=None,
                   help="Comma-separated SITE:ENCODER pairs to omit (e.g. the "
                        "already-completed smoke run).")
    a = p.parse_args()

    if not AUDIT.exists():
        raise SystemExit(f"missing {AUDIT} — run audit_xsite_folds.py first")
    audit = pd.read_csv(AUDIT)
    viable = audit.loc[audit.viable, "site"].astype(str).tolist()

    if a.only:
        pairs = [tuple(s.split(":")) for s in a.only.split(",")]
        bad = [s for s, _ in pairs if s not in viable]
        if bad:
            raise SystemExit(f"site(s) {bad} are not viable folds: {viable}")
    else:
        pairs = [(s, e) for s in viable for e in ENCODERS]

    if a.exclude:
        drop = {tuple(s.split(":")) for s in a.exclude.split(",")}
        pairs = [p_ for p_ in pairs if p_ not in drop]

    out = HERE / a.out
    out.write_text("".join(f"{s} {e}\n" for s, e in pairs))
    print(f"{len(pairs)} jobs -> {out}")
    for s, e in pairs:
        print(f"  {s} {e}")


if __name__ == "__main__":
    main()
