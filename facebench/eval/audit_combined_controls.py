#!/usr/bin/env python3
"""
Audit the label/cohort structure of the COMBINED screening task (C_corrected).
thread facebase3d-paper-2026-08-08.

WHY. Section 2.2 leads the screening claim. Three of the six datasets in the
combined task are normative control cohorts that contribute ZERO Affected
scans, so for that part of the negative class the label is determined by which
cohort the scan came from. This script measures how large that part is, and
what an oracle that reads ONLY cohort membership -- never the geometry --
would score on the same corpus. Those two numbers are the honest bound on how
much of the combined result could in principle be cohort recognition.

WHAT IT IS NOT. It is NOT a claim that the model uses the shortcut. The encoder
receives a point cloud and nothing else; it never sees a dataset column. The
shortcut is available to it only to the extent that cohort and scanner leave a
geometric signature -- which the leave-one-site-out result shows they do
(8-14 pp of AUC lost when the site changes). That is the reason to report the
composition rather than to assume either way.

The comparable CLEAN task is the cleft binary task (A2), whose negatives are
unaffected family members inside the same cohorts, scanned at the same sites,
so no cohort-only rule separates them at all. This script reports A2's
composition too, since the contrast is the point.

Usage: python3 audit_combined_controls.py [--tex]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

# The three normative cohorts: unaffected by construction, no Affected scans.
NORMATIVE = ["FB-TK0", "FB-TX4", "FB-VWP"]


def audit(manifest, label_col, pos, neg, name):
    """neg is the literal negative label; every other value in label_col is
    collapsed to pos. That collapse is how A2 is defined (cleft_type ->
    Unaffected vs any cleft), and it is a no-op for the combined manifest,
    whose binary_label already carries exactly the two values."""
    d = pd.read_csv(manifest).copy()
    d[label_col] = d[label_col].where(d[label_col] == neg, pos)
    n = len(d)
    neg_rows = d[d[label_col] == neg]
    pos_rows = d[d[label_col] == pos]
    from_norm = neg_rows.dataset.isin(NORMATIVE).sum()
    # Cohort-only oracle: "normative cohort -> negative, else positive".
    # It reads the dataset column and nothing else.
    pred_neg = d.dataset.isin(NORMATIVE)
    acc = ((pred_neg & (d[label_col] == neg))
           | (~pred_neg & (d[label_col] == pos))).sum() / n
    tpr = from_norm / len(neg_rows)                 # negatives it recovers
    fpr = pos_rows.dataset.isin(NORMATIVE).sum() / len(pos_rows)
    auc = (tpr + (1 - fpr)) / 2                     # AUC of a single-threshold score
    maj = d[label_col].value_counts(normalize=True).max()
    print(f"\n=== {name} | {Path(manifest).name} | {n} scans ===")
    print(pd.crosstab(d.dataset, d[label_col], margins=True).to_string())
    print(f"  negatives                      {len(neg_rows)}")
    print(f"  negatives from normative-only  {from_norm}  ({100 * from_norm / len(neg_rows):.1f}%)")
    print(f"  majority-class baseline        {maj:.3f}")
    print(f"  cohort-only oracle accuracy    {acc:.3f}")
    print(f"  cohort-only oracle macro-AUC   {auc:.3f}   (TPR {tpr:.3f}, FPR {fpr:.3f})")
    return dict(n=n, n_neg=len(neg_rows), from_norm=int(from_norm),
                frac=from_norm / len(neg_rows), acc=acc, auc=auc, maj=maj)


def main():
    comb = audit(C.TASKS["C_corrected"]["manifest"], "binary_label",
                 "Affected", "Unaffected", "COMBINED screening (task B)")
    ofc = audit(C.TASKS["A2"]["manifest"], "cleft_type",
                "Affected", "Unaffected", "CLEFT screening (task C2/A2)")

    if "--tex" in sys.argv:
        print("\n% ---- paste into numbers.tex ----")
        print(f"\\newcommand{{\\combNegN}}{{{comb['n_neg']:,}}}".replace(",", "{,}"))
        print(f"\\newcommand{{\\combNegNormN}}{{{comb['from_norm']:,}}}".replace(",", "{,}"))
        print(f"\\newcommand{{\\combNegNormFrac}}{{{100 * comb['frac']:.0f}\\%}}")
        print(f"\\newcommand{{\\combCohortOnlyAuc}}{{{comb['auc']:.3f}}}")
        print(f"\\newcommand{{\\combCohortOnlyAcc}}{{{comb['acc']:.3f}}}")
        print(f"\\newcommand{{\\ofcNegNormFrac}}{{{100 * ofc['frac']:.0f}\\%}}")
        print(f"\\newcommand{{\\ofcCohortOnlyAuc}}{{{ofc['auc']:.3f}}}")


if __name__ == "__main__":
    main()
