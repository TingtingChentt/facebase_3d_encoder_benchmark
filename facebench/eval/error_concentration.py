#!/usr/bin/env python3
"""
Are the model's mistakes STRUCTURED or RANDOM?  (Results, sec:core)
thread facebase3d-paper-2026-08-08.

WHY THIS EXISTS. The draft asserted that "under an independence model the
observed AUC would imply a top-1 accuracy far below what we measure, so the
errors are structured rather than random". That sentence had no number, no
macro and no script behind it, and it rested on an unstated independence model
that was never written down. This replaces it with a direct measurement of the
thing actually being claimed.

WHAT IS MEASURED. For each true class with at least MIN_ERRORS misclassified
units, take the fraction of that class's errors that land on its single
most-confused label, then average over classes:

    concentration(c) = max_j n(c -> j, j != c) / sum_j n(c -> j, j != c)

The null is errors spread uniformly over the other K-1 classes, which gives
1/(K-1) — 0.031 at 33 classes, 0.056 at 19. The ratio of the two is the
headline: how many times more concentrated the confusions are than chance.

WHY NOT A CONFUSION-MATRIX EIGENVALUE / ENTROPY. Both are defensible, but this
statistic answers the clinical question directly: when the model is wrong about
a condition, does it tend to be wrong in the SAME way? That is what "confusions
are concentrated among phenotypically related conditions" means, and it is what
a shortlisting instrument's user needs to know.

CLASSES WITH <2 ERRORS ARE EXCLUDED, not counted as concentration 1.0. A class
with a single error trivially puts 100% of its errors on one label and would
inflate the mean toward 1 for free; the count of contributing classes is
reported alongside so the exclusion is visible.

Reads only the cached prediction arrays — no model, no GPU.
Usage: python3 error_concentration.py [--emit]
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402
from trainseed_pred import run_groups                       # noqa: E402

MIN_ERRORS = 2
PRED = C.OUT / "predictions"
OUT_TEX = C.OUT / "numbers_from_errconc.tex"

# (task, K, level, head) — the four cells quoted in Results, plus the scan-level
# and 19-class rows that establish the effect is not specific to one setting.
CELLS = [
    ("B2_corrected", 33, "subject", "softmax"),
    ("B2_corrected", 33, "subject", "proto"),
    ("B2_corrected", 33, "scan", "softmax"),
    ("B1_corrected", 19, "subject", "softmax"),
    ("B1_corrected", 19, "subject", "proto"),
]


def concentration(path):
    d = np.load(path, allow_pickle=True)
    yt, yp = d["y_true"], d["y_pred"]
    wrong = yt != yp
    per_class = []
    for c in np.unique(yt):
        m = (yt == c) & wrong
        if m.sum() < MIN_ERRORS:
            continue
        _, cnt = np.unique(yp[m], return_counts=True)
        per_class.append(cnt.max() / cnt.sum())
    if not per_class:
        return None, 0
    return float(np.mean(per_class)), len(per_class)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true",
                   help="write the LaTeX macro block as well as the table")
    p.add_argument("--family", choices=["legacy", "trainseed"],
                   default="trainseed",
                   help="trainseed (default) reads the five independently "
                        "trained models; legacy reads the single pre-sweep "
                        "checkpoint this script used before 2026-09-08")
    a = p.parse_args()

    rows, missing = [], []
    for task, K, level, head in CELLS:
        try:
            groups = run_groups(PRED, task, head, level, a.family)
        except SystemExit:
            missing.append(f"{task}/{head}/{level}")
            continue
        # Inference seeds are reduced WITHIN each training run before averaging
        # across runs, so the sd below is retraining spread and not the FPS
        # draw. Under --family legacy there is one run and the sd is 0 by
        # construction; the reported sd was the inference spread before
        # 2026-09-08, which is a different quantity with the same name.
        per_run, ns = [], []
        for _, fs in groups:
            vals, n = zip(*[concentration(f) for f in fs])
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            per_run.append(float(np.mean(vals)))
            ns.append(n[0])
        uniform = 1.0 / (K - 1)
        rows.append(dict(task=task, K=K, level=level, head=head,
                         n_seeds=len(per_run), mean=float(np.mean(per_run)),
                         sd=float(np.std(per_run)), lo=float(np.min(per_run)),
                         hi=float(np.max(per_run)), uniform=uniform,
                         ratio=float(np.mean(per_run)) / uniform,
                         n_classes=ns[0]))

    if missing:
        raise SystemExit("missing prediction files for: " + ", ".join(missing))

    hdr = (f"{'task':13} {'K':>3} {'level':8} {'head':8} {'conc':>6} "
           f"{'sd':>6} {'uniform':>8} {'ratio':>7} {'classes':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['task']:13} {r['K']:3} {r['level']:8} {r['head']:8} "
              f"{r['mean']:6.3f} {r['sd']:6.3f} {r['uniform']:8.3f} "
              f"{r['ratio']:6.1f}x {r['n_classes']:8}")

    if not a.emit:
        return

    def pick(task, level, head):
        return next(r for r in rows if r["task"] == task
                    and r["level"] == level and r["head"] == head)

    dx = pick("B2_corrected", "subject", "softmax")
    dxp = pick("B2_corrected", "subject", "proto")
    syn = pick("B1_corrected", "subject", "softmax")
    L = [
        "% " + "=" * 74,
        "% ERROR CONCENTRATION — generated by error_concentration.py. DO NOT HAND-EDIT.",
        f"% arm: {a.family}"
        + ("  (five independently trained models; the sd on each cell is"
           " retraining" if a.family == "trainseed" else
           "  (ONE training run; sd is the inference draw"),
        "%       spread, with inference seeds reduced within each run first)"
        if a.family == "trainseed" else
        "%       only -- superseded for the manuscript on 2026-09-08)",
        "% Fraction of a class's misclassified units that land on its single",
        "% most-confused label, averaged over classes with >=2 errors. The null is",
        "% uniform spread over the other K-1 classes.",
        "% " + "=" * 74,
        "",
        f"% 33 classes, subject level, softmax | {dx['n_classes']} classes contribute"
        f" | seed sd {dx['sd']:.3f}",
        "\\newcommand{\\errConcDx}{%.2f}" % dx["mean"],
        "\\newcommand{\\errConcDxUnif}{%.3f}" % dx["uniform"],
        "\\newcommand{\\errConcDxRatio}{%.0f}" % round(dx["ratio"]),
        "",
        f"% 33 classes, subject level, prototype | {dxp['n_classes']} classes",
        "\\newcommand{\\errConcDxPr}{%.2f}" % dxp["mean"],
        "",
        f"% 19 classes, subject level, softmax | {syn['n_classes']} classes",
        "\\newcommand{\\errConcSyn}{%.2f}" % syn["mean"],
        "\\newcommand{\\errConcSynUnif}{%.3f}" % syn["uniform"],
        "\\newcommand{\\errConcSynRatio}{%.0f}" % round(syn["ratio"]),
        "",
    ]
    OUT_TEX.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
