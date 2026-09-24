#!/usr/bin/env python3
"""
Per-class decomposition of the softmax-vs-prototype contrast (Supplementary
Figs. S1 and S2, Sec. 2.5).  thread facebase3d-a1diag-2026-09-04, 2026-09-08.

WHY THIS SCRIPT EXISTS AT ALL. results/perclass_head_*.npz were
ORPHANS: dated 2026-08-18, loaded by figS1 and figS2, and written by nothing in
scripts/. Nobody could regenerate them, so nobody noticed when the arm they
describe stopped being the arm the manuscript reports.

WHAT CHANGED. The npz files held ONE training run (task "B2_corrected")
averaged over its five INFERENCE seeds -- the farthest-point-sampling draw with
weights frozen. Sec. 2.5's headline moved to the five-TRAINING-seed sweep on
2026-09-03, so the decomposition and the quantity it decomposes came from
different models: the section quoted a +3.6 pp macro-F1 difference in one
sentence and decomposed a 6.1 pp one in the next.

  --family legacy     the 2026-08-18 arm, kept so its npz can be regenerated
                      and so this script can be checked against it
  --family trainseed  B{1,2}_corrected_ts{1..5}; inference seeds are averaged
                      down WITHIN each training run first, exactly as
                      aggregate_trainseed.py does, so FPS noise cannot inflate
                      the training spread

VALIDATION GATE. Under --family legacy this script reproduces all four arrays
of the 2026-08-18 npz to 0.0 before it will write anything. That is what
licenses trusting the trainseed arm, which has no reference to check against.

A NOTE ON THE CLASS COUNTS. "Improved / reduced / unchanged" is computed on the
MEAN per-class difference across the five training runs, and the per-run counts
are emitted alongside as a range. They are not the same statement: a class can
sit inside the band on the mean while individual runs put it outside, and the
range is what says whether the sorting is stable. Both are emitted so the prose
cannot quote the tidier one without the other being available.

Usage:  python3 emit_perclass.py --family legacy
        python3 emit_perclass.py --family trainseed
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

PRED = C.OUT / "predictions"
PAPER = Path(__file__).resolve().parents[2] / "results" / "paper_macros"
ENC = "pointnet2"
LEVEL = "subject"
BAND = 0.01                     # |dF1| <= BAND counted as unchanged
BASES = [("B1_corrected", "syn"), ("B2_corrected", "dx")]
TRAIN_SEEDS = [1, 2, 3, 4, 5]


def _one_run(task, head):
    """Per-class F1 and one-vs-rest AUC for ONE training run, averaged over its
    inference seeds."""
    files = sorted(glob.glob(str(PRED / f"{task}__{ENC}__{head}__{LEVEL}__seed*.npz")))
    if not files:
        raise SystemExit(f"emit_perclass: no prediction files for {task}/{head}")
    F, A = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        yt, yp, ys = d["y_true"], d["y_pred"], d["y_score"]
        K = len(d["label_names"])
        F.append(f1_score(yt, yp, average=None, labels=np.arange(K),
                          zero_division=0))
        # A class absent from the held-out set has no defined one-vs-rest AUC.
        # nan here and nanmean below, never 0.5 -- a missing class must not be
        # silently scored as chance.
        A.append([roc_auc_score((yt == k).astype(int), ys[:, k])
                  if 0 < (yt == k).sum() < len(yt) else np.nan
                  for k in range(K)])
    d0 = np.load(files[0], allow_pickle=True)
    names = [str(x) for x in d0["label_names"]]
    sup = np.bincount(d0["y_true"], minlength=len(names))
    return (np.mean(F, 0), np.nanmean(np.array(A, float), 0), names, sup,
            len(files))


def collect(base, family):
    """Returns per-class arrays for both heads, plus the per-training-run stack."""
    tasks = ([base] if family == "legacy"
             else [f"{base}_ts{s}" for s in TRAIN_SEEDS])
    per = {"softmax": [], "proto": []}
    names = sup = None
    for t in tasks:
        for head in ("softmax", "proto"):
            f, a, nm, sp, nseed = _one_run(t, head)
            per[head].append((f, a))
            if names is None:
                names, sup = nm, sp
            elif nm != names or not (sp == sup).all():
                raise SystemExit(f"emit_perclass: {t} has a different test set")
    out = {}
    for head, tag in (("softmax", "sm"), ("proto", "pr")):
        out[f"f_{tag}"] = np.mean([x[0] for x in per[head]], 0)
        out[f"a_{tag}"] = np.mean([x[1] for x in per[head]], 0)
        out[f"f_{tag}_runs"] = np.array([x[0] for x in per[head]])
        out[f"a_{tag}_runs"] = np.array([x[1] for x in per[head]])
    out["names"] = np.array(names, dtype=object)
    out["support"] = sup
    out["n_runs"] = len(tasks)
    return out


def counts(diff, f_sm, f_pr):
    """up / down / flat / dead under the BAND convention.

    'dead' -- F1 = 0 under BOTH rules -- is split out of 'flat' because a class
    the model never gets right is not unchanged in the same sense as one that
    moved a little, and pooling them would let ten floor-bound classes read as
    ten genuine ties."""
    dead = (f_sm == 0) & (f_pr == 0)
    return (int(((diff > BAND) & ~dead).sum()),
            int(((diff < -BAND) & ~dead).sum()),
            int(((np.abs(diff) <= BAND) & ~dead).sum()),
            int(dead.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["legacy", "trainseed"], required=True)
    args = ap.parse_args()
    fam = args.family
    L, report = [], []

    L.append("% " + "=" * 74)
    L.append(f"% numbers_perclass_{fam}.tex — MACHINE-GENERATED. Do not hand-edit.")
    L.append("%   generator : facebench/eval/emit_perclass.py "
             f"--family {fam}")
    if fam == "trainseed":
        L.append("%   arm       : B{1,2}_corrected_ts{1..5}, five independently")
        L.append("%               trained models; inference seeds averaged down")
        L.append("%               WITHIN each before averaging across.")
        L.append("%   counts    : computed on the MEAN per-class difference. The")
        L.append("%               per-run range is emitted beside every count --")
        L.append("%               quoting one without the other overstates how")
        L.append("%               stable the sorting is.")
    else:
        L.append("%   arm       : ONE training run, five inference seeds. This is")
        L.append("%               the 2026-08-18 arm. Superseded for the")
        L.append("%               manuscript by --family trainseed; kept so the")
        L.append("%               orphan npz can be regenerated.")
    L.append("% " + "=" * 74)
    L.append("")

    P = "ts" if fam == "trainseed" else ""
    for base, tag in BASES:
        d = collect(base, fam)
        f_sm, f_pr, a_sm, a_pr = d["f_sm"], d["f_pr"], d["a_sm"], d["a_pr"]
        diff = f_pr - f_sm
        K = len(diff)
        sup = d["support"]

        dest = C.OUT / (f"perclass_head_{base}"
                        + ("_trainseed" if fam == "trainseed" else "") + ".npz")
        if fam == "legacy":
            ref = np.load(C.OUT / f"perclass_head_{base}.npz", allow_pickle=True)
            for k in ("f_sm", "f_pr", "a_sm", "a_pr"):
                delta = np.nanmax(np.abs(d[k] - ref[k]))
                if delta > 1e-9:
                    raise SystemExit(
                        f"emit_perclass: {base}/{k} differs from the 2026-08-18 "
                        f"npz by {delta:.3e} — the reader is not equivalent, "
                        f"nothing written")
            report.append(f"{base}: reproduces the 2026-08-18 npz exactly")
        np.savez_compressed(dest, **d)

        up, dn, fl, dd = counts(diff, f_sm, f_pr)
        tot = 100 * diff.mean()
        large = sup > 2
        from_large = 100 * diff[large].sum() / K
        from_small = 100 * diff[~large].sum() / K
        mad_f = float(np.abs(diff).mean())
        mad_a = float(np.nanmean(np.abs(a_pr - a_sm)))

        S = tag[0].upper() + tag[1:]
        L.append(f"% ---- {base} ({K} classes), {d['n_runs']} training run(s) ----")
        for nm, val in ((f"{S}HeadClassesUp", up), (f"{S}HeadClassesDown", dn),
                        (f"{S}HeadClassesFlat", fl), (f"{S}HeadClassesDead", dd)):
            L.append("\\newcommand{\\%s%s}{%d}" % (P, nm, val))
        # NOT emitted: the macro-F1 total itself. emit_numbers_trainseed.py
        # already owns \ts{Dx,Syn}HeadFoneDiff for exactly this quantity, and
        # the cross-check below proves the two agree, so a second name for it
        # would be one more thing that can drift.
        L.append("\\newcommand{\\%s%sHeadFoneFromLarge}{%.1f}" % (P, S, from_large))
        L.append("\\newcommand{\\%s%sHeadFoneFromSmall}{%.1f}" % (P, S, from_small))
        L.append("\\newcommand{\\%s%sNLarge}{%d}" % (P, S, int(large.sum())))
        L.append("\\newcommand{\\%s%sNSmall}{%d}" % (P, S, int((~large).sum())))
        L.append("\\newcommand{\\%s%sPerClassFoneMad}{%.3f}" % (P, S, mad_f))
        L.append("\\newcommand{\\%s%sPerClassAucMad}{%.3f}" % (P, S, mad_a))
        L.append("\\newcommand{\\%s%sPerClassRatio}{%.1f}" % (P, S, mad_f / mad_a))

        if fam == "trainseed":
            # CROSS-CHECK against the contrast pipeline. contrast_tests_trainseed
            # .csv computes the same prototype-minus-softmax macro-F1 difference
            # by an independent route (paired within each run, in analyze_ci.py).
            # If the decomposition below and the headline it decomposes ever
            # drift apart again, this is what catches it -- that drift is the
            # whole reason this script exists.
            import csv as _csv
            rows = [r for r in _csv.DictReader(
                        open(C.OUT / "contrast_tests_trainseed.csv"))
                    if r["contrast"] == f"C1a_head_subject[{ENC}]"
                    and r["metric"] == "macro_f1"
                    and r["task"].startswith(base + "_ts")]
            if len(rows) != 5:
                raise SystemExit(f"emit_perclass: {len(rows)} contrast rows for "
                                 f"{base}, expected 5")
            # csv stores softmax-minus-proto; this script computes proto-minus-
            # softmax, hence the negation.
            ref_tot = -100 * np.mean([float(r["diff_mean"]) for r in rows])
            if abs(ref_tot - tot) > 0.05:
                raise SystemExit(
                    f"emit_perclass: {base} per-class total {tot:+.2f} pp "
                    f"disagrees with contrast_tests_trainseed {ref_tot:+.2f} pp")
            report.append(f"{base}: cross-check vs contrast pipeline "
                          f"{tot:+.3f} == {ref_tot:+.3f} pp")

            # Per-run counts. See the module docstring: the mean-difference
            # count and the per-run counts answer different questions.
            pr = [counts(fr - fs, fs, fr)
                  for fs, fr in zip(d["f_sm_runs"], d["f_pr_runs"])]
            tots = [100 * (fr - fs).mean()
                    for fs, fr in zip(d["f_sm_runs"], d["f_pr_runs"])]
            for i, nm in enumerate(["Up", "Down", "Flat", "Dead"]):
                lo, hi = min(x[i] for x in pr), max(x[i] for x in pr)
                L.append("\\newcommand{\\ts%sHeadClasses%sRange}{%d--%d}"
                         % (S, nm, lo, hi))
            # Likewise \ts*HeadFoneDiffRange is emit_numbers_trainseed.py's.
            report.append(f"{base}: mean diff {tot:+.2f} pp "
                          f"(per-run {min(tots):+.2f} to {max(tots):+.2f}), "
                          f"up/down/flat/dead {up}/{dn}/{fl}/{dd}, "
                          f"from n>2 {from_large:+.2f} pp")
        L.append("")

    L = [x for x in L if x != ""] + [""]
    dest_tex = C.OUT / f"numbers_perclass_{fam}.tex"
    dest_tex.write_text("\n".join(L) + "\n")
    if fam == "trainseed":
        (PAPER / "numbers_perclass_trainseed.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {dest_tex}")
    for r in report:
        print(f"  {r}")


if __name__ == "__main__":
    main()
