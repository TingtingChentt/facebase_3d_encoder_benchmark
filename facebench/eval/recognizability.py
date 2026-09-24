#!/usr/bin/env python3
"""
What makes a syndrome recognizable — class size, matched-size spread, and the
RASopathy confusion claim.  (Results, "What makes a syndrome recognizable")
thread facebase3d-paper-2026-08-08.

RESOLVES THE ONLY \\conflict IN THE MANUSCRIPT. Two project documents disagreed:
the ICIBM poster (2026-07-24) reported rho=+0.56 explaining 28% of variance;
results_analysis.md Finding 5 (2026-05-04) reported r=0.256, "weak". Neither was
wrong — THEY WERE COMPUTED ON DIFFERENT TASKS. Recomputed here from the seeded
predictions:

    33-class (clinical diagnosis), subject level : rho = +0.60
    19-class (syndrome category),  subject level : rho = -0.12

The poster's figure matches the 33-class task; results_analysis.md's matches the
19-class/scan-level setting. The relationship is real at the fine-grained label
set and absent at the pathway-level one, which is itself the finding — so the
paper reports both rather than picking the flattering one.

THREE THINGS ARE MEASURED.

  1. rho(class size, per-class F1), Spearman, with a bootstrap-over-classes
     interval. Spearman rather than Pearson because class sizes are heavily
     right-skewed (a few hundred-scan classes against many with <10) and a
     Pearson coefficient there mostly reports the largest class's leverage.
     Pearson R^2 is emitted alongside for continuity with the older documents.

  2. Spread at MATCHED class size. The draft claimed F1 spans 0.33-0.80 at
     matched size without defining "matched". Here the band is stated
     explicitly (BAND_LO..BAND_HI test subjects) and the span is whatever the
     classes in it actually do.

  3. The RASopathy claim. The draft asserted the RASopathies "are confused with
     one another more than with anything else". THAT IS FALSE and this script
     is what falsified it: only ~27% of their errors land on another RASopathy
     and the single most frequent destination is a non-RASopathy. What is true
     is ENRICHMENT — that 27% against a ~9% share of the test set. The emitted
     macros carry the enrichment framing, and the raw destination ranking is
     printed so the prose can name the actual top confusion.

Reads only cached prediction arrays — no model, no GPU.
Usage: python3 recognizability.py [--emit]
"""
import argparse
import collections
import glob
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402
from trainseed_pred import run_groups                       # noqa: E402

PRED = C.OUT / "predictions"
OUT_TEX = C.OUT / "numbers_from_recog.tex"
N_BOOT = 2000
BOOT_SEED = 0          # fixed: this interval must not move between runs
BAND_LO, BAND_HI = 5, 12
RAS_KEYS = ("noonan", "costello", "cfc", "cardiofacio")


def load(task, level, head="softmax", family="trainseed"):
    """Per-class F1 and test size.

    Inference seeds are averaged WITHIN each training run, then across runs --
    the order aggregate_trainseed.py uses. Before 2026-09-08 this globbed a
    single task and so averaged five samplings of ONE model while the
    manuscript reported five models."""
    run_f1, sizes, names = [], [], None
    for _, files in run_groups(PRED, task, head, level, family):
        f1s = []
        for f in files:
            d = np.load(f, allow_pickle=True)
            yt, yp, K = d["y_true"], d["y_pred"], int(d["num_classes"])
            f1s.append(f1_score(yt, yp, average=None,
                                labels=np.arange(K), zero_division=0))
            sizes.append(np.array([(yt == c).sum() for c in range(K)]))
            names = [str(x) for x in d["label_names"]]
        run_f1.append(np.mean(f1s, 0))
    if names is None:
        raise SystemExit(f"no predictions for {task}/{head}/{level}")
    keep = np.mean(sizes, 0) > 0
    return (np.mean(sizes, 0)[keep], np.mean(run_f1, 0)[keep],
            [n for n, k in zip(names, keep) if k])


def rho_ci(size, f1):
    rho = stats.spearmanr(size, f1).statistic
    rng = np.random.default_rng(BOOT_SEED)
    n = len(size)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(np.unique(size[i])) < 3:
            continue
        r = stats.spearmanr(size[i], f1[i]).statistic
        if not np.isnan(r):
            boots.append(r)
    return rho, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rasopathy(task="B2_corrected", level="subject", head="softmax",
              family="trainseed"):
    """RASopathy-to-RASopathy confusion, pooled over every cached prediction.

    Counts are pooled across training runs rather than averaged: they are
    counts, and an average of counts over runs is not one. The per-run WITHIN
    FRACTION is also returned as a range so the pooled figure cannot be quoted
    as if every model behaved the same way."""
    groups = run_groups(PRED, task, head, level, family)
    names = [str(x) for x in
             np.load(groups[0][1][0], allow_pickle=True)["label_names"]]
    ras = [i for i, n in enumerate(names)
           if any(k in n.lower() for k in RAS_KEYS)]
    dest, n_err, per_run = collections.Counter(), 0, []
    for _, files in groups:
        r_within = r_err = 0
        for f in files:
            d = np.load(f, allow_pickle=True)
            yt, yp = d["y_true"], d["y_pred"]
            m = np.isin(yt, ras) & (yt != yp)
            n_err += int(m.sum()); r_err += int(m.sum())
            for p in yp[m]:
                dest[names[p]] += 1
                if any(q in names[p].lower() for q in RAS_KEYS):
                    r_within += 1
        per_run.append(r_within / max(r_err, 1))
    within = sum(v for k, v in dest.items()
                 if any(q in k.lower() for q in RAS_KEYS))
    yt = np.load(groups[0][1][0], allow_pickle=True)["y_true"]
    prev = np.array([(yt == c).sum() for c in range(len(names))], float)
    share = prev[ras].sum() / prev.sum()
    return dict(names=[names[i] for i in ras], n_err=n_err, within=within,
                frac=within / max(n_err, 1), share=float(share),
                enrich=(within / max(n_err, 1)) / share, dest=dest,
                frac_lo=min(per_run), frac_hi=max(per_run), n_runs=len(groups))


def _pretty(name):
    """Class label as the manuscript writes it: 'Noonan Syndrome' -> 'Noonan
    syndrome'. The cached labels are title-cased; the prose is not."""
    return name.replace(" Syndrome", " syndrome")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true")
    p.add_argument("--family", choices=["legacy", "trainseed"],
                   default="trainseed",
                   help="trainseed (default) reads the five independently "
                        "trained models; legacy reads the single pre-sweep "
                        "checkpoint this script used before 2026-09-08")
    a = p.parse_args()

    res = {}
    for task, level, lbl in [("B2_corrected", "subject", "33-class subject"),
                             ("B1_corrected", "subject", "19-class subject")]:
        size, f1, names = load(task, level, family=a.family)
        rho, lo, hi = rho_ci(size, f1)
        r2 = stats.pearsonr(size, f1).statistic ** 2
        res[lbl] = dict(rho=rho, lo=lo, hi=hi, r2=r2, n=len(size),
                        size=size, f1=f1, names=names)
        print(f"{lbl}: n={len(size):2}  rho={rho:+.2f} [{lo:+.2f},{hi:+.2f}]  "
              f"pearson R2={r2:.2f}")

    d = res["33-class subject"]
    band = (d["size"] >= BAND_LO) & (d["size"] <= BAND_HI)
    span = (d["f1"][band].min(), d["f1"][band].max())
    print(f"\nmatched size {BAND_LO}-{BAND_HI} test subjects: "
          f"{int(band.sum())} classes, F1 {span[0]:.2f}-{span[1]:.2f}")
    order = np.argsort(-d["f1"])
    print("  best :", ", ".join(f"{d['names'][i][:22]} {d['f1'][i]:.2f}"
                                for i in order[:5]))
    print("  worst:", ", ".join(f"{d['names'][i][:22]} {d['f1'][i]:.2f}"
                                for i in order[-5:]))

    r = rasopathy(family=a.family)
    print(f"\nRASopathy classes: {r['names']}")
    print(f"  errors {r['n_err']}, to another RASopathy {r['within']} "
          f"({r['frac']:.0%}); test-set share {r['share']:.0%}; "
          f"enrichment {r['enrich']:.1f}x")
    print("  top destinations:", ", ".join(
        f"{k[:26]}={v}" for k, v in r["dest"].most_common(4)))

    if not a.emit:
        return

    dsyn = res["19-class subject"]
    L = [
        "% " + "=" * 74,
        "% RECOGNIZABILITY — generated by recognizability.py. DO NOT HAND-EDIT.",
        "% Resolves the poster-vs-results_analysis rho conflict: the two documents",
        "% computed DIFFERENT TASKS. Both values below are from the seeded 5-seed",
        "%% predictions. Intervals bootstrap over CLASSES (seed %d, %d draws)."
        % (BOOT_SEED, N_BOOT),
        f"% arm: {a.family}"
        + ("  — five independently trained models, inference seeds reduced"
           " within each run first" if a.family == "trainseed"
           else "  — ONE training run; superseded for the manuscript 2026-09-08"),
        "% " + "=" * 74,
        "",
        f"% 33-class clinical diagnosis, subject level, {d['n']} classes",
        "\\newcommand{\\rhoSizeDx}{%+.2f}" % d["rho"],
        "\\newcommand{\\rhoSizeDxCI}{[%+.2f, %+.2f]}" % (d["lo"], d["hi"]),
        "\\newcommand{\\varExplDx}{%.0f\\%%}" % (100 * d["r2"]),
        "",
        f"% 19-class syndrome category, subject level, {dsyn['n']} classes",
        "\\newcommand{\\rhoSizeSyn}{%+.2f}" % dsyn["rho"],
        "\\newcommand{\\rhoSizeSynCI}{[%+.2f, %+.2f]}" % (dsyn["lo"], dsyn["hi"]),
        "",
        f"% classes with {BAND_LO}-{BAND_HI} test subjects: {int(band.sum())}",
        "\\newcommand{\\fOneBandLo}{%d}" % BAND_LO,
        "\\newcommand{\\fOneBandHi}{%d}" % BAND_HI,
        "\\newcommand{\\fOneBandN}{%d}" % int(band.sum()),
        "\\newcommand{\\fOneSpanLo}{%.2f}" % span[0],
        "\\newcommand{\\fOneSpanHi}{%.2f}" % span[1],
        "",
        "% RASopathy confusions — ENRICHMENT, not dominance. The draft's",
        "% \"confused with one another more than with anything else\" is FALSE:",
        f"% the plurality of their errors leave the group, top destination"
        f" {r['dest'].most_common(1)[0][0]}.",
        "\\newcommand{\\rasWithinFrac}{%.0f\\%%}" % (100 * r["frac"]),
        "\\newcommand{\\rasShare}{%.0f\\%%}" % (100 * r["share"]),
        "\\newcommand{\\rasEnrich}{%.0f}" % round(r["enrich"]),
        f"% per-run within-fraction range over {r['n_runs']} training run(s): "
        f"{r['frac_lo']:.0%} to {r['frac_hi']:.0%}",
        "\\newcommand{\\rasWithinFracLo}{%.0f\\%%}" % (100 * r["frac_lo"]),
        "\\newcommand{\\rasWithinFracHi}{%.0f\\%%}" % (100 * r["frac_hi"]),
        "",
        "% These four were hand-maintained in numbers_sal_err.tex until"
        " 2026-09-08 and",
        "% described the single pre-sweep checkpoint. Emitted here so they"
        " cannot drift",
        "% from \\rasWithinFrac / \\rasEnrich again.",
        f"% top destination counted over {r['n_runs']} training run(s)",
        "\\newcommand{\\rasNErr}{%d}" % r["n_err"],
        "\\newcommand{\\rasOutFrac}{%.0f\\%%}" % (100 * (1 - r["frac"])),
        "\\newcommand{\\rasTopDest}{%s}" % _pretty(r["dest"].most_common(1)[0][0]),
        "\\newcommand{\\rasTopDestFrac}{%.0f\\%%}"
        % (100 * r["dest"].most_common(1)[0][1] / max(r["n_err"], 1)),
        "",
    ]
    OUT_TEX.write_text("\n".join(L) + "\n")
    print(f"\nwrote {OUT_TEX}")


if __name__ == "__main__":
    main()
