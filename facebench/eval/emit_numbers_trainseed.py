#!/usr/bin/env python3
"""
Emit the manuscript macros for the TRAINING-SEED sweep.
thread: facebase3d-trainseed-2026-08-31

Companion to emit_numbers.py. That emitter reports one frozen checkpoint per
cell with an inference-seed SD and a subject bootstrap CI; this one reports the
five independently retrained models behind the same cell.

FOUR MACROS PER QUANTITY, never pooled:
  \\tsX        mean over 5 training seeds (inference seeds averaged down WITHIN
              each training seed first, so FPS noise cannot inflate the spread)
  \\tsXSD      SD across those 5 models          <- retraining variance  (NEW)
  \\tsXCI      95% bootstrap CI over test subjects, averaged across the 5
              models                             <- cohort-sampling variance
  \\tsXRange   [min, max] over the 5 models
  \\tsXLegZ    where the published single run sits, in SD units (diagnostic)

CONTRASTS carry \\tsX (mean pp), \\tsXSD (SD pp) and \\tsXNSig ("k of five"),
the number of models whose own subject bootstrap CI excluded zero. A contrast
is only as strong as the number of retrainings that reproduce its sign.

The test split is frozen at seed 42 in all 100 runs, so every SD here is
retraining variance and nothing else.

Usage: python3 emit_numbers_trainseed.py
"""
import csv
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

OUT = C.OUT
DEST = OUT / "numbers_from_trainseed.tex"
PAPER = Path(__file__).resolve().parents[2] / "results" / "paper_macros" / "numbers_trainseed.tex"

SUMMARY = list(csv.DictReader(open(OUT / "trainseed_summary.csv")))
CONTRAST = list(csv.DictReader(open(OUT / "contrast_tests_trainseed.csv")))

P2 = "pointnet2"

# (macro stem matching the existing prose macro, task, encoder, head, level, metric)
CELLS = [
    # -- Fig. 2 headline cells quoted in Results 2.3 -------------------------
    ("combAcc",       "C_corrected",  P2,         "softmax", "scan",    "accuracy"),
    ("combAuc",       "C_corrected",  P2,         "softmax", "scan",    "macro_auc"),
    # A2 reads the A2_fix sweep, not A2. Both are five retrainings of the same
    # task on the same frozen split; they differ only in whether early stopping
    # was allowed to fire, and on this task patience=20 counted on a noisy
    # validation macro-F1 truncated runs inside the ~50-epoch initial plateau.
    # The truncated arm is kept and emitted below as the \ts*ES macros, because
    # the Methods statement about the stopping rule has to be checkable.
    ("ofcBinAcc",     "A2_fix",       P2,         "softmax", "scan",    "accuracy"),
    ("ofcBinAuc",     "A2_fix",       P2,         "softmax", "scan",    "macro_auc"),
    ("ofcBinAccDG",   "A2_fix",       "dgcnn",    "softmax", "scan",    "accuracy"),
    ("ofcBinAucDG",   "A2_fix",       "dgcnn",    "softmax", "scan",    "macro_auc"),
    ("ofcBinAccGM",   "A2_fix",       "geommlp",  "softmax", "scan",    "accuracy"),
    ("ofcBinAucGM",   "A2_fix",       "geommlp",  "softmax", "scan",    "macro_auc"),
    ("ofcBinAccPN",   "A2_fix",       "pointnet", "softmax", "scan",    "accuracy"),
    ("ofcBinAucPN",   "A2_fix",       "pointnet", "softmax", "scan",    "macro_auc"),
    # -- A2 as it came out WITH early stopping (the truncated arm) -----------
    ("ofcBinAccES",   "A2",           P2,         "softmax", "scan",    "accuracy"),
    ("ofcBinAucES",   "A2",           P2,         "softmax", "scan",    "macro_auc"),
    ("ofcBinAccDGES", "A2",           "dgcnn",    "softmax", "scan",    "accuracy"),
    ("ofcBinAucDGES", "A2",           "dgcnn",    "softmax", "scan",    "macro_auc"),
    # -- A1 (four-class cleft subtyping) -------------------------------------
    # ENCODER TOKENS: Pt = PointNet++, DG = DGCNN, GM = GeomMLP, Pnet =
    # PointNet. Pnet rather than PN on purpose -- numbers_reruns.tex binds
    # \ofcFourAucPN to POINTNET++, so a \tsOfcFourAucPN meaning PointNet would
    # put two different encoders behind near-identical names in one subsection.
    # Reads A1_fix for exactly the reason A2 reads A2_fix: the same ln(K)
    # plateau, the same patience-20 rule counting on noise inside it, the same
    # five retrainings on the same frozen split. Confirmed 2026-09-07 by the
    # 20-run sweep — the truncated arm is not a weaker measurement of the task,
    # it is a measurement of the stopping rule (DGCNN 0.547 -> 0.684 macro AUC).
    # GeomMLP is the control: it has no plateau and does not move.
    ("ofcFourAucGM",    "A1_fix",     "geommlp",  "softmax", "scan",    "macro_auc"),
    ("ofcFourAccGM",    "A1_fix",     "geommlp",  "softmax", "scan",    "accuracy"),
    ("ofcFourAccPt",    "A1_fix",     P2,         "softmax", "scan",    "accuracy"),
    ("ofcFourAucPt",    "A1_fix",     P2,         "softmax", "scan",    "macro_auc"),
    ("ofcFourAccDG",    "A1_fix",     "dgcnn",    "softmax", "scan",    "accuracy"),
    ("ofcFourAucDG",    "A1_fix",     "dgcnn",    "softmax", "scan",    "macro_auc"),
    ("ofcFourAccPnet",  "A1_fix",     "pointnet", "softmax", "scan",    "accuracy"),
    ("ofcFourAucPnet",  "A1_fix",     "pointnet", "softmax", "scan",    "macro_auc"),
    ("ofcFourFoneDG",   "A1_fix",     "dgcnn",    "softmax", "scan",    "macro_f1"),
    ("ofcFourFonePt",   "A1_fix",     P2,         "softmax", "scan",    "macro_f1"),
    # -- A1 as it came out WITH early stopping (the truncated arm) -----------
    ("ofcFourAucPtES",  "A1",         P2,         "softmax", "scan",    "macro_auc"),
    ("ofcFourAccPtES",  "A1",         P2,         "softmax", "scan",    "accuracy"),
    ("ofcFourAucDGES",  "A1",         "dgcnn",    "softmax", "scan",    "macro_auc"),
    ("ofcFourAucGMES",  "A1",         "geommlp",  "softmax", "scan",    "macro_auc"),
    ("ofcFourAucPnetES","A1",         "pointnet", "softmax", "scan",    "macro_auc"),
    ("synAucScanDG",  "B1_corrected", "dgcnn",    "softmax", "scan",    "macro_auc"),
    ("dxAucScanDG",   "B2_corrected", "dgcnn",    "softmax", "scan",    "macro_auc"),
    ("synAccScanDG",  "B1_corrected", "dgcnn",    "softmax", "scan",    "accuracy"),
    ("dxAccScanDG",   "B2_corrected", "dgcnn",    "softmax", "scan",    "accuracy"),
    # -- Table 2 grid: 5 rows x 2 tasks x 3 metrics, PointNet++ -------------
    ("synAccScan",    "B1_corrected", P2, "softmax", "scan",    "accuracy"),
    ("synFoneScan",   "B1_corrected", P2, "softmax", "scan",    "macro_f1"),
    ("synAucScan",    "B1_corrected", P2, "softmax", "scan",    "macro_auc"),
    ("dxAccScan",     "B2_corrected", P2, "softmax", "scan",    "accuracy"),
    ("dxFoneScan",    "B2_corrected", P2, "softmax", "scan",    "macro_f1"),
    ("dxAucScan",     "B2_corrected", P2, "softmax", "scan",    "macro_auc"),
    ("synAccSubjSM",  "B1_corrected", P2, "softmax", "subject", "accuracy"),
    ("synFoneSubjSM", "B1_corrected", P2, "softmax", "subject", "macro_f1"),
    ("synAucSubjSM",  "B1_corrected", P2, "softmax", "subject", "macro_auc"),
    ("dxAccSubjSM",   "B2_corrected", P2, "softmax", "subject", "accuracy"),
    ("dxFoneSubjSM",  "B2_corrected", P2, "softmax", "subject", "macro_f1"),
    ("dxAucSubjSM",   "B2_corrected", P2, "softmax", "subject", "macro_auc"),
    ("synAccSubjPr",  "B1_corrected", P2, "proto",   "subject", "accuracy"),
    ("synFoneSubjPr", "B1_corrected", P2, "proto",   "subject", "macro_f1"),
    ("synAucSubjPr",  "B1_corrected", P2, "proto",   "subject", "macro_auc"),
    ("dxAccSubjPr",   "B2_corrected", P2, "proto",   "subject", "accuracy"),
    ("dxFoneSubjPr",  "B2_corrected", P2, "proto",   "subject", "macro_f1"),
    ("dxAucSubjPr",   "B2_corrected", P2, "proto",   "subject", "macro_auc"),
    # Prototype at SCAN level. Without these the only scan-level baseline
    # visible in Table 2 is the softmax one, so a reader differencing the
    # table gets a decision-rule change folded into what reads as an
    # aggregation effect. Restored to the table 2026-09-09;
    # first added 2026-09-08 when she caught the missing row.
    ("synAccScanPr",  "B1_corrected", P2, "proto",   "scan",    "accuracy"),
    ("synFoneScanPr", "B1_corrected", P2, "proto",   "scan",    "macro_f1"),
    ("synAucScanPr",  "B1_corrected", P2, "proto",   "scan",    "macro_auc"),
    ("dxAccScanPr",   "B2_corrected", P2, "proto",   "scan",    "accuracy"),
    ("dxFoneScanPr",  "B2_corrected", P2, "proto",   "scan",    "macro_f1"),
    ("dxAucScanPr",   "B2_corrected", P2, "proto",   "scan",    "macro_auc"),
    ("synAccScanKnn", "B1_corrected", P2, "knn11",   "scan",    "accuracy"),
    ("synFoneScanKnn","B1_corrected", P2, "knn11",   "scan",    "macro_f1"),
    ("synAucScanKnn", "B1_corrected", P2, "knn11",   "scan",    "macro_auc"),
    ("dxAccScanKnn",  "B2_corrected", P2, "knn11",   "scan",    "accuracy"),
    ("dxFoneScanKnn", "B2_corrected", P2, "knn11",   "scan",    "macro_f1"),
    ("dxAucScanKnn",  "B2_corrected", P2, "knn11",   "scan",    "macro_auc"),
    ("synAccSubjKnn", "B1_corrected", P2, "knn11",   "subject", "accuracy"),
    ("synFoneSubjKnn","B1_corrected", P2, "knn11",   "subject", "macro_f1"),
    ("synAucSubjKnn", "B1_corrected", P2, "knn11",   "subject", "macro_auc"),
    ("dxAccSubjKnn",  "B2_corrected", P2, "knn11",   "subject", "accuracy"),
    ("dxFoneSubjKnn", "B2_corrected", P2, "knn11",   "subject", "macro_f1"),
    ("dxAucSubjKnn",  "B2_corrected", P2, "knn11",   "subject", "macro_auc"),
]

# Supplementary Table S1 — combined screening, all four encoders, softmax.
# Stems match the existing prose macros (\combAggGmScan -> \tsCombAggGmScan).
S1_ENC = [("Gm", "geommlp"), ("Pn", "pointnet"), ("Pt", P2), ("Dg", "dgcnn")]
for _tag, _enc in S1_ENC:
    for _lvl, _lab in [("scan", "Scan"), ("subject", "Subj")]:
        for _suf, _met in [("", "accuracy"), ("F", "macro_f1"), ("A", "macro_auc")]:
            CELLS.append((f"combAgg{_tag}{_lab}{_suf}", "C_corrected", _enc,
                          "softmax", _lvl, _met))

# (macro stem, contrast prefix, encoder, task, metric)
CONTRASTS = [
    ("aggGainSyn",       "C2_agg_softmax",   P2, "B1_corrected", "accuracy"),
    ("aggGainSynPr",     "C2_agg_proto",     P2, "B1_corrected", "accuracy"),
    ("aggGainDx",        "C2_agg_softmax",   P2, "B2_corrected", "accuracy"),
    ("aggGainDxPr",      "C2_agg_proto",     P2, "B2_corrected", "accuracy"),
    # The ladder Table 2 shows: scan-level softmax -> subject-level
    # prototype. Changes decision rule AND level together, so it is NOT an
    # aggregation effect and must never be quoted as one. Named at length
    # so it cannot be confused with aggGain*Pr above.
    ("protoSubjVsSmScanSyn", "C2c_proto_subj_vs_softmax_scan", P2,
     "B1_corrected", "accuracy"),
    ("protoSubjVsSmScanDx",  "C2c_proto_subj_vs_softmax_scan", P2,
     "B2_corrected", "accuracy"),
    ("aggGainCombAcc",   "C2_agg_softmax",   P2, "C_corrected",  "accuracy"),
    ("aggGainCombFone",  "C2_agg_softmax",   P2, "C_corrected",  "macro_f1"),
    ("aggGainCombAuc",   "C2_agg_softmax",   P2, "C_corrected",  "macro_auc"),
    ("aggGainSynKnn",    "C2_agg_knn11",     P2, "B1_corrected", "accuracy"),
    ("aggGainSynKnnAuc", "C2_agg_knn11",     P2, "B1_corrected", "macro_auc"),
    ("aggGainDxKnn",     "C2_agg_knn11",     P2, "B2_corrected", "accuracy"),
    ("aggGainDxKnnAuc",  "C2_agg_knn11",     P2, "B2_corrected", "macro_auc"),
    # head contrast is emitted as prototype MINUS softmax, matching the prose
    ("dxHeadAccDiff",    "C1a_head_subject", P2, "B2_corrected", "accuracy"),
    ("dxHeadFoneDiff",   "C1a_head_subject", P2, "B2_corrected", "macro_f1"),
    ("dxHeadAucDiff",    "C1a_head_subject", P2, "B2_corrected", "macro_auc"),
    ("synHeadAccDiff",   "C1a_head_subject", P2, "B1_corrected", "accuracy"),
    ("synHeadFoneDiff",  "C1a_head_subject", P2, "B1_corrected", "macro_f1"),
    ("synHeadAucDiff",   "C1a_head_subject", P2, "B1_corrected", "macro_auc"),
]
for _tag, _enc in S1_ENC:
    for _suf, _met in [("", "accuracy"), ("F", "macro_f1"), ("A", "macro_auc")]:
        CONTRASTS.append((f"combAggD{_tag}{_suf}", "C2_agg_softmax", _enc,
                          "C_corrected", _met))

# C1a is emitted softmax-minus-prototype; the prose states prototype-minus-softmax.
FLIP = {"dxHeadAccDiff", "dxHeadFoneDiff", "dxHeadAucDiff",
        "synHeadAccDiff", "synHeadFoneDiff", "synHeadAucDiff"}

WORDS = {0: "none of five", 1: "one of five", 2: "two of five",
         3: "three of five", 4: "four of five", 5: "all five"}


def num(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def cell(task, enc, head, level, metric):
    r = [x for x in SUMMARY if x["base_task"] == task and x["encoder"] == enc
         and x["head"] == head and x["level"] == level and x["metric"] == metric]
    if len(r) != 1:
        sys.exit(f"{len(r)} rows for {task}/{enc}/{head}/{level}/{metric}")
    return r[0]


def tex(s):
    """LaTeX-safe signed number: a leading minus must be math mode."""
    return f"${s}$" if s.startswith("-") else s


def main():
    L = [
        "% " + "=" * 72,
        "% numbers_trainseed.tex — MACHINE-GENERATED. Do not hand-edit.",
        "%",
        "%   generator : facebench/eval/emit_numbers_trainseed.py",
        f"%   sources   : results/trainseed_summary.csv  ({len(SUMMARY)} rows)",
        f"%               results/contrast_tests_trainseed.csv  ({len(CONTRAST)} rows)",
        "%",
        "% 100 runs = 5 tasks x 4 encoders x 5 TRAINING seeds. The data split is",
        "% frozen at seed 42 in every run, so the test subjects are byte-identical",
        "% across seeds and \\tsXSD is retraining variance alone.",
        "%",
        "%   \\tsX        mean over the 5 retrained models",
        "%   \\tsXSD      SD across those models        (retraining variance)",
        "%   \\tsXCI      mean 95% bootstrap CI over test subjects (cohort variance)",
        "%   \\tsXRange   [min, max] over the 5 models",
        "%   \\tsXLegZ    z of the published single run within the 5-model distribution",
        "% The variance sources are reported side by side and NEVER pooled.",
        "% " + "=" * 72,
        "",
    ]

    L.append("% ---- point estimates " + "-" * 52)
    for stem, task, enc, head, level, metric in CELLS:
        r = cell(task, enc, head, level, metric)
        m, sd = num(r["train_seed_mean"]), num(r["train_seed_sd"])
        lo, hi = num(r["boot_ci_lo"]), num(r["boot_ci_hi"])
        mn, mx = num(r["train_seed_min"]), num(r["train_seed_max"])
        z = num(r["legacy_z"])
        leg = num(r["legacy_value"])
        S = stem[0].upper() + stem[1:]
        pub = f"{leg:.3f}" if leg is not None else "n/a"
        L.append(f"% trainseed_summary.csv | {task}/{enc}/{head}/{level}/{metric} "
                 f"| n_train_seeds={r['n_train_seeds']} n_units={r['n_units']} "
                 f"| published={pub}")
        L.append("\\newcommand{\\ts%s}{%.3f}" % (S, m))
        L.append("\\newcommand{\\ts%sSD}{%.3f}" % (S, sd))
        L.append("\\newcommand{\\ts%sCI}{95\\%% CI %.3f to %.3f}" % (S, lo, hi))
        L.append("\\newcommand{\\ts%sCIb}{[%.3f, %.3f]}" % (S, lo, hi))
        L.append("\\newcommand{\\ts%sRange}{%.3f--%.3f}" % (S, mn, mx))
        L.append("\\newcommand{\\ts%sPct}{%.1f\\%%}" % (S, 100 * m))
        L.append("\\newcommand{\\ts%sLegZ}{%s}" % (S, tex(f"{z:+.2f}") if z is not None else "n/a"))
        L.append("")

    L.append("% ---- contrasts, percentage points " + "-" * 39)
    for stem, pref, enc, task, metric in CONTRASTS:
        rs = sorted([x for x in CONTRAST
                     if x["contrast"].startswith(pref) and f"[{enc}]" in x["contrast"]
                     and x["task"].startswith(task + "_ts") and x["metric"] == metric],
                    key=lambda x: x["task"])
        if len(rs) != 5:
            sys.exit(f"{len(rs)} training seeds for {pref}/{enc}/{task}/{metric}")
        sign = -1.0 if stem in FLIP else 1.0
        d = [sign * 100 * float(x["diff_mean"]) for x in rs]
        nsig = sum(1 for x in rs if x["ci_contains_zero"] == "False")
        S = stem[0].upper() + stem[1:]
        L.append(f"% contrast_tests_trainseed.csv | {pref}[{enc}] {task} {metric}"
                 f"{'  (sign flipped to prototype-minus-softmax)' if sign < 0 else ''}")
        L.append("\\newcommand{\\ts%s}{%s}" % (S, tex(f"{st.mean(d):+.1f}")))
        L.append("\\newcommand{\\ts%sSD}{%.1f}" % (S, st.stdev(d)))
        L.append("\\newcommand{\\ts%sRange}{%s to %s}"
                 % (S, tex(f"{min(d):+.1f}"), tex(f"{max(d):+.1f}")))
        L.append("\\newcommand{\\ts%sNSig}{%s}" % (S, WORDS[nsig]))
        L.append("")

    # ---- paired encoder contrasts, matched by TRAINING SEED --------------
    # DGCNN's AUC edge over PointNet++ is quoted in Results 2.3. Pairing the two
    # encoders within each training seed is the only way to say how often the
    # ordering actually reproduces; marginal SDs cannot answer that.
    PER = list(csv.DictReader(open(OUT / "trainseed_per_run.csv")))

    def series(task, enc, head, level, metric):
        rs = sorted([x for x in PER if x["base_task"] == task and x["encoder"] == enc
                     and x["head"] == head and x["level"] == level
                     and x["metric"] == metric], key=lambda x: int(float(x["train_seed"])))
        if len(rs) != 5:
            sys.exit(f"{len(rs)} training seeds for {task}/{enc}/{head}/{level}/{metric}")
        return [float(x["value"]) for x in rs]

    PAIRED = [("synAucDgVsPt", "B1_corrected", "dgcnn", P2, "macro_auc"),
              ("dxAucDgVsPt",  "B2_corrected", "dgcnn", P2, "macro_auc"),
              ("synAccPtVsDg", "B1_corrected", P2, "dgcnn", "accuracy"),
              ("dxAccPtVsDg",  "B2_corrected", P2, "dgcnn", "accuracy")]
    L.append("% ---- paired encoder differences, matched within training seed " + "-" * 11)
    for stem, task, ea, eb, metric in PAIRED:
        a = series(task, ea, "softmax", "scan", metric)
        b = series(task, eb, "softmax", "scan", metric)
        d = [100 * (x - y) for x, y in zip(a, b)]
        S = stem[0].upper() + stem[1:]
        L.append(f"% trainseed_per_run.csv | {task} {metric}: {ea} minus {eb}, "
                 f"paired by train seed")
        L.append("\\newcommand{\\ts%s}{%s}" % (S, tex(f"{st.mean(d):+.1f}")))
        L.append("\\newcommand{\\ts%sSD}{%.1f}" % (S, st.stdev(d)))
        L.append("\\newcommand{\\ts%sNPos}{%s}"
                 % (S, WORDS[sum(1 for v in d if v > 0)]))
        L.append("")

    # ---- cross-encoder summary quoted in Results 2.5 and the Supplement ----
    f1 = {}
    for tag, enc in S1_ENC:
        rs = [x for x in CONTRAST
              if x["contrast"].startswith("C2_agg_softmax") and f"[{enc}]" in x["contrast"]
              and x["task"].startswith("C_corrected_ts") and x["metric"] == "macro_f1"]
        d = [100 * float(x["diff_mean"]) for x in rs]
        f1[enc] = (st.mean(d), sum(1 for x in rs if x["ci_contains_zero"] == "False"), len(rs))
    neg = [e for e, v in f1.items() if v[0] < 0]
    mags = sorted(abs(v[0]) for v in f1.values())
    L.append("% cross-encoder: macro-F1 change under screening aggregation, "
             "mean over 5 training seeds per encoder")
    for e, v in f1.items():
        L.append(f"%   {e}: {v[0]:+.2f} pp, CI excluded 0 in {v[1]}/{v[2]} models")
    L.append("\\newcommand{\\tsCombAggFoneNRev}{%s}"
             % ("all four" if len(neg) == 4 else f"{len(neg)} of four"))
    L.append("\\newcommand{\\tsCombAggFoneRange}{%.1f to %.1f}" % (mags[0], mags[-1]))
    au = {}
    for tag, enc in S1_ENC:
        rs = [x for x in CONTRAST
              if x["contrast"].startswith("C2_agg_softmax") and f"[{enc}]" in x["contrast"]
              and x["task"].startswith("C_corrected_ts") and x["metric"] == "macro_auc"]
        au[enc] = st.mean(100 * float(x["diff_mean"]) for x in rs)
    amag = sorted(abs(v) for v in au.values())
    L.append("% macro-AUC change under screening aggregation: "
             + ", ".join(f"{e} {v:+.2f}" for e, v in au.items()))
    L.append("\\newcommand{\\tsCombAggAucNDown}{%s}"
             % ("all four" if all(v < 0 for v in au.values())
                else f"{sum(1 for v in au.values() if v < 0)} of four"))
    L.append("\\newcommand{\\tsCombAggAucRange}{%.1f to %.1f}" % (amag[0], amag[-1]))
    L.append("")

    body = "\n".join(L)
    DEST.write_text(body)
    PAPER.write_text(body)
    print(f"wrote {DEST}")
    print(f"wrote {PAPER}")
    print(f"  {len(CELLS)} cells x 6 macros + {len(CONTRASTS)} contrasts x 4 macros")


if __name__ == "__main__":
    main()
