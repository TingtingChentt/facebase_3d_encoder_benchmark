#!/usr/bin/env python3
"""
PART 2 — emit a machine-generated \\newcommand block for paper/numbers.tex.

Reads results/summary_ci.csv + contrast_tests.csv and writes
results/numbers_from_reruns.tex, so the 52 \\prov{} markers in
paper/numbers.tex ("single unseeded run, CI pending") can be retired in one
merge rather than transcribed by hand.

THIS SCRIPT DOES NOT WRITE INTO THE MANUSCRIPT. It only emits the macro block.

THREE MACROS PER QUANTITY. The prose picks which to show:
    \\X        seed-averaged point estimate
    \\XSD      inference-seed SD          — source (i): FPS draw, weights frozen
    \\XCI      bootstrap 95% interval     — source (ii): resampling test SUBJECTS
The two uncertainty sources are NEVER pooled — they answer different questions,
and a single interval spanning both would misrepresent each. See analyze_ci.py.

\\XSD is exactly 0 (not an estimate) for GeomMLP / PointNet / DGCNN: those three
were shown bit-identical across repeat runs, so they are evaluated once. Emitted
rows carry that as a comment rather than a suspiciously clean number.

WHAT THIS DOES NOT MEASURE: training-seed variance. Nothing is retrained;
weights are frozen. Every interval here is inference variance under frozen
weights. Do not relabel it "run-to-run variance" in the prose.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

SUMMARY = C.OUT / "summary_ci.csv"
CONTRASTS = C.OUT / "contrast_tests.csv"
OUT_TEX = C.OUT / "numbers_from_reruns.tex"


# ── the specification ───────────────────────────────────────────────────────
# Every macro is declared here with the exact row it resolves to. This table IS
# the audit trail: nothing is emitted that is not named here, and a spec whose
# row is missing from the CSV is a hard error, never a silent omission.
#
# S(macro, task, encoder, head, level, metric)  -> summary_ci.csv row
# D(macro, contrast, task, metric)              -> contrast_tests.csv row (pp)

def S(macro, task, enc, head, level, metric, fmt="prob", note=""):
    return dict(kind="summary", macro=macro, task=task, encoder=enc, head=head,
                level=level, metric=metric, fmt=fmt, note=note)


def D(macro, contrast, task, metric, fmt="pp", note=""):
    return dict(kind="contrast", macro=macro, contrast=contrast, task=task,
                metric=metric, fmt=fmt, note=note)


SECTIONS = [
    ("Task 5: combined binary (affected vs unaffected, all datasets)", [
        S("combAcc",  "C_corrected", "pointnet2", "softmax", "scan", "accuracy"),
        S("combFone", "C_corrected", "pointnet2", "softmax", "scan", "macro_f1"),
        S("combAuc",  "C_corrected", "pointnet2", "softmax", "scan", "macro_auc"),
    ], "Scan level, matching EXPERIMENTAL_FINDINGS.md line 198 (n_test=3,639)."),

    ("Task 5 at SUBJECT level — NEW, not in numbers.tex. See reply PART 1.", [
        S("combAccSubj",  "C_corrected", "pointnet2", "softmax", "subject", "accuracy"),
        S("combFoneSubj", "C_corrected", "pointnet2", "softmax", "subject", "macro_f1"),
        S("combAucSubj",  "C_corrected", "pointnet2", "softmax", "subject", "macro_auc"),
    ], "C_corrected groups by subject_id (3,639 scans / 2,658 subjects), so unlike\n"
       "the A-family this is a real aggregation, not a relabelled scan row."),

    ("Task 2: OFC binary", [
        S("ofcBinAcc",  "A2", "pointnet2", "softmax", "scan", "accuracy"),
        S("ofcBinFone", "A2", "pointnet2", "softmax", "scan", "macro_f1"),
        S("ofcBinAuc",  "A2", "pointnet2", "softmax", "scan", "macro_auc"),
        S("ofcBinAucDG",   "A2", "dgcnn",    "softmax", "scan", "macro_auc"),
        S("ofcBinAucPnet", "A2", "pointnet",  "softmax", "scan", "macro_auc"),
        S("ofcBinAucGM",   "A2", "geommlp",   "softmax", "scan", "macro_auc"),
    ], "Scan level only: A2's manifest has no subject id, so every scan is its own\n"
       "unit and a subject-level row would duplicate this one exactly.\n"
       "The four macro-AUC rows are the WITHIN-SITE reference for the BINARY\n"
       "leave-one-site-out block in numbers_xsite_bin.tex (\\xsb*): those folds run\n"
       "this same A2 task, so the comparison is like-for-like on the task even\n"
       "though the test sets differ. It is NOT a paired contrast, and it is NOT\n"
       "\\combAuc — the combined-cohort task C_corrected was never run cross-site."),

    ("Task 1: OFC four-class", [
        S("ofcFourAccPN",  "A1", "pointnet2", "softmax", "scan", "accuracy"),
        S("ofcFourAccDG",  "A1", "dgcnn",     "softmax", "scan", "accuracy"),
        S("ofcFourAucGM",  "A1", "geommlp",   "softmax", "scan", "macro_auc"),
        S("ofcFourFoneGM", "A1", "geommlp",   "softmax", "scan", "macro_f1"),
        S("ofcFourAucPN",   "A1", "pointnet2", "softmax", "scan", "macro_auc"),
        S("ofcFourAucDG",   "A1", "dgcnn",     "softmax", "scan", "macro_auc"),
        S("ofcFourAucPnet", "A1", "pointnet",  "softmax", "scan", "macro_auc"),
    ], "Per-encoder picks preserved from numbers.tex: DGCNN best accuracy,\n"
       "GeomMLP best AUC and best macro-F1.\n"
       "The four macro-AUC rows (GM/PN/DG/Pnet) are the WITHIN-SITE reference the\n"
       "cross-site folds in numbers_xsite.tex are read against: the leave-one-site-out\n"
       "runs are this same four-class task, so the comparison is like-for-like on the\n"
       "task even though the test sets differ. It is NOT a paired contrast."),

    ("Task 3: syndrome category, 19 classes, per-scan", [
        S("synAccScan",   "B1_corrected", "pointnet2", "softmax", "scan", "accuracy"),
        S("synFoneScan",  "B1_corrected", "pointnet2", "softmax", "scan", "macro_f1"),
        S("synAucScan",   "B1_corrected", "pointnet2", "softmax", "scan", "macro_auc"),
        S("synAucScanDG", "B1_corrected", "dgcnn",     "softmax", "scan", "macro_auc"),
    ], ""),

    ("Task 4: clinical diagnosis, 33 classes, per-scan", [
        S("dxAccScan",   "B2_corrected", "pointnet2", "softmax", "scan", "accuracy"),
        S("dxFoneScan",  "B2_corrected", "pointnet2", "softmax", "scan", "macro_f1"),
        S("dxAucScan",   "B2_corrected", "pointnet2", "softmax", "scan", "macro_auc"),
        S("dxAucScanDG", "B2_corrected", "dgcnn",     "softmax", "scan", "macro_auc"),
    ], ""),

    ("The head x level grid, 19-class — THE CORE RESULT", [
        S("synAccSubjSM",  "B1_corrected", "pointnet2", "softmax", "subject", "accuracy"),
        S("synFoneSubjSM", "B1_corrected", "pointnet2", "softmax", "subject", "macro_f1"),
        S("synAucSubjSM",  "B1_corrected", "pointnet2", "softmax", "subject", "macro_auc"),
        S("synAccSubjPr",  "B1_corrected", "pointnet2", "proto",   "subject", "accuracy"),
        S("synFoneSubjPr", "B1_corrected", "pointnet2", "proto",   "subject", "macro_f1"),
        S("synAucSubjPr",  "B1_corrected", "pointnet2", "proto",   "subject", "macro_auc"),
    ], "ENCODER-FIXED arm: the *same* checkpoint read through both heads. This is\n"
       "the clean head comparison (contrast C1a)."),

    ("19-class, AS-PUBLISHED proto arm — separate CE-trained checkpoint", [
        S("synAccSubjPrPub",  "B1_proto_cetrain", "pointnet2", "proto", "subject", "accuracy"),
        S("synFoneSubjPrPub", "B1_proto_cetrain", "pointnet2", "proto", "subject", "macro_f1"),
        S("synAucSubjPrPub",  "B1_proto_cetrain", "pointnet2", "proto", "subject", "macro_auc"),
    ], "Different checkpoint from the softmax arm, so this comparison CONFOUNDS\n"
       "head choice with training run (contrast C1b). Reported alongside the\n"
       "encoder-fixed arm so the two are not mistaken for each other."),

    ("The head x level grid, 33-class — THE CORE RESULT", [
        S("dxAccSubjSM",  "B2_corrected", "pointnet2", "softmax", "subject", "accuracy"),
        S("dxFoneSubjSM", "B2_corrected", "pointnet2", "softmax", "subject", "macro_f1"),
        S("dxAucSubjSM",  "B2_corrected", "pointnet2", "softmax", "subject", "macro_auc"),
        S("dxAccSubjPr",  "B2_corrected", "pointnet2", "proto",   "subject", "accuracy"),
        S("dxFoneSubjPr", "B2_corrected", "pointnet2", "proto",   "subject", "macro_f1"),
        S("dxAucSubjPr",  "B2_corrected", "pointnet2", "proto",   "subject", "macro_auc"),
    ], "ENCODER-FIXED arm — see the 19-class note."),

    ("33-class, AS-PUBLISHED proto arm — separate CE-trained checkpoint", [
        S("dxAccSubjPrPub",  "B2_proto_cetrain", "pointnet2", "proto", "subject", "accuracy"),
        S("dxFoneSubjPrPub", "B2_proto_cetrain", "pointnet2", "proto", "subject", "macro_f1"),
        S("dxAucSubjPrPub",  "B2_proto_cetrain", "pointnet2", "proto", "subject", "macro_auc"),
    ], ""),

    ("Aggregation gain (subject minus scan), percentage points", [
        D("aggGainSyn",     "C2_agg_softmax[pointnet2]", "B1_corrected", "accuracy"),
        D("aggGainSynFone", "C2_agg_softmax[pointnet2]", "B1_corrected", "macro_f1"),
        D("aggGainDx",      "C2_agg_softmax[pointnet2]", "B2_corrected", "accuracy"),
        D("aggGainDxFone",  "C2_agg_softmax[pointnet2]", "B2_corrected", "macro_f1"),
    ], "PAIRED bootstrap on the difference (same subject draws in both arms), not\n"
       "two marginal intervals eyeballed against each other. \\XSD here is the SD\n"
       "of the paired difference across inference seeds."),

    ("Head-choice delta (softmax minus prototype), percentage points", [
        D("headDeltaSyn",    "C1a_head_subject[pointnet2]",   "B1_corrected", "accuracy"),
        D("headDeltaDx",     "C1a_head_subject[pointnet2]",   "B2_corrected", "accuracy"),
        D("headDeltaSynPub", "C1b_head_subject_aspublished",  "B1_corrected", "accuracy"),
        D("headDeltaDxPub",  "C1b_head_subject_aspublished",  "B2_corrected", "accuracy"),
    ], "NEW — no macro existed for this. The C1 null claim needs the interval, not\n"
       "just the point difference: 'CI contains 0' alone is equally consistent\n"
       "with being underpowered. See the verdict column in contrast_tests.csv."),

    ("24-class filtered clinical task (B2_r2) — NEW, no macros existed", [
        S("dxFiltAcc",      "B2_r2", "pointnet2", "softmax", "scan", "accuracy"),
        S("dxFiltFone",     "B2_r2", "pointnet2", "softmax", "scan", "macro_f1"),
        S("dxFiltAuc",      "B2_r2", "pointnet2", "softmax", "scan", "macro_auc"),
        S("dxFiltAccSubj",  "B2_r2", "pointnet2", "softmax", "subject", "accuracy"),
        S("dxFiltAucSubj",  "B2_r2", "pointnet2", "softmax", "subject", "macro_auc"),
    ], "numbers.tex already carries \\chanceDxFilt (4.2%) with nothing to compare\n"
       "it against. 377 scans / 149 subjects."),

    ("Class-imbalance strategy variants (A1_S1/S2/S3) — REPLACES \\xsiteAuc*", [
        S("imbWeightedAcc", "A1_S1", "pointnet2", "softmax", "scan", "accuracy"),
        S("imbWeightedAuc", "A1_S1", "pointnet2", "softmax", "scan", "macro_auc"),
        S("imbFocalAcc",    "A1_S2", "pointnet2", "softmax", "scan", "accuracy"),
        S("imbFocalAuc",    "A1_S2", "pointnet2", "softmax", "scan", "macro_auc"),
        S("imbBothAcc",     "A1_S3", "pointnet2", "softmax", "scan", "accuracy"),
        S("imbBothAuc",     "A1_S3", "pointnet2", "softmax", "scan", "macro_auc"),
    ], "These three runs are weighted-sampler / focal-loss / both, trained on the\n"
       "IDENTICAL full-cohort split as A1 (AUDIT_NONDETERMINISM.md section E).\n"
       "They are NOT site splits. \\xsiteAucLo and \\xsiteAucHi are deliberately\n"
       "NOT emitted here: the cross-site claim has no supporting experiment."),

    ("Leakage audit — AFTER values only", [
        S("leakSynAfter",     "B1_corrected", "pointnet2", "softmax", "scan", "accuracy"),
        S("leakDxAfter",      "B2_corrected", "pointnet2", "softmax", "scan", "accuracy"),
        S("leakDxFoneAfter",  "B2_corrected", "pointnet2", "softmax", "scan", "macro_f1"),
        S("leakCombAfter",    "C_corrected",  "pointnet2", "softmax", "scan", "accuracy"),
    ], "Aliases of the corrected-split numbers above, kept so the leakage prose\n"
       "reads independently. The matching *Before* macros are NOT emitted: they\n"
       "come from pre-correction scan-level-split runs that this pipeline never\n"
       "re-ran, so they must stay \\prov{} in numbers.tex."),
]

# Derived quantities: arithmetic on two cells of the SAME summary row. No
# bootstrap interval exists for these because the difference was never
# resampled as a unit — emitting one would be inventing an interval.
DERIVED_GAPS = [
    ("gapSyn",     "B1_corrected", "pointnet2", "softmax", "subject"),
    ("gapSynPr",   "B1_corrected", "pointnet2", "proto",   "subject"),
    ("gapDxScan",  "B2_corrected", "pointnet2", "softmax", "scan"),
    ("gapDxSubj",  "B2_corrected", "pointnet2", "softmax", "subject"),
    ("gapDxSubjPr", "B2_corrected", "pointnet2", "proto",  "subject"),
]

# Inference-seed wobble, reported directly instead of the old \fpsWobble /
# \fpsRunA / \fpsRunB, which were two ad-hoc unseeded runs.
FPS_ROW = ("B2_corrected", "pointnet2", "softmax", "subject", "accuracy")


# ── formatting ──────────────────────────────────────────────────────────────

def f_prob(x):
    return f"{x:.3f}"


def f_pp(x):
    return f"{x * 100:.1f}"


def f_prob_sd(x):
    return f"{x:.4f}"


def f_pp_sd(x):
    return f"{x * 100:.2f}"


FMT = {"prob": (f_prob, f_prob_sd), "pp": (f_pp, f_pp_sd)}


def emit(lines, macro, mean, sd, lo, hi, fmt, src, deterministic):
    fm, fs = FMT[fmt]
    unit = "" if fmt == "prob" else " pp"
    lines.append(f"% {src}")
    lines.append(f"\\newcommand{{\\{macro}}}{{{fm(mean)}}}")
    if deterministic:
        lines.append(f"\\newcommand{{\\{macro}SD}}{{0}}"
                     f"  % exact 0: deterministic encoder, evaluated once")
    else:
        lines.append(f"\\newcommand{{\\{macro}SD}}{{{fs(sd)}}}"
                     f"  % inference-seed SD{unit}, weights frozen")
    lines.append(f"\\newcommand{{\\{macro}CI}}{{[{fm(lo)}, {fm(hi)}]}}"
                 f"  % 95% bootstrap over test subjects")
    lines.append("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_TEX))
    a = ap.parse_args()

    summ = pd.read_csv(SUMMARY)
    con = pd.read_csv(CONTRASTS)

    L = []
    L.append("% " + "=" * 74)
    L.append("% numbers_from_reruns.tex — MACHINE-GENERATED. Do not hand-edit.")
    L.append("%")
    L.append("%   generator : facebench/eval/emit_numbers.py")
    L.append(f"%   sources   : {SUMMARY.relative_to(C.PROJECT)}"
             f"  ({len(summ)} rows)")
    L.append(f"%               {CONTRASTS.relative_to(C.PROJECT)}"
             f"  ({len(con)} rows)")
    L.append("%")
    L.append("% Three macros per quantity — the prose chooses which to show:")
    L.append("%   \\X    seed-averaged point estimate")
    L.append("%   \\XSD  inference-seed SD    (FPS draw; model weights FROZEN)")
    L.append("%   \\XCI  95% bootstrap interval over test SUBJECTS")
    L.append("% The two are different questions and are never pooled into one")
    L.append("% interval. \\XSD is exactly 0 for GeomMLP/PointNet/DGCNN — those")
    L.append("% encoders are bit-identical across repeat runs and are evaluated")
    L.append("% once, so 0 is a fact about the estimator, not a measurement.")
    L.append("%")
    L.append("% NOT MEASURED HERE: training-seed variance. Nothing was retrained;")
    L.append("% every interval is inference variance under frozen weights. A model")
    L.append("% retrained from a different init could land outside these bounds.")
    L.append("%")
    L.append("% Each macro carries the source row it resolves to as a comment.")
    L.append("% " + "=" * 74)
    L.append("")

    missing, n_emitted = [], 0

    for title, specs, note in SECTIONS:
        L.append("% " + "-" * 74)
        L.append(f"% {title}")
        for ln in (note.split("\n") if note else []):
            L.append(f"%   {ln}")
        L.append("% " + "-" * 74)
        for sp in specs:
            if sp["kind"] == "summary":
                r = summ[(summ.task == sp["task"]) & (summ.encoder == sp["encoder"]) &
                         (summ["head"] == sp["head"]) & (summ.level == sp["level"]) &
                         (summ.metric == sp["metric"])]
                if len(r) != 1:
                    missing.append(f"{sp['macro']}: {len(r)} rows matched in "
                                   f"summary_ci.csv for {sp['task']}/{sp['encoder']}/"
                                   f"{sp['head']}/{sp['level']}/{sp['metric']}")
                    continue
                r = r.iloc[0]
                # r['head'] not r.head — Series.head is a method, and the
                # attribute form silently formats the bound method instead.
                src = (f"summary_ci.csv | task={r.task} encoder={r.encoder} "
                       f"head={r['head']} level={r.level} metric={r.metric} "
                       f"n_seeds={r.n_seeds} n_units={r.n_units} "
                       f"n_subjects={r.n_subjects}")
                emit(L, sp["macro"], r["mean"], r.seed_sd, r.boot_ci_lo,
                     r.boot_ci_hi, sp["fmt"], src, bool(r.seed_deterministic))
            else:
                r = con[(con.contrast == sp["contrast"]) & (con.task == sp["task"]) &
                        (con.metric == sp["metric"])]
                if len(r) != 1:
                    missing.append(f"{sp['macro']}: {len(r)} rows matched in "
                                   f"contrast_tests.csv for {sp['contrast']}/"
                                   f"{sp['task']}/{sp['metric']}")
                    continue
                r = r.iloc[0]
                src = (f"contrast_tests.csv | contrast={r.contrast} task={r.task} "
                       f"metric={r.metric} | {r.arm_a} MINUS {r.arm_b} | "
                       f"verdict={r.verdict}")
                emit(L, sp["macro"], r.diff_mean, r.diff_seed_sd, r.boot_ci_lo,
                     r.boot_ci_hi, sp["fmt"], src, False)
            n_emitted += 1

    # ── derived: AUC minus top-1, no interval ──
    L.append("% " + "-" * 74)
    L.append("% The AUC-minus-top-1 gap — DERIVED, POINT ESTIMATE ONLY")
    L.append("%   macro_auc minus accuracy off the same summary row. The")
    L.append("%   difference was never bootstrapped as a unit, so there is no")
    L.append("%   honest interval to attach and none is emitted. numbers.tex was")
    L.append("%   ambiguous about which head \\gapDxSubj came from, so both the")
    L.append("%   softmax and prototype variants are given.")
    L.append("% " + "-" * 74)
    for macro, task, enc, head, level in DERIVED_GAPS:
        sel = ((summ.task == task) & (summ.encoder == enc) &
               (summ["head"] == head) & (summ.level == level))
        acc = summ[sel & (summ.metric == "accuracy")]
        auc = summ[sel & (summ.metric == "macro_auc")]
        if len(acc) != 1 or len(auc) != 1:
            missing.append(f"{macro}: could not resolve acc/auc pair for "
                           f"{task}/{enc}/{head}/{level}")
            continue
        gap = float(auc.iloc[0]["mean"]) - float(acc.iloc[0]["mean"])
        L.append(f"% summary_ci.csv | {task}/{enc}/{head}/{level} "
                 f"macro_auc {auc.iloc[0]['mean']:.4f} - accuracy "
                 f"{acc.iloc[0]['mean']:.4f}")
        L.append(f"\\newcommand{{\\{macro}}}{{{gap:.2f}}}")
        L.append("")
        n_emitted += 1

    # ── inference wobble ──
    t, e, h, lv, m = FPS_ROW
    r = summ[(summ.task == t) & (summ.encoder == e) & (summ["head"] == h) &
             (summ.level == lv) & (summ.metric == m)]
    L.append("% " + "-" * 74)
    L.append("% Inference non-determinism — REPLACES \\fpsWobble/\\fpsRunA/\\fpsRunB")
    L.append("%   Those three came from two ad-hoc unseeded runs. These come from")
    L.append("%   the seeded 5-seed sweep on the same task/head/level.")
    L.append("% " + "-" * 74)
    if len(r) == 1:
        r = r.iloc[0]
        L.append(f"% summary_ci.csv | {t}/{e}/{h}/{lv}/{m} over {r.n_seeds} "
                 f"inference seeds")
        L.append(f"\\newcommand{{\\fpsSeedSD}}{{{r.seed_sd * 100:.2f}}}"
                 f"  % pp, inference-seed SD")
        L.append(f"\\newcommand{{\\fpsSeedLo}}{{{r.seed_min:.3f}}}")
        L.append(f"\\newcommand{{\\fpsSeedHi}}{{{r.seed_max:.3f}}}")
        L.append(f"\\newcommand{{\\fpsSeedSpan}}{{"
                 f"{(r.seed_max - r.seed_min) * 100:.1f}}}  % pp, observed range")
        n_emitted += 4
    else:
        missing.append(f"fps wobble row: {len(r)} rows matched for {FPS_ROW}")
    L.append("")

    # max inference-seed SD anywhere in the sweep — the honest headline number
    pn2 = summ[(summ.encoder == "pointnet2") &
               (summ.metric.isin(["accuracy", "macro_f1", "macro_auc"]))]
    if len(pn2):
        worst = pn2.loc[pn2.seed_sd.idxmax()]
        L.append(f"% summary_ci.csv | worst-case row across the whole sweep: "
                 f"{worst.task}/{worst.encoder}/{worst['head']}/{worst.level}/"
                 f"{worst.metric}")
        L.append(f"\\newcommand{{\\fpsMaxSeedSD}}{{{worst.seed_sd * 100:.2f}}}"
                 f"  % pp, largest inference-seed SD observed")
        L.append("")
        n_emitted += 1

    out = Path(a.out)
    out.write_text("\n".join(L) + "\n")

    print(f"[emit] {n_emitted} quantities -> {out}")
    print(f"[emit] {sum(1 for x in L if x.startswith(chr(92) + 'newcommand'))} "
          f"\\newcommand lines")
    if missing:
        print(f"\n[emit] {len(missing)} UNRESOLVED — these are NOT in the block:")
        for x in missing:
            print(f"  ! {x}")
        sys.exit(1)


if __name__ == "__main__":
    main()
