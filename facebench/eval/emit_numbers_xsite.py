#!/usr/bin/env python3
"""
PART A — emit numbers_from_xsite.tex, the macro block for the cross-site result.
thread facebase3d-paper-2026-08-08.

THIS SCRIPT DOES NOT WRITE INTO THE MANUSCRIPT. It emits the macro block. It also
never writes numbers.tex or numbers_reruns.tex.

Same three-macro convention as numbers_from_reruns.tex, so the prose can choose:
    \\X     seed-averaged point estimate
    \\XSD   inference-seed SD        — FPS draw, weights frozen (0 for the three
                                      deterministic encoders, which run once)
    \\XCI   95% bootstrap interval   — over held-out-site TEST SUBJECTS
The two uncertainty sources are never pooled.

WHAT THESE MACROS REPLACE. numbers.tex carries \\xsiteAucLo / \\xsiteAucHi
(0.48 / 0.51) behind a \\prov marker. I deliberately did NOT emit replacements
for those names on 08-07 because there was no cross-site experiment behind them.
There is one now, but the names are NOT reused: the old pair described a
train-on-one-site design that was never run, and rebinding them would silently
convert a retracted claim into a measured one. New names, and the group split is
carried in the name so no macro can be quoted without its fold type.

GROUPS ARE NEVER POOLED. \\xsGone* covers PH/FC/PR, where both FB-56 and FB-5A
remain in training so the fold isolates SITE from DATASET. \\xsGtwo* covers
GW/PT/LC, which are single-dataset and confound the two. There is deliberately no
macro spanning both.

PREVALENCE. Held-out sites are 72-92% Unaffected and prevalence differs per site,
so accuracy is not comparable across folds. Each fold's majority-class rate is
emitted as \\xs<Site>Maj so no accuracy macro can be quoted without the baseline
it has to beat. macro-AUC is the headline metric (chance 0.50 everywhere).

Usage: python3 emit_numbers_xsite.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

BINARY = "--binary" in sys.argv
# --fix reads the 2026-09-07 retraining of the SAME six folds with early
# stopping disabled (thread facebase3d-xsitediag-2026-09-07). It emits into its
# OWN namespace and rebinds nothing: the published \xs* / \xsb* macros stay
# bound to the patience-20 runs so the manuscript can quote both arms and the
# stopping-rule claim stays checkable from either side. Same principle as
# \ofcBinAuc vs \tsOfcBinAuc vs \tsOfcBinAucES in the trainseed block.
FIX = "--fix" in sys.argv
_stem = ("xsitefix" if FIX else "xsite") + ("_bin" if BINARY else "")
SUMMARY = C.OUT / f"summary_ci_{_stem}.csv"
AUDIT = C.OUT / "xsite_fold_audit.csv"
OUT_TEX = C.OUT / f"numbers_from_{_stem}.tex"
# Task-name prefix in the summary CSV. Stripped EXACTLY, not by substring
# removal: "XS_fix_PH".replace("XS_","") would leave "fix_PH" and silently
# produce macros no fold name matches.
TASK_PREFIX = ("XSB_" if BINARY else "XS_") + ("fix_" if FIX else "")

ENC_TOKEN = {"geommlp": "Gmlp", "pointnet": "Pnet",
             "dgcnn": "Dgcnn", "pointnet2": "Ptwo"}
METRIC_TOKEN = {"accuracy": "Acc", "macro_f1": "Fone", "macro_auc": "Auc"}
HEADLINE_ENC = "pointnet2"
# Distinct namespace so the binary block can be \input alongside the four-class
# one without either silently redefining the other's macros.
PREFIX = ("xsb" if BINARY else "xs") + ("f" if FIX else "")


def f_prob(x):
    return f"{x:.3f}"


def f_sd(x):
    return f"{x:.4f}"


def emit(L, macro, r, src):
    L.append(f"% {src}")
    L.append(f"\\newcommand{{\\{macro}}}{{{f_prob(r['mean'])}}}")
    if bool(r.seed_deterministic):
        L.append(f"\\newcommand{{\\{macro}SD}}{{0}}"
                 f"  % exact 0: deterministic encoder, evaluated once")
    else:
        L.append(f"\\newcommand{{\\{macro}SD}}{{{f_sd(r.seed_sd)}}}"
                 f"  % inference-seed SD, weights frozen")
    L.append(f"\\newcommand{{\\{macro}CI}}"
             f"{{[{f_prob(r.boot_ci_lo)}, {f_prob(r.boot_ci_hi)}]}}"
             f"  % 95% bootstrap over held-out-site test subjects")
    L.append("")


def main():
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY} — run analyze_ci.py --suffix _xsite")
    summ = pd.read_csv(SUMMARY)
    summ = summ[summ.task.str.startswith(TASK_PREFIX)].copy()
    summ["site"] = summ.task.str.slice(len(TASK_PREFIX))
    unknown = sorted(set(summ.site) - set(C.XSITE_SITES))
    if unknown:
        raise SystemExit(f"unrecognised fold names after stripping "
                         f"{TASK_PREFIX!r}: {unknown}")
    audit = pd.read_csv(AUDIT).set_index("site")
    tcols = [c for c in audit.columns if c.startswith("test_")]

    L = []
    L.append("% " + "=" * 74)
    L.append(f"% {OUT_TEX.name} — MACHINE-GENERATED. Do not hand-edit.")
    L.append("%")
    L.append("%   generator : facebench/eval/emit_numbers_xsite.py")
    L.append(f"%   source    : {SUMMARY.relative_to(C.PROJECT)}  ({len(summ)} rows)")
    L.append("%")
    L.append("% Leave-one-site-out cross-site transfer. TEST is one held-out")
    L.append("% FaceBase acquisition site in full; train/val are a stratified")
    L.append("% split over the remaining ten sites. The five sites failing the")
    L.append("% post-attrition per-class viability gate (CL, CO, NG, SF, TX) stay")
    L.append("% in the TRAINING pool and are never scored as folds.")
    L.append("%")
    L.append("% THERE WAS NO PRIOR RESULT HERE. The 'AUC 0.48-0.51' figures in")
    L.append("% earlier drafts came from the A1_S1/S2/S3 runs, which are")
    L.append("% class-imbalance variants on the full-cohort split, not site")
    L.append("% splits. \\xsiteAucLo / \\xsiteAucHi are NOT rebound by this block:")
    L.append("% new measurement, new names.")
    L.append("%")
    L.append("% THE TWO FOLD GROUPS ARE NEVER POOLED:")
    L.append("%   \\xsGone*  PH, FC, PR — both datasets remain in training, so the")
    L.append("%             fold isolates SITE from DATASET")
    L.append("%   \\xsGtwo*  GW, PT, LC — single-dataset; site and dataset move")
    L.append("%             together and are measured jointly")
    L.append("% No macro here spans both groups.")
    L.append("%")
    L.append("% READ ACCURACY AGAINST \\xs<Site>Maj. Held-out sites are 72-92%")
    L.append("% Unaffected and prevalence differs per site, so accuracy is not")
    L.append("% comparable across folds. macro-AUC is the headline metric;")
    L.append("% its chance level is 0.50 regardless of prevalence.")
    L.append("%")
    L.append("% \\X point estimate / \\XSD inference-seed SD / \\XCI bootstrap over")
    L.append("% test SUBJECTS. Never pooled.")
    L.append("%")
    L.append("% NOT MEASURED: training-seed variance. Each cell is ONE training")
    L.append("% run. Every interval is inference variance under frozen weights.")
    if FIX:
        L.append("%")
        L.append("% THIS IS THE CORRECTED ARM (2026-09-07). Same six folds, same")
        L.append("% manifests, same encoders, same label collapse. TWO protocol")
        L.append(f"% changes relative to the \\{PREFIX[:-1]}* block:")
        L.append("%   1. early stopping disabled (patience == epochs == 200).")
        L.append("%      The published folds stopped at 21-46 epochs, inside the")
        L.append("%      ln(K) plateau, so they measured the stopping rule as")
        L.append("%      much as site transfer.")
        L.append("%   2. training is SEEDED (--train-seed 1); the published runs")
        L.append("%      were not, and seeding also gives each dataloader worker")
        L.append("%      its own augmentation stream.")
        L.append("% Both arms are therefore ONE draw each. The difference between")
        L.append("% them is NOT separated into stopping rule vs retraining")
        L.append("% variance, and no macro here should be quoted as if it were.")
        L.append("% What licenses reading it as the stopping rule is the pattern,")
        L.append("% not any single cell: the gain is consistent across folds for")
        L.append("% the encoders that plateau, and absent for GeomMLP, which has")
        L.append("% no plateau and is the control.")
    L.append("% " + "=" * 74)
    L.append("")

    missing = []

    # ── per-fold descriptive macros ─────────────────────────────────────────
    L.append("% " + "-" * 74)
    L.append("% Fold sizes and majority-class baselines (post-attrition)")
    L.append("% " + "-" * 74)
    for site in C.XSITE_SITES:
        row = audit.loc[site, tcols]
        maj = float(row.max() / row.sum())
        grp = "1 (site isolated)" if site in C.XSITE_BOTH_DATASETS \
            else "2 (site+dataset)"
        L.append(f"% site {site} | group {grp} | datasets "
                 f"{audit.loc[site, 'datasets']}")
        L.append(f"\\newcommand{{\\{PREFIX}{site}Ntest}}{{{int(audit.loc[site, 'n_test'])}}}")
        L.append(f"\\newcommand{{\\{PREFIX}{site}Ntrain}}"
                 f"{{{int(audit.loc[site, 'n_train'])}}}")
        L.append(f"\\newcommand{{\\{PREFIX}{site}Maj}}{{{f_prob(maj)}}}"
                 f"  % majority-class rate = accuracy of a constant predictor")
        L.append("")

    # ── per fold x encoder x metric ─────────────────────────────────────────
    for group, sites, label in (
            ("Gone", C.XSITE_BOTH_DATASETS,
             "GROUP 1 — site isolated from dataset (both FB-56 and FB-5A "
             "remain in training)"),
            ("Gtwo", C.XSITE_SINGLE_DATASET,
             "GROUP 2 — site and dataset confounded (single-dataset sites)")):
        L.append("% " + "-" * 74)
        L.append(f"% {label}")
        L.append("% " + "-" * 74)
        for site in sites:
            for enc in C.ENCODERS:
                for metric in ("macro_auc", "macro_f1", "accuracy"):
                    r = summ[(summ.site == site) & (summ.encoder == enc) &
                             (summ.metric == metric) & (summ.level == "scan")]
                    macro = f"{PREFIX}{site}{ENC_TOKEN[enc]}{METRIC_TOKEN[metric]}"
                    if len(r) != 1:
                        missing.append(f"{macro}: {len(r)} rows matched "
                                       f"({site}/{enc}/{metric}/scan)")
                        continue
                    r = r.iloc[0]
                    src = (f"{SUMMARY.name} | task={r.task} "
                           f"encoder={r.encoder} head={r['head']} "
                           f"level={r.level} metric={r.metric} "
                           f"n_seeds={r.n_seeds} n_units={r.n_units} "
                           f"n_subjects={r.n_subjects}")
                    emit(L, macro, r, src)

    # ── group-level ranges, headline encoder, headline metric ──────────────
    L.append("% " + "-" * 74)
    L.append("% Group-level macro-AUC ranges over folds, PointNet++ only.")
    L.append("% Ranges, not means: three folds is too few for an average to mean")
    L.append("% anything, and averaging would also hide a fold that behaves")
    L.append("% differently from the other two.")
    L.append("% " + "-" * 74)
    h = summ[(summ.encoder == HEADLINE_ENC) & (summ.metric == "macro_auc") &
             (summ.level == "scan")]
    for group, sites in (("Gone", C.XSITE_BOTH_DATASETS),
                         ("Gtwo", C.XSITE_SINGLE_DATASET)):
        g = h[h.site.isin(sites)]
        if g.empty:
            missing.append(f"xs{group}Auc*: no rows for {sites}")
            continue
        L.append(f"% folds {', '.join(sorted(g.site))} | encoder "
                 f"{HEADLINE_ENC} | metric macro_auc | level scan")
        L.append(f"\\newcommand{{\\{PREFIX}{group}AucLo}}{{{f_prob(g['mean'].min())}}}")
        L.append(f"\\newcommand{{\\{PREFIX}{group}AucHi}}{{{f_prob(g['mean'].max())}}}")
        L.append(f"\\newcommand{{\\{PREFIX}{group}Nfolds}}{{{len(g)}}}")
        # How many folds clear chance outright, by their own bootstrap interval.
        L.append(f"\\newcommand{{\\{PREFIX}{group}NaboveChance}}"
                 f"{{{int((g.boot_ci_lo > 0.5).sum())}}}"
                 f"  % folds whose 95% CI lies entirely above 0.50")
        L.append("")

    # ── census over ALL fold x encoder cells ───────────────────────────────
    # The group ranges above are PointNet++ only. These count every cell, so
    # the prose can say how much of the grid clears chance without implying
    # the headline encoder is representative — it is not: PointNet++ is the
    # WORST transferring encoder here and GeomMLP the best, which inverts the
    # within-site ordering and is worth stating rather than averaging away.
    L.append("% " + "-" * 74)
    L.append("% Census over all fold x encoder cells, by their own bootstrap CI.")
    L.append("% " + "-" * 74)
    allc = summ[(summ.metric == "macro_auc") & (summ.level == "scan")]
    above = int((allc.boot_ci_lo > 0.5).sum())
    below = int((allc.boot_ci_hi < 0.5).sum())
    L.append(f"\\newcommand{{\\{PREFIX}Ncells}}{{{len(allc)}}}")
    L.append(f"\\newcommand{{\\{PREFIX}NcellsAbove}}{{{above}}}"
             f"  % CI entirely above chance")
    L.append(f"\\newcommand{{\\{PREFIX}NcellsSpan}}{{{len(allc) - above - below}}}"
             f"  % CI contains chance")
    L.append(f"\\newcommand{{\\{PREFIX}NcellsBelow}}{{{below}}}"
             f"  % CI entirely below chance")
    for enc in C.ENCODERS:
        e = allc[allc.encoder == enc]
        L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}NaboveChance}}"
                 f"{{{int((e.boot_ci_lo > 0.5).sum())}}}"
                 f"  % of {len(e)} folds, {enc}")
        L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}AucLo}}"
                 f"{{{f_prob(e['mean'].min())}}}")
        L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}AucHi}}"
                 f"{{{f_prob(e['mean'].max())}}}")
    L.append("")

    # ── attenuation against the within-site reference (FIX arm only) ───────
    # The published arm's equivalents are hand-maintained in numbers.tex with
    # their arithmetic spelled out in comments. They are NOT regenerated here:
    # that block belongs to the patience-20 measurement and rewriting it from
    # corrected folds would silently restate an old claim with new numbers.
    #
    # WITHIN-SITE REFERENCE. A2_fix for the binary panel, A1_fix for the
    # four-class one — the corrected arm on both sides. Measuring a corrected
    # fold against a truncated within-site value is the one arithmetic error
    # this whole block exists to avoid: on four-class it would turn a 12 pp
    # protocol artefact into apparent cross-site transfer.
    #
    # ASYMMETRY, STATED NOT HIDDEN: the within-site value is a mean over five
    # retrainings; each fold is ONE run. The delta therefore has retraining
    # variance on one side only.
    if FIX:
        ts = pd.read_csv(C.OUT / "trainseed_summary.csv")
        within_task = "A2_fix" if BINARY else "A1_fix"
        L.append("% " + "-" * 74)
        L.append(f"% Attenuation vs within-site ({within_task}, 5-retraining "
                 f"mean, scan level, softmax head).")
        L.append("% A fold MEAN is used only for this delta and never as a "
                 "headline: the")
        L.append("% folds differ in size, prevalence and whether dataset moves "
                 "with site,")
        L.append("% so they are not exchangeable and the prose reports ranges.")
        L.append("% " + "-" * 74)
        for enc in C.ENCODERS:
            w = ts[(ts.base_task == within_task) & (ts.encoder == enc)
                   & (ts["head"] == "softmax") & (ts.level == "scan")
                   & (ts.metric == "macro_auc")]
            if len(w) != 1:
                missing.append(f"{PREFIX}{ENC_TOKEN[enc]}Atten: {len(w)} "
                               f"within-site rows for {within_task}/{enc}")
                continue
            wv = float(w["train_seed_mean"].iloc[0])
            fm = float(allc[allc.encoder == enc]["mean"].mean())
            L.append(f"% {enc}: within-site {wv:.3f} -> fold mean {fm:.3f}")
            L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}Within}}"
                     f"{{{f_prob(wv)}}}")
            L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}FoldMean}}"
                     f"{{{f_prob(fm)}}}")
            L.append(f"\\newcommand{{\\{PREFIX}{ENC_TOKEN[enc]}Atten}}"
                     f"{{{100 * (wv - fm):.1f}}}"
                     f"  % percentage points, within-site minus fold mean")
        L.append("")
        # Cells above chance split by fold group, ALL FOUR encoders — the
        # census above is per-encoder over all six folds, which cannot answer
        # "does the site/dataset confound matter", the question the split asks.
        for group, sites in (("Gone", C.XSITE_BOTH_DATASETS),
                             ("Gtwo", C.XSITE_SINGLE_DATASET)):
            g = allc[allc.site.isin(sites)]
            L.append(f"\\newcommand{{\\{PREFIX}{group}CellsAbove}}"
                     f"{{{int((g.boot_ci_lo > 0.5).sum())}}}"
                     f"  % of {len(g)} cells, folds {'/'.join(sites)}")
            L.append(f"\\newcommand{{\\{PREFIX}{group}Ncells}}{{{len(g)}}}")
        L.append("")

    L.append(f"\\newcommand{{\\{PREFIX}Nfolds}}{{{len(C.XSITE_SITES)}}}")
    L.append(f"\\newcommand{{\\{PREFIX}Nencoders}}{{{len(C.ENCODERS)}}}")
    L.append(f"\\newcommand{{\\{PREFIX}Nruns}}"
             f"{{{len(C.XSITE_SITES) * len(C.ENCODERS)}}}"
             f"  % fresh training runs behind this block")
    L.append("")

    if missing:
        print("MISSING ROWS — nothing emitted for these, and that is a hard "
              "error rather than a silent omission:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    OUT_TEX.write_text("\n".join(L) + "\n")
    n_macros = sum(1 for x in L if x.startswith("\\newcommand"))
    print(f"wrote {OUT_TEX}")
    print(f"  {n_macros} \\newcommand lines over {len(summ)} summary rows")


if __name__ == "__main__":
    main()
