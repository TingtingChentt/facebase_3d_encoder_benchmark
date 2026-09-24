#!/usr/bin/env python3
"""
Figures 4 and 5 — two ways to be fooled: split level, and acquisition site.
thread facebase3d-paper-2026-08-08; SPLIT IN TWO 2026-09-14, thread
facebase3d-figsplit-2026-09-14.

THIS SCRIPT EMITS TWO FIGURES. It was one three-panel figure until 2026-09-14,
the reason for the split: the leakage panel and the cross-site panels are
different results supporting different sections (2.7 and 2.8), sharing only the
rhetorical frame "ways to be fooled". They now come out as

    fig4_leakage.png   the old panel (a), alone, authored at SINGLE
    fig5_xsite.png     the old panels (b) and (c), now (a) and (b), at DOUBLE

WHAT THE SPLIT MUST NOT DO is separate the two CROSS-SITE panels from each
other. They are one figure on a shared x and a shared y for the reason given
below, and the split boundary is therefore (a) | (b,c), not (a) | (b) | (c).

FIGURE 4 — SPLIT LEVEL. BOTH ARMS ARE FIVE-RETRAINING MEANS (2026-09-04, thread
facebase3d-figs-2026-09-04), so they are drawn alike. This panel has now had the
same asymmetry removed from it twice. Before 2026-08-18 the scan-level arm was a
single unseeded number from a pre-correction log, drawn hatched so no reader
would take it as equally evidenced; the 2026-08-31 training-seed sweep then made
the subject-disjoint arm a five-model mean while the scan-level arm was still one
checkpoint, putting the asymmetry back in a subtler form. The leaked arm is now
retrained at the same five training seeds (B{1,2}_scanlevel_ts{1..5}, trained on
the frozen manifests under data/processed/legacy_scanlevel/) and scored by
rerun_scanlevel.py --train-seed under the corrected arm's own protocol.

Bars are means over the five retrainings and the dots are the five individual
models, matching Fig. 2's convention. As of 2026-09-04 the subject-level
bootstrap interval is not drawn; it is still computed and still lives in the CSVs
this panel reads. Both arms are read from CSV rather than typed in.

WHAT STILL MAY NOT BE DONE: the two arms have DIFFERENT TEST SETS — that is the
whole point of the comparison — so nothing here is paired and no interval belongs
on the difference between them. The delta annotation is a difference of means and
is labelled as such. The dots are what carries the strength of the result without
pairing: on all three quantities the five scan-level models and the five
subject-disjoint models do not overlap at all.

FIGURE 5, PANELS (a) and (b) — CROSS-SITE. Every fold x encoder cell with its
bootstrap interval over held-out-site subjects, split into the two fold groups
and NEVER pooled. (a) is four-class subtyping, (b) is binary screening — THE
SAME SIX FOLDS, the same manifest, seeds and encoders, differing only in the
label collapse.

WHY THEY ARE STACKED ON A SHARED X AND A SHARED Y. The paper's claim is a
contrast between the two tasks, so the two panels have to be readable as one
comparison: identical fold columns above each other, and one y scale, so the
vertical offset between a fold's four-class cell and its binary cell IS the
result. Giving each panel its own y range would rescale that offset away and
make a collapse to chance look like a modest dip.

BOTH CROSS-SITE PANELS ARE THE CORRECTED ARM (2026-09-07, thread
facebase3d-xsitediag-2026-09-07). The published folds were trained with
patience 20 and stopped at 21-46 epochs, inside the same ln(K) plateau that
truncated the within-site cleft tasks, so the earlier version of the four-class
panel was in part a picture of the stopping rule: PointNet++ moved from a fold mean of 0.500
to 0.621 and from 0/6 to 5/6 folds whose interval clears chance. Both arms were
retrained, not just the four-class one, because the claim is a CONTRAST between
the two label collapses on identical folds and correcting one alone would make
it a protocol comparison.

Each panel carries its OWN within-site reference rule (PointNet++, same task),
because the two tasks start from very different within-site values and that
difference is what stops the four-class result from being read as evidence
about screening. Those rules now read the A1_fix / A2_fix five-retraining means
(0.622 and 0.843), NOT the published single runs (0.573, 0.849): a corrected
fold measured against a truncated within-site reference would show cross-site
beating within-site on four-class, which is an artefact of mixing stopping
rules and not a finding.

WHAT THIS FIGURE STILL CANNOT SEPARATE. Each fold is one training run in both
arms, and the corrected runs are seeded while the published ones were not. The
difference between the arms therefore confounds the stopping rule with
retraining variance, which on the within-site cleft tasks ran 1-10 pp of AUC by
encoder. What licenses reading the four-class panel as the stopping rule is the pattern
rather than any one cell: PointNet++ gains on 6/6 folds and DGCNN on 4/6, while
GeomMLP — the one representation with no plateau, and so the control — moves
-0.8 pp on average.

Chance is at 0.50 for macro-AUC. Cells whose interval clears it are ringed.

Usage: python3 fig4_fooled.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
import figstyle as FS                                       # noqa: E402
import rerun_config as C                                    # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "figures"

# Both arms come from CSV, and both are now five-retraining sweeps keyed on
# base_task. BEFORE was three hardcoded floats out of a pre-correction log until
# 2026-08-18, then one re-scored legacy checkpoint, and is now five retrainings.
BEFORE_KEY = {"19-class\naccuracy": ("B1_scanlevel", "accuracy"),
              "33-class\naccuracy": ("B2_scanlevel", "accuracy"),
              "33-class\nmacro-$F_1$": ("B2_scanlevel", "macro_f1")}
AFTER_KEY = {"19-class\naccuracy": ("B1_corrected", "accuracy"),
             "33-class\naccuracy": ("B2_corrected", "accuracy"),
             "33-class\nmacro-$F_1$": ("B2_corrected", "macro_f1")}


def panel_leak(ax, summ, scan_summ, per, scan_per,
               title="Effect of scan- versus\nsubject-level splitting"):
    def get(frame, task, metric):
        r = frame[(frame.base_task == task) & (frame.encoder == "pointnet2")
                  & (frame["head"] == "softmax") & (frame.level == "scan")
                  & (frame.metric == metric)]
        if r.empty:
            raise SystemExit(f"fig4: no row for {task}/{metric} — have "
                             f"aggregate_trainseed.py and "
                             f"aggregate_scanlevel_trainseed.py been run?")
        return r.iloc[0]

    def runs(frame, task, metric):
        """The five individual retrainings. Anything other than five means the
        sweep is incomplete and the bar above it is not a five-model mean, so
        this refuses rather than drawing a mean over whatever landed."""
        r = frame[(frame.base_task == task) & (frame.encoder == "pointnet2")
                  & (frame["head"] == "softmax") & (frame.level == "scan")
                  & (frame.metric == metric)]
        if len(r) != 5:
            raise SystemExit(f"fig4: {len(r)} training seeds for {task}/"
                             f"{metric}, expected 5")
        return r.value.to_numpy()

    keys = list(BEFORE_KEY)
    xs = np.arange(len(keys), dtype=float)
    w = 0.34
    rng = np.random.default_rng(0)   # jitter only; nothing downstream reads it
    for i, k in enumerate(keys):
        b = get(scan_summ, *BEFORE_KEY[k])
        a = get(summ, *AFTER_KEY[k])
        bv = runs(scan_per, *BEFORE_KEY[k])
        av = runs(per, *AFTER_KEY[k])
        for off, r, v, col in ((-w / 2, b, bv, FS.ACCENT),
                               (+w / 2, a, av, FS.ENC_COLOR["pointnet2"])):
            ax.bar(xs[i] + off, r.train_seed_mean, w, color=col, zorder=3,
                   edgecolor="white", linewidth=0.3)
            jit = rng.uniform(-w * 0.20, w * 0.20, size=len(v))
            ax.scatter(xs[i] + off + jit, v, s=1.6, zorder=6,
                       color=FS.INK, linewidths=0, alpha=0.85)
        ax.annotate(f"$-${100 * (b.train_seed_mean - a.train_seed_mean):.0f} pp",
                    xy=(xs[i], max(bv.max(), av.max()) + 0.030),
                    ha="center", fontsize=6.4, color=FS.MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels(keys)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("score")
    # Wrapped on purpose: at axes.titlesize 8.5 this string is ~3.2 in wide,
    # which is the ENTIRE authored width of the standalone leakage figure, so on
    # one line it runs to the canvas edge. It was wrapped for the same reason
    # when this was panel (a) of the combined figure (~2.1 in of axes then).
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, facecolor=FS.ACCENT,
                       label="scan-level split"),
         plt.Rectangle((0, 0), 1, 1, facecolor=FS.ENC_COLOR["pointnet2"],
                       label="subject-level split")]
    ax.legend(handles=h, frameon=False, loc="upper left", fontsize=6.2,
              handlelength=1.1, borderaxespad=0.2, labelspacing=0.4,
              title="mean of 5 retrainings", title_fontsize=6.2)


def panel_xsite(ax, xs_summ, within, prefix, title, show_group_labels,
                show_fold_labels, within_label, ylim):
    a = xs_summ[(xs_summ.metric == "macro_auc") & (xs_summ.level == "scan")].copy()
    a["site"] = a.task.str.replace(prefix, "", regex=False)
    groups = [(C.XSITE_BOTH_DATASETS, "site varies,\ndataset fixed"),
              (C.XSITE_SINGLE_DATASET, "site and dataset\nvary together")]
    xpos, ticks, labels, bounds = 0.0, [], [], []
    for gi, (sites, glab) in enumerate(groups):
        start = xpos
        for site in sites:
            for enc in FS.ENCODERS:
                r = a[(a.site == site) & (a.encoder == enc)]
                if r.empty:
                    continue
                r = r.iloc[0]
                clears = r.boot_ci_lo > 0.5
                ax.plot([xpos, xpos], [r.boot_ci_lo, r.boot_ci_hi],
                        color=FS.ENC_COLOR[enc], lw=1.0, zorder=3,
                        solid_capstyle="round")
                ax.plot(xpos, r["mean"], "o", ms=3.2,
                        color=FS.ENC_COLOR[enc], zorder=4,
                        markeredgecolor=FS.ACCENT if clears else "white",
                        markeredgewidth=0.9 if clears else 0.3)
                xpos += 1
            ticks.append(xpos - 2.5)
            labels.append(site)
            xpos += 0.9
        bounds.append((start - 0.6, xpos - 1.5, glab))
        xpos += 1.4

    ax.axhline(0.5, color=FS.ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=2)
    # Both reference labels live in a left margin strip that holds no data,
    # so neither can land on a fold's interval.
    # BELOW the chance rule, not above. In (b) the within-site rule sits only
    # 0.07 higher and the two labels collided when both sat above their lines.
    ax.annotate("chance", xy=(-2.45, 0.5), xytext=(0, -2.5),
                textcoords="offset points", fontsize=6.2,
                color=FS.ACCENT, va="top", ha="left")
    ax.axhline(within, color=FS.INK, lw=0.7, ls=(0, (1, 1.6)), zorder=2)
    ax.annotate(within_label, xy=(-2.45, within), fontsize=6.2,
                color=FS.INK, va="center", ha="left")
    if show_group_labels:
        for lo, hi, glab in bounds:
            ax.annotate(glab, xy=((lo + hi) / 2, -0.30),
                        xycoords=("data", "axes fraction"), ha="center",
                        fontsize=6.8, color=FS.MUTED)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels if show_fold_labels else [""] * len(labels),
                       fontsize=6.8)
    ax.set_xlim(-2.6, xpos - 2.2)
    # ONE y range across (b) and (c). See the module docstring: the offset
    # between a fold's two cells is the result, so it must not be rescaled
    # per panel. Computed in main() from BOTH panels' intervals rather than
    # written in: the literal that used to sit here (0.33, 0.90) was set by the
    # patience-20 arm and silently clipped the corrected arm's PR/PointNet++
    # binary interval, which now reaches 0.909.
    ax.set_ylim(*ylim)
    ax.set_ylabel("macro AUC\n(held-out site)")
    ax.set_title(title, pad=4)
    ax.grid(axis="y", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def main():
    FS.apply()
    # summary_ci.csv (single unseeded checkpoints) is no longer read by this
    # figure at all: panel (a) moved to the training-seed sweeps on 2026-09-04
    # and the within-site reference rules in (b)/(c) followed on 2026-09-08.
    ts_summ = pd.read_csv(C.OUT / "trainseed_summary.csv")
    ts_per = pd.read_csv(C.OUT / "trainseed_per_run.csv")
    scan_summ = pd.read_csv(C.OUT / "scanlevel_trainseed_summary.csv")
    scan_per = pd.read_csv(C.OUT / "scanlevel_trainseed_per_run.csv")
    # CORRECTED ARM (2026-09-07, thread facebase3d-xsitediag-2026-09-07). The
    # published folds were trained with patience 20 and stopped at 21-46 epochs,
    # inside the ln(K) plateau, so (b) was in part a picture of the stopping
    # rule. Both arms are redrawn from the retrained folds; the patience-20
    # numbers remain in summary_ci_xsite{,_bin}.csv and in the \xs*/\xsb*
    # macros, and Sec 2.7 quotes them as the truncated comparison.
    xs4 = pd.read_csv(C.OUT / "summary_ci_xsitefix.csv")
    xsb = pd.read_csv(C.OUT / "summary_ci_xsitefix_bin.csv")

    def within_site(task):
        """Within-site reference rule for a panel.

        Reads the TRAINING-SEED sweep, not summary_ci.csv, and reads the _fix
        arm. The rule has to be measured under the same stopping rule as the
        cells it is a reference for -- otherwise (b) would compare corrected
        cross-site folds against a truncated within-site value (A1: 0.499) and
        turn a 12 pp protocol artefact into apparent transfer. It is a
        five-retraining mean where the folds are single runs; that asymmetry is
        stated in the caption rather than hidden by using a single run here.
        """
        r = ts_summ[(ts_summ.base_task == task)
                    & (ts_summ.encoder == "pointnet2")
                    & (ts_summ["head"] == "softmax")
                    & (ts_summ.metric == "macro_auc")
                    & (ts_summ.level == "scan")]
        if len(r) != 1:
            raise SystemExit(f"fig4: {len(r)} within-site rows for {task}")
        return float(r["train_seed_mean"].iloc[0])

    def shared_ylim(*frames):
        """One y range over BOTH cross-site panels, from the drawn intervals.

        Hardcoding this clips silently -- a cell whose interval runs past the
        limit is drawn truncated and reads as a shorter interval, which is a
        data error the eye cannot catch."""
        lo = min(f[(f.metric == "macro_auc") & (f.level == "scan")]
                 .boot_ci_lo.min() for f in frames)
        hi = max(f[(f.metric == "macro_auc") & (f.level == "scan")]
                 .boot_ci_hi.max() for f in frames)
        pad = 0.02
        return (float(lo) - pad, float(hi) + pad)

    OUT.mkdir(exist_ok=True)

    # ---- FIGURE 4: split level -------------------------------------------
    # Authored at SINGLE and included at width=0.492\textwidth (= 3.20/6.50).
    # Including a 3.20 in figure at \textwidth would rescale its type by 2.03x.
    fig_leak = plt.figure(figsize=(FS.SINGLE, 2.95))
    ax_leak = fig_leak.add_subplot(111)
    panel_leak(ax_leak, ts_summ, scan_summ, ts_per, scan_per)
    FS.finish(fig_leak, OUT / "fig4_leakage.png", FS.SINGLE)

    # ---- FIGURE 5: cross-site --------------------------------------------
    # (b) and (c) of the old combined figure, now (a) and (b). They STAY in one
    # figure sharing one x and one y: see the module docstring — the vertical
    # offset between a fold's four-class cell and its binary cell IS the
    # result, and separating them would rescale that offset away.
    fig_xs = plt.figure(figsize=(FS.DOUBLE, 4.05))
    gs = fig_xs.add_gridspec(2, 1, height_ratios=[1, 1])
    ax_four = fig_xs.add_subplot(gs[0, 0])
    ax_bin = fig_xs.add_subplot(gs[1, 0], sharex=ax_four)

    # A1 = four-class cleft subtyping; A2 = the same cohort, binarised. Each
    # panel's rule is its OWN task's within-site value.
    ylim = shared_ylim(xs4, xsb)
    panel_xsite(ax_four, xs4, within_site("A1_fix"), "XS_fix_",
                "(a) Unseen site — four-class subtyping",
                show_group_labels=False, show_fold_labels=False,
                within_label="within-site,\nfour-class", ylim=ylim)
    panel_xsite(ax_bin, xsb, within_site("A2_fix"), "XSB_fix_",
                "(b) Unseen site — binary screening",
                show_group_labels=True, show_fold_labels=True,
                within_label="within-site,\nbinary", ylim=ylim)

    # One key for both cross-site panels, and it goes INSIDE (a). The shared y
    # range runs to ~0.93 for (b)'s sake while (a)'s cells now top out at 0.774,
    # so the top strip of (a) still holds no data. That was true of the
    # patience-20 arm too, but by a wider margin: check it stays true if these
    # panels are ever redrawn again, since the corrected four-class cells sit
    # ~0.10 higher than the ones this placement was chosen against.
    h = [plt.Line2D([], [], color=FS.ENC_COLOR[e], lw=1.0, marker="o", ms=3.2,
                    label=FS.ENC_LABEL[e]) for e in FS.ENCODERS]
    ax_four.legend(handles=h, frameon=False, ncol=4, fontsize=6.2,
                   loc="upper center", handlelength=1.0, columnspacing=0.9,
                   borderaxespad=0.15)

    FS.finish(fig_xs, OUT / "fig5_xsite.png", FS.DOUBLE)


if __name__ == "__main__":
    main()
