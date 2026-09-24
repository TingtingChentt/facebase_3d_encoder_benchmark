#!/usr/bin/env python3
"""
Figure 2 — the AUC/top-1 gap, what aggregation moves, and what the head does not.
thread facebase3d-paper-2026-08-08.

REPLACES figures/fig_block5_paper.png. That was the ICIBM poster's seven-panel
block 5 with its text baked into the raster: included at \\textwidth its type
landed near 6 pt, and it carried explanatory panels sized for a board rather
than for a reader holding a page.

WHAT THIS FIGURE HAS TO DO, and what the poster version got wrong. The claims in
Results 2.3 are all COMPARISONS WITH INTERVALS:
  - top-1 is far below AUC, and the gap grows with K
  - subject aggregation moves top-1 substantially
  - the head does NOT move top-1, but does move macro-F1 at 33 classes
The poster showed these as bar heights with delta badges, which is precisely the
arithmetic the paper says you must not do — a difference of two marginal bars is
not the paired contrast, and the badges invited exactly that reading. Here the
comparisons ARE the plot: panel (b) shows paired differences with their own
bootstrap intervals against a zero line, so "ties" and "moves" are read off the
interval rather than asserted in a caption.

SIGN CONVENTION, stated because the project has been bitten by it. contrast_tests.csv
stores softmax-minus-proto for head contrasts; the manuscript prose reads
proto-minus-softmax. This script NEGATES the head rows and says so on the axis.
Aggregation rows are stored subject-minus-scan, which already matches the prose.

MOVED TO THE TRAINING-SEED SWEEP 2026-09-08 (thread
facebase3d-a1diag-2026-09-04). Until then this figure read summary_ci.csv and
contrast_tests.csv -- ONE unseeded checkpoint per cell -- while Table 2 and the
Sec 2.3 prose beside it had moved to the five-retraining sweep on 2026-09-03.
The two disagreed by up to 4.5 pp: the figure put 19-class subject-level
accuracy at 0.350 where the text said 0.309, and panel (b) drew the 33-class
aggregation gain at +9.1 pp where the text quoted +10.3 +/- 1.6. A reader
checking the text against the figure it cites would have found neither number.
Nothing about the underlying result changed; only which runs are plotted.

WHAT IS DRAWN, and why no bootstrap interval appears in either panel. Per the
2026-09-04 these figures show RETRAINING variability alone -- the same ruling
that removed the subject-bootstrap CI from Fig. 2 and Fig. 5(a). Both panels
therefore show the five independently trained models directly:
  panel (a)  marker = mean over 5 training seeds; dots = the 5 models
  panel (b)  marker = mean paired difference over the 5 models; dots = the 5
             per-model differences, each one paired WITHIN its own training run
The zero line in (b) is still what a comparison is read against, but "we could
not resolve this" is now read as dots falling on both sides of it rather than
as an interval crossing it. That is the stronger statement of the two: it says
retraining alone can flip the sign. The per-model subject-bootstrap intervals
are not drawn but are still in contrast_tests_trainseed.csv, and the Sec 2.3
prose quotes how many of the five excluded zero.

Usage: python3 fig2_core.py
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
TASKS = [("B1_corrected", "19 classes"), ("B2_corrected", "33 classes")]
ENC = "pointnet2"


def panel_gap(ax, summ, per):
    """AUC against top-1 at both units. The vertical distance between the two
    series IS the gap the section is about, so it is drawn as a span rather
    than left to the reader to subtract.

    Markers are five-retraining means and the dots behind them are the five
    models, so the gap is visibly larger than the spread that produces it."""
    def cell(frame, task, level, metric, col):
        r = frame[(frame.base_task == task) & (frame.level == level)
                  & (frame["head"] == "softmax") & (frame.encoder == ENC)
                  & (frame.metric == metric)]
        return r[col]

    xs = np.arange(len(TASKS) * 2, dtype=float)
    xs[2:] += 0.55                       # separate the two tasks
    labels, auc, acc, auc_pts, acc_pts = [], [], [], [], []
    for task, tlab in TASKS:
        for level, llab in [("scan", "scan"), ("subject", "subject")]:
            for metric, mean_out, pts_out in (("macro_auc", auc, auc_pts),
                                              ("accuracy", acc, acc_pts)):
                m = cell(summ, task, level, metric, "train_seed_mean")
                v = cell(per, task, level, metric, "value").to_numpy()
                if len(m) != 1 or len(v) != 5:
                    raise SystemExit(
                        f"fig2_core: {len(m)} summary / {len(v)} per-run rows "
                        f"for {task}/{level}/{metric}, expected 1 and 5 -- has "
                        f"aggregate_trainseed.py been run?")
                mean_out.append(float(m.iloc[0]))
                pts_out.append(v)
            labels.append(llab)

    for x, a, t in zip(xs, auc, acc):
        ax.vlines(x, t, a, color=FS.RULE, lw=3.2, alpha=0.65, zorder=1)
    rng = np.random.default_rng(0)       # jitter only; nothing downstream reads it
    for pts, col in ((auc_pts, FS.ENC_COLOR["pointnet2"]),
                     (acc_pts, FS.ENC_COLOR["pointnet"])):
        for x, v in zip(xs, pts):
            ax.scatter(x + rng.uniform(-0.10, 0.10, size=len(v)), v, s=1.8,
                       color=FS.INK, linewidths=0, alpha=0.85, zorder=5)
    ax.plot(xs, auc, "o", ms=5, color=FS.ENC_COLOR["pointnet2"],
            label="macro AUC", zorder=3)
    ax.plot(xs, acc, "s", ms=4.6, color=FS.ENC_COLOR["pointnet"],
            markeredgecolor=FS.INK, markeredgewidth=0.5,
            label="top-1 accuracy", zorder=3)

    for x, a, t in zip(xs, auc, acc):
        ax.annotate(f"{a - t:.2f}", xy=(x, (a + t) / 2), xytext=(5, 0),
                    textcoords="offset points", fontsize=6.2,
                    color=FS.MUTED, va="center")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    for xc, (_, tlab) in zip([xs[:2].mean(), xs[2:].mean()], TASKS):
        ax.annotate(tlab, xy=(xc, -0.155), xycoords=("data", "axes fraction"),
                    ha="center", fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("(a) Diagnostic discrimination versus top-1 accuracy",
                 pad=4, fontsize=8.0)
    ax.grid(axis="y", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4,
              borderaxespad=0.2)


def panel_contrasts(ax, con):
    """Paired differences with their own intervals. Ticks are read against the
    zero line: an interval crossing it is a comparison we could not resolve."""
    def seeds(contrast, task, metric, sign):
        """The five per-model paired differences for one cell.

        Each row is a contrast computed WITHIN one training run, so the spread
        across them is retraining variance on an already-paired quantity. Fewer
        than five means the sweep is incomplete and the marker above them would
        not be a five-model mean, so this refuses rather than averaging what
        happens to be present."""
        r = con[(con.contrast == contrast) & (con.metric == metric)
                & (con.task.str.startswith(task + "_ts"))]
        if len(r) != 5:
            raise SystemExit(f"fig2_core: {len(r)} training seeds for "
                             f"{contrast}/{task}/{metric}, expected 5")
        return sign * r.sort_values("task").diff_mean.to_numpy()

    rows = []
    for task, tlab in TASKS:
        for metric, mlab in [("accuracy", "top-1"), ("macro_f1", "macro-$F_1$"),
                             ("macro_auc", "macro AUC")]:
            rows.append((f"{mlab}", tlab, "aggregation",
                         seeds(f"C2_agg_softmax[{ENC}]", task, metric, 1.0)))
            # NEGATED: csv is softmax-minus-proto, prose is proto-minus-softmax.
            # Same flip emit_numbers_trainseed.py applies via its FLIP set, so
            # the panel and the \tsDxHead*Diff macros carry the same sign.
            rows.append((f"{mlab}", tlab, "head",
                         seeds(f"C1a_head_subject[{ENC}]", task, metric, -1.0)))

    order, ylab = [], []
    for tlab in [t[1] for t in TASKS][::-1]:
        for mlab in ["macro AUC", "macro-$F_1$", "top-1"]:
            order.append((mlab, tlab))
            ylab.append(f"{mlab}")

    style = {"aggregation": dict(color=FS.ENC_COLOR["pointnet2"], marker="o",
                                 ms=4.4),
             "head": dict(color=FS.ACCENT, marker="D", ms=3.6)}
    ax.axvline(0, color=FS.INK, lw=0.7, zorder=2)
    rng = np.random.default_rng(0)       # jitter only; nothing downstream reads it
    for i, key in enumerate(order):
        for k, dy in [("aggregation", 0.17), ("head", -0.17)]:
            r = next(x for x in rows if (x[0], x[1]) == key and x[2] == k)
            v = r[3] * 100
            # Span of the five models, then the five models, then the mean.
            ax.plot([v.min(), v.max()], [i + dy] * 2, color=style[k]["color"],
                    lw=1.1, zorder=3, solid_capstyle="round", alpha=0.55)
            ax.scatter(v, np.full(len(v), i + dy)
                       + rng.uniform(-0.055, 0.055, size=len(v)), s=1.8,
                       color=FS.INK, linewidths=0, alpha=0.85, zorder=6)
            ax.plot(v.mean(), i + dy, color=style[k]["color"], zorder=4,
                    marker=style[k]["marker"], ms=style[k]["ms"],
                    markeredgecolor="white", markeredgewidth=0.4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(ylab)
    # Extra room at the bottom so the legend sits in empty space rather than
    # on the 33-class rows.
    ax.set_ylim(-1.45, len(order) - 0.4)
    ax.axhline(2.5, color=FS.RULE, lw=0.5)
    for i, tlab in enumerate([t[1] for t in TASKS][::-1]):
        ax.annotate(tlab, xy=(1.0, i * 3 + 1), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", rotation=90,
                    va="center", ha="left", fontsize=7)
    ax.set_xlabel("difference (percentage points)")
    ax.set_title("(b) Effects of subject aggregation and inference strategy",
                 pad=4, fontsize=8.0)
    ax.grid(axis="x", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    h = [plt.Line2D([], [], color=style[k]["color"], marker=style[k]["marker"],
                    ms=style[k]["ms"], lw=1.1, label=lab)
         for k, lab in [("aggregation", "subject − scan"),
                        ("head", "prototype − softmax")]]
    ax.legend(handles=h, frameon=False, loc="lower center", ncol=2,
              handletextpad=0.4, borderaxespad=0.15, columnspacing=1.4)


def main():
    FS.apply()
    # The training-seed sweep, NOT summary_ci.csv / contrast_tests.csv. Those
    # hold one unseeded checkpoint per cell and are what Table 2 stopped using
    # on 2026-09-03; see the module docstring.
    summ = pd.read_csv(C.OUT / "trainseed_summary.csv")
    per = pd.read_csv(C.OUT / "trainseed_per_run.csv")
    con = pd.read_csv(C.OUT / "contrast_tests_trainseed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(FS.DOUBLE, 3.05),
                             gridspec_kw=dict(width_ratios=[1, 1.25]))
    panel_gap(axes[0], summ, per)
    panel_contrasts(axes[1], con)
    # The panel (b) title is wider than its axes, and constrained_layout sizes
    # axes without looking at title WIDTH -- so at the default full-figure rect
    # the title ran a couple of pixels past the right edge and FS.finish's
    # canvas guard refused to save. Reserve a hair of right margin.
    fig.get_layout_engine().set(rect=(0, 0, 0.994, 1))
    OUT.mkdir(exist_ok=True)
    FS.finish(fig, OUT / "fig2_core.png", FS.DOUBLE)


if __name__ == "__main__":
    main()
