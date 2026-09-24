#!/usr/bin/env python3
"""
Figure 2 (fig:results) — five tasks, four encoders, RETRAINED over five
training seeds.  thread facebase3d-trainseed-2026-08-31.

REPLACES the single-checkpoint version drawn by fig1_tasks_encoders.py. That
figure plotted one unreproducible draw per bar (every pre-2026-08-31 checkpoint
was trained with an unseeded torch RNG) and put a subject-bootstrap interval on
it, so the only dispersion a reader could see was "would this survive a
different sample of patients" -- never "would this survive training the model
again". The sweep supplies the second quantity and this figure now shows that
one alone: as of 2026-09-04, the subject-bootstrap CI is no longer drawn,
so every mark in the panel refers to retraining variability.

WHAT IS PLOTTED, per encoder x task, scan level, softmax head:
  bar        mean over 5 independently trained models (train seeds 1-5); the
             inference seeds are averaged down WITHIN each training seed first,
             so PointNet++'s FPS noise does not leak into the training spread.
  dots       the five individual training seeds. Drawn rather than summarised
             because several cells are not symmetric around their mean and an
             SD bar alone would imply they are.

No error bars: the 95% bootstrap CI over test subjects was removed 2026-09-04.
It is a different variance source from the dot spread and the two were never
pooled, but showing both invited the reader to read one as the other. The
bootstrap intervals remain in results/trainseed_summary.csv.

The test split is frozen at seed 42 for every run, so the dot spread is
retraining variance and nothing else.

Usage: python3 fig1_tasks_encoders_trainseed.py
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

# task key, short label, chance accuracy (1/K)
# The two cleft tasks read the A1_fix / A2_fix sweeps: same cohort, same frozen
# split, same five training seeds, early stopping disabled. On both tasks the
# model sits at train loss ln(K) with validation macro-F1 pure noise for tens of
# epochs, and patience counted on that noise truncated runs inside the plateau,
# so the A1_ts/A2_ts arms measured the stopping rule as much as the task. See
# the A1_FIX / A2_FIX blocks in rerun_config.py.
TASKS = [
    ("C_corrected", "Combined\nscreening\n$K$=2", 1 / 2),
    ("A2_fix", "Cleft\nscreening\n$K$=2", 1 / 2),
    # A1_fix landed 2026-09-07. Both cleft columns are now the corrected arm,
    # so this figure no longer mixes stopping rules and the caption no longer
    # has to warn that it does.
    ("A1_fix", "Cleft\ntype\n$K$=4", 1 / 4),
    ("B1_corrected", "Syndrome\ncategory\n$K$=19", 1 / 19),
    ("B2_corrected", "Clinical\ndiagnosis\n$K$=33", 1 / 33),
]


def main():
    FS.apply()
    agg = pd.read_csv(C.OUT / "trainseed_summary.csv")
    per = pd.read_csv(C.OUT / "trainseed_per_run.csv")
    agg = agg[(agg.level == "scan") & (agg["head"] == "softmax")]
    per = per[(per.level == "scan") & (per["head"] == "softmax")]

    fig, axes = plt.subplots(1, 2, figsize=(FS.DOUBLE, 2.95))
    nT, nE = len(TASKS), len(FS.ENCODERS)
    width = 0.78 / nE
    xs = np.arange(nT)
    rng = np.random.default_rng(0)          # dot jitter only, no data effect

    for ax, metric, title in [(axes[0], "accuracy", "Accuracy"),
                              (axes[1], "macro_auc", "Macro AUC (one-vs-rest)")]:
        for j, enc in enumerate(FS.ENCODERS):
            m, seeds = [], []
            for task, _, _ in TASKS:
                r = agg[(agg.base_task == task) & (agg.encoder == enc)
                        & (agg.metric == metric)]
                if len(r) != 1:
                    raise SystemExit(f"{len(r)} rows for {task}/{enc}/{metric}")
                r = r.iloc[0]
                m.append(r.train_seed_mean)
                p = per[(per.base_task == task) & (per.encoder == enc)
                        & (per.metric == metric)]
                if len(p) != 5:
                    raise SystemExit(f"{len(p)} training seeds for "
                                     f"{task}/{enc}/{metric}, expected 5")
                seeds.append(p.value.to_numpy())
            off = (j - (nE - 1) / 2) * width
            ax.bar(xs + off, m, width * 0.92, color=FS.ENC_COLOR[enc],
                   label=FS.ENC_LABEL[enc], zorder=3,
                   edgecolor="white", linewidth=0.3)
            for i, v in enumerate(seeds):
                jit = rng.uniform(-width * 0.20, width * 0.20, size=len(v))
                ax.scatter(xs[i] + off + jit, v, s=1.6, zorder=6,
                           color=FS.INK, linewidths=0, alpha=0.85)

        if metric == "accuracy":
            # Chance is task-specific: draw a segment per group, not one line.
            for i, (_, _, ch) in enumerate(TASKS):
                ax.plot([i - 0.44, i + 0.44], [ch, ch], color=FS.ACCENT,
                        lw=0.8, ls=(0, (3, 2)), zorder=5,
                        solid_capstyle="butt")
            ax.plot([], [], color=FS.ACCENT, lw=0.8, ls=(0, (3, 2)),
                    label="chance (1/$K$)")
        else:
            ax.axhline(0.5, color=FS.ACCENT, lw=0.8, ls=(0, (3, 2)), zorder=5)
            ax.annotate("chance", xy=(0.015, 0.5),
                        xycoords=("axes fraction", "data"),
                        xytext=(0, 2.5), textcoords="offset points",
                        va="bottom", ha="left", fontsize=6.5,
                        color=FS.ACCENT)

        ax.set_title(title, pad=4)
        ax.set_xticks(xs)
        ax.set_xticklabels([t[1] for t in TASKS])
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=FS.RULE, lw=0.4, alpha=0.5, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("score")
    h, l = axes[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], ls="none", marker="o", ms=2.0,
                        color=FS.INK))
    l.append("training seed")
    fig.legend(h, l, loc="outside upper center", ncol=6, frameon=False,
               handlelength=1.2, columnspacing=1.0)
    OUT.mkdir(exist_ok=True)
    FS.finish(fig, OUT / "fig1_tasks_encoders_trainseed.png", FS.DOUBLE)


if __name__ == "__main__":
    main()
