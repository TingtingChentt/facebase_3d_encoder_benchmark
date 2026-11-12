#!/usr/bin/env python3
"""
Paper Figure S1 — per-diagnosis change in subject-level F1, prototype minus softmax
(49-class task, PointNet++). Reads results/perclass_head_CF20_full_v3_trainseed.npz
(written by facebench/evaluation/emit_numbers_cf20.py via emit_perclass.collect).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from figures import figstyle as fs

BAND = 0.01                      # |dF1| <= BAND counted as unchanged
UP, DOWN, FLAT, DEAD = "#1B6B4A", "#B03A2E", "#9A9A9A", "#D5D5D5"

fs.apply()
# _trainseed: five independently trained models, inference seeds averaged down
# within each. Until 2026-09-08 this read perclass_head_B2_corrected.npz -- ONE
# training run, five INFERENCE seeds -- while the macro-F1 difference this panel
# decomposes had moved to the sweep on 2026-09-03. The two disagreed by 2.6 pp.
from facebench.paths import FIG_OUT, ROOT as PROP  # noqa: E402
# Written by facebench/evaluation/emit_numbers_cf20.py (emit_perclass.collect).
d = np.load(PROP / "results" / "perclass_head_CF20_full_v3_trainseed.npz",
            allow_pickle=True)
names = np.array([str(x) for x in d["names"]])
sup, f_sm, f_pr = d["support"], d["f_sm"], d["f_pr"]
diff = f_pr - f_sm

# Classes the model never gets right under EITHER rule are not "unchanged" in
# the same sense as a class that moved a little; they are floor-bound. Separate
# them so the flat count is not read as ten genuine ties.
dead = (f_sm == 0) & (f_pr == 0)
order = np.argsort(diff)
y = np.arange(len(order))

fig = plt.figure(figsize=(fs.DOUBLE * 0.92, 7.4))
ax = fig.add_subplot(1, 1, 1)

colors = []
for i in order:
    if dead[i]:
        colors.append(DEAD)
    elif abs(diff[i]) <= BAND:
        colors.append(FLAT)
    else:
        colors.append(UP if diff[i] > 0 else DOWN)

ax.barh(y, diff[order], color=colors, height=0.72, linewidth=0)
# CF20: every class has >=3 test subjects, so the old panel (b) split by
# support (n<=2 vs n>2) is empty. Instead the five per-run changes are drawn
# on each bar, because the text says the per-class direction is not stable.
runs = d["f_pr_runs"] - d["f_sm_runs"]
for k, i in enumerate(order):
    ax.scatter(runs[:, i], np.full(runs.shape[0], k), s=1.6, color=fs.INK,
               linewidths=0, alpha=0.8, zorder=4)
ax.axvline(0, color=fs.INK, lw=0.7, zorder=3)
ax.axvspan(-BAND, BAND, color=fs.MUTED, alpha=0.14, lw=0, zorder=0)
# Placed in the empty band beside the zero-change classes; anywhere near the
# top collides with the two largest bars.
ax.annotate(f"shaded: $\\pm${BAND:.2f}, counted as unchanged",
            xy=(0.10, float(np.where(dead[order])[0].mean()) if dead.any() else 2.0),
            fontsize=6.4, color=fs.MUTED, va="center", ha="left")

short = [n.replace("Epileptic Encephalopathy Early Infantile Type 2",
                   "Early infantile epileptic enceph. 2")[:40] for n in names]
lab = [f"{short[i]}  (n={sup[i]})" for i in order]
ax.set_yticks(y)
ax.set_yticklabels(lab, fontsize=6.4)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel("Change in per-class $F_1$, prototype $-$ softmax")
ax.set_title("Per-class $F_1$ change, 49 classes",
             loc="left", fontsize=8.5)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

n_up = int(((diff > BAND) & ~dead).sum())
n_dn = int(((diff < -BAND) & ~dead).sum())
n_fl = int(((np.abs(diff) <= BAND) & ~dead).sum())
n_dd = int(dead.sum())
ax.legend(handles=[
    Patch(facecolor=UP, label=f"improved ({n_up})"),
    Patch(facecolor=DOWN, label=f"reduced ({n_dn})"),
    Patch(facecolor=FLAT, label=f"within band ({n_fl})"),
    Patch(facecolor=DEAD, label=f"$F_1=0$ under both ({n_dd})")],
    loc="lower right", frameon=False, fontsize=6.4, handlelength=1.1,
    borderaxespad=0.3)

fs.finish(fig, FIG_OUT / "figS1_perclass_head.png", fs.DOUBLE * 0.92)

print(f"  counts (band {BAND}): up={n_up} down={n_dn} flat={n_fl} dead={n_dd}"
      f"  -> flat+dead={n_fl + n_dd}")
