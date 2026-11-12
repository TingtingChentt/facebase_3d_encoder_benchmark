#!/usr/bin/env python3
"""
Paper Figure S2 — prototype inference redistributes per-class F1 and one-vs-rest AUC
(49-class task). Reads results/perclass_head_CF20_full_v3_trainseed.npz.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
import numpy as np
import matplotlib.pyplot as plt
from figures import figstyle as fs
from facebench.paths import FIG_OUT, ROOT as PROP  # noqa: E402

fs.apply()
TASKS = [("CF20_full_v3", "49-class syndrome diagnosis")]
C_F1, C_AUC = "#123F63", "#B03A2E"
RNG = np.random.default_rng(0)          # jitter only; never touches a value

fig, axes = plt.subplots(1, 2, figsize=(fs.DOUBLE, 2.8), squeeze=False,
                         gridspec_kw={"width_ratios": [1.35, 1.0]})
out = []

for r, (task, title) in enumerate(TASKS):
    # _trainseed: five independently trained models, inference seeds averaged
    # down within each. Until 2026-09-08 this read perclass_head_{task}.npz --
    # ONE training run -- while Sec 2.5's headline came from the sweep. See
    # facebench/evaluation/emit_perclass.py.
    d = np.load(PROP / "results" / f"perclass_head_{task}_trainseed.npz",
                allow_pickle=True)
    sup = d["support"]
    f_sm, f_pr, a_sm, a_pr = d["f_sm"], d["f_pr"], d["a_sm"], d["a_pr"]
    df, da = f_pr - f_sm, a_pr - a_sm
    ok = ~(np.isnan(da))

    # ── left: the two change distributions on one axis ──────────────────────
    ax = axes[r, 0]
    for i, (vals, col, lab) in enumerate([(df, C_F1, "$\\Delta F_1$"),
                                          (da[ok], C_AUC, "$\\Delta$AUC")]):
        yj = 1 - i + RNG.uniform(-0.13, 0.13, len(vals))
        ax.scatter(vals, yj, s=13, facecolor=col, edgecolor="white",
                   linewidth=0.35, alpha=0.85, zorder=3)
        m = float(np.abs(vals).mean())
        ax.plot([-m, m], [1 - i - 0.26] * 2, color=col, lw=2.2,
                solid_capstyle="butt", zorder=2)
        ax.annotate(f"mean |change| {m:.3f}", xy=(0.0, 1 - i - 0.40),
                    ha="center", va="top", fontsize=6.5, color=col)
    ax.axvline(0, color=fs.INK, lw=0.7, zorder=1)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["$\\Delta F_1$", "$\\Delta$AUC"])
    ax.set_ylim(-0.75, 1.45)
    ax.set_xlim(-0.35, 0.48)
    ax.set_xlabel("Per-class change, prototype $-$ softmax")
    ax.set_title(f"({'ac'[r]})  {title}", loc="left", fontsize=8.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # ── right: softmax vs prototype, identical limits ───────────────────────
    ax2 = axes[r, 1]
    ax2.plot([0, 1], [0, 1], color=fs.RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax2.scatter(f_sm, f_pr, s=6 + 2.4 * sup, facecolor=C_F1,
                edgecolor="white", linewidth=0.35, alpha=0.8, zorder=3,
                label="$F_1$")
    ax2.scatter(a_sm[ok], a_pr[ok], s=6 + 2.4 * sup[ok], facecolor=C_AUC,
                edgecolor="white", linewidth=0.35, alpha=0.8, zorder=3,
                label="AUC")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1); ax2.set_aspect("equal")
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xlabel("Softmax"); ax2.set_ylabel("Prototype")
    ax2.set_title(f"({'bd'[r]})", loc="left", fontsize=8.5)
    if r == 0:
        ax2.legend(loc="lower right", frameon=False, fontsize=6.8,
                   handletextpad=0.3, borderaxespad=0.4)
    out.append((task, float(np.abs(df).mean()), float(np.abs(da[ok]).mean())))

fs.finish(fig, FIG_OUT / "figS2_redistribution.png", fs.DOUBLE)
for t, mf, ma in out:
    print(f"  {t:14s} mean|dF1| = {mf:.4f}   mean|dAUC| = {ma:.4f}"
          f"   ratio {mf / ma:.1f}x")
