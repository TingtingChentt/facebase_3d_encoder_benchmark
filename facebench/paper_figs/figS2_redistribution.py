"""
Supplementary Fig. S2 — per-class F1 moves, per-class AUC does not.
thread facebase3d-paper-2026-08-08.

Supports Sec. 2.5 paragraph 4: "prototype retrieval mainly changed which
diagnoses were classified well or poorly, while leaving overall discrimination
similar." Two halves, two kinds of panel:

  LEFT   the change itself, dF1 and dAUC per class on ONE shared axis. This is
         the panel that carries the claim, because both quantities are plotted
         against the same scale and their spreads are directly comparable.
  RIGHT  softmax vs prototype per class, identical 0-1 limits on both axes and
         both panels. Shows WHICH classes moved: F1 leaves the diagonal, AUC
         sits on it, high and tightly clustered.

WHY THE AXES ARE FORCED TO 0-1 ON THE RIGHT. An earlier version let each panel
autoscale, which gave the F1 panels a 0.8-wide axis and the AUC panels a 0.5-wide
one. Equal displacement from the diagonal then LOOKS bigger on the AUC panel, so
the figure understated its own result. Same limits everywhere, or the comparison
is not a comparison.

Reproduces the manuscript's macro values exactly:
  33-class  F1 0.280 -> 0.341,  AUC 0.913 -> 0.897
  19-class  F1 0.326 -> 0.349,  AUC 0.811 -> 0.812
"""
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
TASKS = [("B2_corrected", "33-class clinical diagnosis"),
         ("B1_corrected", "19-class syndrome category")]
C_F1, C_AUC = "#123F63", "#B03A2E"
RNG = np.random.default_rng(0)          # jitter only; never touches a value

fig, axes = plt.subplots(2, 2, figsize=(fs.DOUBLE, 5.4),
                         gridspec_kw={"width_ratios": [1.35, 1.0]})
out = []

for r, (task, title) in enumerate(TASKS):
    # _trainseed: five independently trained models, inference seeds averaged
    # down within each. Until 2026-09-08 this read perclass_head_{task}.npz --
    # ONE training run -- while Sec 2.5's headline came from the sweep. See
    # facebench/eval/emit_perclass.py, which writes both.
    d = np.load(f"../../results/perclass_head_{task}_trainseed.npz",
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

fs.finish(fig, "../../figures/figS2_redistribution.png", fs.DOUBLE)
for t, mf, ma in out:
    print(f"  {t:14s} mean|dF1| = {mf:.4f}   mean|dAUC| = {ma:.4f}"
          f"   ratio {mf / ma:.1f}x")
