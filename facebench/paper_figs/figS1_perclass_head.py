"""
Supplementary Fig. S1 — per-class F1 under softmax vs prototype retrieval,
33-class clinical diagnosis. thread facebase3d-paper-2026-08-08.

Supports Sec. 2.5 paragraph 3: prototype retrieval redistributes performance
across diagnoses rather than lifting it uniformly, and the redistribution is not
an artefact of near-singleton classes.

Source: results/perclass_head_B2_corrected_trainseed.npz, built by
facebench/eval/emit_perclass.py --family trainseed from
B2_corrected_ts{1..5} at subject level. Five INDEPENDENTLY TRAINED models, with
each model's inference seeds averaged down before averaging across models.
The macro-F1 difference this panel decomposes is +3.55 pp, which the generator
checks against contrast_tests_trainseed.csv on every run.

THRESHOLD, STATED BECAUSE IT MATTERS. "Improved / reduced / unchanged" counts
depend on what counts as unchanged. The manuscript's 14/9/10 corresponds to a
+/-0.01 absolute F1 band; exact comparison gives 15/11/7. The band is drawn on
panel (a) so the reader can see the convention rather than infer it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import figstyle as fs

BAND = 0.01                      # |dF1| <= BAND counted as unchanged
UP, DOWN, FLAT, DEAD = "#1B6B4A", "#B03A2E", "#9A9A9A", "#D5D5D5"

fs.apply()
# _trainseed: five independently trained models, inference seeds averaged down
# within each. Until 2026-09-08 this read perclass_head_B2_corrected.npz -- ONE
# training run, five INFERENCE seeds -- while the macro-F1 difference this panel
# decomposes had moved to the sweep on 2026-09-03. The two disagreed by 2.6 pp.
# Written by facebench/eval/emit_perclass.py --family trainseed.
d = np.load("../../results/perclass_head_B2_corrected_trainseed.npz",
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

fig = plt.figure(figsize=(fs.DOUBLE, 5.6))
gs = fig.add_gridspec(1, 2, width_ratios=[2.7, 1.0])
ax = fig.add_subplot(gs[0, 0])

colors = []
for i in order:
    if dead[i]:
        colors.append(DEAD)
    elif abs(diff[i]) <= BAND:
        colors.append(FLAT)
    else:
        colors.append(UP if diff[i] > 0 else DOWN)

ax.barh(y, diff[order], color=colors, height=0.72, linewidth=0)
ax.axvline(0, color=fs.INK, lw=0.7, zorder=3)
ax.axvspan(-BAND, BAND, color=fs.MUTED, alpha=0.14, lw=0, zorder=0)
# Placed in the empty band beside the zero-change classes; anywhere near the
# top collides with the two largest bars.
ax.annotate(f"shaded: $\\pm${BAND:.2f}, counted as unchanged",
            xy=(0.055, float(np.where(dead[order])[0].mean())),
            fontsize=6.4, color=fs.MUTED, va="center", ha="left")

lab = [f"{names[i][:30]}  (n={sup[i]})" for i in order]
ax.set_yticks(y)
ax.set_yticklabels(lab, fontsize=6.4)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel("Change in per-class $F_1$, prototype $-$ softmax")
ax.set_title("(a)  Per-class $F_1$ change, 33-class diagnosis",
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

# ── (b) where the macro-F1 difference comes from ────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
K = len(diff)
large, small = sup > 2, sup <= 2
c_large, c_small = 100 * diff[large].sum() / K, 100 * diff[small].sum() / K

ax2.bar([0], [c_large], width=0.62, color="#123F63", linewidth=0)
ax2.bar([0], [c_small], width=0.62, bottom=[c_large], color="#7FA8CC",
        linewidth=0)
ax2.axhline(0, color=fs.INK, lw=0.7)
ax2.set_xticks([0]); ax2.set_xticklabels(["macro-$F_1$\ndifference"], fontsize=7)
ax2.set_xlim(-0.55, 2.1)
ax2.set_ylabel("Contribution (percentage points)")
ax2.set_title("(b)  By class support", loc="left", fontsize=8.5)
# Both labels sit OUTSIDE the bar: the n>2 caption does not fit inside a bar
# this narrow, and half-clipped bold white text is worse than an offset label.
ax2.annotate(f"{c_large:+.1f} pp\n$n>2$ ({int(large.sum())} classes)",
             xy=(0.36, c_large / 2), ha="left", va="center",
             fontsize=6.8, color=fs.INK, linespacing=1.6)
ax2.annotate(f"{c_small:+.1f} pp\n$n\\leq2$ ({int(small.sum())} classes)",
             xy=(0.36, c_large + c_small / 2), ha="left", va="center",
             fontsize=6.8, color=fs.INK, linespacing=1.6)
ax2.annotate(f"total {c_large + c_small:+.1f} pp",
             xy=(0, c_large + c_small), xytext=(0, 6),
             textcoords="offset points", ha="center", fontsize=7, color=fs.INK)
ax2.set_ylim(0, (c_large + c_small) * 1.28)

fs.finish(fig, "../../figures/figS1_perclass_head.png", fs.DOUBLE)

print(f"  counts (band {BAND}): up={n_up} down={n_dn} flat={n_fl} dead={n_dd}"
      f"  -> flat+dead={n_fl + n_dd}")
print(f"  macro-F1 diff {100 * diff.mean():+.2f} pp"
      f"  = {c_large:+.2f} (n>2) {c_small:+.2f} (n<=2)")
