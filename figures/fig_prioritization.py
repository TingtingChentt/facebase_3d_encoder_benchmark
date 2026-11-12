#!/usr/bin/env python3
"""Figure 4: diagnostic prioritization (49-class syndrome diagnosis, PointNet++).
Reads results/prioritization/*.csv (analyses/prioritization/proto_retrieval.py).
a  candidate coverage: top-K accuracy, K=1..10 (patients)
b  macro top-K recall (each syndrome weighted equally)
c  patient-to-patient retrieval: Recall@K / Precision@K vs random retrieval
d  limited-reference prototypes: top-1 / top-5 vs reference patients per syndrome
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures import figstyle as FS  # noqa: E402
from facebench.paths import FIG_OUT as OUTD, RESULTS  # noqa: E402

FS.apply()
R = RESULTS / "prioritization"
HLAB = {"softmax": "Softmax", "proto": "Prototype"}
HCOL = {"softmax": FS.ENC_COLOR["pointnet2"], "proto": FS.ACCENT}
HMRK = {"softmax": "o", "proto": "D"}
KS = np.arange(1, 11)

A = pd.read_csv(R / "A_summary.csv").set_index("head")
Aci = pd.read_csv(R / "A_topk_patient_ci.csv")
Bs = pd.read_csv(R / "B_summary.csv").iloc[0]
Bch = pd.read_csv(R / "B_chance.csv")
Bci = pd.read_csv(R / "B_patient_ci.csv")
Cs = pd.read_csv(R / "C_summary.csv", dtype={"n_ref": str})

fig, axs = plt.subplots(2, 2, figsize=(FS.DOUBLE, 4.9))
(a, b), (c, d) = axs


def chance(ax, x, y, label):
    ax.plot(x, y, color=FS.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
    ax.annotate(label, xy=(x[-1], y[-1]), xytext=(2, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=FS.MUTED)


import matplotlib as _mpl
FS_TITLE_B = 0.9 * _mpl.font_manager.FontProperties(size=_mpl.rcParams["axes.titlesize"]).get_size_in_points()
for ax, pre, title in ((a, "top", "Correct syndrome within top K candidates"),
                       (b, "macro_top", "Class-balanced diagnostic candidate coverage")):
    for h in ("softmax", "proto"):
        m = np.array([A.loc[h, f"{pre}{k}_mean"] for k in KS])
        s = np.array([A.loc[h, f"{pre}{k}_sd"] for k in KS])
        if pre == "top":
            ci = Aci[Aci["head"] == h].sort_values("K")
            ax.fill_between(KS, ci.ci_lo, ci.ci_hi, color=HCOL[h], alpha=0.15, lw=0)
        ax.errorbar(KS, m, yerr=s, color=HCOL[h], marker=HMRK[h], ms=3.2, lw=1.1,
                    capsize=1.5, elinewidth=0.7, label=HLAB[h])
    chance(ax, KS, KS / 49, "chance")
    ax.set_xticks(KS); ax.set_xlim(0.5, 11.3); ax.set_ylim(0, 0.9)
    ax.set_xlabel("Candidate syndromes reviewed (K)")
    ax.set_ylabel("Top-K accuracy" if pre == "top" else "Macro top-K recall")
    ax.set_title(title, **({} if pre == "top" else {"fontsize": FS_TITLE_B}))
a.legend(loc="upper left", frameon=False)

# c: patient retrieval
kk = np.array([1, 3, 5, 10])
rec = np.array([Bs[f"recall@{k}_mean"] for k in kk]); recsd = np.array([Bs[f"recall@{k}_sd"] for k in kk])
pre = np.array([Bs[f"precision@{k}_mean"] for k in kk]); presd = np.array([Bs[f"precision@{k}_sd"] for k in kk])
c.errorbar(kk, rec, yerr=recsd, color=FS.INK, marker="s", ms=3.2, lw=1.1, capsize=1.5,
           elinewidth=0.7, label="Recall@K")
c.errorbar(kk, pre, yerr=presd, color=FS.INK, marker="s", mfc="white", ms=3.2, lw=1.1, ls="--",
           capsize=1.5, elinewidth=0.7, label="Precision@K")
c.plot(KS, [A.loc["proto", f"top{k}_mean"] for k in KS], color=HCOL["proto"], lw=0.8, alpha=0.3,
       zorder=0, label="Prototype top-K (a), for reference")
chance(c, kk, Bch.chance_recall.to_numpy(), "random recall")
c.plot(kk, Bch.chance_precision, color=FS.MUTED, lw=0.8, ls=(0, (1, 1.5)))
c.annotate("random precision", xy=(kk[-1], Bch.chance_precision.iloc[-1]), xytext=(2, 0),
           textcoords="offset points", va="center", ha="left", fontsize=6.5, color=FS.MUTED)
for k, n in zip(kk, [Bs[f"n_distinct_dx@{k}_mean"] for k in kk]):
    c.annotate(f"{n:.1f}", xy=(k, 0.93), ha="center", fontsize=6, color=FS.MUTED)
c.annotate("distinct syndromes among K patients:", xy=(0.6, 0.98), fontsize=6, color=FS.MUTED)
c.set_xticks(kk); c.set_xlim(0.5, 11.3); c.set_ylim(0, 1.03); c.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
c.set_xlabel("Retrieved training patients (K)")
c.set_ylabel("Proportion of test patients")
c.set_title("Retrieval of previously diagnosed patients")
_h, _l = c.get_legend_handles_labels()
_o = [_l.index(n) for n in ("Recall@K", "Precision@K", "Prototype top-K (a), for reference")]
c.legend([_h[i] for i in _o], [_l[i] for i in _o], loc="upper left", frameon=False, fontsize=6.2,
         bbox_to_anchor=(0.0, 0.90))

# d: limited reference
order = ["1", "3", "5", "10", "all"]
x = np.arange(len(order))
Cs = Cs.set_index("n_ref").loc[order]
for met, mk, ls in (("top5", "D", "-"), ("top1", "D", "--")):
    m = Cs[f"{met}_mean"].to_numpy()
    d.fill_between(x[:-1], Cs[f"{met}_q025"].to_numpy()[:-1], Cs[f"{met}_q975"].to_numpy()[:-1],
                   color=HCOL["proto"], alpha=0.15, lw=0)
    d.errorbar(x, m, yerr=Cs[f"{met}_sd_runs"], color=HCOL["proto"], marker=mk,
               mfc=HCOL["proto"] if met == "top5" else "white", ms=3.2, lw=1.1, ls=ls,
               capsize=1.5, elinewidth=0.7, label=f"Prototype top-{met[-1]}")
    sm = A.loc["softmax", f"{met}_mean"]
    d.axhline(sm, color=HCOL["softmax"], lw=0.8, ls=ls, alpha=0.8)
    d.annotate(f"softmax top-{met[-1]}, all data", xy=(x[0] - 0.3, sm), xytext=(0, 2),
               textcoords="offset points", fontsize=6, color=HCOL["softmax"], va="bottom")
chance(d, np.array([x[0] - 0.3, x[-1] + 0.3]), np.array([5 / 49, 5 / 49]), "chance top-5")
d.plot([x[0] - 0.3, x[-1] + 0.3], [1 / 49] * 2, color=FS.MUTED, lw=0.8, ls=(0, (1, 1.5)))
d.set_xticks(x); d.set_xticklabels(["1", "3", "5", "10", "All\n(14–123)"])
d.set_xlim(-0.5, len(order) - 0.2); d.set_ylim(0, 0.8)
d.set_xlabel("Reference patients per syndrome")
d.set_ylabel("Accuracy (test patients)")
d.set_title("Prototypes from few confirmed patients")
d.legend(loc="center left", frameon=False, bbox_to_anchor=(0.45, 0.58))

for ax, l in zip(axs.flat, "abcd"):
    ax.text(-0.13, 1.06, l, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom")
fig.tight_layout(h_pad=1.6, w_pad=2.0)
OUTD.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTD / "fig_prioritization.pdf")
FS.finish(fig, OUTD / "fig_prioritization.png", FS.DOUBLE)
