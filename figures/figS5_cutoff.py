#!/usr/bin/env python3
"""
Paper Figure S3 (file figS5_cutoff.png): class-size cutoff sensitivity of the
Syndrome-label task.

Seven cutoffs, >=50 scans AND >=K subjects per class, K in 10..75 (55..12
classes); the manuscript's task is K=20 (49 classes). All seven share one
per-class subject split, so their test sets are nested. Subject level,
prototype retrieval, mean +/- SD over five training seeds.

(a) accuracy, with chance 1/classes; (b) macro AUC;
(c) the same 173 test subjects (the 12 classes kept at K=75) scored by every
    cutoff's model: argmax restricted to those 12 classes vs argmax over all
    classes the model was trained on.

Sources: results/cfv3_trainseed_summary.csv (a, b); panel (c) is computed here by
cutoff_panel.cf_fixed_panel from results/predictions/CF*_full_v3_*proto__subject* and
written to results/cutoff_fixed_test_per_run_v3.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from figures import figstyle as FS                                       # noqa: E402
from figures import cutoff_panel as L                                    # noqa: E402

from facebench.paths import FIG_OUT as OUT, ROOT as PROJ  # noqa: E402
from facebench.evaluation.derived import have_predictions  # noqa: E402
KS = [10, 15, 20, 30, 40, 50, 75]
NCLS = {10: 55, 15: 53, 20: 49, 30: 42, 40: 29, 50: 22, 75: 12}
MARK = {"geommlp": "D", "pointnet": "^", "dgcnn": "s", "pointnet2": "o"}


def main():
    FS.apply()
    s = pd.read_csv(PROJ / "results/cfv3_trainseed_summary.csv")
    s = s[(s.level == "subject") & (s["head"] == "proto")]
    fixed = PROJ / "results/cutoff_fixed_test_per_run_v3.csv"     # class-level, shipped
    if have_predictions("CF75_full_v3_ts*__*__proto__subject__*.npz"):
        L.cf_fixed_panel("proto", suffix="_v3").to_csv(fixed, index=False)
    fx = pd.read_csv(fixed)

    fig, axes = plt.subplots(1, 3, figsize=(FS.DOUBLE, 2.55))
    x = np.arange(len(KS))
    for ax, metric, title in [(axes[0], "accuracy", "(a) Accuracy"),
                              (axes[1], "macro_auc", "(b) Macro AUC")]:
        for enc in FS.ENCODERS:
            r = (s[(s.encoder == enc) & (s.metric == metric)]
                 .assign(K=lambda d: d.base_task.str.extract(r"CF(\d+)_full_v3")[0].astype(int))
                 .set_index("K").loc[KS])
            ax.errorbar(x, r.train_seed_mean, yerr=r.train_seed_sd, color=FS.ENC_COLOR[enc],
                        marker=MARK[enc], ms=3.4, lw=1.1, capsize=0, elinewidth=0.7,
                        label=FS.ENC_LABEL[enc], zorder=3)
        if metric == "accuracy":
            ax.plot(x, [1 / NCLS[k] for k in KS], color=FS.ACCENT, lw=0.8,
                    ls=(0, (3, 2)), label="chance")
            ax.set_ylim(0, 0.52)
        else:
            ax.set_ylim(0.70, 0.90)
        ax.set_title(title, pad=4, fontsize=8)
    for enc in ["dgcnn", "pointnet2"]:
        g = fx[fx.encoder == enc].groupby("K")[["restricted", "open"]].agg(["mean", "std"]).loc[KS]
        for col, ls in [("restricted", "-"), ("open", (0, (3, 1.5)))]:
            axes[2].errorbar(x, g[(col, "mean")], yerr=g[(col, "std")], color=FS.ENC_COLOR[enc],
                             marker=MARK[enc], ms=3.4, lw=1.1, ls=ls, elinewidth=0.7,
                             mfc=FS.ENC_COLOR[enc] if col == "restricted" else "white", zorder=3)
    axes[2].set_ylim(0.25, 0.55)
    axes[2].set_title("(c) Same 173 test subjects", pad=4, fontsize=8)
    h = [plt.Line2D([], [], color=FS.INK, ls="-", lw=1.0, label="12 classes only"),
         plt.Line2D([], [], color=FS.INK, ls=(0, (3, 1.5)), lw=1.0, label="all trained classes")]
    axes[2].legend(handles=h, frameon=False, fontsize=6.0, loc="lower right",
                   handlelength=2.0)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{k}\n({NCLS[k]})" for k in KS], fontsize=6.0)
        ax.axvline(KS.index(20), color=FS.RULE, lw=2.6, alpha=0.6, zorder=0)
        ax.grid(axis="y", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
    axes[1].set_xlabel("minimum subjects per class (number of classes)", fontsize=7)
    axes[0].set_ylabel("subject level, prototype", fontsize=7)
    hh, ll = axes[0].get_legend_handles_labels()
    fig.legend(hh, ll, loc="outside upper center", ncol=5, frameon=False, fontsize=6.5)
    OUT.mkdir(parents=True, exist_ok=True)
    FS.finish(fig, OUT / "figS5_cutoff.png", FS.DOUBLE)

    for k in KS:
        a = s[(s.base_task == f"CF{k}_full_v3") & (s.encoder == "pointnet2")]
        print(k, NCLS[k], {m: round(float(a[a.metric == m].train_seed_mean.iloc[0]), 3)
                           for m in ["accuracy", "macro_auc", "top5_accuracy"]})
    print(fx.groupby(["encoder", "K"])[["restricted", "open"]].mean().round(3)
            .loc[["pointnet2", "dgcnn"]])


if __name__ == "__main__":
    main()
