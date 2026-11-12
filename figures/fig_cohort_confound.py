#!/usr/bin/env python3
"""
Figure: cohort composition, demographics and screening performance (Results sec:cohort).

Existing combined-task PointNet++ predictions only (C_full_v3, subject level, softmax);
the model is never retrained, only the subset of test subjects changes.

  a  age at scan by cohort and status (all subjects of the combined task; FB-TK0/FB-TX4 release no age)
  b  false-positive rate among unaffected TEST subjects, by cohort, grouped mixed vs control-only (bar = mean of 5 training runs,
     dots = runs; each run = mean of its 5 inference seeds)
  c  AUC under each sensitivity setting, within FB-TJ0 and within the cleft cohorts.
     PointNet++ rows: filled marker + bar = mean +- SD of 5 training runs (values = results/paper_macros/numbers_cohort.tex).
     Non-image baselines: single value (deterministic, no training-run variation).
     Reference lines (labelled in the top margin): pooled PointNet++, pooled cohort-identity baseline, chance.
Reads results/cohort_sensitivity/*.csv and the predictions (results/predictions/);
writes figures/output/fig_cohort_confound.{png,pdf}.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
from analyses.cohort_sensitivity import cohort_confound as CC   # noqa: E402
from figures import figstyle as FS          # noqa: E402

from facebench.paths import FIG_OUT as OUTD  # noqa: E402
R = CC.OUT
FS.apply()

COH_LABEL = {"FB-5A": "FB-5A\n(cleft)", "FB-56": "FB-56\n(cleft)", "FB-TJ0": "FB-TJ0\n(syndromic)",
             "FB-TK0": "FB-TK0", "FB-VWP": "FB-VWP", "FB-TX4": "FB-TX4"}
ORDER = ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]
C_AFF, C_UNAFF = "#B07A2E", "#3E6E9E"          # affected / unaffected (panel a)
C_TJ, C_CL = "#123F63", "#8C5A9E"              # FB-TJ0 / cleft (panel c)
C_MIXED, C_CTRL = "#3E6E9E", "#BFD3E6"         # cohort with probands / control-only (panel b)
LBOX = dict(facecolor="white", edgecolor="none", pad=0.4)   # value labels mask the reference lines


def panel_age(ax):
    m = CC.manifest(); demo = CC.demographics()
    s = m.groupby("subject_id").agg(dataset=("dataset", "first"), status=("binary_label", "first")).join(demo)
    cohorts = ["FB-5A", "FB-56", "FB-TJ0", "FB-VWP"]
    pos, data, cols = [], [], []
    for i, c in enumerate(cohorts):
        for j, (st, col) in enumerate([("Affected", C_AFF), ("Unaffected", C_UNAFF)]):
            a = s[(s.dataset == c) & (s.status == st)].age_years.dropna().to_numpy()
            if len(a) == 0:
                continue
            pos.append(i + (j - 0.5) * 0.36); data.append(a); cols.append(col)
    bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", lw=1.1), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    for b, col in zip(bp["boxes"], cols):
        b.set(facecolor=col, edgecolor=col, lw=0.6)
    for w, col in zip(np.repeat(cols, 2), bp["whiskers"]):
        col.set_color(w)
    for w, col in zip(np.repeat(cols, 2), bp["caps"]):
        col.set_color(w)
    ax.axhline(18, color=FS.MUTED, lw=0.6, ls=":")
    ax.text(-0.95, 18.8, "18 y", color=FS.MUTED, fontsize=7, ha="left", va="bottom")
    ax.set_xticks(range(len(cohorts)), [COH_LABEL[c] for c in cohorts])
    ax.set_xlim(-1.0, len(cohorts) - 0.4)
    ax.set_ylabel("Age at scan (years)")
    ax.set_ylim(0, 92)
    h = [plt.Rectangle((0, 0), 1, 1, fc=C_AFF), plt.Rectangle((0, 0), 1, 1, fc=C_UNAFF)]
    ax.legend(h, ["Affected", "Unaffected"], loc="upper left", frameon=False, handlelength=1, ncol=2,
              borderaxespad=0.1)
    ax.set_title("Age at scan", loc="left", fontsize=8)


def panel_fpr(ax):
    sub2ds = CC.manifest().drop_duplicates("subject_id").set_index("subject_id").dataset
    rows = []
    for ts in CC.TS:
        ids, y, ds = CC.load_run(ts, "pointnet2")
        co = pd.Series(ids).map(sub2ds).to_numpy()
        for c in ORDER:
            mk = (co == c) & (y == 0)
            rows.append(dict(ts=ts, cohort=c, fpr=100 * np.mean([(d["y_pred"][mk] == 1).mean() for d in ds]),
                             n=int(mk.sum())))
    r = pd.DataFrame(rows)
    # gate against rates_by_cohort.csv (macros FPR*)
    ref = pd.read_csv(R / "rates_by_cohort.csv").query("encoder == 'pointnet2' and metric == 'fpr_unaffected'")
    g = r.groupby("cohort").fpr.mean() / 100
    assert np.allclose(g.reindex(ref.cohort).to_numpy(), ref["mean"].to_numpy(), atol=1e-12)
    rng = np.random.default_rng(0)
    MIXED = ("FB-5A", "FB-56", "FB-TJ0")
    xs = [i + (0.7 if c not in MIXED else 0) for i, c in enumerate(ORDER)]   # gap between the two groups
    for x, c in zip(xs, ORDER):
        v = r[r.cohort == c]
        col = C_MIXED if c in MIXED else C_CTRL
        ax.bar(x, v.fpr.mean(), width=0.62, color=col, edgecolor=col, lw=0.6)
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v.fpr, s=6, color=FS.INK, zorder=3, lw=0)
        ax.text(x, v.fpr.max() + 1.2, f"{v.fpr.mean():.1f}", ha="center", va="bottom", fontsize=7)
    # group brackets
    for x0, x1, lab in ((xs[0], xs[2], "With affected subjects"),
                        (xs[3], xs[5], "Control-only")):
        yb = 56
        ax.plot([x0 - 0.3, x0 - 0.3, x1 + 0.3, x1 + 0.3], [yb - 1.5, yb, yb, yb - 1.5], color=FS.MUTED, lw=0.6)
        ax.text((x0 + x1) / 2, yb + 1.0, lab, ha="center", va="bottom", fontsize=6.8, color=FS.MUTED)
    ax.set_xticks(xs, [f"{c}\n{r[r.cohort == c].n.iloc[0]}" for c in ORDER])
    ax.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax.set_ylim(0, 66)
    ax.set_ylabel("False-positive rate among\nunaffected test subjects (%)")
    ax.set_title("False-positive rate by cohort", loc="left", fontsize=8)


def panel_auc(ax):
    b = pd.read_csv(R / "baselines.csv").set_index(["predictor", "subset"])
    w = pd.read_csv(R / "within_cohort_per_run.csv").query("encoder == 'pointnet2'")
    ps = pd.read_csv(R / "age_adjusted_per_seed.csv")
    groups = {"FB-TJ0": ("FB-TJ0 only", "FB-TJ0", C_TJ, "Within FB-TJ0 (syndromic)"),
              "cleft": ("cleft cohorts (FB-5A + FB-56)", "cleft (FB-5A + FB-56)", C_CL,
                        "Within cleft cohorts (FB-5A + FB-56)")}
    # (row label, kind, getter)
    rows = [
        ("Cohort identity only", "base", lambda g: [b.loc[("cohort membership only", g[0]), "auc"]]),
        ("Age only", "base", lambda g: [b.loc[("age only", g[0]), "auc"]]),
        ("Age + sex", "base",
         lambda g: [ps.query("cohort == @g[1] and subset == 'all ages'")["cv_auc_age+sex"].mean()]),
        ("PointNet++, within cohort", "model", lambda g: w[w.subset == g[0]].auc.to_numpy()),
        ("Age + sex + PointNet++", "model",
         lambda g: ps.query("cohort == @g[1] and subset == 'all ages'")["cv_auc_age+sex+face"].to_numpy()),
        ("PointNet++, matched pairs", "model",
         lambda g: ps.query("cohort == @g[1] and subset == 'age/sex matched (|diff| <= 2 y)'").auc_face.to_numpy()),
        ("PointNet++, age < 18 y", "model",
         lambda g: ps.query("cohort == @g[1] and subset == 'age < 18 y'").auc_face.to_numpy()),
    ]
    pooled_model = pd.read_csv(R / "within_cohort_summary.csv").query("encoder == 'pointnet2' and subset == 'all'")
    pooled_model = float(pooled_model.auc_mean.iloc[0])
    pooled_cohort = b.loc[("cohort membership only", "all"), "auc"]
    ref = [ax.axvline(0.5, color=FS.ACCENT, lw=0.7, ls="--"),
           ax.axvline(pooled_model, color=FS.MUTED, lw=0.8),
           ax.axvline(pooled_cohort, color=FS.MUTED, lw=0.8, ls=(0, (3, 2)))]
    ref_lab = ["Chance", "Combined screening, PointNet++", "Combined screening, cohort identity only"]
    # pooled values written on the figure, above each line
    for xv, ha, dx in ((pooled_cohort, "right", -2), (pooled_model, "left", 2)):
        ax.annotate(f"{xv:.3f}", xy=(xv, 1.0), xycoords=("data", "axes fraction"), xytext=(dx, 1.5),
                    textcoords="offset points", ha=ha, va="bottom", fontsize=6.6, color=FS.MUTED,
                    annotation_clip=False)
    rng = np.random.default_rng(1)
    for i, (lab, kind, get) in enumerate(rows):
        for k, (gk, g) in enumerate(groups.items()):
            yy = i + (k - 0.5) * 0.34
            v = np.asarray(get(g), float)
            if kind == "base":
                ax.scatter(v, [yy], marker="D", s=16, facecolor="white", edgecolor=g[2], lw=1.0, zorder=3)
                ax.text(v[0] + 0.012, yy, f"{v[0]:.3f}", fontsize=6.6, va="center", color=g[2])
            else:
                assert len(v) == 5, (lab, gk, len(v))
                mu, sd = v.mean(), v.std(ddof=1)
                ax.errorbar(mu, yy, xerr=sd, fmt="o", ms=4.2, color=g[2], elinewidth=0.9, capsize=1.8, zorder=3)
                ax.scatter(v, yy + rng.uniform(-0.07, 0.07, len(v)), s=4, color=g[2], alpha=0.45, lw=0, zorder=2)
                ax.text(mu + sd + 0.012, yy, f"{mu:.3f}", fontsize=6.6, va="center", color=g[2])
    # two blocks: no facial information (rows 0-2) vs facial model (rows 3-6), tinted band + header
    ax.axhspan(2.5, len(rows) - 0.5, color="#EEF2F6", lw=0, zorder=0)
    ax.axhline(2.5, color=FS.RULE, lw=0.6)
    hdr = dict(fontsize=7, fontweight="bold", color=FS.MUTED, ha="left", va="top")
    ax.text(0.305, -0.42, "No facial information", bbox=dict(fc="white", ec="none", pad=0.6), **hdr)
    ax.text(0.305, 2.58, "Facial model (existing PointNet++)", bbox=dict(fc="#EEF2F6", ec="none", pad=0.6), **hdr)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.30, 1.0)
    ax.set_xlabel("Subject-level AUC (affected vs unaffected)")
    h = [plt.Line2D([], [], marker="o", ls="", color=g[2], ms=4.2) for g in groups.values()]
    blank = plt.Line2D([], [], ls="", marker="")
    # column 1: the two groups; column 2: the three reference lines (pooled PointNet++, pooled cohort identity, chance)
    ax.legend(h + [blank] + [ref[1], ref[2], ref[0]], [g[3] for g in groups.values()] + [""]
              + [ref_lab[1], ref_lab[2], ref_lab[0]], loc="lower left", bbox_to_anchor=(-0.12, 1.01),
              ncol=2, frameon=False, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)


def main():
    # two subfigures so panel c's long row labels do not squeeze the top row
    fig = plt.figure(figsize=(FS.DOUBLE, 6.3))
    top, bot = fig.subfigures(2, 1, height_ratios=[1, 1.5])
    axa, axb = top.subplots(1, 2, width_ratios=[1, 1.2])
    axc = bot.subplots()
    panel_age(axa); panel_fpr(axb); panel_auc(axc)
    for sf, l in ((top, "a"), (bot, "c")):
        sf.text(0.0, 1.0, l, fontsize=10, fontweight="bold", ha="left", va="top")
    top.text(0.45, 1.0, "b", fontsize=10, fontweight="bold", ha="left", va="top")
    for ext in ("png", "pdf"):
        fig.savefig(OUTD / f"fig_cohort_confound.{ext}")
    print("wrote", OUTD / "fig_cohort_confound.png")


if __name__ == "__main__":
    main()
