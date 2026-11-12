#!/usr/bin/env python3
"""
Figure 3: controlled patient-level aggregation (49-class syndrome diagnosis, PointNet++).
  a  decomposition of the reported scan->subject gain (all 385 test subjects)
  b  top-1 accuracy vs number of scans combined, same patients
  c  rank of the true syndrome vs number of scans combined, same patients
  d  per-patient change in true-syndrome rank, one -> two scans
  e  outcome of adding a second scan (top-1 corrected / lost, rank higher / lower)
Reads results/scan_aggregation/ (analyses/scan_aggregation/scan_count_controlled.py).
Writes figures/output/fig_aggregation.{png,pdf}.
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
R = RESULTS / "scan_aggregation"
rec = pd.read_csv(R / "per_subject_by_m.csv.gz", dtype={"m": str})
scan = pd.read_csv(R / "scan_level_per_seed.csv")
trans = pd.read_csv(R / "per_subject_pair_transitions.csv.gz")
pp = pd.read_csv(R / "B_per_patient_delta.csv")

HEADS = ["softmax", "proto"]
HLAB = {"softmax": "Softmax", "proto": "Prototype"}
HCOL = {"softmax": FS.ENC_COLOR["pointnet2"], "proto": FS.ACCENT}
HMRK = {"softmax": "o", "proto": "D"}


def per_run(df, keys, col):
    """inference seeds -> mean within run; returns Series indexed by keys + run."""
    return df.groupby(keys + ["run", "seed"])[col].mean().groupby(keys + ["run"]).mean()


def tint(color, f):
    """mix `color` with white; f = share of white."""
    import matplotlib.colors as mc
    c = np.array(mc.to_rgb(color))
    return tuple(c + (1 - c) * f)


def dots(ax, x, v, color):
    """per-model dots, small and dark as in the original Fig. 3b."""
    j = np.linspace(-0.06, 0.06, len(v))
    ax.scatter(x + j, v, s=1.8, color=FS.INK, alpha=0.85, lw=0, zorder=5)


def panel_a(ax):
    """decomposition. The 'combining scans' increment (patient one scan -> all scans) is the
    emphasised quantity: drawn as the solid top segment of the all-scans bars and labelled in bold;
    the evaluation-unit step is shown in muted grey."""
    w = 0.36
    xs = np.arange(3)
    gains = {}
    for k, h in enumerate(HEADS):
        s = per_run(scan[scan["head"] == h], [], "top1")
        one = per_run(rec[(rec["head"] == h) & (rec.m == "1")], [], "top1")
        al = per_run(rec[(rec["head"] == h) & (rec.m == "all")], [], "top1")
        gains[h] = (100 * (one - s).mean(), 100 * (al - one).mean())
        x = xs + (k - 0.5) * w
        light = tint(HCOL[h], 0.55)
        ax.bar(x[:2], [s.mean(), one.mean()], w * 0.92, color=light, label=HLAB[h])
        ax.bar(x[2], one.mean(), w * 0.92, color=light)
        ax.bar(x[2], al.mean() - one.mean(), w * 0.92, bottom=one.mean(), color=HCOL[h])
        for xi, v in zip(x, (s, one, al)):
            dots(ax, xi, v.to_numpy(), HCOL[h])
    # evaluation-unit step: muted
    y0 = 0.345
    ax.annotate("", xy=(0.98, y0), xytext=(0.02, y0),
                arrowprops=dict(arrowstyle="->", color=FS.RULE, lw=0.6))
    ax.text(0.5, y0 + 0.008, f"evaluation unit\n+{gains['softmax'][0]:.1f} / +{gains['proto'][0]:.1f} pp",
            ha="center", va="bottom", fontsize=6.2, color=FS.MUTED, linespacing=1.05)
    # combining scans: emphasised
    y1 = 0.445
    ax.annotate("", xy=(1.98, y1), xytext=(1.02, y1),
                arrowprops=dict(arrowstyle="-|>", color=FS.INK, lw=1.1, mutation_scale=8))
    ax.text(1.5, y1 + 0.068, "Combining scans", ha="center", va="bottom", fontsize=7.8,
            fontweight="bold", color=FS.INK)
    t = ax.text(1.5, y1 + 0.014, f"+{gains['softmax'][1]:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=HCOL["softmax"])
    # chain the remaining pieces to the right of the previous one, then centre the whole label
    parts = [t]
    for txt, col, fw in ((" / ", FS.INK, "normal"), (f"+{gains['proto'][1]:.1f}", HCOL["proto"], "bold"),
                         (" pp", FS.INK, "bold")):
        parts.append(ax.annotate(txt, xy=(1, 0), xycoords=parts[-1], va="bottom", ha="left", fontsize=9,
                                 fontweight=fw, color=col))
    fig_ = ax.figure
    fig_.canvas.draw()
    r = fig_.canvas.get_renderer()
    x0 = parts[0].get_window_extent(r).x0
    x1 = parts[-1].get_window_extent(r).x1
    cx = ax.transData.transform((1.5, 0))[0]
    shift = cx - (x0 + x1) / 2
    px0 = ax.transData.transform((1.5, 0))[0] + shift
    parts[0].set_x(ax.transData.inverted().transform((px0, 0))[0])
    ax.set_xticks(xs, ["Scan\nlevel", "Patient,\none scan", "Patient,\nall scans"])
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("Top-1 accuracy")
    ax.axhline(1 / 49, color=FS.MUTED, lw=0.6, ls="--")
    ax.set_title("Gain from combining scans", loc="left")


def curve(ax, col, ylab, title, invert):
    for h in HEADS:
        for kmin, ls, mfc in ((4, "-", None), (5, (0, (3, 2)), "white")):
            ms = [str(m) for m in range(1, kmin + 1)]
            v = per_run(rec[(rec["head"] == h) & (rec.n_scans >= kmin) & rec.m.isin(ms)], ["m"], col).unstack("run")
            v = v.loc[ms]
            x = np.arange(1, kmin + 1) + (0.06 if kmin == 5 else -0.06)
            ax.errorbar(x, v.mean(1), yerr=v.std(1), color=HCOL[h], ls=ls, lw=0.9, marker=HMRK[h],
                        ms=3.6 if h == "softmax" else 3.2, mfc=mfc or HCOL[h], mew=0.8, capsize=1.5,
                        elinewidth=0.6)
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("Scans combined")
    ax.set_ylabel(ylab)
    if invert:
        ax.invert_yaxis()
    ax.set_title(title, loc="left")


def panel_d(ax):
    bins = np.arange(-6.25, 6.3, 0.5)
    for h in HEADS:
        g = -pp[pp["head"] == h].d_rank.to_numpy()          # positive = true syndrome moved up
        ax.hist(np.clip(g, -6, 6), bins=bins, histtype="step", lw=1.0, color=HCOL[h],
                label=HLAB[h])
        ax.hist(np.clip(g, -6, 6), bins=bins, histtype="stepfilled", alpha=0.12, color=HCOL[h])
    ax.axvline(0, color=FS.INK, lw=0.6)
    yl = ax.get_ylim()[1]
    for k, h in enumerate(HEADS):
        g = -pp[pp["head"] == h].d_rank
        ax.text(-6.0, yl * (0.97 - 0.2 * k), f"{HLAB[h]}:\n{(g > 0).sum()} up / {(g < 0).sum()} down",
                ha="left", va="top", fontsize=6.6, color=HCOL[h])
    ax.set_xlabel("Mean change in true-syndrome rank\n(positive = improvement)")
    ax.set_ylabel(f"Patients (n = {pp.subject.nunique()})")
    ax.set_xticks([-6, -4, -2, 0, 2, 4, 6], ["≤−6", "−4", "−2", "0", "2", "4", "≥6"])
    ax.set_title("Per-patient change in diagnostic rank", loc="left")


def panel_e(ax):
    cats = [("wrong_to_right", "Top-1\ncorrected"), ("right_to_wrong", "Top-1\nlost"),
            ("rank_up", "Rank\nimproved"), ("rank_down", "Rank\nworsened")]
    w = 0.3
    xs = np.arange(len(cats))
    for k, h in enumerate(HEADS):
        x = xs + (k - 0.5) * w
        for xi, (c, _) in zip(x, cats):
            v = 100 * per_run(trans[trans["head"] == h], [], c)
            ax.bar(xi, v.mean(), w * 0.92, color=HCOL[h], alpha=0.9, label=HLAB[h] if c == "wrong_to_right" else None)
            dots(ax, xi, v.to_numpy(), HCOL[h])
            ax.text(xi, v.max() + 1.2, f"{v.mean():.1f}", ha="center", va="bottom", fontsize=6.4)
    ax.axvline(1.5, color=FS.RULE, lw=0.6)
    ax.set_xticks(xs, [c[1] for c in cats])
    ax.set_ylabel("Single scan → pair (%)")
    ax.set_ylim(0, 52)
    ax.set_title("Benefits and limitations of adding a second scan", loc="left")


CON = pd.read_csv(RESULTS / "contrast_tests_cf20fullv3.csv")


def panel_overall(ax):
    """Overall patient- minus scan-level change (all 385 patients / 1,056 scans), five models."""
    mets = [("accuracy", "Top-1\naccuracy"), ("macro_f1", "Macro-$F_1$"), ("macro_auc", "Macro AUC")]
    rng = np.random.default_rng(0)
    for i, (m, _) in enumerate(mets):
        for h, dy in (("softmax", 0.17), ("proto", -0.17)):
            r = CON[(CON.contrast == f"C2_agg_{h}[pointnet2]") & (CON.metric == m)
                    & CON.task.str.startswith("CF20_full_v3_ts")]
            assert len(r) == 5, (h, m, len(r))
            v = 100 * r.diff_mean.to_numpy()
            y = len(mets) - 1 - i + dy
            ax.plot([v.min(), v.max()], [y, y], color=HCOL[h], lw=1.1, alpha=0.55, solid_capstyle="round")
            ax.scatter(v, y + rng.uniform(-0.05, 0.05, len(v)), s=1.8, color=FS.INK, alpha=0.85, lw=0, zorder=6)
            ax.plot(v.mean(), y, marker=HMRK[h], ms=4.2 if h == "softmax" else 3.6, color=HCOL[h],
                    mec="white", mew=0.4, zorder=4)
            ax.text(v.max() + 0.35, y, f"+{v.mean():.1f}", va="center", fontsize=6.6, color=HCOL[h])
    ax.set_yticks(range(len(mets)), [l for _, l in mets][::-1])
    ax.set_ylim(-0.6, len(mets) - 0.4)
    ax.set_xlim(0, 12.5)   # left spine is the zero line
    ax.set_xlabel("Patient level − scan level (percentage points)")
    ax.grid(axis="x", color=FS.RULE, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_title("Overall effect of aggregation", loc="left")
    hh = [plt.Line2D([], [], color=HCOL[x], marker=HMRK[x], ms=3.8, lw=1.0, label=HLAB[x]) for x in HEADS]
    ax.legend(handles=hh, loc="lower right", frameon=False, fontsize=6.6, borderaxespad=0.1)


NF4 = rec[rec.n_scans >= 4].subject.nunique()
NF5 = rec[rec.n_scans >= 5].subject.nunique()

fig = plt.figure(figsize=(FS.DOUBLE, 7.5))
r1, r2, r3 = fig.subfigures(3, 1, height_ratios=[1, 1.05, 1.0])
axo, axa = r1.subplots(1, 2, width_ratios=[1.15, 1])
axb, axc = r2.subplots(1, 2)
axe, axd = r3.subplots(1, 2, width_ratios=[1.25, 1])
panel_overall(axo)
panel_a(axa)
curve(axb, "top1", "Top-1 accuracy", "Same patients: top-1 accuracy", False)
curve(axc, "mean_rank", "Rank of true syndrome\n(1 = top candidate)", "Same patients: rank of true syndrome", True)
axc.text(0.03, 0.97, "↑ better", transform=axc.transAxes, ha="left", va="top", fontsize=6.6, color=FS.MUTED)
# legend for c and d as a small key table inside panel c:
# rows = inference rule (colour, marker), columns = patient set (solid/filled vs dashed/open)
def key_table(ax, x0=0.40, y0=0.22):
    tr = ax.transAxes
    cols = {4: (x0 + 0.30, "-", None), 5: (x0 + 0.50, (0, (3, 2)), "white")}
    NK = {4: NF4, 5: NF5}
    for k, (cx, _, _) in cols.items():
        ax.text(cx, y0, f"≥{k} scans\n(n = {NK[k]})", transform=tr, ha="center", va="bottom",
                fontsize=6.6, color=FS.INK, linespacing=1.0)
    for i, h in enumerate(HEADS):
        y = y0 - 0.06 - 0.085 * i
        ax.text(x0 + 0.19, y, HLAB[h], transform=tr, ha="right", va="center", fontsize=6.8, color=HCOL[h])
        for k, (cx, ls, mfc) in cols.items():
            ax.plot([cx - 0.065, cx + 0.065], [y, y], transform=tr, color=HCOL[h], ls=ls, lw=1.0,
                    clip_on=False)
            ax.plot([cx], [y], transform=tr, color=HCOL[h], marker=HMRK[h], ms=3.6, mfc=mfc or HCOL[h],
                    mew=0.8, clip_on=False)


key_table(axb)
panel_e(axe)
panel_d(axd)
for ax, l, dx in ((axo, "a", -0.22), (axa, "b", -0.26), (axb, "c", -0.20), (axc, "d", -0.24),
                  (axe, "e", -0.15), (axd, "f", -0.20)):
    ax.text(dx, 1.06, l, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="left", va="bottom")
for ext in ("png", "pdf"):
    fig.savefig(OUTD / f"fig_aggregation.{ext}")
print("wrote", OUTD / "fig_aggregation.png")
