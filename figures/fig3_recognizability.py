#!/usr/bin/env python3
"""
Paper Figure 4 — per-diagnosis subject-level F1 vs number of test subjects
(49-class task, PointNet++, prototype head, five training seeds). Spearman rho and
its interval come from facebench/evaluation/recognizability.py. Writes figures/output/fig3_recognizability.png.
The x axis is linear (class sizes span 3-25 test subjects); dots that would
coincide are nudged sideways for display only (dodge()).

Design notes from earlier revisions (19/33-class tasks, log axis), kept for context:

REPLACES figures/fig_recognizability.png, which was held pending the class-size
correlation conflict. That conflict is resolved (recognizability.py): the poster
and results_analysis.md had computed DIFFERENT TASKS, and the relationship is
real at 33 classes and absent at 19.

THE FIGURE IS BUILT AROUND THAT CONTRAST, because it is the finding. A single
panel showing only the 33-class scatter would reproduce the poster's claim and
hide the thing that makes it interesting — that regrouping the same scans into
pathway-level categories destroys the relationship.

WHY LOG X. Class sizes span 1 to ~180 test subjects. On a linear axis the whole
long tail collapses against the origin and the eye reads a relationship driven
entirely by the two largest classes.

LABELLED POINTS ARE CHOSEN BY RULE, NOT BY EYE: the extremes of F1 within the
matched-size band the text quotes, plus the conditions the text names. Labelling
whichever points look tidy is how a scatter ends up arguing for a story the data
does not carry.

Usage: python3 fig3_recognizability.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import matplotlib.pyplot as plt

from figures import figstyle as FS                                       # noqa: E402
from facebench.evaluation import recognizability as R                                 # noqa: E402
from figures.adjust_labels import place_labels                      # noqa: E402

from facebench.paths import FIG_OUT as OUT  # noqa: E402

# Conditions the Results text names, so the figure and the prose agree.
# The named conditions are the ones the Results text cites --
# the extremes of F1 among classes with 3-6 test subjects, plus Down syndrome
# and the three RASopathies.
NAMED = ["Down Syndrome", "Achondroplasia", "Rhizomelic", "Coffin-Lowry",
         "Cockayne", "Noonan", "Costello", "Cardiofacio", "Apert", "Fragile X",
         "1p36", "Kabuki", "22q"]


def dodge(size, f1, xpt=10.07, ypt=165.0, sep=4.4, step=0.22):
    """Minimal horizontal offset so no two dots cover each other. Dots are
    placed in F1 order; each stays at its true class size unless it would sit
    within `sep` pt of an already-placed dot of the same size, in which case it
    takes the smallest sideways step (0, +s, -s, +2s, ...) that clears.
    xpt/ypt = points per data unit at the authored size. Class size is an
    integer count, so the sub-unit shift cannot be read as a different n.
    Display only -- rho uses the true sizes.
    (2026-11-02: Apert and Smith-Lemli-Opitz sat on the identical point.)"""
    size = np.asarray(size, float); f1 = np.asarray(f1, float)
    xp = size.copy()
    placed = []          # all dots so far, any class size: a shifted dot must
    for x in np.unique(size):   # also clear the neighbouring column
        for j in np.where(size == x)[0][np.argsort(f1[size == x], kind="stable")]:
            for k in [0, 1, -1, 2, -2, 3, -3, 4, -4]:
                cx = x + k * step
                if all(np.hypot((cx - px) * xpt, (f1[j] - py) * ypt) >= sep
                       for px, py in placed):
                    break
            xp[j] = cx
            placed.append((cx, f1[j]))
    return xp


def panel(ax, size, f1, names, rho, lo, hi, title, label_named):
    size = dodge(size, f1)
    ax.scatter(size, f1, s=14, facecolor=FS.ENC_COLOR["pointnet2"],
               edgecolor="white", linewidth=0.4, zorder=3)
    # LINEAR X (2026-11-02, collaborator review): v3 class sizes span only
    # 3-25 test subjects (<1 decade), so a log axis bought nothing and its
    # "(log)" label was read as log10 values (e.g. "10^19 Down patients").
    ax.set_ylim(-0.16, 0.92)
    ax.set_xlabel("Number of test subjects per diagnosis")
    ax.grid(color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=4)
    # Only \rho goes in mathtext: a leading "+" inside $...$ is typeset as a
    # BINARY operator and renders as "= + 0.60".
    ax.annotate(rf"$\rho$ = {rho:+.2f}" "\n" rf"[{lo:+.2f}, {hi:+.2f}]",
                xy=(0.97, 0.97), xycoords="axes fraction",
                va="top", ha="right", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=FS.RULE, lw=0.4))
    if not label_named:
        return
    pts = []
    for i, nm in enumerate(names):
        if any(k.lower() in nm.lower() for k in NAMED):
            short = (nm.replace(" Syndrome", "").replace(" syndrome", "")
                       .replace("Cardiofaciocutaneous", "CFC")
                       .replace("Rhizomelic Chondrodysplasia Punctata", "RCDP")
                       .replace(" Deletion", " del."))
            pts.append((size[i], f1[i], short[:16]))
    # Hand-set offsets (points) with leader lines: 49 points on a log axis
    # leave the auto-placer no free slot for several labels, and a label
    # floating between two dots is worse than no label.
    # Offsets found by a collision search (2026-11-02): text clears every dot
    # by >=3.5 pt, no label overlaps another, leaders clear non-target dots.
    # (dx, dy, ha) in points. Found by a collision search on the saved layout
    # (2026-11-02): every label clears every dot by >=3.5 pt, no two labels
    # touch, and no leader line crosses a label or a non-target dot.
    OFF = {"1p36 del.": (6, 0, "left"), "Apert": (0, -9, "left"),
           "Fragile X": (0, -21, "center"), "Kabuki": (-1, -9, "center"),
           "Costello": (-7, 3, "right"), "CFC": (11, 0, "left"),
           "Noonan": (2, -9, "left"), "22q 11.2 del.": (14, 6, "left"),
           "Achondroplasia": (-7, -3, "right"), "Cockayne": (-7, 0, "right"),
           "RCDP": (-7, 3, "right"), "Coffin-Lowry": (5, -12, "center"),
           "Down": (-7, 0, "right")}
    # Apert and Fragile X share x (4 test subjects). On v1 both had F1 = 0 and
    # merge into one label; on v3 Fragile X is 0.02, so they are separate points
    # and Fragile X's label is lifted clear of Apert's.
    merged = {}
    for x, y, t in pts:
        merged.setdefault((x, y), []).append(t)
    pts = [(x, y, ", ".join(ts)) for (x, y), ts in merged.items()]
    for x, y, t in pts:
        dx, dy, *ha = OFF.get(t.split(', ')[0], (8, 6))
        ha = ha[0] if ha else ("left" if dx >= 0 else "right")
        ax.scatter([x], [y], s=26, facecolor="none", edgecolor=FS.ACCENT,
                   linewidth=0.7, zorder=4)
        ax.annotate(t, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.0, color=FS.MUTED, zorder=5,
                    ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.35, color=FS.MUTED,
                                    shrinkA=0, shrinkB=2.5))


def main():
    FS.apply()
    fig, ax = plt.subplots(1, 1, figsize=(FS.DOUBLE * 0.62, 2.9))
    size, f1, names = R.load("CF20_full_v3", "subject")
    rho, lo, hi = R.rho_ci(size, f1)
    panel(ax, size, f1, names, rho, lo, hi, "Syndrome diagnosis ($K$=49)", True)
    ax.set_xlim(1.5, max(size) + 2)
    ax.set_xticks([3, 5, 10, 15, 20, 25])
    ax.set_ylabel("per-class $F_1$")
    OUT.mkdir(exist_ok=True)
    FS.finish(fig, OUT / "fig3_recognizability.png", FS.DOUBLE * 0.62)


if __name__ == "__main__":
    main()
