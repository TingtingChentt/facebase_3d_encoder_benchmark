#!/usr/bin/env python3
"""
Figure 3 — what makes a syndrome recognizable.
thread facebase3d-paper-2026-08-08.

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

import numpy as np
import matplotlib.pyplot as plt
from adjust_labels import place_labels

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))
import figstyle as FS                                       # noqa: E402
import recognizability as R                                 # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "figures"

# Conditions the Results text names, so the figure and the prose agree.
NAMED = ["Achondroplasia", "Down Syndrome", "Cockayne", "Cohen",
         "CHARGE", "Jacobsen", "Rett", "Noonan", "Costello", "CFC"]


def panel(ax, size, f1, names, rho, lo, hi, title, label_named):
    ax.scatter(size, f1, s=14, facecolor=FS.ENC_COLOR["pointnet2"],
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.7, max(size) * 1.6)
    ax.set_ylim(-0.04, 0.92)
    ax.set_xlabel("test subjects in class (log)")
    ax.grid(color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title, pad=4)
    # Only \rho goes in mathtext: a leading "+" inside $...$ is typeset as a
    # BINARY operator and renders as "= + 0.60".
    ax.annotate(rf"$\rho$ = {rho:+.2f}" "\n" rf"[{lo:+.2f}, {hi:+.2f}]",
                xy=(0.03, 0.97), xycoords="axes fraction",
                va="top", ha="left", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=FS.RULE, lw=0.4))
    if not label_named:
        return
    pts = []
    for i, nm in enumerate(names):
        if any(k.lower() in nm.lower() for k in NAMED):
            short = (nm.replace(" Syndrome", "").replace(" syndrome", "")
                       .replace("Cardiofaciocutaneous", "CFC"))
            pts.append((size[i], f1[i], short[:16]))
    place_labels(ax, pts)


def main():
    FS.apply()
    fig, axes = plt.subplots(1, 2, figsize=(FS.DOUBLE, 2.7), sharey=True)

    for ax, (task, title, lab) in zip(axes, [
            ("B2_corrected", "Clinical diagnosis ($K$=33)", True),
            ("B1_corrected", "Syndrome category ($K$=19)", False)]):
        size, f1, names = R.load(task, "subject")
        rho, lo, hi = R.rho_ci(size, f1)
        panel(ax, size, f1, names, rho, lo, hi, title, lab)

    axes[0].set_ylabel("per-class $F_1$")
    OUT.mkdir(exist_ok=True)
    FS.finish(fig, OUT / "fig3_recognizability.png", FS.DOUBLE)


if __name__ == "__main__":
    main()
