#!/usr/bin/env python3
"""
Paper Figure 6 (Methods): the four encoders.

Top row: how each encoder reads the 4,096-point cloud, drawn on one real test
scan (the Down syndrome exemplar of Fig. 5a). These four panel renders
(figures/src/enc_<name>.png, made by fig_encoders_renders.py) come from a FaceBase scan and are not
redistributed; the composed figure is in figures/.
Bottom row: block diagrams. Layer sizes are checked against
facebench/models/{geom_mlp,pointnet,pointnet2,dgcnn}.py and the Methods text.
  figures/fig_encoders.{png,pdf}   (authored at DOUBLE width, include at \\textwidth)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figures import figstyle as FS                                       # noqa: E402

from facebench.paths import FIG_OUT, ROOT as PROJECT  # noqa: E402
IMG = PROJECT / "figures/src"          # enc_<name>.png renders, not shipped (see docs/RESULTS_MAP.md)
OUT = FIG_OUT
NAVY, RED, GREY = "#143D64", FS.ACCENT, FS.MUTED

# Rows between the (aligned) input row and the (aligned) embedding row.
# (text, kind): io = input/output, fixed = hand-crafted (not learned),
# learn = learned layer, emb = embedding used for prototype inference.
COLS = [
    ("geommlp", "GeomMLP", "Global shape summary", "0.11 M parameters",
     [("26 shape descriptors\n(hand-crafted)", "fixed"),
      ("MLP 256–256", "learn")],
     "128-d embedding"),
    ("pointnet", "PointNet", "Each point independently", "3.48 M parameters",
     [("Input transform (3×3)\nshared MLP 64–64", "learn"),
      ("Feature transform (64×64)\nshared MLP 64–128–1,024", "learn"),
      ("Max pool → 1,024-d", "learn"),
      ("FC 512 → FC 256", "learn")],
     "256-d embedding"),
    ("pointnet2", "PointNet++", "Local balls at three scales", "1.74 M parameters",
     [("SA1: 512 centroids\nr = 0.1 / 0.2 / 0.4", "learn"),
      ("SA2: 128 centroids\nr = 0.2 / 0.4 / 0.8", "learn"),
      ("Shared MLP 256–512–1,024\nmax pool → 1,024-d", "learn"),
      ("FC 512 → FC 256", "learn")],
     "256-d embedding"),
    ("dgcnn", "DGCNN", "k-nearest-neighbour graph", "1.81 M parameters",
     [("4 × EdgeConv, k = 20\n64–64–128–256", "learn"),
      ("Concat 512 → 1,024", "learn"),
      ("Avg + max pool → 2,048-d", "learn"),
      ("FC 512 → FC 256", "learn")],
     "256-d embedding"),
]
STYLE = {"io": dict(fc="white", ec=GREY, ls="-", tc=FS.INK, bold=False),
         "fixed": dict(fc="white", ec=NAVY, ls="--", tc=NAVY, bold=False),
         "learn": dict(fc=NAVY, ec=NAVY, ls="-", tc="white", bold=False),
         "emb": dict(fc="white", ec=RED, ls="-", tc=RED, bold=True)}

W, H = FS.DOUBLE, 5.05
CW = W / 4                      # column width
BW = CW - 0.14                  # box width
LINE = 0.135                    # height per text line in a box
FS_BOX = 7

# vertical layout, inches from the bottom
Y_TITLE = H - 0.12
Y_SUB = H - 0.29
FACE_TOP, FACE_H = H - 0.38, 1.30
Y_INPUT = FACE_TOP - FACE_H - 0.20      # centre of the input row
Y_EMB = 0.98                            # centre of the embedding row
Y_CLS = 0.66
Y_PARAM = 0.42
Y_LEG = 0.13


def box_h(text):
    return 0.07 + LINE * (text.count("\n") + 1)


def draw_box(ax, xc, yc, text, kind):
    st, h = STYLE[kind], box_h(text)
    ax.add_patch(FancyBboxPatch((xc - BW / 2, yc - h / 2), BW, h,
                                boxstyle="round,pad=0,rounding_size=0.04",
                                fc=st["fc"], ec=st["ec"], ls=st["ls"], lw=0.8))
    ax.text(xc, yc, text, ha="center", va="center", fontsize=FS_BOX, color=st["tc"],
            fontweight="bold" if st["bold"] else "normal", linespacing=1.15)
    return h


def arrow(ax, xc, y0, y1):
    ax.annotate("", (xc, y1), (xc, y0),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.7,
                                mutation_scale=6, shrinkA=0, shrinkB=0))


def column(fig, ax, i, key, name, sub, params, middle, emb):
    x0 = i * CW
    xc = x0 + CW / 2
    ax.text(x0 + 0.04, Y_TITLE, "abcd"[i], fontsize=9, fontweight="bold", va="center")
    ax.text(xc, Y_TITLE, name, fontsize=8.5, fontweight="bold", ha="center", va="center")
    ax.text(xc, Y_SUB, sub, fontsize=7, color=GREY, ha="center", va="center")

    im = plt.imread(IMG / f"enc_{key}.png")
    fw = FACE_H * im.shape[1] / im.shape[0]
    iax = fig.add_axes([(xc - fw / 2) / W, (FACE_TOP - FACE_H) / H, fw / W, FACE_H / H])
    iax.imshow(im, interpolation="lanczos"); iax.axis("off")

    rows = [("4,096 × 3 points", "io")] + middle + [(emb, "emb"), ("Linear → classes", "io")]
    hs = [box_h(t) for t, _ in rows]
    # input and the last two rows are pinned; the middle rows share the gap evenly
    ys = [Y_INPUT]
    top, bot = Y_INPUT - hs[0] / 2, Y_EMB + hs[-2] / 2
    gap = (top - bot - sum(hs[1:-2])) / (len(middle) + 1)
    y = top
    for h in hs[1:-2]:
        y -= gap + h / 2; ys.append(y); y -= h / 2
    ys += [Y_EMB, Y_CLS]
    for (t, k), yc in zip(rows, ys):
        draw_box(ax, xc, yc, t, k)
    for j in range(len(rows) - 1):
        arrow(ax, xc, ys[j] - hs[j] / 2, ys[j + 1] + hs[j + 1] / 2)
    ax.text(xc, Y_PARAM, params, fontsize=7, color=GREY, ha="center", va="center")


def legend(ax):
    items = [("io", "Input / output"), ("fixed", "Hand-crafted, not learned"),
             ("learn", "Learned layers"), ("emb", "Embedding for prototype inference")]
    widths = [1.1, 1.8, 1.15, 2.05]
    x = (W - sum(widths)) / 2
    for (k, label), w in zip(items, widths):
        st = STYLE[k]
        ax.add_patch(FancyBboxPatch((x, Y_LEG - 0.06), 0.22, 0.12,
                                    boxstyle="round,pad=0,rounding_size=0.025",
                                    fc=st["fc"], ec=st["ec"], ls=st["ls"], lw=0.8))
        ax.text(x + 0.29, Y_LEG, label, fontsize=7, va="center")
        x += w


def main():
    FS.apply()
    plt.rcParams["figure.constrained_layout.use"] = False   # absolute layout in inches
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    for i, c in enumerate(COLS):
        column(fig, ax, i, *c)
    for i in range(1, 4):                                    # thin column rules
        ax.plot([i * CW] * 2, [Y_PARAM - 0.08, H - 0.05], color=FS.RULE, lw=0.4)
    legend(ax)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_encoders.png")
    fig.savefig(OUT / "fig_encoders.pdf")
    plt.close(fig)
    print("wrote", OUT / "fig_encoders.png")


if __name__ == "__main__":
    main()
