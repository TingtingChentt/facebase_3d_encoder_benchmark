"""
Minimal deterministic label placement for scatter panels.

WHY NOT adjustText. It is not installed in the `fasebase` env and it is
stochastic — two runs of the same script can produce different figures, which
breaks the project's rule that every artefact regenerates identically.

The algorithm is a small greedy one: try eight candidate offsets around each
point in a fixed order, take the first that collides with neither an already
placed label nor a data point. Deterministic given the input order, so the
figure is reproducible.
"""
import numpy as np

# (dx, dy) in points, tried in this order. Right and above first, since those
# read most naturally, then around the clock.
CANDIDATES = [(5, 3), (5, -6), (-5, 3), (-5, -6), (5, 9), (-5, 9),
              (0, 8), (0, -11)]


def _bbox(ax, x, y, dx, dy, text, fontsize):
    """Approximate label bbox in DISPLAY coords, from a character-width
    estimate. Exact extents would need a draw pass per candidate; the estimate
    is enough to keep labels off each other at this density."""
    px, py = ax.transData.transform((x, y))
    w = 0.55 * fontsize * len(text)
    h = 1.15 * fontsize
    x0 = px + dx if dx >= 0 else px + dx - w
    y0 = py + dy
    return (x0, y0, x0 + w, y0 + h)


def _overlap(a, b, pad=1.5):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def place_labels(ax, points, fontsize=6.0, color=None):
    """points: iterable of (x, y, text) in DATA coords. Draws each label at the
    first non-colliding candidate offset; skips any label with no free slot
    rather than stacking text on text."""
    from figstyle import MUTED
    color = color or MUTED
    ax.figure.canvas.draw()
    placed = []
    pt_boxes = [(*ax.transData.transform((x, y)),) for x, y, _ in points]
    pt_boxes = [(p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3) for p in pt_boxes]
    n_skipped = 0
    for (x, y, text) in points:
        for dx, dy in CANDIDATES:
            bb = _bbox(ax, x, y, dx, dy, text, fontsize)
            if any(_overlap(bb, o) for o in placed):
                continue
            if any(_overlap(bb, o, pad=0.5) for o in pt_boxes):
                continue
            ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                        textcoords="offset points", fontsize=fontsize,
                        color=color, zorder=5,
                        ha="left" if dx >= 0 else "right")
            placed.append(bb)
            break
        else:
            n_skipped += 1
    if n_skipped:
        print(f"    note: {n_skipped} label(s) had no free slot and were "
              f"omitted rather than overplotted")
    return len(placed)
