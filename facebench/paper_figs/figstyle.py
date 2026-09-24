"""
Shared style for the PAPER figures (npj Digital Medicine).
thread facebase3d-paper-2026-08-08.

WHY A SEPARATE MODULE FROM scripts/plot_*.py. Those scripts build POSTER
figures: 48x36 in board, type sized for viewing at two metres, text baked into
the PNG. Reusing them for the paper is what left Fig. 2 rendering at ~6 pt in
the manuscript. Paper figures are sized in real column inches and their type is
specified in points that mean what they say at 100% scale.

RULES THIS MODULE ENFORCES
  * Figures are authored at their FINAL printed width (SINGLE / DOUBLE below)
    and included at that width with no LaTeX rescaling. \\includegraphics with a
    width= that differs from the authored width silently rescales the type.
  * Minimum effective type size is 7 pt. npj's floor is 5-6 pt for panel labels;
    7 pt leaves margin for a reduction at proof.
  * Colour is never the only channel carrying information — the encoder palette
    is also ordered, and greyscale-safe in luminance.
  * Chance/reference levels are drawn, not described, on every axis where a
    reader could mistake a bar for a result.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

# Widths in inches. DOUBLE is this document's ACTUAL \textwidth (measured from
# main.log: 469.755 pt / 72.27 = 6.500 in, i.e. letterpaper with 1 in margins),
# NOT a generic journal column spec. Authoring at 7.2 in and including at
# \textwidth would rescale every figure by 0.90 and take 8 pt type down to 7.2.
# If the class or margins change, re-measure and change this number.
SINGLE = 3.20
DOUBLE = 6.50

# Encoder palette. Ordered light -> dark so the series stays readable in
# greyscale; PointNet++ carries the strongest value because it is the encoder
# the text follows.
ENCODERS = ["geommlp", "pointnet", "dgcnn", "pointnet2"]
ENC_LABEL = {"geommlp": "GeomMLP", "pointnet": "PointNet",
             "dgcnn": "DGCNN", "pointnet2": "PointNet++"}
ENC_COLOR = {"geommlp": "#BFD3E6", "pointnet": "#7FA8CC",
             "dgcnn": "#3E6E9E", "pointnet2": "#123F63"}

INK = "#1A1A1A"
MUTED = "#6B6B6B"
RULE = "#B0B0B0"
ACCENT = "#B03A2E"      # chance lines, warnings — the only warm colour used


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        # NO bbox="tight": it trims to content and silently changes the saved
        # width, which defeats authoring at the final printed width. Layout is
        # handled by constrained_layout instead.
        "savefig.pad_inches": 0.01,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
    })


def chance_line(ax, y, label="chance", x=1.0):
    """Draw a reference level. Always drawn, never left to the caption."""
    ax.axhline(y, color=ACCENT, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.annotate(label, xy=(x, y), xycoords=("axes fraction", "data"),
                xytext=(2, 0), textcoords="offset points",
                va="center", ha="left", fontsize=6.5, color=ACCENT)


def finish(fig, path, width_in):
    """Save at the authored width. The caller must include the figure at this
    same width in LaTeX — see the module docstring. Saved width is asserted, not
    assumed, because a stray bbox setting silently rescales the type."""
    fig.set_size_inches(width_in, fig.get_size_inches()[1])
    fig.canvas.draw()
    _assert_inside_canvas(fig, path)
    fig.savefig(path)
    w_px = fig.get_size_inches()[0] * fig.dpi
    plt.close(fig)
    from PIL import Image
    got = Image.open(path).size[0]
    if abs(got - w_px) > 2:
        raise SystemExit(
            f"{path}: saved {got}px wide, authored {w_px:.0f}px. Something "
            f"rescaled the canvas (bbox='tight'?) — type size in the PDF will "
            f"not match the point sizes set here.")
    print(f"  wrote {path}  ({width_in:.2f} in, {got} px @ {fig.dpi:.0f} dpi)")


def _visible_ticklabels(ax, which):
    lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
    lo, hi = min(lo, hi), max(lo, hi)
    locs = ax.get_xticks() if which == "x" else ax.get_yticks()
    labs = ax.get_xticklabels() if which == "x" else ax.get_yticklabels()
    eps = 1e-9 * max(1.0, abs(hi - lo))
    return [t for loc, t in zip(locs, labs) if lo - eps <= loc <= hi + eps]


def _assert_inside_canvas(fig, path, slack=1.0):
    """Fail loudly if any text runs off the canvas.

    Added after a panel title on the right-hand column was silently clipped at
    the figure edge: constrained_layout sizes the axes, but a title set with
    loc="left" is NOT constrained to the axes width, so it can overflow with no
    warning and no visible error anywhere in the build. Every figure in this
    directory goes through finish(), so this check covers all of them.
    """
    W = fig.get_size_inches()[0] * fig.dpi
    H = fig.get_size_inches()[1] * fig.dpi
    # An explicit renderer is REQUIRED. Without one, Text.get_window_extent()
    # returns a zero-width bbox for titles on constrained-layout axes, and the
    # first version of this guard silently passed a title that was visibly
    # clipped in the PNG.
    rend = fig.canvas.get_renderer()
    bad = []
    for ax in fig.get_axes():
        # Titles and free-standing text are drawn even on an axis("off") panel;
        # tick and axis labels are not, and their stale extents would otherwise
        # report as false positives on every schematic panel.
        # loc="left"/"right" titles are SEPARATE Text artists (_left_title /
        # _right_title); ax.title holds only the centred one and is empty when
        # loc is not "center". Checking ax.title alone is why the first version
        # of this guard passed a clipped left-aligned title.
        titles = [ax.title,
                  getattr(ax, "_left_title", None),
                  getattr(ax, "_right_title", None)]
        items = [t for t in titles if t is not None] + list(ax.texts)
        if getattr(ax, "axison", True):
            items += [ax.xaxis.label, ax.yaxis.label]
            # Matplotlib keeps Text artists for ticks OUTSIDE the view limits.
            # They are never drawn, but they report real extents far off-canvas,
            # so including them buries the true positives in noise. Keep only
            # labels whose tick actually falls inside the axis limits.
            items += _visible_ticklabels(ax, "x") + _visible_ticklabels(ax, "y")
        leg = ax.get_legend()
        if leg is not None:
            items.append(leg)
        for t in items:
            if not getattr(t, "get_visible", lambda: True)():
                continue
            try:
                bb = t.get_window_extent(renderer=rend)
            except TypeError:
                bb = t.get_window_extent()
            except Exception:
                continue
            if bb.width == 0 and bb.height == 0:
                continue
            txt = getattr(t, "get_text", lambda: type(t).__name__)()
            if bb.x1 > W + slack or bb.x0 < -slack or bb.y1 > H + slack \
                    or bb.y0 < -slack:
                bad.append(f"    {str(txt)[:52]!r} bbox=({bb.x0:.0f},{bb.y0:.0f})"
                           f"-({bb.x1:.0f},{bb.y1:.0f})")
    if bad:
        raise SystemExit(
            f"{path}: {len(bad)} text item(s) fall outside the "
            f"{W:.0f}x{H:.0f} px canvas and would be clipped:\n"
            + "\n".join(sorted(set(bad))))
