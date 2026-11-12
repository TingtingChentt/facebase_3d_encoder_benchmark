#!/usr/bin/env python3
"""
Supplementary Fig. S1 — study overview and 3D facial analysis framework.


WHY. Results opened by describing six datasets, five tasks and four encoders in
prose, and a reader had to hold all three in their head before the first result.
This figure carries the setup so section 1 can describe it rather than enumerate
it.

FIVE PANELS, answering the questions a reader has in the order they arise:
  (a) WHY REGISTRATION IS NOT NEUTRAL. A real 22q11.2 scan morphed toward the
      control template, coloured by deviation: the mean deviation that carries
      the diagnostic signal falls from 3.6 mm to 1.9 mm. This is the paper's
      motivating claim, and it was previously asserted in prose only.
  (b) WHO THE PEOPLE ARE. Self-reported ancestry per cohort, joined from the
      FaceBase source tables to the subjects we actually use. The three control
      cohorts ship no ancestry at all, which is drawn rather than omitted: the
      negative class of every screening task has unknown composition.
  (c) WHAT DATA. Every dataset with its true scan and subject counts, coloured by
      the role it plays. The scan/subject distinction is drawn, not stated,
      because it is the fact the whole leakage result turns on: FB-TJ0 is by far
      the largest cohort in scans and nothing like the largest in subjects.
      SITS BESIDE (d) DELIBERATELY: the two share a colour code (the three
      roles), so they are adjacent rather than separated by the ancestry ramp.
  (d) HOW THEY ARE USED. Dataset x task incidence, tasks on y.

  (b), (c) and (d) SIT IN ONE ROW but are three INDEPENDENT panels: each has its
  own x and y axes, its own tick labels and its own key. They were briefly built
  to share one set of dataset rows, and were separated again — a shared axis
  invites reading a neighbour as a continuation of the same plot, and here the
  units differ (scans, percent of subjects, incidence).

  ONE ROW IS ALSO A HARD CONSTRAINT, not a preference. On two rows the figure
  stood 8.10 in tall, the float was 217 pt taller than the text block, and LaTeX
  both banished it to the back matter AND cut its caption off the page
  mid-sentence — panels (c), (d) and (e) were described nowhere in the rendered
  PDF, which nothing in `make check` catches. One row brings it to 4.86 in,
  which fits. Anything that grows H has to be checked against
  `Float too large for page` in paper/main.log.

  (e) WHAT THE MODEL DOES. One real scan through the pipeline.

EVERY NUMBER IS READ FROM THE MANIFESTS AT BUILD TIME. Nothing is typed in, so
this figure cannot drift from Table 1 the way a hand-drawn schematic would.

The face in (e) is one real FB-TJ0 scan (the local exemplar list,
figures/exemplars.py) rendered from its FaceBase mesh and from its stored
4,096-point cloud — the actual model input, not an illustration. Panel (a) shows
renders made by figures/figS_overview_registration.py.

Usage: python figures/figS_overview.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figures import figstyle as FS                                       # noqa: E402

from facebench.paths import (FACEBASE_ROOT as FBSRC, FIG_OUT as OUT, MANIFESTS,  # noqa: E402
                             PROCESSED, ROOT, SPLITS, facebase)

FACE_Y_CUT = -0.19          # neck; see panel_pipeline for the measurement
REG = ROOT / "figures/src"  # fig_regp_* renders from figs/figS_overview_registration.py (local)
ROLE_OF = {"FB-5A": "cleft", "FB-56": "cleft", "FB-TJ0": "syndrome",
           "FB-TK0": "control", "FB-VWP": "control", "FB-TX4": "control"}


def _exemplar_scan():
    from figures.exemplars import exemplar
    return exemplar("figS_overview", "scan")

# Cleft-cohort Race codes. NOT guessed — decoded by cross-tabulating the Race
# column against the RaceWhite/RaceBlack/... indicator flags in
# OFC2_DemoPart1Exported.csv, which agree exactly. Multi-letter codes have two
# indicators set and are therefore "more than one".
RACE1 = {"w": "White", "a": "Asian", "b": "Black or African American",
         "n": "Other or more than one", "o": "Other or more than one",
         "u": "Unknown / not reported"}
ANC_ORDER = ["White", "Asian", "Black or African American",
             "Other or more than one", "Unknown / not reported"]
ANC_COLOR = {"White": "#123F63", "Asian": "#3E6E9E",
             "Black or African American": "#7FA8CC",
             "Other or more than one": "#BFD3E6",
             "Unknown / not reported": "#E4E9EE"}

ROLE_COLOR = {"cleft": "#123F63", "syndrome": "#3E6E9E", "control": "#9EC1DC"}
# Title pad, in points, shared by (b), (c) and (d) so their titles align. Sized
# by the tallest key in the row — (c)'s five ancestry categories in two columns.
TITLE_PAD = 3 * (4.8 + 2.4) + 3.5
TASKS = [
    ("Cleft type",        "$K$=4",  ["FB-5A", "FB-56"]),
    ("Cleft screening",   "$K$=2",  ["FB-5A", "FB-56"]),
    ("Syndrome diagnosis", "$K$=49", ["FB-TJ0"]),
    ("Combined screening", "$K$=2",
     ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]),
]


def cohort_table():
    """Scans and subjects per dataset (Table 1), from data/splits/scans.csv."""
    sc = pd.read_csv(SPLITS / "scans.csv")
    rows = [(ds, len(g), g.subject_id.nunique(), ROLE_OF[ds]) for ds, g in sc.groupby("dataset")]
    df = pd.DataFrame(rows, columns=["dataset", "scans", "subjects", "role"])
    return df.sort_values("scans", ascending=True).reset_index(drop=True)


def task_sizes():
    """Scans entering each task, after the class floors."""
    n = {}
    n["Cleft type"] = len(pd.read_csv(SPLITS / "A1_fix_v3.csv"))
    n["Cleft screening"] = len(pd.read_csv(SPLITS / "A2_fix_v3.csv"))
    n["Syndrome diagnosis"] = len(pd.read_csv(SPLITS / "CF20_full_v3.csv"))
    n["Combined screening"] = len(pd.read_csv(SPLITS / "C_full_v3.csv"))
    return n


def ancestry():
    """Self-reported ancestry for the subjects WE USE, joined from the FaceBase
    source tables. Returns {dataset: {category: n}} plus a set of datasets that
    ship no ancestry at all.

    The join is against our manifests, not the whole release, so this describes
    the study cohort rather than the repository. All three joins match 100% of
    our subjects; if that ever stops being true this function raises rather than
    silently plotting a biased subset."""
    rd = dict(low_memory=False, encoding="latin-1")
    sc = pd.read_csv(SPLITS / "scans.csv")
    sc["sid"] = sc.subject_id.str.split("_", n=1).str[1]
    src = {
        "FB-5A": (sc[sc.dataset == "FB-5A"], "sid",
                  pd.read_csv(FBSRC / "FB-5A_files/Data/OFC1_Demographics.csv", **rd),
                  "StudyID", "code"),
        "FB-56": (sc[sc.dataset == "FB-56"], "sid",
                  pd.read_csv(FBSRC / "FB-56_files/Data/OFC2_DemoPart1Exported.csv", **rd),
                  "StudyID", "code"),
        "FB-TJ0": (sc[sc.dataset == "FB-TJ0"], "sid",
                   pd.read_csv(FBSRC / "FB-TJ0_files/FB00000861_metadata_2020-09-14.csv", **rd),
                   "FBID", "text"),
    }
    out = {}
    for ds, (man, key, tab, skey, kind) in src.items():
        subj = set(man[key].astype(str))
        tab = tab.copy()
        tab[skey] = tab[skey].astype(str)
        m = tab[tab[skey].isin(subj)].drop_duplicates(skey)
        if len(m) != len(subj):
            raise SystemExit(f"{ds}: joined {len(m)} of {len(subj)} subjects — "
                             f"the ancestry panel would describe a subset")
        if kind == "code":
            cat = m.Race.astype(str).map(
                lambda c: RACE1.get(c.strip().lower(),
                                    "Other or more than one"
                                    if len(c.strip()) > 1 else
                                    "Unknown / not reported"))
        else:
            def norm(x):
                x = str(x).strip().lower()
                if x.startswith("white"):
                    return "White"
                if x.startswith("asian"):
                    return "Asian"
                if x.startswith("black"):
                    return "Black or African American"
                if "unknown" in x or x in ("nan", ""):
                    return "Unknown / not reported"
                return "Other or more than one"
            cat = m.Race.map(norm)
        out[ds] = cat.value_counts().to_dict()
    # Verified 2026-08-10: FB-TK0 ships no metadata file at all; FB-TX4 ships a
    # sample-to-subject mapping only; FB-VWP ships sex only. None carries
    # ancestry, so the negative class of every screening task is uncharacterised.
    return out, ["FB-TK0", "FB-VWP", "FB-TX4"]


def key_above(ax, handles, ncol, fontsize=4.8):
    """Put a panel's key between its title and its axes, and reserve the room by
    padding the title rather than by asking constrained_layout for it.

    Each panel keys itself here — a single figure-level key floating between
    panels does not say which panel it belongs to, and inside the axes it
    overprints the data as soon as the panel loses height. in_layout is off
    because the title pad already accounts for the space; leaving both on makes
    constrained_layout reserve it twice."""
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.0),
                    ncol=ncol, frameon=False, fontsize=fontsize,
                    handlelength=0.68, handletextpad=0.22,
                    columnspacing=0.35, labelspacing=0.25, borderaxespad=0.0)
    leg.set_in_layout(False)
    rows = -(-len(handles) // ncol)
    # Never less than TITLE_PAD, so all three panel titles in the row sit on one
    # baseline. Padding each title by exactly its own key's height put them at
    # three different heights, which reads as three misaligned panels rather
    # than one row.
    return max(TITLE_PAD, rows * (fontsize + 2.4) + 3.5)


def centre_titles(axes):
    """Centre each panel title over its own column — axes PLUS its y tick labels.

    Measured, not guessed. A title set loc="center" centres on the AXES, which
    is off-centre for the panel a reader sees, and by a different amount in each
    panel: (d)'s task names are far wider than (b)'s dataset codes, so the three
    titles drift apart by different distances. Must be called after a draw, when
    tick label extents are final."""
    for ax in axes:
        labs = [t for t in ax.get_yticklabels() if t.get_text()]
        ab = ax.get_window_extent()
        left = min([t.get_window_extent().x0 for t in labs], default=ab.x0)
        for t in (ax.title, getattr(ax, "_left_title", None)):
            if t is not None and t.get_text():
                t.set_ha("center")
                t.set_x(((left + ab.x1) / 2 - ab.x0) / ab.width)


LETTER_FS, TITLE_FS = 10.0, 8.5     # matches Fig 1 (9 / 7.6 pt at \textwidth;
                                    # S1 is printed at 0.9\textwidth)


def letter_titles(fig, axes_letters, gap_pt=4.0, align=None):
    """Fig 1 panel style: bold lowercase letter at the panel's column edge
    (axes plus y tick labels), bold title left-aligned right after it on the
    same baseline. Replaces the old "(x) Title" + centre_titles(). The axes
    titles are only used to RESERVE space in the layout; once the layout is
    frozen they are blanked and redrawn as figure text (matplotlib re-positions
    axes titles on every draw, so moving them does not stick). `align` maps a
    panel to another whose baseline it should share. Call after a draw with
    the layout engine frozen."""
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    align = align or {}
    base = {ax: ax._left_title.get_transform().transform(
        ax._left_title.get_position())[1] for ax, _ in axes_letters}
    for ax, letter in axes_letters:
        t = ax._left_title
        text = t.get_text()
        labs = [l for l in ax.get_yticklabels() if l.get_text() and l.get_visible()]
        ab = ax.get_window_extent(r)
        left = min([l.get_window_extent(r).x0 for l in labs], default=ab.x0)
        left = max(left, 0.012 * fig.bbox.width)
        ybase = base[align.get(ax, ax)]
        fx, fy = inv.transform((left, ybase))
        L = fig.text(fx, fy, letter, fontsize=LETTER_FS, fontweight="bold",
                     ha="left", va="baseline")
        lw = L.get_window_extent(r).width + gap_pt / 72.0 * fig.dpi
        tx, _ = inv.transform((left + lw, ybase))
        fig.text(tx, fy, text, fontsize=TITLE_FS, fontweight="bold",
                 ha="left", va="baseline")
        t.set_text("")


def grow_to_title(ax, ref_ax, H, pad_pt=4.0):
    """Give a key-less panel the vertical space the other panels spend on keys.

    (b) and (c) reserve room between title and axes for their keys; (d) has no
    key, so that room was simply empty. constrained_layout will NOT hand it over
    on its own — measured: changing (d)'s title pad moves only the title, and
    all three axes stay at identical heights — so the axes is extended upward
    explicitly and its title pad cut to match, which keeps the title on the same
    baseline as the other two while the panel itself gets taller.

    THE CALLER MUST FREEZE THE LAYOUT ENGINE FIRST. Taking this axes out of the
    layout instead — which is the obvious way to stop the next pass undoing the
    move — makes constrained_layout stop reserving its column, so its neighbours
    expand into it: measured, (c) ran 254 px into (d) and their labels
    overprinted. Freezing keeps every panel where the finished layout put it."""
    ref_baseline = ref_ax._left_title.get_window_extent().y0 / (H * ax.figure.dpi)
    pos = ax.get_position()
    y1 = ref_baseline - pad_pt / 72.0 / H
    if y1 <= pos.y1:
        return
    ax.set_position([pos.x0, pos.y0, pos.width, y1 - pos.y0])
    ax.title.set_y(1.0)
    for t in (ax.title, getattr(ax, "_left_title", None)):
        if t is not None:
            t.set_position((t.get_position()[0], 1.0))
    ax.title.set_in_layout(False)
    ax._left_title.set_in_layout(False)
    ax.set_title(ax.get_title(loc="left"), pad=pad_pt, loc="left")


def check_no_overlap(axes, slack=1.0):
    """Fail if two panels' inked extents overlap.

    Added after a change that moved one panel silently widened its neighbour
    into it — the overlap was 0.64 in and neither the canvas guard nor the key
    guard could see it, because every item was inside the canvas and inside its
    own axes. Compares tight bounding boxes, so tick labels and annotations
    count as part of the panel."""
    boxes = sorted(((ax.get_tightbbox(), ax) for ax in axes),
                   key=lambda t: t[0].x0)
    bad = []
    for (b0, a0), (b1, a1) in zip(boxes, boxes[1:]):
        if b1.x0 < b0.x1 - slack:
            bad.append(f"    {a0.get_title()!r} ends at {b0.x1:.0f} px but "
                       f"{a1.get_title()!r} starts at {b1.x0:.0f} px "
                       f"(overlap {b0.x1 - b1.x0:.0f} px)")
    if bad:
        raise SystemExit("panels overlap:\n" + "\n".join(bad))


def check_keys_fit(fig, slack=2.0):
    """Fail if any panel's key is wider than the panel it keys.

    Eyeballing this does not work: a legend anchored to the axes grows to the
    right without any error, and at these sizes an overhang of a tenth of an
    inch is invisible in the script and obvious in the PDF. Measured against the
    axes' own width, which is what "inside its own panel" means here."""
    rend = fig.canvas.get_renderer()
    bad = []
    for ax in fig.get_axes():
        leg = ax.get_legend()
        if leg is None:
            continue
        lb = leg.get_window_extent(renderer=rend)
        ab = ax.get_window_extent(renderer=rend)
        if lb.x1 > ab.x1 + slack or lb.x0 < ab.x0 - slack:
            bad.append(f"    {ax.get_title(loc='left')!r}: key spans "
                       f"{lb.x0:.0f}-{lb.x1:.0f} px, axes {ab.x0:.0f}-{ab.x1:.0f} "
                       f"(overhang {max(lb.x1 - ab.x1, ab.x0 - lb.x0):.0f} px)")
    if bad:
        raise SystemExit("key(s) wider than their own panel:\n" + "\n".join(bad))


def panel_ancestry(ax, df, anc, missing):
    """(c) Who the people are.

    Its OWN axes, including its own dataset labels: (b), (c) and (d) are three
    independent panels, not three views sharing one row. They were briefly built
    to share y — first as a merged panel, then as a single row with the names
    written once — and they were kept separate both times."""
    for yy, ds in zip(np.arange(len(df)), df.dataset):
        if ds in missing:
            ax.barh(yy, 1.0, 0.62, facecolor="white", edgecolor=FS.RULE,
                    hatch="////", linewidth=0.7, zorder=3)
            ax.annotate("no ancestry recorded", xy=(0.5, yy), ha="center",
                        va="center", fontsize=5.7, color=FS.MUTED, zorder=5)
            continue
        counts = anc[ds]
        tot = sum(counts.values())
        left = 0.0
        for c in ANC_ORDER:
            v = counts.get(c, 0)
            if not v:
                continue
            frac = v / tot
            ax.barh(yy, frac, 0.62, left=left, color=ANC_COLOR[c], zorder=3)
            if frac > 0.12:
                ax.annotate(f"{100 * frac:.0f}", xy=(left + frac / 2, yy),
                            ha="center", va="center", fontsize=5.8, zorder=5,
                            color="white" if c in ANC_ORDER[:2] else FS.INK)
            left += frac
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.dataset)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "50", "100%"])
    ax.set_xlabel("subjects")
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=ANC_COLOR[c], label=c)
         for c in ANC_ORDER]
    pad = key_above(ax, h, ncol=2)
    ax.set_title("Self-reported ancestry", pad=pad, loc="left")


def panel_registration(ax):
    """(a) The motivating claim, on real geometry: the subject, the normative
    template it would be fitted to, and the result of that fit — all three on
    one deviation scale, with the bar shown rather than described.

    The template is included because "attenuated toward the template" cannot be
    read from a before/after pair alone: a reader has to see what it moved
    toward. Colours come from Figure 1's own blue ramp rather than magma (see
    make_registration_paper.py), so the panel sits in the same colour system as
    the rest of the figure."""
    import json
    import matplotlib.image as mpimg
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    meta = json.loads((REG / "fig_regp_meta.json").read_text())
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Template fitting reduces apparent deviation from normative facial shape",
                 pad=2, loc="left")

    panels = [("fig_regp_raw", "Original facial surface",
               f"{meta['dev_raw_mean']:.1f} mm"),
              ("fig_regp_template", "Normative template", None),
              ("fig_regp_registered", "After template fitting",
               f"{meta['dev_post_mean']:.1f} mm")]
    xs = [0.055, 0.300, 0.545]
    w = 0.205
    # These y positions are FRACTIONS OF A STRIP whose printed height changed
    # when the row below it was given clear space, so they were re-derived
    # rather than kept: at the old values the bold deviation line printed
    # through the descenders of "22q11.2 deletion".
    for x, (img, lab, dev) in zip(xs, panels):
        ins = ax.inset_axes([x, 0.345, w, 0.60])
        ins.imshow(mpimg.imread(str(REG / f"{img}.png")))
        ins.axis("off")
        ax.annotate(lab, xy=(x + w / 2, 0.295), ha="center", va="top",
                    fontsize=5.9, color=FS.MUTED)
        if dev:
            ax.annotate(f"mean deviation {dev}", xy=(x + w / 2, 0.085),
                        ha="center", va="top", fontsize=6.6, color=FS.INK,
                        fontweight="bold")
    for x0 in (0.272, 0.517):
        ax.add_patch(FancyArrowPatch((x0, 0.645), (x0 + 0.035, 0.645),
                                     arrowstyle="-|>", mutation_scale=7,
                                     lw=0.8, color=FS.INK, zorder=4))
    cmap = LinearSegmentedColormap.from_list(
        "fb_blues", ["#F7FAFC", "#D6E3EE", "#9EC1DC", "#5E8FBA",
                     "#2C5F8A", "#123F63"])
    cax = ax.inset_axes([0.792, 0.385, 0.019, 0.50])
    cb = ax.figure.colorbar(
        ScalarMappable(norm=Normalize(0, meta["vmax"]), cmap=cmap), cax=cax)
    cb.outline.set_linewidth(0.5)
    cb.ax.tick_params(labelsize=5.8, width=0.5, length=2)
    cb.set_ticks([0, meta["vmax"] / 2, meta["vmax"]])
    cb.set_ticklabels(["0", f"{meta['vmax']/2:.0f}", f"{meta['vmax']:.0f}"])
    ax.annotate("deviation from\ntemplate (mm)", xy=(0.852, 0.635),
                ha="left", va="center", fontsize=5.8, color=FS.MUTED)


def panel_datasets(ax, df):
    """(b) How big each dataset is, in scans and in subjects.

    Sets the row order that panel_ancestry() also uses. The two panels stay
    separate — bar length here is SCANS, whereas ancestry is a percentage of
    SUBJECTS, and for FB-TJ0 (13,307 scans, far fewer subjects) the two are not
    interchangeable, so a single bar carrying both would imply per-scan ancestry
    counts we have never computed."""
    y = np.arange(len(df))
    xmax = df.scans.max() * 1.26
    ax.barh(y, df.scans, 0.62, color=[ROLE_COLOR[r] for r in df.role], zorder=3)
    ax.plot(df.subjects, y, "|", ms=6, mew=1.1, color="white", zorder=5)
    ax.plot(df.subjects, y, "|", ms=6, mew=0.9, color=FS.ACCENT, zorder=6)
    for i, r in df.iterrows():
        # A bar past half the axis has no room for an outside label at this
        # width — the count would run off the panel — so it is set inside in
        # white. Everything else is labelled outside, where the short bars leave
        # the room. Putting them all outside is what forced a 1.45x headroom
        # that squashed the five small cohorts into the first fifth of the axis.
        inside = r.scans > 0.55 * xmax
        ax.annotate(f"{r.scans:,}", xy=(r.scans, i),
                    xytext=(-3 if inside else 3, 0),
                    textcoords="offset points", va="center", fontsize=5.9,
                    ha="right" if inside else "left",
                    color="white" if inside else FS.INK, zorder=7)
    ax.set_yticks(y)
    ax.set_yticklabels(df.dataset)
    ax.set_ylim(-0.6, len(df) - 0.4)
    ax.set_xlim(0, xmax)
    # Explicit ticks in thousands. At a third of the text width the automatic
    # locator picks 2,500 steps and the labels run together ("100001250015000");
    # even four full labels ("15,000") collide, so the axis is in k.
    ax.set_xticks([0, 5000, 10000, 15000])
    ax.set_xticklabels(["0", "5k", "10k", "15k"])
    ax.set_xlabel("scans")
    ax.grid(axis="x", color=FS.RULE, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    h = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLOR[k], label=v)
         for k, v in [("cleft", "cleft cohorts"), ("syndrome", "syndrome library"),
                      ("control", "controls")]]
    h.append(plt.Line2D([], [], color=FS.ACCENT, marker="|", ls="none",
                        ms=6, mew=0.9, label="distinct subjects"))
    pad = key_above(ax, h, ncol=2)
    ax.set_title("Dataset size", pad=pad, loc="left")


def panel_matrix(ax, df, nsize):
    """(d) Which cohorts feed which task, TASKS ON Y — the original orientation.

    It was briefly transposed to put datasets on y so that it could share rows
    with (b) and (c); this one was restored. Tasks on y is the better reading
    anyway: the task names are the long strings, and on y they are set
    horizontally rather than rotated under the columns.

    Its OWN axes throughout — its own y labels (the tasks) and its own x labels
    (the datasets, in the datasets' own order rather than (b)'s size order)."""
    dsets = ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]
    role = dict(zip(df.dataset, df.role))
    ntask = len(TASKS)
    for j, ds in enumerate(dsets):
        for i, (task, k, used) in enumerate(TASKS):
            yy = ntask - 1 - i
            if ds in used:
                ax.plot(j, yy, "o", ms=5.5, color=ROLE_COLOR[role[ds]], zorder=3)
            else:
                ax.plot(j, yy, "o", ms=5.5, mfc="white", mec=FS.RULE,
                        mew=0.7, zorder=2)
    # Corpus size per task, at the right of its row. The x limits are opened past
    # the last dataset column so this sits INSIDE the axes: (d) is the rightmost
    # panel, so an annotation placed beyond axes-fraction 1.0 runs off the canvas
    # rather than into a margin.
    for i, (task, _, _) in enumerate(TASKS):
        ax.annotate(f"{nsize[task]:,}", xy=(len(dsets) - 0.30, ntask - 1 - i),
                    ha="left", va="center", fontsize=5.2, color=FS.MUTED)
    ax.annotate("scans", xy=(len(dsets) - 0.30, ntask - 0.45), ha="left",
                va="center", fontsize=5.2, color=FS.MUTED, style="italic")
    ax.set_xticks(range(len(dsets)))
    ax.set_xticklabels(dsets, fontsize=5.8, rotation=45, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(ntask))
    # Two lines per task — name, then class count. On one line the longest label
    # ("Syndrome category  K=19") set the whole panel's left margin and squeezed
    # the dot columns; wrapped, the margin is the name alone.
    ax.set_yticklabels([f"{t}\n{k}" for t, k, _ in TASKS][::-1], fontsize=5.8,
                       linespacing=1.25)
    ax.set_xlim(-0.6, len(dsets) + 1.75)
    ax.set_ylim(-0.6, ntask - 0.25)
    ax.set_title("Cohorts per task", pad=TITLE_PAD, loc="left")
    ax.grid(color=FS.RULE, lw=0.35, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)


def box(ax, x, y, w, h, text, fc="white", ec=None, fs=6.4, bold=False):
    ec = ec or FS.INK
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.006,rounding_size=0.014",
                                fc=fc, ec=ec, lw=0.7, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            zorder=4, color=FS.INK,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, x1, y, label=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=FS.INK,
                                 zorder=3, shrinkA=0, shrinkB=0))
    if label:
        ax.annotate(label, xy=((x0 + x1) / 2, y), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.9, color=FS.MUTED)


# Raw mesh for (e): the FaceBase file of the SAME acquisition the stored cloud
# was sampled from (preprocess.py).
# Schematic class colours for (e), shared by BOTH branches so that class k in
# the probability bars is prototype k in the embedding sketch.
CLS_COLOR = ["#123F63", "#3E6E9E", "#7FA8CC", "#BFD3E6"]


def exemplar_mesh():
    """The raw surface mesh, put into the stored cloud's normalised frame.

    The frame is not re-derived from preprocess.py's random surface sample: the
    stored cloud IS a subset of this surface, so its bounding box pins the
    similarity transform (one scale, three offsets) directly. Measured
    2026-10-05: re-sampling the cleaned mesh gives the same frame to <1%."""
    import trimesh
    from facebench.preprocessing.mesh_utils import load_face_mesh
    sid = _exemplar_scan()
    src = pd.read_csv(SPLITS / "scans.csv").set_index("scan_id").loc[sid, "src_path"]
    m = load_face_mesh(facebase(src), process=True)
    m = max(m.split(only_watertight=False), key=lambda c: len(c.vertices))
    pts = np.load(PROCESSED / "syndrome" / f"{_exemplar_scan()}.npy")
    V = np.asarray(m.vertices, float)
    lo, hi = V.min(0), V.max(0)
    s = np.mean((pts.max(0) - pts.min(0)) / (hi - lo))
    V = (V - lo) * s + pts.min(0)
    keep = V[:, 1] > FACE_Y_CUT
    F = m.faces[keep[m.faces].all(1)]
    return V, F


def draw_mesh(ax, V, F):
    """Flat-shaded frontal render, painter-sorted, same blue-grey as (a)'s
    template so 'surface mesh' reads the same way in both panels."""
    from matplotlib.collections import PolyCollection
    light = np.array([-0.3, 0.35, 1.0])
    light /= np.linalg.norm(light)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    sh = 0.30 + 0.70 * np.clip(n @ light, 0, 1)
    o = np.argsort(tri[:, :, 2].mean(1))
    fc = np.clip(np.array([0.78, 0.82, 0.87])[None] * sh[o, None], 0, 1)
    ax.add_collection(PolyCollection(tri[o][:, :, :2], facecolors=fc,
                                     edgecolors="none", rasterized=True))


def mini_bars(ax, p, hi=None, ec=None, hic=None):
    """A probability vector over the schematic classes. No axis numbers: the
    values are illustrative (the caption says so), the SHAPE is the point."""
    ax.bar(range(len(p)), p, 0.78, color=CLS_COLOR,
           edgecolor=ec or "none", linewidth=0.6 if ec else 0, zorder=3)
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-0.6, len(p) - 0.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    if hi is not None:
        ax.annotate("▼", xy=(hi, p[hi]), xytext=(0, 1.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=4.2, color=hic or FS.ACCENT)


def panel_pipeline(ax):
    """(e) Raw mesh -> sampled cloud -> encoder -> z, then the TWO inference
    rules as two branches, each showing WHERE a patient's scans are pooled:

      softmax   : per-scan class probabilities, averaged    (prediction space)
      prototype : per-scan embeddings averaged, then the nearest class
                  prototype = mean training embedding, cosine (embedding space)

    That distinction is the paper's (Results, "patient-level aggregation"), and
    a single 'decision rule' box could not show it. With one scan (n = 1) both
    branches reduce to per-scan prediction, so the per-scan/per-patient split is
    written on the pooling step rather than drawn as two more boxes.

    The bar heights and the embedding sketch are SCHEMATIC (four illustrative
    classes, no axis values) — the caption must say so. The mesh and the cloud
    are real: the same FB-TJ0 acquisition, cropped at the neck for DISPLAY only
    (the model input is uncropped; see FACE_Y_CUT)."""
    pts = np.load(PROCESSED / "syndrome" / f"{_exemplar_scan()}.npy")
    pts = pts[pts[:, 1] > FACE_Y_CUT]
    V, F = exemplar_mesh()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("3D facial analysis framework and the two inference rules",
                 pad=1, loc="left")

    # ---- trunk ------------------------------------------------------------
    yc = 0.50
    # Both renders share one frame, so the cloud is visibly a sampling of the
    # mesh beside it rather than a differently-scaled picture.
    # Extents of what is DRAWN (faces above the neck cut), not of the whole
    # vertex array, which still carries the shoulders.
    Vd = V[np.unique(F)]
    xl = (min(Vd[:, 0].min(), pts[:, 0].min()) - 0.02,
          max(Vd[:, 0].max(), pts[:, 0].max()) + 0.02)
    yl = (FACE_Y_CUT - 0.02, max(Vd[:, 1].max(), pts[:, 1].max()) + 0.02)
    # TRUNK LAYOUT IS SOLVED, NOT HAND-PLACED. Arrows of different
    # lengths, and z_i -> fork had none. Two causes, both fixed here:
    #  * the renders are narrower than a guessed inset width (aspect is set by
    #    the head), so arrows started in white space; the insets are now sized
    #    to the head's own drawn width, measured in inches;
    #  * gaps were typed per arrow; now every visible arrow has ONE length A,
    #    solved so the trunk exactly fills the room left of the fork bands.
    X0 = 0.445                       # left edge of the fork bands (see below)
    wr = 0.066                       # drawn head width, axes fraction
    pos = ax.get_position()
    W_in = ax.figure.get_size_inches()[0] * pos.width
    H_in = ax.figure.get_size_inches()[1] * pos.height
    ih = wr * W_in * (yl[1] - yl[0]) / (xl[1] - xl[0]) / H_in
    BOXPAD = 0.008                   # FancyBboxPatch pad + a hair of clearance
    CLR = 0.004                      # clearance between an arrow and an object
    W_ENC, W_Z = 0.092, 0.046
    STEM = 0.5                       # z->elbow and elbow->band, as fractions of A
    fixed = 2 * wr + W_ENC + W_Z + 2 * BOXPAD * 2
    # mesh->cloud, cloud->enc, enc->z: 3 arrows; then the stem and the branch
    # arrow (STEM*A each); clearances at both ends of every arrow.
    A = (X0 - fixed - 3 * 2 * CLR - 2 * CLR) / (3 + 2 * STEM)
    x = 0.0
    xm = x; x += wr
    a1 = (x + CLR, x + CLR + A); x = a1[1] + CLR
    xc = x; x += wr
    a2 = (x + CLR, x + CLR + A); x = a2[1] + CLR + BOXPAD
    xe = x; x += W_ENC + BOXPAD
    a3 = (x + CLR, x + CLR + A); x = a3[1] + CLR + BOXPAD
    xzb = x; x += W_Z + BOXPAD
    xz = x + CLR
    xf = xz + STEM * A
    print(f"  (e) trunk: head width {wr:.3f}, arrow length {A:.3f} = {A * W_in:.2f} in")
    for x0, kind in [(xm, "mesh"), (xc, "cloud")]:
        ins = ax.inset_axes([x0, yc - ih / 2 + 0.06, wr, ih])
        ins.set_xlim(*xl)
        ins.set_ylim(*yl)
        ins.set_aspect("equal")
        ins.axis("off")
        if kind == "mesh":
            draw_mesh(ins, V, F)
        else:
            o = np.argsort(pts[:, 2])
            z = pts[o, 2]
            ins.scatter(pts[o, 0], pts[o, 1], s=0.35, c=z, cmap="bone_r",
                        vmin=np.percentile(z, 25), vmax=np.percentile(z, 99),
                        linewidths=0, rasterized=True)
    ax.annotate("raw 3D scan\n(.obj mesh)", xy=(max(xm + wr / 2, 0.040), yc - ih / 2 + 0.06),
                ha="center", va="top", fontsize=5.3, color=FS.MUTED)
    ax.annotate("4,096 sampled\npoints", xy=(xc + wr / 2, yc - ih / 2 + 0.06),
                ha="center", va="top", fontsize=5.3, color=FS.MUTED)
    bh = 0.36
    box(ax, xe, yc - bh / 2 + 0.06, W_ENC, bh,
        "encoder $f_\\theta$\nPointNet\nPointNet++\nDGCNN\nGeomMLP", fs=5.3)
    box(ax, xzb, yc - bh / 2 + 0.06, W_Z, bh, "embed-\nding $\\mathbf{z}_i$", fs=5.6)
    for a0, a1_ in (a1, a2, a3):
        arrow(ax, a0, a1_, yc + 0.06)
    ax.annotate("scan $i = 1,\\ldots,n$ of one patient",
                xy=((xe + xzb + W_Z) / 2, yc - bh / 2 + 0.03), ha="center", va="top",
                fontsize=5.4, color=FS.MUTED)
    # Key for the embedding sketches, under the trunk where there is room.
    for i, (mk, txt) in enumerate([("*", "prototype $\\mathbf{c}_k$ (class-mean\ntraining embedding)"),
                                   ("o", "scan embedding $\\mathbf{z}_i$"),
                                   ("O", "patient mean $\\bar{\\mathbf{z}}$")]):
        y = 0.215 - i * 0.075 - (0.02 if i else 0)
        kx = xe
        if mk == "*":
            ax.plot(kx, y, "*", ms=5, color=CLS_COLOR[0], mec=FS.INK, mew=0.3)
        elif mk == "o":
            ax.plot(kx, y, "o", ms=2.6, mfc="white", mec=FS.ACCENT, mew=0.6)
        else:
            ax.plot(kx, y, "o", ms=3.6, color=FS.ACCENT)
        ax.annotate(txt, xy=(kx + 0.010, y), ha="left", va="center",
                    fontsize=5.0, color=FS.MUTED, linespacing=1.15)

    # ---- the fork: inference rule (rows) x level (columns) ----------------
    # A 2x2, because the paper crosses the two: each rule is reported at the
    # scan level (each scan predicted alone) AND at the patient level (the
    # patient's n scans pooled). The rules differ in WHERE they pool —
    # probabilities for softmax, embeddings for prototypes — which is the red
    # 'average' step that appears only in the right-hand column.
    XD, X1 = 0.660, 0.997            # column divider, right edge (X0 above)
    top = (0.585, 0.925)
    bot = (0.010, 0.560)
    for (y0, y1) in (top, bot):
        ax.add_patch(FancyBboxPatch((X0, y0), X1 - X0, y1 - y0,
                                    boxstyle="round,pad=0.0,rounding_size=0.012",
                                    fc="#F2F6FA", ec="none", zorder=0))
    ax.plot([XD, XD], [bot[0] + 0.02, top[1] - 0.02], color=FS.RULE, lw=0.6,
            ls=(0, (2, 2)), zorder=1)
    for xc, txt in [((X0 + 0.030 + XD) / 2, "scan-level  (each scan alone)"),
                    ((XD + X1) / 2, "patient-level  ($n$ scans pooled)")]:
        ax.annotate(txt, xy=(xc, 0.990), ha="center", va="top", fontsize=5.9,
                    color=FS.INK, fontweight="bold")
    sy = 0.745                       # top-row flow line
    ey = 0.285                       # bottom-row centre
    for y, l1, l2 in [(sy, "softmax", "$\\mathbf{p}_i=\\sigma(W\\mathbf{z}_i)$"),
                      (ey, "prototype", "nearest $\\mathbf{c}_k$")]:
        ax.annotate(l1, xy=(X0 + 0.040, y + 0.012), ha="center", va="bottom",
                    fontsize=5.8, color=FS.INK, fontweight="bold")
        ax.annotate(l2, xy=(X0 + 0.040, y - 0.004), ha="center", va="top",
                    fontsize=5.2, color=FS.MUTED)
    # Fork drawn as an explicit elbow: z -> stem -> up and down -> bands.
    # xz, xf come from the trunk solve, so the stem and the two branch arrows
    # share the trunk's arrow length.
    ax.plot([xz, xf], [yc + 0.06] * 2, color=FS.INK, lw=0.8, zorder=3)
    ax.plot([xf, xf], [ey, sy], color=FS.INK, lw=0.8, zorder=3)
    for y in (sy, ey):
        arrow(ax, xf, X0 - CLR, y)

    # Illustrative numbers. Chosen so that scan 1 ALONE picks a different class
    # from the pooled patient, in both rows — the pooling benefit the paper
    # reports, made visible. Not data.
    P = np.array([[0.28, 0.44, 0.18, 0.10],
                  [0.47, 0.24, 0.19, 0.10],
                  [0.50, 0.20, 0.17, 0.13]])
    pbar = P.mean(0)

    # softmax x scan: p_1 -> argmax
    bx_ = ax.inset_axes([0.522, sy - 0.085, 0.040, 0.19])
    bx_.patch.set_alpha(0)
    mini_bars(bx_, P[0], hi=int(P[0].argmax()), hic=FS.INK)
    ax.annotate("$\\mathbf{p}_i$", xy=(0.542, sy - 0.095), ha="center",
                va="top", fontsize=5.2, color=FS.MUTED)
    arrow(ax, 0.570, 0.590, sy)
    ax.annotate("$\\arg\\max_k\\,p_{ik}$", xy=(0.594, sy), ha="left",
                va="center", fontsize=5.6, color=FS.INK)

    # softmax x patient: p_1..p_n -> average -> p-bar -> argmax
    for k, p in enumerate(P):
        bx_ = ax.inset_axes([0.668 + k * 0.036, sy - 0.085, 0.031, 0.19])
        bx_.patch.set_alpha(0)
        mini_bars(bx_, p)
        ax.annotate(f"$\\mathbf{{p}}_{k + 1}$", xy=(0.6835 + k * 0.036, sy - 0.095),
                    ha="center", va="top", fontsize=5.2, color=FS.MUTED)
    arrow(ax, 0.778, 0.828, sy)
    ax.annotate("average", xy=(0.803, sy + 0.012), ha="center", va="bottom",
                fontsize=5.4, color=FS.ACCENT)
    bx_ = ax.inset_axes([0.832, sy - 0.085, 0.045, 0.19])
    bx_.patch.set_alpha(0)
    mini_bars(bx_, pbar, hi=int(pbar.argmax()), ec=FS.ACCENT)
    ax.annotate("$\\bar{\\mathbf{p}}$", xy=(0.8545, sy - 0.095), ha="center",
                va="top", fontsize=5.6, color=FS.ACCENT)
    arrow(ax, 0.884, 0.904, sy)
    ax.annotate("$\\arg\\max_k\\,\\bar{p}_k$", xy=(0.908, sy), ha="left",
                va="center", fontsize=5.6, color=FS.INK)

    # prototype row: the SAME class layout in both cells, so the only change
    # between them is z_1 alone vs the pooled z-bar.
    rng = np.random.RandomState(3)
    C = np.array([[1.45, 0.55], [0.35, 0.78], [2.45, 0.80], [3.05, 0.30]])
    clouds = [c + rng.normal(0, [0.13, 0.075], (16, 2)) for c in C]
    Z = np.array([[0.80, 0.42], [1.75, 0.30], [1.35, 0.06]])

    def sketch(rect, q, Zs):
        ex = ax.inset_axes(rect)
        ex.set_xlim(-0.05, 3.45)
        ex.set_ylim(-0.05, 1.0)
        ex.axis("off")
        for k, (c, cl) in enumerate(zip(C, clouds)):
            ex.scatter(cl[:, 0], cl[:, 1], s=1.4, color=CLS_COLOR[k],
                       alpha=0.55, linewidths=0, zorder=2)
            ex.scatter(*c, marker="*", s=30, color=CLS_COLOR[k],
                       edgecolors=FS.INK, linewidths=0.35, zorder=4)
            ex.annotate(f"$\\mathbf{{c}}_{k + 1}$", xy=c, xytext=(3, 2),
                        textcoords="offset points", fontsize=4.8,
                        color=FS.INK, zorder=5)
        near = int(np.argmin(np.linalg.norm(C - q, axis=1)))
        for k, c in enumerate(C):
            ex.plot(*zip(q, c), ls=(0, (2, 1.5)), lw=0.8 if k == near else 0.4,
                    color=FS.INK if k == near else FS.RULE, zorder=3)
        if Zs is not None:
            for zi in Zs:
                ex.plot(*zip(zi, q), color=FS.ACCENT, lw=0.5, alpha=0.7,
                        zorder=3)
            ex.scatter(Zs[:, 0], Zs[:, 1], s=6, facecolors="white",
                       edgecolors=FS.ACCENT, linewidths=0.6, zorder=5)
            ex.scatter(*q, s=12, color=FS.ACCENT, zorder=6)
            ex.annotate("$\\bar{\\mathbf{z}}$", xy=q, xytext=(-4, -2),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=5.6, color=FS.ACCENT)
        else:
            ex.scatter(*q, s=8, facecolors="white", edgecolors=FS.ACCENT,
                       linewidths=0.7, zorder=6)
            ex.annotate("$\\mathbf{z}_i$", xy=q, xytext=(-3, -2),
                        textcoords="offset points", ha="right", va="top",
                        fontsize=5.4, color=FS.ACCENT)
        return near

    sh = 0.40
    sketch([X0 + 0.074, bot[0] + 0.115, XD - X0 - 0.080, sh], Z[0], None)
    ax.annotate("$\\arg\\min_k\\,d_{\\cos}(\\mathbf{z}_i,\\mathbf{c}_k)$",
                xy=((X0 + 0.074 + XD) / 2, bot[0] + 0.085), ha="center",
                va="top", fontsize=5.6, color=FS.INK)
    sketch([XD + 0.008, bot[0] + 0.115, X1 - XD - 0.016, sh], Z.mean(0), Z)
    ax.annotate("average $\\mathbf{z}_1,\\ldots,\\mathbf{z}_n$, then  "
                "$\\arg\\min_k\\,d_{\\cos}(\\bar{\\mathbf{z}},\\mathbf{c}_k)$",
                xy=((XD + X1) / 2, bot[0] + 0.085), ha="center", va="top",
                fontsize=5.6, color=FS.INK)


def main():
    FS.apply()
    df = cohort_table()
    nsize = task_sizes()
    anc, missing = ancestry()
    # HEIGHT IS A HARD CONSTRAINT, NOT A PREFERENCE. At 8.10 in the float was
    # 217 pt taller than the text block, so LaTeX pushed it to the back matter
    # AND cut the caption off the page mid-sentence — panels (c), (d) and (e)
    # were described nowhere in the rendered PDF. Putting (b), (c) and (d) in
    # one row is what buys the height back. Anything that grows this number has
    # to be checked against `Float too large for page` in main.log.
    H = 5.32
    fig = plt.figure(figsize=(FS.DOUBLE, H))
    # constrained_layout is restricted to the top of the canvas so the pipeline
    # strip can be placed manually across the FULL width. Inside the gridspec it
    # gets aligned to the columns above and inherits their tick-label margins,
    # which is why it did not read as a full-width row.
    # (a) and (e) are schematic strips and are placed MANUALLY at full bleed;
    # constrained_layout governs only the three data panels between them, so
    # those stay mutually aligned while the strips use the whole width.
    # The two schematic strips keep their PRINTED size (inches), so they are
    # converted to figure fractions against H rather than left as the fractions
    # that happened to be right at 8.10 in. GAP is the clear space asked
    # for between the data row and the pipeline strip.
    # The two strips give up some of their printed height to the data row.
    # At H=5.06 with the old 1.474/1.183 in strips the row was 1.97 in tall,
    # too short for (b)'s role key to sit in the bars' whitespace without
    # overprinting them; the strips lose 0.29 in between them and the row
    # gets it back. Neither strip's TYPE shrinks — both are sized in points.
    A_H, P_H = 1.22 / H, 1.52 / H
    GAP_TOP, GAP_BOT = 0.20 / H, 0.20 / H     # (a)->row, row->(e)
    a_bot = 1.0 - 0.030 - A_H          # headroom for (a)'s title
    m_bot = 0.004 + P_H + GAP_BOT
    m_top = a_bot - GAP_TOP
    # GAP_TOP is why (a)'s deviation figures no longer sit on top of the row's
    # panel titles. It is paid for by (a)'s strip rather than by the row: the
    # row is what carries six bars and five task rows, and H cannot grow without
    # putting the caption back over the page.
    fig.get_layout_engine().set(rect=(0.0, m_bot, 1.0, m_top - m_bot))
    # ONE row of three INDEPENDENT panels: (b) size, (c) composition, (d) use.
    # Each carries its own x and y axes and its own key — nothing is shared, so
    # none of them can be misread as a continuation of its neighbour. The
    # columns are unequal because their labels are: (b) and (c) pay for dataset
    # names, (d) for task names on y and a corpus-size column on the right.
    # Ancestry LEFT, dataset size MIDDLE: (d)'s dots carry the same three role
    # colours as the size panel's bars, so the two panels that share a colour
    # code sit next to each other instead of with the ancestry ramp between
    # them. (d) is the widest of the three because it is the only one with no
    # key to pay for, and its dot grid is what benefits from the room.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.38, 0.99, 0.83])
    axa = fig.add_axes([0.012, a_bot, 0.976, A_H])
    axa.set_in_layout(False)
    panel_registration(axa)
    ax_anc = fig.add_subplot(gs[0, 0])
    ax_size = fig.add_subplot(gs[0, 1])
    ax_task = fig.add_subplot(gs[0, 2])
    panel_ancestry(ax_anc, df, anc, missing)
    panel_datasets(ax_size, df)
    panel_matrix(ax_task, df, nsize)
    axp = fig.add_axes([0.012, 0.004, 0.976, P_H])
    axp.set_in_layout(False)
    panel_pipeline(axp)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    # Freeze BEFORE touching any position: everything below moves artists that
    # constrained_layout would otherwise re-solve around, and re-solving is what
    # let (c) expand across (d).
    fig.set_layout_engine("none")
    grow_to_title(ax_task, ax_size, H)
    fig.canvas.draw()
    letter_titles(fig, [(axa, "a"), (ax_anc, "b"), (ax_size, "c"),
                        (ax_task, "d"), (axp, "e")],
                  align={ax_task: ax_size})
    fig.canvas.draw()
    check_no_overlap([ax_anc, ax_size, ax_task])
    check_keys_fit(fig)
    FS.finish(fig, OUT / "figS_overview.png", FS.DOUBLE)
    print(f"  height {H:.2f} in = {H * 72.27:.0f} pt of the text block")


if __name__ == "__main__":
    main()
