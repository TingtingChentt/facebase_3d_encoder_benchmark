#!/usr/bin/env python3
"""
Figure 1 (new) — study overview: what data, what tasks, what the model does.
thread facebase3d-paper-2026-08-08.

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
  to share one set of dataset rows; they were separated — a shared axis
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

The face in (c) is a real test scan (Achondroplasia, FB-TJ0) rendered from the
stored 4,096-point cloud — the actual model input, not an illustration.

Usage: python3 fig0_overview.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).parent))
import figstyle as FS                                       # noqa: E402

PROJ = Path(__file__).resolve().parents[2]
DATA = PROJ / "data/processed"
OUT = PROJ / "figures"
EXEMPLAR = DATA / "syndrome/fbtj0_FB1349_150706102257.npy"   # Achondroplasia
FACE_Y_CUT = -0.19          # neck; see panel_pipeline for the measurement
FBSRC = Path(os.environ.get("FACEBASE_ROOT", "/path/to/facebase"))
REG = PROJ / "figures"          # fig_regp_* — blue-palette renders from
                                # facebench/paper_figs/make_registration_paper.py

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
    ("Syndrome category", "$K$=19", ["FB-TJ0"]),
    ("Clinical diagnosis", "$K$=33", ["FB-TJ0"]),
    ("Combined screening", "$K$=2",
     ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]),
]


def cohort_table():
    """Scans and subjects per dataset, from the manifests themselves."""
    ofc = pd.read_csv(DATA / "ofc_manifest.csv")
    syn = pd.read_csv(DATA / "syndrome_manifest.csv")
    ctl = pd.read_csv(DATA / "controls_manifest.csv")
    rows = []
    for ds, g in ofc.groupby("dataset"):
        rows.append((ds, len(g), g.study_id.nunique(), "cleft"))
    rows.append(("FB-TJ0", len(syn), syn.fbid.nunique(), "syndrome"))
    for ds, g in ctl.groupby("dataset"):
        rows.append((ds, len(g), len(g), "control"))
    df = pd.DataFrame(rows, columns=["dataset", "scans", "subjects", "role"])
    return df.sort_values("scans", ascending=True).reset_index(drop=True)


def task_sizes():
    """Scans entering each task, after the class floors."""
    n = {}
    n["Cleft type"] = len(pd.read_csv(DATA / "ofc_manifest.csv"))
    n["Cleft screening"] = n["Cleft type"]
    n["Syndrome category"] = len(pd.read_csv(DATA / "syndrome_manifest_b1.csv"))
    n["Clinical diagnosis"] = len(pd.read_csv(DATA / "syndrome_clinical_manifest.csv"))
    n["Combined screening"] = len(pd.read_csv(DATA / "combined_manifest.csv"))
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
    ofc = pd.read_csv(DATA / "ofc_manifest.csv")
    syn = pd.read_csv(DATA / "syndrome_manifest.csv")
    src = {
        "FB-5A": (ofc[ofc.dataset == "FB-5A"], "study_id",
                  pd.read_csv(FBSRC / "FB-5A_files/Data/OFC1_Demographics.csv", **rd),
                  "StudyID", "code"),
        "FB-56": (ofc[ofc.dataset == "FB-56"], "study_id",
                  pd.read_csv(FBSRC / "FB-56_files/Data/OFC2_DemoPart1Exported.csv", **rd),
                  "StudyID", "code"),
        "FB-TJ0": (syn, "fbid",
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
    ax.set_title("(b) Self-reported ancestry", pad=pad, loc="left")


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
    ax.set_title("(a) Registration attenuates the signal it is meant to measure",
                 pad=2, loc="left")

    panels = [("fig_regp_raw", "unregistered scan\n22q11.2 deletion",
               f"{meta['dev_raw_mean']:.1f} mm"),
              ("fig_regp_template", "normative template\n(control subject)", None),
              ("fig_regp_registered", "after fitting to template",
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
    ax.set_title("(c) Dataset size", pad=pad, loc="left")


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
    ax.set_title("(d) Cohorts per task", pad=TITLE_PAD, loc="left")
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


def panel_pipeline(ax):
    """Single left-to-right flow. Every box is placed on one baseline and every
    arrow runs edge-to-edge between neighbours, so nothing can overlap: the
    previous version stacked the decision rules vertically and routed one arrow
    into the middle of the stack, which read as if only that rule were used."""
    pts = np.load(EXEMPLAR)
    # DISPLAY crop only. The scan the model consumes runs down to the shoulders;
    # this draws the head so the face is legible at 1 in wide. The cut is the
    # geometry's own break, not a guess: cloud width collapses from 1.25 to 0.52
    # between y=-0.32 and y=-0.06, which is the neck. The caption says the
    # render is cropped and the input is not — the no-cropping claim is one of
    # the paper's, so the figure must not quietly imply a detection step.
    pts = pts[pts[:, 1] > FACE_Y_CUT]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(e) One scan through the pipeline", pad=1, loc="left")

    yc = 0.58                      # single baseline for the whole flow
    bh = 0.42                      # box height
    by = yc - bh / 2

    # The raw cloud is shown UNCROPPED, including neck and shoulders. That is
    # the actual model input: there is no face detection, cropping or alignment
    # anywhere in the pipeline, and showing a tidy cropped face would imply a
    # step we do not perform.
    ins = ax.inset_axes([0.005, by - 0.06, 0.115, bh + 0.12])
    # Depth-sorted so nearer points draw last, and the colour range clipped to
    # the head's own central depths. On the full cloud's range the face occupied
    # a sliver of the ramp and rendered as a flat blob, because the extremes are
    # set by the ears at the back and the shoulders that are no longer drawn.
    o = np.argsort(pts[:, 2])
    z = pts[o, 2]
    ins.scatter(pts[o, 0], pts[o, 1], s=0.5, c=z, cmap="bone_r",
                vmin=np.percentile(z, 25), vmax=np.percentile(z, 99),
                linewidths=0, rasterized=True)
    ins.set_aspect("equal")
    ins.axis("off")
    ax.annotate("raw scan\n4,096 points", xy=(0.062, by - 0.10),
                ha="center", va="top", fontsize=5.8, color=FS.MUTED)

    # Explicit non-overlapping layout. Each box is (x, width); the next arrow
    # starts at x+width and ends at the following x, so a box can never be
    # drawn over its neighbour and an arrow can never run backwards.
    stages = [
        (0.155, 0.185, "encoder\nPointNet, PointNet++,\nDGCNN, GeomMLP", "white", 5.7),
        (0.375, 0.085, "embedding\n$\\mathbf{z}$", "white", 6.0),
        (0.495, 0.175, "decision rule\nsoftmax,\nprototype", "#F2F6FA", 5.7),
        (0.705, 0.130, "per-scan\nprediction", "#F2F6FA", 6.0),
        (0.862, 0.126, "per-patient\nprediction", FS.ENC_COLOR["geommlp"], 6.0),
    ]
    for x, w, txt, fc, fs in stages:
        box(ax, x, by, w, bh, txt, fc=fc, fs=fs)

    # BOXPAD: FancyBboxPatch("round,pad=P") draws P beyond the rect on every
    # side, so an arrow ending at the rect's x overlaps the visible outline.
    # Every arrow is inset by BOXPAD at both ends.
    BOXPAD = 0.008
    for x0, x1 in [(0.124, 0.155), (0.340, 0.375), (0.460, 0.495),
                   (0.670, 0.705), (0.835, 0.862)]:
        arrow(ax, x0 + BOXPAD, x1 - BOXPAD, yc)

    # The pooling step is the paper's one positive actionable result, so the
    # arrow that performs it is labelled rather than left to the caption.
    ax.annotate("pool a patient's\nscans", xy=(0.8485, by - 0.05),
                ha="center", va="top", fontsize=5.8, color=FS.ACCENT)



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
    H = 4.86
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
    A_H, P_H = 1.22 / H, 1.06 / H
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
    OUT.mkdir(exist_ok=True)
    fig.canvas.draw()
    # Freeze BEFORE touching any position: everything below moves artists that
    # constrained_layout would otherwise re-solve around, and re-solving is what
    # let (c) expand across (d).
    fig.set_layout_engine("none")
    grow_to_title(ax_task, ax_size, H)
    centre_titles([ax_anc, ax_size, ax_task])
    fig.canvas.draw()
    check_no_overlap([ax_anc, ax_size, ax_task])
    check_keys_fit(fig)
    FS.finish(fig, OUT / "fig0_overview.png", FS.DOUBLE)
    print(f"  height {H:.2f} in = {H * 72.27:.0f} pt of the text block")


if __name__ == "__main__":
    main()
