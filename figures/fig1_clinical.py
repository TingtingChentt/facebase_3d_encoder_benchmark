#!/usr/bin/env python3
"""
Figure 1 — from 3D facial screening to diagnostic prioritization.

  (a) Clinical motivation: one held-out FB-TJ0 scan and the two questions the
      benchmark asks of it, screening (is a craniofacial condition present?) and
      diagnosis (which syndromes should be considered?), with the real PointNet++
      outputs for that scan (combined-screening model, and scan-level prototype
      inference of the 49-class model; training run 1, inference seed 0).
  (b) The six FaceBase cohorts (scans / subjects, as Table 1) -> 4,096-point cloud
      -> one of four encoders -> facial embedding.
  (c) The four benchmark tasks, from broad screening to 49-class syndrome
      diagnosis (cohorts and number of classes; score bars are schematic).
  (d) Patient-level diagnostic prioritization for one held-out patient with
      several scans (PointNet++, 49-class task, run 1, seed 0): the mean scan
      embedding compared by cosine similarity with the class prototypes (mean
      training embedding per syndrome) and with every training patient (mean of
      that patient's scan embeddings) — the same rules as
      analyses/prioritization/proto_retrieval.py.

Visual grammar: screening outputs blue (score on a track with a threshold and a
call), diagnostic outputs red (ranked list).

Inputs: labelled manifests (data/manifests/), cached features
(results/features/, from facebench/inference/extract_features.py), the FaceBase
mesh of the panel-(a) scan, results/fig1/scan_quality.csv (from
figures/fig1_scan_quality.py; used only to pick the most complete scan to draw
for each patient) and the local exemplar list (figures/exemplars.py).

Usage: python figures/fig1_clinical.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from figures import figstyle as FS                                       # noqa: E402
from figures import figS_overview as F0                     # noqa: E402  (mesh helpers, cohort table)
from figures.exemplars import exemplar                       # noqa: E402
from facebench.paths import FEATURES as FEAT, FIG_OUT as OUTD, MACROS, MANIFESTS as MAN3, RESULTS, SPLITS, facebase  # noqa: E402

TEAL = FS.ENC_COLOR["dgcnn"]   # screening outputs (mid-blue)
TEAL_L = "#DCE6F0"
RED = FS.ACCENT           # diagnosis (also the prototype colour in Figs 3-4)
RED_L = "#F3DAD7"
RBLUE = FS.ENC_COLOR["pointnet2"]           # patient-level retrieval branch, panel (d)
BLUE = "#123F63"
ROLE = F0.ROLE_COLOR      # cleft / syndrome / control
EX_PATIENT = exemplar("fig1", "patient")      # panel (d)
FACE_SCAN = exemplar("fig1", "face_scan")     # panels (a) and (b); held out in both tasks
EX_RUN, EX_SEED = 1, 0                        # training run and inference seed shown
QUAL = pd.read_csv(RESULTS / "fig1/scan_quality.csv")
W, H = FS.DOUBLE, 5.23


# ───────────────────────────── primitives ──────────────────────────────────
def canvas(fig, x, y, w, h):
    """Axes in INCHES (x, y, w, h in inches from the figure's bottom-left)."""
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return ax


def inset(ax, x, y, w, h):
    return ax.inset_axes([x, y, w, h], transform=ax.transData)


def rbox(ax, x, y, w, h, fc="white", ec=FS.INK, lw=0.6, r=0.04, z=2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, ls=ls))


def arr(ax, p0, p1, color=FS.INK, lw=0.7, ms=6, cs="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw,
                                 color=color, shrinkA=0, shrinkB=0, zorder=4,
                                 connectionstyle=cs))


def title(ax, letter, text, sub=None, y=None):
    y = ax.get_ylim()[1] if y is None else y
    ax.text(0, y, letter, fontsize=9, fontweight="bold", va="top", ha="left")
    ax.text(0.17, y - 0.005, text, fontsize=7.6, fontweight="bold", va="top", ha="left")
    if sub:
        ax.text(0.17, y - 0.15, sub, fontsize=6.3, color=FS.MUTED, va="top", ha="left")


def cloud(ax, P, s=0.22, ycut=-0.72):
    """Frontal render of a stored cloud: depth-sorted, depth-shaded."""
    P = P[P[:, 1] > ycut]
    o = np.argsort(P[:, 2]); z = P[o, 2]
    # Full depth range on a colormap TRUNCATED at its light end, so the deepest
    # points (ears, jaw edge) stay visible. With vmin at the
    # 20th percentile and bone_r's white end, the ears rendered as background.
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("bone_trunc", plt.get_cmap("bone_r")(np.linspace(0.30, 0.85, 64)))
    ax.scatter(P[o, 0], P[o, 1], s=s, c=z, cmap=cm, linewidths=0, rasterized=True,
               vmin=z.min(), vmax=np.percentile(z, 99.5))
    ax.set_aspect("equal"); ax.axis("off")


# v15: no longer used; panel (c) now uses screen_result (same glyph as (a), no "refer").
def screen_glyph(ax, x, y, w, score, h=0.075, flag=True, fs=5.0):
    """SCREENING output: score on an unaffected..affected track + threshold + flag."""
    ax.add_patch(Rectangle((x, y), w, h, fc=TEAL_L, ec="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w * score, h, fc=TEAL, ec="none", zorder=3))
    ax.plot([x + w * 0.5] * 2, [y - 0.035, y + h + 0.035], color=FS.INK, lw=0.6, zorder=4)
    ax.text(x, y - 0.03, "unaffected", fontsize=fs - 0.6, color=FS.MUTED, ha="left", va="top")
    ax.text(x + w, y - 0.03, "affected", fontsize=fs - 0.6, color=FS.MUTED, ha="right", va="top")
    if flag:
        fx = x + w + 0.07
        ax.plot([fx, fx], [y - 0.02, y + h + 0.10], color=TEAL, lw=0.8, zorder=4)
        ax.add_patch(plt.Polygon([[fx, y + h + 0.10], [fx + 0.10, y + h + 0.065],
                                  [fx, y + h + 0.03]], fc=TEAL, ec="none", zorder=4))
        ax.text(fx + 0.13, y + h + 0.065, "refer", fontsize=fs - 0.2, color=TEAL, ha="left",
                va="center", fontweight="bold")


def screen_result(ax, x, y, w, score, thr=0.5, h=0.075, fs=5.0, show_val=True):
    """SCREENING output for ONE REAL scan (panel a). The generic
    glyph did not show the call. Here: labelled threshold, a marker at the score,
    and the explicit call to the right."""
    ax.add_patch(Rectangle((x, y), w, h, fc=TEAL_L, ec="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w * score, h, fc=TEAL, ec="none", zorder=3, alpha=0.55))
    tx = x + w * thr
    ax.plot([tx, tx], [y - 0.035, y + h + 0.035], color=FS.INK, lw=0.6, zorder=4)
    sx = x + w * score
    ax.plot(sx, y + h / 2, "o", ms=3.4, mfc=TEAL, mec="white", mew=0.5, zorder=5)
    if show_val:
        ax.text(sx, y + h + 0.045, f"{score:.3f}", fontsize=fs - 0.4, color=TEAL, ha="right",
                va="bottom", fontweight="bold")
    # 2026-11-10: "unaffected"/"affected" end labels dropped;
    # threshold tick + explicit call carry it.
    call = "Affected" if score >= thr else "Unaffected"
    ax.text(x + w + 0.10, y + h / 2, call, fontsize=fs + 0.6, color=TEAL, ha="left",
            va="center", fontweight="bold")


def rank_glyph(ax, x, y, w, labels, vals, row=0.105, hi=None, fs=5.0, bar_w=None,
               vmax=None, show_vals=False, mark="✓"):
    """DIAGNOSIS output: candidates ranked by score, top row first (y = top)."""
    bar_w = bar_w if bar_w is not None else w * 0.32
    vmax = vmax or max(vals)
    for i, (lab, v) in enumerate(zip(labels, vals)):
        yy = y - i * row
        a = 1.0 - 0.55 * i / max(len(labels) - 1, 1)
        ax.text(x, yy, f"{i + 1}", fontsize=fs, color=RED, ha="left", va="center",
                fontweight="bold")
        ax.add_patch(Rectangle((x + 0.07, yy - row * 0.32), bar_w * v / vmax, row * 0.64,
                               fc=RED, alpha=a, ec="none", zorder=3))
        txt = lab + (f"  {v:.2f}" if show_vals else "") + (f"  {mark}" if i == hi and mark else "")
        ax.text(x + 0.10 + bar_w, yy, txt, fontsize=fs, ha="left", va="center",
                color=FS.INK, fontweight="bold" if i == hi else "normal")


# ───────────────────────────── data ────────────────────────────────────────
def task_table():
    """Scans and subjects entering each task (v3 manifests)."""
    ofc = pd.read_csv(MAN3 / "ofc_manifest.csv")
    comb = pd.read_csv(MAN3 / "combined_manifest.csv")
    syn = pd.read_csv(MAN3 / "syndrome_k20_manifest.csv")
    n_ofc_subj = ofc.groupby("dataset").study_id.nunique().sum()
    return {
        "comb": (len(comb), comb.subject_id.nunique()),
        "cleft": (len(ofc), n_ofc_subj),
        "syn": (len(syn), syn.fbid.nunique(), syn.syndrome_name.nunique()),
        "cleft_types": ofc.cleft_type.value_counts().to_dict(),
    }


def example_patient():
    def l2(X):
        return X / np.linalg.norm(X, axis=-1, keepdims=True)
    f = lambda p: np.load(FEAT / f"CF20_full_v3_ts{EX_RUN}__pointnet2__seed{EX_SEED}__{p}.npz",
                          allow_pickle=True)
    tr, te = f("train"), f("test")
    names = [str(n) for n in tr["label_names"]]
    Etr, ytr, itr = tr["embed"].astype(float), tr["labels"], tr["ids"]
    m = te["ids"] == EX_PATIENT
    assert m.sum() >= 2
    y = int(te["labels"][m][0])
    P = l2(np.stack([Etr[ytr == k].mean(0) for k in range(len(names))]))
    q = l2(te["embed"][m].astype(float).mean(0))
    sc = P @ q
    order = np.argsort(-sc)
    subs = list(dict.fromkeys(itr))
    Ref = l2(np.stack([Etr[itr == s].mean(0) for s in subs]))
    lab = np.array([ytr[itr == s][0] for s in subs])
    sim = Ref @ q
    nn = np.argsort(-sim, kind="stable")[:3]
    rank_true = int(1 + (sc > sc[y]).sum())
    return dict(true=names[y], y=y, top=[(names[k], float(sc[k]), k == y) for k in order[:5]],
                rank_true=rank_true, n_ref=len(subs), K=len(sc),
                nn=[(subs[j], names[lab[j]], float(sim[j])) for j in nn])


def face_outputs():
    """Real outputs for FACE_SCAN. Feature rows follow manifest order (verified:
    CF20 test rows == manifest fixed_split=='test'; combined test rows == manifest
    rows of the test subjects)."""
    from scipy.special import softmax
    def l2(X):
        return X / np.linalg.norm(X, axis=-1, keepdims=True)
    c = np.load(FEAT / "C_full_v3_ts1__pointnet2__seed0__test.npz", allow_pickle=True)
    cm = pd.read_csv(MAN3 / "combined_manifest.csv")
    sub = cm[cm.subject_id.isin(set(c["ids"]))].reset_index(drop=True)
    assert (sub.subject_id.values == c["ids"]).all()
    i = sub.index[sub.scan_id == FACE_SCAN][0]
    assert list(c["label_names"]) == ["Unaffected", "Affected"]
    p_aff = float(softmax(c["logits"][i].astype(float))[1])
    s = np.load(FEAT / "CF20_full_v3_ts1__pointnet2__seed0__test.npz", allow_pickle=True)
    tr = np.load(FEAT / "CF20_full_v3_ts1__pointnet2__seed0__train.npz", allow_pickle=True)
    m = pd.read_csv(MAN3 / "syndrome_k20_manifest.csv")
    mt = m[m.fixed_split == "test"].reset_index(drop=True)
    assert (mt.fbid.values == s["ids"]).all()
    j = mt.index[mt.scan_id == FACE_SCAN][0]
    names = [str(n) for n in tr["label_names"]]
    P = l2(np.stack([tr["embed"][tr["labels"] == k].astype(float).mean(0) for k in range(len(names))]))
    sc = P @ l2(s["embed"][j].astype(float))
    o = np.argsort(-sc)[:3]
    y = int(s["labels"][j])
    return dict(p_aff=p_aff, true=names[y], top=[(names[k], float(sc[k]), k == y) for k in o])


def face_mesh():
    """FaceBase mesh of FACE_SCAN, largest component, centred and scaled to unit size."""
    from facebench.preprocessing.mesh_utils import load_face_mesh
    src = pd.read_csv(SPLITS / "scans.csv").set_index("scan_id").loc[FACE_SCAN, "src_path"]
    m = load_face_mesh(facebase(src), process=True)
    m = max(m.split(only_watertight=False), key=lambda c: len(c.vertices))
    V = np.asarray(m.vertices, float)
    V = (V - V.mean(0)) / np.abs(V - V.mean(0)).max()
    return V, m.faces


def clean_scan(fbid):
    """The patient's scan with the most complete surface (display only: a reference
    patient is represented by the mean of ALL their scans)."""
    return QUAL[QUAL.fbid == fbid].sort_values(["hole_frac", "top_spread"]).out_path.iloc[0]


def scan_paths(fbid, n=None):
    from facebench.data.dataset import resolve_out_path
    s = resolve_out_path(pd.read_csv(MAN3 / "syndrome_k20_manifest.csv"))
    p = s[s.fbid == fbid].sort_values("scan_id").out_path.tolist()
    return p if n is None else p[:n]


SHORT = {"Coffin-Lowry Syndrome": "Coffin–Lowry syndrome",
         "Pitt-Hopkins Syndrome": "Pitt–Hopkins syndrome",
         "Rett Syndrome": "Rett syndrome",
         "Osteogenesis Imperfecta": "Osteogenesis imperfecta",
         "1p36 Deletion": "1p36 deletion",
         "Down Syndrome": "Down syndrome",
         "Pierre Robin Sequence": "Pierre Robin sequence",
         "Wolf-Hirschhorn syndrome": "Wolf–Hirschhorn syndrome",
         "Marfan Syndrome": "Marfan syndrome",
         "Klinefelter Syndrome": "Klinefelter syndrome",
         "Treacher Collins Syndrome": "Treacher Collins syndrome",
         "Rubinstein-Taybi Syndrome": "Rubinstein–Taybi syndrome",
         "Loeys-Dietz Syndrome": "Loeys–Dietz syndrome",
         "5p Deletion Cri du Chat": "5p deletion (Cri du Chat)",
         "Jacobsen Syndrome": "Jacobsen syndrome",
         "Phelan McDermid Syndrome": "Phelan–McDermid syndrome",
         "Sotos Syndrome": "Sotos syndrome",
         "Coffin-Siris Syndrome": "Coffin–Siris syndrome"}
short = lambda n: SHORT.get(n, n)


# ───────────────────────────── panels ──────────────────────────────────────
def panel_a(fig, x0, y0, w, h):
    ax = canvas(fig, x0, y0, w, h)
    title(ax, "a", "Clinical motivation")
    V, Fc = face_mesh()
    Vd = V
    fw = 0.70
    fh = fw * (Vd[:, 1].max() - Vd[:, 1].min()) / (Vd[:, 0].max() - Vd[:, 0].min())
    AV = h - 0.20      # usable height below the panel title
    fy = AV / 2 - fh / 2 + 0.02
    ia = inset(ax, 0.05, fy, fw, fh)
    ia.set_xlim(Vd[:, 0].min(), Vd[:, 0].max()); ia.set_ylim(Vd[:, 1].min(), Vd[:, 1].max())
    ia.set_aspect("equal"); ia.axis("off")
    F0.draw_mesh(ia, V, Fc)
    ax.text(0.05 + fw / 2, fy - 0.04, "3D facial scan", fontsize=5.6, color=FS.MUTED,
            ha="center", va="top")
    # two questions — REAL outputs for this held-out scan
    O = face_outputs()
    print("  (a) outputs:", O)
    qx, qw = 0.92, w - 0.95
    ys = [AV * 0.73, AV * 0.25]
    for (yc, col, colL, head, q) in [
            (ys[0], TEAL, TEAL_L, "Screening", "Is a craniofacial condition present?"),
            (ys[1], RED, RED_L, "Diagnosis", "Which genetic syndromes should be considered?")]:
        rbox(ax, qx, yc - 0.30, qw, 0.60, fc=colL, ec=col, lw=0.6)
        ax.text(qx + 0.07, yc + 0.22, head, fontsize=6.4, fontweight="bold", color=col, va="center")
        ax.text(qx + 0.07, yc + 0.08, q, fontsize=5.1, va="center")
        arr(ax, (0.05 + fw + 0.03, fy + fh / 2), (qx - 0.01, yc), cs="arc3,rad=0")
    screen_result(ax, qx + 0.10, ys[0] - 0.19, qw - 0.95, score=O["p_aff"])
    hi = [t for _, _, t in O["top"]].index(True)
    rank_glyph(ax, qx + 0.07, ys[1] - 0.06, qw - 0.15,
               [short(n) for n, _, _ in O["top"]], [v for _, v, _ in O["top"]], row=0.085,
               hi=hi, bar_w=0.40, vmax=max(v for _, v, _ in O["top"]), show_vals=True)
    return ax


def panel_b(fig, x0, y0, w, h, df):
    ax = canvas(fig, x0, y0, w, h)
    title(ax, "b", "FaceBase cohort and 3D representation learning")
    top = h - 0.36
    # cohort bars, grouped by role
    order = ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]
    d = df.set_index("dataset").loc[order]
    bx, bw, row = 0.42, 0.62, 0.155
    vmax = d.scans.max()
    ax.text(0.0, top + 0.02, "Six cohorts", fontsize=6.0, fontweight="bold", va="bottom")
    ax.text(bx + bw + 0.02, top + 0.02, "scans / subjects", fontsize=4.8, color=FS.MUTED,
            va="bottom", ha="left")
    for i, (ds, r) in enumerate(d.iterrows()):
        yy = top - 0.10 - i * row - (0.04 if i >= 2 else 0) - (0.04 if i >= 3 else 0)
        ax.text(bx - 0.04, yy, ds, fontsize=5.3, ha="right", va="center")
        ax.add_patch(Rectangle((bx, yy - 0.045), bw * r.scans / vmax, 0.09,
                               fc=ROLE[r.role], ec="none"))
        ax.text(bx + bw * r.scans / vmax + 0.03, yy, f"{r.scans:,} / {r.subjects:,}",
                fontsize=4.8, va="center", color=FS.INK)
    ky = top - 0.10 - 5 * row - 0.08 - 0.17
    for j, (k, lab) in enumerate([("cleft", "cleft"), ("syndrome", "syndrome"),
                                  ("control", "unaffected")]):
        kx = 0.10 + [0, 0.36, 0.86][j]
        ax.add_patch(Rectangle((kx, ky - 0.03), 0.06, 0.06, fc=ROLE[k], ec="none"))
        ax.text(kx + 0.08, ky, lab, fontsize=4.8, va="center")
    # point cloud
    P = np.load(QUAL.set_index("scan_id").loc[FACE_SCAN, "out_path"])
    cx = 1.62
    cw = 0.52
    ic = inset(ax, cx, top - 0.92, cw, 0.74)
    cloud(ic, P, s=0.09, ycut=-9)
    arr(ax, (1.42, top - 0.55), (cx - 0.03, top - 0.55))
    ax.text(cx + cw / 2, top - 0.96, "4,096 points", fontsize=5.2,
            ha="center", va="top", color=FS.INK)
    # 2026-11-10: "no landmarks or template" dropped (in caption)
    # encoders
    ex, ew, eh = 2.40, 0.62, 0.17
    arr(ax, (cx + cw + 0.04, top - 0.55), (ex - 0.03, top - 0.55))
    ax.text(ex + ew / 2, top - 0.04, "encoder", fontsize=5.6, ha="center", va="bottom",
            fontweight="bold")
    for i, k in enumerate(["geommlp", "pointnet", "pointnet2", "dgcnn"]):
        yy = top - 0.24 - i * (eh + 0.05)
        c = FS.ENC_COLOR[k]
        rbox(ax, ex, yy - eh / 2, ew, eh, fc=c, ec="none", r=0.03)
        dark = k in ("pointnet2", "dgcnn")
        ax.text(ex + ew / 2, yy, FS.ENC_LABEL[k], fontsize=5.4, ha="center", va="center",
                color="white" if dark else FS.INK)
    # 2026-11-10: "each evaluated separately" dropped (in caption)
    zx = ex + ew + 0.20
    arr(ax, (ex + ew + 0.03, top - 0.55), (zx - 0.02, top - 0.55))
    for i in range(7):
        ax.add_patch(Rectangle((zx, top - 0.55 - 0.245 + i * 0.07), 0.08, 0.06,
                               fc=BLUE, alpha=0.25 + 0.1 * (i % 4), ec="none"))
    ax.text(zx + 0.04, top - 0.55 - 0.30, "facial\nembedding", fontsize=5.0, ha="center",
            va="top", linespacing=1.1)
    return ax


def task_card(ax, x, y, w, h, kind, name, sub, cohorts, n, glyph, tag):
    col, colL = (TEAL, TEAL_L) if kind == "screen" else (RED, RED_L)
    rbox(ax, x, y, w, h, fc="white", ec=col, lw=0.8)
    # One-line header: task name left, type tag right.
    ax.add_patch(FancyBboxPatch((x, y + h - 0.20), w, 0.20,
                                boxstyle="round,pad=0,rounding_size=0.04", fc=colL, ec="none",
                                zorder=2.5))
    ax.text(x + 0.06, y + h - 0.10, name, fontsize=6.0, fontweight="bold", color=col,
            va="center", zorder=3)
    ax.text(x + w - 0.06, y + h - 0.10, tag, fontsize=4.6, color=col, va="center", ha="right",
            style="italic", zorder=3)
    ax.text(x + 0.06, y + h - 0.31, sub, fontsize=5.0, va="center")
    cx = x + 0.06
    for ds, role in cohorts:
        ax.add_patch(Rectangle((cx, y + h - 0.47), 0.045, 0.045, fc=ROLE[role], ec="none",
                               zorder=3))
        ax.text(cx + 0.06, y + h - 0.448, ds, fontsize=4.4, va="center")
        cx += 0.06 + 0.032 * len(ds) + 0.04
    # 2026-11-10: K on the cohort row (right-aligned), bars move up.
    ax.text(x + w - 0.06, y + h - 0.448, n, fontsize=4.6, va="center", ha="right",
            color=FS.MUTED)
    glyph(x + 0.06, y + 0.04)


def panel_c(fig, x0, y0, w, h, T):
    ax = canvas(fig, x0, y0, w, h)
    title(ax, "c", "Clinical benchmark tasks")
    gap = 0.10
    cw = (w - 3 * gap) / 4
    ch = h - 0.35      # card height; r2h 1.41 -> 1.30 removes the old K row
    cy = 0.18
    CL = [("FB-5A", "cleft"), ("FB-56", "cleft")]
    ALL6 = CL + [("FB-TJ0", "syndrome"), ("3 controls", "control")]
    ct = T["cleft_types"]
    cards = [
        ("screen", "Combined screening", "Any craniofacial condition vs unaffected", ALL6,
         f"$K$=2",
         lambda gx, gy: screen_result(ax, gx + 0.05, gy + 0.12, cw - 0.74, 0.86, show_val=False)),
        ("screen", "Cleft screening", "Orofacial cleft vs unaffected", CL,
         f"$K$=2",
         lambda gx, gy: screen_result(ax, gx + 0.05, gy + 0.12, cw - 0.74, 0.22, show_val=False)),
        ("dx", "Cleft subtype", "Unaffected, CL, CP or CLP", CL,
         f"$K$=4",
         lambda gx, gy: rank_glyph(ax, gx, gy + 0.27, cw - 0.12,
                                   ["cleft lip and palate", "cleft lip", "cleft palate",
                                    "unaffected"], [0.55, 0.25, 0.12, 0.08], row=0.068,
                                   fs=4.5, bar_w=0.30)),
        ("dx", "Syndrome diagnosis", f"{T['syn'][2]} genetic syndromes", [("FB-TJ0", "syndrome")],
         f"$K$={T['syn'][2]}",
         lambda gx, gy: rank_glyph(ax, gx, gy + 0.27, cw - 0.12,
                                   ["syndrome A", "syndrome B", "syndrome C",
                                    f"… to {T['syn'][2]}"], [0.42, 0.30, 0.24, 0.05],
                                   row=0.068, fs=4.5, bar_w=0.30)),
    ]
    # 2026-11-10: italic type tags dropped (blue/red carries it);
    # card count line is K only (scans/subjects in Table 1).
    tags = ["", "", "", ""]
    for i, (kind, name, sub, coh, n, g) in enumerate(cards):
        task_card(ax, i * (cw + gap), cy, cw, ch, kind, name, sub, coh, n, g, tags[i])
    # specificity arrow, labelled in three stages under the cards they describe
    ax.add_patch(FancyArrowPatch((0.0, 0.07), (w, 0.07), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=FS.MUTED, shrinkA=0, shrinkB=0))
    # 2026-11-10: one arrow label replaces subtitle + three stage labels.
    ax.text(w / 2, 0.095, "Increasing diagnostic specificity", fontsize=4.9, color=FS.MUTED,
            ha="center", va="bottom")
    return ax


def panel_d(fig, x0, y0, w, h, E):
    """v14: (1) confirmed diagnosis shown as a badge next to the
    scans; (2)/(4) the two cosine comparisons drawn as separate framed branches,
    Syndrome-level ranking (vs diagnosis prototypes) and Patient-level retrieval
    (vs individual confirmed reference patients), each naming its output;
    (3) outcome box "Candidate diagnoses for expert review" (the model does not
    establish a differential independently; the confirmed label is a
    retrospective check only)."""
    ax = canvas(fig, x0, y0, w, h)
    title(ax, "d", "Patient-level diagnostic prioritization")
    top = h - 0.30
    # ---- confirmed diagnosis badge
    rbox(ax, 0.04, top - 0.05, 1.88, 0.17, fc=RED, ec="none", r=0.03)
    ax.text(0.98, top + 0.035, f"Confirmed diagnosis: {short(E['true']).replace('-', '–')}", fontsize=5.0,
            color="white", fontweight="bold", ha="center", va="center")
    # ---- repeated scans
    paths = scan_paths(EX_PATIENT)
    n = len(paths)
    stop = top - 0.12
    step = (stop - 0.50) / (n - 1)
    sh = min(0.53, step * 1.05)   # 0.46 -> 0.53
    sw = sh * 0.84
    ys = [stop - 0.25 - i * step for i in range(n)]
    for i, (p, yy) in enumerate(zip(paths, ys)):
        ic = inset(ax, 0.04, yy - sh / 2, sw, sh)
        cloud(ic, np.load(p), s=0.07)
    ax.text(0.04, stop - 0.02, f"{n} scans, one patient", fontsize=5.0, ha="left",
            va="bottom")
    # encoder -> z_i -> average box
    ex = 0.60
    rbox(ax, ex, ys[-1] - 0.12, 0.30, ys[0] - ys[-1] + 0.24, fc=FS.ENC_COLOR["pointnet2"], ec="none")
    ax.text(ex + 0.15, (ys[0] + ys[-1]) / 2, "PointNet++", rotation=90, fontsize=5.4,
            color="white", ha="center", va="center")
    zx = 1.08
    bxl, bw_, bh_ = 1.36, 0.42, 0.34
    my = (ys[0] + ys[-1]) / 2
    for i, yy in enumerate(ys):
        arr(ax, (0.04 + sw + 0.01, yy), (ex - 0.02, yy), lw=0.5, ms=4)
        arr(ax, (ex + 0.32, yy), (zx - 0.05, yy), lw=0.5, ms=4)
        ax.plot(zx, yy, "o", ms=3.0, mfc="white", mec=RED, mew=0.7, zorder=5)
        ax.text(zx, yy + 0.08, f"$\\mathbf{{z}}_{i + 1}$", fontsize=5.0, ha="center",
                va="bottom", color=RED)
        ax.plot([zx + 0.03, bxl], [yy, my - (i - (n - 1) / 2) * 0.07], color=RED, lw=0.5)
    rbox(ax, bxl, my - bh_ / 2, bw_, bh_, fc="white", ec=RED, lw=0.7, r=0.03)
    ax.text(bxl + bw_ / 2, my + 0.08, "average", fontsize=4.8, color=RED, ha="center",
            va="center")
    ax.text(bxl + bw_ / 2, my - 0.05, "$\\bar{\\mathbf{z}}$", fontsize=6.5, color=RED,
            ha="center", va="center")
    ax.text(bxl + bw_ / 2, my - bh_ / 2 - 0.05, "Patient-level\nfacial embedding", fontsize=4.7,
            ha="center", va="top", linespacing=1.1, fontweight="bold")
    xo = bxl + bw_ + 0.02
    # ---- two framed branches
    bx0, bx1 = 1.98, 4.50
    def branch(yb, yt, name, sub, out, ec):
        rbox(ax, bx0, yb, bx1 - bx0, yt - yb, fc="white", ec=ec, lw=0.7, r=0.03)
        ax.text(bx0 + 0.07, yt - 0.08, name, fontsize=5.7, fontweight="bold", va="center")
        ax.text(bx1 - 0.07, yt - 0.08, out, fontsize=4.4, style="italic", color=ec,
                ha="right", va="center")
        ax.text(bx0 + 0.07, yt - 0.19, sub, fontsize=4.5, color=FS.MUTED, va="center")
        return (yb + yt) / 2
    b1y = branch(0.99, top + 0.08, "Syndrome-level ranking",
                 f"vs. {E['K']} syndrome prototypes",
                 "Output: ranked diagnostic candidates", RED)
    labs = [short(n) for n, _, _ in E["top"]]
    vals = [v for _, v, _ in E["top"]]
    hi = [t for _, _, t in E["top"]].index(True)
    rank_glyph(ax, bx0 + 0.07, top + 0.08 - 0.33, 1.6, labs, vals, row=0.098, hi=hi, fs=5.0,
               bar_w=0.55, vmax=max(vals), show_vals=True)
    b2y = branch(0.02, 0.93, "Patient-level retrieval",
                 f"vs. {E['n_ref']:,} confirmed reference patients",
                 "Output: similar reference cases", RBLUE)
    arr(ax, (xo, my + 0.05), (bx0 - 0.03, b1y), cs="arc3,rad=-0.25")
    arr(ax, (xo, my - 0.05), (bx0 - 0.03, b2y), color=RBLUE, cs="arc3,rad=0.25")
    WRAP = {"Treacher Collins Syndrome": "Treacher\nCollins\nsyndrome",
            "Coffin-Lowry Syndrome": "Coffin–Lowry\nsyndrome",
            "Sotos Syndrome": "Sotos\nsyndrome",
            "Pitt-Hopkins Syndrome": "Pitt–Hopkins\nsyndrome"}
    for i, (sid, dx, s) in enumerate(E["nn"]):
        px = bx0 + 0.07 + i * 0.80
        P = np.load(clean_scan(sid))
        ic = inset(ax, px - 0.02, 0.03, 0.40, 0.58)   # was 0.34x0.50
        cloud(ic, P, s=0.06)
        same = dx == E["true"]
        c = RED if same else FS.INK
        ax.text(px + 0.39, 0.50, f"#{i + 1}", fontsize=5.0, fontweight="bold", color=RBLUE,
                va="center")
        ax.text(px + 0.39, 0.31, WRAP.get(dx, short(dx)), fontsize=4.6, va="center",
                linespacing=1.05, color=RBLUE)
        ax.text(px + 0.39, 0.12, f"cos {s:.2f}", fontsize=4.4, color=FS.MUTED, va="center")
    # ---- outcome box: "Potential information for expert review".
    # The two outputs stay in SEPARATE sections (each fed by its own arrow, dashed
    # divider) so the box does not read as a computational fusion of the branches.
    # The cohort-level top-5 statistic is removed (it belongs to Fig. prior).
    ox, ow = 4.62, w - 4.64
    oy, oh = 0.02, top + 0.08 - 0.02
    rbox(ax, ox, oy, ow, oh, fc=RED_L, ec=RED, lw=0.8)
    yt = oy + oh
    ax.text(ox + 0.09, yt - 0.08, "Potential information\nfor expert review", fontsize=6.2,
            fontweight="bold", color=RED, va="top", linespacing=1.1)
    # 2026-11-10: "from ..." source lines dropped (the
    # arrows and colours give the source), retrieved cases collapsed to one line,
    # the 3-line retrospective sentence replaced by a check-mark key (sentence is in
    # the caption).
    # section 1: ranked candidates (prototype inference)
    s1 = b1y + 0.02     # clears the two-line box title
    arr(ax, (bx1 + 0.01, b1y), (ox - 0.03, b1y))
    ax.text(ox + 0.11, s1, "Ranked candidate syndromes", fontsize=5.3, fontweight="bold",
            va="center")
    for i, (nm, v, t) in enumerate(E["top"][:3]):
        ax.text(ox + 0.15, s1 - 0.14 - i * 0.12, f"{i + 1}  {short(nm)}" + ("  ✓" if t else ""),
                fontsize=5.0, va="center", fontweight="bold" if t else "normal")
    ax.plot([ox + 0.11, ox + ow - 0.11], [(s1 - 0.50 + b2y + 0.24) / 2] * 2, color=RED,
            lw=0.5, ls=(0, (2, 2)))
    # section 2: similar confirmed cases (patient-level retrieval)
    s2 = b2y + 0.12
    arr(ax, (bx1 + 0.01, b2y), (ox - 0.03, b2y), color=RBLUE)
    ax.text(ox + 0.11, s2, "Similar confirmed cases", fontsize=5.3, fontweight="bold",
            va="center")
    dxs = [d for _, d, _ in E["nn"]]
    if len(set(dxs)) == 1:
        lines = [(f"#1–#{len(dxs)}  {short(dxs[0])}", dxs[0] == E["true"])]
    else:
        lines = [(f"#{k + 1}  {short(d)}", d == E["true"]) for k, d in enumerate(dxs)]
    for k, (t, ok) in enumerate(lines):
        ax.text(ox + 0.15, s2 - 0.14 - k * 0.12, t + ("  ✓" if ok else ""), fontsize=5.0,
                va="center", fontweight="bold" if ok else "normal")
    ax.text(ox + 0.11, oy + 0.08, "✓ confirmed diagnosis", fontsize=4.5, color=FS.MUTED,
            va="bottom")
    return ax


def check_against_manuscript(df, T):
    """Panel (b) counts must equal the Table 1 macros (results/paper_macros/)."""
    import re
    tex = "\n".join(f.read_text() for f in sorted(MACROS.glob("numbers*.tex")))
    def mac(n):
        m = re.findall(r"\\(?:re)?newcommand\{\\" + n + r"\}\{((?:[^{}]|\{[^{}]*\})*)\}", tex)
        assert m, n
        return int(m[-1].replace("{,}", ""))
    d = df.set_index("dataset")
    for ds, (a, b) in {"FB-5A": ("dsFiveA", "dsFiveASubj"), "FB-56": ("dsFiveSix", "dsFiveSixSubj"),
                       "FB-TJ0": ("dsTJZeroAll", "dsTJZeroSubj"), "FB-TK0": ("dsTKZero", "dsTKZeroSubj"),
                       "FB-VWP": ("dsVWP", "dsVWPSubj"), "FB-TX4": ("dsTXFour", "dsTXFourSubj")}.items():
        assert (d.loc[ds, "scans"], d.loc[ds, "subjects"]) == (mac(a), mac(b)), ds
    assert (d.scans.sum(), d.subjects.sum()) == (mac("NScansProc"), mac("NSubjectsProc"))
    assert T["syn"][2] == 49
    print("  (b) counts match the Table 1 macros")


def main():
    FS.apply()
    plt.rcParams["figure.constrained_layout.use"] = False
    df = F0.cohort_table()
    T = task_table()
    check_against_manuscript(df, T)
    E = example_patient()
    print("example:", E)
    assert E["rank_true"] <= 3 and E["nn"][0][1] == E["true"], "example no longer as described"
    fig = plt.figure(figsize=(W, H))
    r1h, r2h, r3h = 1.62, 1.30, 2.02
    gap = 0.12
    y3 = 0.02
    y2 = y3 + r3h + gap
    y1 = y2 + r2h + gap
    panel_a(fig, 0.02, y1, 2.85, r1h)
    panel_b(fig, 3.02, y1, W - 3.04, r1h, df)
    panel_c(fig, 0.02, y2, W - 0.04, r2h, T)
    panel_d(fig, 0.02, y3, W - 0.04, r3h, E)
    print(f"  rows end at {y1 + r1h:.2f} in of H={H}")
    OUTD.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTD / "fig1_clinical.pdf")
    FS.finish(fig, OUTD / "fig1_clinical.png", FS.DOUBLE)


if __name__ == "__main__":
    main()
