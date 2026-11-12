"""
Region-occlusion interpretability figures: paper Figure 5 and Figures S5-S6.

  figures/fig_interp_main.{png,pdf}                         (Figure 5)
      (a) one consistently correct test subject per syndrome (subjects chosen by the authors from the consistently correct cases):
          3D facial scan | anatomical regions | importance map | region scores
      (b) syndrome x region heatmap over ALL test subjects of each syndrome (correct or
          not): cell = mean importance, text = mean and % of subjects with positive
          importance, * = one-sample t-test over subjects, Benjamini-Hochberg q < 0.05
          across the 35 displayed syndrome x region cells (2026-10-23; previously the
          uncorrected t-based 95% CI; the bootstrap CI, also in the csv, is too liberal at
          n = 5-7)
  figures/figS_interp_misclassified.{png,pdf}               (Figure S6)
      consistently misclassified subjects grouped by failure mode (A/B/C), same columns
      plus the 3 most similar training faces
  figures/figS_interp_retrieval.{png,pdf}                   (Figure S5)
      the main-figure subjects with their 3 most similar training faces
  results/interp/tables/syndrome_region_all_subjects.csv
      mean, 95% CI, t-test p, % positive per syndrome x region; all subjects and correct-only;
      q_bh (BH over the 35 displayed cells) on the displayed all-subject rows

Reads results/interp/{pilot,ext}_scans.csv, cases{,_ext}.csv, nn_scans{,_ext}.csv,
{pilot,ext}/region_occlusion.csv (run occlusion.py first), the landmark files
results/interp/landmarks*/ (landmark_scans.py; not shipped), the CF20_full_v3
PointNet++ prediction files, and the raw FaceBase meshes (FACEBASE_ROOT) for the
geometry-only renders. Rendering drops detached mesh fragments and, in the
supplementary figures, trims surface far from the landmarks; display only, the
model input and every score are unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from scipy import stats                                             # noqa: E402
from scipy.spatial import cKDTree                                   # noqa: E402

from analyses.interpretability.interp_lib import REGIONS, ALL_REGIONS, landmark_regions, point_regions  # noqa: E402
from analyses.interpretability import render_lib as RL                                             # noqa: E402
from analyses.interpretability.interp_lib import abs_paths                                    # noqa: E402
from figures.exemplars import exemplar                               # noqa: E402
from analyses.interpretability.panels_common import (IN, IMG, IMP_CMAP, REG_COLS, LM_DIRS, LM_NN_DIRS,  # noqa: E402
                                 find, geometry)

from facebench.paths import FIG_OUT as FIG, ROOT as PROJECT  # noqa: E402
# Exemplar subjects come from the local data/exemplars.csv (not distributed; see
# figures/exemplars.py). The syndromes are fixed by the analysis.
PICK_SYNDROMES = ["Achondroplasia", "Cockayne Syndrome", "Down Syndrome",
                  "Pallister-Killian Syndrome", "Rhizomelic Chondrodysplasia Punctata"]
WRONG_GROUPS = [("A  Clinically related syndrome confusion",
                 ["Cardiofaciocutaneous Syndrome", "Marfan Syndrome"], "A"),
                ("B  Incorrect diagnosis despite positive true-label region importance",
                 ["Down Syndrome"], "B"),
                ("C  Mixed-label nearest-neighbor retrieval",
                 ["Ehlers Danlos Syndrome", "Williams-Beuren Syndrome"], "C")]


def _picks():
    return [(sy, exemplar("fig_interp", sy)) for sy in PICK_SYNDROMES]


def _wrong():
    return [(g, [(sy, exemplar("fig_interp_wrong", f"{k}:{sy}")) for sy in syns])
            for g, syns, k in WRONG_GROUPS]


SHORT = {"Rhizomelic Chondrodysplasia Punctata": "Rhizomelic CDP", "Cardiofaciocutaneous Syndrome": "CFC",
         "X-Linked Hypohidrotic Ectodermal Dysplasia": "X-linked HED", "22q 11.2 Deletion Syndrome": "22q11.2 del.",
         "Ehlers Danlos Syndrome": "Ehlers-Danlos"}
LIM = 0.16                                   # colour limit of the importance map (saturates above)
N_BOOT = 10000
WIDTHS = [1, 1, 1, .42, 1.05]                # scan | regions | map | label gap | bars
NN_WIDTHS = [.14, 1, 1, 1]                   # gap | NN1-3
MIN_PART = 0.01                              # rendering only: drop detached mesh fragments < 1% of faces
TRIM = 0.25                                  # supp figures, rendering only: keep mesh within TRIM x landmark span
                                             # of the nearest landmark (cuts attached hair/headband/backdrop)


def pretty(r):
    return "Chin/jaw" if r == "chin_jaw" else r.capitalize()


def drop_fragments(F):
    """Faces of mesh components with >= MIN_PART of all faces (removes floating scanner noise, e.g. above
    the head in one Cockayne exemplar; display only, the model input cloud is untouched)."""
    import trimesh
    m = trimesh.Trimesh(np.zeros((F.max() + 1, 3)), F, process=False)
    lab = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(F))
    cnt = np.bincount(lab)
    return F[cnt[lab] >= MIN_PART * len(F)]


def trim(Vn, F, lm):
    """Faces whose vertices all lie within TRIM x landmark span of a landmark (display only)."""
    d = cKDTree(lm).query(Vn)[0]
    return F[(d[F] < TRIM * np.ptp(lm[:, :2], 0).max()).all(1)]


def face_render(scan_id, row, lm_dir, img=300):
    """Plain grey render as in interp_analyze.scan_render, with fragments dropped and noise trimmed."""
    lm, Vn, F, R, crop = geometry(scan_id, row, lm_dir)
    return RL.render_scalar(Vn, trim(Vn, drop_fragments(F), lm), R, crop, img=img)


def short(s):
    return SHORT.get(s, s.replace(" Syndrome", ""))


def load():
    scans = abs_paths(pd.concat([pd.read_csv(IN / "pilot_scans.csv"), pd.read_csv(IN / "ext_scans.csv")])) \
        .set_index("scan_id")
    cases = pd.concat([pd.read_csv(IN / "cases.csv"), pd.read_csv(IN / "cases_ext.csv")])
    cases["fbid"] = cases.fbid.astype(str)
    nn = abs_paths(pd.concat([pd.read_csv(IN / "nn_scans.csv"), pd.read_csv(IN / "nn_scans_ext.csv")])) \
        .drop_duplicates("scan_id").set_index("scan_id")
    oc = pd.concat([pd.read_csv(IN / "pilot/region_occlusion.csv"), pd.read_csv(IN / "ext/region_occlusion.csv")])
    imp = oc.groupby(["scan_id", "region"]).importance.mean().unstack()[ALL_REGIONS]
    return scans, cases, nn, imp


def case_scan(cases, imp, syn, fb):
    c = cases[(cases.syndrome == syn) & (cases.fbid == fb)].iloc[0]
    return c, [x for x in c.scans.split(";") if x in imp.index][0]   # first scan, as in the panels


def draw_row(fig, gs, r, c, s, row, v, nn, bar_lim, with_nn, fs=8):
    n_face = len(REGIONS)
    ax = [fig.add_subplot(gs[r, j]) for j in ((0, 1, 2, 4, 6, 7, 8) if with_nn else (0, 1, 2, 4))]
    lm, Vn, F, R, crop = geometry(s, row, find(LM_DIRS, s))
    F = drop_fragments(F)
    cloud_F = F
    if with_nn:                                       # supplementary figure: also trim attached noise
        F = trim(Vn, F, lm)
    cloud = np.load(row["out_path"]).astype(float)
    preg = point_regions(cloud, lm)
    ax[0].imshow(RL.render_scalar(Vn, F, R, crop, img=IMG))
    xy = RL.project(lm, R, crop, IMG)
    ax[0].scatter(xy[:, 0], xy[:, 1], s=2.0, c=REG_COLS[landmark_regions(lm)], lw=0)
    on = np.zeros(len(Vn), bool); on[cloud_F.ravel()] = True
    show = on[cKDTree(Vn).query(cloud)[1]]            # hide cloud points on the dropped fragments (display only)
    if with_nn:                                       # supplementary figure: same landmark-distance trim
        show &= cKDTree(lm).query(cloud)[0] < TRIM * np.ptp(lm[:, :2], 0).max()
    q = cloud @ R.T
    o = np.argsort(q[:, 2]); o = o[show[o]]
    ax[1].scatter(q[o, 0], q[o, 1], c=REG_COLS[preg[o]], s=4)
    ax[1].set_xlim(crop[0], crop[1]); ax[1].set_ylim(crop[2], crop[3])
    ax[1].set_aspect("equal")
    vmap = np.append(v[:n_face], 0.0)                 # outside the face uncoloured
    _, j = cKDTree(cloud).query(Vn)
    ax[2].imshow(RL.render_signed(Vn, F, R, crop, vmap[preg[j]], LIM, IMP_CMAP, img=IMG))
    ax[3].barh(range(n_face), v[:n_face], color=REG_COLS[:n_face], ec="k", lw=.4)
    ax[3].set_yticks(range(n_face)); ax[3].set_yticklabels([pretty(x) for x in REGIONS], fontsize=fs)
    ax[3].axvline(0, c="k", lw=.6); ax[3].invert_yaxis()
    ax[3].set_xlim(-bar_lim, bar_lim); ax[3].set_box_aspect(1.0)
    ax[3].set_xticks([-0.2, -0.1, 0, 0.1, 0.2] if bar_lim >= 0.2 else [-0.1, 0, 0.1])
    ax[3].tick_params(axis="x", labelsize=fs - 1)
    if with_nn:
        for k in range(3):
            a = ax[4 + k]
            ns = c[f"nn{k+1}_scan"]
            d = find(LM_NN_DIRS, ns)
            if d is not None:
                a.imshow(face_render(ns, nn.loc[ns], d))
            else:
                a.text(.5, .5, "no landmarks", ha="center", transform=a.transAxes)
            a.set_title(f"#{k+1} {short(c[f'nn{k+1}_label'])}  ({c[f'nn{k+1}_cos']:.2f})", fontsize=fs,
                        color="green" if c[f"nn{k+1}_label"] == c.syndrome else "firebrick")
    for a in ax[:3] + ax[4:]:
        a.axis("off")
    return ax


def colour_bar(fig, ax_map, ax_bar, fs=8):
    ax_bar.set_xlabel("importance (vs. random patch)", fontsize=fs)
    cax = ax_map.inset_axes([0.1, -0.08, 0.8, 0.05])
    sm = plt.cm.ScalarMappable(cmap=IMP_CMAP, norm=plt.Normalize(-LIM, LIM))
    fig.colorbar(sm, cax=cax, orientation="horizontal").ax.tick_params(labelsize=fs - 1)


def headers(fig, ax, with_nn, bold=False, fs=10):
    for a, t in zip(ax[:4], ["3D facial scan", "Anatomical regions", "Importance map", "Region scores"]):
        a.set_title(t + ("\n" if with_nn else ""), fontsize=fs, fontweight="bold" if bold else "normal")
    if with_nn:
        fig.canvas.draw()
        b4, b6 = ax[4].get_position(), ax[6].get_position()
        fig.text((b4.x0 + b6.x1) / 2, b4.y1 + 0.022, "Most similar training faces (similarity)",
                 ha="center", va="bottom", fontsize=fs, fontweight="bold" if bold else "normal")


def row_label(fig, ax0, text):
    b = ax0.get_position()
    fig.text(b.x0 - 0.008, (b.y0 + b.y1) / 2, text, rotation=90, ha="right", va="center",
             fontsize=10, fontweight="bold")


def names_left(fig, ax0s, texts):
    """Horizontal syndrome labels left of the scans, right-aligned on one common edge.
    Returns the left edge of the widest label (figure fraction)."""
    fig.canvas.draw()
    x = min(a.get_position().x0 for a in ax0s) - 0.005
    tx = []
    for a, t in zip(ax0s, texts):
        b = a.get_position()
        tx.append(fig.text(x, (b.y0 + b.y1) / 2, t, ha="right", va="center", fontsize=12))
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    return min(t.get_window_extent(r).transformed(fig.transFigure.inverted()).x0 for t in tx)


def syndrome_table(imp, scans, cases, syns):
    """Subject-level importance (mean over the subject's scans) -> syndrome x region stats."""
    rng = np.random.default_rng(0)
    fb = scans.fbid.astype(str)
    sub = imp[REGIONS].groupby(imp.index.map(fb)).mean()
    cat = cases.set_index("fbid").category
    rows = []
    for syn in syns:
        ids = scans[scans.syndrome_name == syn].fbid.astype(str).unique()
        for subset, keep in (("all", ids), ("consistent_correct", [i for i in ids if cat.get(i) == "consistent_correct"])):
            M = sub.loc[sub.index.isin(keep)].values
            bs = M[rng.integers(0, len(M), (N_BOOT, len(M)))].mean(1)
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            se = M.std(0, ddof=1) / np.sqrt(len(M))
            tq = stats.t.ppf(0.975, len(M) - 1)
            pv = stats.ttest_1samp(M, 0, axis=0).pvalue
            for k, rg in enumerate(REGIONS):
                rows.append(dict(syndrome=syn, subset=subset, region=rg, n_subjects=len(M),
                                 mean=M[:, k].mean(), t_ci_lo=M[:, k].mean() - tq * se[k],
                                 t_ci_hi=M[:, k].mean() + tq * se[k], boot_ci_lo=lo[k], boot_ci_hi=hi[k],
                                 p=pv[k], frac_positive=(M[:, k] > 0).mean()))
    t = pd.DataFrame(rows)
    fam = (t.subset == "all") & t.syndrome.isin(PICK_SYNDROMES)   # the 35 cells of Fig. b
    t.loc[fam, "q_bh"] = stats.false_discovery_control(t.loc[fam, "p"], method="bh")
    t.to_csv(IN / "tables/syndrome_region_all_subjects.csv", index=False)
    return t


def heatmap(fig, ax, t):
    syns = list(PICK_SYNDROMES)
    a = t[t.subset == "all"]
    M = a.pivot(index="syndrome", columns="region", values="mean").loc[syns, REGIONS]
    P = a.pivot(index="syndrome", columns="region", values="frac_positive").loc[syns, REGIONS]
    sig = a.assign(s=a.q_bh < 0.05).pivot(index="syndrome", columns="region", values="s") \
        .loc[syns, REGIONS]
    n = a.groupby("syndrome").n_subjects.first()
    hl = float(np.ceil(np.abs(M.values).max() * 50) / 50)
    im = ax.imshow(M.values, cmap=IMP_CMAP, vmin=-hl, vmax=hl, aspect="auto")
    for i in range(len(syns)):
        for j in range(len(REGIONS)):
            m = M.iloc[i, j]
            star = "*" if sig.iloc[i, j] else ""
            ax.text(j, i, f"{m:+.3f}{star}\n{P.iloc[i, j]:.0%} pos.", ha="center", va="center", fontsize=10,
                    color="white" if abs(m) > 0.6 * hl else "black",
                    fontweight="bold" if star else "normal")
    ax.set_xticks(range(len(REGIONS))); ax.set_xticklabels([pretty(x) for x in REGIONS], fontsize=12)
    ax.set_yticks(range(len(syns)))
    ax.set_yticklabels([f"{short(s)} (n = {n[s]})" for s in syns], fontsize=12)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=10.5)
    cb.set_label("mean importance", fontsize=12)
    ax.set_title("All test subjects of each syndrome (correctly or incorrectly classified); "
                 "* = FDR q < 0.05", fontsize=12)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png", flush=True)


def bar_limit(imp, scans_):
    return float(np.ceil(np.abs(imp.loc[scans_, REGIONS].values).max() * 20) / 20)   # bars never clipped


def main_figure(scans, cases, nn, imp, t):
    picks = [case_scan(cases, imp, *p) for p in _picks()]
    for c, _ in picks:
        assert c.category == "consistent_correct", (c.syndrome, c.fbid)
    bl = bar_limit(imp, [s for _, s in picks])
    nr = len(picks)
    fig = plt.figure(figsize=(12, 2.9 * nr + 3.8))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.9 * nr, 3.2], hspace=0.14)
    top = outer[0].subgridspec(nr, len(WIDTHS), width_ratios=WIDTHS, wspace=0.06, hspace=0.18)
    ax0s = []
    for i, (c, s) in enumerate(picks):
        ax = draw_row(fig, top, i, c, s, scans.loc[s], imp.loc[s].values, nn, bl, False, fs=10.5)
        if i == 0:
            headers(fig, ax, False, bold=True, fs=13)
        ax0s.append(ax[0])
        if i == nr - 1:
            colour_bar(fig, ax[2], ax[3], fs=10.5)
    names_left(fig, ax0s, [short(c.syndrome) for c, _ in picks])
    axh = fig.add_subplot(outer[1])
    heatmap(fig, axh, t)
    fig.canvas.draw()
    for lab, gs in (("a", outer[0]), ("b", outer[1])):
        fig.text(0.04, gs.get_position(fig).y1 + 0.01, lab, fontsize=16, fontweight="bold")
    save(fig, "fig_interp_main")


def misclassified_figure(scans, cases, nn, imp):
    rows = [(g, case_scan(cases, imp, *p)) for g, ps in _wrong() for p in ps]
    for _, (c, _) in rows:
        assert c.category == "consistent_wrong", (c.syndrome, c.fbid)
    bl = bar_limit(imp, [s for _, (_, s) in rows])
    hr = []
    for i, (g, ps) in enumerate(_wrong()):
        hr += [0.9 if i == 0 else 0.45] + [2.9] * len(ps)     # first group heading sits above the column headers
    fig = plt.figure(figsize=(21, sum(hr)))
    gs = fig.add_gridspec(len(hr), len(WIDTHS) + len(NN_WIDTHS), height_ratios=hr,
                          width_ratios=WIDTHS + NN_WIDTHS, wspace=0.06, hspace=0.25)
    r, first, prev, last_ax = 0, True, None, None
    heads, ax0s, names = [], [], []
    for g, (c, s) in rows:
        if g != prev:
            ax_h = fig.add_subplot(gs[r, :]); ax_h.axis("off")
            heads.append((ax_h, g, first))
            r += 1; prev = g
        ax = draw_row(fig, gs, r, c, s, scans.loc[s], imp.loc[s].values, nn, bl, True, fs=10.5)
        agree = int(cases_agree(c))
        ax0s.append(ax[0])
        names.append(f"{short(c.syndrome)}\n\u2192 {short(c.mode_pred_name)} ({agree}/5)")
        if first:
            headers(fig, ax, True, bold=True, fs=13)
            first = False
        last_ax = ax
        r += 1
    colour_bar(fig, last_ax[2], last_ax[3], fs=10.5)
    x0 = names_left(fig, ax0s, names)
    for ax_h, g, top in heads:                         # group headings start at the label column
        b = ax_h.get_position()
        fig.text(x0, b.y1 if top else b.y0, g, fontsize=14, fontweight="bold", va="top" if top else "bottom")
    save(fig, "figS_interp_misclassified")


_AGREE = None


def cases_agree(c):
    """Number of the 5 models that made the most common prediction for this subject."""
    global _AGREE
    if _AGREE is None:
        from analyses.interpretability.panels_common import wrong_agreement
        _AGREE = wrong_agreement()
    return _AGREE.get(str(c.fbid), 0)


def retrieval_figure(scans, cases, nn):
    picks = [cases[(cases.syndrome == sy) & (cases.fbid == f)].iloc[0] for sy, f in _picks()]
    fig, axs = plt.subplots(len(picks), 5, figsize=(11, 2.7 * len(picks)),
                            gridspec_kw=dict(wspace=0.08, hspace=0.3, width_ratios=[1, .12, 1, 1, 1]))
    for i, c in enumerate(picks):
        s = c.scans.split(";")[0]
        d = find(LM_DIRS, s)
        axs[i, 0].imshow(face_render(s, scans.loc[s], d))
        for k in range(3):
            a = axs[i, k + 2]
            ns = c[f"nn{k+1}_scan"]
            dn = find(LM_NN_DIRS, ns)
            if dn is not None:
                a.imshow(face_render(ns, nn.loc[ns], dn))
            a.set_title(f"#{k+1} {short(c[f'nn{k+1}_label'])} ({c[f'nn{k+1}_cos']:.2f})", fontsize=10.5,
                        color="green" if c[f"nn{k+1}_label"] == c.syndrome else "firebrick")
        for a in axs[i]:
            a.axis("off")
    axs[0, 0].set_title("3D facial scan\n", fontsize=13, fontweight="bold")
    fig.canvas.draw()
    b2, b4 = axs[0, 2].get_position(), axs[0, 4].get_position()
    fig.text((b2.x0 + b4.x1) / 2, b2.y1 + 0.022, "Most similar training faces (similarity)",
             ha="center", va="bottom", fontsize=13, fontweight="bold")
    names_left(fig, list(axs[:, 0]), [short(c.syndrome) for c in picks])
    save(fig, "figS_interp_retrieval")


def main():
    scans, cases, nn, imp = load()
    t = syndrome_table(imp, scans, cases, [p[0] for p in _picks()] +
                       ["Mucopolysaccharidosis", "Phelan McDermid Syndrome"])
    main_figure(scans, cases, nn, imp, t)
    misclassified_figure(scans, cases, nn, imp)
    retrieval_figure(scans, cases, nn)


if __name__ == "__main__":
    main()
