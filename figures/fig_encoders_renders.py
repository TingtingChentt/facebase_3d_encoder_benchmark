#!/usr/bin/env python3
"""Top-row panels of Fig. 6 (encoder architectures): how each encoder reads the
4,096-point cloud, drawn on one real test scan (the Down syndrome exemplar of
Fig. 5a; local exemplar list, figures/exemplars.py).

  figures/src/enc_{pointnet,pointnet2,dgcnn,geommlp}.png   (read by fig_encoders.py)

Needs the interpretability scan lists, cases and landmarks (results/interp/;
analyses/interpretability/) for the frontal view, and the FaceBase mesh.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from scipy.spatial import cKDTree                                  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from analyses.interpretability.interp_lib import abs_paths, point_regions   # noqa: E402
from analyses.interpretability.panels_common import IN, LM_DIRS, find, geometry  # noqa: E402
from facebench.paths import ROOT                                             # noqa: E402
from figures.exemplars import exemplar                                       # noqa: E402

OUT = ROOT / "figures/src"
SYNDROME = "Down Syndrome"
DPI = 200


def load_scan():
    fbid = exemplar("fig_interp", SYNDROME)
    scans = abs_paths(pd.read_csv(IN / "pilot_scans.csv")).set_index("scan_id")
    cases = pd.read_csv(IN / "cases.csv"); cases["fbid"] = cases.fbid.astype(str)
    c = cases[(cases.syndrome == SYNDROME) & (cases.fbid == fbid)].iloc[0]
    occ = pd.read_csv(IN / "pilot/region_occlusion.csv")
    s = [x for x in c.scans.split(";") if x in set(occ.scan_id)][0]
    row = scans.loc[s]
    lm, Vn, F, R, crop = geometry(s, row, find(LM_DIRS, s))
    cloud = np.load(row["out_path"]).astype(float)
    return cloud, point_regions(cloud, lm), R, crop


def canvas(R, crop, cloud):
    q = cloud @ R.T
    pad = 0.06
    lo, hi = q[:, :2].min(0) - pad, q[:, :2].max(0) + pad
    fig = plt.figure(figsize=(3, 3 * (hi[1] - lo[1]) / (hi[0] - lo[0])), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal"); ax.axis("off")
    return fig, ax, q


def scatter(ax, q, idx, col, s=3.0, z=1, **kw):
    o = idx[np.argsort(q[idx, 2])]
    ax.scatter(q[o, 0], q[o, 1], c=col if np.ndim(col) == 1 or len(col) != len(q) else col[o], s=s, lw=0,
               zorder=z, **kw)


def encoder_panels(cloud, reg, R, crop):
    allp = np.arange(len(cloud))
    navy = np.array([.08, .24, .39, 1]); light = np.array([.62, .70, .80, 1])
    rng = np.random.default_rng(0)

    fig, ax, q = canvas(R, crop, cloud)                       # PointNet: every point alone
    scatter(ax, q, allp, navy, s=2.5)
    fig.savefig(OUT / "enc_pointnet.png", transparent=True); plt.close(fig)

    fig, ax, q = canvas(R, crop, cloud)                       # PointNet++: local balls at sampled centres
    scatter(ax, q, allp, light, s=2.0)
    ctrs = [allp[0]]
    d = np.linalg.norm(cloud - cloud[ctrs[0]], axis=1)
    for _ in range(11):                                       # farthest-point sampling, 12 centres
        ctrs.append(int(d.argmax())); d = np.minimum(d, np.linalg.norm(cloud - cloud[ctrs[-1]], axis=1))
    tree = cKDTree(cloud)
    for c in ctrs:
        nb = tree.query_ball_point(cloud[c], 0.2)
        scatter(ax, q, np.array(nb), navy, s=2.5, z=2)
        for r_, a_ in ((0.1, .9), (0.2, .6)):
            ax.add_patch(plt.Circle(q[c, :2], r_, fill=False, ec="#B03A2E", lw=1.0, alpha=a_, zorder=3))
        ax.scatter(*q[c, :2], c="#B03A2E", s=14, zorder=4)
    fig.savefig(OUT / "enc_pointnet2.png", transparent=True); plt.close(fig)

    fig, ax, q = canvas(R, crop, cloud)                       # DGCNN: k-NN graph edges
    scatter(ax, q, allp, light, s=2.0)
    sub = rng.choice(allp, 260, replace=False)
    _, nb = tree.query(cloud[sub], k=9)
    for i, n in zip(sub, nb):
        for j in n[1:]:
            ax.plot(q[[i, j], 0], q[[i, j], 1], c="#143D64", lw=.5, alpha=.8, zorder=2)
    fig.savefig(OUT / "enc_dgcnn.png", transparent=True); plt.close(fig)

    fig, ax, q = canvas(R, crop, cloud)                       # GeomMLP: global summary only
    scatter(ax, q, allp, light, s=2.0)
    c2 = q[:, :2].mean(0)
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    for a, b in (((lo[0], c2[1]), (hi[0], c2[1])), ((c2[0], lo[1]), (c2[0], hi[1]))):
        ax.annotate("", b, a, arrowprops=dict(arrowstyle="<->", color="#B03A2E", lw=1.6), zorder=4)
    ax.scatter(*c2, c="#B03A2E", s=18, zorder=5)
    ax.add_patch(plt.Rectangle(lo, *(hi - lo), fill=False, ec="#143D64", lw=1.2, ls="--", zorder=3))
    ax.add_patch(plt.Circle(c2, 0.5, fill=False, ec="#143D64", lw=1.0, zorder=3))
    fig.savefig(OUT / "enc_geommlp.png", transparent=True); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    encoder_panels(*load_scan())
    print("wrote", ", ".join(f"enc_{k}.png" for k in ("pointnet", "pointnet2", "dgcnn", "geommlp")), "->", OUT)
