#!/usr/bin/env python3
"""
Supplementary Fig. S1a — how template fitting attenuates apparent facial deviation.

One FB-TJ0 scan with 22q11.2 deletion syndrome is compared with a normative
control scan (chosen as the control scan nearest the PCA centroid), before and
after a regularized non-rigid fit toward that template; colour = per-vertex
deviation from the template (mm). Both subjects come from the local exemplar
list (figures/exemplars.py, keys reg_subject / reg_template); the meshes are
read from the FaceBase download.

Writes figures/src/fig_regp_{raw,template,registered}.png and fig_regp_meta.json,
which figures/figS_overview.py places in panel (a). Needs open3d.

Panels
  A  Raw unregistered scan (real), coloured by deviation from the template.
  B  Normative template (real control), aligned into the subject's frontal frame.
  C  Subject after a regularized non-rigid morph TOWARD the template -> atypical
     anatomy pulled toward the mean; the same deviation field collapses.

All geometry is real. The morph is an illustrative regularized template fit
(what dense/non-rigid registration pipelines do); labelled as such on the poster.
"""
import numpy as np
import trimesh
import open3d as o3d
from matplotlib.collections import PolyCollection
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from facebench.paths import ROOT, SPLITS, facebase  # noqa: E402
from facebench.preprocessing.mesh_utils import load_face_mesh  # noqa: E402
from figures.exemplars import exemplar  # noqa: E402

_SRC = pd.read_csv(SPLITS / "scans.csv").set_index("scan_id").src_path
SUBJ = facebase(_SRC[exemplar("figS_overview", "reg_subject")])
TMPL = facebase(_SRC[exemplar("figS_overview", "reg_template")])
FIGDIR = ROOT / "figures/src"
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT = FIGDIR / "fig_regp_attenuation.png"

rng = np.random.RandomState(0)
try:
    o3d.utility.random.seed(0)                       # deterministic RANSAC across runs
except Exception:
    pass


# ------------------------------------------------------------------ geometry
def load_face(path, ycrop):
    m = load_face_mesh(path, process=False)
    m = max(m.split(only_watertight=False), key=lambda c: len(c.vertices)) if False else m
    V = np.asarray(m.vertices, float)
    keep = V[:, 1] > ycrop
    idx = np.where(keep)[0]
    remap = -np.ones(len(V), int); remap[idx] = np.arange(len(idx))
    fmask = keep[m.faces].all(1)
    return trimesh.Trimesh(V[idx], remap[m.faces[fmask]], process=False)


def simplify(m, target_faces):
    ms = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(m.vertices, float)),
        o3d.utility.Vector3iVector(np.asarray(m.faces)))
    ms = ms.simplify_quadric_decimation(int(target_faces))
    ms.remove_degenerate_triangles(); ms.remove_unreferenced_vertices()
    return trimesh.Trimesh(np.asarray(ms.vertices), np.asarray(ms.triangles), process=False)


def crop_box(m, lo, hi):
    """Keep vertices inside [lo,hi] box, drop faces touching removed verts."""
    V = np.asarray(m.vertices); keep = np.all((V > lo) & (V < hi), axis=1)
    idx = np.where(keep)[0]; remap = -np.ones(len(V), int); remap[idx] = np.arange(len(idx))
    fm = keep[m.faces].all(1)
    return trimesh.Trimesh(V[idx], remap[m.faces[fm]], process=False)


def largest_component(m):
    comps = m.split(only_watertight=False)
    return max(comps, key=lambda c: len(c.vertices)) if len(comps) else m


print("[load] subject + template meshes")
Sfull = largest_component(simplify(load_face(SUBJ, -95.0), 60000))   # +Z front; registration target
# tight facial crop: drop hair (top/back), ears (sides), neck (bottom)
cx = np.median(Sfull.vertices[:, 0]); zmax = Sfull.vertices[:, 2].max()
FLO = np.array([cx - 65, -78, zmax - 64]); FHI = np.array([cx + 65, 69, zmax + 60])
S = largest_component(crop_box(Sfull, FLO, FHI))      # subject face for morph + render
print(f"[subject] face crop: {len(S.vertices)} verts (from {len(Sfull.vertices)})")
Tfull = load_face_mesh(TMPL, process=False)            # tilted control; align via registration


# --- global + rigid registration:  template -> subject frame ---------------
def to_pcd(V, vox):
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(V)
    p = p.voxel_down_sample(vox)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=vox*2, max_nn=30))
    return p

VT = np.asarray(Tfull.vertices, float)
VT = VT - VT.mean(0) + Sfull.vertices.mean(0)        # coarse translate to subject
src = to_pcd(VT, 4.0)
tgt = to_pcd(Sfull.vertices, 4.0)
fsrc = o3d.pipelines.registration.compute_fpfh_feature(src, o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=100))
ftgt = o3d.pipelines.registration.compute_fpfh_feature(tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=100))
ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    src, tgt, fsrc, ftgt, True, 12.0,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(12.0)],
    o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 0.999))
icp = o3d.pipelines.registration.registration_icp(
    to_pcd(VT, 2.0), tgt, 10.0, ransac.transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
print(f"[reg] ransac fit {ransac.fitness:.2f}  icp fit {icp.fitness:.2f} rmse {icp.inlier_rmse:.2f}")
Th = (icp.transformation @ np.c_[VT, np.ones(len(VT))].T).T[:, :3]
Tm = trimesh.Trimesh(Th, Tfull.faces, process=False)
# crop template to the subject's facial box (shared frame), generous in depth
bmin, bmax = S.vertices.min(0), S.vertices.max(0)
Tm = largest_component(crop_box(Tm, np.array([bmin[0]-6, bmin[1]-6, bmin[2]-45]),
                                    np.array([bmax[0]+6, bmax[1]+6, bmax[2]+45])))
Tm = simplify(Tm, 22000)
print(f"[template] face crop: {len(Tm.vertices)} verts")


# --- deviation field + regularized non-rigid morph -------------------------
from scipy.spatial import cKDTree
kd = cKDTree(Tm.vertices)
dist, nn = kd.query(S.vertices, k=1)                 # nearest template point
target = Tm.vertices[nn]
disp = target - S.vertices                            # raw pull toward template

# regularize: Laplacian-smooth the displacement over the subject mesh
A = S.vertex_adjacency_graph
nbrs = [list(A[i]) for i in range(len(S.vertices))]
d = disp.copy()
for _ in range(25):
    d = 0.5 * d + 0.5 * np.array([d[n].mean(0) if n else d[i]
                                  for i, n in enumerate(nbrs)])
ALPHA = 0.72
Smorph = S.vertices + ALPHA * d
dev_raw = np.linalg.norm(disp, axis=1)               # diagnostic deviation
dev_post = np.linalg.norm(target - Smorph, axis=1)   # residual after morph
print(f"[morph] mean deviation  raw {dev_raw.mean():.1f}  ->  after {dev_post.mean():.1f} mm")
# light smoothing of the scalar field so the heatmap reads as anatomy, not NN noise
for _ in range(4):
    dev_raw = 0.5*dev_raw + 0.5*np.array([dev_raw[n].mean() if n else dev_raw[i] for i, n in enumerate(nbrs)])
    dev_post = 0.5*dev_post + 0.5*np.array([dev_post[n].mean() if n else dev_post[i] for i, n in enumerate(nbrs)])


# ------------------------------------------------------------------ rendering
# Emit the three faces as SEPARATE transparent images at a shared scale, so the
# poster can place them alongside editable pptx titles / arrows / colorbar.
import json
LIGHT = np.array([-0.3, 0.35, 1.0]); LIGHT /= np.linalg.norm(LIGHT)

def face_shade(V, Faces):
    tri = V[Faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    sh = 0.35 + 0.65 * np.clip(n @ LIGHT, 0, 1)
    return sh, tri[:, :, 2].mean(1)

vmax = float(np.percentile(dev_raw, 95))
dn = Normalize(0, vmax)
# Sequential blue from Figure 1's palette: near-white at zero deviation
# through to the darkest navy used for the encoder series.
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "fb_blues", ["#F7FAFC", "#D6E3EE", "#9EC1DC", "#5E8FBA", "#2C5F8A", "#123F63"])

# shared data extent (so all three faces are to-scale + identical pixel size)
allV = np.vstack([S.vertices[:, :2], Tm.vertices[:, :2], Smorph[:, :2]])
pad = 6.0
xlo, ylo = allV.min(0) - pad; xhi, yhi = allV.max(0) + pad
AR = (xhi - xlo) / (yhi - ylo)

def render_panel(V, Faces, out, base=None, vscalar=None):
    sh, depth = face_shade(V, Faces)
    order = np.argsort(depth)
    tris2d = V[Faces][:, :, :2][order]
    if vscalar is not None:
        fc = cmap(dn(vscalar[Faces].mean(1)))[:, :3][order]
    else:
        fc = np.tile(np.array(base), (len(Faces), 1))[order]
    fc = np.clip(fc * sh[order, None], 0, 1)
    fig = plt.figure(figsize=(4.2 * AR, 4.2)); fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.axis("off")
    ax.add_collection(PolyCollection(tris2d, facecolors=fc, edgecolors="none", antialiaseds=True))
    fig.savefig(out, dpi=220, transparent=True)
    plt.close(fig)
    print("WROTE", out.split("/")[-1])

render_panel(S.vertices, S.faces, f"{FIGDIR}/fig_regp_raw.png",       vscalar=dev_raw)
render_panel(Tm.vertices, Tm.faces, f"{FIGDIR}/fig_regp_template.png", base=(0.78, 0.82, 0.87))
render_panel(Smorph,      S.faces,  f"{FIGDIR}/fig_regp_registered.png", vscalar=dev_post)

# colorbar swatches (magma) + metadata for the editable pptx legend
N = 24
colors = ["%02X%02X%02X" % tuple(int(round(255*c)) for c in cmap(i/(N-1))[:3]) for i in range(N)]
meta = {"ar": round(AR, 4), "vmax": vmax, "vmax_round": int(round(vmax)), "colors": colors,
        "dev_raw_mean": round(float(dev_raw.mean()), 1), "dev_post_mean": round(float(dev_post.mean()), 1)}
with open(f"{FIGDIR}/fig_regp_meta.json", "w") as fh:
    json.dump(meta, fh, indent=2)
print(f"WROTE fig_regp_meta.json  (ar={meta['ar']}, vmax={meta['vmax_round']}mm)")

# also keep the combined flat PNG as a fallback reference
fig, ax = plt.subplots(1, 3, figsize=(13.2, 5.2)); fig.patch.set_alpha(0)
for a, (V, F, kw) in zip(ax, [(S.vertices, S.faces, dict(vscalar=dev_raw)),
                              (Tm.vertices, Tm.faces, dict(base=(0.80, 0.82, 0.85))),
                              (Smorph, S.faces, dict(vscalar=dev_post))]):
    sh, depth = face_shade(V, F); order = np.argsort(depth)
    if "vscalar" in kw:
        fc = cmap(dn(kw["vscalar"][F].mean(1)))[:, :3][order]
    else:
        fc = np.tile(np.array(kw["base"]), (len(F), 1))[order]
    fc = np.clip(fc * sh[order, None], 0, 1)
    a.add_collection(PolyCollection(V[F][:, :, :2][order], facecolors=fc, edgecolors="none"))
    a.set_xlim(xlo, xhi); a.set_ylim(ylo, yhi); a.set_aspect("equal"); a.axis("off")
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.02)
plt.savefig(OUT, dpi=150, transparent=True); plt.close(fig)
print("WROTE", OUT.name, "(fallback combined)")
