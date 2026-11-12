"""
Automatic sparse 3D landmarks for FB-TJ0 scans, in the coordinate frame of the
stored v3 point cloud. 

Runs in a separate environment with MediaPipe 0.10.14 (see docs/REPRODUCE.md), not the main one.

Per scan:
  1. load the raw 3dMD PLY (vertex colour);
  2. render a frontal orthographic image by dense surface sampling + z-buffer
     (colour x Lambert shading); camera on +z, y up (3dMD frame; the model's
     training augmentation also rotates about y, so y is up in the clouds);
  3. MediaPipe FaceMesh (468 points). If no face is found, retry with the
     camera on -z and with in-plane rotations;
  4. back-project each 2D landmark to the 3D surface point that owns that
     pixel (nearest filled pixel);
  5. recover the v3 normalisation (cloud = (mesh_pt - c) / s) by a
     scale+translation ICP of the stored cloud onto the mesh surface, and
     express the landmarks in cloud coordinates.

The colour render is used ONLY for detection on the cluster; figures for the
paper use geometry-only renders (no texture), see make_figures.py.

Usage: landmark_scans.py --list scans.csv --out-dir DIR [--qc-dir DIR]
       scans.csv needs scan_id, src_path, out_path
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import time

import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy import ndimage
import cv2
from PIL import Image, ImageDraw
import mediapipe as mp
from plyfile import PlyData

IMG = 768
N_SURF = 1_500_000


def render(V, F, vc, view):
    """Orthographic z-buffer render. view = (flip_z, rot90 k).
    Returns rgb (IMG,IMG,3) uint8, xyz (IMG,IMG,3) of the owning surface point
    in the ORIGINAL mesh frame (nan where empty), and the 2D->pixel mapping."""
    flip, k = view
    R = np.eye(3)
    if flip:                                   # camera on -z: rotate 180 about y
        R = np.diag([-1.0, 1.0, -1.0])
    th = k * np.pi / 2
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    R = Rz @ R
    m = trimesh.Trimesh(V, F, process=False)
    pts, fi = trimesh.sample.sample_surface(m, N_SURF, seed=0)
    col = vc[F[fi]].mean(1)[:, :3] / 255.0
    nrm = m.face_normals[fi] @ R.T
    q = pts @ R.T
    front = nrm[:, 2] > 0                      # back-face cull
    q, pts, col, nrm = q[front], pts[front], col[front], nrm[front]
    light = np.array([0.3, 0.4, 1.0]); light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
    rgb = np.clip(col * shade[:, None], 0, 1)
    lo, hi = q[:, :2].min(0), q[:, :2].max(0)
    span = (hi - lo).max() * 1.05
    ctr = (lo + hi) / 2
    px = ((q[:, 0] - ctr[0]) / span + 0.5) * (IMG - 1)
    py = (0.5 - (q[:, 1] - ctr[1]) / span) * (IMG - 1)
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    pix = iy * IMG + ix
    order = np.lexsort((-q[:, 2], pix))        # per pixel, nearest (max z) first
    pix_s = pix[order]
    first = order[np.r_[True, pix_s[1:] != pix_s[:-1]]]
    img = np.ones((IMG * IMG, 3)); geo = np.ones(IMG * IMG)
    xyz = np.full((IMG * IMG, 3), np.nan)
    img[pix[first]] = rgb[first]; geo[pix[first]] = shade[first]
    xyz[pix[first]] = pts[first]
    img = img.reshape(IMG, IMG, 3); geo = geo.reshape(IMG, IMG)
    xyz = xyz.reshape(IMG, IMG, 3)
    # fill sampling pin-holes (empty pixels within 3 px of the surface)
    empty = np.isnan(xyz[..., 0])
    dist, (ri, ci) = ndimage.distance_transform_edt(empty, return_indices=True)
    hole = empty & (dist <= 3)
    img[hole] = img[ri[hole], ci[hole]]; geo[hole] = geo[ri[hole], ci[hole]]
    img = (img * 255).astype(np.uint8)
    # variant 2: contrast-normalised colour (CLAHE on L) -- dark or low-contrast skin
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
    img_eq = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    # variant 3: geometry-only grey shading (independent of skin colour)
    g = (np.clip(geo, 0, 1) * 255).astype(np.uint8)
    img_geo = np.repeat(g[..., None], 3, 2)
    return {"rgb": img, "clahe": img_eq, "geo": img_geo}, xyz


def detect(fm, rgb):
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * IMG, p.y * IMG] for p in lm])


def backproject(lm2d, xyz):
    filled = np.argwhere(~np.isnan(xyz[..., 0]))           # (row, col)
    tree = cKDTree(filled)
    d, j = tree.query(np.c_[lm2d[:, 1], lm2d[:, 0]])
    rc = filled[j]
    return xyz[rc[:, 0], rc[:, 1]], d


def fit_norm(cloud, V, F):
    """cloud ~= (surface - c)/s ; ICP over scale+translation only (no rotation:
    preprocessing never rotates)."""
    surf, _ = trimesh.sample.sample_surface(trimesh.Trimesh(V, F, process=False), 400_000, seed=1)
    tree = cKDTree(surf)
    c = surf.mean(0)
    s = np.linalg.norm(surf - c, axis=1).max()
    for _ in range(60):
        X = cloud * s + c
        _, j = tree.query(X)
        Q = surf[j]
        P0 = cloud - cloud.mean(0); Q0 = Q - Q.mean(0)
        s_new = (P0 * Q0).sum() / (P0 * P0).sum()
        c_new = Q.mean(0) - s_new * cloud.mean(0)
        conv = abs(s_new - s) / s < 1e-7 and np.linalg.norm(c_new - c) < 1e-6 * s
        s, c = s_new, c_new
        if conv:
            break
    d, _ = tree.query(cloud * s + c)
    return s, c, float(np.median(d) / s), float(np.percentile(d, 95) / s)


def load_ply(path):
    """3dMD binary PLY (plyfile; trimesh rejects these files)."""
    d = PlyData.read(str(path))
    v = d["vertex"]
    V = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
    names = v.data.dtype.names
    vc = np.stack([v[c] for c in ("red", "green", "blue")], 1).astype(float) \
        if "red" in names else np.full((len(V), 3), 200.0)
    fd = d["face"].data
    fkey = "vertex_index" if "vertex_index" in fd.dtype.names else "vertex_indices"
    F = np.vstack(fd[fkey]).astype(int)
    return V, F, vc


VIEWS = [(False, 0), (True, 0), (False, 1), (False, 3), (False, 2), (True, 1), (True, 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--qc-dir")
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    a = ap.parse_args()
    from analyses.interpretability.interp_lib import abs_paths
    df = abs_paths(pd.read_csv(a.list)).iloc[a.worker_id::a.n_workers]
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    qc = Path(a.qc_dir) if a.qc_dir else None
    if qc:
        qc.mkdir(parents=True, exist_ok=True)
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=False,
                                         min_detection_confidence=0.3)
    rows = []
    for r in df.itertuples():
        t0 = time.time()
        rec = dict(scan_id=r.scan_id, status="ok", view="", fit_med=np.nan,
                   fit_p95=np.nan, bp_px_max=np.nan)
        try:
            V, F, vc = load_ply(r.src_path)
            lm2d = None
            for v in VIEWS:
                imgs, xyz = render(V, F, vc, v)
                for variant in ("rgb", "clahe", "geo"):
                    lm2d = detect(fm, imgs[variant])
                    if lm2d is not None:
                        break
                if lm2d is not None:
                    break
            rgb = imgs["rgb"]
            if lm2d is None:
                rec["status"] = "no_face"
                rows.append(rec); print(r.scan_id, "no_face", flush=True); continue
            lm3d, dpx = backproject(lm2d, xyz)
            cloud = np.load(r.out_path).astype(float)
            s, c, fmed, fp95 = fit_norm(cloud, V, F)
            lm_norm = (lm3d - c) / s
            np.savez_compressed(out / f"{r.scan_id}.npz", lm_raw=lm3d, lm_norm=lm_norm,
                                lm2d=lm2d, s=s, c=c, view=np.array(v), variant=variant)
            rec.update(view=f"{int(v[0])}{v[1]}", variant=variant, fit_med=fmed, fit_p95=fp95,
                       bp_px_max=float(dpx.max()))
            if qc:
                im = Image.fromarray(rgb); dr = ImageDraw.Draw(im)
                for x, y in lm2d:
                    dr.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=(0, 255, 0))
                im.resize((384, 384)).save(qc / f"{r.scan_id}.jpg", quality=85)
        except Exception as ex:                                  # noqa: BLE001
            rec.update(status="error:" + repr(ex)[:150])
        rec["sec"] = round(time.time() - t0, 1)
        rows.append(rec)
        print(rec, flush=True)
    pd.DataFrame(rows).to_csv(out / f"_status_w{a.worker_id}.csv", index=False)


if __name__ == "__main__":
    sys.exit(main())
