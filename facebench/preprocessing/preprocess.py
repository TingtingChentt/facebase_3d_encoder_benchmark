#!/usr/bin/env python3
"""
Preprocessing v3 — one cleaning rule for every dataset.

Why: preprocess.py kept only the largest connected component. 3dMD two-pod
exports (9/11 cleft sites, FB-VWP) are two unstitched half-faces, so the stored
cloud was one half of the face. preprocess_v2.py (controls) skipped the split
for meshes >100k vertices, so datasets were cleaned by different rules.

Cleaning rule (all datasets, no size-dependent branches):
  1. weld coincident vertices (rounded to 1e-4 units) so connectivity reflects
     the surface, not the file's vertex duplication;
  2. connected components over the welded face graph;
  3. seed = largest component by surface area; grow: repeatedly add any
     component with a vertex within R of the kept set, R = GROW_FRAC x the
     whole-mesh bounding-box diagonal (~10 mm on a 250 mm face; scale-free, so
     the half-scale CO meshes get the same relative tolerance);
  4. sample exactly as preprocess.py (surface sample n*3 -> FPS to 4096),
     normalise to the unit sphere. Per-scan RNG seeded from scan_id.
Point clouds (no faces) pass through unchanged, as before.

QC columns per scan: n_comp, n_kept, kept_area_frac, kept_vert_frac,
largest_area_frac. Flag: kept_area_frac < 0.90.

Writes to data/processed/<subdir>/<scan_id>.npy (subdir = ofc | syndrome | controls,
taken from out_path). src_path is relative to FACEBASE_ROOT.
Usage: preprocess.py --list data/splits/scans.csv --worker-id K --n-workers N
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse
import gc
import os
import hashlib
import time
import traceback

import numpy as np
import pandas as pd
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from facebench.preprocessing.mesh_utils import N_POINTS, farthest_point_sample, normalize  # noqa: E402
from facebench.preprocessing.mesh_utils import load_mesh_v1, load_mesh_v2  # noqa: E402


def load_mesh(src):
    """v2 loader first (memory-light); fall back to the v1 loader, which reads
    some FB-TJ0 PLYs that plyfile rejects. Welding below makes the two equivalent."""
    try:
        return load_mesh_v2(src)
    except Exception:                                        # noqa: BLE001
        return load_mesh_v1(src)

from facebench.paths import FACEBASE_ROOT, PROCESSED as OUT_ROOT  # noqa: E402
GROW_FRAC = 0.04
WELD = 1e-4
MAX_QUERY = 40_000          # vertices per component used for the proximity query


def weld(V, F):
    key = np.round(V / WELD).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    F = inv[F]
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    return V[first], F


def components(nv, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(nv, nv))
    return connected_components(g, directed=False)[1]


def clean_v3(V, F):
    """Return (kept face mask, qc dict)."""
    lab = components(len(V), F)
    flab = lab[F[:, 0]]
    tri = V[F]
    farea = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    ncomp = lab.max() + 1
    carea = np.bincount(flab, weights=farea, minlength=ncomp)
    used = np.unique(F)                                   # ignore isolated vertices
    R = GROW_FRAC * np.linalg.norm(V[used].max(0) - V[used].min(0))
    vlab = lab[used]; tree = cKDTree(V[used])
    seed = int(np.argmax(carea))
    kept = {seed}; frontier = [seed]
    while frontier:
        c = frontier.pop()
        pts = V[used[vlab == c]]
        if len(pts) > MAX_QUERY:
            pts = pts[np.random.default_rng(0).choice(len(pts), MAX_QUERY, replace=False)]
        hits = tree.query_ball_point(pts, R, return_sorted=False)
        nb = set(np.unique(vlab[np.concatenate([np.asarray(h, int) for h in hits])]).tolist()) - kept
        kept |= nb; frontier.extend(nb)
    fmask = np.isin(flab, list(kept))
    tot = carea.sum()
    qc = dict(n_comp=int((carea > 0).sum()), n_kept=len(kept),
              kept_area_frac=float(carea[list(kept)].sum() / tot),
              largest_area_frac=float(carea[seed] / tot),
              kept_vert_frac=float(np.isin(vlab, list(kept)).mean()), grow_radius=float(R))
    return fmask, qc


def sample(mesh, n):
    pts, _ = trimesh.sample.sample_surface_even(mesh, n * 3)
    if len(pts) < n:
        pts, _ = trimesh.sample.sample_surface(mesh, n * 3)
    return pts[farthest_point_sample(pts, n)]


def process(scan_id, src, out):
    t0 = time.time()
    r = dict(scan_id=scan_id, src_path=str(src), out_path=str(out), status="ok", error="")
    seed = int(hashlib.md5(scan_id.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    try:
        m = load_mesh(Path(src))
        if isinstance(m, trimesh.PointCloud) or not hasattr(m, "faces") or len(m.faces) == 0:
            V = np.asarray(m.vertices, np.float64)
            pts = V[farthest_point_sample(V, N_POINTS)]
            r.update(kind="pointcloud", n_verts_raw=len(V))
        else:
            V0, F0 = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
            del m
            V, F = weld(V0, F0)
            fmask, qc = clean_v3(V, F)
            r.update(kind="mesh", n_verts_raw=len(V0), n_verts_welded=len(V), **qc)
            mesh = trimesh.Trimesh(V, F[fmask], process=False)
            pts = sample(mesh, N_POINTS)
            del mesh, V0, F0
        pts = normalize(pts).astype(np.float32)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, pts)
        r["elapsed"] = round(time.time() - t0, 2)
    except Exception as ex:                                  # noqa: BLE001
        r.update(status="error", error=repr(ex)[:200], traceback=traceback.format_exc()[-800:])
    return r


def v3_path(old_out):
    p = Path(old_out)
    return OUT_ROOT / p.parent.name / p.name


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--max-scans", type=int)
    a = ap.parse_args()
    L = pd.read_csv(a.list).iloc[a.worker_id::a.n_workers]
    if a.max_scans:
        L = L.head(a.max_scans)
    logdir = OUT_ROOT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, x in enumerate(L.itertuples()):
        out = v3_path(x.out_path)
        if out.exists():
            continue
        src = x.src_path if os.path.isabs(x.src_path) else str(FACEBASE_ROOT / x.src_path)
        rows.append(process(x.scan_id, src, out))
        if rows[-1]["status"] != "ok":
            print("FAIL", x.scan_id, rows[-1]["error"], flush=True)
        if i % 10 == 0:
            gc.collect()
        if i % 100 == 0:
            print(f"{i}/{len(L)}", flush=True)
            pd.DataFrame(rows).to_csv(logdir / f"log_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_{a.worker_id:03d}.csv", index=False)
    pd.DataFrame(rows).to_csv(logdir / f"log_{os.environ.get('SLURM_ARRAY_JOB_ID', 'local')}_{a.worker_id:03d}.csv", index=False)
    print("done", len(rows))
