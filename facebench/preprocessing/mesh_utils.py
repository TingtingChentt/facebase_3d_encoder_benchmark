"""
Mesh I/O and point-sampling helpers used by preprocess.py, copied verbatim from the
earlier pipeline revisions they come from:

  load_mesh_v1           v1 loader (trimesh process=True); reads some FB-TJ0 PLYs plyfile rejects
  load_mesh_v2           v2 loader (process=False, memory-light; both PLY face-field spellings)
  farthest_point_sample  greedy FPS on the global numpy RNG (preprocess.py seeds it per scan)
  normalize              centre on the centroid, scale to the unit sphere
"""
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

N_POINTS = 4096


def farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Farthest point sampling. Returns indices of n selected points."""
    N = len(points)
    if N <= n:
        idx = np.concatenate([np.arange(N), np.random.choice(N, n - N)])
        return idx
    selected = np.zeros(n, dtype=np.int64)
    dists = np.full(N, np.inf)
    selected[0] = np.random.randint(N)
    for i in range(1, n):
        last = points[selected[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return selected


def normalize(pts: np.ndarray) -> np.ndarray:
    """Center to centroid, scale to unit sphere."""
    pts = pts - pts.mean(axis=0)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts


def load_mesh_v1(src_path: Path) -> trimesh.Trimesh:
    """Load OBJ or PLY, returning a Trimesh. Uses plyfile for binary PLY."""
    suffix = src_path.suffix.lower()
    if suffix == ".ply":
        try:
            data = PlyData.read(str(src_path))
            verts = data["vertex"]
            xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
            if "face" in data:
                faces_raw = data["face"]["vertex_index"]  # 3dMD PLY uses singular name
                faces = np.array([f.tolist() for f in faces_raw], dtype=np.int64)
                return trimesh.Trimesh(vertices=xyz, faces=faces, process=True)
            else:
                return trimesh.PointCloud(xyz)
        except Exception as e:
            # Fallback to trimesh native loader — skip_materials avoids loading .jpg textures
            return trimesh.load(str(src_path), process=True, force="mesh",
                                skip_materials=True)
    else:
        mesh = trimesh.load(str(src_path), process=True, force="mesh",
                            skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
        return mesh


def load_mesh_v2(src_path: Path) -> trimesh.Trimesh:
    """Load OBJ or PLY, returning a Trimesh. Uses plyfile for binary PLY."""
    suffix = src_path.suffix.lower()
    if suffix == ".ply":
        try:
            data = PlyData.read(str(src_path))
            verts = data["vertex"]
            xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
            if "face" in data:
                # Try both field name variants (3dMD uses 'vertex_index'; others use 'vertex_indices')
                face_data = None
                for field in ("vertex_index", "vertex_indices"):
                    try:
                        face_data = data["face"][field]
                        break
                    except (ValueError, KeyError):
                        continue
                if face_data is not None:
                    faces = np.array([f.tolist() for f in face_data], dtype=np.int64)
                    # process=False: skip expensive merge_vertices/fix_winding — saves 10-50x peak memory
                    return trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
                else:
                    return trimesh.PointCloud(xyz)
            else:
                return trimesh.PointCloud(xyz)
        except Exception:
            # Fallback to trimesh native loader — skip_materials avoids loading .jpg textures
            return trimesh.load(str(src_path), process=False, force="mesh",
                                skip_materials=True)
    else:
        mesh = trimesh.load(str(src_path), process=False, force="mesh",
                            skip_materials=True)
        if isinstance(mesh, trimesh.Scene):
            geoms = list(mesh.geometry.values())
            mesh = trimesh.util.concatenate(geoms) if geoms else trimesh.Trimesh()
        return mesh


def load_face_mesh(src_path, process: bool = True) -> trimesh.Trimesh:
    """Surface mesh of one FaceBase scan, for DISPLAY (figures that render faces).

    Reads the FaceBase file directly (PLY via plyfile, both face-index field
    names; OBJ via trimesh). Vertices not referenced by any face are dropped, so
    process=False gives the same vertex/face arrays as trimesh's OBJ loader on an
    OBJ export of the same file (checked on five FB-TJ0 scans), and process=True
    merges duplicate vertices as trimesh does. Never used for model input; the
    point clouds come from preprocess.py."""
    m = load_mesh_v2(Path(src_path))
    m = trimesh.Trimesh(vertices=np.asarray(m.vertices), faces=np.asarray(m.faces),
                        process=process)
    if not process:
        m.remove_unreferenced_vertices()
    return m
