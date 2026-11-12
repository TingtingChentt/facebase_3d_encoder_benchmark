"""
Geometry-only mesh rendering for the interpretability figures (no texture, so
no skin colour is shown). Main environment.
"""
import numpy as np
import trimesh
from plyfile import PlyData
from scipy import ndimage
from scipy.spatial import cKDTree


def load_ply(path):
    d = PlyData.read(str(path))
    v = d["vertex"]
    V = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)
    fd = d["face"].data
    key = "vertex_index" if "vertex_index" in fd.dtype.names else "vertex_indices"
    return V, np.vstack(fd[key]).astype(int)


def view_matrix(flip):
    return np.diag([-1.0, 1.0, -1.0]) if flip else np.eye(3)


def render_scalar(V, F, R, crop, scalar=None, img=480, n_surf=900_000,
                  cmap=None, vmin=0.5, vmax=1.0):
    """Orthographic render of mesh (already in cloud coords) under rotation R.
    crop = (xmin, xmax, ymin, ymax) in rotated coords. scalar: per-vertex value
    in [0,1] drawn with cmap over grey shading, transparent below vmin."""
    m = trimesh.Trimesh(V, F, process=False)
    pts, fi = trimesh.sample.sample_surface(m, n_surf, seed=0)
    q = pts @ R.T
    nrm = m.face_normals[fi] @ R.T
    keep = (nrm[:, 2] > 0) & (q[:, 0] > crop[0]) & (q[:, 0] < crop[1]) & \
           (q[:, 1] > crop[2]) & (q[:, 1] < crop[3])
    q, nrm, fi = q[keep], nrm[keep], fi[keep]
    light = np.array([0.25, 0.35, 1.0]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
    rgb = np.repeat(shade[:, None], 3, 1) * np.array([0.86, 0.86, 0.88])
    if scalar is not None:
        val = scalar[F[fi]].mean(1)
        a = np.clip((val - vmin) / (vmax - vmin), 0, 1) ** 1.2
        col = cmap(np.clip((val - vmin) / (vmax - vmin), 0, 1))[:, :3]
        rgb = rgb * (1 - 0.95 * a[:, None]) + col * (0.55 + 0.45 * shade[:, None]) * 0.95 * a[:, None]
    return _raster(q, rgb, crop, img)


def render_signed(V, F, R, crop, values, lim, cmap, img=480, n_surf=900_000):
    """Like render_scalar, for a signed per-vertex value: colour = cmap at
    0.5 + 0.5*value/lim (diverging), opacity grows with |value|/lim."""
    m = trimesh.Trimesh(V, F, process=False)
    pts, fi = trimesh.sample.sample_surface(m, n_surf, seed=0)
    q = pts @ R.T
    nrm = m.face_normals[fi] @ R.T
    keep = (nrm[:, 2] > 0) & (q[:, 0] > crop[0]) & (q[:, 0] < crop[1]) & \
           (q[:, 1] > crop[2]) & (q[:, 1] < crop[3])
    q, nrm, fi = q[keep], nrm[keep], fi[keep]
    light = np.array([0.25, 0.35, 1.0]); light /= np.linalg.norm(light)
    shade = 0.35 + 0.65 * np.clip(nrm @ light, 0, 1)
    rgb = np.repeat(shade[:, None], 3, 1) * np.array([0.86, 0.86, 0.88])
    val = values[F[fi]].mean(1)
    t = np.clip(val / lim, -1, 1)
    a = 0.9 * np.abs(t) ** 0.7
    col = cmap(0.5 + 0.5 * t)[:, :3]
    rgb = rgb * (1 - a[:, None]) + col * (0.55 + 0.45 * shade[:, None]) * a[:, None]
    return _raster(q, rgb, crop, img)


def project(pts, R, crop, img=480):
    """Pixel (x, y) of 3D points (cloud coords) in a render_scalar/render_signed image."""
    q = pts @ R.T
    w = crop[1] - crop[0]; h = crop[3] - crop[2]
    W = img; H = int(round(img * h / w))
    return np.c_[(q[:, 0] - crop[0]) / w * (W - 1), (crop[3] - q[:, 1]) / h * (H - 1)]


def _raster(q, rgb, crop, img):
    w = crop[1] - crop[0]; h = crop[3] - crop[2]
    W = img; H = int(round(img * h / w))
    ix = np.clip(((q[:, 0] - crop[0]) / w * (W - 1)).round().astype(int), 0, W - 1)
    iy = np.clip(((crop[3] - q[:, 1]) / h * (H - 1)).round().astype(int), 0, H - 1)
    pix = iy * W + ix
    order = np.lexsort((-q[:, 2], pix))
    ps = pix[order]
    first = order[np.r_[True, ps[1:] != ps[:-1]]]
    out = np.ones((H * W, 3)); filled = np.zeros(H * W, bool)
    out[pix[first]] = rgb[first]; filled[pix[first]] = True
    out = out.reshape(H, W, 3); filled = filled.reshape(H, W)
    dist, (ri, ci) = ndimage.distance_transform_edt(~filled, return_indices=True)
    hole = ~filled & (dist <= 3)
    out[hole] = out[ri[hole], ci[hole]]
    return np.clip(out, 0, 1)


def to_vertices(cloud, values, Vn, k=8):
    """Interpolate per-point values onto mesh vertices (inverse distance, k nn)."""
    d, j = cKDTree(cloud).query(Vn, k=k)
    w = 1.0 / (d + 1e-4)
    return (values[j] * w).sum(1) / w.sum(1)


def face_crop(lm, R, pad_x=0.30, top=0.55, bottom=0.12):
    q = lm @ R.T
    x0, x1 = q[:, 0].min(), q[:, 0].max(); y0, y1 = q[:, 1].min(), q[:, 1].max()
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    half = max(0.5 * w * (1 + 2 * pad_x), 0.8 * h)    # turned faces: landmark bbox is narrow
    return (cx - half, cx + half, y0 - bottom * h, y1 + top * h)
