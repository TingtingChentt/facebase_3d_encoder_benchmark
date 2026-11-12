"""
Shared helpers for the landmark-based interpretability analysis
No MediaPipe import (runs in the main environment).

Regions are defined from MediaPipe FaceMesh indices: a handful of ANCHOR
landmarks per region; every one of the 468 landmarks takes the region of its
nearest anchor (in that scan's own 3D landmark set), and every cloud point takes
the region of its nearest landmark, or "outside" (ears, neck, scalp, clothing)
if it is farther than TAU_MULT x the median landmark spacing from all of them.
Left and right are pooled.
"""
import numpy as np
from scipy.spatial import cKDTree

ANCHORS = {
    "forehead":    [10, 151, 9, 108, 337, 67, 297, 109, 338, 103, 332, 54, 284,
                    69, 299, 104, 333, 21, 251],
    "periorbital": [33, 133, 159, 145, 246, 7, 263, 362, 386, 374, 466, 249,
                    70, 63, 105, 66, 107, 300, 293, 334, 296, 336,
                    230, 229, 450, 449],
    "nose":        [168, 6, 197, 195, 5, 4, 1, 19, 94, 48, 64, 98, 278, 294, 327,
                    129, 358, 115, 344],
    "cheek":       [50, 280, 123, 352, 116, 345, 117, 346, 205, 425, 187, 411,
                    147, 376, 213, 433, 207, 427, 234, 454, 93, 323, 132, 361],
    "philtrum":    [164, 167, 393, 165, 391, 92, 322],
    "lips":        [0, 13, 14, 17, 37, 267, 84, 314, 61, 291, 78, 308, 40, 270,
                    91, 321],
    "chin_jaw":    [152, 175, 199, 200, 18, 148, 377, 176, 400, 136, 365, 172,
                    397, 150, 379, 58, 288],
}
REGIONS = list(ANCHORS)                      # face regions, display order
ALL_REGIONS = REGIONS + ["outside"]
TAU_MULT = 2.5


def landmark_regions(lm):
    """(468,) region index for each landmark, by nearest anchor."""
    idx = np.concatenate([ANCHORS[r] for r in REGIONS])
    lab = np.concatenate([[i] * len(ANCHORS[r]) for i, r in enumerate(REGIONS)])
    _, j = cKDTree(lm[idx]).query(lm)
    return lab[j]


def point_regions(cloud, lm):
    """(N,) region index per cloud point; len(REGIONS) = outside."""
    lreg = landmark_regions(lm)
    t = cKDTree(lm)
    nn, _ = t.query(lm, k=2)
    tau = TAU_MULT * np.median(nn[:, 1])
    d, j = t.query(cloud)
    reg = lreg[j]
    reg[d > tau] = len(REGIONS)
    return reg


def abs_paths(df):
    """Scan lists in results/interp/ store src_path relative to the FaceBase download
    (FACEBASE_ROOT) and out_path relative to the repository (FACEBENCH_DATA_ROOT,
    default = repository root). Return a copy with both made absolute."""
    import os
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    fb = Path(os.environ.get("FACEBASE_ROOT", "FACEBASE_ROOT_not_set"))
    data = Path(os.environ.get("FACEBENCH_DATA_ROOT", root))
    df = df.copy()
    if "src_path" in df:
        df["src_path"] = df["src_path"].astype(str).map(
            lambda p: p if os.path.isabs(p) else str(fb / p))
    if "out_path" in df:
        df["out_path"] = df["out_path"].astype(str).map(
            lambda p: p if os.path.isabs(p) else str(data / p))
    return df
