"""
Shared constants and loaders for the region-occlusion figures
(analyses/interpretability/main_figure_interp.py).

  find(dirs, scan_id)        landmark directory that holds <scan_id>.npz
  geometry(scan_id, row, d)  normalised mesh + landmark frame for rendering
  wrong_agreement()          fbid -> how many of the 5 models made the most common prediction
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

from analyses.interpretability import render_lib as RL                                             # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
IN = PROJECT / "results/interp"
OUT = IN / "tables"
PRED = PROJECT / "results/predictions"
IMG = 360
IMP_CMAP = plt.cm.RdBu_r
REG_COLS = plt.cm.tab10(np.arange(8)); REG_COLS[7] = [.85, .85, .85, 1]
MAIN = ["Rhizomelic Chondrodysplasia Punctata", "Mucopolysaccharidosis", "Cockayne Syndrome",
        "Pallister-Killian Syndrome", "Phelan McDermid Syndrome", "Down Syndrome", "Achondroplasia"]
LM_DIRS = [IN / "landmarks", IN / "landmarks_ext"]
LM_NN_DIRS = [IN / "landmarks_nn", IN / "landmarks_nn_ext"]
N_BOOT = 10000


def find(dirs, scan_id):
    for d in dirs:
        if (d / f"{scan_id}.npz").exists():
            return d
    return None


def wrong_agreement():
    """fbid -> number of the 5 models that made the most common (wrong) prediction."""
    rows = []
    for ts in range(1, 6):
        d = np.load(PRED / f"CF20_full_v3_ts{ts}__pointnet2__proto__subject__seed0.npz", allow_pickle=True)
        rows.append(pd.DataFrame(dict(fbid=d["subject_of_unit"].astype(str), p=d["y_pred"])))
    pr = pd.concat(rows, ignore_index=True)
    return pr.groupby(["fbid", "p"]).size().groupby("fbid").max()


def geometry(scan_id, row, lm_dir):
    d = np.load(lm_dir / f"{scan_id}.npz")
    V, F = RL.load_ply(row["src_path"])
    Vn = (V - d["c"]) / d["s"]
    R = RL.view_matrix(bool(d["view"][0]))
    return d["lm_norm"], Vn, F, R, RL.face_crop(d["lm_norm"], R)
