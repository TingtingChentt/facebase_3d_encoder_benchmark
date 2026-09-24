#!/usr/bin/env python3
"""
Gradient saliency on EXEMPLAR test scans, rendered as faces (frontal projection).

Why this exists: the pre-existing saliency PNGs
(runs/analysis/saliency_B2_pn2_*.png) average saliency over every
scan of a class and draw it in a 3D axes box -- the result is a diffuse blob that
reads as noise, which is why the earlier poster deliberately omitted it. Here we
instead take ONE correctly-classified test scan per syndrome and project it
frontally, so the output actually looks like a face.

Model:  PointNet2 B2_corrected (33-class clinical diagnosis), CPU.
Split:  same as every other B2 number -- clinical, seed 42, subject-level test.
Saliency: |d logit_true / d xyz| per point, L2 over the 3 coords, then
          percentile-normalised per scan for display only.

Emits runs/analysis/saliency_exemplars.npz with, per syndrome:
points (4096,3), saliency (4096,), scan_id, fbid, predicted label, margin.

Usage: scripts/gen_saliency_exemplars.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "facebench/models"))
from dataset import FaceBaseDataset          # noqa: E402
from pointnet2 import PointNet2              # noqa: E402

MANIFEST = str(PROJECT / "data/processed/syndrome_clinical_manifest.csv")
CKPT = str(PROJECT / "runs/B2_corrected/pointnet2/checkpoints/"
                     "PointNet2_B2_corrected_best.pt")
OUT = PROJECT / "runs/analysis/saliency_exemplars.npz"
SEED, N_CLASSES = 42, 33

# MATCHED-CLASS-SIZE PAIRS. Class size correlates with recognisability
# (Spearman rho = +0.53 subject-level), so a phenotype claim can only be made by
# holding n fixed. Each column of the figure is a pair with the SAME n_subjects:
#   n=10  Achondroplasia (F1 0.80)  vs  Phelan McDermid (0.33)
#   n= 7  Costello       (F1 0.67)  vs  CHARGE          (0.20)
#   n= 9/4 Williams      (F1 0.53)  vs  Rett            (0.40)
# Apert and Crouzon are NOT here: under the corrected subject-level split they
# have n=1 test subject each and F1 = 0.000, so they cannot be cited as
# "best-recognised" (their high F1 in B2_perclass_metrics.csv comes from a
# different, larger 530-scan split, not this 475-scan corrected one).
WANT = ["Achondroplasia", "Phelan McDermid Syndrome",
        "Costello Syndrome", "CHARGE Syndrome",
        "Williams Syndrome", "Rett Syndrome"]

device = torch.device("cpu")
model = PointNet2(num_classes=N_CLASSES)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device).eval()

ds = FaceBaseDataset(MANIFEST, exp="clinical", split="test",
                     val_frac=0.15, test_frac=0.15, seed=SEED, augment=False)
label_names = [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]
print(f"[exemplars] {len(ds)} test scans, {len(label_names)} classes")

out = {}
for target in WANT:
    if target not in ds.label_map:
        print(f"  !! {target}: not in label map, skipping")
        continue
    tgt = ds.label_map[target]
    idxs = [i for i in range(len(ds)) if int(ds.df["_label"].iloc[i]
            if "_label" in ds.df.columns else -1) == tgt]
    if not idxs:                       # fall back to reading labels from the dataset
        idxs = [i for i in range(len(ds)) if int(ds[i][2]) == tgt]
    best = None
    for i in idxs:
        pts, aux, lbl = ds[i]
        x = pts.unsqueeze(0).clone().requires_grad_(True)
        logit = model(x)
        p = torch.softmax(logit, 1)[0]
        pred = int(logit.argmax(1))
        # margin between the true class and the strongest rival
        rival = float(p[[c for c in range(N_CLASSES) if c != tgt]].max())
        margin = float(p[tgt]) - rival
        model.zero_grad()
        logit[0, tgt].backward()
        sal = x.grad.detach()[0].norm(dim=1).numpy()      # (4096,)
        rec = dict(pts=pts.numpy(), sal=sal, pred=pred, margin=margin,
                   correct=(pred == tgt), scan=str(ds.df["scan_id"].iloc[i]),
                   fbid=str(ds.df["fbid"].iloc[i]))
        # prefer a correctly-classified scan, then the most confident one
        key = (rec["correct"], rec["margin"])
        if best is None or key > (best["correct"], best["margin"]):
            best = rec
    if best is None:
        print(f"  !! {target}: no test scans")
        continue
    out[target] = best
    print(f"  {target:<22} n_test={len(idxs):>3}  exemplar={best['scan']}  "
          f"correct={best['correct']}  margin={best['margin']:+.3f}  "
          f"pred={label_names[best['pred']]}")

np.savez_compressed(
    OUT,
    **{f"{k}|{f}": np.asarray(v[f]) for k, v in out.items()
       for f in ("pts", "sal", "margin", "correct", "scan", "fbid", "pred")})
print(f"WROTE {OUT}  ({len(out)} syndromes)")
