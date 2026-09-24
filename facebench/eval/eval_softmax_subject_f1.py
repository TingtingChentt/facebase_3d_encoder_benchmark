#!/usr/bin/env python3
"""
Compute SUBJECT-LEVEL per-class F1 for the SOFTMAX head (PN++ B2_corrected),
to pair with the CE+Proto subject-level per-class F1 (already in its report).

Same test split (seed 42, syndrome_clinical_manifest.csv) as B2_proto_cetrain,
so the 169 subjects match. Softmax probabilities are averaged per subject
(fbid) -> argmax -> subject prediction, then per-class F1.

Output: runs/analysis/softmax_subject_perclass_f1.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, classification_report

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "facebench/models"))
from dataset import FaceBaseDataset          # noqa: E402
from pointnet2 import PointNet2              # noqa: E402
from torch.utils.data import DataLoader      # noqa: E402

MANIFEST = str(PROJECT / "data/processed/syndrome_clinical_manifest.csv")
CKPT     = str(PROJECT / "runs/B2_corrected/pointnet2/checkpoints/PointNet2_B2_corrected_best.pt")
OUT      = str(PROJECT / "runs/analysis/softmax_subject_perclass_f1.csv")
SEED, N_CLASSES = 42, 33

device = torch.device("cpu")
model = PointNet2(num_classes=N_CLASSES)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device).eval()

ds = FaceBaseDataset(MANIFEST, exp="clinical", split="test",
                     val_frac=0.15, test_frac=0.15, seed=SEED, augment=False)
label_names = [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]
fbids = ds.df["fbid"].tolist()
print(f"[eval] {len(ds)} test scans, {N_CLASSES} classes")

loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
probs, lbls = [], []
with torch.no_grad():
    for bi, (pts, aux, lbl) in enumerate(loader):
        logit = model(pts.to(device))
        probs.append(torch.softmax(logit, -1).cpu().numpy())
        lbls.append(lbl.numpy())
        print(f"  batch {bi+1}/{len(loader)}", flush=True)
probs = np.concatenate(probs); lbls = np.concatenate(lbls)

# subject aggregate: mean prob per fbid
subj_p, subj_y = {}, {}
for p, y, f in zip(probs, lbls, fbids):
    subj_p.setdefault(f, []).append(p)
    subj_y[f] = y
keys = list(subj_p.keys())
P = np.stack([np.stack(subj_p[k]).mean(0) for k in keys])
Y = np.array([subj_y[k] for k in keys])
pred = P.argmax(1)
print(f"[eval] {len(keys)} subjects | subject-level acc = {(pred==Y).mean():.4f} | "
      f"macro F1 = {f1_score(Y, pred, average='macro', zero_division=0):.4f}")

f1s = f1_score(Y, pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
support = np.bincount(Y, minlength=N_CLASSES)
df = pd.DataFrame({"class_name": label_names, "softmax_subj_f1": f1s, "n_subjects": support})
df.to_csv(OUT, index=False)
print("WROTE", OUT)
