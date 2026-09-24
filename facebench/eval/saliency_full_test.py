#!/usr/bin/env python3
"""
Gradient saliency over the FULL B2_corrected test set, for Sec. 2.6 paragraph 4.
thread facebase3d-paper-2026-08-08.

WHY THIS EXISTS. The manuscript states that saliency maps "were generally
diffuse and did not reveal anatomical regions that were consistently associated
with diagnostic predictions". The only quantitative backing for that sentence
was runs/analysis/B2_saliency_stats.csv, which is unusable for the
paper on two counts:

  1. WRONG CHECKPOINT. It was produced by scripts/interp_saliency.py from
     runs/B2/ -- the ORIGINAL scan-level split. That is the
     leakage-affected run Sec. 2.7 reports as inflated. Every other number in
     the paper comes from B2_corrected.
  2. ENTROPY OF A CLASS MEAN, NOT OF A MAP. It averages saliency across all
     scans of a class and then measures the entropy of that average. Averaging
     point-indexed saliency across subjects who are NOT in dense correspondence
     destroys per-scan structure by construction, so the near-uniform entropy it
     reports (98-99.5% of maximum) is partly an artefact of the averaging.

This script recomputes on B2_corrected and separates the two questions the
original conflated:

  (a) WITHIN a scan, is saliency concentrated?      -> entropy, top-decile mass
  (b) ACROSS scans, is it concentrated in the SAME
      place, i.e. is there a consistent anatomical
      region?                                        -> voxel-grid agreement
                                                        between scans, against a
                                                        within-scan-shuffled null

Emits saliency_full_B2_corrected.npz (per-scan saliency + points + labels) and
saliency_full_B2_corrected_stats.csv.
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
OUTD = PROJECT / "runs/analysis"
SEED, N_CLASSES = 42, 33

device = torch.device("cpu")
torch.manual_seed(0)
model = PointNet2(num_classes=N_CLASSES)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.to(device).eval()

ds = FaceBaseDataset(MANIFEST, exp="clinical", split="test",
                     val_frac=0.15, test_frac=0.15, seed=SEED, augment=False)
label_names = [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]
print(f"[saliency] {len(ds)} test scans, {len(label_names)} classes", flush=True)

P, S, Y, PR = [], [], [], []
for i in range(len(ds)):
    pts, aux, lbl = ds[i]
    t = int(lbl)
    x = pts.unsqueeze(0).clone().requires_grad_(True)
    logit = model(x)
    pred = int(logit.argmax(1))
    model.zero_grad()
    logit[0, t].backward()
    sal = x.grad.detach()[0].norm(dim=1).numpy()
    P.append(pts.numpy().astype(np.float32))
    S.append(sal.astype(np.float32))
    Y.append(t); PR.append(pred)
    if (i + 1) % 25 == 0:
        print(f"  {i + 1}/{len(ds)}", flush=True)

P = np.stack(P); S = np.stack(S)
Y = np.array(Y); PR = np.array(PR)
np.savez_compressed(OUTD / "saliency_full_B2_corrected.npz",
                    pts=P, sal=S, y=Y, pred=PR,
                    label_names=np.array(label_names, dtype=object))
print(f"WROTE {OUTD / 'saliency_full_B2_corrected.npz'}  "
      f"pts={P.shape} sal={S.shape}")
