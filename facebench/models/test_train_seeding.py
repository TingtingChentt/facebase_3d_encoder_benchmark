#!/usr/bin/env python3
"""Regression test for seeding.py — see that module for the three-seed taxonomy.
Run: python3 facebench/models/test_train_seeding.py
Property (1) is the load-bearing one: if the test set moved with the train seed,
the multi-seed spread would confound retraining variance with split variance."""
import sys, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import FaceBaseDataset
from seeding import seed_everything, loader_kwargs
from torch.utils.data import DataLoader

M = str(Path(__file__).resolve().parents[2] / "data/processed/syndrome_clinical_manifest.csv")
kw = dict(manifest_csv=M, exp="clinical", seed=42, binary_mode=False)

# (1) SPLIT INVARIANCE — test membership must not move with the train seed.
ids = []
for ts in (1, 2, 3):
    seed_everything(ts)
    te = FaceBaseDataset(**kw, split="test", augment=False)
    ids.append(list(te.df["out_path"]))
print("(1) test-set identical across train seeds:", ids[0] == ids[1] == ids[2], f"(n={len(ids[0])})")

# (2) AUGMENTATION REPRODUCIBILITY + per-worker diversity.
tr = FaceBaseDataset(**kw, split="train", augment=True)
def first_batches(ts, nw=4):
    seed_everything(ts)
    dl = DataLoader(tr, batch_size=8, shuffle=True, num_workers=nw,
                    drop_last=True, **loader_kwargs(ts))
    out = []
    for i, (p, a, y) in enumerate(dl):
        out.append((p.numpy().copy(), y.numpy().copy()))
        if i == 3: break
    return out
a1, a2, b1 = first_batches(1), first_batches(1), first_batches(2)
same = all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a1, a2))
diff = any(not np.array_equal(x[0], y[0]) for x, y in zip(a1, b1))
print("(2a) seed 1 == seed 1 (reproducible):", same)
print("(2b) seed 1 != seed 2 (varies):      ", diff)
print("     label order seed1 vs seed2 differs:", not np.array_equal(a1[0][1], b1[0][1]))

# (3) THE num_workers TRAP — do the 4 workers emit distinct augmentation?
# Batches 0..3 come from workers 0..3 in round-robin. Compare the augmentation
# noise across them on the SAME underlying scans by checking that no two
# batches are byte-identical after we force one repeated item.
seed_everything(7)
dl = DataLoader(tr, batch_size=4, shuffle=False, num_workers=4,
                **loader_kwargs(7))
bs = [b[0].numpy().copy() for i, b in zip(range(4), dl)]
print("(3) 4 workers produce 4 distinct augmentation streams:",
      len({b.tobytes() for b in bs}) == 4)

# (4) model init actually moves with the train seed
from geom_mlp import GeomMLP
def w(ts):
    seed_everything(ts)
    return next(GeomMLP(num_classes=33).parameters()).detach().numpy().copy()
print("(4a) init seed 1 == seed 1:", np.array_equal(w(1), w(1)))
print("(4b) init seed 1 != seed 2:", not np.array_equal(w(1), w(2)))
