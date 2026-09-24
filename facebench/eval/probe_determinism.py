#!/usr/bin/env python3
"""
TASK 1 — non-determinism audit probe.

Empirically tests every encoder's INFERENCE path for run-to-run variation,
rather than assuming it from code reading. For each encoder we run the same
frozen checkpoint over the same fixed batch twice and report max|delta|.

Encoders: GeomMLP, PointNet, DGCNN, PointNet++.
Outputs: results/audit/determinism_probe.csv

INFERENCE ONLY. Loads checkpoints read-only, never writes them.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "facebench/models"))

from dataset import FaceBaseDataset          # noqa: E402
from pointnet import PointNet                # noqa: E402
from pointnet2 import PointNet2              # noqa: E402
from dgcnn import DGCNN                      # noqa: E402
from geom_mlp import GeomMLP                 # noqa: E402
from torch.utils.data import DataLoader      # noqa: E402

MODEL_CLASSES = {
    "geommlp": GeomMLP,
    "pointnet": PointNet,
    "dgcnn": DGCNN,
    "pointnet2": PointNet2,
}
CKPT_NAME = {
    "geommlp": "GeomMLP", "pointnet": "PointNet",
    "dgcnn": "DGCNN", "pointnet2": "PointNet2",
}

# Probe on B2_corrected — smallest syndrome task, carries the core claim.
EXP = "B2_corrected"
MANIFEST = str(PROJECT / "data/processed/syndrome_clinical_manifest.csv")
EXP_TYPE, N_CLASSES, AUX_DIM = "clinical", 33, 0
# Per-encoder probe batch. DGCNN materialises a (B,4096,4096) neighbour graph
# (~2.1 GB at B=32), so it gets a small batch; the others can take more.
N_PROBE = {"geommlp": 32, "pointnet": 16, "dgcnn": 4, "pointnet2": 8}

OUT_DIR = PROJECT / "results/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_key, num_classes, aux_dim, exp):
    cls = MODEL_CLASSES[model_key]
    ckpt = (PROJECT / "runs" / exp / model_key /
            "checkpoints" / f"{CKPT_NAME[model_key]}_{exp}_best.pt")
    model = cls(num_classes=num_classes, aux_dim=aux_dim)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()
    return model, ckpt


def main():
    device = torch.device("cpu")
    ds = FaceBaseDataset(MANIFEST, exp=EXP_TYPE, split="test",
                         val_frac=0.15, test_frac=0.15, seed=42, augment=False)
    print(f"[probe] test split: {len(ds)} scans")

    rows = []
    for key in ["geommlp", "pointnet", "dgcnn", "pointnet2"]:
        n = N_PROBE[key]
        # Fixed batch, loaded once — removes the data path from the comparison.
        loader = DataLoader(ds, batch_size=n, shuffle=False, num_workers=0)
        pts, aux, lbl = next(iter(loader))
        pts = pts.to(device)

        model, ckpt = load_model(key, N_CLASSES, AUX_DIM, EXP)

        # Confirm eval() actually took: no module left in training mode.
        training_mods = [n_ for n_, m in model.named_modules() if m.training]

        t0 = time.time()
        with torch.no_grad():
            f1 = model.encode(pts)
        dt = time.time() - t0
        with torch.no_grad():
            f2 = model.encode(pts)

        delta = (f1 - f2).abs().max().item()
        rows.append({
            "encoder": key,
            "max_abs_delta_encode": delta,
            "deterministic": bool(delta == 0.0),
            "modules_left_in_train_mode": len(training_mods),
            "sec_per_scan_cpu": round(dt / n, 4),
            "n_probe": n,
            "checkpoint": str(ckpt.relative_to(PROJECT)),
        })
        print(f"  {key:10s} max|delta|={delta:.3e}  "
              f"{'DETERMINISTIC' if delta == 0 else 'NON-DETERMINISTIC'}  "
              f"{dt / n:.3f}s/scan (B={n})  train-mode-modules={len(training_mods)}",
              flush=True)

    import csv
    out = OUT_DIR / "determinism_probe.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWROTE {out}")

    print("\n[env]")
    print(json.dumps({
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "num_threads": torch.get_num_threads(),
    }, indent=2))


if __name__ == "__main__":
    main()
