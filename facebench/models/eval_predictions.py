#!/usr/bin/env python3
"""
Standalone script: load existing checkpoint, run test evaluation, save predictions.npz.

Usage:
  python eval_predictions.py --model pointnet2 --exp-id A1 \
      --manifest data/processed/ofc_manifest.csv --exp-type ofc
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).parent.parent.parent
MODELS_DIR = Path(__file__).parent
sys.path.insert(0, str(MODELS_DIR))

from dataset import FaceBaseDataset
from evaluate import evaluate
from pointnet import PointNet
from pointnet2 import PointNet2
from dgcnn import DGCNN
from geom_mlp import GeomMLP

MODEL_CLASSES = {
    "pointnet":  PointNet,
    "pointnet2": PointNet2,
    "dgcnn":     DGCNN,
    "geommlp":   GeomMLP,
}

CLASS_NAME_MAP = {
    "pointnet":  "PointNet",
    "pointnet2": "PointNet2",
    "dgcnn":     "DGCNN",
    "geommlp":   "GeomMLP",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(MODEL_CLASSES))
    p.add_argument("--exp-id", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--exp-type", required=True,
                   choices=["ofc", "ofc_a3", "ofc_a4", "syndrome", "clinical", "combined"])
    p.add_argument("--binary-mode", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aux-dim", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = PROJECT / "experiments" / "runs" / args.exp_id / args.model
    ckpt_dir = run_dir / "checkpoints"

    dataset_kwargs = dict(
        manifest_csv=args.manifest,
        exp=args.exp_type,
        seed=args.seed,
        binary_mode=args.binary_mode,
        augment=False,
    )
    test_ds = FaceBaseDataset(**dataset_kwargs, split="test")
    label_names = list(test_ds.label_map.keys())
    num_classes = test_ds.num_classes

    print(f"Test set: {len(test_ds)} scans | {num_classes} classes")

    model = MODEL_CLASSES[args.model](num_classes=num_classes)
    class_name = CLASS_NAME_MAP[args.model]
    ckpt_path = ckpt_dir / f"{class_name}_{args.exp_id}_best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)
    metrics = evaluate(model, test_loader, device, label_names,
                       use_aux=args.aux_dim > 0)

    out_path = run_dir / "predictions.npz"
    np.savez(
        out_path,
        scan_ids=np.array(test_ds.df["scan_id"].tolist()),
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        y_prob=metrics["y_prob"],
        label_names=np.array(label_names),
    )

    print(f"predictions.npz saved to {out_path}")
    print(f"  n_samples={metrics['n_samples']}  acc={metrics['accuracy']:.4f}  "
          f"auc={metrics['auc_macro_ovr']:.4f}")


if __name__ == "__main__":
    main()
