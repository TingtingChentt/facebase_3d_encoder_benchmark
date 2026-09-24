#!/usr/bin/env python3
"""
M4: Per-class precision/recall/F1/AUC for B1 (19-class) and B2 (33-class).
Outputs: CSVs + per-class F1 bar charts + PN2 confusion matrix heatmaps.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "experiments" / "models"))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
)

from dataset import FaceBaseDataset
from pointnet2 import PointNet2
from dgcnn import DGCNN

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST_B2   = str(PROJECT / "data/processed/syndrome_clinical_manifest.csv")
MANIFEST_B1   = str(PROJECT / "data/processed/syndrome_manifest_b1.csv")
CKPT_B2_PN2   = str(PROJECT / "runs/B2/pointnet2/checkpoints/PointNet2_B2_best.pt")
CKPT_B2_DGCNN = str(PROJECT / "runs/B2/dgcnn/checkpoints/DGCNN_B2_best.pt")
CKPT_B1_PN2   = str(PROJECT / "runs/B1/pointnet2/checkpoints/PointNet2_B1_best.pt")
CKPT_B1_DGCNN = str(PROJECT / "runs/B1/dgcnn/checkpoints/DGCNN_B1_best.pt")
OUT_DIR = PROJECT / "runs/analysis"
SEED    = 42
BATCH   = 32


def load_model(model_class, num_classes, ckpt_path, device):
    model = model_class(num_classes=num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def run_inference(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds, labels, probs = [], [], []
    with torch.no_grad():
        for pts, lbl in loader:
            pts = pts.to(device)
            logits = model(pts)
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            preds.append(logits.argmax(dim=-1).cpu().numpy())
            labels.append(lbl.numpy())
    return np.concatenate(labels), np.concatenate(preds), np.concatenate(probs)


def perclass_metrics(y_true, y_pred, y_prob, label_names, model_name):
    n = len(label_names)
    cls_range = list(range(n))
    prec    = precision_score(y_true, y_pred, average=None, zero_division=0, labels=cls_range)
    rec     = recall_score(y_true, y_pred, average=None, zero_division=0, labels=cls_range)
    f1      = f1_score(y_true, y_pred, average=None, zero_division=0, labels=cls_range)
    support = np.bincount(y_true, minlength=n)

    auc_ovr = []
    for i in range(n):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            auc_ovr.append(float("nan"))
            continue
        try:
            auc_ovr.append(roc_auc_score(y_bin, y_prob[:, i]))
        except Exception:
            auc_ovr.append(float("nan"))

    return pd.DataFrame({
        "class_name": label_names,
        "precision":  prec,
        "recall":     rec,
        "f1":         f1,
        "support":    support.astype(int),
        "auc_ovr":    auc_ovr,
        "model":      model_name,
    })


def plot_perclass_f1(df_pn2, df_dgcnn, title, out_path):
    classes  = df_pn2.sort_values("f1", ascending=False)["class_name"].tolist()
    pn2_f1   = df_pn2.set_index("class_name").loc[classes, "f1"].values
    dgcnn_f1 = df_dgcnn.set_index("class_name").loc[classes, "f1"].values
    y = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(10, max(6, len(classes) * 0.35)))
    ax.barh(y + 0.2, pn2_f1,   height=0.38, label="PointNet++", color="#2196F3")
    ax.barh(y - 0.2, dgcnn_f1, height=0.38, label="DGCNN",     color="#FF5722")
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("F1 Score")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confmat(y_true, y_pred, label_names, title, out_path):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    n       = len(label_names)
    sz      = max(10, n * 0.42)

    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=90, fontsize=6)
    ax.set_yticklabels(label_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def label_names_from_dataset(ds):
    return [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("Loading B2 test split (clinical, 33 classes)...")
    ds_b2 = FaceBaseDataset(MANIFEST_B2, exp="clinical", split="test",
                             val_frac=0.15, test_frac=0.15, seed=SEED)
    b2_names = label_names_from_dataset(ds_b2)
    print(f"  {len(ds_b2)} samples | {ds_b2.num_classes} classes")

    print("Loading B1 test split (syndrome, 19 classes)...")
    ds_b1 = FaceBaseDataset(MANIFEST_B1, exp="syndrome", split="test",
                             val_frac=0.15, test_frac=0.15, seed=SEED)
    b1_names = label_names_from_dataset(ds_b1)
    print(f"  {len(ds_b1)} samples | {ds_b1.num_classes} classes")

    # Load models
    print("Loading models...")
    b2_pn2   = load_model(PointNet2, 33, CKPT_B2_PN2,   device)
    b2_dgcnn = load_model(DGCNN,     33, CKPT_B2_DGCNN, device)
    b1_pn2   = load_model(PointNet2, 19, CKPT_B1_PN2,   device)
    b1_dgcnn = load_model(DGCNN,     19, CKPT_B1_DGCNN, device)

    # Inference
    print("Running inference...")
    b2_true, b2_pn2_pred,   b2_pn2_prob   = run_inference(b2_pn2,   ds_b2, device, BATCH)
    _,       b2_dgcnn_pred, b2_dgcnn_prob  = run_inference(b2_dgcnn, ds_b2, device, BATCH)
    b1_true, b1_pn2_pred,   b1_pn2_prob   = run_inference(b1_pn2,   ds_b1, device, BATCH)
    _,       b1_dgcnn_pred, b1_dgcnn_prob  = run_inference(b1_dgcnn, ds_b1, device, BATCH)

    # Per-class metrics
    df_b2_pn2   = perclass_metrics(b2_true, b2_pn2_pred,   b2_pn2_prob,   b2_names, "PN2")
    df_b2_dgcnn = perclass_metrics(b2_true, b2_dgcnn_pred, b2_dgcnn_prob, b2_names, "DGCNN")
    df_b1_pn2   = perclass_metrics(b1_true, b1_pn2_pred,   b1_pn2_prob,   b1_names, "PN2")
    df_b1_dgcnn = perclass_metrics(b1_true, b1_dgcnn_pred, b1_dgcnn_prob, b1_names, "DGCNN")

    # Save CSVs
    pd.concat([df_b2_pn2, df_b2_dgcnn], ignore_index=True).to_csv(
        OUT_DIR / "B2_perclass_metrics.csv", index=False)
    pd.concat([df_b1_pn2, df_b1_dgcnn], ignore_index=True).to_csv(
        OUT_DIR / "B1_perclass_metrics.csv", index=False)
    print(f"Saved CSVs: B2_perclass_metrics.csv, B1_perclass_metrics.csv")

    # Figures
    plot_perclass_f1(df_b2_pn2, df_b2_dgcnn,
                     "B2 Per-Class F1 — 33 Clinical Diagnoses",
                     OUT_DIR / "fig_B2_perclass_f1.png")
    plot_perclass_f1(df_b1_pn2, df_b1_dgcnn,
                     "B1 Per-Class F1 — 19 Syndrome Categories",
                     OUT_DIR / "fig_B1_perclass_f1.png")
    plot_confmat(b2_true, b2_pn2_pred, b2_names,
                 "B2 PointNet++ Confusion Matrix (row-normalized)",
                 OUT_DIR / "fig_B2_confmat_pn2.png")
    plot_confmat(b1_true, b1_pn2_pred, b1_names,
                 "B1 PointNet++ Confusion Matrix (row-normalized)",
                 OUT_DIR / "fig_B1_confmat_pn2.png")

    print("\nM4 complete.")


if __name__ == "__main__":
    main()
