#!/usr/bin/env python3
"""
Interp: Gradient-based point saliency on B2 PN++ test set.
For each test sample: |d(logit_true)/d(input)| → per-point importance.
Aggregates per syndrome and generates 3D saliency visualizations.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "experiments" / "models"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

from dataset import FaceBaseDataset
from pointnet2 import PointNet2

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST_B2  = str(PROJECT / "data/processed/syndrome_clinical_manifest.csv")
CKPT_B2_PN2  = str(PROJECT / "runs/B2/pointnet2/checkpoints/PointNet2_B2_best.pt")
OUT_DIR = PROJECT / "runs/analysis"
SEED    = 42


def load_model(ckpt_path, device):
    model = PointNet2(num_classes=33)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def compute_saliency(model, pts_np, y_true, device):
    """Returns (saliency: (N,), y_pred: int)."""
    pts = torch.from_numpy(pts_np).float().unsqueeze(0).to(device)
    pts.requires_grad_(True)
    with torch.enable_grad():
        logits = model(pts)
        logits[0, y_true].backward()
    saliency = pts.grad.abs().sum(dim=-1).squeeze(0).detach().cpu().numpy()
    y_pred   = logits.detach().argmax(dim=-1).item()
    return saliency, y_pred


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading B2 test split...")
    ds          = FaceBaseDataset(MANIFEST_B2, exp="clinical", split="test",
                                  val_frac=0.15, test_frac=0.15, seed=SEED)
    label_names = [k for k, _ in sorted(ds.label_map.items(), key=lambda x: x[1])]
    n_classes   = ds.num_classes
    print(f"  {len(ds)} samples | {n_classes} classes")

    print("Loading B2 PN2 model...")
    model = load_model(CKPT_B2_PN2, device)

    # Per-class storage
    class_data = {i: {"pts": [], "sal": [], "correct": []} for i in range(n_classes)}

    print("Computing saliency for all test samples ...")
    for i in range(len(ds)):
        pts_t, lbl_t = ds[i]
        y_true = lbl_t.item()
        sal, y_pred = compute_saliency(model, pts_t.numpy(), y_true, device)
        is_correct  = (y_pred == y_true)

        class_data[y_true]["pts"].append(pts_t.numpy())
        class_data[y_true]["sal"].append(sal)
        class_data[y_true]["correct"].append(is_correct)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ds)}")

    print("Aggregating per-syndrome saliency ...")
    class_mean_pts = {}
    class_mean_sal = {}
    stats_rows     = []

    for cls_idx in range(n_classes):
        d = class_data[cls_idx]
        n_total   = len(d["pts"])
        n_correct = sum(d["correct"])
        if n_total == 0:
            continue

        # Prefer correctly predicted samples; fall back to all if n_correct < 5
        if n_correct >= 5:
            sel_pts = [p for p, ok in zip(d["pts"], d["correct"]) if ok]
            sel_sal = [s for s, ok in zip(d["sal"], d["correct"]) if ok]
        else:
            sel_pts = d["pts"]
            sel_sal = d["sal"]

        mean_pts = np.stack(sel_pts).mean(axis=0)  # (N, 3)
        mean_sal = np.stack(sel_sal).mean(axis=0)  # (N,)
        class_mean_pts[cls_idx] = mean_pts
        class_mean_sal[cls_idx] = mean_sal

        max_sal  = float(mean_sal.max())
        norm     = mean_sal / (mean_sal.sum() + 1e-8)
        entropy  = float(-np.sum(norm * np.log(norm + 1e-8)))
        stats_rows.append({
            "syndrome":              label_names[cls_idx],
            "n_test":                n_total,
            "n_correct":             n_correct,
            "mean_max_saliency":     max_sal,
            "mean_entropy_saliency": entropy,
        })

    # Save CSV
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUT_DIR / "B2_saliency_stats.csv", index=False)
    print(f"Saved: {OUT_DIR}/B2_saliency_stats.csv")

    # Top-10 by test sample count
    top10 = stats_df.sort_values("n_test", ascending=False).head(10)
    top10_idx = [ds.label_map[name] for name in top10["syndrome"]]
    print(f"Top-10 syndromes: {top10['syndrome'].tolist()}")

    # Individual 3D saliency plots
    print("Generating individual saliency plots ...")
    for cls_idx in top10_idx:
        pts = class_mean_pts[cls_idx]
        sal = class_mean_sal[cls_idx]
        name     = label_names[cls_idx]
        row      = stats_df[stats_df["syndrome"] == name].iloc[0]
        safe     = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")

        fig = plt.figure(figsize=(7, 6))
        ax  = fig.add_subplot(111, projection="3d")
        sc  = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                         c=sal, cmap="plasma", s=2, alpha=0.7)
        plt.colorbar(sc, ax=ax, shrink=0.5, label="Mean Saliency")
        ax.set_title(f"{name}\n(n={int(row['n_test'])}, correct={int(row['n_correct'])})",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        out_path = OUT_DIR / f"saliency_B2_pn2_{safe}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # 2×5 summary grid
    print("Generating summary grid ...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9),
                              subplot_kw={"projection": "3d"})
    for ax, cls_idx in zip(axes.flat, top10_idx):
        pts  = class_mean_pts[cls_idx]
        sal  = class_mean_sal[cls_idx]
        name = label_names[cls_idx]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c=sal, cmap="plasma", s=1, alpha=0.7)
        ax.set_title(name[:28], fontsize=6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    fig.suptitle("B2 PointNet++ — Top-10 Syndrome Gradient Saliency Maps", fontsize=12)
    plt.tight_layout()
    out_path = OUT_DIR / "fig_interp_saliency_B2_top10.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    print("\nInterp saliency complete.")


if __name__ == "__main__":
    main()
