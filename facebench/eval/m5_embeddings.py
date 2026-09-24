#!/usr/bin/env python3
"""
M5: Extract B1/B2 embeddings, run UMAP + t-SNE, plot 2×2 scatter grids.
Falls back to t-SNE-only if umap-learn is not available.
"""

import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "experiments" / "models"))

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
OUT_DIR    = PROJECT / "runs/analysis"
SEED       = 42
TSNE_MAX   = 3000


def load_model(model_class, num_classes, ckpt_path, device):
    model = model_class(num_classes=num_classes)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_all_splits(manifest_csv, exp, seed=42):
    """Return (ConcatDataset, int_labels_array, label_map) for all 3 splits."""
    kw = dict(manifest_csv=manifest_csv, exp=exp,
               val_frac=0.15, test_frac=0.15, seed=seed)
    ds_tr = FaceBaseDataset(split="train", **kw)
    ds_va = FaceBaseDataset(split="val",   **kw)
    ds_te = FaceBaseDataset(split="test",  **kw)
    label_map = ds_tr.label_map
    lc = ds_tr.label_col

    labels = np.array(
        [label_map[r] for r in ds_tr.df[lc]] +
        [label_map[r] for r in ds_va.df[lc]] +
        [label_map[r] for r in ds_te.df[lc]],
        dtype=np.int64,
    )
    combined = ConcatDataset([ds_tr, ds_va, ds_te])
    return combined, labels, label_map


def extract_embeddings(model, dataset, device, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    feats = []
    with torch.no_grad():
        for pts, _ in loader:
            feats.append(model.encode(pts.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0)


def try_umap(feats, seed=SEED):
    try:
        import umap as umap_module
    except ImportError:
        print("umap-learn not installed — attempting pip install ...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "umap-learn", "--quiet"],
                       check=False)
        try:
            import umap as umap_module
        except ImportError:
            print("umap-learn install failed — falling back to t-SNE only.")
            return None
    reducer = umap_module.UMAP(n_components=2, random_state=seed,
                               n_neighbors=30, min_dist=0.1)
    return reducer.fit_transform(feats)


def run_tsne(feats, seed=SEED, max_n=TSNE_MAX):
    if len(feats) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(feats), max_n, replace=False)
        xy  = TSNE(n_components=2, random_state=seed, perplexity=40,
                   n_iter=1000).fit_transform(feats[idx])
        return xy, idx
    xy = TSNE(n_components=2, random_state=seed, perplexity=40,
              n_iter=1000).fit_transform(feats)
    return xy, None


def scatter_2x2(panels, suptitle, out_path, label_map):
    """panels: list of 4 (xy, labels, subtitle) tuples."""
    label_names = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes   = len(label_names)
    cmap        = plt.cm.get_cmap("tab20", n_classes)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    for ax, (xy, labels, subtitle) in zip(axes.flat, panels):
        for i, name in enumerate(label_names):
            mask = labels == i
            if mask.sum() == 0:
                continue
            ax.scatter(xy[mask, 0], xy[mask, 1], s=5, alpha=0.6,
                       color=cmap(i), label=name, rasterized=True)
        ax.set_title(subtitle, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="center right", fontsize=6,
               bbox_to_anchor=(1.16, 0.5), markerscale=3)
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def process_experiment(feats_pn2, feats_dgcnn, labels, label_map,
                       exp_tag, has_umap):
    """Run UMAP + t-SNE and return panel list for 2×2 plot."""
    # t-SNE (subsample if needed)
    print(f"  [{exp_tag}] t-SNE PN2 ...")
    tsne_pn2,   idx_pn2   = run_tsne(feats_pn2)
    labels_pn2  = labels[idx_pn2] if idx_pn2 is not None else labels

    print(f"  [{exp_tag}] t-SNE DGCNN ...")
    tsne_dgcnn, idx_dgcnn = run_tsne(feats_dgcnn)
    labels_dgcnn = labels[idx_dgcnn] if idx_dgcnn is not None else labels

    if has_umap:
        print(f"  [{exp_tag}] UMAP PN2 ...")
        umap_pn2   = try_umap(feats_pn2)
        print(f"  [{exp_tag}] UMAP DGCNN ...")
        umap_dgcnn = try_umap(feats_dgcnn)
        panels = [
            (umap_pn2,   labels,       "UMAP — PointNet++"),
            (umap_dgcnn, labels,       "UMAP — DGCNN"),
            (tsne_pn2,   labels_pn2,   "t-SNE — PointNet++"),
            (tsne_dgcnn, labels_dgcnn, "t-SNE — DGCNN"),
        ]
    else:
        panels = [
            (tsne_pn2,   labels_pn2,   "t-SNE — PointNet++ (UMAP unavailable)"),
            (tsne_dgcnn, labels_dgcnn, "t-SNE — DGCNN (UMAP unavailable)"),
            (tsne_pn2,   labels_pn2,   "t-SNE — PointNet++ (duplicate row)"),
            (tsne_dgcnn, labels_dgcnn, "t-SNE — DGCNN (duplicate row)"),
        ]
    return panels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check UMAP availability once
    print("Checking umap-learn ...")
    test_umap = try_umap(np.random.randn(10, 8).astype(np.float32))
    has_umap  = test_umap is not None
    print(f"UMAP available: {has_umap}")

    # Load all data
    print("Loading all B2 samples (train+val+test)...")
    b2_all, b2_labels, b2_lmap = load_all_splits(MANIFEST_B2, "clinical")
    print(f"  B2 total: {len(b2_all)} samples | {len(b2_lmap)} classes")

    print("Loading all B1 samples (train+val+test)...")
    b1_all, b1_labels, b1_lmap = load_all_splits(MANIFEST_B1, "syndrome")
    print(f"  B1 total: {len(b1_all)} samples | {len(b1_lmap)} classes")

    # Load models
    print("Loading models...")
    b2_pn2   = load_model(PointNet2, 33, CKPT_B2_PN2,   device)
    b2_dgcnn = load_model(DGCNN,     33, CKPT_B2_DGCNN, device)
    b1_pn2   = load_model(PointNet2, 19, CKPT_B1_PN2,   device)
    b1_dgcnn = load_model(DGCNN,     19, CKPT_B1_DGCNN, device)

    # Extract embeddings
    print("Extracting B2 embeddings...")
    b2_feat_pn2   = extract_embeddings(b2_pn2,   b2_all, device)
    b2_feat_dgcnn = extract_embeddings(b2_dgcnn, b2_all, device)
    print("Extracting B1 embeddings...")
    b1_feat_pn2   = extract_embeddings(b1_pn2,   b1_all, device)
    b1_feat_dgcnn = extract_embeddings(b1_dgcnn, b1_all, device)

    # Save B2 embeddings
    np.save(OUT_DIR / "embeddings_B2_pn2.npy",   b2_feat_pn2)
    np.save(OUT_DIR / "embeddings_B2_dgcnn.npy", b2_feat_dgcnn)
    np.save(OUT_DIR / "embeddings_B2_labels.npy", b2_labels)
    print(f"Saved B2 embeddings: pn2={b2_feat_pn2.shape} dgcnn={b2_feat_dgcnn.shape}")

    # Dimensionality reduction + plots
    print("Running dimensionality reduction for B2 ...")
    b2_panels = process_experiment(b2_feat_pn2, b2_feat_dgcnn, b2_labels,
                                   b2_lmap, "B2", has_umap)
    scatter_2x2(b2_panels, "B2 Embeddings — 33 Clinical Diagnoses",
                OUT_DIR / "fig_M5_embeddings_B2.png", b2_lmap)

    print("Running dimensionality reduction for B1 ...")
    b1_panels = process_experiment(b1_feat_pn2, b1_feat_dgcnn, b1_labels,
                                   b1_lmap, "B1", has_umap)
    scatter_2x2(b1_panels, "B1 Embeddings — 19 Syndrome Categories",
                OUT_DIR / "fig_M5_embeddings_B1.png", b1_lmap)

    print("\nM5 complete.")


if __name__ == "__main__":
    main()
