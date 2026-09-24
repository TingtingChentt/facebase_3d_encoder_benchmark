# FaceBase 3D Encoder — Model Implementations

## Input Format
All models accept: `(B, N, 3)` float32 tensor — batch of normalized point clouds, N=4096 points, unit sphere.

## Common Interface
All models expose:
- `encode(x)` → feature vector `(B, feat_dim)`
- `classify(feat)` → class logits `(B, num_classes)`
- `forward(x)` → class logits (encode + classify)
- `num_classes` attribute

## Models

### PointNet (`pointnet.py`)
Qi et al., CVPR 2017.  
Input/feature transform networks (T-Net) for pose invariance.  
Global max-pool aggregation.

| Component | Params |
|-----------|--------|
| T-Net (3→3) | ~0.08M |
| T-Net (64→64) | ~2.8M |
| MLP encoder | ~0.9M |
| Classifier head | ~0.7M |
| **Total** | **~4.5M** |

Feature dim: 1024  
T-Net regularization loss: orthogonality constraint (weight 0.001, applied in trainer).

---

### PointNet++ with MSG (`pointnet2.py`)
Qi et al., NeurIPS 2017.  
Multi-scale grouping set abstraction layers.  
Hierarchical local feature learning.

| Layer | npoint | radii | out_channels |
|-------|--------|-------|--------------|
| SA1 (MSG) | 512 | [0.1, 0.2, 0.4] | 320 |
| SA2 (MSG) | 128 | [0.2, 0.4, 0.8] | 640 |
| Global SA | — | — | 1024 |

**Total params: ~3.7M**  
Feature dim: 1024

---

### DGCNN (`dgcnn.py`)
Wang et al., TOG 2019.  
Dynamic graph construction in feature space (k=20 neighbors).  
EdgeConv + global avg+max pool.

| Component | Out channels |
|-----------|-------------|
| EdgeConv 1 | 64 |
| EdgeConv 2 | 64 |
| EdgeConv 3 | 128 |
| EdgeConv 4 | 256 |
| MLP fusion | 1024 |
| Global pool (avg+max) | 2048 |

**Total params: ~1.8M**  
Feature dim: 2048

---

### Geometric MLP (`geom_mlp.py`)
Handcrafted feature baseline (no learned 3D processing).  
PCA eigenvalues, projected coordinate statistics, bounding box, radial distribution.

| Feature group | Dim |
|---------------|-----|
| PCA eigenvalues | 3 |
| PCA proj mean/std/range | 9 |
| Bounding box | 6 |
| Radial stats | 5 |
| Asymmetry | 3 |
| **Total** | **26** |

MLP: 26 → 256 → 256 → 128 → num_classes  
**Total params: ~0.13M**  
Feature dim: 26

---

## Training Infrastructure

- **`dataset.py`** — PyTorch Dataset reading manifest CSV + .npy files. Stratified split, data augmentation (Y-axis rotation + jitter), inverse-frequency class weights.
- **`trainer.py`** — AdamW + cosine LR schedule + weighted cross-entropy + early stopping (patience=20) + CSV logging. PointNet T-Net regularization applied automatically.
- **`evaluate.py`** — accuracy, per-class F1, macro AUC (OvR), confusion matrix via sklearn.

## SLURM

- **M1 preprocessing:** A100 (`gpuq`), 8-way sharded array job (`m1_preprocess.sh`)
- **M3 training:** H100 (`gpu-xe9680q`, `time=10-00:00:00`) — scripts in `experiments/slurm/`

## Unit Tests

```bash
conda run -n fasebase python3 test_models.py
```
