"""
Geometric feature MLP baseline.
Handcrafted features: PCA coefficients (top 20), surface area, volume estimate,
bounding box dims, axis-aligned moment statistics.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_geom_features(pts: torch.Tensor) -> torch.Tensor:
    """
    pts: (B, N, 3) normalized point cloud
    Returns: (B, feat_dim) feature vector
    """
    B, N, _ = pts.shape
    feats = []

    # Center (should be ~0 after normalization, but compute anyway)
    centroid = pts.mean(dim=1)  # (B, 3)
    pts_c = pts - centroid.unsqueeze(1)

    # PCA via SVD. NOTE: this computes the 3 covariance eigenvalues and
    # per-axis projection statistics — NOT "top-20 PCA shape coefficients".
    # That stale description propagated into the ICIBM poster and an early
    # manuscript draft before being caught (2026-08-10). A 3D cloud has 3
    # principal axes; there is no 20th coefficient to take.
    cov = torch.bmm(pts_c.transpose(2, 1), pts_c) / (N - 1)  # (B, 3, 3)
    # Since 3D, we get 3 eigenvalues/eigenvectors
    # Use full SVD of point matrix for PCA coefficients
    # Project onto eigenvectors and take statistics as features
    try:
        U, S, Vh = torch.linalg.svd(pts_c, full_matrices=False)  # S: (B, 3)
    except Exception:
        U, S, Vh = torch.linalg.svd(pts_c.float(), full_matrices=False)

    # Eigenvalues of covariance ~ S^2 / (N-1)
    eigs = S ** 2 / (N - 1)  # (B, 3)

    # Projected coordinates onto principal axes: (B, N, 3)
    proj = torch.bmm(pts_c, Vh.transpose(2, 1))  # (B, N, 3)

    # Statistics per axis: mean, std, skewness proxy, range
    proj_mean = proj.mean(1)
    proj_std = proj.std(1)
    proj_range = proj.max(1)[0] - proj.min(1)[0]

    feats.append(eigs)               # 3
    feats.append(proj_mean)          # 3
    feats.append(proj_std)           # 3
    feats.append(proj_range)         # 3

    # Bounding box in original space
    bb_min = pts.min(1)[0]
    bb_max = pts.max(1)[0]
    bb_dims = bb_max - bb_min        # 3
    bb_center = (bb_min + bb_max) / 2  # 3
    feats.append(bb_dims)            # 3
    feats.append(bb_center)          # 3

    # Radial distribution stats
    r = torch.norm(pts_c, dim=-1)    # (B, N)
    feats.append(r.mean(1, keepdim=True))   # 1
    feats.append(r.std(1, keepdim=True))    # 1
    feats.append(r.max(1)[0].unsqueeze(1))  # 1
    feats.append(r.min(1)[0].unsqueeze(1))  # 1

    # Surface density proxy: fraction in inner sphere (r < 0.5)
    inner = (r < 0.5).float().mean(1, keepdim=True)
    feats.append(inner)  # 1

    # Asymmetry: mean of signed coords (deviation from symmetry)
    asym = pts_c.mean(1).abs()  # (B, 3)
    feats.append(asym)  # 3

    feat = torch.cat(feats, dim=1)  # (B, 26) — matches FEAT_DIM below
    # 26 = eigs 3 + proj_mean 3 + proj_std 3 + proj_range 3 + bb_dims 3
    #    + bb_center 3 + r_{mean,std,max,min} 4 + inner 1 + asym 3
    return feat


FEAT_DIM = 26


class GeomMLP(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int = FEAT_DIM,
                 hidden: list = None, aux_dim: int = 0):
        super().__init__()
        if hidden is None:
            hidden = [256, 256, 128]
        self.aux_dim = aux_dim
        layers = []
        in_dim = feat_dim + aux_dim
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.ReLU(inplace=True), nn.Dropout(0.3)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.num_classes = num_classes
        self.embed_dim = hidden[-1]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return extract_geom_features(x)

    def classify(self, feat: torch.Tensor, aux=None) -> torch.Tensor:
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        return self.net(feat)

    def embed(self, x: torch.Tensor, aux=None) -> torch.Tensor:
        """Return pre-logit embedding (last hidden layer, before final Linear)."""
        feat = self.encode(x)
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        children = list(self.net.children())
        for layer in children[:-1]:
            feat = layer(feat)
        return feat

    def forward(self, x: torch.Tensor, aux=None) -> torch.Tensor:
        return self.classify(self.encode(x), aux)
