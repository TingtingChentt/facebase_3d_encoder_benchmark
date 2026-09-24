"""
PointNet++ with Multi-Scale Grouping (MSG).
Qi et al., 2017. https://arxiv.org/abs/1706.02413
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def square_distance(src, dst):
    """Per-point squared distances. src/dst: (B, N, 3)."""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.bmm(src, dst.permute(0, 2, 1))
    dist += (src ** 2).sum(-1, keepdim=True)
    dist += (dst ** 2).sum(-1).unsqueeze(1)
    return dist  # (B, N, M)


def farthest_point_sample(xyz, npoint, generator=None):
    """
    FPS on GPU. xyz: (B, N, 3) → idx: (B, npoint).

    The start index is drawn uniformly at random, which is what makes PN++
    inference non-deterministic. Pass `generator` (a torch.Generator) to make
    that draw reproducible. The DISTRIBUTION is unchanged — it is still a
    uniform draw over [0, N); only the random stream is now controlled. Do not
    "fix" this by hard-coding a start index: that would change the estimator.

    generator=None preserves the original unseeded behaviour so existing
    callers are unaffected.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    if generator is None:
        farthest = torch.randint(0, N, (B,), device=device)
    else:
        # Draw on the generator's own device, then move, so a CPU generator
        # stays valid for a CUDA model (and vice versa).
        farthest = torch.randint(0, N, (B,), generator=generator,
                                 device=generator.device).to(device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[torch.arange(B), farthest].unsqueeze(1)  # (B,1,3)
        d = ((xyz - centroid) ** 2).sum(-1)
        dist = torch.min(dist, d)
        farthest = dist.argmax(-1)
    return idx


def index_points(pts, idx):
    """Gather points by index. pts: (B,N,C), idx: (B,S) or (B,S,K)."""
    B = pts.shape[0]
    device = pts.device
    if idx.dim() == 2:
        return pts[torch.arange(B, device=device).unsqueeze(1).expand_as(idx), idx]
    B, S, K = idx.shape
    out = pts[torch.arange(B, device=device).view(B,1,1).expand(B,S,K), idx]
    return out


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Ball query. xyz: (B,N,3), new_xyz: (B,S,3) → idx: (B,S,nsample)."""
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape
    dists = square_distance(new_xyz, xyz)  # (B,S,N)
    idx = dists.argsort(dim=-1)[:, :, :nsample]
    mask = dists.sort(dim=-1)[0][:, :, :nsample] > radius ** 2
    # Replace out-of-radius with first neighbour
    idx[mask] = idx[:, :, 0:1].expand_as(idx)[mask]
    return idx


class PointNetSetAbstractionMSG(nn.Module):
    def __init__(self, npoint, radii, nsamples, in_channel, mlp_lists):
        super().__init__()
        self.npoint = npoint
        self.radii = radii
        self.nsamples = nsamples
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for mlp in mlp_lists:
            convs, bns = nn.ModuleList(), nn.ModuleList()
            last = in_channel + 3
            for out_c in mlp:
                convs.append(nn.Conv2d(last, out_c, 1))
                bns.append(nn.BatchNorm2d(out_c))
                last = out_c
            self.convs.append(convs)
            self.bns.append(bns)

    def forward(self, xyz, points, generator=None):
        # xyz: (B,N,3), points: (B,N,C) or None
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint, generator=generator)
        new_xyz = index_points(xyz, fps_idx)  # (B,npoint,3)
        new_pts_list = []
        for i, (radius, nsample) in enumerate(zip(self.radii, self.nsamples)):
            idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B,npoint,nsample)
            grouped_xyz = index_points(xyz, idx) - new_xyz.unsqueeze(2)  # (B,npoint,nsample,3)
            if points is not None:
                grouped_pts = index_points(points, idx)  # (B,npoint,nsample,C)
                grouped = torch.cat([grouped_xyz, grouped_pts], dim=-1)
            else:
                grouped = grouped_xyz
            grouped = grouped.permute(0, 3, 2, 1)  # (B,C+3,nsample,npoint)
            for conv, bn in zip(self.convs[i], self.bns[i]):
                grouped = F.relu(bn(conv(grouped)))
            new_pts = grouped.max(dim=2)[0].permute(0, 2, 1)  # (B,npoint,out_C)
            new_pts_list.append(new_pts)
        return new_xyz, torch.cat(new_pts_list, dim=-1)


class PointNet2(nn.Module):
    def __init__(self, num_classes: int, aux_dim: int = 0):
        super().__init__()
        self.aux_dim = aux_dim
        self.sa1 = PointNetSetAbstractionMSG(
            npoint=512, radii=[0.1, 0.2, 0.4], nsamples=[16, 32, 128],
            in_channel=0,
            mlp_lists=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMSG(
            npoint=128, radii=[0.2, 0.4, 0.8], nsamples=[32, 64, 128],
            in_channel=64 + 128 + 128,
            mlp_lists=[[64, 64, 128], [128, 128, 256], [128, 128, 256]]
        )
        # Global SA
        self.conv1 = nn.Conv1d(128 + 256 + 256, 256, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)
        self.conv3 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024 + aux_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.4)
        self.num_classes = num_classes
        self.embed_dim = 256

    def encode(self, x, generator=None):
        # x: (B, N, 3)
        xyz, pts = self.sa1(x, None, generator=generator)
        xyz, pts = self.sa2(xyz, pts, generator=generator)
        pts_t = pts.permute(0, 2, 1)  # (B, C, npoint)
        pts_t = F.relu(self.bn1(self.conv1(pts_t)))
        pts_t = F.relu(self.bn2(self.conv2(pts_t)))
        pts_t = F.relu(self.bn3(self.conv3(pts_t)))
        feat = pts_t.max(dim=2)[0]  # (B, 1024)
        return feat

    def classify(self, feat, aux=None):
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.relu(self.bn4(self.fc1(feat)))
        x = self.drop(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.drop(x)
        return self.fc3(x)

    def embed(self, x, aux=None, generator=None):
        """Return 256-dim pre-logit embedding (output of fc2, before fc3)."""
        feat = self.encode(x, generator=generator)
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.relu(self.bn4(self.fc1(feat)))
        x = self.drop(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.drop(x)
        return x  # (B, 256)

    def forward(self, x, aux=None, generator=None):
        return self.classify(self.encode(x, generator=generator), aux)
