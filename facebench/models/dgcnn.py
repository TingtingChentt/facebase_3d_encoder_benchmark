"""
DGCNN — Dynamic Graph CNN.
Wang et al., 2019. https://arxiv.org/abs/1801.07829
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """Return k-NN indices. x: (B, C, N) → idx: (B, N, k)."""
    inner = -2 * torch.bmm(x.transpose(2, 1), x)  # (B, N, N)
    xx = (x ** 2).sum(1, keepdim=True)             # (B, 1, N)
    dist = inner + xx + xx.transpose(2, 1)         # (B, N, N)
    return dist.topk(k, dim=-1, largest=False)[1]  # (B, N, k)


def get_graph_feature(x, k, idx=None):
    """EdgeConv feature construction. x: (B, C, N) → (B, 2C, N, k)."""
    B, C, N = x.shape
    if idx is None:
        idx = knn(x, k)
    device = x.device
    idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N
    idx = idx + idx_base  # (B, N, k)
    idx = idx.view(-1)
    x_t = x.transpose(2, 1).contiguous().view(B * N, C)
    neighbor = x_t[idx].view(B, N, k, C).permute(0, 3, 1, 2)  # (B,C,N,k)
    x_exp = x.unsqueeze(-1).expand_as(neighbor)
    return torch.cat([x_exp, neighbor - x_exp], dim=1)  # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_c, out_c, k):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_c * 2, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        feat = get_graph_feature(x, self.k)        # (B, 2C, N, k)
        feat = self.conv(feat)                     # (B, out_c, N, k)
        return feat.max(dim=-1)[0]                 # (B, out_c, N)


class DGCNN(nn.Module):
    def __init__(self, num_classes: int, k: int = 20, emb_dims: int = 1024,
                 aux_dim: int = 0):
        super().__init__()
        self.k = k
        self.aux_dim = aux_dim
        self.ec1 = EdgeConv(3, 64, k)
        self.ec2 = EdgeConv(64, 64, k)
        self.ec3 = EdgeConv(64, 128, k)
        self.ec4 = EdgeConv(128, 256, k)
        self.conv = nn.Sequential(
            nn.Conv1d(64 + 64 + 128 + 256, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc1 = nn.Linear(emb_dims * 2 + aux_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.5)
        self.num_classes = num_classes
        self.embed_dim = 256

    def encode(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # (B, 3, N)
        x1 = self.ec1(x)
        x2 = self.ec2(x1)
        x3 = self.ec3(x2)
        x4 = self.ec4(x3)
        cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)
        x = self.conv(cat)                          # (B, emb, N)
        x_avg = x.mean(dim=-1)
        x_max = x.max(dim=-1)[0]
        feat = torch.cat([x_avg, x_max], dim=1)    # (B, emb*2)
        return feat

    def classify(self, feat, aux=None):
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(feat)), 0.2)
        x = self.drop(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.drop(x)
        return self.fc3(x)

    def embed(self, x, aux=None):
        """Return 256-dim pre-logit embedding (output of fc2, before fc3)."""
        feat = self.encode(x)
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.leaky_relu(self.bn1(self.fc1(feat)), 0.2)
        x = self.drop(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.2)
        x = self.drop(x)
        return x  # (B, 256)

    def forward(self, x, aux=None):
        return self.classify(self.encode(x), aux)
