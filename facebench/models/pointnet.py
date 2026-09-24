"""
PointNet with T-Net (input + feature transforms).
Qi et al., 2017. https://arxiv.org/abs/1612.00593
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    """Spatial transformer network for k-dim input."""
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x: (B, k, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]  # (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)
        eye = torch.eye(self.k, device=x.device).flatten().unsqueeze(0)
        x = x + eye
        return x.view(-1, self.k, self.k)


class PointNet(nn.Module):
    def __init__(self, num_classes: int, use_tnet_feat: bool = True, aux_dim: int = 0):
        super().__init__()
        self.use_tnet_feat = use_tnet_feat
        self.aux_dim = aux_dim
        self.tnet3 = TNet(3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        if use_tnet_feat:
            self.tnet64 = TNet(64)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024 + aux_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(0.3)
        self.num_classes = num_classes
        self.embed_dim = 256

    def encode(self, x):
        # x: (B, N, 3)
        x = x.transpose(2, 1)  # (B, 3, N)
        t3 = self.tnet3(x)
        x = torch.bmm(t3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.use_tnet_feat:
            t64 = self.tnet64(x)
            x = torch.bmm(t64, x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        feat = x.max(dim=2)[0]  # (B, 1024) global max pool
        return feat

    def classify(self, feat, aux=None):
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.relu(self.bn6(self.fc1(feat)))
        x = self.drop(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.drop(x)
        return self.fc3(x)

    def embed(self, x, aux=None):
        """Return 256-dim pre-logit embedding (output of fc2, before fc3)."""
        feat = self.encode(x)
        if self.aux_dim > 0 and aux is not None:
            feat = torch.cat([feat, aux], dim=1)
        x = F.relu(self.bn6(self.fc1(feat)))
        x = self.drop(x)
        x = F.relu(self.bn7(self.fc2(x)))
        x = self.drop(x)
        return x  # (B, 256)

    def forward(self, x, aux=None):
        return self.classify(self.encode(x), aux)

    def tnet_regularization(self, x):
        """Orthogonality regularization loss for feature T-Net."""
        if not self.use_tnet_feat:
            return torch.tensor(0.0, device=x.device)
        x = x.transpose(2, 1)
        t3 = self.tnet3(x)
        x = torch.bmm(t3, x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        t64 = self.tnet64(x)
        k = t64.size(1)
        I = torch.eye(k, device=t64.device).unsqueeze(0)
        reg = torch.mean(torch.norm(torch.bmm(t64, t64.transpose(2, 1)) - I, dim=(1, 2)))
        return reg
