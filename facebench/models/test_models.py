"""Unit tests: forward pass shape check for all models."""

import torch
import pytest

from pointnet import PointNet
from pointnet2 import PointNet2
from dgcnn import DGCNN
from geom_mlp import GeomMLP

B, N, C = 4, 4096, 3
CONFIGS = [
    ("ofc", 4),
    ("syndrome", 15),
]


def make_batch(b=B, n=N):
    return torch.randn(b, n, 3)


@pytest.mark.parametrize("exp,num_classes", CONFIGS)
def test_pointnet_forward(exp, num_classes):
    model = PointNet(num_classes=num_classes)
    x = make_batch()
    logits = model(x)
    assert logits.shape == (B, num_classes), f"Expected ({B},{num_classes}), got {logits.shape}"
    feat = model.encode(x)
    assert feat.shape == (B, 1024)


@pytest.mark.parametrize("exp,num_classes", CONFIGS)
def test_pointnet2_forward(exp, num_classes):
    model = PointNet2(num_classes=num_classes)
    x = make_batch()
    logits = model(x)
    assert logits.shape == (B, num_classes)
    feat = model.encode(x)
    assert feat.shape == (B, 1024)


@pytest.mark.parametrize("exp,num_classes", CONFIGS)
def test_dgcnn_forward(exp, num_classes):
    model = DGCNN(num_classes=num_classes)
    x = make_batch()
    logits = model(x)
    assert logits.shape == (B, num_classes)
    feat = model.encode(x)
    assert feat.shape == (B, 2048)


@pytest.mark.parametrize("exp,num_classes", CONFIGS)
def test_geom_mlp_forward(exp, num_classes):
    model = GeomMLP(num_classes=num_classes)
    x = make_batch()
    logits = model(x)
    assert logits.shape == (B, num_classes)
    feat = model.encode(x)
    assert feat.shape[0] == B


def test_all_models_consistent_interface():
    """All models expose encode() and classify()."""
    num_classes = 4
    x = make_batch()
    for ModelClass in [PointNet, PointNet2, DGCNN, GeomMLP]:
        m = ModelClass(num_classes=num_classes)
        feat = m.encode(x)
        logits = m.classify(feat)
        assert logits.shape == (B, num_classes), f"{ModelClass.__name__} classify shape mismatch"


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    print("Running forward pass tests...")
    for exp, nc in CONFIGS:
        for name, cls in [("PointNet", PointNet), ("PointNet2", PointNet2),
                           ("DGCNN", DGCNN), ("GeomMLP", GeomMLP)]:
            m = cls(num_classes=nc)
            x = make_batch()
            out = m(x)
            n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f"  {name:12s} [{exp:8s}] output={tuple(out.shape)}  "
                  f"params={n_params/1e6:.2f}M")
    print("All forward passes OK")
