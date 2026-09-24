"""
Prototypical Network head for FaceBase 3D encoder.
Replaces the final linear classifier with nearest-prototype classification.

Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ProtoHead(nn.Module):
    """
    Nearest-prototype classifier operating on pre-logit embeddings.

    Training mode  (labels provided):  compute class prototypes from current
        batch as the mean embedding per class, return (full_logits, proto_loss).
    Inference mode (prototypes provided): return distance-based logits.
    """

    def __init__(self, embed_dim: int, n_classes: int, distance: str = "cosine"):
        super().__init__()
        assert distance in ("cosine", "euclidean"), f"Unknown distance: {distance}"
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        self.distance = distance

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    def _dist(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a: (N, D), b: (M, D) → pairwise distances (N, M).
        cosine:    1 - cosine_similarity  ∈ [0, 2]
        euclidean: squared L2 distance
        """
        if self.distance == "cosine":
            a_n = F.normalize(a, dim=1)
            b_n = F.normalize(b, dim=1)
            return 1.0 - a_n @ b_n.t()
        else:
            diff = a.unsqueeze(1) - b.unsqueeze(0)  # (N, M, D)
            return (diff ** 2).sum(-1)               # (N, M)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor = None,
                prototypes: torch.Tensor = None):
        """
        Training (labels provided):
            Computes per-class mean prototypes from the batch.
            Returns (full_logits [N, C], proto_loss).
        Inference (prototypes provided):
            Returns logits [N, C] = negative distances to each prototype.
        """
        if prototypes is not None:
            # ── Inference mode ───────────────────────────────────────
            dists = self._dist(embeddings, prototypes)   # (N, C)
            return -dists                                 # logits

        # ── Training mode ─────────────────────────────────────────────
        device = embeddings.device
        n = len(labels)

        # Identify which classes are present in this batch
        present_idx = labels.unique()                    # (C_p,) sorted

        # Compute prototypes for each present class (differentiable)
        protos_p = torch.stack([
            embeddings[labels == c].mean(0) for c in present_idx
        ])                                               # (C_p, D)

        # Distances from every embedding to each present prototype
        dists = self._dist(embeddings, protos_p)        # (N, C_p)
        logits_p = -dists                                # (N, C_p)

        # Remap original class indices to consecutive local indices
        global_to_local = torch.full((self.n_classes,), -1, dtype=torch.long,
                                     device=device)
        for local_i, c in enumerate(present_idx):
            global_to_local[c] = local_i
        local_labels = global_to_local[labels]           # (N,)

        loss = F.cross_entropy(logits_p, local_labels)

        # Expand to full C-dim logits for metric logging
        full_logits = torch.full((n, self.n_classes), -1e9, device=device)
        for local_i, c in enumerate(present_idx):
            full_logits[:, c] = logits_p[:, local_i]

        return full_logits, loss


# ──────────────────────────────────────────────────────────────────────
# Prototype computation utilities
# ──────────────────────────────────────────────────────────────────────

def compute_prototypes(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n_classes: int,
    use_aux: bool = False,
) -> torch.Tensor:
    """
    One forward pass over dataloader to compute per-class mean embeddings.

    Returns: prototypes tensor (n_classes, embed_dim) on CPU.
    """
    model.eval()
    embed_dim = model.embed_dim
    proto_sums = torch.zeros(n_classes, embed_dim)
    proto_counts = torch.zeros(n_classes)

    with torch.no_grad():
        for pts, aux, labels in dataloader:
            pts = pts.to(device)
            aux = aux.to(device) if use_aux else None
            emb = model.embed(pts, aux)          # (B, D)
            emb_cpu = emb.cpu()
            for c in range(n_classes):
                mask = labels == c
                if mask.any():
                    proto_sums[c] += emb_cpu[mask].sum(0)
                    proto_counts[c] += mask.sum()

    counts_safe = proto_counts.clamp(min=1).unsqueeze(1)
    return proto_sums / counts_safe              # (C, D)


def compute_subject_prototypes(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    n_classes: int,
    fbids: list,
    use_aux: bool = False,
) -> torch.Tensor:
    """
    Compute prototypes using subject-level averaging:
    1. Average all scan embeddings per subject.
    2. Compute per-class mean over subject-level embeddings.

    fbids: list of subject IDs aligned with the dataloader's dataset (no shuffle).
    Returns: prototypes tensor (n_classes, embed_dim) on CPU.
    """
    model.eval()
    embed_dim = model.embed_dim

    all_embs, all_labs = [], []
    with torch.no_grad():
        for pts, aux, labels in dataloader:
            pts = pts.to(device)
            aux = aux.to(device) if use_aux else None
            emb = model.embed(pts, aux)
            all_embs.append(emb.cpu())
            all_labs.append(labels)

    all_embs = torch.cat(all_embs)       # (N, D)
    all_labs = torch.cat(all_labs)       # (N,)

    # Average per subject
    subj_embs: dict = {}
    subj_labs: dict = {}
    for i, fid in enumerate(fbids):
        subj_embs.setdefault(fid, []).append(all_embs[i])
        subj_labs[fid] = all_labs[i].item()

    subj_mean_embs = torch.stack([
        torch.stack(v).mean(0) for v in subj_embs.values()
    ])                                   # (S, D)
    subj_label_list = [subj_labs[k] for k in subj_embs]

    # Per-class mean over subjects
    proto_sums = torch.zeros(n_classes, embed_dim)
    proto_counts = torch.zeros(n_classes)
    for i, lab in enumerate(subj_label_list):
        proto_sums[lab] += subj_mean_embs[i]
        proto_counts[lab] += 1

    counts_safe = proto_counts.clamp(min=1).unsqueeze(1)
    return proto_sums / counts_safe      # (C, D)
