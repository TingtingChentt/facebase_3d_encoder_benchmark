"""
Shared training loop for all FaceBase encoder models.
AdamW + cosine LR + weighted cross-entropy + early stopping + CSV logging.
Supports: --sampler weighted (WeightedRandomSampler) and --loss focal (Focal Loss).
Supports: --head proto (ProtoHead replaces final linear, proto loss during training).
"""

import csv
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from evaluate import evaluate, metrics_summary


class FocalLoss(nn.Module):
    """Multiclass focal loss: FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)."""
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class alpha (same as CE class weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_pt = F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_w = (1.0 - pt) ** self.gamma
        if self.weight is not None:
            focal_w = focal_w * self.weight[targets]
        return -(focal_w * log_pt).mean()


def train(model, train_dataset, val_dataset, cfg: dict, proto_head=None):
    """
    cfg keys:
      exp, num_classes, batch_size, epochs, lr, weight_decay,
      patience, checkpoint_dir, log_csv, device,
      sampler ('none'|'weighted'), loss ('ce'|'focal'), focal_gamma,
      proto_train_loss ('proto'|'ce')  — only used when proto_head is provided

    proto_head: ProtoHead instance or None. When provided and
        cfg['proto_train_loss'] == 'proto', uses proto loss during training
        and proto-based validation. When 'ce', training is unchanged.
    """
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(device)
    if proto_head is not None:
        proto_head = proto_head.to(device)

    class_weights = train_dataset.class_weights()

    # DataLoader
    # loader_kwargs carries {generator, worker_init_fn} when --train-seed was
    # given, {} otherwise. See seeding.py: without worker_init_fn all four
    # forked workers replay one identical augmentation stream.
    lkw = cfg.get("loader_kwargs", {})
    sampler = cfg.get("sampler", "none")
    if sampler == "weighted":
        label_col = train_dataset.label_col
        label_map = train_dataset.label_map
        sample_labels = torch.tensor(
            [label_map[r] for r in train_dataset.df[label_col]], dtype=torch.long
        )
        sample_weights = class_weights[sample_labels]
        wrs = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset),
                                    replacement=True,
                                    generator=lkw.get("generator"))
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                  sampler=wrs, num_workers=4, pin_memory=True,
                                  drop_last=True, **lkw)
    else:
        train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True, **lkw)

    # Separate loader for prototype computation (no shuffle)
    # NOTE: proto_loader iterates train_dataset, which has augment=True. It is
    # shuffle=False but the augmentation draw still moves, so it takes the
    # worker seeding too or prototype computation stays unseeded.
    proto_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True,
                              worker_init_fn=lkw.get("worker_init_fn"))

    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"] * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    class_weights = class_weights.to(device)

    # Loss function (used for CE-based training and softmax head)
    loss_type = cfg.get("loss", "ce")
    if loss_type == "focal":
        criterion = FocalLoss(gamma=cfg.get("focal_gamma", 2.0), weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # All parameters: model + proto_head (proto_head has no learnable params, but include anyway)
    params = list(model.parameters())
    if proto_head is not None:
        params += list(proto_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01
    )

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg["log_csv"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1
    patience_ctr = 0
    label_names = list(train_dataset.label_map.keys())

    use_proto_loss = (proto_head is not None and
                      cfg.get("proto_train_loss", "proto") == "proto")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "val_f1_macro",
                         "val_auc", "lr", "elapsed_s"])

    aux_dim = getattr(model, "aux_dim", 0)
    head_str = "proto" if use_proto_loss else "softmax"
    print(f"Training {type(model).__name__} on {device} | "
          f"{cfg['epochs']} epochs | classes={cfg['num_classes']} | "
          f"head={head_str} | aux_dim={aux_dim}")

    # Proto import deferred to avoid circular import at module level
    if use_proto_loss:
        from proto_head import compute_prototypes as _compute_prototypes

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        if proto_head is not None:
            proto_head.train()
        total_loss = 0.0

        for pts, aux, labels in train_loader:
            pts, aux, labels = pts.to(device), aux.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_proto_loss:
                emb = model.embed(pts, aux if aux_dim > 0 else None)
                _, loss = proto_head(emb, labels=labels)
            else:
                logits = model(pts, aux) if aux_dim > 0 else model(pts)
                loss = criterion(logits, labels)

            # PointNet T-Net orthogonality regularization (no-op for PN2/DGCNN)
            if hasattr(model, "tnet_regularization"):
                loss = loss + 0.001 * model.tnet_regularization(pts)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # Validation
        if use_proto_loss:
            # Compute training-set prototypes for this epoch's model
            train_protos = _compute_prototypes(
                proto_loader, model, device, cfg["num_classes"],
                use_aux=aux_dim > 0
            )
            val_metrics = evaluate(model, val_loader, device, label_names,
                                   use_aux=aux_dim > 0,
                                   proto_head=proto_head,
                                   prototypes=train_protos)
        else:
            val_metrics = evaluate(model, val_loader, device, label_names,
                                   use_aux=aux_dim > 0)

        elapsed = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch:3d}/{cfg['epochs']} | loss={avg_loss:.4f} | "
              f"val_acc={val_metrics['accuracy']:.4f} | "
              f"val_f1={val_metrics['f1_macro']:.4f} | "
              f"lr={lr:.2e} | {elapsed:.1f}s")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, avg_loss, val_metrics["accuracy"],
                val_metrics["f1_macro"], val_metrics["auc_macro_ovr"],
                lr, round(elapsed, 1)
            ])

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            patience_ctr = 0
            torch.save(model.state_dict(),
                       ckpt_dir / f"{type(model).__name__}_{cfg['exp']}_best.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"Early stopping at epoch {epoch} (patience={cfg['patience']})")
                break

    print(f"Best val F1 (macro): {best_val_f1:.4f}")
    return best_val_f1
