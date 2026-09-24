"""
Evaluation utilities: accuracy, per-class F1, AUC (OvR), confusion matrix.
Supports both softmax head (default) and prototypical head.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)


def evaluate(model, loader, device, label_names=None, use_aux=False,
             proto_head=None, prototypes=None):
    """
    Run model over loader, return metrics dict.

    proto_head / prototypes: if provided, use nearest-prototype classification
        via model.embed() instead of model().
    """
    model.eval()
    if proto_head is not None:
        proto_head.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            pts, aux, labels = batch
            pts, aux, labels = pts.to(device), aux.to(device), labels.to(device)

            if proto_head is not None and prototypes is not None:
                emb = model.embed(pts, aux if use_aux else None)
                logits = proto_head(emb, prototypes=prototypes.to(device))
            else:
                logits = model(pts, aux) if use_aux else model(pts)

            probs = torch.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    n_classes = y_prob.shape[1]
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    present_labels = sorted(set(y_true) | set(y_pred))
    filtered_names = [label_names[i] for i in present_labels] if label_names else None
    report = classification_report(y_true, y_pred, labels=present_labels,
                                   target_names=filtered_names, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class.tolist(),
        "auc_macro_ovr": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob.astype(np.float32),
    }


def evaluate_subject_level(model, test_ds, device, label_names,
                            proto_head, prototypes, use_aux=False,
                            batch_size=32):
    """
    Subject-level evaluation with proto head:
    1. Extract scan embeddings for the test set.
    2. Average embeddings per subject (fbid).
    3. Classify averaged embedding by nearest prototype.

    Requires test_ds.df to have a 'fbid' column.
    Returns metrics dict same as evaluate().
    """
    from torch.utils.data import DataLoader

    if "fbid" not in test_ds.df.columns:
        raise ValueError("evaluate_subject_level requires 'fbid' column in dataset")

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    model.eval()
    proto_head.eval()

    all_embs, all_labs = [], []
    with torch.no_grad():
        for pts, aux, labels in loader:
            pts = pts.to(device)
            aux = aux.to(device) if use_aux else None
            emb = model.embed(pts, aux)
            all_embs.append(emb.cpu())
            all_labs.append(labels)

    all_embs = torch.cat(all_embs)   # (N, D)
    all_labs = torch.cat(all_labs)   # (N,)
    fbids = test_ds.df["fbid"].tolist()

    # Aggregate per subject
    subj_embs: dict = {}
    subj_labs: dict = {}
    for i, fid in enumerate(fbids):
        subj_embs.setdefault(fid, []).append(all_embs[i])
        subj_labs[fid] = all_labs[i].item()

    agg_embs = torch.stack([
        torch.stack(v).mean(0) for v in subj_embs.values()
    ]).to(device)                     # (S, D)
    agg_labs = torch.tensor(list(subj_labs.values()))  # (S,)

    with torch.no_grad():
        logits = proto_head(agg_embs, prototypes=prototypes.to(device))
        probs = torch.softmax(logits, dim=-1)

    y_true = agg_labs.numpy()
    y_pred = logits.cpu().argmax(dim=-1).numpy()
    y_prob = probs.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    n_classes = y_prob.shape[1]
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    present_labels = sorted(set(y_true) | set(y_pred))
    filtered_names = [label_names[i] for i in present_labels] if label_names else None
    report = classification_report(y_true, y_pred, labels=present_labels,
                                   target_names=filtered_names, zero_division=0)

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "f1_per_class": f1_per_class.tolist(),
        "auc_macro_ovr": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n_samples": len(y_true),
        "n_subjects": len(subj_embs),
    }


def metrics_summary(metrics: dict) -> str:
    lines = [
        f"  Accuracy:      {metrics['accuracy']:.4f}",
        f"  F1 (macro):    {metrics['f1_macro']:.4f}",
        f"  F1 (weighted): {metrics['f1_weighted']:.4f}",
        f"  AUC (OvR):     {metrics['auc_macro_ovr']:.4f}",
        f"  N samples:     {metrics['n_samples']}",
    ]
    if "n_subjects" in metrics:
        lines.append(f"  N subjects:    {metrics['n_subjects']}")
    return "\n".join(lines)
