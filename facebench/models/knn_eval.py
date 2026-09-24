"""
kNN retrieval evaluation on PointNet2 shape embeddings.
Extracts embeddings via model.encode(pts), L2-normalizes, then predicts
test labels by majority-vote kNN over the training gallery.
Scan-level and subject-level (average embeddings per subject).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from dataset import FaceBaseDataset
from pointnet2 import PointNet2

KS = [1, 3, 5, 7, 11]


def extract_embeddings(model, loader, device):
    model.eval()
    embs, labs = [], []
    with torch.no_grad():
        for pts, aux, labels in loader:
            e = model.encode(pts.to(device))
            embs.append(e.cpu())
            labs.append(labels)
    return torch.cat(embs), torch.cat(labs)


def l2_norm(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-8)


def eval_knn(sim, gallery_lab, query_lab, num_classes):
    """
    sim: (Q, G) cosine similarity matrix (already computed once).
    Returns list of row dicts, one per k.
    AUC uses per-class mean similarity to ALL gallery samples.
    """
    q_lab_np = query_lab.numpy()
    g_lab_np = gallery_lab.numpy()
    n_q = sim.shape[0]

    # AUC scores: mean similarity to each class over ALL gallery
    # Only use classes present in both gallery and query to avoid NaN
    auc_scores = np.zeros((n_q, num_classes))
    for c in range(num_classes):
        mask = g_lab_np == c
        if mask.sum() > 0:
            auc_scores[:, c] = sim[:, mask].mean(axis=1)
    present = sorted(set(q_lab_np.tolist()) & set(g_lab_np.tolist()))
    try:
        if len(present) < 2:
            auc = float('nan')
        elif len(present) == 2:
            pos_col = auc_scores[:, present[1]]
            auc = roc_auc_score((q_lab_np == present[1]).astype(int), pos_col)
        else:
            # Softmax similarity scores → probabilities summing to 1 (required by sklearn OvR)
            prob = softmax(auc_scores[:, present], axis=1)
            auc = roc_auc_score(q_lab_np, prob,
                                multi_class='ovr', average='macro',
                                labels=present)
    except Exception as e:
        print(f"  [AUC warning] {e}")
        auc = float('nan')

    rows = []
    for k in KS:
        topk_idx = np.argsort(-sim, axis=1)[:, :k]
        topk_labs = g_lab_np[topk_idx]
        preds = np.array([np.bincount(topk_labs[i], minlength=num_classes).argmax()
                          for i in range(n_q)])
        acc = accuracy_score(q_lab_np, preds)
        f1m = f1_score(q_lab_np, preds, average='macro', zero_division=0)
        f1w = f1_score(q_lab_np, preds, average='weighted', zero_division=0)
        rows.append({'k': k, 'acc': acc, 'f1_macro': f1m,
                     'f1_weighted': f1w, 'auc': auc, 'n': n_q})
    return rows


def subject_aggregate(embs, labs, fbids):
    subj = {}
    for i, fid in enumerate(fbids):
        if fid not in subj:
            subj[fid] = {'embs': [], 'labs': []}
        subj[fid]['embs'].append(embs[i])
        subj[fid]['labs'].append(labs[i].item())
    out_e, out_l = [], []
    for v in subj.values():
        out_e.append(torch.stack(v['embs']).mean(0))
        out_l.append(max(set(v['labs']), key=v['labs'].count))
    return l2_norm(torch.stack(out_e)), torch.tensor(out_l)


def run_experiment(exp_id, checkpoint, manifest, exp_type, num_classes,
                   aux_dim, results_dir, device):
    print(f"\n{'='*60}")
    print(f"{exp_id}  classes={num_classes}  aux_dim={aux_dim}")

    model = PointNet2(num_classes=num_classes, aux_dim=aux_dim)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
    model = model.to(device).eval()

    ds_kw = dict(manifest_csv=manifest, exp=exp_type, augment=False)
    train_ds = FaceBaseDataset(**ds_kw, split='train')
    test_ds  = FaceBaseDataset(**ds_kw, split='test')
    tr_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=4,
                           pin_memory=device.type == 'cuda')
    te_loader = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4,
                           pin_memory=device.type == 'cuda')

    print(f"Extracting embeddings (train={len(train_ds)}, test={len(test_ds)})...")
    tr_emb, tr_lab = extract_embeddings(model, tr_loader, device)
    te_emb, te_lab = extract_embeddings(model, te_loader, device)
    tr_emb_n = l2_norm(tr_emb)
    te_emb_n = l2_norm(te_emb)

    tr_fbids = train_ds.df['fbid'].tolist()
    te_fbids = test_ds.df['fbid'].tolist()

    # Similarity matrix (computed ONCE, reused for all k and AUC)
    sim_scan = (te_emb_n @ tr_emb_n.T).numpy()

    header = ['exp_id', 'k', 'level', 'accuracy', 'f1_macro',
              'f1_weighted', 'auc_macro_ovr', 'n_test']
    print(f"\n{'k':>4}  {'level':>8}  {'acc':>6}  {'f1_mac':>7}  {'auc':>6}")
    print("-" * 45)

    all_rows = []
    for level, sim_m, q_lab in [
        ('scan',    sim_scan, te_lab),
        ('subject', None,     None),
    ]:
        if level == 'subject':
            tr_emb_s, tr_lab_s = subject_aggregate(tr_emb_n, tr_lab, tr_fbids)
            te_emb_s, te_lab_s = subject_aggregate(te_emb_n, te_lab, te_fbids)
            sim_m = (te_emb_s @ tr_emb_s.T).numpy()
            q_lab = te_lab_s
            g_lab = tr_lab_s
        else:
            g_lab = tr_lab

        for r in eval_knn(sim_m, g_lab, q_lab, num_classes):
            print(f"{r['k']:>4}  {level:>8}  {r['acc']:.4f}  {r['f1_macro']:.4f}  "
                  f"  {r['auc']:.4f}")
            all_rows.append([exp_id, r['k'], level,
                             round(r['acc'], 4), round(r['f1_macro'], 4),
                             round(r['f1_weighted'], 4), round(r['auc'], 4),
                             r['n']])

    out = results_dir / f"knn_results_{exp_id}.csv"
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(all_rows)
    print(f"Saved {out}")
    return all_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--manifest',    required=True)
    p.add_argument('--exp-type',    required=True)
    p.add_argument('--num-classes', type=int, required=True)
    p.add_argument('--exp-id',      required=True)
    p.add_argument('--aux-dim',     type=int, default=0)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    results_dir = PROJECT / 'outputs' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = run_experiment(
        exp_id=args.exp_id,
        checkpoint=str(PROJECT / args.checkpoint),
        manifest=str(PROJECT / args.manifest),
        exp_type=args.exp_type,
        num_classes=args.num_classes,
        aux_dim=args.aux_dim,
        results_dir=results_dir,
        device=device,
    )

    combined = results_dir / 'knn_results.csv'
    header = ['exp_id', 'k', 'level', 'accuracy', 'f1_macro',
              'f1_weighted', 'auc_macro_ovr', 'n_test']
    write_header = not combined.exists()
    with open(combined, 'a', newline='') as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)
    print(f"Appended to {combined}")


if __name__ == '__main__':
    main()
