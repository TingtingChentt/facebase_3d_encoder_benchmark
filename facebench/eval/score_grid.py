#!/usr/bin/env python3
"""
TASK 3, stage 2 — score the head x level decision grid from cached features.

Consumes the .npz files written by extract_features.py and emits, for every
(task, encoder, head, level, seed), the per-unit predictions:
    y_true, y_pred, y_prob, subject_id
Point metrics and bootstrap CIs are computed downstream by analyze_ci.py from
these arrays, so the expensive point-cloud trunk is never re-run.

Head semantics deliberately mirror the already-published eval code, so these
numbers are comparable to the ones in EXPERIMENTAL_FINDINGS.md:
  softmax  — evaluate.py / eval_softmax_subject_f1.py
             scan: softmax(logits); subject: MEAN PROBABILITY per subject
  proto    — proto_head.compute_prototypes (class means over TRAIN *scan*
             embeddings, per train.py:194) + cosine distance
             subject: MEAN EMBEDDING per subject, then nearest prototype
             (matches evaluate.evaluate_subject_level)
  knn11    — knn_eval.py: L2-normalised encode() features, cosine similarity
             to the train gallery, majority vote over k=11
             subject: mean of normalised features, renormalised

INFERENCE ONLY — reads cached features, touches no checkpoint.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402


def load(task, encoder, seed, split):
    p = C.feature_path(task, encoder, seed, split)
    if not p.exists():
        raise FileNotFoundError(p)
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def softmax_np(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def group_mean(mat, ids):
    """Mean of `mat` rows within each id, preserving first-appearance order.
    Returns (stacked_means, unique_ids_in_order, index_of_first_occurrence)."""
    order, seen = [], {}
    for i, s in enumerate(ids):
        if s not in seen:
            seen[s] = len(order)
            order.append(s)
    out = np.zeros((len(order), mat.shape[1]), dtype=np.float64)
    cnt = np.zeros(len(order))
    first = np.zeros(len(order), dtype=int)
    for i, s in enumerate(ids):
        j = seen[s]
        out[j] += mat[i]
        if cnt[j] == 0:
            first[j] = i
        cnt[j] += 1
    return out / cnt[:, None], np.array(order, dtype=object), first


def head_scores(head, tr, te, num_classes):
    """Return (scan_pred, scan_score, subj_pred, subj_score, subj_ids,
               subj_labels).

    Predictions and AUC scores are returned SEPARATELY because knn11 derives
    them differently (vote count vs. mean similarity), exactly as knn_eval.py
    does. For softmax/proto the prediction is simply argmax of the score.
    """
    te_ids = te["ids"]

    # tr may be None when the caller only asked for softmax — see main(). Fail
    # with the reason rather than a TypeError three lines down.
    if head in ("proto", "knn11") and tr is None:
        raise ValueError(f"head={head} needs the TRAIN gallery, but none was "
                         f"loaded (softmax-only task?)")

    if head == "softmax":
        scan = softmax_np(te["logits"].astype(np.float64), axis=1)
        subj, sids, first = group_mean(scan, te_ids)      # mean probability
        return (scan.argmax(1), scan, subj.argmax(1), subj,
                sids, te["labels"][first])

    if head == "proto":
        # Class prototypes = mean TRAIN scan embedding per class
        # (proto_head.compute_prototypes, as invoked at train.py:194).
        emb_tr, lab_tr = tr["embed"].astype(np.float64), tr["labels"]
        protos = np.zeros((num_classes, emb_tr.shape[1]))
        for c in range(num_classes):
            m = lab_tr == c
            if m.sum():
                protos[c] = emb_tr[m].mean(0)
        pn = l2norm(protos)

        emb_te = te["embed"].astype(np.float64)
        scan = -(1.0 - l2norm(emb_te) @ pn.T)             # logits = -cos dist
        # Subject level: average the EMBEDDING, then classify (evaluate.py)
        emb_subj, sids, first = group_mean(emb_te, te_ids)
        subj = -(1.0 - l2norm(emb_subj) @ pn.T)
        return (scan.argmax(1), softmax_np(scan, 1),
                subj.argmax(1), softmax_np(subj, 1),
                sids, te["labels"][first])

    if head == "knn11":
        g = l2norm(tr["feat"].astype(np.float64))
        lab_g = tr["labels"]

        def knn(qm):
            sim = qm @ g.T
            k = min(C.KNN_K, sim.shape[1])
            # Full argsort, matching knn_eval.py's np.argsort(-sim)[:, :k].
            topk = np.argsort(-sim, axis=1)[:, :k]
            labs = lab_g[topk]
            # np.bincount(...).argmax() — ties go to the LOWEST class index.
            # Reproduced deliberately: this is the published estimator, and
            # the task is to measure its uncertainty, not to improve it.
            pred = np.array([
                np.bincount(labs[i], minlength=num_classes).argmax()
                for i in range(sim.shape[0])
            ])
            # AUC score: per-class mean similarity over the FULL gallery.
            aucs = np.zeros((sim.shape[0], num_classes))
            for c in range(num_classes):
                m = lab_g == c
                if m.sum():
                    aucs[:, c] = sim[:, m].mean(1)
            return pred, softmax_np(aucs, axis=1)

        q = l2norm(te["feat"].astype(np.float64))
        scan_pred, scan_sc = knn(q)
        q_subj, sids, first = group_mean(q, te_ids)
        subj_pred, subj_sc = knn(l2norm(q_subj))
        return (scan_pred, scan_sc, subj_pred, subj_sc,
                sids, te["labels"][first])

    raise ValueError(head)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    out_dir = Path(a.out) if a.out else (C.OUT / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in a.tasks.split(","):
        cfg = C.TASKS[task]
        nc = cfg["num_classes"]
        heads, levels = C.heads_for(task), C.levels_for(task)
        # The TRAIN split is a GALLERY: proto needs it for class prototypes and
        # knn11 for its neighbour set. softmax reads test logits alone and needs
        # no gallery at all. Loading it lazily lets a softmax-only task skip
        # extracting the train split entirely — which for the cross-site folds
        # is 4,178-6,812 scans per fold x 8 (encoder, seed) combinations that
        # would otherwise be computed and never read. Tasks that DO request
        # proto/knn11 still require it, so nothing already scored changes.
        needs_gallery = bool(set(heads) & {"proto", "knn11"})
        for enc in C.task_encoders(task):
            for seed in C.seeds_for(enc):
                try:
                    tr = load(task, enc, seed, "train") if needs_gallery else None
                    te = load(task, enc, seed, "test")
                except FileNotFoundError as e:
                    print(f"  SKIP {task}/{enc}/seed{seed}: missing {e}")
                    continue
                for head in heads:
                    (sc_pred, sc_score, sj_pred, sj_score,
                     sids, slab) = head_scores(head, tr, te, nc)
                    for level, pred, score, ids_, lab_ in [
                        r for r in [
                            ("scan", sc_pred, sc_score, te["ids"], te["labels"]),
                            ("subject", sj_pred, sj_score, sids, slab),
                        ] if r[0] in levels
                    ]:
                        f = out_dir / f"{task}__{enc}__{head}__{level}__seed{seed}.npz"
                        np.savez_compressed(
                            f,
                            y_true=lab_, y_pred=pred,
                            y_score=score.astype(np.float32),
                            # Bootstrap resamples SUBJECTS. At scan level each
                            # row's subject is its grouping id, so all scans of
                            # a drawn subject travel together.
                            subject_of_unit=np.array(ids_, dtype=object),
                            num_classes=nc,
                            label_names=te["label_names"],
                        )
                    acc = (sc_pred == te["labels"]).mean()
                    sacc = (sj_pred == slab).mean()
                    subj = f"subj_acc={sacc:.4f}" if "subject" in levels else \
                        "subj_acc=n/a (scan-level task)"
                    print(f"  {task:18s} {enc:10s} {head:8s} seed{seed}  "
                          f"scan_acc={acc:.4f}  {subj}", flush=True)
    print(f"[score] predictions -> {out_dir}")


if __name__ == "__main__":
    main()
