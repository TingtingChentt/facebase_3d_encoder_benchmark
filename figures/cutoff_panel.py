"""
Fixed-test-set panel of the class-size sensitivity figure (Fig. S5c).

The CF75 task's test subjects (12 core diagnoses) are a subset of every other
cutoff's test set, because all seven cutoffs share one subject split. Each
cutoff's models are scored on exactly those subjects, two ways:
  restricted : argmax over the 12 core classes only (the same task for every model)
  open       : argmax over the model's whole label space (distractor classes included)
Reads results/predictions/CF{K}_full_v3_ts{1..5}__<encoder>__proto__subject__seed*.npz.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from facebench.paths import PREDICTIONS as PRED, ROOT as PROJ  # noqa: E402
ENCS = ["pointnet2", "dgcnn", "pointnet", "geommlp"]
TS = [1, 2, 3, 4, 5]
CF_K = [10, 15, 20, 30, 40, 50, 75]


def load_pred(task, enc, head, level="subject"):
    """Returns (subjects sorted, y_true, list of score matrices over inference seeds, labels)."""
    fs = sorted(PRED.glob(f"{task}__{enc}__{head}__{level}__seed*.npz"))
    assert fs, f"no predictions for {task} {enc} {head} {level}"
    scores, ref = [], None
    for f in fs:
        z = np.load(f, allow_pickle=True)
        o = np.argsort(z["subject_of_unit"])
        subj = z["subject_of_unit"][o]
        if ref is None:
            ref = (subj, z["y_true"][o], list(z["label_names"]))
        assert (subj == ref[0]).all() and (z["y_true"][o] == ref[1]).all()
        scores.append(z["y_score"][o])
    return ref[0], ref[1], scores, ref[2]


def cf_fixed_panel(head="proto", suffix=""):
    """Score the CF75 test subjects (12 core classes) with every cutoff's model.
    'restricted': argmax over the 12 core classes only (same task for every model).
    'open': argmax over the model's whole label space (distractor classes included).
    suffix "_v3" scores the CF*_full_v3 runs (v3 clouds; added 2026-10-14)."""
    rows = []
    for enc in ENCS:
        for K in CF_K:
            for ts in TS:
                s75, y75, _, lab75 = load_pred(f"CF75_full{suffix}_ts{ts}", enc, head)
                subj, y, scores, lab = load_pred(f"CF{K}_full{suffix}_ts{ts}", enc, head)
                keep = np.isin(subj, s75)
                assert keep.sum() == len(s75), "CF75 test subjects not nested"
                col = np.array([lab.index(c) for c in lab75])
                yt = np.array([lab75.index(lab[i]) for i in y[keep]])
                assert (subj[keep] == s75).all() and (yt == y75).all()
                r = np.mean([(s[keep][:, col].argmax(1) == yt).mean() for s in scores])
                o = np.mean([(s[keep].argmax(1) == y[keep]).mean() for s in scores])
                rows.append(dict(encoder=enc, K=K, ts=ts, restricted=r, open=o,
                                 n_subj=int(keep.sum()), n_cls_model=len(lab)))
    return pd.DataFrame(rows)
