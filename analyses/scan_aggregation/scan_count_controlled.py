#!/usr/bin/env python3
"""
Controlled patient-level aggregation analysis.

Question: does the subject-level gain on the 49-class
syndrome task come from COMBINING scans, or from the change of evaluation unit
(scan-level metrics weight each patient by their number of scans, subject-level
metrics weight every patient once)?

Design: existing PointNet++ CF20_full_v3 models (5 training runs x 5 inference
seeds), no retraining. Within one fixed set of patients, each patient is
represented by m of their own scans (m = 1..k). Every evaluation is one unit per
patient, so only the number of combined scans changes. For each patient and m
the metric is the EXPECTATION over all m-subsets of their scans (exact
enumeration; random sample of MAX_SUBSETS subsets when there are more).
Aggregation is identical to score_grid.py: softmax = mean probability,
proto = mean embedding -> cosine to train-class prototypes.

Reduction order as elsewhere in the paper: metric per inference seed -> mean
within training run -> mean +- SD across the 5 runs.

Outputs: results/scan_aggregation/*.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import itertools
from math import comb

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[2]
from facebench.training import task_registry as C                                    # noqa: E402
from facebench.evaluation.deidentify import Deidentifier                  # noqa: E402
from facebench.inference.score_grid import l2norm, softmax_np                   # noqa: E402

TASK, ENC, K = "CF20_full_v3", "pointnet2", 49
RUNS, SEEDS = (1, 2, 3, 4, 5), (0, 1, 2, 3, 4)
MMAX = 5
MAX_SUBSETS = 300
OUT = C.OUT / "scan_aggregation"
OUT.mkdir(parents=True, exist_ok=True)
FEAT = C.OUT / "features"
rng = np.random.default_rng(20261105)


def scores_for(head, run, seed):
    te = np.load(FEAT / f"{TASK}_ts{run}__{ENC}__seed{seed}__test.npz", allow_pickle=True)
    ids, y = te["ids"], te["labels"]
    if head == "softmax":
        return ids, y, softmax_np(te["logits"].astype(np.float64), 1), None
    tr = np.load(FEAT / f"{TASK}_ts{run}__{ENC}__seed{seed}__train.npz", allow_pickle=True)
    emb_tr, lab_tr = tr["embed"].astype(np.float64), tr["labels"]
    protos = np.stack([emb_tr[lab_tr == c].mean(0) if (lab_tr == c).any()
                       else np.zeros(emb_tr.shape[1]) for c in range(K)])
    return ids, y, te["embed"].astype(np.float64), l2norm(protos)


def agg_score(head, X, pn, idx):
    v = X[list(idx)].mean(0)
    if head == "softmax":
        return v
    return l2norm(v[None])[0] @ pn.T          # monotone in -cos distance


def rank_of_true(s, t):
    return 1 + int((s > s[t]).sum())


# fixed subset lists per subject (same subsets for every run/seed/head)
ids0 = np.load(FEAT / f"{TASK}_ts1__{ENC}__seed0__test.npz", allow_pickle=True)["ids"]
subj_rows = {}
for i, s in enumerate(ids0):
    subj_rows.setdefault(s, []).append(i)
SUBSETS = {}
for s, rows in subj_rows.items():
    n = len(rows)
    for m in range(1, min(n, MMAX) + 1):
        if comb(n, m) <= MAX_SUBSETS:
            SUBSETS[s, m] = list(itertools.combinations(rows, m))
        else:
            SUBSETS[s, m] = [tuple(sorted(rng.choice(rows, m, replace=False)))
                             for _ in range(MAX_SUBSETS)]
    SUBSETS[s, "all"] = [tuple(rows)]
    # ordered pairs (i -> {i,j}) for the transition analysis
    if n >= 2:
        SUBSETS[s, "pairs"] = [(i, j) for i in rows for j in rows if i != j]

rec, trans, scanlvl = [], [], []
for head in ("softmax", "proto"):
    for run in RUNS:
        for seed in SEEDS:
            ids, y, X, pn = scores_for(head, run, seed)
            assert (ids == ids0).all()
            # conventional scan-level metrics (each scan one unit)
            r = [rank_of_true(agg_score(head, X, pn, (i,)), y[i]) for i in range(len(y))]
            r = np.array(r)
            scanlvl.append(dict(head=head, run=run, seed=seed, top1=(r == 1).mean(),
                                top5=(r <= 5).mean(), mrr=(1 / r).mean(), mean_rank=r.mean()))
            single = {}
            for s, rows in subj_rows.items():
                t = y[rows[0]]
                for m in [mm for mm in range(1, MMAX + 1) if (s, mm) in SUBSETS] + ["all"]:
                    rk = np.array([rank_of_true(agg_score(head, X, pn, ss), t) for ss in SUBSETS[s, m]])
                    rec.append(dict(head=head, run=run, seed=seed, subject=s, n_scans=len(rows),
                                    label=t, m=m, top1=(rk == 1).mean(), top5=(rk <= 5).mean(),
                                    mrr=(1 / rk).mean(), mean_rank=rk.mean()))
                if (s, "pairs") in SUBSETS:
                    r1 = {i: rank_of_true(agg_score(head, X, pn, (i,)), t) for i in rows}
                    pr = [(r1[i], rank_of_true(agg_score(head, X, pn, (i, j)), t))
                          for i, j in SUBSETS[s, "pairs"]]
                    pr = np.array(pr)
                    a, b = pr[:, 0], pr[:, 1]
                    trans.append(dict(head=head, run=run, seed=seed, subject=s, n_scans=len(rows),
                                      label=t,
                                      wrong_to_right=((a > 1) & (b == 1)).mean(),
                                      right_to_wrong=((a == 1) & (b > 1)).mean(),
                                      right_to_right=((a == 1) & (b == 1)).mean(),
                                      wrong_to_wrong=((a > 1) & (b > 1)).mean(),
                                      rank_up=(b < a).mean(), rank_down=(b > a).mean(),
                                      rank_same=(b == a).mean(),
                                      d_rank=(b - a).mean(),
                                      d_logrank=(np.log2(b) - np.log2(a)).mean(),
                                      d_rr=(1 / b - 1 / a).mean()))
            print(head, run, seed, flush=True)

rec = pd.DataFrame(rec); rec["m"] = rec.m.astype(str)
trans = pd.DataFrame(trans); scanlvl = pd.DataFrame(scanlvl)
# Per-patient tables are written without FaceBase IDs or diagnoses (random codes,
# shuffled rows; facebench/evaluation/deidentify.py). Everything downstream uses
# them only to count and group, so all results are unchanged.
DEID = Deidentifier()
DEID(rec).to_csv(OUT / "per_subject_by_m.csv.gz", index=False)
DEID(trans).to_csv(OUT / "per_subject_pair_transitions.csv.gz", index=False)
scanlvl.to_csv(OUT / "scan_level_per_seed.csv", index=False)

MET = ["top1", "top5", "mrr", "mean_rank"]


def runwise(df, keys):
    """per seed -> mean within run -> mean/SD across runs."""
    a = df.groupby(keys + ["run", "seed"])[MET].mean().groupby(keys + ["run"]).mean()
    g = a.groupby(keys)
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    out["runs_all"] = 1
    return out.reset_index(), a.reset_index()


# ---- 0. decomposition on all 385 patients --------------------------------
rows = []
sl = scanlvl.groupby(["head", "run"])[MET].mean()
for h in ("softmax", "proto"):
    rows.append(dict(head=h, setting="scan-level (each scan one unit, 1056 scans)",
                     **{f"{k}_mean": sl.loc[h][k].mean() for k in MET},
                     **{f"{k}_sd": sl.loc[h][k].std() for k in MET}))
for m, lab in (("1", "patient-level, one random scan per patient (385)"),
               ("all", "patient-level, all scans combined (385)")):
    s, _ = runwise(rec[rec.m == m], ["head"])
    for _, r in s.iterrows():
        rows.append(dict(head=r["head"], setting=lab, **{c: r[c] for c in s.columns if c.endswith(("_mean", "_sd"))}))
pd.DataFrame(rows).to_csv(OUT / "A0_decomposition_all385.csv", index=False)

# ---- A/C. same-patient curves --------------------------------------------
curves = []
for kmin in (2, 3, 4, 5):
    sub = rec[rec.n_scans >= kmin]
    sub = sub[sub.m.isin([str(m) for m in range(1, kmin + 1)] + ["all"])]
    s, a = runwise(sub, ["head", "m"])
    s.insert(0, "cohort", f">={kmin} scans")
    s.insert(1, "n_patients", sub.subject.nunique())
    s.insert(2, "n_classes", sub.label.nunique())
    curves.append(s)
pd.concat(curves).to_csv(OUT / "AC_same_patient_curves.csv", index=False)

# ---- paired contrasts with patient bootstrap + Wilcoxon ------------------
B = 2000
con = []
def contrast(head, kmin, m_a, m_b):
    sub = rec[(rec["head"] == head) & (rec.n_scans >= kmin) & rec.m.isin([m_a, m_b])]
    per = sub.groupby(["subject", "m", "run"])[MET].mean()           # mean over seeds
    runs_d = {}
    for k in MET:
        w = per[k].unstack("m")
        runs_d[k] = (w[m_b] - w[m_a]).unstack("run")                 # subject x run
    out = dict(head=head, cohort=f">={kmin} scans", contrast=f"{m_b} vs {m_a}",
               n_patients=runs_d["top1"].shape[0])
    for k in MET:
        D = runs_d[k].to_numpy()
        per_run = D.mean(0)
        pat = D.mean(1)
        idx = rng.integers(0, len(pat), (B, len(pat)))
        bs = pat[idx].mean(1)
        out[f"d_{k}"] = per_run.mean(); out[f"d_{k}_sd"] = per_run.std(ddof=1)
        out[f"d_{k}_ci_lo"], out[f"d_{k}_ci_hi"] = np.percentile(bs, [2.5, 97.5])
        out[f"d_{k}_runs_improved"] = int((per_run > 0).sum() if k != "mean_rank" else (per_run < 0).sum())
        nz = pat[pat != 0]
        out[f"d_{k}_wilcoxon_p"] = wilcoxon(nz).pvalue if len(nz) > 10 else np.nan
    return out
for h in ("softmax", "proto"):
    for kmin, pairs in ((2, [("1", "2"), ("1", "all")]),
                        (3, [("1", "2"), ("2", "3"), ("1", "3")]),
                        (4, [("1", "2"), ("2", "3"), ("3", "4"), ("1", "4")]),
                        (5, [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("1", "5")])):
        for ma, mb in pairs:
            con.append(contrast(h, kmin, ma, mb))
pd.DataFrame(con).to_csv(OUT / "AC_paired_contrasts.csv", index=False)

# ---- B. per-patient consistency (1 -> 2 scans, >=2-scan patients) --------
TC = ["wrong_to_right", "right_to_wrong", "right_to_right", "wrong_to_wrong",
      "rank_up", "rank_down", "rank_same", "d_rank", "d_logrank", "d_rr"]
tw = trans.groupby(["head", "run", "seed"])[TC].mean().groupby(["head", "run"]).mean()
g = tw.groupby("head")
g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).to_csv(OUT / "B_pair_transitions.csv")
# patient-level direction (expected rank change, averaged over runs and seeds)
pp = trans.groupby(["head", "subject", "n_scans", "label"])[["d_rank", "d_logrank", "d_rr"]].mean().reset_index()
DEID(pp).to_csv(OUT / "B_per_patient_delta.csv", index=False)
dirn = []
for h, d in pp.groupby("head"):
    for tol in (0.0, 0.5, 1.0):
        dirn.append(dict(head=h, tolerance_rank=tol, n=len(d),
                         improved=int((d.d_rank < -tol).sum()), worsened=int((d.d_rank > tol).sum()),
                         unchanged=int((d.d_rank.abs() <= tol).sum()),
                         median_d_rank=d.d_rank.median()))
pd.DataFrame(dirn).to_csv(OUT / "B_patient_direction.csv", index=False)
print("done")
