#!/usr/bin/env python3
"""
Diagnostic prioritization (Fig. 4).

Clinical question: can learned 3D facial representations
prioritise a SHORT LIST of candidate syndromes, rather than assign one label?

Substrate: existing PointNet++ CF20_full_v3 models (5 training runs x 5
inference seeds), 49-class syndrome task, no retraining. Every query is one
held-out test PATIENT (all scans averaged, exactly as the subject-level rows of
Table 2: softmax = mean probability; prototype = mean embedding -> cosine).

  A. Candidate coverage   top-K accuracy K=1..10, macro top-K recall
                          (per-syndrome recall@K, averaged over 49), MRR,
                          for softmax and prototype inference.
  B. Patient retrieval    reference library = training patients, one averaged
                          embedding per patient; cosine; Recall@K (>=1 of K
                          retrieved patients shares the diagnosis), Precision@K
                          (fraction of K that share it), number of distinct
                          diagnoses among the K, and the analytic chance level
                          for random retrieval from the same library.
  C. Limited reference    prototype from n in {1,3,5,10,all} randomly sampled
                          training patients per syndrome (each patient = mean
                          of their scan embeddings; prototype = mean of those),
                          N_DRAWS reference draws; draws are shared across
                          runs/seeds so conditions are paired.

Reduction: metric per inference seed -> mean within run -> mean +- SD over the
5 runs (as everywhere in the paper). Patient-level CIs: bootstrap over test
patients (2,000) of per-patient values averaged over runs and seeds; paired
tests: two-sided Wilcoxon signed-rank on per-patient differences, zeros dropped.

Outputs: results/prioritization/*.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import wilcoxon

PROJECT = Path(__file__).resolve().parents[2]
from facebench.training import task_registry as C                                    # noqa: E402
from facebench.inference.score_grid import l2norm, softmax_np                   # noqa: E402

TASK, ENC, K = "CF20_full_v3", "pointnet2", 49
RUNS, SEEDS = (1, 2, 3, 4, 5), (0, 1, 2, 3, 4)
KS = list(range(1, 11))
RET_KS = (1, 3, 5, 10)
REF_SIZES = (1, 3, 5, 10, "all")
N_DRAWS = 50
B = 2000
OUT = C.OUT / "prioritization"
OUT.mkdir(parents=True, exist_ok=True)
FEAT = C.OUT / "features"
rng = np.random.default_rng(20261109)


def load(run, seed, part):
    z = np.load(FEAT / f"{TASK}_ts{run}__{ENC}__seed{seed}__{part}.npz", allow_pickle=True)
    return z["ids"], z["labels"], z["embed"].astype(np.float64), z["logits"].astype(np.float64)


def by_subject(ids, y):
    subj = list(dict.fromkeys(ids))
    rows = {s: np.where(ids == s)[0] for s in subj}
    lab = np.array([y[rows[s][0]] for s in subj])
    assert all((y[rows[s]] == y[rows[s][0]]).all() for s in subj)
    return subj, rows, lab


def subj_mean(X, subj, rows):
    return np.stack([X[rows[s]].mean(0) for s in subj])


def ranks(S, t):
    """rank of true class = 1 + #classes scoring strictly higher (paper rule)."""
    return 1 + (S > S[np.arange(len(t)), t][:, None]).sum(1)


def cand_metrics(r, t):
    d = {f"top{k}": (r <= k).mean() for k in KS}
    d.update({f"macro_top{k}": np.mean([(r[t == c] <= k).mean() for c in range(K)]) for k in KS})
    d["mrr"] = (1 / r).mean()
    d["macro_mrr"] = np.mean([(1 / r[t == c]).mean() for c in range(K)])
    d["mean_rank"] = r.mean()
    return d


# fixed subject structure (verified identical for every run/seed)
ids_te0, y_te0, *_ = load(1, 0, "test")
ids_tr0, y_tr0, *_ = load(1, 0, "train")
S_te, R_te, Y_te = by_subject(ids_te0, y_te0)
S_tr, R_tr, Y_tr = by_subject(ids_tr0, y_tr0)
NTE, NTR = len(S_te), len(S_tr)
print(f"test patients {NTE}, train patients {NTR}", flush=True)
tr_by_class = [np.where(Y_tr == c)[0] for c in range(K)]

# C. reference draws (indices into train-patient list), shared by all runs/seeds
DRAWS = {n: [[rng.choice(tr_by_class[c], n, replace=False) for c in range(K)]
             for _ in range(N_DRAWS)] for n in REF_SIZES if n != "all"}

recA, perA = [], []          # A: run/seed-level and per-patient ranks
recB, perB = [], []
recC, perC = [], []
for run in RUNS:
    for seed in SEEDS:
        ids_te, y_te, E_te, L_te = load(run, seed, "test")
        ids_tr, y_tr, E_tr, _ = load(run, seed, "train")
        assert (ids_te == ids_te0).all() and (ids_tr == ids_tr0).all()
        Q = l2norm(subj_mean(E_te, S_te, R_te))                  # query patients
        P_sm = subj_mean(softmax_np(L_te, 1), S_te, R_te)
        Ref = subj_mean(E_tr, S_tr, R_tr)                        # raw patient means

        # ---------------- A. candidate coverage ----------------
        proto_scan = l2norm(np.stack([E_tr[y_tr == c].mean(0) for c in range(K)]))  # paper rule
        for head, S in (("softmax", P_sm), ("proto", Q @ proto_scan.T)):
            r = ranks(S, Y_te)
            recA.append(dict(head=head, run=run, seed=seed, **cand_metrics(r, Y_te)))
            perA.append(pd.DataFrame(dict(head=head, run=run, seed=seed, subject=S_te,
                                          label=Y_te, rank=r)))

        # ---------------- B. patient-to-patient retrieval ----------------
        sim = Q @ l2norm(Ref).T                                 # NTE x NTR
        order = np.argsort(-sim, axis=1, kind="stable")[:, :max(RET_KS)]
        same = Y_tr[order] == Y_te[:, None]
        first = np.where(same.any(1), same.argmax(1) + 1, np.inf)   # rank of 1st same-dx patient (<=10)
        row = dict(run=run, seed=seed)
        for k in RET_KS:
            rec_k = same[:, :k].any(1); prec_k = same[:, :k].mean(1)
            ndx = np.array([len(set(Y_tr[o[:k]])) for o in order])
            row.update({f"recall@{k}": rec_k.mean(), f"precision@{k}": prec_k.mean(),
                        f"macro_recall@{k}": np.mean([rec_k[Y_te == c].mean() for c in range(K)]),
                        f"macro_precision@{k}": np.mean([prec_k[Y_te == c].mean() for c in range(K)]),
                        f"n_distinct_dx@{k}": ndx.mean()})
            perB.append(pd.DataFrame(dict(run=run, seed=seed, subject=S_te, label=Y_te, k=k,
                                          recall=rec_k.astype(float), precision=prec_k)))
        recB.append(row)

        # ---------------- C. limited-reference prototypes ----------------
        for n in REF_SIZES:
            draws = [None] if n == "all" else DRAWS[n]
            for di, dr in enumerate(draws):
                if n == "all":
                    pr = np.stack([Ref[tr_by_class[c]].mean(0) for c in range(K)])
                else:
                    pr = np.stack([Ref[dr[c]].mean(0) for c in range(K)])
                r = ranks(Q @ l2norm(pr).T, Y_te)
                m = cand_metrics(r, Y_te)
                recC.append(dict(n_ref=str(n), draw=di, run=run, seed=seed,
                                 **{k: m[k] for k in ("top1", "top3", "top5", "top10", "mrr",
                                                      "macro_top1", "macro_top5", "macro_mrr",
                                                      "mean_rank")}))
                perC.append(pd.DataFrame(dict(n_ref=str(n), draw=di, run=run, seed=seed,
                                              subject=S_te, rank=r)))
        print("run", run, "seed", seed, flush=True)

recA, recB, recC = map(pd.DataFrame, (recA, recB, recC))
perA, perB, perC = pd.concat(perA), pd.concat(perB), pd.concat(perC)
recA.to_csv(OUT / "A_per_seed.csv", index=False)
recB.to_csv(OUT / "B_per_seed.csv", index=False)
recC.to_csv(OUT / "C_per_seed_draw.csv.gz", index=False)
perA.to_csv(OUT / "A_per_patient_rank.csv.gz", index=False)


def runwise(df, keys, cols):
    a = df.groupby(keys + ["run", "seed"])[cols].mean().groupby(keys + ["run"]).mean()
    g = a.groupby(keys)
    return g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).reset_index()


def boot_ci(x):
    idx = rng.integers(0, len(x), (B, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def wil(d):
    nz = d[d != 0]
    return wilcoxon(nz).pvalue if len(nz) > 10 else np.nan


# ---------------- A summary ----------------
colsA = [c for c in recA.columns if c not in ("head", "run", "seed")]
sA = runwise(recA, ["head"], colsA)
sA.to_csv(OUT / "A_summary.csv", index=False)
# chance: top-K = K/49 (uniform random ranking), same for macro
pd.DataFrame(dict(K=KS, chance=[k / K for k in KS])).to_csv(OUT / "A_chance.csv", index=False)
# paired proto - softmax per patient (expected over runs x seeds)
pp = perA.pivot_table(index=["subject", "run", "seed"], columns="head", values="rank").reset_index()
pat = pp.groupby("subject")
con = []
for k in KS + ["mrr", "rank"]:
    if k == "mrr":
        f = lambda r: 1 / r
    elif k == "rank":
        f = lambda r: r
    else:
        f = (lambda kk: (lambda r: (r <= kk).astype(float)))(k)
    d_run = pp.assign(d=f(pp.proto) - f(pp.softmax)).groupby(["run", "seed"]).d.mean().groupby("run").mean()
    d_pat = pp.assign(d=f(pp.proto) - f(pp.softmax)).groupby("subject").d.mean().to_numpy()
    lo, hi = boot_ci(d_pat)
    better = (d_run < 0).sum() if k == "rank" else (d_run > 0).sum()
    con.append(dict(metric=f"top{k}" if isinstance(k, int) else k, d_mean=d_run.mean(),
                    d_sd=d_run.std(), ci_lo=lo, ci_hi=hi, runs_proto_better=int(better),
                    wilcoxon_p=wil(d_pat)))
pd.DataFrame(con).to_csv(OUT / "A_proto_vs_softmax.csv", index=False)
# per-patient CIs on prototype top-K (for figure bands)
ci = []
for head in ("softmax", "proto"):
    sub = perA[perA["head"] == head]
    for k in KS:
        x = sub.assign(h=(sub["rank"] <= k).astype(float)).groupby("subject").h.mean().to_numpy()
        lo, hi = boot_ci(x)
        ci.append(dict(head=head, K=k, ci_lo=lo, ci_hi=hi))
pd.DataFrame(ci).to_csv(OUT / "A_topk_patient_ci.csv", index=False)
# per-class top-5 recall (expected), for the macro story
pc = perA.assign(t5=(perA["rank"] <= 5).astype(float), t1=(perA["rank"] == 1).astype(float))
pc = pc.groupby(["head", "label", "run", "seed"])[["t1", "t5"]].mean().groupby(["head", "label"]).mean()
pc.reset_index().to_csv(OUT / "A_per_class_top1_top5.csv", index=False)

# ---------------- B summary ----------------
colsB = [c for c in recB.columns if c not in ("run", "seed")]
sB = runwise(recB.assign(g="retrieval"), ["g"], colsB).drop(columns="g")
sB.to_csv(OUT / "B_summary.csv", index=False)
# analytic chance for random retrieval of k patients without replacement from the
# training library: precision = n_c/N, recall = 1 - C(N-n_c, k)/C(N, k)
n_c = np.bincount(Y_tr, minlength=K)
ch = []
for k in RET_KS:
    prec = n_c[Y_te] / NTR
    rec = 1 - comb(NTR - n_c[Y_te], k) / comb(NTR, k)
    ch.append(dict(k=k, chance_recall=rec.mean(), chance_precision=prec.mean(),
                   chance_macro_recall=np.mean([rec[Y_te == c].mean() for c in range(K)]),
                   chance_macro_precision=np.mean([prec[Y_te == c].mean() for c in range(K)])))
pd.DataFrame(ch).to_csv(OUT / "B_chance.csv", index=False)
ciB = []
for k in RET_KS:
    sub = perB[perB.k == k].groupby("subject")[["recall", "precision"]].mean()
    for m in ("recall", "precision"):
        lo, hi = boot_ci(sub[m].to_numpy())
        ciB.append(dict(k=k, metric=m, ci_lo=lo, ci_hi=hi))
pd.DataFrame(ciB).to_csv(OUT / "B_patient_ci.csv", index=False)
# patient retrieval Recall@K vs prototype top-K syndrome accuracy, same patients
pr_pat = perA[perA["head"] == "proto"]
cmp = []
for k in RET_KS:
    a = pr_pat.assign(h=(pr_pat["rank"] <= k).astype(float)).groupby("subject").h.mean()
    b = perB[perB.k == k].groupby("subject").recall.mean()
    d = (b - a.loc[b.index]).to_numpy()
    lo, hi = boot_ci(d)
    cmp.append(dict(k=k, patient_recall=b.mean(), proto_topk=a.mean(), d=d.mean(),
                    ci_lo=lo, ci_hi=hi, wilcoxon_p=wil(d)))
pd.DataFrame(cmp).to_csv(OUT / "B_vs_proto_topk.csv", index=False)
perB.to_csv(OUT / "B_per_patient.csv.gz", index=False)

# ---------------- C summary ----------------
colsC = ["top1", "top3", "top5", "top10", "mrr", "macro_top1", "macro_top5", "macro_mrr", "mean_rank"]
# (i) expected over draws: per draw -> runwise; then mean over draws of run means
a = recC.groupby(["n_ref", "draw", "run", "seed"])[colsC].mean().groupby(["n_ref", "draw", "run"]).mean()
exp_run = a.groupby(["n_ref", "run"]).mean()                          # expectation over draws
g = exp_run.groupby("n_ref")
sC = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd_runs"))
# (ii) draw-to-draw variability: average over runs per draw, then spread over draws
per_draw = a.groupby(["n_ref", "draw"]).mean()
gd = per_draw.groupby("n_ref")
sC = sC.join(gd.std().add_suffix("_sd_draws")).join(gd.quantile(0.025).add_suffix("_q025")) \
       .join(gd.quantile(0.975).add_suffix("_q975"))
sC = sC.reset_index()
sC["order"] = sC.n_ref.map({"1": 1, "3": 3, "5": 5, "10": 10, "all": 999})
sC.sort_values("order").drop(columns="order").to_csv(OUT / "C_summary.csv", index=False)
# paired contrasts: each n vs all (per patient, expectation over draws, runs, seeds)
perC["top1"] = (perC["rank"] == 1).astype(float); perC["top5"] = (perC["rank"] <= 5).astype(float)
perC["rr"] = 1 / perC["rank"]
pm = perC.groupby(["n_ref", "subject"])[["top1", "top5", "rr", "rank"]].mean()
conC = []
for n in [str(x) for x in REF_SIZES if x != "all"]:
    for m in ("top1", "top5", "rr", "rank"):
        d = (pm.loc[n][m] - pm.loc["all"][m]).to_numpy()
        lo, hi = boot_ci(d)
        conC.append(dict(n_ref=n, metric=m, d=d.mean(), ci_lo=lo, ci_hi=hi, wilcoxon_p=wil(d)))
    # also each step vs the previous size
conC = pd.DataFrame(conC)
conC.to_csv(OUT / "C_vs_all.csv", index=False)
# per-class: what happens to the 1-shot prototype by syndrome (top-5 recall)
pcC = perC.merge(pd.DataFrame(dict(subject=S_te, label=Y_te)), on="subject")
pcC.groupby(["n_ref", "label"])[["top1", "top5"]].mean().reset_index() \
   .to_csv(OUT / "C_per_class.csv", index=False)
pd.DataFrame(dict(label=range(K), n_train_patients=n_c,
                  n_test_patients=np.bincount(Y_te, minlength=K))).to_csv(OUT / "class_sizes.csv", index=False)
print("done")
