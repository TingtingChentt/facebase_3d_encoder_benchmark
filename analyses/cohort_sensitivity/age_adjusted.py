#!/usr/bin/env python3
"""
Age/sex-adjusted within-cohort screening.
Follow-up to cohort_confound.py: within each affected cohort, age alone separates
affected (mostly children) from unaffected (mostly adults) almost as well as the
face model. Three checks on the combined-screening PointNet++ test predictions
(ensemble score = mean over 5 training x 5 inference seeds, with bootstrap CI; per training run =
metric per inference seed averaged within the run, the paper's convention):

  (1) age band     : AUC among subjects < 18 y, and among 5-17 y
  (2) matched      : each affected subject matched 1:1 (without replacement, greedy,
                     random order, seed 0) to an unaffected subject of the same cohort
                     and sex with |age diff| <= 2 y; AUC on the matched set
  (3) incremental  : 5-fold CV (repeated 20x) logistic regression within the test set,
                     age+sex vs age+sex+face score; AUC of each
Outputs results/cohort_sensitivity/age_adjusted.csv (+ per-seed file).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from analyses.cohort_sensitivity import cohort_confound as CC          # noqa: E402
from facebench.evaluation.analyze_ci import macro_auc      # noqa: E402

OUT = CC.OUT
COHORTS = {"FB-TJ0": ["FB-TJ0"], "cleft (FB-5A + FB-56)": ["FB-5A", "FB-56"],
           "FB-5A": ["FB-5A"], "FB-56": ["FB-56"]}


def auc(y, p1):
    return macro_auc(np.asarray(y).astype(int), None, np.c_[1 - p1, p1], 2) \
        if 0 < np.sum(y) < len(y) else np.nan


def boot(y, p1, n=2000, seed=1):
    r = np.random.default_rng(seed); v = []
    for _ in range(n):
        b = r.integers(0, len(y), len(y))
        if 0 < y[b].sum() < len(b):
            v.append(auc(y[b], p1[b]))
    return np.percentile(v, [2.5, 97.5])


def match(d, tol=2.0, seed=0):
    r = np.random.default_rng(seed)
    aff = d[d.y == 1].sample(frac=1, random_state=seed)
    pool = d[d.y == 0].copy()
    keep = []
    for i, a in aff.iterrows():
        c = pool[(pool.cohort == a.cohort) & (pool.sex == a.sex) & ((pool.age - a.age).abs() <= tol)]
        if len(c):
            j = (c.age - a.age).abs().idxmin()
            keep += [i, j]
            pool = pool.drop(j)
    return d.loc[keep]


def incremental(d, reps=20):
    X0 = np.c_[d.age, np.log1p(d.age.clip(lower=0)), d.sex.eq("F")]
    X1 = np.c_[X0, np.log(d.p / (1 - d.p)).clip(-15, 15)]
    y = d.y.to_numpy()
    out = {"age+sex": [], "age+sex+face": [], "face only": []}
    for rep in range(reps):
        p0, p1 = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(X0, y):
            p0[te] = LogisticRegression(max_iter=2000).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1]
            p1[te] = LogisticRegression(max_iter=2000).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1]
        out["age+sex"].append(auc(y, p0)); out["age+sex+face"].append(auc(y, p1))
    out["face only"] = [auc(y, d.p.to_numpy())]
    return {k: float(np.mean(v)) for k, v in out.items()}


def frame(score):
    m = CC.manifest(); demo = CC.demographics()
    sub2ds = m.drop_duplicates("subject_id").set_index("subject_id").dataset
    ids, y, _ = CC.load_run(1, "pointnet2")
    d = pd.DataFrame(dict(subject_id=ids, y=y, p=score)).set_index("subject_id")
    d["cohort"] = d.index.map(sub2ds)
    d = d.join(demo[["age_years", "sex"]]).rename(columns={"age_years": "age"})
    return d


def scores():
    """PointNet++ P(affected): per (training seed, inference seed) and the 25-model mean."""
    per = {}
    for ts in CC.TS:
        _, _, ds = CC.load_run(ts, "pointnet2")
        # average the two columns' complement so p is the same quantity macro_auc ranks
        per[ts] = [(d["y_score"][:, 1] + 1 - d["y_score"][:, 0]) / 2 for d in ds]
    return per, np.mean([p for v in per.values() for p in v], 0)


def analyse(d, tag):
    rows = []
    for name, cs in COHORTS.items():
        c = d[d.cohort.isin(cs)].dropna(subset=["age", "sex"])
        sets = {"all ages": c, "age < 18 y": c[c.age < 18], "age 5-17 y": c[(c.age >= 5) & (c.age < 18)],
                "age/sex matched (|diff| <= 2 y)": match(c)}
        for sname, s in sets.items():
            y, p = s.y.to_numpy(), s.p.to_numpy()
            row = dict(model=tag, cohort=name, subset=sname, n_aff=int(y.sum()), n_unaff=int((1 - y).sum()),
                       age_med_aff=s[s.y == 1].age.median(), age_med_unaff=s[s.y == 0].age.median(),
                       auc_face=auc(y, p))
            if sname == "all ages":
                row.update({f"cv_auc_{k}": v for k, v in incremental(s).items()})
            if tag == "ensemble":
                row["ci_lo"], row["ci_hi"] = boot(y, p)
                # age-only AUC on the same subset (direction-free: younger = affected)
                row["auc_age_only"] = auc(y, -s.age.to_numpy() + s.age.max() + 1) if len(s) else np.nan
            rows.append(row)
    return rows


def main():
    per, ens = scores()
    rows = analyse(frame(ens), "ensemble")
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "age_adjusted.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    # per training run, paper convention (Methods): metric computed per inference seed, then
    # averaged over the 5 inference seeds within the run. 2026-11-04: previously the 5 inference
    # seeds' scores were averaged first (backup age_adjusted_per_seed.csv.bak_20261104).
    ps = []
    for ts, plist in per.items():
        rs = [pd.DataFrame(analyse(frame(sc), "per_seed")) for sc in plist]
        key = ["model", "cohort", "subset"]
        r = pd.concat(rs).groupby(key, sort=False).mean(numeric_only=True).reset_index()
        ps.append(r.assign(train_seed=ts, n_inference_seeds=len(rs)))
    ps = pd.concat(ps, ignore_index=True)
    ps.to_csv(OUT / "age_adjusted_per_seed.csv", index=False)
    print(ps.groupby(["cohort", "subset"])[["auc_face", "cv_auc_age+sex", "cv_auc_age+sex+face"]]
          .agg(["mean", "std"]).round(3))


if __name__ == "__main__":
    main()
