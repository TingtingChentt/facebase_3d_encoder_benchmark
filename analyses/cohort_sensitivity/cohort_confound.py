#!/usr/bin/env python3
"""
Cohort confounding in combined affected-vs-unaffected screening (C_full_v3).
Reads finished test-set predictions only (results/predictions/, from score_grid.py); no training.

Why: in the combined task, three source cohorts (FB-TK0, FB-VWP, FB-TX4) contribute
ONLY unaffected subjects and FB-TJ0 is mostly affected, so "which cohort is this scan
from" predicts the label. A model can score well by recognising the cohort.

Outputs (results/cohort_sensitivity/):
  cohort_table.csv          per cohort x status: subjects, scans, age, sex (combined-task cohort, v3 clouds)
  within_cohort_per_run.csv per encoder x train seed x subset: AUC / sens / spec (subject level, softmax;
                            PointNet++ averaged over its 5 inference seeds first)
  within_cohort_summary.csv mean +- SD over the 5 training seeds
  baselines.csv             AUC of cohort-membership-only and age/sex-only predictors on the same test subjects
  unaffected_by_cohort.csv  per-cohort false-positive rate of unaffected subjects (5-seed mean)
Gate: the all-subject AUC per run must reproduce results/cfullv3_trainseed_per_run.csv.
Needs FACEBASE_ROOT (age/sex tables) and the labelled combined manifest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from facebench.evaluation.analyze_ci import macro_auc   # the paper's AUC: mean of both class columns (float32 ties at p=1)  # noqa: E402

from facebench.paths import FACEBASE_ROOT as F, MANIFESTS, PREDICTIONS as PRED, RESULTS  # noqa: E402

OUT = RESULTS / "cohort_sensitivity"
ENC = ["geommlp", "pointnet", "dgcnn", "pointnet2"]
TS = [1, 2, 3, 4, 5]
N_BOOT = 2000
RNG = np.random.default_rng(20261103)


# ---------------------------------------------------------------- cohort + demographics
def manifest():
    # The manifest holds exactly the scans the models were trained and tested on
    # (data/splits/C_full_v3.csv), so no point-cloud existence filter is needed.
    return pd.read_csv(MANIFESTS / "combined_manifest.csv")


def demographics():
    """subject_id -> age_years, sex ('F'/'M'). NaN where the cohort ships no field."""
    rd = dict(low_memory=False, encoding="latin-1")
    sx = {1: "M", 2: "F"}
    rows = []
    o1 = pd.read_csv(F / "FB-5A_files/Data/OFC1_Demographics.csv", **rd)
    rows.append(pd.DataFrame(dict(subject_id="fb5a_" + o1.StudyID.astype(str),
                                  age_years=pd.to_numeric(o1.age, errors="coerce"),
                                  sex=o1.Sex.map(sx))))
    o2 = pd.read_csv(F / "FB-56_files/Data/OFC2_DemoPart1Exported.csv", **rd)
    rows.append(pd.DataFrame(dict(subject_id="fb56_" + o2.StudyID.astype(str),
                                  age_years=pd.to_numeric(o2.Age, errors="coerce"),
                                  sex=o2.Sex.map(sx))))
    t = pd.read_csv(F / "FB-TJ0_files/FB00000861_metadata_2020-09-14.csv", **rd)
    rows.append(pd.DataFrame(dict(subject_id="fbtj0_" + t.FBID.astype(str),
                                  age_years=pd.to_numeric(t["Collection Age (months)"], errors="coerce") / 12,
                                  sex=t.Sex.where(t.Sex.isin(["F", "M"])),
                                  tj0_syndrome=t.Syndrome)))
    v = pd.read_csv(F / "FB-VWP_files/TDFN-Export (2014-06-13).txt", usecols=["ID", "Sex", "Age"])
    corr = pd.read_csv(F / "FB-VWP_files/TDFN-Export-GenderCorrections.csv")
    fix = dict(zip(corr.FaceBaseID.astype(str), corr["New Sex from Genotyping"]))
    v["Sex"] = [fix.get(str(i), s) for i, s in zip(v.ID, v.Sex)]
    rows.append(pd.DataFrame(dict(subject_id="fbvwp_" + v.ID.astype(str),
                                  age_years=pd.to_numeric(v.Age, errors="coerce"),
                                  sex=v.Sex.map(sx))))
    d = pd.concat(rows, ignore_index=True)
    assert not d.subject_id.duplicated().any()
    return d.set_index("subject_id")


def iqr(x):
    x = x.dropna()
    if len(x) == 0:
        return "n/a"
    q = np.percentile(x, [25, 50, 75])
    return f"{q[1]:.1f} [{q[0]:.1f}-{q[2]:.1f}]"


def cohort_table(m, demo, test_subj):
    s = m.groupby("subject_id").agg(dataset=("dataset", "first"), status=("binary_label", "first"),
                                    n_scans=("scan_id", "size"))
    assert (m.groupby("subject_id").binary_label.nunique() == 1).all()
    s = s.join(demo, how="left")
    s["test"] = s.index.isin(test_subj)
    rows = []
    order = ["FB-5A", "FB-56", "FB-TJ0", "FB-TK0", "FB-VWP", "FB-TX4"]
    for ds in order:
        for st in ["Affected", "Unaffected"]:
            g = s[(s.dataset == ds) & (s.status == st)]
            if len(g) == 0:
                rows.append(dict(cohort=ds, status=st, subjects=0, scans=0, test_subjects=0))
                continue
            has_age, has_sex = g.age_years.notna().sum(), g.sex.notna().sum()
            rows.append(dict(
                cohort=ds, status=st, subjects=len(g), scans=int(g.n_scans.sum()),
                test_subjects=int(g.test.sum()),
                age_median_iqr_years=iqr(g.age_years), age_n=int(has_age),
                age_under18_pct=round(100 * (g.age_years < 18).sum() / has_age, 1) if has_age else np.nan,
                female_pct=round(100 * (g.sex == "F").sum() / has_sex, 1) if has_sex else np.nan,
                sex_n=int(has_sex)))
    t = pd.DataFrame(rows)
    # affected share of each cohort (subject level)
    share = s.groupby("dataset").status.apply(lambda x: (x == "Affected").mean())
    t["cohort_affected_pct"] = t.cohort.map((100 * share).round(1))
    return t, s


# ---------------------------------------------------------------- predictions
def load_run(ts, enc, level="subject"):
    seeds = range(5) if enc == "pointnet2" else [0]
    ds = [np.load(PRED / f"C_full_v3_ts{ts}__{enc}__softmax__{level}__seed{k}.npz", allow_pickle=True)
          for k in seeds]
    ids = ds[0]["subject_of_unit"].astype(str)
    for d in ds[1:]:
        assert (d["subject_of_unit"].astype(str) == ids).all()
    assert list(ds[0]["label_names"]) == ["Unaffected", "Affected"]
    return ids, ds[0]["y_true"], ds


SUBSETS = {
    "all": lambda d: np.ones(len(d), bool),
    "FB-TJ0 only": lambda d: d == "FB-TJ0",
    "cleft cohorts (FB-5A + FB-56)": lambda d: np.isin(d, ["FB-5A", "FB-56"]),
    "FB-5A only": lambda d: d == "FB-5A",
    "FB-56 only": lambda d: d == "FB-56",
    "affected cohorts only (no control-only cohorts)": lambda d: np.isin(d, ["FB-5A", "FB-56", "FB-TJ0"]),
}


def metrics(y, score, pred):
    out = dict(n=len(y), n_aff=int(y.sum()))
    out["auc"] = macro_auc(y, None, score, 2) if 0 < y.sum() < len(y) else np.nan
    out["sens"] = (pred[y == 1] == 1).mean() if (y == 1).any() else np.nan
    out["spec"] = (pred[y == 0] == 0).mean() if (y == 0).any() else np.nan
    return out


def per_run(sub2ds):
    rows, gate = [], []
    ref = pd.read_csv(RESULTS / "cfullv3_trainseed_per_run.csv")
    for enc in ENC:
        for ts in TS:
            ids, y, ds = load_run(ts, enc)
            cohort = pd.Series(ids).map(sub2ds).to_numpy()
            assert not pd.isna(cohort).any()
            # metrics per inference seed, then averaged (paper convention)
            per_seed = []
            for d in ds:
                sc, pr = d["y_score"], d["y_pred"]
                per_seed.append({k: metrics(y[f(cohort)], sc[f(cohort)], pr[f(cohort)])
                                 for k, f in SUBSETS.items()})
                gate.append((enc, ts, per_seed[-1]["all"]["auc"]))
            for k in SUBSETS:
                r = dict(encoder=enc, train_seed=ts, subset=k,
                         **{m: np.mean([p[k][m] for p in per_seed]) for m in ("auc", "sens", "spec")},
                         n=per_seed[0][k]["n"], n_aff=per_seed[0][k]["n_aff"])
                rows.append(r)
    # gate against the paper's per-run file (inference-seed mean of macro AUC, subject, softmax)
    g = pd.DataFrame(gate, columns=["encoder", "train_seed", "auc"]).groupby(["encoder", "train_seed"]).auc.mean()
    r = ref[(ref["head"] == "softmax") & (ref.level == "subject") & (ref.metric == "macro_auc")] \
        .set_index(["encoder", "train_seed"]).value
    diff = (g - r.reindex(g.index)).abs().max()
    print(f"GATE all-subject AUC vs cfullv3_trainseed_per_run: max |diff| = {diff:.2e}")
    assert diff < 1e-9, "does not reproduce the paper's AUC"
    return pd.DataFrame(rows)


def boot_ci(y, score, n=N_BOOT):
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = RNG.choice(idx, len(idx))
        if 0 < y[b].sum() < len(b):
            vals.append(macro_auc(y[b], None, score[b], 2))
    return np.percentile(vals, [2.5, 97.5])


def pooled_pn2(sub2ds):
    """PointNet++ score averaged over 5 train x 5 inference seeds, with subject bootstrap CI per subset."""
    sc = []
    for ts in TS:
        ids, y, ds = load_run(ts, "pointnet2")
        sc.append(np.mean([d["y_score"] for d in ds], 0))
        if ts == 1:
            ids0, y0 = ids, y
        assert (ids == ids0).all()
    score = np.mean(sc, 0)
    cohort = pd.Series(ids0).map(sub2ds).to_numpy()
    rows = []
    for k, f in SUBSETS.items():
        m = f(cohort)
        lo, hi = boot_ci(y0[m], score[m])
        rows.append(dict(subset=k, auc_ensemble=macro_auc(y0[m], None, score[m], 2), ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows), ids0, y0, score, cohort


# ---------------------------------------------------------------- baselines (no image)
def baselines(s, test_ids, y_test, cohort_test):
    """Predictors that never see a face, scored on the same test subjects."""
    train = s[~s.test]
    test = s.loc[test_ids]
    assert (test.status.eq("Affected").astype(int).to_numpy() == y_test).all()
    rows = []
    # (1) cohort membership only: score = affected share of the subject's cohort in NON-test subjects
    prev = train.groupby("dataset").status.apply(lambda x: (x == "Affected").mean())
    sc = test.dataset.map(prev).to_numpy()
    for k, f in SUBSETS.items():
        mk = f(cohort_test)
        auc = roc_auc_score(y_test[mk], sc[mk]) if len(np.unique(sc[mk])) > 1 else 0.5
        rows.append(dict(predictor="cohort membership only", subset=k, auc=auc, n=int(mk.sum())))
    # (2) age (+ sex) only: logistic regression fit on non-test subjects of the SAME subset
    for feats, name in [(["age_years"], "age only"), (["age_years", "female"], "age + sex")]:
        for k, f in SUBSETS.items():
            tr = train[f(train.dataset.to_numpy())].assign(female=lambda d: (d.sex == "F").astype(float))
            te = test[f(test.dataset.to_numpy())].assign(female=lambda d: (d.sex == "F").astype(float))
            tr, te = tr.dropna(subset=["age_years", "sex"]), te.dropna(subset=["age_years", "sex"])
            if tr.status.nunique() < 2 or te.status.nunique() < 2:
                continue
            X = lambda d: np.c_[d[feats].to_numpy(), np.log1p(np.clip(d[["age_years"]].to_numpy(), 0, None))]
            clf = LogisticRegression(max_iter=1000).fit(X(tr), tr.status == "Affected")
            auc = roc_auc_score(te.status == "Affected", clf.predict_proba(X(te))[:, 1])
            rows.append(dict(predictor=name, subset=k, auc=auc, n=len(te),
                             note="subjects with age and sex recorded"))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = manifest()
    demo = demographics()
    sub2ds = m.drop_duplicates("subject_id").set_index("subject_id").dataset
    ids, _, _ = load_run(1, "pointnet2")
    table, s = cohort_table(m, demo, set(ids))
    table.to_csv(OUT / "cohort_table.csv", index=False)
    print(table.to_string(index=False))

    pr = per_run(sub2ds)
    pr.to_csv(OUT / "within_cohort_per_run.csv", index=False)
    summ = pr.groupby(["subset", "encoder"]).agg(
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"), sens_mean=("sens", "mean"), sens_sd=("sens", "std"),
        spec_mean=("spec", "mean"), spec_sd=("spec", "std"), n=("n", "first"), n_aff=("n_aff", "first")).reset_index()
    summ.to_csv(OUT / "within_cohort_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    ens, tid, ty, tsc, tco = pooled_pn2(sub2ds)
    ens.to_csv(OUT / "pn2_ensemble_auc_ci.csv", index=False)
    print(ens.round(3).to_string(index=False))

    b = baselines(s, tid, ty, tco)
    b.to_csv(OUT / "baselines.csv", index=False)
    print(b.round(3).to_string(index=False))

    # unaffected false-positive rate and affected sensitivity per cohort, 5-seed mean, every encoder
    rows = []
    for enc in ENC:
        for ts in TS:
            i, y, ds = load_run(ts, enc)
            co = pd.Series(i).map(sub2ds).to_numpy()
            for d in ds:
                for c in np.unique(co):
                    for lab, nm in [(0, "fpr_unaffected"), (1, "sens_affected")]:
                        mk = (co == c) & (y == lab)
                        if mk.any():
                            rows.append(dict(encoder=enc, train_seed=ts, cohort=c, metric=nm,
                                             value=(d["y_pred"][mk] == 1).mean(), n=int(mk.sum())))
    u = pd.DataFrame(rows).groupby(["encoder", "cohort", "metric", "train_seed"]).value.mean() \
        .groupby(["encoder", "cohort", "metric"]).agg(["mean", "std"]).reset_index()
    u.to_csv(OUT / "rates_by_cohort.csv", index=False)

    # FB-TJ0 unaffected subjects of the combined task: relatives vs unrelated
    # controls (the `Syndrome` field), counts only.
    tj = demo.loc[demo.index.isin(m.loc[(m.dataset == "FB-TJ0") & (m.binary_label == "Unaffected"),
                                         "subject_id"].unique()), "tj0_syndrome"]
    tj.value_counts().rename_axis("syndrome_field").reset_index(name="subjects") \
      .to_csv(OUT / "tj0_unaffected_composition.csv", index=False)
    print(u[u.encoder == "pointnet2"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
