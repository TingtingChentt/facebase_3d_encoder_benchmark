#!/usr/bin/env python3
"""
results/paper_macros/numbers_cohort.tex — macros for the Results subsection "Cohort
composition and demographic differences influence screening performance".
Every value is read from results/cohort_sensitivity/*.csv; nothing is typed by hand.
Subject-level, softmax, PointNet++ unless the macro name says otherwise.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from facebench.paths import MACROS, RESULTS  # noqa: E402

R = RESULTS / "cohort_sensitivity"
S = {"all": "all", "TJ": "FB-TJ0 only", "Cleft": "cleft cohorts (FB-5A + FB-56)",
     "FiveA": "FB-5A only", "FiveSix": "FB-56 only", "AffCoh": "affected cohorts only (no control-only cohorts)"}
A = {"TJ": "FB-TJ0", "Cleft": "cleft (FB-5A + FB-56)", "FiveA": "FB-5A", "FiveSix": "FB-56"}
out = []


def mac(name, v, fmt="{:.3f}"):
    out.append(f"\\newcommand{{\\cc{name}}}{{{fmt.format(v) if not isinstance(v, str) else v}}}")


w = pd.read_csv(R / "within_cohort_summary.csv")
w = w[w.encoder == "pointnet2"].set_index("subset")
for k, s in S.items():
    for m in ("auc", "sens", "spec"):
        mac(f"{k}{m.capitalize()}", w.loc[s, f"{m}_mean"]); mac(f"{k}{m.capitalize()}SD", w.loc[s, f"{m}_sd"])
    mac(f"{k}N", int(w.loc[s, "n"]), "{:,}"); mac(f"{k}NAff", int(w.loc[s, "n_aff"]), "{:,}")
    mac(f"{k}NUnaff", int(w.loc[s, "n"] - w.loc[s, "n_aff"]), "{:,}")
# other encoders, within FB-TJ0 / cleft
w2 = pd.read_csv(R / "within_cohort_summary.csv").set_index(["subset", "encoder"])
for k in ("TJ", "Cleft"):
    for e, tag in (("dgcnn", "DG"), ("pointnet", "PN"), ("geommlp", "GM")):
        mac(f"{k}Auc{tag}", w2.loc[(S[k], e), "auc_mean"]); mac(f"{k}Auc{tag}SD", w2.loc[(S[k], e), "auc_sd"])
    mac(f"{k}AucAllGM", w2.loc[("all", "geommlp"), "auc_mean"])

b = pd.read_csv(R / "baselines.csv").set_index(["predictor", "subset"])
mac("CohortOnlyAuc", b.loc[("cohort membership only", "all"), "auc"])
mac("CohortOnlyAffCohAuc", b.loc[("cohort membership only", S["AffCoh"]), "auc"])
for k in ("TJ", "Cleft", "FiveA", "FiveSix", "all"):
    mac(f"{k}AgeOnlyAuc", b.loc[("age only", S[k]), "auc"])
    mac(f"{k}AgeSexAuc", b.loc[("age + sex", S[k]), "auc"])

r = pd.read_csv(R / "rates_by_cohort.csv")
r = r[(r.encoder == "pointnet2")].set_index(["cohort", "metric"])
for c, tag in (("FB-TJ0", "TJ"), ("FB-5A", "FiveA"), ("FB-56", "FiveSix"), ("FB-TK0", "TK"), ("FB-VWP", "VWP"), ("FB-TX4", "TX")):
    mac(f"FPR{tag}", 100 * r.loc[(c, "fpr_unaffected"), "mean"], "{:.1f}")
    mac(f"FPR{tag}SD", 100 * r.loc[(c, "fpr_unaffected"), "std"], "{:.1f}")

e = pd.read_csv(R / "pn2_ensemble_auc_ci.csv").set_index("subset")
for k, s in S.items():
    mac(f"{k}EnsAuc", e.loc[s, "auc_ensemble"])
    mac(f"{k}EnsCI", f"[{e.loc[s, 'ci_lo']:.3f}, {e.loc[s, 'ci_hi']:.3f}]")

aa = pd.read_csv(R / "age_adjusted.csv").set_index(["cohort", "subset"])
ps = pd.read_csv(R / "age_adjusted_per_seed.csv").groupby(["cohort", "subset"]).auc_face.agg(["mean", "std"])
SUB = {"Kid": "age < 18 y", "School": "age 5-17 y", "Match": "age/sex matched (|diff| <= 2 y)", "AllAge": "all ages"}
for k, c in A.items():
    for t, s in SUB.items():
        mac(f"{k}{t}Auc", ps.loc[(c, s), "mean"]); mac(f"{k}{t}AucSD", ps.loc[(c, s), "std"])
        row = aa.loc[(c, s)]
        mac(f"{k}{t}EnsAuc", row.auc_face)
        mac(f"{k}{t}EnsCI", f"[{row.ci_lo:.3f}, {row.ci_hi:.3f}]")
        mac(f"{k}{t}EnsLo", row.ci_lo); mac(f"{k}{t}EnsHi", row.ci_hi)
        mac(f"{k}{t}NAff", int(row.n_aff), "{:,}"); mac(f"{k}{t}NUnaff", int(row.n_unaff), "{:,}")
        mac(f"{k}{t}AgeAff", row.age_med_aff, "{:.1f}"); mac(f"{k}{t}AgeUnaff", row.age_med_unaff, "{:.1f}")
        mac(f"{k}{t}AgeOnlyAuc", row.auc_age_only)
    # CV logistic models: per training run (paper convention), mean +- SD; ensemble value kept as *Ens
    pcv = pd.read_csv(R / "age_adjusted_per_seed.csv").query("subset == 'all ages' and cohort == @c")
    mac(f"{k}CvAgeSex", pcv["cv_auc_age+sex"].mean())
    mac(f"{k}CvAgeSexFace", pcv["cv_auc_age+sex+face"].mean()); mac(f"{k}CvAgeSexFaceSD", pcv["cv_auc_age+sex+face"].std())
    row = aa.loc[(c, "all ages")]
    mac(f"{k}CvAgeSexFaceEns", row["cv_auc_age+sex+face"])

ct = pd.read_csv(R / "cohort_table.csv").set_index(["cohort", "status"])
mac("TJAffPct", ct.loc[("FB-TJ0", "Affected"), "cohort_affected_pct"], "{:.1f}")
mac("FiveAAffPct", ct.loc[("FB-5A", "Affected"), "cohort_affected_pct"], "{:.1f}")
mac("FiveSixAffPct", ct.loc[("FB-56", "Affected"), "cohort_affected_pct"], "{:.1f}")
mac("NControlOnlySubj", int(ct.loc[[("FB-TK0", "Unaffected"), ("FB-VWP", "Unaffected"), ("FB-TX4", "Unaffected")], "subjects"].sum()), "{:,}")
mac("NControlOnlyTest", int(ct.loc[[("FB-TK0", "Unaffected"), ("FB-VWP", "Unaffected"), ("FB-TX4", "Unaffected")], "test_subjects"].sum()), "{:,}")
_tj = pd.read_csv(R / "tj0_unaffected_composition.csv").set_index("syndrome_field").subjects
mac("TJRelatives", int(_tj.get("Relative", 0)), "{:,}")       # Syndrome == 'Relative'
mac("TJUnrelated", int(_tj.get("Unaffected Unrelated", 0)), "{:,}")

hdr = ["% GENERATED by analyses/cohort_sensitivity/write_numbers_cohort.py from results/cohort_sensitivity/*.csv",
       "% Subject level, softmax head; PointNet++ unless noted. Mean +- SD over 5 training seeds",
       "% (each the mean of 5 inference seeds); *Ens* = 25-model mean score with subject bootstrap 95% CI."]
MACROS.mkdir(parents=True, exist_ok=True)
(MACROS / "numbers_cohort.tex").write_text("\n".join(hdr + out) + "\n")
print(len(out), "macros")
