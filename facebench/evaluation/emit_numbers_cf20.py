#!/usr/bin/env python3
"""
Macros (\\cf*) for the 49-class syndrome-diagnosis task: FB-TJ0 `Syndrome` field,
>=50 scans and >=20 subjects per class (CF20_full_v3, early stopping off, train
seeds 1-5). Covers Table 2 cells, head / aggregation contrasts, per-class
redistribution, recognizability (class size vs F1), error concentration, and the
task-definition counts.

Reads finished results only: results/cf20fullv3_trainseed_{summary,per_run}.csv,
results/contrast_tests_cf20fullv3.csv, results/predictions/CF20_full_v3_*.npz.
Writes results/paper_macros/numbers_cf20.{tex,json} and
results/perclass_head_CF20_full_v3_trainseed.npz (read by Figs. S1, S2). The
analysis functions are IMPORTED from error_concentration.py, recognizability.py
and emit_perclass.py, so every value uses the same estimator throughout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import csv
import json
import os
import statistics as st

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
from facebench.training import task_registry as C                                   # noqa: E402
from facebench.evaluation import error_concentration as EC                           # noqa: E402
from facebench.evaluation import recognizability as RG                               # noqa: E402
from facebench.evaluation import emit_perclass as EP                                 # noqa: E402
from facebench.evaluation.trainseed_pred import run_groups                      # noqa: E402
from facebench.evaluation.derived import DERIVED, cached_json, have_predictions   # noqa: E402

TASK, K = "CF20_full_v3", 49
P2 = "pointnet2"
DEST = PROJECT / "results/paper_macros"
DEST.mkdir(parents=True, exist_ok=True)
PRED = C.OUT / "predictions"

SUMMARY = list(csv.DictReader(open(C.OUT / "cf20fullv3_trainseed_summary.csv")))
CONTRAST = list(csv.DictReader(open(C.OUT / "contrast_tests_cf20fullv3.csv")))
PER = list(csv.DictReader(open(C.OUT / "cf20fullv3_trainseed_per_run.csv")))
WORDS = {0: "none of five", 1: "one of five", 2: "two of five",
         3: "three of five", 4: "four of five", 5: "all five"}
ENC_TAG = {P2: "", "dgcnn": "DG", "pointnet": "Pnet", "geommlp": "GM"}

L, J = [], {}


def emit(name, val, note=""):
    if note:
        L.append(f"% {note}")
    L.append("\\newcommand{\\cf%s}{%s}" % (name, val))
    J[name] = val


def tex(s):
    return f"${s}$" if s.startswith("-") else s


def cell(enc, head, level, metric):
    r = [x for x in SUMMARY if x["base_task"] == TASK and x["encoder"] == enc
         and x["head"] == head and x["level"] == level and x["metric"] == metric]
    assert len(r) == 1, (enc, head, level, metric, len(r))
    return r[0]


def series(enc, head, level, metric):
    rs = sorted([x for x in PER if x["base_task"] == TASK and x["encoder"] == enc
                 and x["head"] == head and x["level"] == level
                 and x["metric"] == metric], key=lambda x: int(float(x["train_seed"])))
    assert len(rs) == 5
    return [float(x["value"]) for x in rs]


# ---- 1. grid cells: every encoder x head x level x metric ------------------
L.append("% ---- cells: mean / SD over 5 training seeds (cf_trainseed_summary.csv)")
for enc, et in ENC_TAG.items():
    for head, ht in [("softmax", "SM"), ("proto", "Pr")]:
        for level, lt in [("scan", "Scan"), ("subject", "Subj")]:
            for metric, mt in [("accuracy", "Acc"), ("macro_f1", "Fone"),
                               ("macro_auc", "Auc"), ("top5_accuracy", "TopFive"),
                               ("top3_accuracy", "TopThree")]:
                r = cell(enc, head, level, metric)
                s = f"{mt}{lt}{ht}{et}"
                m, sd = float(r["train_seed_mean"]), float(r["train_seed_sd"])
                emit(s, f"{m:.3f}")
                emit(s + "SD", f"{sd:.3f}")
                emit(s + "Range", f"{float(r['train_seed_min']):.3f}--"
                                  f"{float(r['train_seed_max']):.3f}")
                emit(s + "Pct", f"{100*m:.1f}\\%")
                emit(s + "CIb", f"[{float(r['boot_ci_lo']):.3f}, "
                                f"{float(r['boot_ci_hi']):.3f}]")
r0 = cell(P2, "softmax", "subject", "accuracy")
emit("NTestSubj", f"{int(r0['n_subjects'])}", "test subjects")
emit("NTestScans", f"{int(cell(P2, 'softmax', 'scan', 'accuracy')['n_units'])}")
emit("Chance", f"{1/K:.3f}")

# ---- 2. paired contrasts (percentage points) -------------------------------
L.append("")
L.append("% ---- contrasts, pp, mean/SD over 5 training seeds (contrast_tests_cf.csv)")
CONTRASTS = [("AggGain", "C2_agg_softmax", "accuracy", 1),
             ("AggGainFone", "C2_agg_softmax", "macro_f1", 1),
             ("AggGainAuc", "C2_agg_softmax", "macro_auc", 1),
             ("AggGainPr", "C2_agg_proto", "accuracy", 1),
             ("AggGainPrFone", "C2_agg_proto", "macro_f1", 1),
             ("AggGainPrAuc", "C2_agg_proto", "macro_auc", 1),
             ("HeadAccDiff", "C1a_head_subject", "accuracy", -1),
             ("HeadFoneDiff", "C1a_head_subject", "macro_f1", -1),
             ("HeadAucDiff", "C1a_head_subject", "macro_auc", -1),
             ("ProtoSubjVsSmScan", "C2c_proto_subj_vs_softmax_scan", "accuracy", 1)]
for enc, et in ENC_TAG.items():
    for stem, pref, metric, sign in CONTRASTS:
        rs = sorted([x for x in CONTRAST if x["contrast"] == f"{pref}[{enc}]"
                     and x["task"].startswith(TASK + "_ts") and x["metric"] == metric],
                    key=lambda x: x["task"])
        assert len(rs) == 5, (stem, enc, len(rs))
        d = [sign * 100 * float(x["diff_mean"]) for x in rs]
        nsig = sum(1 for x in rs if x["ci_contains_zero"] == "False")
        npos = sum(1 for v in d if v > 0)
        s = stem + et
        emit(s, tex(f"{st.mean(d):+.1f}"))
        emit(s + "SD", f"{st.stdev(d):.1f}")
        emit(s + "Range", f"{tex(f'{min(d):+.1f}')} to {tex(f'{max(d):+.1f}')}")
        emit(s + "NSig", WORDS[nsig])
        emit(s + "NPos", WORDS[npos])

# ---- 3. paired encoder differences, matched by training seed --------------
L.append("")
L.append("% ---- encoder pairs (scan-level softmax and subject-level proto), pp")
for stem, ea, eb, head, level, metric in [
        ("AucDgVsPt", "dgcnn", P2, "softmax", "scan", "macro_auc"),
        ("AccPtVsDg", P2, "dgcnn", "softmax", "scan", "accuracy"),
        ("AccPtVsDgSubjPr", P2, "dgcnn", "proto", "subject", "accuracy"),
        ("AucPtVsDgSubjPr", P2, "dgcnn", "proto", "subject", "macro_auc")]:
    a, b = series(ea, head, level, metric), series(eb, head, level, metric)
    d = [100 * (x - y) for x, y in zip(a, b)]
    emit(stem, tex(f"{st.mean(d):+.1f}"))
    emit(stem + "SD", f"{st.stdev(d):.1f}")
    emit(stem + "NPos", WORDS[sum(1 for v in d if v > 0)])

# ---- 4. error concentration (error_concentration.concentration) -----------
L.append("")
L.append("% ---- error concentration, PN++ subject level")
def _errconc(head):
    per_run, ns = [], []
    for _, fs in run_groups(PRED, TASK, head, "subject", "trainseed"):
        vals, n = zip(*[EC.concentration(f) for f in fs])
        per_run.append(float(np.mean([v for v in vals if v is not None])))
        ns.append(int(n[0]))
    return dict(per_run=per_run, ns=ns)


for head, ht in [("softmax", ""), ("proto", "Pr")]:
    ec = cached_json(f"errconc_{TASK}_{head}_subject", f"{TASK}_ts*__pointnet2__{head}__subject__*.npz",
                     lambda: _errconc(head))
    per_run, ns = ec["per_run"], ec["ns"]
    emit("ErrConc" + ht, f"{np.mean(per_run):.2f}",
         f"{head}: {ns[0]} classes with >=2 errors; per-run {min(per_run):.2f}-{max(per_run):.2f}")
    emit("ErrConc" + ht + "Ratio", f"{round(np.mean(per_run) * (K - 1)):.0f}")
    emit("ErrConc" + ht + "NCls", f"{ns[0]}")
emit("ErrConcUnif", f"{1/(K-1):.3f}")

# ---- 5. recognizability (recognizability.load / rho_ci) --------------------
L.append("")
L.append("% ---- class size vs per-class F1, PN++ softmax subject level")
size, f1, names = RG.load(TASK, "subject")
rho, lo, hi = RG.rho_ci(size, f1)
emit("RhoSize", f"{rho:+.2f}")
emit("RhoSizeCI", f"[{lo:+.2f}, {hi:+.2f}]")
# The manuscript's band was 5-12 test subjects on d33; CF20 test classes run
# 3..N, so report the band where most classes sit and say so.
BLO, BHI = 3, 6
band = (size >= BLO) & (size <= BHI)
emit("BandLo", f"{BLO}"); emit("BandHi", f"{BHI}")
emit("BandN", f"{int(band.sum())}")
emit("SpanLo", f"{f1[band].min():.2f}"); emit("SpanHi", f"{f1[band].max():.2f}")
order = np.argsort(-f1)
J["recog_best"] = [(names[i], float(size[i]), float(f1[i])) for i in order[:8]]
J["recog_worst"] = [(names[i], float(size[i]), float(f1[i])) for i in order[-12:]]
J["recog_band"] = sorted([(names[i], float(size[i]), float(f1[i]))
                          for i in np.where(band)[0]], key=lambda x: -x[2])
emit("NZeroFone", f"{int((f1 == 0).sum())}", "classes with mean F1 = 0")
emit("TestMin", f"{int(size.min())}"); emit("TestMax", f"{int(size.max())}")
emit("TestMedian", f"{int(np.median(size))}")

# ---- 6. RASopathy (recognizability.rasopathy) -----------------------------
L.append("")
r = RG.rasopathy(task=TASK)
J["ras_names"] = r["names"]
J["ras_dest"] = r["dest"].most_common(8)
emit("RasWithinFrac", f"{100*r['frac']:.0f}\\%", f"RAS classes: {r['names']}")
emit("RasShare", f"{100*r['share']:.0f}\\%")
emit("RasWithinFracLo", f"{100*r['frac_lo']:.0f}\\%")
emit("RasWithinFracHi", f"{100*r['frac_hi']:.0f}\\%")
emit("RasNErr", f"{r['n_err']}")
emit("RasOutFrac", f"{100*(1-r['frac']):.0f}\\%")
top, topn = r["dest"].most_common(1)[0]
emit("RasTopDest", RG._pretty(top))
emit("RasTopDestFrac", f"{100*topn/max(r['n_err'],1):.0f}\\%")
emit("RasEnrich", f"{r['enrich']:.1f}")

# ---- 7. per-class head redistribution (emit_perclass.collect / counts) -----
L.append("")
PERCLASS = PROJECT / "results" / "perclass_head_CF20_full_v3_trainseed.npz"   # class-level, shipped
if have_predictions(f"{TASK}_ts*__pointnet2__*__subject__*.npz"):
    d = EP.collect(TASK, "trainseed")
    np.savez_compressed(PERCLASS, **d)
d = dict(np.load(PERCLASS, allow_pickle=True))
f_sm, f_pr, a_sm, a_pr, sup = d["f_sm"], d["f_pr"], d["a_sm"], d["a_pr"], d["support"]
diff = f_pr - f_sm
up, dn, fl, dd = EP.counts(diff, f_sm, f_pr)
emit("HeadClassesUp", f"{up}"); emit("HeadClassesDown", f"{dn}")
emit("HeadClassesFlat", f"{fl}"); emit("HeadClassesDead", f"{dd}")
pr = [EP.counts(fr - fs, fs, fr) for fs, fr in zip(d["f_sm_runs"], d["f_pr_runs"])]
for i, nm in enumerate(["Up", "Down", "Flat", "Dead"]):
    emit(f"HeadClasses{nm}Range", f"{min(x[i] for x in pr)}--{max(x[i] for x in pr)}")
large = sup > 2
tot = 100 * diff.mean()
emit("HeadFoneTot", f"{tot:+.1f}")
emit("HeadFoneFromLarge", f"{100*diff[large].sum()/K:+.1f}")
emit("HeadFoneFromSmall", f"{100*diff[~large].sum()/K:+.1f}")
emit("NLarge", f"{int(large.sum())}"); emit("NSmall", f"{int((~large).sum())}")
emit("PerClassFoneMad", f"{np.abs(diff).mean():.3f}")
emit("PerClassAucMad", f"{np.nanmean(np.abs(a_pr - a_sm)):.3f}")
emit("HeadBand", f"{EP.BAND:.2f}")

# ---- 8. cohort / task definition ------------------------------------------
L.append("")
def _task_counts():
    """Task-definition counts; needs the labelled manifest (class sizes)."""
    man = pd.read_csv(C.TASKS[TASK]["manifest"])
    spc = man.groupby("fbid").size()
    out = dict(n_scans=len(man), n_subj=int(man.fbid.nunique()), n_classes=int(man.syndrome_name.nunique()),
               spc_median=int(spc.median()), spc_max=int(spc.max()),
               class_subjects={k: int(v) for k, v in man.groupby("syndrome_name").fbid.nunique()
                               .sort_values(ascending=False).items()},
               test_class_subjects={k: int(v) for k, v in man[man.fixed_split == "test"]
                                    .groupby("syndrome_name").fbid.nunique().items()})
    for sp in ["train", "val", "test"]:
        m = man[man.fixed_split == sp]
        out[f"{sp}_scans"], out[f"{sp}_subj"] = len(m), int(m.fbid.nunique())
    return out


TC_FILE = DERIVED / f"task_counts_{TASK}.json"
if Path(C.TASKS[TASK]["manifest"]).exists():
    DERIVED.mkdir(parents=True, exist_ok=True)
    TC_FILE.write_text(json.dumps(_task_counts(), indent=1))
tc = json.loads(TC_FILE.read_text())
emit("NScans", f"{tc['n_scans']:,}".replace(",", "{,}"))
emit("NSubj", f"{tc['n_subj']:,}".replace(",", "{,}"))
emit("NClasses", f"{tc['n_classes']}")
for sp in ["train", "val", "test"]:
    emit(f"Split{sp.capitalize()}Scans", f"{tc[sp + '_scans']:,}".replace(",", "{,}"))
    emit(f"Split{sp.capitalize()}Subj", f"{tc[sp + '_subj']:,}".replace(",", "{,}"))
emit("ScansPerSubjMedian", f"{tc['spc_median']}")
emit("ScansPerSubjMax", f"{tc['spc_max']}")
cls = tc["class_subjects"]
emit("ClsSubjMin", f"{min(cls.values())}"); emit("ClsSubjMax", f"{max(cls.values())}")
J["class_subjects"] = cls
test_cls = tc["test_class_subjects"]
emit("ClsTestMin", f"{min(test_cls.values())}"); emit("ClsTestMax", f"{max(test_cls.values())}")
hdr = ["% numbers_cf20.tex — GENERATED by facebench/evaluation/emit_numbers_cf20.py",
       "% 49-class syndrome-diagnosis task (CF20_full_v3). Do not edit by hand.", ""]
(DEST / "numbers_cf20.tex").write_text("\n".join(hdr + L) + "\n")
(DEST / "numbers_cf20.json").write_text(json.dumps(J, indent=1, default=str))
print(f"wrote {DEST/'numbers_cf20.tex'} ({sum(1 for x in L if x.startswith(chr(92)))} macros)")
