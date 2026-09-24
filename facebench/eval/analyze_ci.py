#!/usr/bin/env python3
"""
TASKS 4 & 5 — uncertainty quantification and paired contrast tests.

TWO UNCERTAINTY SOURCES, DELIBERATELY NEVER POOLED
--------------------------------------------------
  (i)  INFERENCE-SEED variance — spread across inference seeds {0..4} with
       model weights FROZEN. Captures only the PointNet++ FPS draw.
       Reported as seed_sd / seed_min / seed_max.
  (ii) SUBJECT-SAMPLING variance — bootstrap over TEST SUBJECTS, >=2000
       resamples. Reported as boot_ci_lo / boot_ci_hi.
Pooling them into one interval would misrepresent both, so every output column
is tagged with which source it came from.

WHAT THIS PROTOCOL DOES *NOT* MEASURE
-------------------------------------
TRAINING-seed variance. Nothing is retrained here — weights are frozen by
design. So every interval below is "inference variance under frozen weights",
and the manuscript must never call it "run-to-run variance" unqualified. A
model retrained from a different init could land outside these intervals.

BOOTSTRAP UNIT IS THE SUBJECT, NOT THE SCAN. Subjects contribute multiple
correlated scans; resampling scans would treat those as independent and
produce optimistically narrow intervals. At scan level we draw subjects and
carry ALL of their scans along.

PAIRING. One set of bootstrap subject draws is generated per task and reused
across every configuration of that task, so contrasts are genuinely paired:
we take the CI of the DIFFERENCE, not two marginal CIs and an eyeball test.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

PRED_DIR = C.OUT / "predictions"
ALPHA = 0.05
# Equivalence margin for the C1 null claim. 2 pp is the scale below which a
# head-choice difference is not clinically or practically meaningful here; it
# is also well under the ~3 pp inference noise band that motivated this task.
EQUIV_MARGIN = 0.02


# ── metrics (vectorised; no sklearn in the bootstrap inner loop) ────────────

def accuracy(y_true, y_pred, _score, nc):
    return float((y_true == y_pred).mean())


def macro_f1(y_true, y_pred, _score, nc):
    """Macro-F1 over classes PRESENT IN y_true, matching sklearn's
    average='macro' with zero_division=0 on the labels that occur."""
    cm = np.bincount(y_true * nc + y_pred, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(0).astype(float)
    true_pos = cm.sum(1).astype(float)
    denom = pred_pos + true_pos
    f1 = np.divide(2 * tp, denom, out=np.zeros(nc), where=denom > 0)
    present = true_pos > 0
    return float(f1[present].mean()) if present.any() else float("nan")


def macro_auc(y_true, _y_pred, score, nc):
    """Macro one-vs-rest AUC via the Mann-Whitney rank identity, averaged over
    classes that have both positives and negatives present.

    Ranks come from scipy's C-implemented rankdata (ties averaged). A previous
    pure-Python tie loop here cost ~350M interpreter iterations over the full
    bootstrap and dominated runtime; this is numerically identical and orders
    of magnitude faster. Verified against sklearn.roc_auc_score to 1e-16.
    """
    ranks_all = rankdata(score, method="average", axis=0)   # (n, nc)
    n = len(y_true)
    aucs = []
    for c in range(nc):
        pos = y_true == c
        n_pos = int(pos.sum())
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        aucs.append((ranks_all[pos, c].sum() - n_pos * (n_pos + 1) / 2.0)
                    / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else float("nan")


def top_k_accuracy(k):
    """Fraction of units whose true class is among the k highest-scoring.

    Added 2026-08-31 for the 19- and 33-class tasks, where a
    clinician's realistic use of the model is a shortlist, not a single guess,
    and top-1 alone understates a model that reliably brackets the answer.

    Defined only where k < n_classes (rerun_config.topk_for) — top-3 of 4
    classes is near-chance-free and top-3 of 2 is identically 1.0.

    THE SCORE MUST RANK THE SAME WAY THE PREDICTION DOES. For softmax and
    proto, y_pred IS argmax(y_score), so top-1 == accuracy and the ladder is
    coherent. knn11 predicts by an 11-neighbour majority VOTE while its
    y_score is mean gallery similarity: the two disagree, so top-1 there would
    not equal the reported accuracy. verify_topk_consistency() enforces this
    and knn11 is excluded upstream rather than silently reported.
    """
    def _topk(y_true, _y_pred, score, nc):
        if k >= nc:
            return float("nan")
        # argpartition: O(n*nc), no full sort of a (n, 33) score matrix.
        idx = np.argpartition(-score, kth=k - 1, axis=1)[:, :k]
        return float((idx == y_true[:, None]).any(axis=1).mean())
    return _topk


METRICS = {"accuracy": accuracy, "macro_f1": macro_f1, "macro_auc": macro_auc}

# Heads whose y_pred is argmax(y_score) and can therefore carry a top-k
# ladder. See top_k_accuracy.__doc__ for why knn11 is not one of them.
TOPK_HEADS = {"softmax", "proto"}


def metrics_for(task, head):
    """Metric set for one (task, head). Base three everywhere; the top-k
    ladder only where the class count and the head both support it."""
    m = dict(METRICS)
    if head in TOPK_HEADS:
        for k in C.topk_for(task):
            m[f"top{k}_accuracy"] = top_k_accuracy(k)
    return m


def verify_topk_consistency(d, head):
    """Guard the invariant top_k_accuracy relies on: for a top-k head the
    stored prediction must equal argmax of the stored score. Raises rather
    than emitting a top-1 that silently disagrees with `accuracy`."""
    if head not in TOPK_HEADS:
        return
    if not np.array_equal(d["y_score"].argmax(1), d["y_pred"]):
        n = int((d["y_score"].argmax(1) != d["y_pred"]).sum())
        raise AssertionError(
            f"head={head}: y_pred disagrees with argmax(y_score) on {n} units; "
            f"a top-k ladder built on this score would not have accuracy as "
            f"its top-1 rung")


def per_class_f1(y_true, y_pred, nc):
    cm = np.bincount(y_true * nc + y_pred, minlength=nc * nc).reshape(nc, nc)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    return np.divide(2 * tp, denom, out=np.zeros(nc), where=denom > 0)


# ── loading ────────────────────────────────────────────────────────────────

def pred_file(task, enc, head, level, seed):
    return PRED_DIR / f"{task}__{enc}__{head}__{level}__seed{seed}.npz"


def load_pred(task, enc, head, level, seed):
    f = pred_file(task, enc, head, level, seed)
    if not f.exists():
        return None
    d = np.load(f, allow_pickle=True)
    return dict(y_true=d["y_true"], y_pred=d["y_pred"], y_score=d["y_score"],
                subject=d["subject_of_unit"], nc=int(d["num_classes"]),
                label_names=d["label_names"])


def subject_index(subjects):
    """subject -> row indices, in stable first-appearance order."""
    idx = defaultdict(list)
    for i, s in enumerate(subjects):
        idx[s].append(i)
    subs = list(idx.keys())
    return subs, [np.array(idx[s]) for s in subs]


def make_boot_draws(n_subjects, n_boot, seed=12345):
    """Bootstrap subject draws, generated ONCE per task and shared by every
    configuration so contrasts stay paired."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_subjects, size=(n_boot, n_subjects))


def boot_rows(draw, per_subject_rows):
    return np.concatenate([per_subject_rows[s] for s in draw])


# ── main ───────────────────────────────────────────────────────────────────

def collect(tasks, n_boot):
    per_seed, summary, boot_cache = [], [], {}

    for task in tasks:
        cfg = C.TASKS[task]
        draws_by_level = {}

        for enc in C.task_encoders(task):
            for head in C.heads_for(task):
                for level in C.levels_for(task):
                    seeds = C.seeds_for(enc)
                    loaded = {s: load_pred(task, enc, head, level, s)
                              for s in seeds}
                    loaded = {s: v for s, v in loaded.items() if v is not None}
                    if not loaded:
                        continue

                    ref = loaded[min(loaded)]
                    nc = ref["nc"]
                    subs, rows = subject_index(ref["subject"])
                    key = (task, level)
                    if key not in draws_by_level:
                        draws_by_level[key] = (
                            subs, make_boot_draws(len(subs), n_boot))
                    ref_subs, draws = draws_by_level[key]
                    assert ref_subs == subs, (
                        f"subject list mismatch for {task}/{level} — bootstrap "
                        f"pairing would be invalid")

                    # Metric set is per (task, head): the top-k ladder only
                    # attaches to many-class tasks and score-ranked heads.
                    MSET = metrics_for(task, head)

                    # ── (i) inference-seed variance, on the real data ──
                    point = {m: {} for m in MSET}
                    for s, d in loaded.items():
                        verify_topk_consistency(d, head)
                        for m, fn in MSET.items():
                            v = fn(d["y_true"], d["y_pred"], d["y_score"], nc)
                            point[m][s] = v
                            per_seed.append(dict(
                                task=task, encoder=enc, head=head, level=level,
                                seed=s, metric=m, value=v,
                                n_units=len(d["y_true"]), n_subjects=len(subs)))
                        pcf = per_class_f1(d["y_true"], d["y_pred"], nc)
                        for ci, cname in enumerate(d["label_names"]):
                            per_seed.append(dict(
                                task=task, encoder=enc, head=head, level=level,
                                seed=s, metric=f"f1_class::{cname}",
                                value=float(pcf[ci]),
                                n_units=len(d["y_true"]), n_subjects=len(subs)))

                    # ── (ii) subject bootstrap, averaged over seeds ──
                    for m, fn in MSET.items():
                        vals = np.array([point[m][s] for s in sorted(loaded)])
                        bkey = (task, enc, head, level, m)
                        bs = np.empty(len(draws))
                        for b, draw in enumerate(draws):
                            r = boot_rows(draw, rows)
                            # average over seeds within the replicate so the
                            # interval describes the seed-averaged estimate
                            bs[b] = np.mean([
                                fn(loaded[s]["y_true"][r],
                                   loaded[s]["y_pred"][r],
                                   loaded[s]["y_score"][r], nc)
                                for s in sorted(loaded)])
                        boot_cache[bkey] = bs
                        lo, hi = np.percentile(bs, [100 * ALPHA / 2,
                                                    100 * (1 - ALPHA / 2)])
                        summary.append(dict(
                            task=task, encoder=enc, head=head, level=level,
                            metric=m, n_seeds=len(vals),
                            mean=float(vals.mean()),
                            seed_sd=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                            seed_min=float(vals.min()), seed_max=float(vals.max()),
                            seed_deterministic=bool(enc not in C.NONDETERMINISTIC),
                            boot_mean=float(bs.mean()),
                            boot_ci_lo=float(lo), boot_ci_hi=float(hi),
                            n_boot=len(draws),
                            n_subjects=len(subs),
                            n_units=len(ref["y_true"])))
                    print(f"  done {task} {enc} {head} {level}", flush=True)
        # free per-task draws
        draws_by_level.clear()

    return (pd.DataFrame(per_seed), pd.DataFrame(summary), boot_cache)


def contrasts(tasks, n_boot):
    """C1 and C2 as PAIRED differences with bootstrap CIs on the difference."""
    rows = []

    def arm(task, enc, head, level):
        seeds = C.seeds_for(enc)
        d = {s: load_pred(task, enc, head, level, s) for s in seeds}
        d = {k: v for k, v in d.items() if v is not None}
        return d or None

    def paired(name, task, a, b, expect, note):
        """a, b = (task, enc, head, level)."""
        A, B = arm(*a), arm(*b)
        if not A or not B:
            print(f"  SKIP {name} ({a} vs {b}): missing predictions")
            return
        refA, refB = A[min(A)], B[min(B)]
        subsA, rowsA = subject_index(refA["subject"])
        subsB, rowsB = subject_index(refB["subject"])
        if subsA != subsB:
            print(f"  SKIP {name}: subject sets differ between arms")
            return
        draws = make_boot_draws(len(subsA), n_boot)
        nc = refA["nc"]
        # Pair seeds positionally; deterministic encoders have a single seed
        # which is then compared against every seed of the other arm.
        seedsA, seedsB = sorted(A), sorted(B)
        pairs = [(sa, sb) for sa, sb in zip(seedsA, seedsB)] \
            if len(seedsA) == len(seedsB) else \
            [(sa, sb) for sa in seedsA for sb in seedsB]

        for m, fn in METRICS.items():
            # seed-level paired differences on the real data
            dif_seed = np.array([
                fn(A[sa]["y_true"], A[sa]["y_pred"], A[sa]["y_score"], nc) -
                fn(B[sb]["y_true"], B[sb]["y_pred"], B[sb]["y_score"], nc)
                for sa, sb in pairs])
            # bootstrap the paired difference on the SAME subject draws
            bs = np.empty(len(draws))
            for i, draw in enumerate(draws):
                rA = boot_rows(draw, rowsA)
                rB = boot_rows(draw, rowsB)
                bs[i] = np.mean([
                    fn(A[sa]["y_true"][rA], A[sa]["y_pred"][rA],
                       A[sa]["y_score"][rA], nc) -
                    fn(B[sb]["y_true"][rB], B[sb]["y_pred"][rB],
                       B[sb]["y_score"][rB], nc)
                    for sa, sb in pairs])
            lo, hi = np.percentile(bs, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
            mean_d = float(dif_seed.mean())
            spans_zero = bool(lo <= 0.0 <= hi)
            # Whole CI inside +/- EQUIV_MARGIN => the difference is not just
            # "not significant", it is bounded small. A null claim needs this;
            # "CI spans 0" alone is equally consistent with being underpowered.
            within_margin = bool(abs(lo) < EQUIV_MARGIN and abs(hi) < EQUIV_MARGIN)

            if expect == 0:
                # C1 asserts a TIE. Spanning zero SUPPORTS the claim; the
                # question is whether the interval is tight enough to mean it.
                if not spans_zero:
                    verdict = "TIE REFUTED (CI excludes 0 — real difference)"
                elif within_margin:
                    verdict = (f"TIE CONFIRMED, tight "
                               f"(CI within +/-{EQUIV_MARGIN:.0%})")
                else:
                    verdict = ("TIE NOT ESTABLISHED (CI contains 0 but is too "
                               "wide to exclude a material difference)")
            else:
                if spans_zero:
                    verdict = "DISSOLVES (CI spans 0)"
                elif (mean_d > 0) == (expect > 0):
                    verdict = "SURVIVES (CI excludes 0, expected sign)"
                else:
                    verdict = "REVERSES (CI excludes 0, opposite sign)"

            rows.append(dict(
                contrast=name, task=task, metric=m,
                arm_a=f"{a[1]}/{a[2]}/{a[3]}@{a[0]}",
                arm_b=f"{b[1]}/{b[2]}/{b[3]}@{b[0]}",
                diff_mean=mean_d,
                diff_seed_sd=float(dif_seed.std(ddof=1)) if len(dif_seed) > 1 else 0.0,
                boot_diff_mean=float(bs.mean()),
                boot_ci_lo=float(lo), boot_ci_hi=float(hi),
                ci_contains_zero=spans_zero,
                ci_within_equiv_margin=within_margin,
                equiv_margin=EQUIV_MARGIN,
                expected_sign="positive" if expect > 0 else "tie/zero",
                verdict=verdict, n_boot=len(draws),
                n_subjects=len(subsA), note=note))
            print(f"  {name:34s} {m:10s} d={mean_d:+.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {verdict}", flush=True)

    # Which contrasts a task supports is a property of what was SCORED for it,
    # not of a hand-maintained list. A task with one head cannot support C1; a
    # task whose every scan is its own subject cannot support C2. Deriving the
    # gate from heads_for/levels_for means the A-family drops out on its own
    # (single head, single level -> per-arm CIs only) while B2_r2 and
    # C_corrected, which do have real subject grouping, contribute the
    # aggregation contrast without anything being invented for them.
    for task in tasks:
        heads, levels = C.heads_for(task), C.levels_for(task)
        both_levels = "scan" in levels and "subject" in levels
        proto_task = task.replace("_corrected", "_proto_cetrain")

        for enc in C.task_encoders(task):
            # C1a — encoder held FIXED, only the head changes. This is the
            # clean test of "does head choice matter".
            if "softmax" in heads and "proto" in heads and "subject" in levels:
                paired(f"C1a_head_subject[{enc}]", task,
                       (task, enc, "softmax", "subject"),
                       (task, enc, "proto", "subject"),
                       expect=0,
                       note="encoder-fixed: same checkpoint, softmax vs prototype head")

            # C2 — aggregation effect, computed separately for each head.
            if both_levels:
                for head in heads:
                    paired(f"C2_agg_{head}[{enc}]", task,
                           (task, enc, head, "subject"),
                           (task, enc, head, "scan"),
                           expect=+1,
                           note="subject-aggregated minus per-scan, same head")

            # C2c — the ladder the Sec 2.4 prose quotes: proto/subject
            # against the scan-level SOFTMAX baseline, so both numbers in
            # that paragraph share one reference point. It changes the
            # decision rule AND the prediction level at once and is
            # therefore NOT an aggregation effect; it is the combined effect
            # of both. Emitted under its own name so it can never be
            # mistaken for C2. (Added 2026-09-08 in place of a proto/scan
            # table row; the row was then added anyway on 2026-09-09, so
            # Table 2 now carries all four cells and BOTH readings are
            # available -- C2 for aggregation within a rule, C2c for the
            # paragraph's shared-baseline ladder. They are different
            # quantities; do not quote one for the other.)
            if both_levels and "softmax" in heads and "proto" in heads:
                paired(f"C2c_proto_subj_vs_softmax_scan[{enc}]", task,
                       (task, enc, "proto", "subject"),
                       (task, enc, "softmax", "scan"),
                       expect=+1,
                       note="prototype+subject minus softmax+scan — CONFOUNDS "
                            "head with level, not an aggregation effect")

        # C1b — the AS-PUBLISHED comparison: softmax from the CE-trained
        # checkpoint vs proto head from a SEPARATELY TRAINED checkpoint. This
        # confounds head choice with training run; reported alongside C1a so
        # the two are not mistaken for each other.
        # `proto_task` only differs from `task` when the name actually carried
        # "_corrected". For a task that is itself a proto_cetrain arm (or B2_r2)
        # the substitution is a no-op, and firing here would compare the task
        # against ITSELF — a guaranteed zero difference dressed up as a result.
        if proto_task != task and proto_task in C.TASKS:
            paired(f"C1b_head_subject_aspublished", task,
                   (task, "pointnet2", "softmax", "subject"),
                   (proto_task, "pointnet2", "proto", "subject"),
                   expect=0,
                   note="AS-PUBLISHED: different checkpoints — confounds head "
                        "with training run")

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", required=True)
    p.add_argument("--n-boot", type=int, default=C.N_BOOTSTRAP)
    p.add_argument("--suffix", default="")
    p.add_argument("--skip-collect", action="store_true",
                   help="recompute contrasts only, leaving per_seed_results / "
                        "summary_ci untouched. The bootstrap in collect() takes "
                        "~1h over all eleven tasks and does not depend on the "
                        "contrast definitions, so it is not repeated when only "
                        "the contrast set changes.")
    a = p.parse_args()

    tasks = a.tasks.split(",")
    out = C.OUT
    out.mkdir(parents=True, exist_ok=True)

    if a.skip_collect:
        print("[analyze] --skip-collect: reusing existing per_seed_results / "
              "summary_ci")
    else:
        print("[analyze] per-seed metrics + subject bootstrap")
        per_seed, summary, _ = collect(tasks, a.n_boot)
        per_seed.to_csv(out / f"per_seed_results{a.suffix}.csv", index=False)
        summary.to_csv(out / f"summary_ci{a.suffix}.csv", index=False)
        print(f"  wrote per_seed_results{a.suffix}.csv ({len(per_seed)} rows)")
        print(f"  wrote summary_ci{a.suffix}.csv ({len(summary)} rows)")

    print("\n[analyze] paired contrasts C1 / C2")
    con = contrasts(tasks, a.n_boot)
    con.to_csv(out / f"contrast_tests{a.suffix}.csv", index=False)
    print(f"  wrote contrast_tests{a.suffix}.csv ({len(con)} rows)")


if __name__ == "__main__":
    main()
