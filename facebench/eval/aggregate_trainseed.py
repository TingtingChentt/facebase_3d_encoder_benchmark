#!/usr/bin/env python3
"""
Collapse the training-seed sweep into the numbers the manuscript reports.

thread: facebase3d-trainseed-2026-08-31

The gap this closes (2026-08-31): "the confidence intervals account for test-subject sampling,
not independent retraining/model-initialization variance."  Correct. This
script produces the missing quantity.

THREE VARIANCE SOURCES. THEY ARE NEVER POOLED.
----------------------------------------------
  (T) TRAINING-seed SD   — 5 independent retrainings, split frozen at 42 so
      every one is scored on IDENTICAL test subjects. Answers "would this
      result survive training the model again". THIS IS THE NEW HEADLINE
      DISPERSION and the one the paper previously had no measurement of.
  (I) INFERENCE-seed SD  — PointNet++ FPS start index, weights frozen. Already
      averaged down WITHIN each training seed before (T) is taken, and also
      reported on its own so the two are separable.
  (S) SUBJECT bootstrap  — 95% CI over resampled test subjects. Answers "would
      this result survive a different sample of patients". Orthogonal to (T):
      a wide CI with a tight seed SD means the cohort is small, not that
      training is unstable.

Quoting (S) as though it covered (T) is exactly the conflation this sweep
exists to remove. Any one of the three may be the binding constraint, so all
three are carried into the output columns and the emitter prints (T) and (S)
side by side.

THE LEGACY COLUMN. Every pre-2026-08-31 checkpoint was trained with an
unseeded torch RNG, so it is one unreproducible draw. legacy_z reports where
it sits in the retrained distribution, in SD units. It is a diagnostic, NOT a
result: a large |z| means the published number was a lucky or unlucky draw,
which is the finding, not something to correct away.

Usage:
  python3 aggregate_trainseed.py                 # after analyze_ci --suffix _trainseed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

OUT = C.OUT
MIN_SEEDS_TO_REPORT = 3   # below this an SD is not worth printing
# Below this an SD is float dust, not dispersion, and dividing by it makes
# legacy_z meaningless. See the note at step 4.
SD_FLOOR = 1e-9


def load_trainseed():
    f = OUT / "per_seed_results_trainseed.csv"
    if not f.exists():
        sys.exit(f"missing {f} — run analyze_ci.py --suffix _trainseed first")
    df = pd.read_csv(f)
    # `seed` in this file is the INFERENCE seed; the TRAINING seed is encoded
    # in the task name. Split them apart before anything aggregates.
    df["base_task"] = df["task"].map(lambda t: C.TASKS[t].get("base_task", t))
    df["train_seed"] = df["task"].map(lambda t: C.TASKS[t].get("train_seed"))
    df = df.rename(columns={"seed": "infer_seed"})
    missing = df["train_seed"].isna()
    if missing.any():
        bad = sorted(df.loc[missing, "task"].unique())
        sys.exit(f"tasks with no train_seed in rerun_config: {bad}")
    return df


def load_legacy():
    """Point estimates from the frozen single-run checkpoints, for the
    diagnostic legacy_z column only."""
    f = OUT / "summary_ci.csv"
    if not f.exists():
        print(f"  note: {f} absent — legacy comparison columns will be blank")
        return None
    lg = pd.read_csv(f)
    return lg.rename(columns={"task": "base_task", "mean": "legacy_value",
                              "boot_ci_lo": "legacy_ci_lo",
                              "boot_ci_hi": "legacy_ci_hi"})[
        ["base_task", "encoder", "head", "level", "metric",
         "legacy_value", "legacy_ci_lo", "legacy_ci_hi"]]


def add_legacy_aliases(lg, wanted):
    """Carry a base task's legacy row over to a re-run of the same task.

    A2_fix is task A2 retrained with early stopping disabled, so its published
    comparison point is A2's frozen single checkpoint — the same cohort, split
    and metric. Without this the legacy columns are blank for every A2_fix row
    and the manuscript cannot say where the published value sits. Declared in
    rerun_config as legacy_base so the mapping lives with the task definition,
    not here.

    `wanted` is the set of base tasks actually present in this aggregation.
    Other families declare legacy_base too — the cross-site retrains, whose
    published rows live in summary_ci_xsite.csv and are scored by their own
    pipeline — and they are not this script's business.
    """
    extra = []
    for name, cfg in C.TASKS.items():
        src = cfg.get("legacy_base")
        if src is None or name not in wanted:
            continue
        rows = lg[lg["base_task"] == src].copy()
        if not len(rows):
            print(f"  note: {name} declares legacy_base={src} but "
                  f"summary_ci.csv has no {src} rows")
            continue
        rows["base_task"] = name
        extra.append(rows)
    return pd.concat([lg] + extra, ignore_index=True) if extra else lg


def main():
    df = load_trainseed()
    keys = ["base_task", "encoder", "head", "level", "metric"]

    # Step 1 — collapse the inference seeds WITHIN each training seed, so each
    # retrained model contributes exactly one point estimate and PointNet++'s
    # FPS noise does not inflate the training-seed SD.
    per_train = (df.groupby(keys + ["train_seed"], as_index=False)
                   .agg(value=("value", "mean"),
                        infer_seed_sd=("value", lambda v: float(np.std(v, ddof=1))
                                       if len(v) > 1 else 0.0),
                        n_infer_seeds=("value", "size"),
                        n_units=("n_units", "first"),
                        n_subjects=("n_subjects", "first")))

    # Step 2 — dispersion ACROSS training seeds. This is the new quantity.
    agg = (per_train.groupby(keys, as_index=False)
                    .agg(train_seed_mean=("value", "mean"),
                         train_seed_sd=("value", lambda v: float(np.std(v, ddof=1))
                                        if len(v) > 1 else np.nan),
                         train_seed_min=("value", "min"),
                         train_seed_max=("value", "max"),
                         n_train_seeds=("value", "size"),
                         infer_seed_sd=("infer_seed_sd", "mean"),
                         n_units=("n_units", "first"),
                         n_subjects=("n_subjects", "first")))
    agg["train_seed_range"] = agg["train_seed_max"] - agg["train_seed_min"]

    # Step 3 — carry the subject bootstrap across, averaged over training
    # seeds. Deliberately a SEPARATE pair of columns from the seed SD.
    sf = OUT / "summary_ci_trainseed.csv"
    if sf.exists():
        sm = pd.read_csv(sf)
        sm["base_task"] = sm["task"].map(
            lambda t: C.TASKS[t].get("base_task", t))
        boot = (sm.groupby(keys, as_index=False)
                  .agg(boot_ci_lo=("boot_ci_lo", "mean"),
                       boot_ci_hi=("boot_ci_hi", "mean")))
        agg = agg.merge(boot, on=keys, how="left")
        agg["boot_ci_width"] = agg["boot_ci_hi"] - agg["boot_ci_lo"]

    # Step 4 — legacy diagnostic.
    lg = load_legacy()
    if lg is not None:
        lg = add_legacy_aliases(lg, set(agg['base_task'].unique()))
        agg = agg.merge(lg, on=keys, how="left")
        # An SD of exactly 0 and an SD of 3e-17 are the same statement — the
        # metric did not move across retrainings — but only the first was
        # blanked here, so a seed-invariant cell could report a z of +0.89 built
        # entirely out of last-bit float noise (found 2026-09-01, B1 / GeomMLP /
        # kNN-11). The guard is an epsilon floor, not an equality test.
        sd = agg["train_seed_sd"].mask(agg["train_seed_sd"].abs() < SD_FLOOR)
        agg["legacy_z"] = (agg["legacy_value"] - agg["train_seed_mean"]) / sd
        agg["legacy_inside_seed_range"] = (
            (agg["legacy_value"] >= agg["train_seed_min"]) &
            (agg["legacy_value"] <= agg["train_seed_max"]))

    thin = agg[agg["n_train_seeds"] < MIN_SEEDS_TO_REPORT]
    if len(thin):
        print(f"  WARNING: {len(thin)} rows have <{MIN_SEEDS_TO_REPORT} "
              f"training seeds; their SD is not reportable")
        for _, r in thin.head(10).iterrows():
            print(f"    {r.base_task}/{r.encoder}/{r.head}/{r.level}/"
                  f"{r.metric}: n={r.n_train_seeds}")

    agg = agg.sort_values(keys).reset_index(drop=True)
    per_train.to_csv(OUT / "trainseed_per_run.csv", index=False)
    agg.to_csv(OUT / "trainseed_summary.csv", index=False)
    print(f"wrote trainseed_per_run.csv ({len(per_train)} rows)")
    print(f"wrote trainseed_summary.csv ({len(agg)} rows)")

    # Console view of the headline rows.
    print("\n" + "=" * 100)
    print("TRAINING-SEED SPREAD — core-claim tasks, softmax/scan")
    print("=" * 100)
    view = agg[(agg["head"] == "softmax") & (agg["level"] == "scan") &
               (agg["metric"].isin(["accuracy", "macro_auc",
                                    "top3_accuracy", "top5_accuracy"]))]
    for bt in ["B1_corrected", "B2_corrected"]:
        v = view[view["base_task"] == bt]
        if not len(v):
            continue
        print(f"\n{bt} — {C.TASKS[bt]['note']}")
        print(f"  {'encoder':10s} {'metric':16s} {'mean+/-SD (train)':>22s} "
              f"{'range':>7s} {'boot 95% CI':>18s} {'legacy z':>9s}")
        for _, r in v.iterrows():
            ci = (f"[{r.boot_ci_lo:.3f}, {r.boot_ci_hi:.3f}]"
                  if "boot_ci_lo" in r and pd.notna(r.get("boot_ci_lo")) else "—")
            lz = (f"{r.legacy_z:+.2f}" if pd.notna(r.get("legacy_z")) else "—")
            print(f"  {r.encoder:10s} {r.metric:16s} "
                  f"{r.train_seed_mean:9.4f} +/- {r.train_seed_sd:<8.4f} "
                  f"{r.train_seed_range:7.4f} {ci:>18s} {lz:>9s}")


if __name__ == "__main__":
    main()
