#!/usr/bin/env python3
"""
Collapse the LEAKED arm's training-seed sweep into the numbers Sec. 2.7 reports.
thread: facebase3d-figs-2026-09-04

Companion to aggregate_trainseed.py, which does the same for the corrected arm.
Run after the ten rerun_scanlevel.py --train-seed jobs.

WHAT THIS FIXES. After the 2026-08-31 sweep the subject-disjoint arm of Fig. 5(a)
was a five-training-seed mean while the leaked arm was one unseeded checkpoint
per task. The 2026-08-18 rework had already removed exactly this asymmetry once,
in its inference-seed form; the sweep put it back in a training-seed form. Both
arms are now five retrainings.

VARIANCE SOURCES, NOT POOLED -- same discipline as aggregate_trainseed.py:
  (T) across the 5 retrainings, split frozen at 42 so all five see IDENTICAL
      test scans. Inference seeds are collapsed WITHIN each retraining first, so
      PointNet++'s FPS noise cannot leak into (T).
  (I) across the 5 inference seeds, weights frozen.
  (S) 95% bootstrap over test SUBJECTS, carried across from the per-seed
      summaries and averaged over training seeds.

THE DELTA IS STILL NOT PAIRED. The two arms have different test sets -- that is
the whole point of the comparison -- so no interval may be attached to the
difference, and the per-seed deltas are NOT a paired sample. What IS reportable
without pairing is separation: whether the WORST leaked retraining still beats
the BEST corrected one. That is `separated` below, and it is the honest form of
"this gap is not a seed artefact".

NOTHING HERE IS TASK PERFORMANCE. ~79% (B1) and ~85% (B2) of leaked-arm test
scans come from subjects also seen in training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

OUT = C.OUT
ARMS = {"B1_scanlevel": "B1_corrected", "B2_scanlevel": "B2_corrected"}
KEYS = ["base_task", "encoder", "head", "level", "metric"]


def load_leaked():
    """Per (training seed, inference seed) values from the ten scoring jobs."""
    frames, missing = [], []
    for task in ("B1", "B2"):
        for ts in C.TRAIN_SEEDS:
            f = OUT / f"scanlevel_per_seed_{task}_ts{ts}.csv"
            if not f.exists():
                missing.append(f.name)
                continue
            frames.append(pd.read_csv(f))
    if missing:
        sys.exit("missing scoring output — run the rerun_scanlevel_trainseed "
                 "array first:\n  " + "\n  ".join(missing))
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"seed": "infer_seed"})
    # Guard: an SD across training seeds is meaningless if the arms were scored
    # on different test sets. n_units is per (task, train_seed) constant by
    # construction; check it rather than assume it.
    return df


def load_boot():
    """Bootstrap intervals, one summary per (task, training seed)."""
    frames = []
    for task in ("B1", "B2"):
        for ts in C.TRAIN_SEEDS:
            f = OUT / f"scanlevel_summary_ci_{task}_ts{ts}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True) if frames else None


def load_corrected():
    """The subject-disjoint arm, already aggregated by aggregate_trainseed.py."""
    f = OUT / "trainseed_summary.csv"
    if not f.exists():
        sys.exit(f"missing {f} — run aggregate_trainseed.py first")
    c = pd.read_csv(f)
    c = c[(c.encoder == "pointnet2") & (c["head"] == "softmax")
          & (c.level == "scan") & (c.base_task.isin(ARMS.values()))]
    return c


def load_legacy_leaked():
    """The single unseeded leaky checkpoint, for the legacy_z diagnostic only."""
    f = OUT / "scanlevel_summary_ci_all.csv"
    if not f.exists():
        return None
    lg = pd.read_csv(f)
    return lg.rename(columns={"task": "base_task", "mean": "legacy_value"})[
        ["base_task", "encoder", "head", "level", "metric", "legacy_value"]]


def main():
    df = load_leaked()

    # Step 1 — collapse inference seeds WITHIN each retraining.
    per_train = (df.groupby(KEYS + ["train_seed"], as_index=False)
                   .agg(value=("value", "mean"),
                        infer_seed_sd=("value", lambda v: float(np.std(v, ddof=1))
                                       if len(v) > 1 else 0.0),
                        n_infer_seeds=("value", "size")))

    # Step 2 — dispersion ACROSS retrainings.
    agg = (per_train.groupby(KEYS, as_index=False)
                    .agg(train_seed_mean=("value", "mean"),
                         train_seed_sd=("value", lambda v: float(np.std(v, ddof=1))
                                        if len(v) > 1 else np.nan),
                         train_seed_min=("value", "min"),
                         train_seed_max=("value", "max"),
                         n_train_seeds=("value", "size"),
                         infer_seed_sd=("infer_seed_sd", "mean")))
    agg["train_seed_range"] = agg["train_seed_max"] - agg["train_seed_min"]

    # Step 3 — subject bootstrap and the leak fraction, carried across.
    bt = load_boot()
    if bt is not None:
        b = (bt.groupby(KEYS, as_index=False)
               .agg(boot_ci_lo=("boot_ci_lo", "mean"),
                    boot_ci_hi=("boot_ci_hi", "mean"),
                    leaked_scan_frac=("leaked_scan_frac", "first"),
                    n_units=("n_units", "first"),
                    n_subjects=("n_subjects", "first")))
        # The five retrainings must have been scored on the same test set, or
        # the SD in step 2 is a split SD wearing a training-seed label.
        spread = bt.groupby(KEYS)["n_units"].nunique()
        if (spread > 1).any():
            sys.exit("test-set size varies across training seeds — the sweep is "
                     "not scored on a frozen split; refusing to aggregate")
        agg = agg.merge(b, on=KEYS, how="left")

    # Step 4 — the corrected arm, and the unpaired contrast.
    corr = load_corrected().rename(columns={
        "train_seed_mean": "corr_mean", "train_seed_sd": "corr_sd",
        "train_seed_min": "corr_min", "train_seed_max": "corr_max"})
    corr["base_task"] = corr["base_task"].map(
        {v: k for k, v in ARMS.items()})
    agg = agg.merge(corr[KEYS + ["corr_mean", "corr_sd", "corr_min",
                                 "corr_max"]], on=KEYS, how="left")
    agg["delta_pp"] = 100 * (agg["train_seed_mean"] - agg["corr_mean"])
    # Separation: does the WORST leaked retraining still beat the BEST corrected
    # one? A claim available without pairing the arms.
    agg["separated"] = agg["train_seed_min"] > agg["corr_max"]

    # Step 5 — legacy diagnostic for the leaked arm itself.
    lg = load_legacy_leaked()
    if lg is not None:
        agg = agg.merge(lg, on=KEYS, how="left")
        sd = agg["train_seed_sd"].where(agg["train_seed_sd"] > 1e-9)
        agg["legacy_z"] = (agg["legacy_value"] - agg["train_seed_mean"]) / sd
        agg["legacy_inside_seed_range"] = (
            (agg["legacy_value"] >= agg["train_seed_min"]) &
            (agg["legacy_value"] <= agg["train_seed_max"]))

    agg = agg.sort_values(KEYS).reset_index(drop=True)
    per_train.to_csv(OUT / "scanlevel_trainseed_per_run.csv", index=False)
    agg.to_csv(OUT / "scanlevel_trainseed_summary.csv", index=False)
    print(f"wrote scanlevel_trainseed_per_run.csv ({len(per_train)} rows)")
    print(f"wrote scanlevel_trainseed_summary.csv ({len(agg)} rows)")

    print("\n" + "=" * 104)
    print("LEAKAGE CONTRAST — both arms as five-retraining means, PointNet++ "
          "softmax/scan")
    print("=" * 104)
    print(f"  {'task':14s} {'metric':10s} {'leaked mean+/-SD':>20s} "
          f"{'subj-disjoint mean+/-SD':>25s} {'delta pp':>9s} {'sep':>5s} "
          f"{'legacy z':>9s}")
    for _, r in agg.iterrows():
        lz = f"{r.legacy_z:+.2f}" if pd.notna(r.get("legacy_z")) else "—"
        print(f"  {r.base_task:14s} {r.metric:10s} "
              f"{r.train_seed_mean:8.4f} +/- {r.train_seed_sd:<7.4f} "
              f"{r.corr_mean:12.4f} +/- {r.corr_sd:<7.4f} "
              f"{r.delta_pp:9.1f} {str(bool(r.separated)):>5s} {lz:>9s}")


if __name__ == "__main__":
    main()
