"""
Group cached prediction files by TRAINING run.  2026-09-08.

The prediction cache names every file `<task>__<enc>__<head>__<level>__seed<N>`,
where seed<N> is the INFERENCE seed -- the farthest-point-sampling draw, taken
with the model weights frozen. The training run is carried in the TASK name
(`B2_corrected` for the legacy checkpoint, `B2_corrected_ts{1..5}` for the
sweep), which is easy to miss: a glob over `seed*` looks like it averages five
models and averages five samplings of one model instead.

That is exactly how Figs. S1-S4 and Fig. 4 kept describing a single checkpoint
after the manuscript's headline numbers moved to the five-training-seed sweep
(2026-09-03). This helper makes the distinction explicit at the call site: it
returns one entry PER TRAINING RUN, each holding that run's inference-seed
files, so a caller has to decide how to reduce within a run before it can
average across runs.

The order matters and is not interchangeable. Reduce inference seeds WITHIN a
run first, then across runs -- the order aggregate_trainseed.py uses. Pooling
all 25 files at once would let sampling noise inflate what is reported as
retraining spread.
"""
import glob

TRAIN_SEEDS = (1, 2, 3, 4, 5)


def run_groups(pred_dir, base, head, level, family, enc="pointnet2",
               seeds=TRAIN_SEEDS):
    """[(task_name, [file, ...]), ...] — one entry per training run.

    family "legacy"    -> the single pre-sweep checkpoint, one entry
    family "trainseed" -> the five independently trained models
    """
    if family not in ("legacy", "trainseed"):
        raise ValueError(f"unknown family {family!r}")
    tasks = [base] if family == "legacy" else [f"{base}_ts{s}" for s in seeds]
    groups = []
    for t in tasks:
        fs = sorted(glob.glob(
            str(pred_dir / f"{t}__{enc}__{head}__{level}__seed*.npz")))
        if not fs:
            raise SystemExit(f"trainseed_pred: no predictions for "
                             f"{t}/{enc}/{head}/{level}")
        groups.append((t, fs))
    if family == "trainseed" and len(groups) != len(seeds):
        raise SystemExit(f"trainseed_pred: {len(groups)} training runs for "
                         f"{base}, expected {len(seeds)}")
    return groups
