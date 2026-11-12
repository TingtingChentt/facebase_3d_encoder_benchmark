"""
Task registry for every model reported in the paper.

THREE different "seeds" live in this pipeline. Do not conflate them:
  * DATA SPLIT SEED  — fixed at 42. Defines train/val/test membership (for the
    syndrome tasks the split is stored in the manifest's `fixed_split` column
    instead). Every checkpoint was trained under it. Never varied.
  * INFERENCE SEED   — {0,1,2,3,4}. Controls only the PointNet++
    farthest-point-sampling start index at evaluation time, weights frozen.
    The other three encoders are deterministic and are scored once.
  * TRAIN SEED       — {1,2,3,4,5}. Weight init, batch order, augmentation.
    Retrains the model; the split is untouched, so all five seeds are scored
    on IDENTICAL test subjects and the spread is retraining variance alone.
    Implemented in facebench/training/seeding.py; swept by the *_ts<N> tasks.

TASK NAMES. The keys are the names the experiments were run under, kept so the
result files in results/ match them one-to-one:

  A1_fix_v3     four-class cleft subtyping            (FB-5A, FB-56)
  A2_fix_v3     binary cleft screening                (FB-5A, FB-56)
  C_full_v3     combined affected-vs-unaffected screening (all six datasets)
  CF20_full_v3  49-class syndrome diagnosis, FB-TJ0 `Syndrome` field,
                >=50 scans and >=20 subjects per class — the main fine-grained task
  CF{K}_full_v3 class-size sensitivity sweep, >=50 scans and >=K subjects per
                class, K in {10, 15, 30, 40, 50, 75} (CF20 is the main task)

  "_v3"   = third revision of the point-cloud preprocessing
            (facebench/preprocessing/preprocess.py: weld coincident vertices,
            then grow from the largest component). It is the only preprocessing
            in this repository.
  "_fix" / "_full" = early stopping disabled (patience == epochs == 200); the
            checkpoint is still the epoch with the best validation macro-F1.
            On the cleft tasks the training loss sits on a ln(K) plateau for
            tens of epochs and validation macro-F1 is noise there, so a short
            patience stopped some runs inside the plateau; the rule is applied
            identically to all four encoders so the encoder comparison stays an
            encoder comparison.
  "_ts<N>" = train seed N.

The A-family tasks have id_col=None: extract_features.py gives each scan a
synthetic id, so every scan is its own unit and only scan-level softmax is
scored (a subject-level row would duplicate it).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from facebench.paths import MANIFESTS, ROOT, RUNS, RESULTS  # noqa: E402

PROJECT = ROOT
DATA = MANIFESTS        # labelled manifests, built by facebench/preprocessing/build_manifests.py
OUT = RESULTS

DATA_SPLIT_SEED = 42          # frozen — see docstring
INFERENCE_SEEDS = [0, 1, 2, 3, 4]
TRAIN_SEEDS = [1, 2, 3, 4, 5]
EPOCHS = 200                  # == patience for every task, i.e. no early stopping

ENCODERS = ["geommlp", "pointnet", "dgcnn", "pointnet2"]
CKPT_PREFIX = {"geommlp": "GeomMLP", "pointnet": "PointNet",
               "dgcnn": "DGCNN", "pointnet2": "PointNet2"}
# PointNet++ draws its FPS start index from an RNG at inference; the other
# three encoders are bit-deterministic in eval mode.
NONDETERMINISTIC = {"pointnet2"}
# Extraction batch sizes (memory only; results do not depend on them).
BATCH_SIZE = {"geommlp": 64, "pointnet": 32, "dgcnn": 4, "pointnet2": 16}

HEADS = ["softmax", "proto", "knn11"]
LEVELS = ["scan", "subject"]
KNN_K = 11
N_BOOTSTRAP = 2000


def trainseed_task(base, seed):
    return f"{base}_ts{seed}"


V3 = " — v3 clouds"
BASES = {
    "A1_fix_v3": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0, id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC 4-class — early stopping disabled"),
    "A2_fix_v3": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=2, binary_mode=True, aux_dim=0, id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC binary — early stopping disabled"),
    "C_full_v3": dict(
        manifest=DATA / "combined_manifest.csv", exp_type="combined",
        num_classes=2, binary_mode=True, aux_dim=0, id_col="subject_id",
        heads=["softmax"], levels=["scan", "subject"],
        note="combined binary — no early stopping (patience 200)"),
}
# Class-size sweep: K -> number of classes retained (>=50 scans AND >=K subjects).
# All seven manifests share ONE subject split assigned per class, so test sets
# are nested across K.
CF_CUTOFFS = {10: 55, 15: 53, 20: 49, 30: 42, 40: 29, 50: 22, 75: 12}
for _k, _nc in CF_CUTOFFS.items():
    BASES[f"CF{_k}_full_v3"] = dict(
        manifest=DATA / f"syndrome_k{_k}_manifest.csv", exp_type="syndrome_name",
        num_classes=_nc, binary_mode=False, aux_dim=0, id_col="fbid", cutoff=_k,
        heads=["softmax", "proto"], levels=["scan", "subject"],
        note=f"{_nc}-class Syndrome, >=50 scans & >=%d subj/class — no early stopping" % _k)

TASKS = {}
for _b, _cfg in BASES.items():
    TASKS[_b] = dict(_cfg, run_dir=_b, note=_cfg["note"] + V3)
    for _ts in TRAIN_SEEDS:
        _n = trainseed_task(_b, _ts)
        TASKS[_n] = dict(_cfg, run_dir=_n, train_seed=_ts, base_task=_b,
                         note=f"{_cfg['note']} — train seed {_ts}{V3}")
del _b, _cfg, _ts, _n, _k, _nc

# Families: base -> its five seeded tasks. One analysis suffix per family
# (results/*_<suffix>*.csv); never pooled.
FAMILIES = {b: [trainseed_task(b, s) for s in TRAIN_SEEDS] for b in BASES}
MAIN_FAMILIES = ["A1_fix_v3", "A2_fix_v3", "C_full_v3", "CF20_full_v3"]
CF_FAMILIES = [f"CF{k}_full_v3" for k in CF_CUTOFFS]
SUFFIX = {"A1_fix_v3": "_a1fixv3", "A2_fix_v3": "_a2fixv3",
          "C_full_v3": "_cfullv3", "CF20_full_v3": "_cf20fullv3"}
CF_SUFFIX = "_cfv3"           # all seven cutoffs analysed together
TRAINED_TASKS = [t for b in BASES for t in FAMILIES[b]]

# Top-k accuracy, reported only for the many-class tasks.
TOPK = [3, 5]
TOPK_MIN_CLASSES = 10


def topk_for(task):
    """k values to score for `task`. Empty for the binary and 4-class tasks."""
    nc = TASKS[task]["num_classes"]
    if nc < TOPK_MIN_CLASSES:
        return []
    return [k for k in TOPK if k < nc]


def task_encoders(task):
    return TASKS[task].get("encoders", ENCODERS)


def heads_for(task):
    return TASKS[task].get("heads", HEADS)


def levels_for(task):
    """A task whose id_col is None has one synthetic id per scan, so a
    subject-level row would duplicate the scan-level row exactly."""
    return TASKS[task].get("levels", LEVELS)


def seeds_for(encoder):
    """Deterministic encoders are run once; repeating them 5x would only
    reproduce identical numbers at 5x the cost."""
    return INFERENCE_SEEDS if encoder in NONDETERMINISTIC else [0]


def checkpoint_path(task, encoder):
    run = TASKS[task]["run_dir"]
    return RUNS / run / encoder / "checkpoints" / f"{CKPT_PREFIX[encoder]}_{run}_best.pt"


def feature_path(task, encoder, seed, split):
    return OUT / "features" / f"{task}__{encoder}__seed{seed}__{split}.npz"
