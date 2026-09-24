"""
Shared registry for the 5-seed paper re-eval (blocker B1, thread
facebase3d-paper-2026-08-04).

IMPORTANT — THREE different "seeds" live in this pipeline. Do not conflate them:
  * DATA SPLIT SEED  — fixed at 42 forever. Defines train/val/test membership.
    Every checkpoint was trained under it; changing it would invalidate the
    frozen weights. Never varied.
  * INFERENCE SEED   — {0,1,2,3,4}. Controls only the PointNet++ FPS start
    index at eval time, weights frozen.
  * TRAIN SEED       — {1,2,3,4,5}, added 2026-08-31.
    Weight init, batch order, augmentation. Retrains the model; the split is
    untouched, so all five seeds are scored on IDENTICAL test subjects and
    the spread is retraining variance alone. Implemented in
    facebench/models/seeding.py; swept by the *_ts<N> tasks below.

    The point behind it: the published intervals are a bootstrap
    over test subjects plus an inference-seed spread with weights frozen.
    Neither answers "would this hold if we trained again". The pre-2026-08-31
    checkpoints were trained with an UNSEEDED torch RNG, so they are single
    unreproducible draws and no seed can be assigned to them retroactively.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "data/processed"
RUNS = PROJECT / "runs"
OUT = PROJECT / "results"

DATA_SPLIT_SEED = 42          # frozen — see docstring
INFERENCE_SEEDS = [0, 1, 2, 3, 4]

ENCODERS = ["geommlp", "pointnet", "dgcnn", "pointnet2"]
CKPT_PREFIX = {"geommlp": "GeomMLP", "pointnet": "PointNet",
               "dgcnn": "DGCNN", "pointnet2": "PointNet2"}

# Encoders whose inference path is non-deterministic and therefore genuinely
# need multiple inference seeds. Established empirically by
# facebench/eval/probe_determinism.py — not assumed.
NONDETERMINISTIC = {"pointnet2"}

# Per-encoder eval batch size. DGCNN builds a (B, 4096, 4096) neighbour graph,
# so it needs a much smaller batch than the rest or it exhausts RAM.
BATCH_SIZE = {"geommlp": 64, "pointnet": 32, "dgcnn": 4, "pointnet2": 16}

TASKS = {
    # A-family: id_col=None, so extract_features.py assigns each scan a synthetic
    # id ("scan0", "scan1", ...). Every scan is therefore its own subject and a
    # subject-level row would be a byte-identical copy of the scan-level row.
    # Scored scan-level / softmax only — see heads_for() / levels_for().
    "A1": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0,
        run_dir="A1", id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC 4-class",
    ),
    # NOTE: A1_S1/S2/S3 are NOT site splits despite the name — see
    # AUDIT_NONDETERMINISM.md. They are class-imbalance strategy variants
    # trained on the identical full-cohort split as A1.
    "A1_S1": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0,
        run_dir="A1_S1", id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC 4-class, weighted sampler (MISLABELLED as 'site 1')",
    ),
    "A1_S2": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0,
        run_dir="A1_S2", id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC 4-class, focal loss (MISLABELLED as 'site 2')",
    ),
    "A1_S3": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0,
        run_dir="A1_S3", id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC 4-class, weighted+focal (MISLABELLED as 'site 3')",
    ),
    "A2": dict(
        manifest=DATA / "ofc_manifest.csv", exp_type="ofc",
        num_classes=2, binary_mode=True, aux_dim=0,
        run_dir="A2", id_col=None,
        heads=["softmax"], levels=["scan"],
        note="OFC binary",
    ),
    "B1_corrected": dict(
        manifest=DATA / "syndrome_manifest_b1.csv", exp_type="syndrome",
        num_classes=19, binary_mode=False, aux_dim=0,
        run_dir="B1_corrected", id_col="fbid",
        note="19-class syndrome category — CORE CLAIM",
    ),
    "B2_corrected": dict(
        manifest=DATA / "syndrome_clinical_manifest.csv", exp_type="clinical",
        num_classes=33, binary_mode=False, aux_dim=0,
        run_dir="B2_corrected", id_col="fbid",
        note="33-class clinical diagnosis — CORE CLAIM",
    ),
    "B2_r2": dict(
        manifest=DATA / "syndrome_clinical_manifest_r2.csv", exp_type="clinical",
        num_classes=24, binary_mode=False, aux_dim=2,
        run_dir="B2_r2", id_col="fbid",
        note="24-class filtered (age+sex covariates)",
    ),
    "C_corrected": dict(
        manifest=DATA / "combined_manifest.csv", exp_type="combined",
        num_classes=2, binary_mode=True, aux_dim=0,
        run_dir="C_corrected", id_col="subject_id",
        heads=["softmax"], levels=["scan", "subject"],
        note="combined binary",
    ),
    # PointNet++-only separately-trained CE-loss/proto-head checkpoints.
    # These reproduce the AS-PUBLISHED C1 comparison, which confounds head
    # choice with training run. Kept distinct from the encoder-fixed contrast.
    "B1_proto_cetrain": dict(
        manifest=DATA / "syndrome_manifest_b1.csv", exp_type="syndrome",
        num_classes=19, binary_mode=False, aux_dim=0,
        run_dir="B1_proto_cetrain", id_col="fbid",
        encoders=["pointnet2"],
        note="B1 CE-trained proto-head checkpoint (as-published arm)",
    ),
    "B2_proto_cetrain": dict(
        manifest=DATA / "syndrome_clinical_manifest.csv", exp_type="clinical",
        num_classes=33, binary_mode=False, aux_dim=0,
        run_dir="B2_proto_cetrain", id_col="fbid",
        encoders=["pointnet2"],
        note="B2 CE-trained proto-head checkpoint (as-published arm)",
    ),
}

# ── Cross-site leave-one-site-out folds (thread ...-2026-08-08, PART A) ─────
# TEST is one held-out acquisition site in full; train/val are a stratified
# split over the remaining ten. The five sites failing the post-attrition
# per-class gate (CL, CO, NG, SF, TX) stay in the TRAINING pool but are never
# scored as folds. Fold viability is derived in audit_xsite_folds.py, not
# asserted here.
#
# THE VERDICT MUST BE REPORTED SPLIT BY GROUP AND NEVER AVERAGED:
#   XSITE_BOTH_DATASETS  — both FB-56 and FB-5A remain in training, so holding
#                          this site out isolates SITE from DATASET.
#   XSITE_SINGLE_DATASET — single-dataset site; site and dataset move together
#                          and are measured jointly.
#
# Reads ofc_manifest_xsite.csv (= ofc_manifest.csv + a composite
# dataset|study_id subject column) so the bootstrap unit is literally the test
# subject. make_xsite_manifest.py verifies the splits are row-identical to the
# manifest the models were trained on.
XSITE_BOTH_DATASETS = ["PH", "FC", "PR"]
XSITE_SINGLE_DATASET = ["GW", "PT", "LC"]
XSITE_SITES = XSITE_BOTH_DATASETS + XSITE_SINGLE_DATASET

for _site in XSITE_SITES:
    _grp = ("site isolated from dataset" if _site in XSITE_BOTH_DATASETS
            else "site+dataset confounded")
    TASKS["XS_" + _site] = dict(
        manifest=DATA / "ofc_manifest_xsite.csv", exp_type="ofc",
        num_classes=4, binary_mode=False, aux_dim=0,
        run_dir="XS_" + _site, id_col="subject_id",
        holdout_site=_site,
        heads=["softmax"], levels=["scan"],
        note="cross-site LOSO, held-out test site %s (%s)" % (_site, _grp),
    )
for _site in XSITE_SITES:
    _grp = ("site isolated from dataset" if _site in XSITE_BOTH_DATASETS
            else "site+dataset confounded")
    TASKS["XSB_" + _site] = dict(
        manifest=DATA / "ofc_manifest_xsite.csv", exp_type="ofc",
        num_classes=2, binary_mode=True, aux_dim=0,
        run_dir="XSB_" + _site, id_col="subject_id",
        holdout_site=_site,
        heads=["softmax"], levels=["scan"],
        note="cross-site LOSO, OFC BINARY, held-out test site %s (%s)"
             % (_site, _grp),
    )
del _site, _grp

# XSB_* is the SAME six folds as XS_*, on the binary Affected/Unaffected task.
# It exists because the 4-class LOSO sweep cannot speak to this paper's lead
# claim: 4-class is weak WITHIN site too (PointNet++ macro-AUC 0.573), whereas
# screening rests on the binary tasks. Same manifest, folds, seed and encoders
# as XS_* so the two are comparable — the only difference is the label
# collapse. Held-out sites are 71-92% Unaffected, so under binarisation the
# majority-class predictor scores that outright: macro-AUC is the headline
# metric and accuracy must never be quoted without the fold's majority rate.

# ── Cross-site LOSO retrained without early stopping ───────────────────────
# thread facebase3d-xsitediag-2026-09-07.
#
# WHY. The A1/A2 diagnosis applies here unchanged. Every XS_*/XSB_* checkpoint
# was trained with patience 20 counted on a validation macro-F1 that is pure
# noise during each cleft task's initial plateau, and the four-class folds show
# it: PointNet++ stopped at 21, 38, 46, 39, 39 and 46 epochs across the six
# sites, all inside the plateau the A1 probe needed far longer to escape. So
# Sec 2.7's "four-class cleft subtyping does not transfer to unseen sites" is
# not yet separable from "four-class cleft training was stopped before it
# learned anything", which is a different claim about a different thing.
#
# BOTH ARMS, NOT JUST THE FOUR-CLASS ONE. The result Sec 2.7 reports is a
# CONTRAST between four-class and binary cross-site transfer on identical folds.
# Correcting one arm's stopping rule and not the other would make that contrast
# a comparison of protocols — the same reason the A1/A2 fixes were applied to
# all four encoders rather than only the ones that visibly suffered.
#
# NOT A TRAINING-SEED SWEEP. The published cross-site arm is one run per
# (fold, encoder) and its intervals are subject bootstraps within a held-out
# site. These retrains keep that design exactly; the ONLY thing that changes is
# --patience. Adding a seed axis here would be a second change and would make
# the before/after uninterpretable.
#
# Everything else is pinned: same ofc_manifest_xsite.csv, same six folds, same
# group assignment, same split seed 42, same encoders, same scan-level softmax
# scoring.
XSITE_FIX_EPOCHS = 200        # == patience, i.e. early stopping never fires

for _site in XSITE_SITES:
    for _pref, _bin, _nc in [("XS_", False, 4), ("XSB_", True, 2)]:
        _src = _pref + _site
        _dst = _pref + "fix_" + _site
        TASKS[_dst] = dict(TASKS[_src])
        TASKS[_dst]["run_dir"] = _dst
        TASKS[_dst]["legacy_base"] = _src
        TASKS[_dst]["note"] = TASKS[_src]["note"] + " — early stopping disabled"
del _site, _pref, _bin, _nc, _src, _dst

XSITE_FIX_TASKS = ["XS_fix_" + s for s in XSITE_SITES]
XSITE_FIX_BIN_TASKS = ["XSB_fix_" + s for s in XSITE_SITES]

# ── Training-seed sweep (thread facebase3d-trainseed-2026-08-31) ───────────
# One task entry per (base task, train seed). Everything is inherited from the
# base task except run_dir, so the manifest, class count, heads and levels are
# guaranteed identical to the arm the seeded runs are meant to replace — and
# checkpoint_path() resolves them with no change, because trainer.py names
# checkpoints {ModelClass}_{exp_id}_best.pt and exp_id is the run_dir.
TRAIN_SEEDS = [1, 2, 3, 4, 5]
TRAINSEED_BASE_TASKS = ["A1", "A2", "B1_corrected", "B2_corrected",
                        "C_corrected"]


def trainseed_task(base, seed):
    return f"{base}_ts{seed}"


for _base in TRAINSEED_BASE_TASKS:
    for _ts in TRAIN_SEEDS:
        _name = trainseed_task(_base, _ts)
        TASKS[_name] = dict(TASKS[_base])
        TASKS[_name]["run_dir"] = _name
        TASKS[_name]["train_seed"] = _ts
        TASKS[_name]["base_task"] = _base
        TASKS[_name]["note"] = f"{TASKS[_base]['note']} — train seed {_ts}"
del _base, _ts, _name

TRAINSEED_TASKS = [trainseed_task(b, s)
                   for b in TRAINSEED_BASE_TASKS for s in TRAIN_SEEDS]

# ── A2 retrained without early stopping ────────────────────────────────────
# thread facebase3d-a2diag-2026-09-04.
#
# WHY. A2's five training-seed runs did not fail to learn the task; early
# stopping killed them inside the task's initial plateau. On A2 the model sits
# at train_loss ~= ln(2) with validation macro-F1 pure noise for roughly the
# first 50 epochs. Patience is 20 and it is counted on that noisy macro-F1, so
# survival past the plateau depends on whether F1 noise happens to set a new
# maximum within some 20-epoch window. The legacy unseeded PointNet++ run won
# that draw (F1 ratcheted at epochs 8, 15, 27, 35, 40) and reached val AUC 0.878
# by epoch 111; all five retrainings lost it and died at epochs 21-38 with val
# AUC 0.59-0.60. Confirmed directly: A2_ts1 rerun with patience disabled and
# nothing else changed passed val AUC 0.797 by epoch 78, against 0.592 as
# shipped.
#
# It is not PointNet++-specific. DGCNN loses the same draw on 1 of 5 seeds
# (ts1 died at 22 epochs, val AUC 0.625, against 0.79-0.84 for the four that
# survived), which is what inflates its reported A2 AUC SD to 0.105. GeomMLP has
# no plateau and is unaffected; PointNet never separates on this task either way.
#
# THE FIX IS UNIFORM ACROSS THE FOUR ENCODERS, NOT APPLIED TO POINTNET++ ALONE.
# A2's row compares encoders, so a protocol that differed between them would
# turn that comparison into a protocol comparison. All four are retrained with
# early stopping disabled (--patience 200 == --epochs) and the checkpoint still
# selected by best validation macro-F1, which is the same selection rule as
# everywhere else. Disabling early stopping can only change a result where a run
# was previously stopped BEFORE its best checkpoint would have occurred, so it is
# a strictly safer protocol, not a more permissive one.
#
# The broken A2_ts* runs are kept, not overwritten. They are the evidence.
A2_FIX_BASE = "A2_fix"
TASKS[A2_FIX_BASE] = dict(TASKS["A2"])
TASKS[A2_FIX_BASE]["run_dir"] = A2_FIX_BASE
TASKS[A2_FIX_BASE]["note"] = "OFC binary — early stopping disabled"
# The published A2 checkpoint is the comparison point for this retrain: same
# cohort, same frozen split, same metric, only the stopping rule differs. Read
# by aggregate_trainseed.add_legacy_aliases to fill the legacy_* columns.
TASKS[A2_FIX_BASE]["legacy_base"] = "A2"
for _ts in TRAIN_SEEDS:
    _name = trainseed_task(A2_FIX_BASE, _ts)
    TASKS[_name] = dict(TASKS[A2_FIX_BASE])
    TASKS[_name]["run_dir"] = _name
    TASKS[_name]["train_seed"] = _ts
    TASKS[_name]["base_task"] = A2_FIX_BASE
    TASKS[_name]["note"] = f"{TASKS[A2_FIX_BASE]['note']} — train seed {_ts}"
del _ts, _name

A2_FIX_TASKS = [trainseed_task(A2_FIX_BASE, s) for s in TRAIN_SEEDS]
A2_FIX_ENCODERS = ["pointnet2", "dgcnn", "pointnet", "geommlp"]
A2_FIX_EPOCHS = 200          # == patience, i.e. early stopping never fires

# ── A1 retrained without early stopping ────────────────────────────────────
# thread facebase3d-a1diag-2026-09-04, run 2026-09-07.
#
# WHY. The A2 diagnosis (see the A2_FIX block) raised the same question for A1,
# and the probe run at the time did NOT clear it. On A1 every run of every
# encoder — the legacy ones included — ends at train_loss 1.38-1.45, i.e. at
# ln(4) = 1.3863, the chance loss for a four-class task, and unlike A2 the
# legacy runs and the retrainings agree (|legacy z| < 1.2 in 11 of 12 cells),
# so there was no lucky-draw split pointing at the stopping rule. What the probe
# showed instead is that the plateau is ESCAPABLE: with patience disabled at
# train seed 1, train loss falls off ln(4) and macro AUC goes 0.499 -> 0.637
# (PointNet++) and 0.547 -> 0.703 (DGCNN), both well above their legacy single
# runs (0.573, 0.551).
#
# That makes the shipped A1 row unreportable as it stands. Near-chance
# four-class subtyping is a RESULT in this paper — it carries the task-hierarchy
# claim and Sec 2.8's "no PointNet++ interval excludes chance" — and the lab now
# holds evidence that the number is a property of the stopping rule rather than
# of the task. Under the corrected protocol the claim either survives or it does
# not; either way it is measured, not assumed.
#
# Same protocol as A2_FIX and for the same reason: all four encoders, not just
# the two probed, because A1's row is an encoder comparison and a stopping rule
# that differed between encoders would make it a protocol comparison. Checkpoint
# selection is unchanged (best validation macro-F1).
#
# The truncated A1_ts* runs are kept. They are the evidence for the contrast.
A1_FIX_BASE = "A1_fix"
TASKS[A1_FIX_BASE] = dict(TASKS["A1"])
TASKS[A1_FIX_BASE]["run_dir"] = A1_FIX_BASE
TASKS[A1_FIX_BASE]["note"] = "OFC 4-class — early stopping disabled"
TASKS[A1_FIX_BASE]["legacy_base"] = "A1"
for _ts in TRAIN_SEEDS:
    _name = trainseed_task(A1_FIX_BASE, _ts)
    TASKS[_name] = dict(TASKS[A1_FIX_BASE])
    TASKS[_name]["run_dir"] = _name
    TASKS[_name]["train_seed"] = _ts
    TASKS[_name]["base_task"] = A1_FIX_BASE
    TASKS[_name]["note"] = f"{TASKS[A1_FIX_BASE]['note']} — train seed {_ts}"
del _ts, _name

A1_FIX_TASKS = [trainseed_task(A1_FIX_BASE, s) for s in TRAIN_SEEDS]
A1_FIX_ENCODERS = ["pointnet2", "dgcnn", "pointnet", "geommlp"]
A1_FIX_EPOCHS = 200          # == patience, i.e. early stopping never fires

# One registry for the no-early-stopping retrains, so the gate, the extraction
# joblist and the merge are one parameterised script each rather than one pair
# per task. A family is (base task, its five seeded task names, the encoders it
# retrains, the epoch budget that doubles as its patience).
FIX_FAMILIES = {
    A1_FIX_BASE: dict(tasks=A1_FIX_TASKS, encoders=A1_FIX_ENCODERS,
                      epochs=A1_FIX_EPOCHS, legacy_base="A1"),
    A2_FIX_BASE: dict(tasks=A2_FIX_TASKS, encoders=A2_FIX_ENCODERS,
                      epochs=A2_FIX_EPOCHS, legacy_base="A2"),
}


def fix_family(name):
    if name not in FIX_FAMILIES:
        raise SystemExit(f"unknown fix family {name!r}; "
                         f"have {sorted(FIX_FAMILIES)}")
    return FIX_FAMILIES[name]


# ── leaked scan-level arm of Fig. 5(a), retrained across training seeds ─────
# thread facebase3d-figs-2026-09-04. Fig. 5(a) compares a scan-level (leaky)
# split against the subject-disjoint split. Its corrected arm is B1/B2_corrected
# and already has five training seeds; its leaked arm was one unseeded
# checkpoint re-scored by rerun_scanlevel.py. These tasks retrain the leaked arm
# so both bars are five-model means.
#
# The manifests are the ordinary B1/B2 manifests with the fbid column dropped,
# frozen under data/processed/legacy_scanlevel/. dataset.py falls back to a
# scan-level stratified split when the subject-id column is absent, and that
# fallback reproduces rerun_scanlevel.legacy_split() row-for-row and in order
# for all six B1/B2 splits — checked before these were queued. No leaky split
# flag was added to dataset.py.
#
# NOTHING HERE IS TASK PERFORMANCE. ~79% (B1) and ~85% (B2) of test scans come
# from subjects also present in training. These numbers are only ever the
# inflated arm of the leakage contrast.
SCANLEVEL_DIR = PROJECT / "data/processed/legacy_scanlevel"
SCANLEVEL_BASE = {
    "B1_scanlevel": dict(manifest=SCANLEVEL_DIR / "B1_scanlevel_manifest.csv",
                         exp_type="syndrome", num_classes=19, binary_mode=False,
                         aux_dim=0, run_dir="B1_scanlevel", id_col=None,
                         note="19-class syndrome — LEAKED scan-level split"),
    "B2_scanlevel": dict(manifest=SCANLEVEL_DIR / "B2_scanlevel_manifest.csv",
                         exp_type="clinical", num_classes=33, binary_mode=False,
                         aux_dim=0, run_dir="B2_scanlevel", id_col=None,
                         note="33-class clinical — LEAKED scan-level split"),
}
for _b, _cfg in SCANLEVEL_BASE.items():
    TASKS[_b] = dict(_cfg)
    for _ts in TRAIN_SEEDS:
        _name = trainseed_task(_b, _ts)
        TASKS[_name] = dict(_cfg)
        TASKS[_name]["run_dir"] = _name
        TASKS[_name]["train_seed"] = _ts
        TASKS[_name]["base_task"] = _b
        TASKS[_name]["note"] = f"{_cfg['note']} — train seed {_ts}"
del _b, _cfg, _ts, _name

SCANLEVEL_BASE_TASKS = list(SCANLEVEL_BASE)
SCANLEVEL_TRAINSEED_TASKS = [trainseed_task(b, s)
                             for b in SCANLEVEL_BASE_TASKS for s in TRAIN_SEEDS]

# Execution staging — B1/B2 carry the core claim and go first.
PRIORITY_TASKS = ["B1_corrected", "B2_corrected",
                  "B1_proto_cetrain", "B2_proto_cetrain"]
SECONDARY_TASKS = ["A1", "A1_S1", "A1_S2", "A1_S3", "A2", "B2_r2", "C_corrected"]
XSITE_TASKS = [f"XS_{s}" for s in XSITE_SITES]
XSITE_BIN_TASKS = [f"XSB_{s}" for s in XSITE_SITES]

# The full head x level decision grid is only required for the B1/B2 core
# claim. A/C tasks are single-scan-level classification tasks.
GRID_TASKS = ["B1_corrected", "B2_corrected"]
HEADS = ["softmax", "proto", "knn11"]
LEVELS = ["scan", "subject"]
KNN_K = 11
N_BOOTSTRAP = 2000


# Top-k accuracy ladder (2026-08-31). Reported only for the
# many-class tasks: top-3 of a 4-class or binary task is not informative, and
# top-k is undefined for k >= n_classes.
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
    """Heads scored for a task. Defaults to the full set so the already-scored
    B1/B2 tasks are unaffected; A-family and C override it down to softmax,
    which is the only head those runs were ever published under."""
    return TASKS[task].get("heads", HEADS)


def levels_for(task):
    """Levels scored for a task. A task whose id_col is None has one synthetic
    id per scan, so a subject-level row would duplicate the scan-level row
    exactly — those tasks override this down to ["scan"]."""
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
