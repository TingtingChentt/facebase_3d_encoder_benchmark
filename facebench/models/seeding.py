"""
Training-seed control for the FaceBase 3D encoder experiments.

WHY THIS MODULE EXISTS
----------------------
Until 2026-08-31 nothing in this repo seeded the RNGs that decide a training
run's outcome. `train.py --seed` fed ONLY FaceBaseDataset, where it selects
train/val/test membership. Weight init, DataLoader shuffling and the
rotation/jitter augmentation all ran off unseeded global RNGs, so every
checkpoint in runs/ is one unreproducible draw from a distribution
nobody measured.

TWO SEEDS, NEVER CONFLATED — a third joins the two in rerun_config.py:
  * DATA SPLIT SEED (`--seed`, frozen at 42) — defines split membership.
    Varying it changes WHICH SUBJECTS ARE IN THE TEST SET, so results across
    different values are not comparable. Never varied.
  * TRAIN SEED (`--train-seed`) — weight init, batch order, augmentation
    draws. The split is untouched, so every seed is scored on the IDENTICAL
    test subjects and the spread is pure retraining variance. This is what
    the multi-seed sweep varies.
  * INFERENCE SEED (rerun_config.INFERENCE_SEEDS) — PointNet++ FPS start
    index at eval time, weights frozen.

THE num_workers TRAP
--------------------
FaceBaseDataset._augment draws from the GLOBAL np.random. DataLoader workers
are forked, so they inherit one numpy state and all four emit the IDENTICAL
augmentation stream — seeding the parent alone does not fix this, it freezes
it. worker_init_fn below gives each worker its own derived stream, which is
both reproducible and actually distinct per worker.

DETERMINISM IS NOT PROMISED, REPRODUCIBILITY OF THE DISTRIBUTION IS.
cudnn.benchmark is left ON: turning it off costs ~30% wall-clock across a
100-run sweep, and the goal here is to MEASURE run-to-run spread, not to
eliminate it. Two runs at the same train seed on the same GPU model land very
close but need not be bit-identical. Do not quote this module as "fully
deterministic training" in the manuscript.
"""

import os
import random

import numpy as np
import torch


def seed_everything(train_seed: int) -> None:
    """Seed every RNG that influences a training run's trajectory."""
    os.environ["PYTHONHASHSEED"] = str(train_seed)
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)


def make_generator(train_seed: int) -> torch.Generator:
    """Generator for DataLoader(shuffle=True) / WeightedRandomSampler, so batch
    order is a function of the train seed rather than of global RNG state that
    earlier code may have advanced."""
    g = torch.Generator()
    g.manual_seed(train_seed)
    return g


def make_worker_init_fn(train_seed: int):
    """Per-worker seeding for numpy and random — see THE num_workers TRAP."""
    def _init(worker_id: int) -> None:
        s = (train_seed + 1) * 10_000 + worker_id
        np.random.seed(s % (2 ** 32))
        random.seed(s)
    return _init


def loader_kwargs(train_seed) -> dict:
    """DataLoader kwargs implementing the seed. Returns {} when train_seed is
    None so the legacy unseeded path stays byte-for-byte what it was."""
    if train_seed is None:
        return {}
    return dict(generator=make_generator(train_seed),
                worker_init_fn=make_worker_init_fn(train_seed))
