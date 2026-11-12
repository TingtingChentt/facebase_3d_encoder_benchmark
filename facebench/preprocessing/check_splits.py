#!/usr/bin/env python3
"""Check that the labelled manifests reproduce the shipped partitions exactly.

Run after build_manifests.py. For every task family the dataset class draws
train/val/test exactly as during training (stratified, seed 42, or the stored
`fixed_split` for the syndrome tasks) and the result is compared row for row
with data/splits/<task>.csv. Point clouds are not needed.

  python facebench/preprocessing/check_splits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import pandas as pd

from facebench.data.dataset import FaceBaseDataset
from facebench.paths import SPLITS
from facebench.training import task_registry as C


def drawn_split(cfg):
    parts = []
    for split in ("train", "val", "test"):
        ds = FaceBaseDataset(str(cfg["manifest"]), cfg["exp_type"], split=split,
                             seed=C.DATA_SPLIT_SEED, binary_mode=cfg["binary_mode"],
                             require_clouds=False)
        parts.append(pd.DataFrame({"scan_id": ds.df.scan_id, "split": split}))
    return pd.concat(parts).set_index("scan_id").split


def main():
    bad = 0
    for base, cfg in C.BASES.items():
        shipped = pd.read_csv(SPLITS / f"{base}.csv").set_index("scan_id").split
        drawn = drawn_split(cfg)
        same = set(drawn.index) == set(shipped.index) and (drawn[shipped.index] == shipped).all()
        n = shipped.value_counts()
        print(f"{base:14s} {len(shipped):6,d} scans  train {n['train']:5,d}  val {n['val']:5,d}  "
              f"test {n['test']:5,d}  {'OK' if same else 'MISMATCH'}")
        bad += not same
    if bad:
        sys.exit(f"{bad} task(s) do not reproduce the shipped split")
    print("all partitions reproduce data/splits/")


if __name__ == "__main__":
    main()
