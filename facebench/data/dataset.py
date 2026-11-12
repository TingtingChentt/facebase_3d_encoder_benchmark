"""
PyTorch Dataset for FaceBase 3D encoder experiments.
Reads from manifest CSV + .npy point cloud files.

PATHS. Manifests (built by facebench/preprocessing/build_manifests.py) store
`out_path` relative, e.g. "data/processed/ofc/<scan>.npy". resolve_out_path()
makes it absolute against FACEBENCH_DATA_ROOT (default: the repository root),
so the point clouds can live on scratch storage without editing any manifest.

Rows whose point cloud does not exist are dropped BEFORE the split is drawn, so
a manifest row with no cloud and an absent row give identical splits.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
from facebench.paths import DATA_ROOT  # noqa: E402


def resolve_out_path(df):
    """Make the manifest's out_path column absolute. Idempotent."""
    df = df.copy()
    df["out_path"] = df["out_path"].astype(str).apply(
        lambda p: p if os.path.isabs(p) else str(DATA_ROOT / p))
    return df


LABEL_MAPS = {
    "ofc": {
        "Unaffected": 0,
        "Cleft Lip": 1,
        "Cleft Palate": 2,
        "Cleft Lip and Palate": 3,
    },
    "syndrome_name": None,  # built from the manifest (sorted class names)
}

# Maps each label to its binary class (0=Unaffected, 1=Affected)
BINARY_COLLAPSE = {
    "ofc": {
        "Unaffected": "Unaffected",
        "Cleft Lip": "Affected",
        "Cleft Palate": "Affected",
        "Cleft Lip and Palate": "Affected",
    },
}

LABEL_COLS = {
    "ofc": "cleft_type",              # cleft subtyping / cleft screening
    "syndrome_name": "syndrome_name",  # syndrome diagnosis (FB-TJ0 `Syndrome` field, curated)
    "combined": "binary_label",        # combined screening — Affected / Unaffected
}

# FB-TJ0 has ~2.9 scans/subject; split at subject level to prevent leakage.
# OFC (FB-5A) is ~1 scan/subject so scan-level split is fine.
SUBJECT_ID_COL = {
    "syndrome_name": "fbid",
    "ofc": None,
    "combined": "subject_id",
}


class FaceBaseDataset(Dataset):
    def __init__(self, manifest_csv: str, exp: str,
                 split: str = "train", val_frac: float = 0.15,
                 test_frac: float = 0.15, seed: int = 42,
                 augment: bool = False, label_map: dict = None,
                 binary_mode: bool = False, require_clouds: bool = True):
        """
        Args:
            manifest_csv: path to manifest CSV
            exp: "ofc", "combined" or "syndrome_name"
            split: "train", "val", or "test"
            augment: random rotation + jitter for training
            label_map: override label→int mapping (ignored when binary_mode=True)
            binary_mode: collapse to 2-class Affected/Unaffected
            require_clouds: drop rows whose point cloud is missing before the
                split is drawn (always True for training/inference). False is
                for checking the partitions without the clouds; on a complete
                preprocessing run both give the same split.
        """
        df = resolve_out_path(pd.read_csv(manifest_csv))
        label_col = LABEL_COLS[exp]

        # Drop rows where processed file doesn't exist
        if require_clouds:
            df = df[df["out_path"].apply(lambda p: os.path.exists(p))]
        df = df.reset_index(drop=True)

        if binary_mode:
            # Collapse multi-class labels to Affected / Unaffected
            if exp == "ofc":
                collapse = BINARY_COLLAPSE["ofc"]
                df = df[df[label_col].isin(collapse)].copy()
                df[label_col] = df[label_col].map(collapse)
            elif exp == "combined":
                pass  # already Affected/Unaffected — no collapse needed
            else:
                raise ValueError(f"binary_mode is not defined for exp={exp!r}")
            self.label_map = {"Unaffected": 0, "Affected": 1}
        elif label_map is not None:
            self.label_map = label_map
        elif LABEL_MAPS.get(exp) is not None:
            self.label_map = LABEL_MAPS[exp]
        else:
            classes = sorted(df[label_col].unique())
            self.label_map = {c: i for i, c in enumerate(classes)}

        df = df[df[label_col].isin(self.label_map)].reset_index(drop=True)

        subject_id_col = SUBJECT_ID_COL.get(exp)
        if "fixed_split" in df.columns:
            # Split stored in the manifest (syndrome tasks): one subject split
            # per class, shared by all seven class-size cutoffs so their test
            # sets are nested (facebench/preprocessing/labels/).
            for s in ("train", "val", "test"):
                if not (df["fixed_split"] == s).any():
                    raise ValueError(f"fixed_split: no '{s}' rows in {manifest_csv}")
            if subject_id_col and subject_id_col in df.columns:
                if (df.groupby(subject_id_col)["fixed_split"].nunique() > 1).any():
                    raise ValueError(f"fixed_split: a subject spans splits in {manifest_csv}")
            self.df = df[df["fixed_split"] == split].reset_index(drop=True)
            if "age_months" in df.columns:
                train_ages = df.loc[df["fixed_split"] == "train", "age_months"]
                self.age_mean = float(train_ages.mean())
                self.age_std = float(train_ages.std()) or 1.0
            else:
                self.age_mean, self.age_std = 0.0, 1.0
        elif subject_id_col and subject_id_col in df.columns:
            # Subject-level stratified split: all scans for a subject go to one split.
            # Drop the rare subjects with inconsistent labels across their scans.
            consistent = df.groupby(subject_id_col)[label_col].nunique() == 1
            df = df[df[subject_id_col].isin(consistent[consistent].index)].reset_index(drop=True)

            subj_label = df.drop_duplicates(subject_id_col)[[subject_id_col, label_col]].reset_index(drop=True)
            rng = np.random.default_rng(seed)
            train_subj, val_subj, test_subj = [], [], []
            for cls in subj_label[label_col].unique():
                subjs = subj_label.loc[subj_label[label_col] == cls, subject_id_col].tolist()
                rng.shuffle(subjs)
                n = len(subjs)
                n_test = max(1, int(n * test_frac))
                n_val = max(1, int(n * val_frac))
                test_subj.extend(subjs[:n_test])
                val_subj.extend(subjs[n_test:n_test + n_val])
                train_subj.extend(subjs[n_test + n_val:])

            subj_set = {"train": set(train_subj), "val": set(val_subj), "test": set(test_subj)}[split]
            self.df = df[df[subject_id_col].isin(subj_set)].reset_index(drop=True)
            # Age normalization stats from train subjects
            if "age_months" in df.columns:
                train_ages = df.loc[df[subject_id_col].isin(set(train_subj)), "age_months"]
                self.age_mean = float(train_ages.mean())
                self.age_std = float(train_ages.std()) or 1.0
            else:
                self.age_mean, self.age_std = 0.0, 1.0
        else:
            # Scan-level stratified split (cleft tasks: the cleft manifests carry
            # no subject column; one subject in 6,872 has two scans).
            rng = np.random.default_rng(seed)
            train_idx, val_idx, test_idx = [], [], []
            for cls in df[label_col].unique():
                idx = df.index[df[label_col] == cls].tolist()
                rng.shuffle(idx)
                n = len(idx)
                n_test = max(1, int(n * test_frac))
                n_val = max(1, int(n * val_frac))
                test_idx.extend(idx[:n_test])
                val_idx.extend(idx[n_test:n_test + n_val])
                train_idx.extend(idx[n_test + n_val:])

            split_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
            self.df = df.iloc[split_idx].reset_index(drop=True)
            if "age_months" in df.columns:
                train_ages = df.iloc[train_idx]["age_months"]
                self.age_mean = float(train_ages.mean())
                self.age_std = float(train_ages.std()) or 1.0
            else:
                self.age_mean, self.age_std = 0.0, 1.0
        self.label_col = label_col
        self.binary_mode = binary_mode
        self.augment = augment and split == "train"
        self.num_classes = len(self.label_map)

    def class_counts(self) -> dict:
        """Per-class scan counts in THIS split, in label_map order."""
        vc = self.df[self.label_col].value_counts()
        return {c: int(vc.get(c, 0))
                for c, _ in sorted(self.label_map.items(), key=lambda x: x[1])}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pts = np.load(row["out_path"]).astype(np.float32)  # (4096, 3)

        if self.augment:
            pts = self._augment(pts)

        # Auxiliary covariates: [age_norm, sex_bin]. Zeros if columns absent.
        aux = np.zeros(2, dtype=np.float32)
        if "age_months" in row.index and not pd.isna(row["age_months"]):
            aux[0] = (float(row["age_months"]) - self.age_mean) / (self.age_std + 1e-6)
        if "sex" in row.index and not pd.isna(row["sex"]):
            val = row["sex"]
            # Numeric encoding (0=Male, 1=Female) stored directly in manifest
            try:
                aux[1] = float(val)
            except (ValueError, TypeError):
                aux[1] = 1.0 if str(val).strip().upper() == "M" else 0.0
        else:
            aux[1] = 0.5  # unknown

        label = self.label_map[row[self.label_col]]
        return (torch.from_numpy(pts),
                torch.from_numpy(aux),
                torch.tensor(label, dtype=torch.long))

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        # Random rotation around Y axis
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        pts = pts @ R.T
        # Random jitter
        pts += np.random.normal(0, 0.01, pts.shape).astype(np.float32)
        pts = np.clip(pts, -1, 1)
        return pts

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for weighted cross-entropy."""
        counts = self.df[self.label_col].value_counts()
        total = len(self.df)
        weights = torch.zeros(self.num_classes)
        for cls, idx in self.label_map.items():
            n = counts.get(cls, 1)
            weights[idx] = total / (self.num_classes * n)
        return weights
