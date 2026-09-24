"""
PyTorch Dataset for FaceBase 3D encoder experiments.
Reads from manifest CSV + .npy point cloud files.

PATHS. The released manifests store `out_path` relative to the repository root
(e.g. "data/processed/ofc/<scan>.npy"). resolve_out_path() below makes those
absolute against FACEBENCH_DATA_ROOT, defaulting to the repository root, so the
processed point clouds can live on a scratch filesystem without editing any
manifest. Absolute out_path values are passed through unchanged, which is what
the manifests held while the experiments in the paper were run.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_ROOT = Path(os.environ.get("FACEBENCH_DATA_ROOT",
                                Path(__file__).resolve().parents[2]))


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
    "ofc_a3": {"Cleft Lip": 0, "Cleft Palate": 1},
    "ofc_a4": {"Unaffected": 0, "Cleft Lip": 1, "Cleft Palate": 2},
    "syndrome": None,  # built dynamically from manifest
    "clinical": None,  # built dynamically from syndrome_clinical_manifest.csv
}

# Maps each label to its binary class (0=Unaffected, 1=Affected)
BINARY_COLLAPSE = {
    "ofc": {
        "Unaffected": "Unaffected",
        "Cleft Lip": "Affected",
        "Cleft Palate": "Affected",
        "Cleft Lip and Palate": "Affected",
    },
    "syndrome": None,  # built dynamically: Control→Unaffected, rest→Affected
}

LABEL_COLS = {
    "ofc": "cleft_type",
    "ofc_a3": "cleft_type",
    "ofc_a4": "cleft_type",
    "syndrome": "syndrome_category",
    "clinical": "clinical_diagnosis",
    "combined": "binary_label",  # Exp C — pre-computed Affected/Unaffected
}

# FB-TJ0 has ~2.9 scans/subject; split at subject level to prevent leakage.
# OFC (FB-5A) is ~1 scan/subject so scan-level split is fine.
SUBJECT_ID_COL = {
    "syndrome": "fbid",
    "clinical": "fbid",
    "ofc": None,
    "ofc_a3": None,
    "ofc_a4": None,
    "combined": "subject_id",
}


class FaceBaseDataset(Dataset):
    def __init__(self, manifest_csv: str, exp: str,
                 split: str = "train", val_frac: float = 0.15,
                 test_frac: float = 0.15, seed: int = 42,
                 augment: bool = False, label_map: dict = None,
                 binary_mode: bool = False, icp: bool = False,
                 icp_v2: bool = False, holdout_site: str = None):
        """
        Args:
            manifest_csv: path to manifest CSV
            exp: "ofc", "syndrome", or "clinical"
            split: "train", "val", or "test"
            augment: random rotation + jitter for training
            label_map: override label→int mapping (ignored when binary_mode=True)
            binary_mode: collapse to 2-class Affected/Unaffected
            holdout_site: leave-one-site-out mode. When set, TEST is exactly the
                scans from that site and train/val are a stratified split over
                the remaining sites — test_frac is ignored, because the test set
                is defined by the site, not by a fraction. None (the default)
                leaves the random-split behaviour byte-for-byte unchanged.
        """
        df = resolve_out_path(pd.read_csv(manifest_csv))
        label_col = LABEL_COLS[exp]

        # Redirect to ICP-registered data directories when requested
        if icp_v2:
            df["out_path"] = df["out_path"].str.replace(
                "/data/processed/syndrome/", "/data/processed/syndrome_icp_v2/",
                regex=False
            ).str.replace(
                "/data/processed/ofc/", "/data/processed/ofc_icp_v2/",
                regex=False
            )
        elif icp:
            df["out_path"] = df["out_path"].str.replace(
                "/data/processed/syndrome/", "/data/processed/syndrome_icp/",
                regex=False
            ).str.replace(
                "/data/processed/ofc/", "/data/processed/ofc_icp/",
                regex=False
            )

        # Drop rows where processed file doesn't exist
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
                # syndrome / clinical: Control→Unaffected, everything else→Affected
                df[label_col] = df[label_col].apply(
                    lambda x: "Unaffected" if x == "Control" else "Affected"
                )
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
        if holdout_site is not None:
            # ── Leave-one-site-out ────────────────────────────────────────────
            # TEST is the held-out site in full; train/val are a stratified
            # split over every remaining site. test_frac is deliberately unused:
            # the test set is a site, not a fraction.
            if "site" not in df.columns:
                raise ValueError(
                    f"holdout_site={holdout_site!r} requested but manifest "
                    f"{manifest_csv} has no 'site' column")
            site = df["site"].astype(str)
            present = sorted(site.unique())
            if holdout_site not in present:
                raise ValueError(
                    f"holdout_site={holdout_site!r} not in manifest sites {present}")

            is_held = (site == holdout_site).values
            if not is_held.any():
                raise ValueError(f"site {holdout_site!r} has 0 usable scans")
            if is_held.all():
                raise ValueError(f"site {holdout_site!r} is the entire manifest")

            # A subject with scans at both the held-out site and elsewhere would
            # leak across the fold boundary. Checked rather than assumed — this
            # is the whole premise of a site split.
            if subject_id_col and subject_id_col in df.columns:
                spanning = df.loc[is_held, subject_id_col].isin(
                    df.loc[~is_held, subject_id_col]).sum()
                if spanning:
                    raise ValueError(
                        f"{spanning} scans belong to subjects present both at "
                        f"held-out site {holdout_site!r} and in the training "
                        f"pool — site split would leak")

            test_idx = df.index[is_held].tolist()
            rng = np.random.default_rng(seed)
            train_idx, val_idx = [], []

            if subject_id_col and subject_id_col in df.columns:
                # Group train/val by subject so repeat scans stay together.
                rest = df[~is_held]
                consistent = rest.groupby(subject_id_col)[label_col].nunique() == 1
                rest = rest[rest[subject_id_col].isin(consistent[consistent].index)]
                subj_label = rest.drop_duplicates(subject_id_col)
                for cls in subj_label[label_col].unique():
                    subjs = subj_label.loc[
                        subj_label[label_col] == cls, subject_id_col].tolist()
                    rng.shuffle(subjs)
                    n_val = max(1, int(len(subjs) * val_frac))
                    val_s, train_s = set(subjs[:n_val]), set(subjs[n_val:])
                    val_idx.extend(rest.index[rest[subject_id_col].isin(val_s)].tolist())
                    train_idx.extend(rest.index[rest[subject_id_col].isin(train_s)].tolist())
            else:
                for cls in df.loc[~is_held, label_col].unique():
                    idx = df.index[(df[label_col] == cls) & (~is_held)].tolist()
                    rng.shuffle(idx)
                    n_val = max(1, int(len(idx) * val_frac))
                    val_idx.extend(idx[:n_val])
                    train_idx.extend(idx[n_val:])

            split_idx = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
            self.df = df.loc[split_idx].reset_index(drop=True)
            if "age_months" in df.columns:
                train_ages = df.loc[train_idx, "age_months"]
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
            # Scan-level stratified split (OFC: ~1 scan/subject; combined).
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
        self.holdout_site = holdout_site

    def class_counts(self) -> dict:
        """Per-class scan counts in THIS split, in label_map order. Used by the
        cross-site smoke check to show what a held-out site actually contains
        after preprocessing attrition, which the manifest count does not."""
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
