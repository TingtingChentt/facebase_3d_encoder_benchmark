#!/usr/bin/env python3
"""
Clean and filter syndrome_clinical_manifest.csv.
Step 1: merge known duplicate/typo clinical diagnosis entries.
Step 2: drop categories with fewer than MIN_COUNT scans.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data/processed/syndrome_clinical_manifest.csv"
MIN_COUNT = 50

LABEL_FIXES = {
    "Costello":         "Costello Syndrome",
    "Marfans":          "Marfan Syndrome",
    "Williams Sydrome": "Williams Syndrome",
}

# Classes excluded on clinical review — not syndrome diagnoses
EXCLUDE_CLASSES = {"Non-relative control"}


def main():
    df = pd.read_csv(MANIFEST_PATH)
    original_n = len(df)
    print(f"Loaded {original_n} rows")

    # Step 1: normalize labels
    n_fixed = 0
    for wrong, right in LABEL_FIXES.items():
        mask = df["clinical_diagnosis"] == wrong
        n_fixed += mask.sum()
        df.loc[mask, "clinical_diagnosis"] = right
    print(f"Label fixes applied: {n_fixed} rows updated")

    # Step 2: exclude non-diagnosis classes
    df = df[~df["clinical_diagnosis"].isin(EXCLUDE_CLASSES)].reset_index(drop=True)
    print(f"After exclusions: {len(df)} rows")

    # Step 4: apply cutoff
    counts = df["clinical_diagnosis"].value_counts()
    keep_cats = counts[counts >= MIN_COUNT].index
    df_clean = df[df["clinical_diagnosis"].isin(keep_cats)].reset_index(drop=True)
    dropped_n = original_n - len(df_clean)
    print(f"Categories kept: {len(keep_cats)} (≥{MIN_COUNT} scans)")
    print(f"Scans retained: {len(df_clean)} / {original_n} (dropped {dropped_n})")

    print("\nFinal class distribution:")
    final_counts = df_clean["clinical_diagnosis"].value_counts()
    for cat, cnt in final_counts.items():
        print(f"  {cat:<50} {cnt}")

    df_clean.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest overwritten: {MANIFEST_PATH}")

    return df_clean, final_counts


if __name__ == "__main__":
    main()
