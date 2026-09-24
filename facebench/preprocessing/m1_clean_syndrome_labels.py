#!/usr/bin/env python3
"""
Clean and filter syndrome_manifest.csv.
Step 1: fix known label inconsistencies.
Step 2: drop categories with fewer than MIN_COUNT scans.
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data/processed"
MANIFEST_PATH = PROCESSED / "syndrome_manifest.csv"
LOG_PATH = PROCESSED / "syndrome_processing_log.csv"
MIN_COUNT = 175

LABEL_FIXES = {
    "cartilate": "Cartilage",
    "Muscular dystrophy": "Muscular Dystrophy",
    "Lysozome": "Lysosome",
}


def main():
    df = pd.read_csv(MANIFEST_PATH)
    original_n = len(df)
    print(f"Loaded {original_n} rows from {MANIFEST_PATH.name}")

    # Step 1: normalize labels
    n_fixed = 0
    for wrong, right in LABEL_FIXES.items():
        mask = df["syndrome_category"] == wrong
        n_fixed += mask.sum()
        df.loc[mask, "syndrome_category"] = right
    print(f"Label fixes applied: {n_fixed} rows updated")

    # Step 2: apply cutoff
    counts = df["syndrome_category"].value_counts()
    keep_cats = counts[counts >= MIN_COUNT].index
    df_clean = df[df["syndrome_category"].isin(keep_cats)].reset_index(drop=True)
    dropped_n = original_n - len(df_clean)
    print(f"Categories kept: {len(keep_cats)} (≥{MIN_COUNT} scans)")
    print(f"Scans retained: {len(df_clean)} / {original_n} (dropped {dropped_n})")

    # Step 3: print final distribution
    print("\nFinal class distribution:")
    final_counts = df_clean["syndrome_category"].value_counts()
    for cat, cnt in final_counts.items():
        print(f"  {cat:<25} {cnt}")

    # Overwrite manifest
    df_clean.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest overwritten: {MANIFEST_PATH}")

    # Update processing log if it exists
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        log_clean = log[log["scan_id"].isin(df_clean["scan_id"])].reset_index(drop=True)
        log_clean.to_csv(LOG_PATH, index=False)
        print(f"Processing log updated: {len(log_clean)} rows (was {len(log)})")

    return df_clean, final_counts


if __name__ == "__main__":
    main()
