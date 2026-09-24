#!/usr/bin/env python3
"""
T4 — Build combined manifest for Exp C (binary Affected/Unaffected).

Sources and label mapping:
  FB-5A, FB-56 (OFC):
    Unaffected → Unaffected
    CLP / CL / CP → Affected
  FB-TJ0 (Syndrome):
    Control → Unaffected
    Uncategorized → EXCLUDED (ambiguous; not used in Exp C per 163000 message)
    All other Syndrome Categories → Affected
  FB-TK0, FB-TX4, FB-VWP (Controls):
    All scans → Unaffected

Output: data/processed/combined_manifest.csv
  Columns: scan_id, src_path, dataset, binary_label, out_path
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"


# ── OFC ──────────────────────────────────────────────────────────────────────
ofc = pd.read_csv(PROCESSED / "ofc_manifest.csv")

OFC_BINARY = {
    "Unaffected": "Unaffected",
    "Cleft Lip": "Affected",
    "Cleft Palate": "Affected",
    "Cleft Lip and Palate": "Affected",
}
ofc["binary_label"] = ofc["cleft_type"].map(OFC_BINARY)
ofc = ofc.dropna(subset=["binary_label"])
ofc_out = ofc[["scan_id", "src_path", "dataset", "binary_label", "out_path"]].copy()
print(f"OFC: {len(ofc_out)} scans")
print(f"  {ofc_out['binary_label'].value_counts().to_dict()}")


# ── Syndrome ─────────────────────────────────────────────────────────────────
syn = pd.read_csv(PROCESSED / "syndrome_manifest.csv")
syn["dataset"] = "FB-TJ0"

EXCLUDE_SYNDROME = {"Uncategorized"}
UNAFFECTED_SYNDROME = {"Control"}

def syndrome_binary(cat):
    if cat in EXCLUDE_SYNDROME:
        return None
    if cat in UNAFFECTED_SYNDROME:
        return "Unaffected"
    return "Affected"

syn["binary_label"] = syn["syndrome_category"].apply(syndrome_binary)
syn = syn.dropna(subset=["binary_label"])
syn = syn.rename(columns={"syndrome_category": "syndrome_category"})
syn_out = syn[["scan_id", "src_path", "dataset", "binary_label", "out_path"]].copy()
print(f"Syndrome (FB-TJ0): {len(syn_out)} scans (Uncategorized excluded)")
print(f"  {syn_out['binary_label'].value_counts().to_dict()}")


# ── Controls ─────────────────────────────────────────────────────────────────
ctrl = pd.read_csv(PROCESSED / "controls_manifest.csv")
ctrl_out = ctrl[["scan_id", "src_path", "dataset", "binary_label", "out_path"]].copy()
print(f"Controls: {len(ctrl_out)} scans")
print(f"  {ctrl_out['dataset'].value_counts().to_dict()}")


# ── Combine ───────────────────────────────────────────────────────────────────
combined = pd.concat([ofc_out, syn_out, ctrl_out], ignore_index=True)
out_path = PROCESSED / "combined_manifest.csv"
combined.to_csv(out_path, index=False)

print(f"\nCombined manifest: {len(combined)} total scans → {out_path}")
print(f"  Affected:   {(combined['binary_label']=='Affected').sum()}")
print(f"  Unaffected: {(combined['binary_label']=='Unaffected').sum()}")
print(f"  By dataset: {combined['dataset'].value_counts().to_dict()}")
