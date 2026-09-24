#!/usr/bin/env python3
"""
M2 Control Preprocessing — FaceBase 3D Encoder
Preprocesses OBJ scans from three normative control datasets:
  FB-TK0  (686 OBJ in FB-TK0_files/)
  FB-TX4  (3605 OBJ in FB-TX4_files/Spritz/Images/)
  FB-VWP  (2454 OBJ in FB-VWP_files/Images/)
All scans receive label = 'Unaffected' in the combined Exp C manifest.
Reuses load/sample/normalize logic from preprocess_v2.py.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Reuse preprocessing functions from preprocess_v2.py
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_v2 import process_scan, run_experiment  # noqa: E402

FACEBASE_ROOT = Path(os.environ.get("FACEBASE_ROOT", "/path/to/facebase"))
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CONTROLS_DIR = PROCESSED_ROOT / "controls"


# ── Manifest builders ────────────────────────────────────────────────────────

def build_fbtk0_manifest() -> pd.DataFrame:
    """686 OBJ files from FB-TK0_files/ (original quality, GWAS controls)."""
    src_dir = FACEBASE_ROOT / "FB-TK0_files"
    rows = []
    for obj in sorted(src_dir.glob("*.obj")):
        stem = obj.stem
        scan_id = f"fbtk0_{stem}"
        rows.append({
            "scan_id": scan_id,
            "src_path": str(obj),
            "dataset": "FB-TK0",
            "binary_label": "Unaffected",
            "out_path": str(CONTROLS_DIR / f"{scan_id}.npy"),
        })
    return pd.DataFrame(rows)


def build_fbtx4_manifest() -> pd.DataFrame:
    """3605 OBJ files from FB-TX4_files/Spritz/Images/ (Spritz normative)."""
    src_dir = FACEBASE_ROOT / "FB-TX4_files" / "Spritz" / "Images"
    rows = []
    for obj in sorted(src_dir.glob("*.obj")):
        stem = obj.stem
        scan_id = f"fbtx4_{stem}"
        rows.append({
            "scan_id": scan_id,
            "src_path": str(obj),
            "dataset": "FB-TX4",
            "binary_label": "Unaffected",
            "out_path": str(CONTROLS_DIR / f"{scan_id}.npy"),
        })
    return pd.DataFrame(rows)


def build_fbvwp_manifest() -> pd.DataFrame:
    """2454 OBJ files from FB-VWP_files/Images/ (Weinberg TDFN normative)."""
    src_dir = FACEBASE_ROOT / "FB-VWP_files" / "Images"
    rows = []
    for obj in sorted(src_dir.glob("*.obj")):
        stem = obj.stem
        scan_id = f"fbvwp_{stem}"
        rows.append({
            "scan_id": scan_id,
            "src_path": str(obj),
            "dataset": "FB-VWP",
            "binary_label": "Unaffected",
            "out_path": str(CONTROLS_DIR / f"{scan_id}.npy"),
        })
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fbtk0", "fbtx4", "fbvwp", "all"],
                        default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--manifest-only", action="store_true",
                        help="Build and save manifests only, no processing")
    args = parser.parse_args()

    CONTROLS_DIR.mkdir(parents=True, exist_ok=True)
    log_csv = PROCESSED_ROOT / "controls_processing_log.csv"

    builders = {
        "fbtk0": build_fbtk0_manifest,
        "fbtx4": build_fbtx4_manifest,
        "fbvwp": build_fbvwp_manifest,
    }

    datasets = list(builders.keys()) if args.dataset == "all" else [args.dataset]
    all_rows = []
    for ds in datasets:
        print(f"Building {ds} manifest...")
        df = builders[ds]()
        print(f"  {ds}: {len(df)} scans")
        all_rows.append(df)

    manifest = pd.concat(all_rows, ignore_index=True)
    manifest_path = PROCESSED_ROOT / "controls_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Controls manifest: {len(manifest)} scans → {manifest_path}")

    if args.manifest_only:
        print("--manifest-only: skipping preprocessing")
        sys.exit(0)

    # Preprocess using run_experiment from preprocess_v2
    run_experiment(
        exp="controls",
        manifest=manifest,
        out_csv=log_csv,
        dry_run=args.dry_run,
        worker_id=args.worker_id,
        n_workers=args.n_workers,
    )
    print("Done.")
