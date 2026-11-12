"""Filesystem locations used across the repository.

Everything is relative to the repository root unless overridden:

  FACEBENCH_DATA_ROOT  where processed point clouds (data/processed/...) and
                       trained checkpoints (runs/) live; default = repo root.
                       Point it at scratch storage for a full rebuild.
  FACEBASE_ROOT        root of your FaceBase download (the directory holding
                       FB-5A_files/, FB-56_files/, FB-TJ0_files/, ...).
                       Needed only by preprocessing, label building and the
                       figures that render faces.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("FACEBENCH_DATA_ROOT", ROOT))
FACEBASE_ROOT = Path(os.environ.get("FACEBASE_ROOT", "FACEBASE_ROOT_not_set"))

SPLITS = ROOT / "data" / "splits"          # shipped: scan/subject IDs + partition, no labels
MANIFESTS = ROOT / "data" / "manifests"    # built locally by build_manifests.py (labels; gitignored)
PROCESSED = DATA_ROOT / "data" / "processed"   # point clouds (.npy)
RUNS = DATA_ROOT / "runs"                  # checkpoints
RESULTS = ROOT / "results"                 # shipped aggregate tables + local outputs
FEATURES = RESULTS / "features"            # cached encoder outputs (local only)
PREDICTIONS = RESULTS / "predictions"      # per-unit predictions (local only)
MACROS = RESULTS / "paper_macros"          # LaTeX \newcommand blocks used in the paper
FIG_OUT = ROOT / "figures" / "output"


def facebase(*parts) -> Path:
    """Path inside the FaceBase download; fails loudly if FACEBASE_ROOT is unset."""
    if not FACEBASE_ROOT.exists():
        raise FileNotFoundError(
            "set FACEBASE_ROOT to your FaceBase download (see docs/DATA.md)")
    return FACEBASE_ROOT.joinpath(*parts)
