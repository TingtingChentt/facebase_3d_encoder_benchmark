"""Exemplar subjects shown in the figures that render faces.

Which FaceBase subject or scan appears in a figure is read from
data/exemplars.csv (columns: figure, key, id). That file is NOT distributed:
together with the figure labels it would link FaceBase IDs to diagnoses, and
the repository ships subject-level partitions without labels only.

The face-rendering figures (Fig. 1, Fig. 5, Supplementary Figs. S1 and the
interpretability supplements) need the FaceBase meshes in any case. To render
them, create data/exemplars.csv with your own choice of subjects (see
data/exemplars.example.csv for the keys each figure uses). No reported number
depends on which exemplar is shown.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from facebench.paths import ROOT  # noqa: E402

FILE = ROOT / "data" / "exemplars.csv"


def exemplar(figure: str, key: str) -> str:
    if not FILE.exists():
        raise FileNotFoundError(
            f"{FILE} not found. Face-rendering figures need a local exemplar list; "
            "see figures/exemplars.py and data/exemplars.example.csv.")
    t = pd.read_csv(FILE, dtype=str)
    hit = t[(t.figure == figure) & (t.key == key)]
    if len(hit) != 1:
        raise KeyError(f"data/exemplars.csv: need exactly one row for figure={figure!r}, key={key!r}")
    return hit.id.iloc[0]
