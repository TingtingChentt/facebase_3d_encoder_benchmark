#!/usr/bin/env python3
"""Display-quality score for every scan of the 49-class syndrome task, used only to
choose which scan of a patient to draw in Fig. 1 (exemplar faces without holes or hair).

hole_frac : fraction of empty cells in a 20x20 frontal raster of the central face
            box (|x|<0.38, -0.50<y<0.50) — holes in the surface show up as empty cells.
top_spread: SD of depth (z) for points above y>0.55 — hair/scalp artefacts are
            ragged in depth, a clean forehead is smooth.
Display choice only; nothing here feeds an analysis.
Output: results/fig1/scan_quality.csv (local; keyed by scan)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root
from facebench.data.dataset import resolve_out_path  # noqa: E402
from facebench.paths import MANIFESTS, RESULTS  # noqa: E402

m = resolve_out_path(pd.read_csv(MANIFESTS / "syndrome_k20_manifest.csv"))
rows = []
for r in m.itertuples():
    X = np.load(r.out_path)
    b = (np.abs(X[:, 0]) < 0.38) & (X[:, 1] > -0.50) & (X[:, 1] < 0.50)
    H, _, _ = np.histogram2d(X[b, 0], X[b, 1], bins=20, range=[[-0.38, 0.38], [-0.5, 0.5]])
    t = X[X[:, 1] > 0.55]
    rows.append(dict(scan_id=r.scan_id, fbid=r.fbid, split=r.fixed_split,
                     syndrome=r.syndrome_name, out_path=r.out_path,
                     hole_frac=(H == 0).mean(), top_spread=t[:, 2].std() if len(t) > 20 else np.nan))
d = pd.DataFrame(rows)
(RESULTS / "fig1").mkdir(parents=True, exist_ok=True)
d.to_csv(RESULTS / "fig1/scan_quality.csv", index=False)
print(d.describe())
