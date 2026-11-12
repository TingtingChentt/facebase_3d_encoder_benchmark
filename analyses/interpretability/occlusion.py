#!/usr/bin/env python3
"""
Region-importance tables from the raw occlusion scores written by interp_compute.py.

For every scan x trained model x anatomical region:
  drop_region   = cos(embed(x), true prototype) - cos(embed(x without region), true prototype)
  drop_control  = the same for a size-matched random facial patch (mean of N_CTRL=3 patches)
  importance    = drop_region - drop_control          (> 0: the region matters more than a
                                                        random patch of the same size)
Scores are averaged over the two PointNet++ FPS inference seeds stored in occ_ts*.csv.

The analysis was run in two batches over the same five models (CF20_full_v3_ts1-5):
  pilot : results/interp/pilot_scans.csv  -> compute/occ_ts*.csv     -> pilot/region_occlusion.csv
  ext   : results/interp/ext_scans.csv    -> compute_ext/occ_ts*.csv -> ext/region_occlusion.csv
The figures and macros concatenate the two tables.

Usage: python3 analyses/interpretability/occlusion.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import numpy as np
import pandas as pd

from analyses.interpretability import render_lib as RL                                             # noqa: E402
from analyses.interpretability.interp_lib import abs_paths                                    # noqa: E402

PROJECT = Path(__file__).resolve().parents[2]
IN = PROJECT / "results/interp"
TS = [1, 2, 3, 4, 5]
BATCHES = [("pilot_scans.csv", "compute", "pilot"), ("ext_scans.csv", "compute_ext", "ext")]


def region_tables(occ, scans):
    """Per scan x model x region importance (see module docstring)."""
    syn = scans.set_index("scan_id").syndrome_name
    o = occ[occ.kind != "base"].groupby(["scan_id", "ts", "region", "kind"]).cos_true.mean() \
        .unstack().rename(columns={"region": "cos_region", "control": "cos_control"})
    base = occ[occ.kind == "base"].groupby(["scan_id", "ts"]).cos_true.mean()
    o = o.reset_index()
    o["base"] = o.set_index(["scan_id", "ts"]).index.map(base)
    o["drop_region"] = o.base - o.cos_region
    o["drop_control"] = o.base - o.cos_control
    o["importance"] = o.drop_region - o.drop_control
    ctrl_max = occ[occ.kind == "control"].groupby(["scan_id", "ts", "region", "fps"]).cos_true.min()
    reg_val = occ[occ.kind == "region"].set_index(["scan_id", "ts", "region", "fps"]).cos_true
    beats = (reg_val < ctrl_max.reindex(reg_val.index)).groupby(["scan_id", "ts", "region"]).mean()
    o = o.merge(beats.rename("beats_all_controls").reset_index(), on=["scan_id", "ts", "region"])
    o["syndrome"] = o.scan_id.map(syn)
    return o


def scan_render(scan_id, row, lm_dir, scalar=None, cmap=None, img=360):
    lmf = lm_dir / f"{scan_id}.npz"
    d = np.load(lmf)
    V, F = RL.load_ply(row["src_path"])
    Vn = (V - d["c"]) / d["s"]
    R = RL.view_matrix(bool(d["view"][0]))
    crop = RL.face_crop(d["lm_norm"], R)
    sv = None
    if scalar is not None:
        cloud = np.load(row["out_path"])
        # display-only smoothing over the 16 nearest input points; stats use raw values
        from scipy.spatial import cKDTree
        _, j = cKDTree(cloud).query(cloud, k=16)
        sv = RL.to_vertices(cloud, scalar[j].mean(1), Vn)
    return RL.render_scalar(Vn, F, R, crop, sv, img=img, cmap=cmap)


def main():
    for lst, comp, out in BATCHES:
        scans = abs_paths(pd.read_csv(IN / lst))
        occ = pd.concat(pd.read_csv(IN / comp / f"occ_ts{ts}.csv") for ts in TS)
        o = region_tables(occ, scans)
        (IN / out).mkdir(exist_ok=True)
        o.to_csv(IN / out / "region_occlusion.csv", index=False)
        print(f"{out}/region_occlusion.csv: {len(o)} rows, {o.scan_id.nunique()} scans")


if __name__ == "__main__":
    main()
