"""
Region occlusion for ONE trained PointNet++ (CF20_full_v3_ts{k}) on the scans of a list.

The paper's headline inference is prototype retrieval: class prototype = mean
TRAIN-scan 256-d embedding (score_grid.py), score = cosine to the prototype. The
occlusion therefore measures that score, not the softmax logit.

Occlusion: delete the points of one landmark region, refill to 4096 by
duplicating random remaining points, re-embed. Control: delete an equally
sized contiguous random patch (k nearest points around a random face point),
N_CTRL draws. Reported per region, per control draw and per FPS seed: cosine to
the true-diagnosis prototype and margin (true minus best other), base vs occluded.
occlusion.py turns these rows into importance = drop(region) - drop(control).

Requires the trained checkpoint (runs/) and the cached train-split features
(results/features/, from extract_features.py) for the prototypes, the landmark
files written by landmark_scans.py, and the v3 point clouds.

Usage: interp_compute.py --ts 1 --list results/interp/pilot_scans.csv \
           --lm-dir results/interp/landmarks --out-dir results/interp/compute
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
from facebench.models.pointnet2 import PointNet2                     # noqa: E402
from facebench.training import task_registry as C                            # noqa: E402
from analyses.interpretability.interp_lib import REGIONS, ALL_REGIONS, point_regions, abs_paths  # noqa: E402

N_FPS = 4                # occlusion is scored with the first N_FPS // 2 = 2 FPS seeds
N_CTRL = 3
NPTS = 4096


def gen(seed):
    g = torch.Generator(); g.manual_seed(seed); return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", type=int, required=True)
    ap.add_argument("--list", required=True)
    ap.add_argument("--lm-dir", default=str(PROJECT / "results/interp/landmarks"))
    ap.add_argument("--out-dir", default=str(PROJECT / "results/interp/compute"))
    a = ap.parse_args()
    task = f"CF20_full_v3_ts{a.ts}"
    torch.manual_seed(0)
    model = PointNet2(num_classes=49, aux_dim=0)
    model.load_state_dict(torch.load(C.checkpoint_path(task, "pointnet2"), map_location="cpu"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tr = np.load(C.feature_path(task, "pointnet2", 0, "train"), allow_pickle=True)
    names = list(tr["label_names"])
    emb, lab = tr["embed"].astype(np.float64), tr["labels"]
    protos = np.stack([emb[lab == c].mean(0) for c in range(len(names))])
    protos /= np.linalg.norm(protos, axis=1, keepdims=True)
    P = torch.tensor(protos, dtype=torch.float32)

    df = abs_paths(pd.read_csv(a.list))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    lm_dir = Path(a.lm_dir)
    rng = np.random.default_rng(1000 + a.ts)
    occ_rows = []

    def cos_all(x, g):
        e = model.embed(x, None, generator=g)
        return torch.nn.functional.normalize(e, dim=1) @ P.T     # (B, 49)

    for r in df.itertuples():
        lmf = lm_dir / f"{r.scan_id}.npz"
        if not lmf.exists():
            continue
        cloud = np.load(r.out_path).astype(np.float32)
        reg = point_regions(cloud.astype(float), np.load(lmf)["lm_norm"])
        y = names.index(r.syndrome_name)
        x0 = torch.from_numpy(cloud)[None]

        # ---- occlusion batch: base, each region, N_CTRL size-matched patches per region
        face = np.where(reg < len(REGIONS))[0]
        tree = None
        variants, meta = [cloud], [("base", -1, 0)]
        from scipy.spatial import cKDTree
        tree = cKDTree(cloud)
        for ri, rn in enumerate(ALL_REGIONS):
            drop = np.where(reg == ri)[0]
            if len(drop) == 0 or len(drop) > NPTS - 256:
                continue
            masks = [drop]
            for _ in range(N_CTRL):
                ctr = cloud[rng.choice(face)]
                masks.append(tree.query(ctr, k=len(drop))[1].ravel())
            for mi, m in enumerate(masks):
                keep = np.setdiff1d(np.arange(NPTS), m)
                fill = rng.choice(keep, NPTS - len(keep), replace=True)
                variants.append(cloud[np.r_[keep, fill]])
                meta.append((rn, mi, len(drop)))
        X = torch.from_numpy(np.stack(variants))
        with torch.no_grad():
            for s in range(N_FPS // 2):
                cs = cos_all(X, gen(s)).numpy()
                for (rn, mi, n), cv in zip(meta, cs):
                    oth = np.delete(cv, y)
                    occ_rows.append(dict(scan_id=r.scan_id, ts=a.ts, fps=s, region=rn,
                                         draw=mi, kind="base" if mi < 0 else
                                         ("region" if mi == 0 else "control"),
                                         n_drop=n, cos_true=cv[y],
                                         margin=cv[y] - oth.max(), pred=int(cv.argmax())))
        print(f"{r.scan_id} y={y}", flush=True)

    pd.DataFrame(occ_rows).to_csv(out / f"occ_ts{a.ts}.csv", index=False)


if __name__ == "__main__":
    main()
