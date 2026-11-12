"""
Case selection + nearest training neighbours for the pilot syndromes.
 Cached features/predictions only.

Selection rule (fixed before looking at any map):
  consistent_correct : subject-level prototype prediction correct in 5/5 training seeds
  consistent_wrong   : wrong in 5/5; the "confused-with" class is the modal prediction
  mixed              : everything else (not used for exemplars)
Nearest neighbours: subject-mean 256-d embedding (inference seed 0), cosine to
every TRAIN subject's mean embedding, averaged over the 5 training seeds (ts1-5);
top 3, with label, mean cosine and one scan.

Writes results/interp/cases.csv and nn_scans.csv (train scans to landmark).
--list/--tag run it on another scan list; e.g. --list ext_scans.csv --tag ext -> cases_ext.csv, nn_scans_ext.csv.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root
import argparse

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
from facebench.data.dataset import FaceBaseDataset, DATA_ROOT            # noqa: E402

R = PROJECT / "results"
OUT = PROJECT / "results/interp"
from facebench.paths import MANIFESTS  # noqa: E402
M = str(MANIFESTS / "syndrome_k20_manifest.csv")
K_NN = 3


def l2(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def subj_mean(emb, ids):
    u = pd.unique(ids)
    return u, np.stack([emb[ids == s].mean(0) for s in u])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default="pilot_scans.csv")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    pilot = pd.read_csv(OUT / a.list)
    names = None
    rows = []
    for ts in range(1, 6):
        d = np.load(R / f"predictions/CF20_full_v3_ts{ts}__pointnet2__proto__subject__seed0.npz",
                    allow_pickle=True)
        names = list(d["label_names"])
        rows.append(pd.DataFrame(dict(ts=ts, fbid=d["subject_of_unit"].astype(str),
                                      y=d["y_true"], p=d["y_pred"])))
    pr = pd.concat(rows)
    pr["ok"] = pr.y == pr.p
    g = pr.groupby("fbid").agg(n_correct=("ok", "sum"),
                               mode_pred=("p", lambda s: s.mode().iloc[0])).reset_index()
    g["category"] = np.where(g.n_correct == 5, "consistent_correct",
                             np.where(g.n_correct == 0, "consistent_wrong", "mixed"))
    g["mode_pred_name"] = [names[i] for i in g.mode_pred]

    tr_ds = FaceBaseDataset(M, exp="syndrome_name", split="train", val_frac=0.15,
                            test_frac=0.15, seed=42, augment=False)
    trdf = tr_ds.df.reset_index(drop=True)
    sims = []
    for ts in range(1, 6):
        ftr = np.load(R / f"features/CF20_full_v3_ts{ts}__pointnet2__seed0__train.npz", allow_pickle=True)
        fte = np.load(R / f"features/CF20_full_v3_ts{ts}__pointnet2__seed0__test.npz", allow_pickle=True)
        assert (trdf.fbid.astype(str).values == ftr["ids"].astype(str)).all()
        u_tr, tr_e = subj_mean(ftr["embed"].astype(float), ftr["ids"].astype(str))
        u_te, te_e = subj_mean(fte["embed"].astype(float), fte["ids"].astype(str))
        if ts == 1:
            tr_u, te_u = u_tr, u_te
        assert (u_tr == tr_u).all() and (u_te == te_u).all()
        sims.append(l2(te_e) @ l2(tr_e).T)
    sim = np.mean(sims, 0)                                   # cosine averaged over 5 models
    tr_lab = trdf.groupby("fbid").syndrome_name.first()
    tr_scan = trdf.groupby("fbid").first()

    cases, nn_scans = [], []
    for sid in pilot.fbid.astype(str).unique():
        i = list(te_u).index(sid)
        top = np.argsort(-sim[i])[:K_NN]
        rec = g[g.fbid == sid].iloc[0].to_dict()
        rec["syndrome"] = pilot[pilot.fbid.astype(str) == sid].syndrome_name.iloc[0]
        rec["scans"] = ";".join(pilot[pilot.fbid.astype(str) == sid].scan_id)
        for k, j in enumerate(top):
            nb = tr_u[j]
            rec[f"nn{k+1}_fbid"] = nb
            rec[f"nn{k+1}_label"] = tr_lab[nb]
            rec[f"nn{k+1}_cos"] = round(float(sim[i, j]), 4)
            rec[f"nn{k+1}_scan"] = tr_scan.loc[nb, "scan_id"]
            nn_scans.append(tr_scan.loc[nb, ["scan_id", "src_path", "out_path"]].to_dict()
                            | {"syndrome_name": tr_lab[nb], "fbid": nb})
        cases.append(rec)
    cases = pd.DataFrame(cases)
    cases.to_csv(OUT / f"cases{sfx}.csv", index=False)
    nn_df = pd.DataFrame(nn_scans).drop_duplicates("scan_id")
    # store out_path relative to the data root, like every other released list
    nn_df["out_path"] = nn_df.out_path.map(lambda p: str(Path(p).relative_to(DATA_ROOT)))
    nn_df.to_csv(OUT / f"nn_scans{sfx}.csv", index=False)
    print(cases.groupby(["syndrome", "category"]).size().unstack(fill_value=0))
    w = cases[cases.category == "consistent_wrong"]
    print(w.groupby(["syndrome", "mode_pred_name"]).size())
    print("NN label agrees with true:",
          (cases.nn1_label == cases.syndrome).groupby(cases.category).mean().round(2).to_dict())


if __name__ == "__main__":
    main()
