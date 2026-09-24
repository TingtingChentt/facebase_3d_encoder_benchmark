#!/usr/bin/env python3
"""
Re-evaluate the PRE-CORRECTION scan-level split under the paper's own protocol.
thread facebase3d-paper-2026-08-08.

WHY. Sec. 2.7 contrasts a scan-level split against the subject-disjoint split.
The subject-disjoint arm is a five-inference-seed mean with a bootstrap interval;
the scan-level arm was a single unseeded run carried over from a pre-correction
experiment log, marked \prov in numbers.tex and drawn hatched in Fig. 4(a) so a
reader would not read the two as equally evidenced. That asymmetry is the last
thing in the subsection that is weaker than the claim resting on it.

WHAT THIS DOES. Rebuilds the legacy scan-level split, loads the ORIGINAL
checkpoints (runs/{B1,B2,C}), and evaluates them with the same five
inference seeds and the same bootstrap the corrected arm uses, so the two arms
become like-for-like.

THE SPLIT IS RECOVERED, NOT REIMPLEMENTED. dataset.py always groups by subject
now, so it cannot produce the old split. legacy_split() below is the split code
as it stood at commit 1d4ec45, before subject grouping was introduced. It is
kept here rather than added to dataset.py as a flag: it exists only to score
superseded checkpoints, and nothing new should ever be trained with it.

VERIFIED BEFORE USE. Scoring the old B2 checkpoint on the rebuilt split returns
accuracy 0.5075 against the published \leakDxBefore of 0.509 -- i.e. the split
is the one those checkpoints were trained under. Without that check the whole
comparison would be meaningless, because evaluating an old checkpoint on a
DIFFERENT split would score it on scans it may have trained on.

TRAINING-SEED ARM (added 2026-09-04, thread facebase3d-figs-2026-09-04).
--train-seed k scores the RETRAINED leaky checkpoint from
runs/{key}_scanlevel_ts{k}/ instead of the original one. Those were
trained on data/processed/legacy_scanlevel/, the same manifests with fbid
dropped, which sends dataset.py down its scan-level fallback. That fallback
reproduces legacy_split() below row-for-row and IN ORDER for train/val/test on
both B1 and B2 -- re-checked before this arm was scored -- so the split object
built here is the split those checkpoints were trained under, and the fbid
column it carries (the frozen manifests do not have one) is what makes the
subject bootstrap possible at all.

Emits results/scanlevel_{per_seed,summary_ci,leakage}.csv
"""
import argparse, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "facebench/models"))
sys.path.insert(0, str(PROJECT / "facebench/eval"))
from dataset import LABEL_COLS, LABEL_MAPS, resolve_out_path  # noqa: E402
from pointnet2 import PointNet2                     # noqa: E402
import rerun_config as C                            # noqa: E402

DATA = PROJECT / "data/processed"
OUT = PROJECT / "results"
N_BOOT = 2000

# The three pre-correction runs, with the manifest and label space each of their
# checkpoints was actually trained on. Class counts were read back off the
# checkpoints (fc3.weight) rather than assumed: B1 19, B2 33, C 2.
LEGACY = {
    "B1": dict(manifest=DATA / "syndrome_manifest_b1.csv", exp="syndrome",
               binary=False, num_classes=19, id_col="fbid",
               run_dir="B1", ckpt="PointNet2_B1_best.pt",
               corrected="B1_corrected", note="19-class syndrome category"),
    "B2": dict(manifest=DATA / "syndrome_clinical_manifest.csv", exp="clinical",
               binary=False, num_classes=33, id_col="fbid",
               run_dir="B2", ckpt="PointNet2_B2_best.pt",
               corrected="B2_corrected", note="33-class clinical diagnosis"),
    "C":  dict(manifest=DATA / "combined_manifest.csv", exp="combined",
               binary=True, num_classes=2, id_col="subject_id",
               run_dir="C", ckpt="PointNet2_C_best.pt",
               corrected="C_corrected", note="combined binary screening"),
}


def legacy_split(manifest, exp, split, binary=False,
                 val_frac=0.15, test_frac=0.15, seed=42):
    """Stratified split over SCANS -- dataset.py as of commit 1d4ec45.

    Reproduced verbatim, including two details that change the result if
    'tidied': the classes are iterated in df[label_col].unique() order, which
    fixes the RNG consumption order, and rows are taken with .iloc on positional
    indices. Either change silently yields a different split.
    """
    df = resolve_out_path(pd.read_csv(manifest))
    label_col = LABEL_COLS[exp]
    df = df[df["out_path"].apply(os.path.exists)].reset_index(drop=True)
    if binary and exp != "combined":
        df[label_col] = df[label_col].apply(
            lambda x: "Unaffected" if x == "Control" else "Affected")
        label_map = {"Unaffected": 0, "Affected": 1}
    elif binary:
        label_map = {"Unaffected": 0, "Affected": 1}
    else:
        label_map = LABEL_MAPS.get(exp) or {
            c: i for i, c in enumerate(sorted(df[label_col].unique()))}
    df = df[df[label_col].isin(label_map)].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for cls in df[label_col].unique():
        idx = df.index[df[label_col] == cls].tolist()
        rng.shuffle(idx)
        n = len(idx)
        n_test, n_val = max(1, int(n * test_frac)), max(1, int(n * val_frac))
        te.extend(idx[:n_test])
        va.extend(idx[n_test:n_test + n_val])
        tr.extend(idx[n_test + n_val:])
    sel = {"train": tr, "val": va, "test": te}[split]
    return df.iloc[sel].reset_index(drop=True), label_map, label_col


class ScanDS(Dataset):
    def __init__(self, frame, label_map, label_col):
        self.df, self.lm, self.lc = frame, label_map, label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pts = np.load(r["out_path"]).astype(np.float32)
        return torch.from_numpy(pts), self.lm[r[self.lc]]


def make_gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


# ── metrics, matching facebench/eval/analyze_ci.py ────────────────────
def accuracy(yt, yp, _s, _nc):
    return float((yt == yp).mean())


def macro_f1(yt, yp, _s, nc):
    f = []
    for c in range(nc):
        tp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def macro_auc(yt, _yp, score, nc):
    aucs = []
    for c in range(nc):
        pos = yt == c
        if pos.sum() == 0 or pos.sum() == len(yt):
            continue
        s = score[:, c]
        r = np.argsort(np.argsort(s)) + 1.0
        n1 = pos.sum()
        n0 = len(yt) - n1
        aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs)) if aucs else float("nan")


METRICS = {"accuracy": accuracy, "macro_f1": macro_f1, "macro_auc": macro_auc}


def run_task(key, device, seeds, out_dir, train_seed=None):
    cfg = LEGACY[key]
    # The split is IDENTICAL for every training seed -- it is a function of the
    # manifest and split seed 42 alone, neither of which the sweep varies. That
    # is what makes the five retrainings comparable to each other and to the
    # corrected arm: same test scans, same subjects, only the initialization
    # differs.
    if train_seed is None:
        run_dir, ckpt_name = cfg["run_dir"], cfg["ckpt"]
        task_name = f"{key}_scanlevel"
    else:
        run_dir = f"{key}_scanlevel_ts{train_seed}"
        ckpt_name = f"PointNet2_{run_dir}_best.pt"
        task_name = run_dir
    te, lm, lc = legacy_split(cfg["manifest"], cfg["exp"], "test", cfg["binary"])
    tr, _, _ = legacy_split(cfg["manifest"], cfg["exp"], "train", cfg["binary"])
    nc = cfg["num_classes"]
    assert len(lm) == nc, f"{key}: label map {len(lm)} != checkpoint {nc}"

    idc = cfg["id_col"]
    leaked = float(te[idc].isin(set(tr[idc])).mean()) if idc in te.columns else np.nan
    print(f"[{task_name}] test={len(te)} train={len(tr)} classes={nc} "
          f"leaked_scans={leaked:.1%}", flush=True)

    model = PointNet2(num_classes=nc)
    ckpt_path = (PROJECT / "runs" / run_dir / "pointnet2/checkpoints"
                 / ckpt_name)
    if not ckpt_path.exists():
        sys.exit(f"missing checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()

    ds = ScanDS(te, lm, lc)
    bs = C.BATCH_SIZE["pointnet2"]
    rows = []
    per_seed_pred = {}
    for sd in seeds:
        # ONE generator for the whole split, advanced batch to batch -- the same
        # estimator extract_features.py uses. Re-seeding per batch would make
        # every batch draw the same FPS start index.
        g = make_gen(sd)
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        ys, logits = [], []
        t0 = time.time()
        with torch.no_grad():
            for bi, (pts, lbl) in enumerate(loader):
                out = model(pts.to(device), generator=g)
                logits.append(out.cpu().numpy())
                ys.append(lbl.numpy())
                if bi % 20 == 0:
                    print(f"   seed{sd} batch {bi}/{len(loader)} "
                          f"{time.time()-t0:.0f}s", flush=True)
        yt = np.concatenate(ys)
        lg = np.concatenate(logits)
        yp = lg.argmax(1)
        e = np.exp(lg - lg.max(1, keepdims=True))
        sc = e / e.sum(1, keepdims=True)
        per_seed_pred[sd] = (yt, yp, sc)
        np.savez_compressed(
            OUT / f"scanlevel_pred_{task_name}_seed{sd}.npz",
            y_true=yt, y_pred=yp, y_score=sc.astype(np.float32),
            subject_of_unit=np.array(te[idc].astype(str).tolist(), dtype=object))
        for m, fn in METRICS.items():
            rows.append(dict(task=task_name, base_task=f"{key}_scanlevel",
                             train_seed=train_seed, encoder="pointnet2",
                             head="softmax", level="scan", metric=m,
                             seed=sd, value=fn(yt, yp, sc, nc)))
        print(f"[{task_name}] seed {sd}: acc={accuracy(yt,yp,sc,nc):.4f}",
              flush=True)

    # ── bootstrap over test SUBJECTS ───────────────────────────────────────
    # Resample SUBJECTS, not scans, matching analyze_ci.py. It matters here for
    # the same reason it matters there: scans of one subject are correlated, so
    # a scan bootstrap treats 530 correlated scans as 530 independent draws and
    # reports an interval that is too narrow. On this split that understatement
    # is worst of all, because 85% of test scans share a subject with training.
    # NOTE the intervals on the two arms are still NOT paired -- the arms have
    # different test sets -- so no CI may be attached to the difference itself.
    subs = te[idc].astype(str).values
    uniq, sub_idx = np.unique(subs, return_inverse=True)
    by_sub = [np.where(sub_idx == u)[0] for u in range(len(uniq))]
    rng = np.random.default_rng(12345)
    draws = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    summary = []
    for m, fn in METRICS.items():
        vals = [r["value"] for r in rows if r["metric"] == m]
        bs_vals = []
        for d in draws:
            sel = np.concatenate([by_sub[j] for j in d])
            acc = [fn(yt[sel], yp[sel], sc[sel], nc)
                   for yt, yp, sc in per_seed_pred.values()]
            bs_vals.append(np.mean(acc))
        bs_vals = np.array(bs_vals)
        lo, hi = np.percentile(bs_vals, [2.5, 97.5])
        summary.append(dict(
            task=task_name, base_task=f"{key}_scanlevel",
            train_seed=train_seed, encoder="pointnet2", head="softmax",
            level="scan", metric=m, n_seeds=len(seeds),
            mean=float(np.mean(vals)), seed_sd=float(np.std(vals, ddof=0)),
            seed_min=float(np.min(vals)), seed_max=float(np.max(vals)),
            boot_mean=float(bs_vals.mean()),
            boot_ci_lo=float(lo), boot_ci_hi=float(hi),
            n_boot=N_BOOT, n_units=len(te), n_subjects=len(uniq),
            leaked_scan_frac=leaked, corrected_task=cfg["corrected"]))
        print(f"[{task_name}] {m}: {np.mean(vals):.4f} "
              f"[{lo:.4f}, {hi:.4f}]", flush=True)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["B1", "B2", "C"])
    ap.add_argument("--seeds", nargs="+", type=int, default=C.INFERENCE_SEEDS)
    ap.add_argument("--tag", default="")
    # Score the RETRAINED leaky checkpoints instead of the original ones. The
    # original arm stays reachable (omit the flag) because Sec 2.7 still reports
    # C_scanlevel, which was not retrained.
    ap.add_argument("--train-seed", type=int, default=None,
                    choices=C.TRAIN_SEEDS)
    # DEVICE AND THREADS ARE PART OF THE ESTIMATOR HERE, NOT A PREFERENCE.
    # AUDIT_NONDETERMINISM.md line 58: the corrected arm was deliberately run on
    # CPU to avoid cuDNN non-determinism, and every cached feature file was
    # produced at 16 BLAS threads. The whole point of this script is a
    # like-for-like "before" arm, so it must match on both. Defaults are CPU/16;
    # --device cuda exists only for a fast smoke test and must not be used for
    # numbers that go in the paper.
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    if a.device == "cuda":
        print("WARNING: cuda requested — smoke testing only, NOT paper numbers",
              flush=True)
    print(f"device={device} threads={torch.get_num_threads()} "
          f"tasks={a.tasks} seeds={a.seeds} train_seed={a.train_seed}",
          flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, all_sum = [], []
    for k in a.tasks:
        r, s = run_task(k, device, a.seeds, OUT, train_seed=a.train_seed)
        all_rows += r
        all_sum += s
    tag = a.tag or "_".join(a.tasks)
    if a.train_seed is not None and not a.tag:
        tag = f"{tag}_ts{a.train_seed}"
    pd.DataFrame(all_rows).to_csv(OUT / f"scanlevel_per_seed_{tag}.csv", index=False)
    pd.DataFrame(all_sum).to_csv(OUT / f"scanlevel_summary_ci_{tag}.csv", index=False)
    print(f"WROTE {OUT}/scanlevel_summary_ci_{tag}.csv")


if __name__ == "__main__":
    main()
