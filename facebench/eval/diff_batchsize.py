#!/usr/bin/env python3
"""
PART B follow-up — does EVAL BATCH SIZE move the PointNet++ numbers?
thread facebase3d-paper-2026-08-08.

Device is ruled out (diff_gpu_cpu.py: metrics identical). Thread count is ruled
out (diff_threads.py: zero argmax flips, zero metric movement). What remains
unexplained is that the published PN++ point estimates are not merely outside
the 5-seed range but SYSTEMATICALLY ABOVE it — +0.7 to +1.8 pp on every
comparable row. Symmetric noise cannot do that, so the asymmetry has to be in
the estimator.

THE CANDIDATE. farthest_point_sample draws its start index as
torch.randint(0, N, (B,)) — once per SAMPLE per BATCH, off a generator advanced
batch to batch. So the batch size is part of the estimator, not a throughput
knob: the same scans, rebatched, get different FPS start indices and therefore
different sampled points.

  published : train.py's end-of-training eval, test loader at batch_size*2 = 32
  re-eval   : extract_features.py at C.BATCH_SIZE["pointnet2"] = 16

If batch size moves the metrics, the two were never like-for-like, and the
honest description of numbers_from_reruns.tex gains ", at eval batch 16".

Both arms are CPU at threads=16, matching the cache, so thread count is held
fixed and the only thing varying is the batching.

Usage: python3 diff_batchsize.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402
from analyze_ci import METRICS                              # noqa: E402
from score_grid import head_scores                          # noqa: E402

TASK, ENC = "B2_corrected", "pointnet2"
SEEDS = C.INFERENCE_SEEDS
BATCHES = [16, 32]

# Published PN++ values on this task's family, for orientation only. These are
# NOT thresholds and nothing is tuned toward them.
PUBLISHED_GAP_PP = "published sits +0.7 to +1.8 pp above the 5-seed mean"


def load(d, seed, split="test"):
    p = Path(d) / f"{TASK}__{ENC}__seed{seed}__{split}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def main():
    nc = C.TASKS[TASK]["num_classes"]
    # Gallery for the head comes from the cache; softmax needs only test
    # logits, but head_scores takes a train dict, so pass the cached one.
    tr = load(C.OUT / "features", 0, "train")

    rows = []
    for bs in BATCHES:
        for seed in SEEDS:
            te = load(C.OUT / f"features_bs{bs}", seed)
            sc_p, sc_s, *_ = head_scores("softmax", tr, te, nc)
            r = dict(batch_size=bs, seed=seed)
            for m, fn in METRICS.items():
                r[m] = fn(te["labels"], sc_p, sc_s, nc)
            rows.append(r)
    df = pd.DataFrame(rows)

    print("=" * 78)
    print(f"EVAL BATCH SIZE PROBE — {TASK}/{ENC}/softmax/scan, CPU, threads=16")
    print("=" * 78)
    print("\nper-seed metrics")
    print(df.set_index(["batch_size", "seed"]).round(6).to_string())

    print("\nsummary over the 5 inference seeds")
    g = df.groupby("batch_size")[list(METRICS)].agg(["mean", "std", "min", "max"])
    print(g.round(6).to_string())

    print("\nbatch 32 minus batch 16, percentage points")
    a = df[df.batch_size == 16].set_index("seed")[list(METRICS)]
    b = df[df.batch_size == 32].set_index("seed")[list(METRICS)]
    d = (b - a) * 100
    print(d.round(4).to_string())
    print("\n  per-metric mean offset (pp):")
    for m in METRICS:
        v = d[m]
        same = bool(np.all(np.sign(v.values) == np.sign(v.mean()))
                    and v.mean() != 0)
        print(f"    {m:10s} mean={v.mean():+.4f}  sd={v.std():.4f}  "
              f"min={v.min():+.4f}  max={v.max():+.4f}  "
              f"same_sign_all_seeds={same}")

    # Does rebatching change which points get sampled at all?
    print("\n  sanity — do the two batchings actually differ in the features?")
    for seed in SEEDS[:2]:
        x, y = load(C.OUT / "features_bs16", seed), load(C.OUT / "features_bs32", seed)
        dl = float(np.abs(x["logits"].astype("f8") - y["logits"].astype("f8")).max())
        flips = int((x["logits"].argmax(1) != y["logits"].argmax(1)).sum())
        print(f"    seed{seed}: max|d_logits|={dl:.3e}  argmax_flips="
              f"{flips}/{len(x['labels'])}")

    df.to_csv(C.OUT / "batchsize_probe.csv", index=False)
    print(f"\n-> {C.OUT.name}/batchsize_probe.csv")
    print(f"\nOrientation only, not a threshold: {PUBLISHED_GAP_PP}.")


if __name__ == "__main__":
    main()
