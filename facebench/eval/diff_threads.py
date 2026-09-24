#!/usr/bin/env python3
"""
PART B follow-up — is CPU inference bit-identical across BLAS thread counts?
thread facebase3d-paper-2026-08-08.

extract_features.py routes the whole 5-seed re-eval to CPU on the stated
grounds that "CPU: bit-identical by construction". The PART B control found
that a threads=8 re-extraction does NOT reproduce the threads=16 cache exactly:
the trunk output `feat` is bit-identical but the fully-connected head differs at
~1e-6. This separates the two candidate causes — thread count vs. the --device
edit — by re-extracting the same config at 16, 8 and 1 threads.

threads=16 MUST reproduce the cache exactly. If it does, the edit is innocent
and the residual is purely a thread-count effect; the "bit-identical by
construction" claim then needs the qualifier "at a fixed thread count".

Usage: python3 diff_threads.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402
from analyze_ci import METRICS                              # noqa: E402
from score_grid import head_scores                          # noqa: E402

TASK, ENC, SEED, SPLIT = "B2_corrected", "pointnet2", 0, "test"
THREADS = [16, 8, 1]


def load(d):
    p = Path(d) / f"{TASK}__{ENC}__seed{SEED}__{SPLIT}.npz"
    if not p.exists():
        raise SystemExit(f"missing {p}")
    z = np.load(p, allow_pickle=True)
    return {k: z[k] for k in z.files}


def dmax(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def main():
    ref = load(C.OUT / "features")          # the cache, built at threads=16
    print("=" * 78)
    print(f"CPU thread-count probe — {TASK}/{ENC}/seed{SEED}/{SPLIT}")
    print("reference = results/features/ (built at threads=16)")
    print("=" * 78)
    print(f"{'threads':>8} {'d_feat':>11} {'d_embed':>11} {'d_logits':>11} "
          f"{'argmax_flips':>13} {'identical':>10}")
    rows = {}
    for t in THREADS:
        f = load(C.OUT / f"features_thr{t}")
        d = {k: dmax(ref[k], f[k]) for k in ("feat", "embed", "logits")}
        flips = int((ref["logits"].argmax(1) != f["logits"].argmax(1)).sum())
        ident = all(v == 0.0 for v in d.values())
        rows[t] = (f, ident)
        print(f"{t:>8} {d['feat']:>11.3e} {d['embed']:>11.3e} "
              f"{d['logits']:>11.3e} {flips:>13d} {str(ident):>10}")

    print("\n--- do the metrics move? (softmax head, scan level) ---")
    nc = C.TASKS[TASK]["num_classes"]
    tr = np.load(C.OUT / "features" /
                 f"{TASK}__{ENC}__seed{SEED}__train.npz", allow_pickle=True)
    tr = {k: tr[k] for k in tr.files}
    base = None
    for t in THREADS:
        te = rows[t][0]
        sc_p, sc_s, *_ = head_scores("softmax", tr, te, nc)
        vals = {m: fn(te["labels"], sc_p, sc_s, nc) for m, fn in METRICS.items()}
        if base is None:
            base = vals
        delta = {m: 100 * (vals[m] - base[m]) for m in vals}
        print(f"  threads={t:<3d} " +
              "  ".join(f"{m}={vals[m]:.6f} ({delta[m]:+.4f}pp)" for m in vals))

    print("\n" + "=" * 78)
    if rows[16][1]:
        print("threads=16 REPRODUCES THE CACHE EXACTLY.")
        print("  => the --device edit is innocent; the ~1e-6 residual seen in the")
        print("     PART B control is a BLAS THREAD-COUNT effect.")
        print("  => 'CPU is bit-identical by construction' holds only AT A FIXED")
        print("     THREAD COUNT. Every cached feature file was produced at 16,")
        print("     so the published re-eval numbers are internally consistent —")
        print("     but the qualifier belongs in the audit note.")
    else:
        print("threads=16 does NOT reproduce the cache — the edit or the")
        print("environment changed something. Investigate before proceeding.")
    print("=" * 78)


if __name__ == "__main__":
    main()
