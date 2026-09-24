#!/usr/bin/env python3
"""
TASK 2 — determinism self-test for the seeded PointNet++ inference path.

Asserts BOTH directions, because only checking the first would pass trivially
if the seed never reached the sampler at all:
  (A) same seed  -> bit-identical outputs, max|delta| == 0 exactly
  (B) diff seeds -> outputs actually differ (the seed is plumbed, not ignored)
  (C) generator=None still behaves as before (backward compatible, unseeded)
  (D) the sampling DISTRIBUTION is unchanged: seeded FPS start indices are
      uniform over [0, N), matching the original estimator.

Run:  python3 facebench/eval/test_seeding.py
Exit code 0 = all pass.
"""
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "facebench/models"))

from pointnet2 import PointNet2, farthest_point_sample   # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def gen(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def main():
    torch.manual_seed(0)
    model = PointNet2(num_classes=33).eval()
    x = torch.randn(4, 4096, 3)

    print("\n(A) same seed -> bit-identical")
    with torch.no_grad():
        a1 = model.encode(x, generator=gen(0))
        a2 = model.encode(x, generator=gen(0))
    d_same = (a1 - a2).abs().max().item()
    check("encode() same seed is bit-identical", d_same == 0.0,
          f"max|delta| = {d_same:.3e}")

    with torch.no_grad():
        l1 = model(x, generator=gen(7))
        l2 = model(x, generator=gen(7))
    d_logit = (l1 - l2).abs().max().item()
    check("forward() same seed is bit-identical", d_logit == 0.0,
          f"max|delta| = {d_logit:.3e}")

    with torch.no_grad():
        e1 = model.embed(x, generator=gen(3))
        e2 = model.embed(x, generator=gen(3))
    d_emb = (e1 - e2).abs().max().item()
    check("embed() same seed is bit-identical", d_emb == 0.0,
          f"max|delta| = {d_emb:.3e}")

    print("\n(B) different seeds -> outputs differ (seed truly reaches the sampler)")
    with torch.no_grad():
        b1 = model.encode(x, generator=gen(0))
        b2 = model.encode(x, generator=gen(1))
    d_diff = (b1 - b2).abs().max().item()
    check("encode() differs across seeds", d_diff > 0.0,
          f"max|delta| = {d_diff:.3e}")

    print("\n(C) generator=None -> original unseeded behaviour preserved")
    with torch.no_grad():
        c1 = model.encode(x)
        c2 = model.encode(x)
    d_none = (c1 - c2).abs().max().item()
    check("unseeded path still non-deterministic (backward compatible)",
          d_none > 0.0, f"max|delta| = {d_none:.3e}")

    print("\n(D) sampling distribution unchanged (uniform start index)")
    # Draw many seeded FPS start indices and check they span [0, N) uniformly.
    N = 4096
    xyz = torch.randn(1, N, 3)
    starts = []
    for s in range(2000):
        idx = farthest_point_sample(xyz, 1, generator=gen(s))
        starts.append(idx[0, 0].item())
    starts_t = torch.tensor(starts, dtype=torch.float)
    mean, lo, hi = starts_t.mean().item(), min(starts), max(starts)
    n_unique = len(set(starts))
    # Uniform over [0,4096) has mean ~2047.5; allow generous tolerance.
    check("seeded start index is uniform over [0, N)",
          abs(mean - (N - 1) / 2) < N * 0.05 and lo < N * 0.05 and hi > N * 0.95,
          f"mean={mean:.1f} (expect ~{(N-1)/2:.1f}), range=[{lo},{hi}], "
          f"{n_unique} unique / 2000 draws")

    print("\n" + "=" * 60)
    if FAILS:
        print(f"FAILED: {len(FAILS)} check(s): {FAILS}")
        return 1
    print("ALL CHECKS PASSED — inference path is seed-controlled and "
          "distribution-preserving.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
