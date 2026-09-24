#!/usr/bin/env python3
"""
Format the verdict table and the headline summary tables as markdown, from
summary_ci.csv and contrast_tests.csv.

Emits fragments that are pasted into RERUN_REPORT.md, so every number in the
report is machine-generated from the CSVs rather than transcribed by hand.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C            # noqa: E402

PP = lambda x: f"{100*x:+.1f}"      # noqa: E731  (percentage points)


def verdict_table(con, metric="accuracy"):
    d = con[con.metric == metric].copy()
    if d.empty:
        return "_(no contrasts)_\n"
    lines = [
        f"| Contrast | Task | Encoder | diff ({metric}) | 95% CI (subject bootstrap) | seed SD | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in d.iterrows():
        enc = r.contrast.split("[")[-1].rstrip("]") if "[" in r.contrast else "pointnet2"
        name = r.contrast.split("[")[0]
        lines.append(
            f"| {name} | {r.task} | {enc} | {PP(r.diff_mean)} pp | "
            f"[{PP(r.boot_ci_lo)}, {PP(r.boot_ci_hi)}] pp | "
            f"{100*r.diff_seed_sd:.2f} pp | {r.verdict} |")
    return "\n".join(lines) + "\n"


def grid_table(summ, task, metric="accuracy"):
    d = summ[(summ.task == task) & (summ.metric == metric)]
    if d.empty:
        return f"_(no results for {task})_\n"
    lines = [f"| Encoder | Head | Level | mean | seed SD | seed range | 95% CI (subjects) |",
             "|---|---|---|---|---|---|---|"]
    for enc in C.ENCODERS:
        for head in C.HEADS:
            for lvl in C.LEVELS:
                r = d[(d.encoder == enc) & (d.head == head) & (d.level == lvl)]
                if r.empty:
                    continue
                r = r.iloc[0]
                rng = ("deterministic" if r.seed_deterministic
                       else f"[{r.seed_min:.4f}, {r.seed_max:.4f}]")
                lines.append(
                    f"| {enc} | {head} | {lvl} | {r['mean']:.4f} | "
                    f"{r.seed_sd:.4f} | {rng} | "
                    f"[{r.boot_ci_lo:.4f}, {r.boot_ci_hi:.4f}] |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suffix", default="")
    a = p.parse_args()

    summ = pd.read_csv(C.OUT / f"summary_ci{a.suffix}.csv")
    con = pd.read_csv(C.OUT / f"contrast_tests{a.suffix}.csv")

    print("## Verdict table — accuracy\n")
    print(verdict_table(con, "accuracy"))
    print("\n## Verdict table — macro-F1\n")
    print(verdict_table(con, "macro_f1"))
    for t in ["B1_corrected", "B2_corrected"]:
        print(f"\n## Decision grid — {t} (accuracy)\n")
        print(grid_table(summ, t, "accuracy"))

    # Inference-seed noise band, the thing that started all this.
    print("\n## Inference-seed spread (PointNet++ only; others are exactly 0)\n")
    d = summ[(summ.encoder == "pointnet2") & (summ.metric == "accuracy")]
    print("| Task | Head | Level | mean | seed SD | seed min | seed max | swing (pp) |")
    print("|---|---|---|---|---|---|---|---|")
    for _, r in d.iterrows():
        print(f"| {r.task} | {r['head']} | {r.level} | {r['mean']:.4f} | "
              f"{r.seed_sd:.4f} | {r.seed_min:.4f} | {r.seed_max:.4f} | "
              f"{100*(r.seed_max-r.seed_min):.2f} |")


if __name__ == "__main__":
    main()
