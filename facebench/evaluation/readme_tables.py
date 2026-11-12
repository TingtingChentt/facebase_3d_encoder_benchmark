#!/usr/bin/env python3
"""
Print the README's headline tables (markdown) from the shipped training-seed
summaries, so the README never carries a hand-typed number.

  Table A  every task x encoder, scan level, softmax head (paper Figure 2)
  Table B  49-class syndrome diagnosis, subject level, softmax vs prototype (paper Table 2)

Every cell is mean +/- SD over five independently trained models (train seeds 1-5);
PointNet++'s inference seeds are averaged within each training seed first.

Usage: python3 facebench/evaluation/readme_tables.py
"""
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[2] / "results"
ENC = [("geommlp", "GeomMLP"), ("pointnet", "PointNet"), ("dgcnn", "DGCNN"), ("pointnet2", "PointNet++")]
TASKS = [("C_full_v3", "cfullv3", "Combined screening", 2),
         ("A2_fix_v3", "a2fixv3", "Cleft screening", 2),
         ("A1_fix_v3", "a1fixv3", "Cleft subtyping", 4),
         ("CF20_full_v3", "cf20fullv3", "Syndrome diagnosis", 49)]


def cell(df, enc, head, level, metric):
    r = df[(df.encoder == enc) & (df["head"] == head) & (df.level == level) & (df.metric == metric)]
    assert len(r) == 1, (enc, head, level, metric, len(r))
    r = r.iloc[0]
    return f"{r.train_seed_mean:.3f} ± {r.train_seed_sd:.3f}"


def main():
    print("| Task | *K* | Chance acc. | Encoder | Accuracy | Macro AUC |")
    print("|---|---|---|---|---|---|")
    for task, suf, name, k in TASKS:
        df = pd.read_csv(RES / f"{suf}_trainseed_summary.csv")
        df = df[df.base_task == task]
        for i, (enc, lab) in enumerate(ENC):
            lead = f"| {name} | {k} | {1 / k:.3f} " if i == 0 else "| | | "
            print(f"{lead}| {lab} | {cell(df, enc, 'softmax', 'scan', 'accuracy')} "
                  f"| {cell(df, enc, 'softmax', 'scan', 'macro_auc')} |")
    print()
    df = pd.read_csv(RES / "cf20fullv3_trainseed_summary.csv")
    print("| Encoder | Head | Accuracy | Top-5 | Macro F1 | Macro AUC |")
    print("|---|---|---|---|---|---|")
    for enc, lab in ENC:
        for head in ["softmax", "proto"]:
            m = [cell(df, enc, head, "subject", x) for x in ["accuracy", "top5_accuracy", "macro_f1", "macro_auc"]]
            print(f"| {lab} | {'prototype' if head == 'proto' else 'softmax'} | " + " | ".join(m) + " |")


if __name__ == "__main__":
    main()
