#!/usr/bin/env python3
"""
PART A — cross-site leave-one-site-out report.
thread facebase3d-paper-2026-08-08.

THE VERDICT IS SPLIT BY FOLD TYPE AND NEVER AVERAGED INTO ONE HEADLINE.

  GROUP 1  PH, FC, PR   Both FB-56 and FB-5A remain in the training pool, so
                        holding one of these sites out varies SITE while
                        holding DATASET fixed. These folds isolate the
                        acquisition-site effect.
  GROUP 2  GW, PT, LC   Single-dataset sites. Holding one out removes a site
                        AND shifts the dataset mix, so site and dataset move
                        together and are measured jointly.

main.tex currently concedes that acquisition effect and population effect
cannot be separated. For Group 1 they can. Collapsing the two groups into one
average would throw away exactly the thing that makes this run worth having,
so this script refuses to print a pooled number.

PREVALENCE WARNING, APPLIED THROUGHOUT. Held-out sites are 72-92% Unaffected,
and the prevalence differs per site. Accuracy is therefore not comparable
across folds and is dominated by the majority class — a constant "Unaffected"
predictor scores 0.80 at PH. The majority-class rate is printed beside every
accuracy so no row can be read without it, and macro-AUC (chance = 0.50
regardless of prevalence) is treated as the headline metric. That is also the
metric the retired claim was stated in.

Usage: python3 report_xsite.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import rerun_config as C                                    # noqa: E402

SUMMARY = C.OUT / "summary_ci_xsite.csv"
AUDIT = C.OUT / "xsite_fold_audit.csv"
REPORT = C.OUT / "XSITE_REPORT.md"
HEADLINE = "macro_auc"


def load():
    if not SUMMARY.exists():
        raise SystemExit(f"missing {SUMMARY} — run analyze_ci.py --suffix _xsite")
    df = pd.read_csv(SUMMARY)
    df = df[df.task.str.startswith("XS_")].copy()
    df["site"] = df.task.str.replace("XS_", "", regex=False)
    df["group"] = np.where(df.site.isin(C.XSITE_BOTH_DATASETS),
                           "1_site_isolated", "2_site+dataset")
    return df


def prevalence():
    """Majority-class rate per held-out site — the accuracy a constant
    predictor achieves. Read from the post-attrition fold audit."""
    a = pd.read_csv(AUDIT).set_index("site")
    cols = [c for c in a.columns if c.startswith("test_")]
    out = {}
    for site in C.XSITE_SITES:
        r = a.loc[site, cols]
        out[site] = dict(n_test=int(a.loc[site, "n_test"]),
                         majority=float(r.max() / r.sum()),
                         majority_class=cols[int(np.argmax(r.values))]
                         .replace("test_", ""))
    return out


def fmt(r):
    """point [ci_lo, ci_hi] +/-seed_sd, all in percentage points."""
    sd = "" if r.seed_deterministic else f" ±{100*r.seed_sd:.2f}sd"
    return f"{r['mean']:.3f} [{r.boot_ci_lo:.3f},{r.boot_ci_hi:.3f}]{sd}"


def section(df, prev, group, title, lines):
    sub = df[df.group == group]
    if sub.empty:
        return
    lines.append(f"\n## {title}\n")
    for site in [s for s in C.XSITE_SITES if s in set(sub.site)]:
        p = prev[site]
        lines.append(f"\n### Held-out site {site} "
                     f"(n_test={p['n_test']}, majority class "
                     f"{p['majority_class']} = {p['majority']:.3f})\n")
        lines.append("| encoder | macro_auc (headline) | macro_f1 | accuracy | "
                     "acc - majority |")
        lines.append("|---|---|---|---|---|")
        for enc in C.ENCODERS:
            e = sub[(sub.site == site) & (sub.encoder == enc)]
            if e.empty:
                lines.append(f"| {enc} | *(missing)* | | | |")
                continue
            row = {m: e[e.metric == m].iloc[0] for m in
                   ("macro_auc", "macro_f1", "accuracy")}
            gap = row["accuracy"]["mean"] - p["majority"]
            lines.append(f"| {enc} | {fmt(row['macro_auc'])} | "
                         f"{fmt(row['macro_f1'])} | {fmt(row['accuracy'])} | "
                         f"{gap:+.3f} |")


def verdict_block(df, lines):
    """Per-group statements about the headline metric, kept separate."""
    lines.append("\n## Headline verdict, by group — NOT averaged\n")
    h = df[df.metric == HEADLINE]
    for group, label in (("1_site_isolated",
                          "GROUP 1 — site isolated from dataset (PH, FC, PR)"),
                         ("2_site+dataset",
                          "GROUP 2 — site and dataset confounded (GW, PT, LC)")):
        g = h[h.group == group]
        if g.empty:
            continue
        lines.append(f"\n**{label}**\n")
        lines.append(f"- folds: {', '.join(sorted(set(g.site)))}")
        lines.append(f"- macro-AUC range over fold x encoder: "
                     f"{g['mean'].min():.3f} - {g['mean'].max():.3f}")
        # A fold x encoder cell is "above chance" only if its bootstrap
        # interval clears 0.50 outright.
        above = g[g.boot_ci_lo > 0.5]
        at = g[(g.boot_ci_lo <= 0.5) & (g.boot_ci_hi >= 0.5)]
        below = g[g.boot_ci_hi < 0.5]
        lines.append(f"- cells with CI entirely ABOVE chance (0.50): "
                     f"{len(above)}/{len(g)}")
        lines.append(f"- cells whose CI CONTAINS chance: {len(at)}/{len(g)}")
        lines.append(f"- cells with CI entirely BELOW chance: {len(below)}/{len(g)}")
        for _, r in g.sort_values("mean").iterrows():
            mark = ("above" if r.boot_ci_lo > 0.5 else
                    "spans" if r.boot_ci_hi >= 0.5 else "below")
            lines.append(f"    - {r.site:3s} {r.encoder:10s} "
                         f"{r['mean']:.3f} [{r.boot_ci_lo:.3f},"
                         f"{r.boot_ci_hi:.3f}]  ({mark} chance)")


def main():
    df, prev = load(), prevalence()
    lines = [
        "# Cross-site leave-one-site-out — results",
        "",
        "thread facebase3d-paper-2026-08-08, PART A.",
        "",
        "Generated by `facebench/eval/report_xsite.py` from "
        "`summary_ci_xsite.csv`.",
        "",
        "**There was no prior result here.** The 'cross-site transfer collapses "
        "to chance, AUC 0.48-0.51' figures in earlier drafts came from the "
        "A1_S1/S2/S3 runs, which are class-imbalance variants trained on the "
        "full-cohort split, not site splits. No leave-one-site-out experiment "
        "had been run before this one, so nothing below is a confirmation or a "
        "contradiction of a measured number.",
        "",
        "**Uncertainty.** `±sd` is the inference-seed SD (PointNet++ FPS draw, "
        "weights frozen; 5 seeds). `[lo,hi]` is the 95% bootstrap interval over "
        "held-out-site TEST SUBJECTS. The two are never pooled. Training-seed "
        "variance is NOT measured here — every model is a single training run.",
        "",
        "**Read accuracy against the majority-class rate in each site header.** "
        "Held-out sites are 72-92% Unaffected and prevalence differs per site, "
        "so raw accuracy is not comparable across folds. macro-AUC is the "
        "headline metric; its chance level is 0.50 everywhere.",
    ]
    verdict_block(df, lines)
    section(df, prev, "1_site_isolated",
            "GROUP 1 — site isolated from dataset "
            "(both FB-56 and FB-5A remain in training)", lines)
    section(df, prev, "2_site+dataset",
            "GROUP 2 — site and dataset confounded (single-dataset sites)",
            lines)
    lines.append("")
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {REPORT}")


if __name__ == "__main__":
    main()
