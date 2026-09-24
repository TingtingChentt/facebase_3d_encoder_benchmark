# Legacy scan-level (leaky) manifests — DO NOT USE FOR HEADLINE RESULTS

These are the pre-correction manifests for the 19-class (B1) and 33-class (B2)
syndrome tasks with the `fbid` subject-id column **removed**.

## Why they exist
Figure 5(a) contrasts a scan-level split against the subject-disjoint split to
measure how much repeated scans of the same patient inflate performance. Until
2026-09-04 the leaky arm of that panel was a single unseeded checkpoint per task,
re-scored by `scripts/paper_reruns/rerun_scanlevel.py`, while the corrected arm
had five training seeds. Both arms were rerun as five-training-seed means
so the panel matches Table 2's convention. That requires *training* on the leaky
split, which nothing in the pipeline previously did.

## Why dropping one column is the whole mechanism
`dataset.py` splits by subject whenever the manifest carries the experiment's
subject-id column, and falls back to a scan-level stratified split otherwise
(that fallback is legitimate and long-standing — the cleft cohorts really are
~1 scan/subject). With `fbid` gone, the syndrome manifests take that fallback,
and it reproduces `rerun_scanlevel.legacy_split()` — the split as it stood at
commit 1d4ec45 — exactly: identical rows in identical order for train/val/test
on both B1 and B2. Verified before any run was queued.

This deliberately does NOT add a leaky-split flag to `dataset.py`. The leak
lives in these frozen CSVs, where it is visible and inert, not in a code path
that some future run could pick up by accident.

## Standing constraint
Models trained from these manifests are valid ONLY as the leaked arm of the
leakage comparison (Sec 2.7, Fig 5a). ~79% of B1 and ~85% of B2 test scans come
from subjects also seen in training. Any number from them is inflated by
construction and must never be quoted as task performance.

Runs: `experiments/runs/{B1,B2}_scanlevel_ts{1..5}/`
Thread: facebase3d-figs-2026-09-04
