# Task 1 — Non-determinism audit of the inference path

**Thread:** facebase3d-paper-2026-08-04 · **Agent:** coder · **Date:** 2026-08-04
**Scope:** every source of run-to-run variation in the *evaluation* path for
`facebase_3d_encoder`. Inference only — nothing retrained.

Findings are **empirical, not code-reading**: each encoder was run twice over an
identical fixed batch from the same frozen checkpoint and the outputs differenced
(`scripts/paper_reruns/probe_determinism.py`,
`results/paper_reruns/audit/determinism_probe.csv`).

---

## A. Answer to the gate question (asked first, as instructed)

> **GATE: if the 4,096-point sampling in `dataset.py` is ALSO unseeded, STOP and report.**

**The gate is NOT triggered. Scope is unchanged.**

`dataset.py.__getitem__` (line 184) does:

```python
pts = np.load(row["out_path"]).astype(np.float32)   # (4096, 3)
```

It **loads a pre-computed point set from disk**. It does not sample. The 4,096-point
draw happened once, at preprocessing time
(`experiments/preprocessing/preprocess_v2.py:37,41,72` — `np.random.choice` /
`np.random.randint`), and the result was frozen into `.npy` files. Every eval run
reads byte-identical points.

Two further checks confirm the data path is inert at eval time:
* `self.augment = augment and split == "train"` (line 176) — augmentation, the only
  other stochastic element in the dataset, is structurally unreachable on test.
* `DataLoader(..., shuffle=False)` in every eval script, so ordering is fixed;
  `num_workers` does not affect assembled batch order.

**Consequence:** the 4-encoder blow-up you were worried about does not happen. Only
PointNet++ is affected. Cost stayed in the range I've reported below rather than
quadrupling.

*Caveat worth recording in Methods:* because the point sets were frozen at
preprocessing, **point-sampling variance is not measured by anything we do** — it is
baked into the dataset. Re-running preprocessing with a different seed would give a
different (equally valid) dataset. This is a fixed-dataset study.

---

## B. Inventory — every source examined

| # | Source | Location | Affects | Deterministic after fix? |
|---|---|---|---|---|
| 1 | **FPS start index drawn unseeded** | `pointnet2.py:27` `torch.randint(0,N,(B,))` | **PointNet++ only** (all tasks) | **YES** — now seeded via `torch.Generator` |
| 2 | 4,096-point sampling | `dataset.py:184` → frozen `.npy` | none at eval | N/A — frozen on disk (see §A) |
| 3 | Train-time augmentation | `dataset.py:176,208` | none at eval | N/A — disabled on test by construction |
| 4 | Dropout | all 4 encoders | none | already inert: `model.eval()` verified, 0 modules left in train mode |
| 5 | BatchNorm | all 4 encoders | none | already inert: eval mode uses frozen running stats |
| 6 | cuDNN non-deterministic kernels | GPU eval path | would affect all | **avoided** — re-eval runs on **CPU**, bit-identical **at a fixed BLAS thread count** (probe 2026-08-09: 16 vs 8 vs 1 threads differ at ~1e-6 in `embed`/`logits`, `feat` identical at 16; every cached file was produced at 16 threads, and metrics agree to 6 dp with 0 argmax flips, so nothing needs re-running) |
| 7 | DataLoader worker seeding / shuffle | all eval scripts | none | `shuffle=False`; workers do not reorder |
| 8 | kNN gallery construction order | `knn_eval.py` | kNN head | deterministic (insertion-ordered dicts, `shuffle=False`) |
| 9 | kNN tie-breaking | `knn_eval.py` `np.bincount(...).argmax()` | kNN head | deterministic but **arbitrary** — see §D |
| 10 | Subject label aggregation | `knn_eval.py` vs `evaluate.py` | subject level | deterministic but **inconsistent between scripts** — see §D |

### Measured result

| Encoder | max abs delta, repeat `encode()` | Verdict | CPU cost |
|---|---|---|---|
| GeomMLP | 0.000e+00 | **deterministic** | 0.001 s/scan |
| PointNet | 0.000e+00 | **deterministic** | 0.092 s/scan |
| DGCNN | 0.000e+00 | **deterministic** | 1.102 s/scan |
| **PointNet++** | **1.886e+00** | **NON-DETERMINISTIC** | 0.604 s/scan |

The PN++ delta of **1.886** is on the 1024-d trunk feature on *real* scans — large
enough to flip argmax decisions, which is exactly the ~3 pp accuracy swing
(0.396 vs 0.367) that motivated this task.

**This measurement is load-bearing for the whole design:** because three of four
encoders are provably bit-identical across repeat runs, sweeping them over 5 seeds
would reproduce five identical numbers at 4× the compute. They are run once and
their inference-seed SD is reported as exactly **0 by construction**, not as an
estimate. Only PointNet++ is swept over {0,1,2,3,4}.

---

## C. The fix (Task 2)

`farthest_point_sample(xyz, npoint, generator=None)` now accepts an explicit
`torch.Generator`, threaded through `PointNetSetAbstractionMSG.forward` →
`PointNet2.encode` / `.embed` / `.forward`. Plumbed, not a global side effect.

Constraints honoured:
* **Distribution unchanged.** Still a uniform draw over `[0, N)`. It is *not*
  pinned to a fixed start index — that would change the estimator and silently
  move every result. Verified empirically: 2,000 seeded draws span [0, 4094],
  mean 2093.8 vs uniform expectation 2047.5.
* **One generator per split**, advanced batch to batch. Re-seeding per batch would
  force every batch to draw the same start index — a different, worse estimator.
* **Backward compatible.** `generator=None` reproduces the original unseeded
  behaviour, so existing scripts are unaffected.
* **Checkpoints untouched.** Loaded read-only; nothing retrained.

Self-test — `scripts/paper_reruns/test_seeding.py`, all pass:

| Check | Result |
|---|---|
| same seed → bit-identical `encode()` / `forward()` / `embed()` | max delta **exactly 0.0** |
| different seeds → outputs differ (seed really reaches the sampler) | delta 1.384e-04 ≠ 0 |
| `generator=None` → still non-deterministic (backward compatible) | delta 1.137e-04 ≠ 0 |
| seeded start index still uniform over [0, N) | mean 2093.8, range [0,4094], 1562 unique / 2000 |

The second check exists because a "fix" that never reaches the sampler would pass
the first check trivially.

---

## D. Two deterministic-but-wrong things found on the way

Neither causes run-to-run variance, so neither explains the 3 pp swing. Both are
correctness issues that would survive into the paper unnoticed.

**D1 — kNN ties break by lowest class index.** `knn_eval.py` predicts with
`np.bincount(topk_labels).argmax()`. On a tied vote among 19 or 33 classes, this
always awards the class with the smallest integer id — i.e. whichever label sorted
first alphabetically when the label map was built. With k=11 over 19–33 classes ties
are common. This is a systematic bias toward alphabetically-early classes, not a
coin flip.

*I deliberately reproduced this behaviour rather than fixing it*, because the task is
to measure the uncertainty of the **published estimator**, not to substitute a better
one. Fixing it is a separate decision — flagging it for that decision to be made
explicitly.

**D2 — the two eval scripts disagree on subject labels.** For a subject whose scans
carry inconsistent labels:
* `knn_eval.py:subject_aggregate` takes the **majority** label,
* `evaluate.py:evaluate_subject_level` takes the **last** label encountered.

`dataset.py:127` drops label-inconsistent subjects for the syndrome/clinical tasks,
so this is currently latent rather than active for B1/B2. It is a live hazard for any
task that does not go through that filter.

---

## E. Out-of-scope finding that blocks a manuscript claim

**The "A1 cross-site generalization" experiment is not a cross-site experiment.**

`EXPERIMENTAL_FINDINGS.md` describes A1_S1/S2/S3 as *"train on one FB-56 acquisition
site, test on full set"*, and concludes *"cross-site generalization essentially
fails… site/scanner confounds are a major bottleneck."*

The runs do not do this:

* `ofc_manifest.csv` **does** have a `site` column (11 sites: PH 1960, FC 1259, GW 744, …).
* **Neither `train.py` nor `dataset.py` ever reads it** — `grep -n site` over
  `experiments/models/*.py` returns nothing. There is no site-filtering code path.
* All three launch scripts pass the **full, unfiltered** `ofc_manifest.csv`:
  * `train_A1_S1_pointnet2.sh` → `--sampler weighted`
  * `train_A1_S2_pointnet2.sh` → `--loss focal --focal-gamma 2`
  * `train_A1_S3_pointnet2.sh` → `--sampler weighted --loss focal --focal-gamma 2`
* Every `config.json` reports **identical split sizes** to plain A1:
  `n_train=4815, n_val=1029, n_test=1029`.

**"S" meant *strategy*, not *site*.** These are three class-imbalance variants trained
on the identical full-cohort split. The near-zero accuracies (0.023, 0.027) are what a
weighted sampler plus focal loss does to a 75%-majority-class problem — it stops
predicting the majority class — not evidence about scanner transfer.

**Impact:** the cross-site claim has no supporting experiment. A real cross-site test
is runnable (the `site` column exists) but requires **retraining**, which is out of
scope here. I re-evaluated these three checkpoints under their true identity
(imbalance-strategy variants) and labelled them as such in `rerun_config.py`.

Escalating rather than deciding: this is a claim-level change for the manuscript.
