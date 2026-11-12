# Paper → artifact map

Every figure, table and quoted number in the manuscript, with the script that
produces it and the files it reads. Paths are relative to the repository root.
"Level 1" = regenerable from the shipped files; "Level 2" = needs the FaceBase
data and/or locally rebuilt predictions (see `docs/REPRODUCE.md`).

## Main figures

| Paper | Output | Generator | Reads | Level |
|---|---|---|---|---|
| Fig. 1 — screening to diagnostic prioritization | `figures/output/fig1_clinical.{png,pdf}` | `figures/fig1_clinical.py` (+ `fig1_scan_quality.py`) | features of the 49-class and combined PointNet++ models, labelled manifests, one FaceBase mesh, local exemplar list; panel (b) counts checked against the Table 1 macros | 2 |
| Fig. 2 — four tasks × four encoders | `figures/output/fig1_tasks_encoders_trainseed.png` | `figures/fig1_tasks_encoders_trainseed.py` | `results/{a1fixv3,a2fixv3,cfullv3,cf20fullv3}_trainseed_{summary,per_run}.csv` | 1 |
| Fig. 3 — cohort composition and screening | `figures/output/fig_cohort_confound.{png,pdf}` | `figures/fig_cohort_confound.py` | `results/cohort_sensitivity/*.csv`, combined-screening predictions, FaceBase age/sex tables | 2 |
| Fig. 4 — repeated-scan aggregation | `figures/output/fig_aggregation.{png,pdf}` | `figures/fig_aggregation.py` | `results/scan_aggregation/*` (per-patient tables de-identified), `results/contrast_tests_cf20fullv3.csv` | 1 |
| Fig. 5 — diagnostic prioritization and retrieval | `figures/output/fig_prioritization.{png,pdf}` | `figures/fig_prioritization.py` | `results/prioritization/*_summary.csv`, `*_ci.csv`, `*_chance.csv` | 1 |
| Fig. 6 — class size vs diagnostic performance | `figures/output/fig3_recognizability.png` | `figures/fig3_recognizability.py` | per-class F1 and test size (`results/derived/perclass_f1_*.json`, via `facebench/evaluation/recognizability.py`) | 1 |
| Fig. 7 — anatomical interpretability | `figures/output/fig_interp_main.{png,pdf}` | `analyses/interpretability/main_figure_interp.py` | `results/interp/` (scan lists, cases, occlusion scores, landmarks), FaceBase meshes, local exemplar list | 2 |
| Fig. 8 — encoder architectures (Methods) | `figures/output/fig_encoders.{png,pdf}` | `figures/fig_encoders.py` (+ `fig_encoders_renders.py`) | four renders of one test scan (`figures/src/enc_*.png`, local) | 2 |

## Supplementary figures

| Paper | Output | Generator | Reads | Level |
|---|---|---|---|---|
| Fig. S1 — study overview and 3D facial analysis framework | `figures/output/figS_overview.png` | `figures/figS_overview.py` (+ `figS_overview_registration.py` for panel a) | `data/splits/`, FaceBase demographic tables, two FaceBase meshes, one point cloud | 2 |
| Fig. S2 — per-class F1 change, prototype − softmax | `figures/output/figS1_perclass_head.png` | `figures/figS1_perclass_head.py` | `results/perclass_head_CF20_full_v3_trainseed.npz` | 1 |
| Fig. S3 — redistribution of F1 and AUC | `figures/output/figS2_redistribution.png` | `figures/figS2_redistribution.py` | `results/perclass_head_CF20_full_v3_trainseed.npz` | 1 |
| Fig. S4 — class-size threshold sensitivity | `figures/output/figS5_cutoff.png` | `figures/figS5_cutoff.py` (+ `cutoff_panel.py`) | `results/cfv3_trainseed_summary.csv`, `results/cutoff_fixed_test_per_run_v3.csv` | 1 |
| Fig. S5 — structure of classification errors | `figures/output/figS3_error_structure.png` | `figures/figS3_error_structure.py` | `results/derived/confusion_CF20_full_v3_pointnet2_softmax_subject.json` | 1 |
| Fig. S6 — nearest-neighbour retrieval | `figures/output/figS_interp_retrieval.{png,pdf}` | `analyses/interpretability/main_figure_interp.py` | as Fig. 7 | 2 |
| Fig. S7 — misclassified subjects | `figures/output/figS_interp_misclassified.{png,pdf}` | `analyses/interpretability/main_figure_interp.py` | as Fig. 7 | 2 |

Output file names keep the names used during the study; the paper number is in
the first column.

## Tables

| Paper | Source |
|---|---|
| Table 1 — cohorts and task contributions | `results/paper_macros/numbers_corpus.tex` (from `data/splits/scans.csv`; `facebench/evaluation/emit_numbers_corpus.py`) |
| Table 2 — aggregation × inference strategy, 49-class task | `results/cf20fullv3_trainseed_summary.csv`; macros `\cf*` in `numbers_cf20.tex` |
| Table 3 — combined-screening composition by cohort | `results/cohort_sensitivity/cohort_table.csv` (`analyses/cohort_sensitivity/cohort_confound.py`); relatives / unrelated counts in `tj0_unaffected_composition.csv` |
| Table 4 — comparison with published methods | literature values, `\lit*` in `numbers_static.tex` |
| Table S1 — cohort sensitivity analyses | `results/paper_tables/table_s_cohort_sensitivity.tex` (`analyses/cohort_sensitivity/write_supp_table_cohort.py`) |
| Table S2 — aggregation on combined screening | `results/cfullv3_trainseed_summary.csv`, `results/contrast_tests_cfullv3.csv`; macros `\cfu*` |
| Table S3 — controlled same-patient aggregation steps | `results/paper_tables/table_s_aggctl_steps.tex` (`analyses/scan_aggregation/write_supp_table_aggctl.py`) |

## LaTeX macro blocks (`results/paper_macros/`)

| Macro file | Emitter | Covers |
|---|---|---|
| `numbers_corpus.tex` | `facebench/evaluation/emit_numbers_corpus.py` | processed scans / subjects per cohort (`\NScansProc`, `\ds*`) |
| `numbers_cf20.tex` (+ `.json`) | `facebench/evaluation/emit_numbers_cf20.py` | 49-class task: cells, head and aggregation contrasts, per-class redistribution, class size, error structure, task-definition counts (`\cf*`) |
| `numbers_cfull.tex` | `facebench/evaluation/emit_numbers_cfull.py` | combined screening (`\cfu*`) |
| `numbers_ofc.tex` | `facebench/evaluation/emit_numbers_ofc.py` | cleft screening and subtyping (`\tsOfc*`) |
| `numbers_cohort.tex` | `analyses/cohort_sensitivity/write_numbers_cohort.py` | cohort sensitivity (`\cc*`) |
| `numbers_aggctl.tex` | `analyses/scan_aggregation/write_numbers_aggctl.py` | controlled aggregation (`\ag*`) |
| `numbers_protoret.tex` | `analyses/prioritization/write_numbers_protoret.py` | diagnostic prioritization (`\pq*`) |
| `numbers_interp.tex` | `analyses/interpretability/write_numbers_interp.py` | region occlusion (`\interp*`, `\ipm*`, `\ipp*`, `\ipn*`) |
| `numbers_static.tex` | hand-maintained | literature values; the labelled combined-manifest row count |

Checked against the final manuscript: all 307 macros its text and tables use
are produced by these files with values identical to the manuscript's, from the
shipped files alone.

## Result tables in `results/`

One suffix per task family (the class-size sweep is analysed as one family of
all seven cutoffs):

| Suffix | Family |
|---|---|
| `a1fixv3` | cleft subtyping, `A1_fix_v3_ts{1..5}` |
| `a2fixv3` | cleft screening, `A2_fix_v3_ts{1..5}` |
| `cfullv3` | combined screening, `C_full_v3_ts{1..5}` |
| `cf20fullv3` | 49-class syndrome diagnosis, `CF20_full_v3_ts{1..5}` |
| `cfv3` | class-size sweep, `CF{10,15,20,30,40,50,75}_full_v3_ts{1..5}` |

| File | Content |
|---|---|
| `per_seed_results_<s>.csv` | one row per task × encoder × head × level × inference seed × metric |
| `summary_ci_<s>.csv` | per training run: mean over inference seeds + 95% subject-bootstrap CI |
| `contrast_tests_<s>.csv` | paired contrasts (head, aggregation, encoder) per training run |
| `<s>_trainseed_per_run.csv` | one row per training run (inference seeds averaged) |
| `<s>_trainseed_summary.csv` | mean / SD / min / max over the five training runs, bootstrap CI alongside |
| `cohort_sensitivity/` | within-cohort, age/sex-adjusted and baseline screening results |
| `scan_aggregation/` | controlled same-patient aggregation (per-patient tables de-identified) |
| `prioritization/` | top-K, retrieval and limited-reference results (summaries and per-class tables) |
| `interp/tables/` | syndrome × region importance with intervals; analysis counts |
| `derived/` | class-level summaries of the predictions used by figures and macros |
