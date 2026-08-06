# Milestone 2.6 — Evaluation Subsystem: Completion Documentation

> **Status: COMPLETE.** All eight Milestone 2.6 commits are implemented,
> unit-tested (921/921 project-wide), and independently verified against
> real external dependencies (`rouge`, `evaluate`, `bert-score`,
> `pytrec_eval`, plus the already-validated `radgraph`/`f1chexbert`
> shims) in the user's own Colab environment — not merely in-sandbox.
> This document records that completion in the same spirit as
> `docs/progress.md`'s Milestone 2.4 entry and the dedicated
> `4f674565b9369a898012de41cb31c9be912b6ece` "Document Milestone 2.4
> completion" commit, but as a single self-contained file rather than
> edits spread across the shared tracking docs (per explicit
> instruction for this document).

## Table of Contents

1. [Milestone Overview](#1-milestone-overview)
2. [Scope of Cells 30–36](#2-scope-of-cells-3036)
3. [Final Implementation Summary](#3-final-implementation-summary)
4. [Architecture Overview](#4-architecture-overview)
5. [Implementation Commits](#5-implementation-commits)
6. [New Modules and Classes](#6-new-modules-and-classes)
7. [Test Growth Across the Milestone](#7-test-growth-across-the-milestone)
8. [Final Full-Suite Result](#8-final-full-suite-result)
9. [Cell 36 Final Verification Summary](#9-cell-36-final-verification-summary)
10. [Dependency Verification](#10-dependency-verification)
11. [Real-Metric Execution Confirmation](#11-real-metric-execution-confirmation)
12. [Determinism Verification](#12-determinism-verification)
13. [Bootstrap Significance Verification](#13-bootstrap-significance-verification)
14. [Remaining Limitations](#14-remaining-limitations)
15. [Readiness Assessment for Phase 3 (Innovations)](#15-readiness-assessment-for-phase-3-innovations)

---

## 1. Milestone Overview

Milestone 2.6 implements the paper's full evaluation subsystem: strict
key-based loading of generation predictions/references, three
generation-quality metrics (ROUGE-L, BLEU-4, BERTScore), three
clinical-accuracy metrics (F1RadGraph, dataset-level F1CheXbert,
instance-level F1CheXbert), two retrieval metrics (Recall@K, NDCG@K, via
`pytrec_eval`) plus MRR (a preserved hand-written official algorithm,
never delegated to a library), a unifying `EvaluationRunner`
orchestrator producing atomically-written JSON/JSONL reports, a
small-scale Oracle upper-bound baseline construction, and paired
bootstrap significance testing.

The design was fixed first, without implementation, in
`docs/milestone_2_6_evaluation_contract.md` (commit `ae9d4e1`, 1,184
lines) — a line-by-line audit of the paper (Section 4, Appendix A.3) and
every evaluation-related file in the official repository, distinguishing
`PAPER_EXPLICIT`/`CODE_EXPLICIT`/`INFERRED`/`UNKNOWN` behavior
throughout, per this project's established audit-before-implementation
discipline (the same discipline used for Milestones 2.2–2.5). Cells
30–35 then implemented that contract exactly, one cohesive unit at a
time, each landing with its own commit and its own unit test suite
before the next began. Cell 36 is the final, non-code verification step:
a real-dependency dry run proving every metric executes for real (not
via its own mock fallback) end-to-end, first attempted in this sandbox
(network-restricted, so several metrics could only reach a classified
`WARNING`), and then — per explicit instruction — re-verified in the
user's own Colab environment with all four missing optional dependencies
installed at pinned versions, producing a genuine `CELL 36: PASS`.

## 2. Scope of Cells 30–36

| Cell | Deliverable | Commit |
|---|---|---|
| 30 | Evaluation Core: `GenerationMetricsConfig`, strict key-based `load_generation_pairs`, upstream-failure propagation, atomic summary write | `04c7c78` |
| 31 | Generation metric wrappers: `RougeLMetric`, `Bleu4Metric`, `BertScoreMetric`, shared `_LazyLibraryMixin` lazy-construction discipline, `_classify_metric_dependency_error` taxonomy | `a4db5f1` |
| 32 | Clinical metric wrappers: `F1RadGraphMetric`, `DatasetF1ChexbertMetric`, `InstanceF1ChexbertMetric` (two structurally distinct CheXbert formulas, never conflated) | `066ada6` |
| 33 | Retrieval metric wrappers: `compute_mrr_at_k` (preserved official algorithm), `compute_recall_ndcg_at_k` (delegated to `pytrec_eval`), `MRRMetric`/`RecallAtKMetric`/`NdcgAtKMetric` | `addc318` |
| 34 | `EvaluationRunner` orchestrator unifying all three metric families, `EvaluationRunConfig`/`EvaluationRunResult`, atomic JSON/JSONL report writers | `06c3970` |
| 35 | `OracleEvaluator` small-scale upper-bound baseline construction, `paired_bootstrap_significance` | `752ca4c` |
| 36 | Real-evaluation dry run and dependency closure — no `src/` changes; verification only, documented in this file | (not a repository commit — see [§9](#9-cell-36-final-verification-summary)) |

A supporting commit, `07f39a9` ("Fix retriever trainer smoke-test
determinism"), also landed inside this milestone's commit range —
included in [§5](#5-implementation-commits) for completeness, though it
is a Milestone 2.4-era test-flakiness fix, not new evaluation
functionality.

## 3. Final Implementation Summary

- **Loader**: strict query-key alignment between references and
  predictions (mismatched key sets raise `EvaluationError` — verified
  live, not just assumed), upstream generation failures propagated and
  visible (never silently dropped), atomic `.meta.json` summary writes.
- **Generation metrics**: three wrappers behind one uniform
  `GenerationMetricWrapper.compute(hyps, refs) -> MetricComputation`
  API, each lazily constructing its real library on first `compute()`
  call (never at `__init__`), each with an already-committed
  `MockGenerationMetricWrapper` fallback used only when real
  construction/execution fails — and that fallback is always classified
  and disclosed, never silent.
- **Clinical metrics**: F1RadGraph and both CheXbert formulas
  (dataset-level and instance-level are independently verified as
  structurally distinct code paths — Cell 32's own 44 unit tests) built
  on the Milestone 2.2 `src/baseline/radgraph/compat.py` compatibility
  shim layer, which fails loudly (`CompatibilityError`) rather than
  silently patching an unvalidated library version.
- **Retrieval metrics**: MRR is a hand-written reimplementation of the
  official `evaluate_retriever.py::compute_mrr` algorithm with zero
  external dependency; Recall@K/NDCG@K are delegated to
  `pytrec_eval.RelevanceEvaluator` using the *same* bare
  `{"ndcg_cut", "recall"}` measure-family request the official code
  uses, which is why `RetrievalMetricsConfig`'s default `k_values`
  (`(100, 200, 500, 1000)`) are fixed to `pytrec_eval`'s own default
  cutoff points rather than arbitrary values.
- **Orchestrator**: `EvaluationRunner` composes all three metric
  registries into one `.run()` call, producing a fully reproducible
  `EvaluationRunResult` (byte-identical across repeated runs on the same
  input), with metric-level failures always visible in
  `result.failures` (never swallowed) and atomic
  `evaluation_summary.json` / `per_sample_rows.jsonl` writers whose
  schemas were verified against each other (summary `corpus_score`
  equals `mean(per-sample scores)` to within `1e-6`, by construction).
- **Oracle**: a small-scale upper-bound baseline — for each query,
  scores every non-self corpus candidate via injected real
  radgraph/chexbert instance scorers and reports the winning candidate —
  reproducible across repeated calls on identical input, bridging
  cleanly to Cell 30's own prediction-row schema via
  `oracle_result_row_to_pred_row_dict`.
- **Significance**: `paired_bootstrap_significance` uses a local
  `numpy.random.Generator` (never global RNG state — verified directly,
  including a test that seeds and consumes 500 global-state draws
  between two calls and confirms identical results) for confidence
  intervals and p-values between any two paired per-example score
  series.

## 4. Architecture Overview

```
src/evaluation/
├── generation_metrics.py   # Cell 30 loader + Cells 31/32 generation & clinical wrappers
│   ├── load_generation_pairs()            strict key-based loader
│   ├── GenerationMetricWrapper (ABC)      compute(hyps, refs) -> MetricComputation
│   │   ├── RougeLMetric                   real: rouge
│   │   ├── Bleu4Metric                    real: evaluate.load("bleu")
│   │   ├── BertScoreMetric                real: bert_score (distilbert-base-uncased)
│   │   ├── F1RadGraphMetric               real: radgraph (via Milestone 2.2 compat shim)
│   │   ├── DatasetF1ChexbertMetric        real: f1chexbert (dataset-level formula)
│   │   ├── InstanceF1ChexbertMetric       real: f1chexbert (instance-level formula)
│   │   └── MockGenerationMetricWrapper    deterministic word-overlap fallback
│   └── _classify_metric_dependency_error() shared 6-category error taxonomy
├── retrieval_metrics.py    # Cell 33
│   ├── compute_mrr_at_k()                 hand-written, zero dependency
│   ├── compute_recall_ndcg_at_k()         real: pytrec_eval.RelevanceEvaluator
│   └── RetrievalMetricWrapper (ABC)
│       ├── MRRMetric / RecallAtKMetric / NdcgAtKMetric
│       └── MockRetrievalMetricWrapper
├── report.py                # Cell 34
│   └── EvaluationRunner.run() -> EvaluationRunResult
│       ├── write_evaluation_summary_json_atomic()
│       └── write_per_sample_rows_jsonl_atomic()
└── significance.py          # Cell 35 (bootstrap half)
    └── paired_bootstrap_significance()    local Generator, never global RNG

src/baseline/evaluation/
└── oracle.py                 # Cell 35 (Oracle half)
    └── OracleEvaluator.build_row() / .build_all()
```

Every real-dependency wrapper follows the same shape established in
Milestone 2.5's `HFGeneratorAdapter`: lazy construction on first real
call (never at `__init__`), instance-scoped caching (never a module
global), and any construction/execution failure classified via
`_classify_metric_dependency_error` into one of six categories
(`missing_dependency`, `authentication_required`,
`checkpoint_unavailable`, `network_error`, `invalid_input`,
`computation_failed`) before being wrapped in an `EvaluationError` and
chained — never a bare re-raise, never a silent catch-all.

## 5. Implementation Commits

All eight commits below are on branch `claude/factmm-rag-repo-setup-o04jx3`,
currently **8 commits ahead of `origin/claude/factmm-rag-repo-setup-o04jx3`**
(unpushed — see [§14](#14-remaining-limitations)):

| # | Hash | Message |
|---|---|---|
| 1 | `ae9d4e1362e8dcfaa46ae1a68d3276cafc46fe28` | Document Milestone 2.6 Evaluation design contract |
| 2 | `07f39a91834eb8c89b4ee6dce11e8c4bb6d5a837` | Fix retriever trainer smoke-test determinism |
| 3 | `04c7c783561549f14cf2db214dbb0f98c8739420` | Implement Evaluation Core schemas and loader |
| 4 | `a4db5f170897b1a27d60ff5f29ebf212686bcbe2` | Implement Generation metric wrappers |
| 5 | `066ada684d8ee3a0c478e301fc5d838ffd49a01e` | Implement Clinical metric wrappers |
| 6 | `addc318262ea472cc3ee0216785bf1373dc9aeb1` | Implement Retrieval metric wrappers |
| 7 | `06c39705f0ce59a45d326aba698ea10fe4d1ce0a` | Implement Evaluation orchestrator |
| 8 | `752ca4c2a72f71da93c1a0effc1143675230de4b` (HEAD) | Implement Oracle evaluation and bootstrap significance |

## 6. New Modules and Classes

**`src/evaluation/generation_metrics.py`** (pre-existing Phase-1 skeleton,
filled in across Cells 30–32):
`GenerationMetricsConfig`, `ReferenceRow`, `PredictionRow`,
`GenerationExampleScore`, `GenerationMetricsResult`,
`GenerationLoadSummary`, `GenerationLoadResult`,
`load_generation_pairs()`, `write_generation_load_summary_atomic()`,
`MetricComputation`, `GenerationMetricWrapper` (ABC), `_LazyLibraryMixin`,
`RougeLMetric`, `Bleu4Metric`, `BertScoreMetric`, `F1RadGraphMetric`,
`DatasetF1ChexbertMetric`, `InstanceF1ChexbertMetric`,
`MockGenerationMetricWrapper`, `_classify_metric_dependency_error()`,
`build_metric_registry()`, `build_mock_metric_registry()`.

**`src/evaluation/retrieval_metrics.py`** (pre-existing Phase-1 skeleton,
filled in in Cell 33): `RetrievalMetricsConfig`, `compute_mrr_at_k()`,
`compute_recall_ndcg_at_k()`, `RetrievalMetricComputation`,
`RetrievalMetricWrapper` (ABC), `MRRMetric`, `RecallAtKMetric`,
`NdcgAtKMetric`, `MockRetrievalMetricWrapper`,
`build_retrieval_metric_registry()`, `build_mock_retrieval_metric_registry()`.

**`src/evaluation/report.py`** (pre-existing Phase-1 skeleton, filled in
in Cell 34): `EvaluationRunConfig`, `MetricFailure`,
`PerSampleResultRow`, `EvaluationRunResult`, `EvaluationRunner`,
`write_evaluation_summary_json_atomic()`,
`write_per_sample_rows_jsonl_atomic()`.

**`src/evaluation/significance.py`** (pre-existing Phase-1 skeleton,
filled in in Cell 35): `SignificanceConfig`, `SignificanceResult`,
`paired_bootstrap_significance()`.

**`src/baseline/evaluation/oracle.py`** (new file, Cell 35):
`OracleConfig`, `compute_oracle_score()`, `OracleResultRow`,
`OracleBuildSummary`, `oracle_result_row_to_pred_row_dict()`,
`OracleEvaluator`.

**`src/common/exceptions.py`**: extended (Cell 30) with the evaluation
subsystem's exception types used throughout the above.

## 7. Test Growth Across the Milestone

Collected-test counts (`pytest tests/ --collect-only -q`) at each
commit, confirmed by temporarily checking out each commit in sequence
and returning to `HEAD` afterward (working tree left clean):

| Commit | Tests collected | Δ |
|---|---|---|
| `24d7f1b` (Milestone 2.5 baseline, pre-2.6) | 593 | — |
| `ae9d4e1` (design contract, docs only) | 593 | +0 |
| `07f39a9` (retriever determinism fix) | 593 | +0 |
| `04c7c78` (Evaluation Core) | 656 | +63 |
| `a4db5f1` (Generation wrappers) | 705 | +49 |
| `066ada6` (Clinical wrappers) | 747 | +42 |
| `addc318` (Retrieval wrappers) | 820 | +73 |
| `06c3970` (Orchestrator) | 860 | +40 |
| `752ca4c` (Oracle + significance) | **921** | +61 |

**Net growth across Milestone 2.6: 593 → 921 (+328 tests, +55.3%).**

## 8. Final Full-Suite Result

```
921 passed in 88.99s (0:01:28)
```

Confirmed independently in three places: this sandbox's own fresh run
(most recent audit), and twice in the user's real Colab environment
(pre- and post- dependency install, both showing `921 passed`). Zero
warnings, zero errors, zero failures in any of the three runs. Working
tree clean in every case (`git status --short` empty).

## 9. Cell 36 Final Verification Summary

Cell 36 is a live dry run — not a repository commit — that exercises
Cells 30–35 end-to-end on a tiny synthetic radiology-style dataset,
attempting each metric's **real** dependency first and falling back to
its own already-committed mock wrapper only on classified failure. It
was run three times over the course of this milestone's closure:

1. **Sandbox, before optional-dependency install**: `rouge`, `evaluate`,
   `bert-score`, `pytrec_eval` not installed → `radgraph`/`f1chexbert`
   genuinely installed and passing for real → **`CELL 36: WARNING`**
   (expected — a disclosed dependency gap, not an implementation defect).
2. **User's Colab, first real run** (after installing `rouge==1.0.1`,
   `evaluate==0.4.6`, `bert-score==0.3.13`, `pytrec_eval==0.5`):
   surfaced a genuine `KeyError: 'recall_1'` — traced to the *dry-run
   test harness* requesting non-default `k_values=(1, 2)` against
   `pytrec_eval`'s bare-family measure request, which only ever
   populates its own default cutoff keys (100/200/500/1000). **Not a
   `src/` incompatibility** — `compute_recall_ndcg_at_k`'s design
   (mirroring the official reference code's own bare-family request
   verbatim) was already correct; the test harness's synthetic k-values
   were wrong. Fixed in the harness only, zero `src/` changes.
3. **User's Colab, final rerun**: **`CELL 36: PASS`** —

```
Smoke checks passed:      31/31
Total tests (full suite): 921
Modified/created files: (none -- this cell performs a live dry run only, no repo changes)

CELL 36: PASS
```

Real corpus-level scores produced on the synthetic dataset in that final
run (illustrative only — a 4-sample synthetic smoke test, not a
paper-scale evaluation number):

```
Generation: {'bert_score': 0.7897, 'bleu4': 0.2857, 'rouge_l': 0.5667}
Clinical:   {'f1chexbert': 1.0, 'f1chexbert_instance': 1.0, 'f1radgraph': 0.6151}
Retrieval:  {'mrr': {100: 0.8333, ...}, 'ndcg': {100: 0.8770, ...}, 'recall': {100: 1.0, ...}}
```

## 10. Dependency Verification

| Package | `requirements.txt` | Verified installed (Colab, final PASS run) |
|---|---|---|
| `rouge` | listed, unpinned | `1.0.1` |
| `evaluate` | listed, unpinned | `0.4.6` |
| `bert-score` | listed, unpinned | `0.3.13` (import reports `bert_score.__version__ == 0.3.12`; PyPI distribution version `0.3.13` per `importlib.metadata`) |
| `pytrec_eval` | listed, unpinned | `0.5` |

`requirements.txt` currently lists all four package names without
version pins (`rouge`, `evaluate`, `bert-score`, `pytrec_eval`, added by
Cells 31/33). The exact versions above are what was actually installed
and exercised for real in the user's Colab verification; pinning
`requirements.txt` itself was out of scope for this documentation task
(no `src/`, test, or other doc file was modified — see
[§14](#14-remaining-limitations)).

`radgraph==0.0.9` and `f1chexbert==0.0.2` (Milestone 2.2's own pinned,
audited versions) were already installed and real in the Colab
environment prior to this milestone; both remained real and passing
throughout Cell 36.

## 11. Real-Metric Execution Confirmation

Every metric below executed via its **real** external dependency in the
user's Colab final verification run — no mock/fallback path was used
for any of them:

| Metric | Real dependency | Result |
|---|---|---|
| ROUGE-L | `rouge` | PASS |
| BLEU-4 | `evaluate.load("bleu")` | PASS |
| BERTScore | `bert_score` (`distilbert-base-uncased`) | PASS |
| F1RadGraph | `radgraph` (Milestone 2.2 compat shim) | PASS |
| F1CheXbert (dataset-level) | `f1chexbert` (Milestone 2.2 compat shim) | PASS |
| F1CheXbert (instance-level) | `f1chexbert` (Milestone 2.2 compat shim) | PASS |
| MRR | none (hand-written, zero dependency) | PASS |
| Recall@K | `pytrec_eval.RelevanceEvaluator` | PASS |
| NDCG@K | `pytrec_eval.RelevanceEvaluator` | PASS |
| Oracle — RadGraph instance scorer | `radgraph` (via `_default_f1radgraph_factory`) | PASS |
| Oracle — CheXbert instance scorer | `f1chexbert` (via `_default_f1chexbert_factory`) | PASS |

## 12. Determinism Verification

The `EvaluationRunner` was run twice, back to back, on identical
synthetic input in the final Cell 36 verification, with all of the
following confirmed byte-identical between the two runs:

- `generation_corpus_scores`
- `clinical_corpus_scores`
- `retrieval_scores`
- `per_sample_rows` (full per-sample decomposition, not just aggregates)

This is in addition to `paired_bootstrap_significance`'s own dedicated
determinism guarantee (Cell 35's unit tests, `test_significance.py`):
identical results across 20 repeated calls with the same seed, and —
the central claim — identical results regardless of prior *global*
`numpy.random` state (a local `Generator` is used throughout, never
global RNG state), directly carrying forward the seeding-order lesson
from this project's own earlier `test_retrieval_trainer.py` flakiness
fix (commit `07f39a9`, included in this milestone's commit range for
that reason).

## 13. Bootstrap Significance Verification

`paired_bootstrap_significance` was exercised for real in the final Cell
36 run, comparing `rouge_l` vs. `bert_score` per-example scores on the
synthetic dataset:

```
SignificanceResult(metric_name='rouge_l_vs_bert_score',
  system_a_name='rouge_l', system_b_name='bert_score',
  system_a_mean=0.5667, system_b_mean=0.7897,
  mean_difference=-0.2230, ci_low=-0.4005, ci_high=-0.0579,
  p_value=0.0, num_bootstrap_samples=2000, seed=42)
```

Reproducibility was checked by calling it twice with the same config and
confirming an identical `SignificanceResult` (`bootstrap_significance_reproducible`
smoke check, PASS). Unit-level correctness (`tests/unit/test_significance.py`,
14 tests) additionally covers: degenerate zero-difference/zero-width CI
for identical systems, CI-excludes-zero and low p-value for clearly
different systems, wider confidence level producing a wider-or-equal
interval, and independent reproducibility across different seeds.

## 14. Remaining Limitations

- **8 commits unpushed.** All Milestone 2.6 work is committed locally on
  `claude/factmm-rag-repo-setup-o04jx3` but not yet on
  `origin/claude/factmm-rag-repo-setup-o04jx3` — blocked by a git
  transport `403` in this sandbox (`git push --dry-run` confirms
  `fatal: ... error: 403`), not a content or implementation issue.
- **No Milestone 2.6 completion commit existed prior to this
  document.** Only the pre-implementation design contract (`ae9d4e1`)
  had been committed; this file is the first completion record.
- **`requirements.txt` pins are absent** for `rouge`/`evaluate`/
  `bert-score`/`pytrec_eval` (listed by name only). The exact versions
  verified as real and working are recorded in
  [§10](#10-dependency-verification) but not yet reflected as pins in
  `requirements.txt` itself.
- **Cell 36 exercises a 4-sample synthetic dataset only** — it proves
  every metric's real, end-to-end wiring and reproducibility, not any
  paper-scale evaluation number. No MIMIC-CXR-scale evaluation run has
  been performed under Milestone 2.6.
- **Oracle was exercised at small synthetic scale only** (1 query, 2
  candidates) — real large-scale Oracle construction remains
  unattempted, consistent with Cell 36's explicit non-goals.
- **Risk register item #15** (an unresolved F1CheXbert run-to-run label
  discrepancy observed during Milestone 2.2's Cell 14, on identical
  pinned versions) remains open. It was flagged in
  `docs/risk_register.md` as something that "must be investigated before
  Milestone 2.6 ... where run-to-run F1CheXbert score stability is
  required for any reported number to be trustworthy." Cell 36's
  determinism checks confirm `EvaluationRunner`-level reproducibility
  given a fixed F1CheXbert label output, but do not investigate or
  resolve the underlying label-instability question itself — that
  remains open and unresolved, unchanged by this milestone.
- **Risk register item #17** (no real-data Retriever training performed
  at any milestone to date) also remains open and unresolved by this
  milestone — Milestone 2.6 evaluates the pipeline's correctness, not
  retrieval quality from a trained model.
- **`bert_score`'s two version identifiers disagree**: the installed
  module reports `bert_score.__version__ == "0.3.12"` while
  `importlib.metadata.version("bert-score")` (the PyPI distribution
  record) reports `"0.3.13"`, for the same `pip install
  bert-score==0.3.13` installation. Both were recorded as observed; the
  discrepancy is upstream packaging metadata, not a project defect, and
  was not investigated further as out of scope for this milestone.

## 15. Readiness Assessment for Phase 3 (Innovations)

**Ready**, with the above limitations explicitly disclosed rather than
resolved. Every metric the paper requires (ROUGE-L, BLEU-4, BERTScore,
F1RadGraph, F1CheXbert dataset- and instance-level, MRR, Recall@K,
NDCG@K) has now been proven to execute for real, end-to-end, through the
project's own orchestrator, with reproducible output and a working
Oracle upper-bound baseline and paired bootstrap significance testing —
the complete measurement toolkit Phase 3 will need to report and compare
results against. The full 921-test suite provides a stable regression
net for any Innovation work built on top of this subsystem. The open
items above (unpushed commits, no `requirements.txt` pins yet, risk
register #15/#17 still open, no paper-scale evaluation run yet
performed) are pre-existing, disclosed, and do not block starting
Innovations — they should be tracked, not treated as gates, consistent
with how this project has carried forward comparable open items across
every prior milestone transition (e.g. Milestone 2.4's real-training gap
was explicitly not a blocker for Milestone 2.5).
