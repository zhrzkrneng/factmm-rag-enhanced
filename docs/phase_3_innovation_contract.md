# Phase 3 — Innovation Design Contract

> **DESIGN CONTRACT — NOT EXECUTABLE CODE.** Every class, function, and
> config shown below is a signature-level specification, not implemented
> source. No file under `src/baseline/`, `src/evaluation/`, `src/common/`,
> `src/data/`, `src/retrieval/`, `src/generation/`, or `tests/` is created
> or modified by this document. This contract grounds every design
> decision in the now-complete, real-verified baseline (Milestones
> 2.1–2.6, `CELL 36: PASS`, 921/921 tests) and in this project's own
> pre-existing `docs/innovation_proposals.md` design sketches. No claim
> of improvement is made anywhere in this document — every "success"
> criterion below is a falsifiable, untested hypothesis pending real
> experiments that have not yet run. Implementation does not begin until
> this contract is approved.

## Table of Contents

1. [Status & Scope](#1-status--scope)
2. [Baseline Foundation — What Phase 3 Inherits](#2-baseline-foundation--what-phase-3-inherits)
3. [Task-Framing Decision: Radiology Report Generation](#3-task-framing-decision-radiology-report-generation)
4. [Design Principles](#4-design-principles)
5. [Architecture Overview](#5-architecture-overview)
6. [Module Layout & Responsibility Boundaries](#6-module-layout--responsibility-boundaries)
7. [Core Innovation 1 — Iterative Evidence Refinement](#7-core-innovation-1--iterative-evidence-refinement)
8. [Supporting Innovation 2 — Confidence-Aware Retrieval](#8-supporting-innovation-2--confidence-aware-retrieval)
9. [Supporting Innovation 3 — Adaptive Retrieval Depth](#9-supporting-innovation-3--adaptive-retrieval-depth)
10. [Supporting Innovation 4 — Evidence-Aware Prompt Construction](#10-supporting-innovation-4--evidence-aware-prompt-construction)
11. [Supporting Innovation 5 — Self-Verification and Correction](#11-supporting-innovation-5--self-verification-and-correction)
12. [Supporting Innovation 6 — Cross-Modal Evidence Fusion](#12-supporting-innovation-6--cross-modal-evidence-fusion)
13. [Baseline/Innovation Separation & Anti-Contamination Protocol](#13-baselineinnovation-separation--anti-contamination-protocol)
14. [Branch Strategy](#14-branch-strategy)
15. [Experiment Naming & Versioning](#15-experiment-naming--versioning)
16. [Reproducibility Metadata](#16-reproducibility-metadata)
17. [Comparison Tables (Reserved Shape, Not Populated)](#17-comparison-tables-reserved-shape-not-populated)
18. [Bootstrap Significance Plan](#18-bootstrap-significance-plan)
19. [Error-Analysis Protocol](#19-error-analysis-protocol)
20. [Known Design Conflicts With Current Baseline](#20-known-design-conflicts-with-current-baseline)
21. [Open Questions (`UNKNOWN`)](#21-open-questions-unknown)

---

## 1. Status & Scope

Phase 3 (Innovations) begins only after Milestone 2.6 (Evaluation) was
independently verified real (`CELL 36: PASS`, 921/921 tests, 31/31 smoke
checks, real `rouge`/`evaluate`/`bert-score`/`pytrec_eval` plus
already-real `radgraph`/`f1chexbert`, no mock/fallback path used) — see
`docs/milestone_2_6_completion.md`. This document is the **audit-first
design contract** for Phase 3, produced before any innovation code is
written, per this project's established discipline (the same discipline
used for `docs/milestone_2_5_generator_contract.md` and
`docs/milestone_2_6_evaluation_contract.md`).

**No file under `src/baseline/`, `src/evaluation/`, `src/common/`,
`src/data/`, `src/retrieval/`, `src/generation/`, or `tests/` is
touched by this document.** The only new file is this contract itself.
Implementation of any innovation module begins only after this contract
is approved, one module at a time, following the same discipline used
for Cells 30–36.

This contract defines six innovation modules — one core, five
supporting — each specified in full (research hypothesis through
falsification criterion) before any of them are implemented. It also
defines the cross-cutting protocol every innovation must follow:
strict separation from baseline code, anti-contamination of baseline
results, branch strategy, experiment naming, reproducibility metadata,
the (currently empty) comparison-table shape, the bootstrap
significance plan, and the error-analysis protocol.

## 2. Baseline Foundation — What Phase 3 Inherits

Everything below is real, verified, unmodified, and treated as a fixed,
read-only dependency by every innovation in this contract:

| Layer | Module | Verified state |
|---|---|---|
| Retrieval | `src.baseline.retrieval.model.MultiModalRetriever` (`encode_images_only`/`encode_text_only`/`encode_images_with_text`/`scaled_similarity`) | Real CLIP-ViT + T5 forward pass verified against actual Hugging Face weights (Milestone 2.4G) |
| Retrieval index | `src.retrieval.index.FaissFlatIPIndex` | Exact FlatIP, matches official code's own index type |
| Generation | `src.baseline.generation.prompt_builder.PromptBuilder`/`PromptBuilderConfig`/`PromptResult`, `src.baseline.generation.adapter.GeneratorAdapter`/`HFGeneratorAdapter` | Real HF adapter + checkpoint compatibility verified (Milestone 2.5) |
| Evaluation | `src.evaluation.generation_metrics.*` (ROUGE-L, BLEU-4, BERTScore, F1RadGraph, F1CheXbert dataset/instance), `src.evaluation.retrieval_metrics.*` (MRR, Recall@K, NDCG@K), `src.evaluation.report.EvaluationRunner` | All real, `CELL 36: PASS` (Milestone 2.6) |
| Oracle | `src.baseline.evaluation.oracle.OracleEvaluator`, `compute_oracle_score` | Real radgraph/f1chexbert instance scorers verified (Milestone 2.6, Cell 35) |
| Significance | `src.evaluation.significance.paired_bootstrap_significance` | Real, deterministic, local-`Generator`-based (Milestone 2.6, Cell 35) |
| Compatibility shims | `src.baseline.radgraph.compat` (`CompatibilityError`) | Real `radgraph==0.0.9`/`f1chexbert==0.0.2` (Milestone 2.2) |

**Pre-existing Phase-1 innovation skeleton** (docstring-only stubs,
zero implementation, from the "Add Phase 1 project architecture"
commit), referenced and extended by this contract:

```
src/innovation/
├── adaptive_retrieval/adaptive_k.py      -> Supporting Innovation 3
├── confidence_gating/gate.py             -> Supporting Innovation 2
├── factual_reranking/reranker.py         -> not in scope for this contract (see §20.3)
├── multi_report_fusion/fusion.py         -> Supporting Innovation 6
├── uncertainty/detector.py               -> Supporting Innovation 5
└── evaluation/innovation_metrics.py      -> §17 (comparison-table generator)
```

`docs/innovation_proposals.md`'s Innovations A–F are this project's
own earlier design sketches; this contract formalizes and extends four
of them (A→§9, D→§8, C→§12, E→§11) into full module specifications,
adds one genuinely new core module (§7, not previously proposed), and
adds one new supporting module with no prior sketch (§10). Innovation
B (reranking) and Innovation F (longitudinal context) remain out of
scope — see §20.3 and `docs/innovation_proposals.md` respectively.

## 3. Task-Framing Decision: Radiology Report Generation

### 3.1 Decision

**RESOLVED.** The project continues as:

> **Iterative Multimodal Retrieval-Augmented Generation for Radiology
> Report Generation**

A stronger alternative title, offered as an optional variant carrying
the same scope and decision below:

> **Fact-Aware Iterative Multimodal Retrieval-Augmented Generation for
> Radiology Report Generation**

| Path | Status |
|---|---|
| **Path A — Radiology Report Generation** | **SELECTED** |
| Path B — Medical Visual Question Answering | **DEFERRED / OUT OF SCOPE** |

This decision is final for the current thesis implementation and
applies to every innovation module in this contract (§7–§12). It
supersedes the "Medical Question Answering" framing named in an earlier
draft of this document — see §3.5 for the historical record of why that
path was considered and why it was deferred.

### 3.2 What this means concretely

- The verified FactMM-RAG baseline (Milestones 2.1–2.6) **is a
  radiology report generation system** — given an image (+ optional
  prior text), it generates a finding/impression text, scored against a
  reference report.
- **MIMIC-CXR and CheXpert are used in the report-generation setting**
  — the same setting the paper itself uses, and the same setting this
  project's baseline was built and verified against.
- **The output of every innovation module is a generated radiology
  report — never a question/answer pair.**
- **The existing evaluation stack (§2) remains valid without schema
  changes.** ROUGE-L, BLEU-4, BERTScore, F1RadGraph, F1CheXbert
  (dataset and instance), MRR, Recall@K, and NDCG@K — all real,
  verified `CELL 36: PASS` — are reused unmodified by every innovation
  below, with no new metric family required by this decision.
- **No QA dataset, QA schema, or QA-specific metric will be introduced
  in Phase 3.**

### 3.3 Rationale

- Direct compatibility with the completed, real-verified baseline — no
  new data pipeline, retriever, generator, or evaluation subsystem is
  required.
- Direct comparison with the original FactMM-RAG paper's own reported
  results and metrics (Table 1 and Appendix A.3), since both use the
  same task and the same metric family.
- Lower implementation risk — every innovation module in this contract
  composes with already-real, already-verified components (§2) rather
  than requiring new, unvalidated infrastructure.
- No need to rebuild dataset schemas or metrics — Cell 30's strict
  key-based loader, `PredictionRow`/`pred.jsonl`, and every metric
  wrapper (Milestone 2.6) apply unchanged.
- Clearer ablation and significance analysis — every comparison reuses
  `paired_bootstrap_significance` (§18) on the same metric family the
  baseline was already validated against, with no new metric whose
  statistical behavior is untested.
- Stronger reproducibility — the full reproducibility discipline
  established across Milestones 2.1–2.6 (atomic writes, deterministic
  reruns, classified fallback, real-dependency verification) transfers
  directly, with nothing new to re-validate at the data/metric level.

### 3.4 Terminology

Every innovation module in this contract (§7–§12) uses this vocabulary
consistently:

**Use**: input image, retrieved reports/evidence, draft radiology
report, evidence gap, second retrieval, fused evidence, final radiology
report.

**Do not use** (except in §3.5, the historical record explaining why
Path B was rejected): question, answer, QA pair, VQA, medical question
answering.

### 3.5 Historical note: why Medical Visual Question Answering was considered and deferred

An earlier draft of this contract explored **"Medical Question
Answering"** as the thesis framing, on the reasoning that "what does
this image show" could be treated as an implicit question and the
existing report-generation metrics reused as the quality signal without
change. That framing was evaluated and explicitly rejected for the
current thesis implementation: the verified baseline (Milestones
2.1–2.6) has no question/answer-pair data, no QA-specific evaluation
metric (exact match, answer accuracy, span-F1, etc.), and no
QA-formatted dataset anywhere in the project, and adopting a genuine,
separately-evaluated VQA task would have required its own audit-first
contract revision, new dataset schema work, and new, unvalidated
metrics — directly contradicting the rationale in §3.3.

**True Medical VQA may be explored as a future independent extension**,
outside the current thesis, if a later phase's scope and available data
support it. It is explicitly **not** part of the current thesis
implementation, and no innovation module in this contract (§7–§12)
depends on it or is designed around it.

## 4. Design Principles

1. **Read-only dependency on the baseline.** Every innovation module may
   call baseline/evaluation code through its existing public API. No
   innovation module may modify, monkeypatch, or fork baseline logic.
2. **Disclosed, classified fallback — never silent.** Every innovation
   step that depends on a real external library or a real baseline
   component reuses the same six-category error taxonomy
   (`_classify_metric_dependency_error`, Milestone 2.6) and the same
   discipline established across Cells 31–36: real path attempted
   first, fallback only on classified failure, always disclosed.
3. **Bounded, not agentic.** Every "iterative" or "self-correcting"
   mechanism in this contract has an explicit, small, fixed upper bound
   on rounds/attempts (specified per module below) — never an unbounded
   loop.
4. **Deterministic by construction wherever no model sampling is
   involved.** Pure decision/fusion/scoring logic must be
   RNG-free and byte-identical across reruns, verified the same way
   Milestone 2.6 verified `EvaluationRunner` determinism (rerun ×2,
   compare byte-for-byte).
5. **Validation-only tuning.** No innovation threshold, weight, or
   bound may be chosen by looking at test-split data — enforced as a
   code-level discipline (§13) as well as a manual review step (§19).
6. **No improvement claimed before experiments.** Every "what counts as
   success" / "what would falsify" pair below is a pending hypothesis.
   This contract produces zero result numbers.

## 5. Architecture Overview

```
                         ┌─────────────────────────────┐
                         │   Baseline (read-only, §2)   │
                         │ Retrieval · Generation ·      │
                         │ Evaluation · Oracle ·         │
                         │ Significance                  │
                         └───────────────┬───────────────┘
                                         │ (public API only)
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
┌───────▼────────┐              ┌────────▼─────────┐            ┌─────────▼────────┐
│ Supporting       │              │ Core Innovation 1  │            │ Supporting        │
│ Innovation 2      │◄────────────┤ Iterative Evidence  ├───────────►│ Innovation 6       │
│ Confidence-Aware   │  gate      │ Refinement          │  fused    │ Cross-Modal        │
│ Retrieval          │  decision  │ (retrieval round 1  │  evidence │ Evidence Fusion     │
│                     │            │  -> draft radiology │           │                     │
└────────────────────┘  feeds     │  report -> evidence  │  feeds    └─────────┬──────────┘
                          round 1  │  gap -> retrieval    │  Innov. 4           │
┌────────────────────┐   sizing   │  round 2 -> fused    │                     │
│ Supporting          │◄───────────┤  evidence -> final   │                     │
│ Innovation 3         │            │  radiology report)   │                     │
│ Adaptive Retrieval    │           └─────────┬─────────────┘                    │
│ Depth                 │                     │                                  │
└───────────────────────┘                     │                                  │
                                                │                                  │
┌─────────────────────────────┐        ┌───────▼─────────────────────────────────▼──┐
│ Supporting Innovation 4       │◄───────┤ (fused evidence, from Innov. 1 or 6)       │
│ Evidence-Aware Prompt          │        └─────────────────────────────────────────┘
│ Construction                   │
└───────────────┬─────────────────┘
                │ prompt -> real HFGeneratorAdapter (unmodified)
┌───────────────▼─────────────────┐
│ Supporting Innovation 5           │
│ Self-Verification and Correction   │
│ (reuses Oracle's real instance      │
│  scorers, unmodified)               │
└────────────────────────────────────┘
```

Each innovation is independently testable and independently
switchable off (returning exactly the baseline's own behavior) — no
innovation module requires another to be present, though Core
Innovation 1 is designed to *compose with* Innovations 2/3/4/6 as
optional injected strategies (never a hard dependency; see each
module's own config).

## 6. Module Layout & Responsibility Boundaries

```
src/innovation/
├── iterative_refinement/        # Core Innovation 1 (new)
│   ├── config.py
│   ├── pipeline.py
│   ├── gap_detector.py
│   └── fusion.py
├── confidence_gating/           # Supporting Innovation 2 (fills existing stub)
│   ├── config.py
│   ├── gate.py
│   └── calibration.py
├── adaptive_retrieval/          # Supporting Innovation 3 (fills existing stub)
│   ├── config.py
│   └── adaptive_k.py
├── evidence_aware_prompting/    # Supporting Innovation 4 (new)
│   ├── config.py
│   └── builder.py
├── self_verification/           # Supporting Innovation 5 (extends existing stub)
│   ├── config.py
│   ├── verifier.py
│   └── corrector.py
├── multi_report_fusion/         # Supporting Innovation 6 (fills + extends existing stub)
│   ├── config.py
│   ├── fusion.py
│   └── modality_attribution.py
├── evaluation/
│   └── innovation_metrics.py    # extended: comparison-table generator, §17
├── factual_reranking/           # out of scope for this contract (§20.3)
└── uncertainty/                 # superseded in role by self_verification/ (§11); stub retained, unmodified
```

Every new file under `src/innovation/` is additive only. No existing
stub file's current docstring-only content is deleted by this
contract — implementation fills them in, per module, one at a time,
exactly as Cells 30–35 filled in the pre-existing `src/evaluation/*.py`
skeletons.

---

## 7. Core Innovation 1 — Iterative Evidence Refinement

**Stages**: first retrieval → draft radiology report → evidence-gap
detection → second retrieval → fused evidence → final radiology
report.

**Research hypothesis**: A single-shot retrieve-then-generate pipeline
cannot correct for retrieval misses that only become apparent after
seeing a draft radiology report. Performing a second, gap-targeted
retrieval — conditioned on gaps detected in the draft radiology report
— and fusing both rounds' evidence before generating the final
radiology report will improve clinical-accuracy metrics (F1RadGraph,
F1CheXbert) and, at minimum, hold fluency metrics (ROUGE-L, BLEU-4,
BERTScore) steady, relative to the single-shot baseline, with
round-2-vs-round-1 retrieval quality (Recall@K, NDCG@K, MRR) reported
as a supporting diagnostic — at a disclosed and bounded additional
compute cost.

**Exact baseline limitation addressed**: the baseline's generation path
(`RAGDatasetBuilder` → `PromptBuilder` → `HFGeneratorAdapter`, Milestone
2.5) retrieves once and never revisits retrieval after generation — no
feedback loop from generated output back to retrieval exists anywhere
in Milestones 2.4/2.5.

**Input/output contract**:
- Input: `IterativeRefinementRequest` — the same image/text query
  inputs `RAGDatasetBuilder`/`PromptBuilder` already consume, plus
  injected (not owned) handles to a `FaissFlatIPIndex`,
  `MultiModalRetriever`, and `GeneratorAdapter`.
- Output: `IterativeRefinementResult` — the final radiology report
  text, both rounds' retrieved evidence, the `EvidenceGapReport`, a
  fusion trace, and `rounds_used` (1 or 2). Bridges cleanly to Cell
  30's `PredictionRow`/`pred.jsonl` schema (reused, not reinvented) so
  output can be scored by the unmodified evaluation subsystem.

**Module layout**: `src/innovation/iterative_refinement/{config.py,
pipeline.py, gap_detector.py, fusion.py}` (see §6).

**Dataclasses/configs**: `IterativeRefinementConfig` (frozen:
`max_rounds` fixed at 2 for this contract, `gap_detection_threshold`,
`second_retrieval_k`, `fusion_strategy` enum, `seed`);
`IterativeRefinementRequest`; `EvidenceGapReport`;
`IterativeRefinementResult`.

**Deterministic behavior**: given identical weights, identical corpus
index, and identical seed, two runs on the same query must produce
byte-identical intermediate artifacts (round-1 ranking, gap report,
round-2 ranking, fusion output) and final text — verified via rerun ×2
byte-comparison, the same method used for `EvaluationRunner` in
Milestone 2.6. The underlying HF generator's decoding must be pinned
(greedy or seeded) for this claim to hold, per Milestone 2.5's own
`HFGeneratorAdapter` discipline.

**Stopping criteria**: exactly 2 rounds maximum. Round 2 triggers only
if the gap score exceeds `gap_detection_threshold`; otherwise the
pipeline stops after round 1, and its output in that case must be
provably identical to the plain baseline single-shot path — itself a
tested invariant.

**Failure modes**: gap detector's dependency fails →
classified via the shared taxonomy, pipeline falls back to baseline
single-shot output (never crashes); round-2 retrieval returns zero new
candidates → logged, pipeline stops at round-1 output; fusion produces
an unresolved contradiction between round-1 and round-2 evidence → the
contradiction is preserved and disclosed, never silently resolved by
arbitrary selection.

**Unit-testing strategy**: pure synthetic fakes for retriever/generator
(matching `MockGeneratorAdapter`'s existing discipline) — no real model
required. Test matrix: (a) no-gap path exactly matches single-shot
baseline output, (b) gap-triggered path performs exactly one second
retrieval call, never more, (c) determinism (rerun ×2), (d) each
failure mode above has a dedicated forced-failure test asserting
graceful, classified degradation, (e) fusion conflict-preservation
test.

**Colab execution plan**: a dry-run cell (numbering assigned at
implementation time), same discipline as Cell 36 — synthetic tiny data
first; only with explicit approval, a small real-model smoke test
reusing Milestone 2.4G's real retriever weights and Milestone 2.5's
real HF generator adapter, never a paper-scale run at this stage.

**Ablation design**: baseline (single-shot) vs. iterative
(gap-triggered, 2-round max) vs. iterative-forced (always 2 rounds
regardless of detected gap) — isolating "does round 2 help when
triggered" from "does gap-triggering itself add value over always
doing round 2."

**Evaluation metrics**: F1RadGraph, F1CheXbert (both formulations),
ROUGE-L, BLEU-4, BERTScore (all reused unmodified from Milestone 2.6);
Recall@K/NDCG@K/MRR computed separately for round-1-only vs.
round-1+2-combined evidence pools; a new gap-detection precision/recall
diagnostic, reported as unvalidated unless/until a held-out
human-labeled gap set exists.

**Computational cost**: up to 2× retrieval and 2× generation calls per
query, only for queries where round 2 actually triggers. Must report,
per experiment: mean rounds used and % of queries triggering round 2.

**Expected risks**: gap detector miscalibration (over/under-triggering);
round-2 retrieval returning near-duplicates of round 1 rather than
genuinely new evidence; fusion introducing contradictions the generator
cannot resolve; cost/latency not justified by any accuracy gain.

**What counts as success**: on held-out validation data, the
gap-triggered arm statistically significantly outperforms the baseline
on at least one of F1RadGraph/F1CheXbert (bootstrap CI excludes 0, via
`paired_bootstrap_significance` unmodified) without a significant
regression on any other reused metric, at disclosed bounded cost.

**What would falsify the hypothesis**: no clinical metric improves
significantly; any metric regresses significantly; the gap-triggered
subset shows no measurable difference from queries that never trigger
round 2 (implying the detector isn't finding real opportunities); or
round-2 evidence is empirically redundant with round-1 evidence in the
majority of triggered cases (near-zero new information at fusion time).

---

## 8. Supporting Innovation 2 — Confidence-Aware Retrieval

*(formalizes `docs/innovation_proposals.md` Innovation D)*

**Research hypothesis**: a calibrated confidence signal over the
retrieval score distribution (top-1 similarity, top-1-vs-top-2 margin,
or score entropy) can identify queries where retrieval is unlikely to
help, and gating those queries to non-RAG generation will not
significantly harm — and may improve — the generated radiology
report's quality (ROUGE-L, BLEU-4, BERTScore, F1RadGraph, F1CheXbert)
relative to always retrieving unconditionally, while measurably
reducing total retrieval calls (fewer real Recall@K/NDCG@K/MRR
evaluations needed per run, reported as a supporting diagnostic).

**Exact baseline limitation addressed**: the baseline always retrieves
exactly one report per query, unconditionally — the official pipeline
and this project's reproduction of it have no retrieve/no-retrieve
decision point anywhere.

**Input/output contract**:
- Input: ranked candidates + scores from `MultiModalRetriever`/
  `FaissFlatIPIndex` search (unmodified).
- Output: `RetrievalConfidenceDecision` (bool retrieve/no-retrieve,
  confidence score, per-signal breakdown, threshold used).

**Module layout**: `src/innovation/confidence_gating/{config.py,
gate.py, calibration.py}`.

**Dataclasses/configs**: `ConfidenceGatingConfig` (signal-choice enum,
threshold, calibration split identifier); `RetrievalConfidenceDecision`.

**Deterministic behavior**: pure function of `(scores, config)`, no
RNG — trivially deterministic; tested via repeated-call identity.

**Stopping criteria**: single decision point per query, computed once,
never retried.

**Failure modes**: degenerate score distribution (empty or
all-identical candidate scores) → defined fallback (always retrieve,
a disclosed default bias), never a divide-by-zero crash; threshold
accidentally calibrated on the wrong (test) split → prevented as a
code-level invariant (§13), tested explicitly.

**Unit-testing strategy**: monotonicity (higher margin → more likely to
retrieve, for the margin signal); edge cases (single candidate, empty
candidate list); determinism; a test asserting the calibration
threshold is loaded from a frozen validation-tuned artifact, never an
inline literal computed from test data.

**Colab execution plan**: synthetic dry run computing gating decisions
over Milestone 2.4-exported real/synthetic embeddings — no new model
training required.

**Ablation design**: always-retrieve (baseline) vs. always-no-retrieve
(non-RAG path) vs. confidence-gated, crossed with each candidate signal
(top-1 similarity, margin, entropy) as separate arms.

**Evaluation metrics**: ROUGE-L, BLEU-4, BERTScore, F1RadGraph,
F1CheXbert split by gated-in vs. gated-out subsets; total retrieval
calls made (Recall@K/NDCG@K/MRR reported only for the gated-in subset,
since gated-out queries never retrieve); gate agreement with an oracle
"would retrieval have helped" label, defined post hoc via the
unmodified Oracle module (§2) rather than a newly invented metric.

**Computational cost**: near-zero overhead (scalar computation over
already-computed scores); net cost should be *lower* than baseline via
retrieval+generation calls saved on gated-out queries — savings must be
measured and reported, not assumed.

**Expected risks**: gate may systematically exclude genuinely-hard
queries that retrieval would have helped; the chosen signal may not
calibrate consistently across query difficulty distributions.

**What counts as success**: the gated subset's paired quality delta
(gated vs. always-retrieve, same queries) is non-negative within CI,
and the gate reduces total retrieval calls by a nontrivial margin
without significant quality loss.

**What would falsify the hypothesis**: gating removes retrieval from
queries the Oracle module shows would have clearly benefited from it;
or gate decisions show no better correlation with downstream quality
than a random gating baseline at matched gate-on rate.

---

## 9. Supporting Innovation 3 — Adaptive Retrieval Depth

*(formalizes `docs/innovation_proposals.md` Innovation A)*

**Research hypothesis**: choosing the number of retrieved reports (`k`)
adaptively per query — fewer for confident/easy queries, more for
ambiguous ones — outperforms any single fixed-`k` baseline on
retrieval metrics (Recall@K, NDCG@K, MRR) and on the generated
radiology report's quality (ROUGE-L, BLEU-4, BERTScore, F1RadGraph,
F1CheXbert), at equal or lower average `k`.

**Exact baseline limitation addressed**: `RAGDatasetBuilder`/the
official pipeline use a fixed `k` (effectively top-1 post
patient/study filtering) for every query regardless of query
difficulty.

**Input/output contract**:
- Input: ranked candidates + scores (same source as §8).
- Output: `AdaptiveKDecision` (selected `k`, clipped to configured
  `[k_min, k_max]`, signal value and rationale logged per query).

**Module layout**: `src/innovation/adaptive_retrieval/{config.py,
adaptive_k.py}`.

**Dataclasses/configs**: `AdaptiveRetrievalConfig` (`k_min`, `k_max`,
signal choice, mapping-function identifier); `AdaptiveKDecision`.

**Deterministic behavior**: pure function of `(scores, config)`, no
RNG.

**Stopping criteria**: single decision point per query; the
`[k_min, k_max]` bound is the enforced ceiling, itself validated by
tests, not an iterative process.

**Failure modes**: fewer than `k_min` real candidates exist in the
corpus (small-corpus edge case) → `k` clipped to the available
candidate count, disclosed, never silently padded with invalid
candidates.

**Unit-testing strategy**: chosen `k` always within
`[k_min, k_max]`; monotonic relationship between the chosen signal and
`k` for the configured mapping direction; determinism; edge case where
`k_max` exceeds corpus size.

**Colab execution plan**: dry run over synthetic and Milestone
2.4-exported real embeddings. A rule-based/calibrated mapping function
only, in this contract's initial scope — a *learned* k-selector is
explicitly out of scope here and named as a possible future extension
(§21), not silently assumed.

**Ablation design**: fixed `k` ∈ {1, 2, 3, 5} (matching
`docs/innovation_proposals.md`'s own stated comparison) vs.
adaptive-`k`, on Recall@K/MRR and downstream generation metrics, at
matched *average* `k` across arms for a fair cost comparison.

**Evaluation metrics**: Recall@K, NDCG@K, MRR at the per-query chosen
`k` (aggregated); downstream F1RadGraph/F1CheXbert/ROUGE-L/BLEU-4/
BERTScore; mean/median/distribution of chosen `k` vs. fixed-`k` arms.

**Computational cost**: variable per query, bounded by `k_max`; report
the full distribution of chosen `k` and total compute relative to a
fixed-`k_max` baseline (expected lower on average, by design — must be
measured, not assumed).

**Expected risks**: the mapping function may be miscalibrated (wrong
monotonic direction empirically); benefit may not materialize if
generation quality is insensitive to `k` for this generator/prompt
family.

**What counts as success**: adaptive-`k` matches or exceeds the best
fixed-`k` arm on ROUGE-L/BLEU-4/BERTScore/F1RadGraph/F1CheXbert and on
Recall@K/NDCG@K/MRR at a lower or equal average retrieval budget.

**What would falsify the hypothesis**: adaptive-`k` underperforms the
best fixed-`k` arm on any of those metrics at equal-or-higher average
budget; or the chosen signal shows no correlation with actual
per-query difficulty (measured via post hoc quality-vs-`k` curves).

---

## 10. Supporting Innovation 4 — Evidence-Aware Prompt Construction

*(new — no prior sketch in `docs/innovation_proposals.md`)*

**Research hypothesis**: constructing the generation prompt from a
structured, fact-annotated evidence representation — rather than the
baseline's raw concatenated report text — improves factual grounding
(F1RadGraph/F1CheXbert) without degrading fluency (ROUGE-L/BLEU-4/
BERTScore).

**Exact baseline limitation addressed**: `PromptBuilder`/
`PromptBuilderConfig`/`PromptResult` (Milestone 2.5) interpolate
retrieved report text directly into the prompt template, exactly
matching the official code (`docs/paper_analysis.md` §9) — no
structured fact representation, source attribution, or
evidence-highlighting exists in the baseline prompt path.

**Input/output contract**:
- Input: a structured evidence object (from Innovation 1's fusion step
  or Innovation 6's `CrossModalEvidence`, §12) plus the same
  query/image inputs `PromptBuilder` already takes.
- Output: `EvidenceAwarePromptResult`, a strict superset of the
  baseline `PromptResult` schema — a drop-in replacement wherever
  `PromptResult` is consumed downstream (`GeneratorAdapter`), unmodified.

**Module layout**: `src/innovation/evidence_aware_prompting/{config.py,
builder.py}`.

**Dataclasses/configs**: `EvidenceAwarePromptConfig` (template-variant
enum, `max_facts_per_report`, attribution style);
`EvidenceAwarePromptResult`.

**Deterministic behavior**: pure templating from already-computed
evidence, no RNG — byte-identical output for identical evidence input,
verified the same way as `PromptBuilder`'s own existing determinism
tests (Milestone 2.5).

**Stopping criteria**: N/A — single deterministic construction step,
not iterative.

**Failure modes**: evidence object missing expected fields (upstream
fusion failure) → defined fallback to the baseline raw-concatenation
prompt, never a crash, with the fallback path always disclosed per
query.

**Unit-testing strategy**: schema superset compatibility with baseline
`PromptResult` (existing generator code must accept it unmodified);
fallback-to-baseline test on malformed evidence; determinism; coverage
of each template variant.

**Colab execution plan**: synthetic dry run building prompts from
synthetic structured evidence, fed through the real, already-verified
`HFGeneratorAdapter` (Milestone 2.5G) unmodified — confirms real
end-to-end compatibility without retraining anything.

**Ablation design**: baseline raw-concatenation prompt vs.
evidence-aware prompt (each template variant a separate arm), holding
retrieval and generator model fixed. A **length-matched raw-
concatenation control** arm is required (see falsification below).

**Evaluation metrics**: F1RadGraph/F1CheXbert (primary — this targets
factual grounding specifically); ROUGE-L/BLEU-4/BERTScore (fluency
regression check). Retrieval metrics (Recall@K/NDCG@K/MRR) are not
applicable — this innovation changes prompt construction only and does
not alter retrieval.

**Computational cost**: negligible extra compute (string construction
only); longer/more structured prompts may marginally increase
generator forward-pass cost — token-count deltas must be measured and
reported.

**Expected risks**: longer/more structured prompts may confuse rather
than help the generator (model-dependent, unverified for this exact
template family); structured evidence is only as faithful as the
upstream fusion step (Innovation 6) feeding it.

**What counts as success**: evidence-aware prompting improves
F1RadGraph and/or F1CheXbert significantly (bootstrap CI excludes 0)
without significant ROUGE-L/BLEU-4/BERTScore regression.

**What would falsify the hypothesis**: no clinical-metric improvement;
any fluency-metric regression; or the improvement disappears once
controlled for prompt length alone via the length-matched
raw-concatenation control arm.

---

## 11. Supporting Innovation 5 — Self-Verification and Correction

*(extends `docs/innovation_proposals.md` Innovation E from
detection-only to detection **and** bounded correction, per explicit
request)*

**Research hypothesis**: comparing the generated radiology report text
against the same retrieved evidence it was conditioned on (via the
real F1RadGraph/F1CheXbert instance scorers already used by Oracle,
reused unmodified) can detect unsupported or contradicted claims; a
bounded, single-pass correction step (re-prompting with the discrepancy
made explicit) reduces the rate of evidence-unsupported claims and
improves F1RadGraph/F1CheXbert relative to no self-verification,
without regressing ROUGE-L, BLEU-4, or BERTScore. Retrieval metrics are
not applicable — this innovation does not change retrieval.

**Exact baseline limitation addressed**: baseline generation is a
single forward pass with no post-hoc consistency check against its own
retrieved evidence — nothing in Milestones 2.5/2.6 verifies a generated
radiology report's claims against what was actually retrieved before
returning it as final output.

**Input/output contract**:
- Input: the generated radiology report text + the evidence it was
  conditioned on.
- Output: `VerificationReport` (per-claim support/contradiction flags,
  overall consistency score) and, if triggered, a
  `CorrectedGenerationResult`.

**Module layout**: `src/innovation/self_verification/{config.py,
verifier.py, corrector.py}` (fills and extends the existing
`uncertainty/detector.py` stub's responsibility).

**Dataclasses/configs**: `SelfVerificationConfig` (consistency
threshold, `max_correction_attempts` fixed at 1 for this contract —
bounded, not agentic); `VerificationReport`;
`CorrectedGenerationResult`.

**Deterministic behavior**: verification score computation is
deterministic given fixed instance-scorer outputs, matching Oracle's
own determinism guarantee (Milestone 2.6). The correction step's
determinism depends on the underlying generator's decoding, which must
be pinned (greedy/seeded), the same caveat as Core Innovation 1.

**Stopping criteria**: at most **one** correction attempt. If the
corrected output still fails verification, it is returned as-is with
the failure disclosed — never retried further.

**Failure modes**: instance-scorer dependency unavailable
(`radgraph`/`f1chexbert`) → classified WARNING via the existing
`_classify_metric_dependency_error` taxonomy, verification skipped and
disclosed (never silently assumed "consistent"); correction degrading
fluency → caught by the same fluency-regression check as Innovation 4.

**Unit-testing strategy**: synthetic fake instance scorers (same
pattern as Cell 36's dry run) for: (a) fully-supported claim → no
correction triggered, (b) unsupported claim → correction triggered
exactly once, (c) correction attempted but still fails → returned with
a disclosed failure flag, no second attempt, (d) scorer unavailable →
classified skip, not a silent pass.

**Colab execution plan**: dry run reusing Milestone 2.6's real
radgraph/f1chexbert instance scorers and Milestone 2.5's real HF
generator adapter — no new checkpoint, no new training.

**Ablation design**: no-verification (baseline) vs. verify-only (report
but never correct) vs. verify-and-correct — isolating detection value
from correction value.

**Evaluation metrics**: F1RadGraph/F1CheXbert (primary); a new
claim-level support-rate diagnostic, itself validated against the
already-real Oracle instance scorers rather than an unvalidated new
metric; fluency-regression check (ROUGE-L/BLEU-4/BERTScore).

**Computational cost**: +1 verification pass (cheap, reuses existing
instance scorers); up to +1 full generation forward pass only for the
correction-triggered subset. Report % of queries triggering correction
and total added cost.

**Expected risks**: correction re-prompting may overcorrect (removing
correct-but-flagged claims — a false-positive cost); the verifier
inherits the already-open, unresolved F1CheXbert run-to-run label
instability risk (`docs/risk_register.md` #15) — this innovation
cannot be more reliable than that unresolved upstream risk, and must
disclose the dependency explicitly rather than assume it away.

**What counts as success**: the verify-and-correct arm shows a
significant reduction in evidence-unsupported claims and/or
F1RadGraph/F1CheXbert improvement relative to baseline, without
significant fluency regression.

**What would falsify the hypothesis**: no reduction in unsupported-claim
rate; correction significantly harms fluency; or verify-only already
captures the full benefit (meaning correction adds cost with no added
value) — independently testable via the three ablation arms.

---

## 12. Supporting Innovation 6 — Cross-Modal Evidence Fusion

*(broadens `docs/innovation_proposals.md` Innovation C from
text-only multi-report fusion to genuinely cross-modal fusion, per
explicit request)*

**Research hypothesis**: fusing image-derived evidence (the retriever's
own image/text similarity components) together with text-derived
evidence (RadGraph entities/relations, CheXbert labels of retrieved
reports) into one structured evidence object yields a more factually
grounded generated radiology report — measured by F1RadGraph and
F1CheXbert improvement, without ROUGE-L/BLEU-4/BERTScore regression —
than text-only fusion (the original Innovation C) or no fusion at all
(raw concatenation). Retrieval metrics are not applicable — this
innovation fuses already-retrieved evidence and does not change
retrieval itself.

**Exact baseline limitation addressed**: `MultiModalRetriever` computes
a single fused image+text embedding for ranking/similarity only
(`encode_images_with_text`/`scaled_similarity`) — that cross-modal
signal is discarded after retrieval and never surfaced to the
generator; `PromptBuilder` only ever receives raw retrieved report
text, never any image-derived evidence signal.

**Input/output contract**:
- Input: retrieved candidates' report text + the retriever's own
  per-candidate image/text similarity components (reusing
  `MultiModalRetriever`'s existing encode methods, unmodified, no
  retraining) + RadGraph/CheXbert annotations of retrieved reports
  (reusing Milestone 2.2's compat shim, unmodified).
- Output: `CrossModalEvidence` — `supporting_facts`,
  `conflicting_facts`, `source_ids`, `confidence_scores` (the original
  Innovation C schema) plus a new `modality_attribution` field
  distinguishing image-supported vs. text-supported facts. This is the
  direct input Innovation 4 consumes.

**Module layout**: `src/innovation/multi_report_fusion/{config.py,
fusion.py, modality_attribution.py}`.

**Dataclasses/configs**: `CrossModalFusionConfig` (fusion strategy,
redundancy-penalty weight, diversity-reward weight — matching the
original Innovation C design); `CrossModalEvidence`.

**Deterministic behavior**: pure function of already-computed
retriever/annotation outputs, no RNG — byte-identical fusion output for
identical inputs, verified via rerun comparison (the same method used
for Milestone 2.6's summary/per-sample agreement check).

**Stopping criteria**: N/A for a standalone call (single fusion pass
over a fixed candidate set); it is, however, the fusion step Core
Innovation 1 invokes up to twice, once per round.

**Failure modes**: RadGraph/CheXbert annotation unavailable for a
candidate → reuses Milestone 2.2's `CompatibilityError`/classified-
WARNING discipline; that candidate's text-derived evidence is marked
unavailable and disclosed, never silently dropped from `source_ids`.
Image-derived signal unavailable (e.g. missing image) →
`modality_attribution` marks image evidence absent for that candidate;
fusion proceeds text-only for it.

**Unit-testing strategy**: agreement/contradiction detection
correctness on hand-constructed synthetic report pairs (same style as
Cell 32's clinical-metric tests); redundancy/diversity scoring
correctness; `modality_attribution` correctness (image-only vs.
text-only vs. both-supported labeling); determinism; each failure mode
above.

**Colab execution plan**: dry run over Milestone 2.4G's real retriever
embeddings and Milestone 2.2's real RadGraph/CheXbert annotations
(both already proven real) — no new checkpoint, no training.

**Ablation design**: no fusion (raw concatenation, baseline) vs.
text-only fusion (original Innovation C) vs. cross-modal fusion (this
innovation), each feeding into Innovation 4 unchanged — isolating the
marginal value of the image-derived signal specifically.

**Evaluation metrics**: F1RadGraph/F1CheXbert (primary, per
`docs/innovation_proposals.md`'s own stated evaluation discipline for
Innovation C); ROUGE-L/BLEU-4/BERTScore (fluency check); a new
fusion-diagnostic metric (contradiction count per query,
redundancy-reduction ratio).

**Computational cost**: fusion itself is cheap (no model forward
pass — pure aggregation over already-computed outputs); cost is
dominated by the number of candidates fused (bounded by Innovation 3's
adaptive-`k`, if composed together, or a fixed top-N otherwise).

**Expected risks**: image-derived "evidence" from a similarity signal
is a much weaker/noisier proxy for factual grounding than text-derived
RadGraph/CheXbert signal — the cross-modal contribution may prove
negligible or net-negative; contradiction resolution across many
candidates may not scale cleanly beyond tiny synthetic test scale.

**What counts as success**: cross-modal fusion significantly improves
F1RadGraph/F1CheXbert over **text-only** fusion specifically (not just
over the no-fusion baseline, which the original Innovation C already
claims), without fluency regression.

**What would falsify the hypothesis**: cross-modal fusion performs no
better (or worse) than text-only fusion — meaning the image-derived
signal adds no measurable factual-grounding value at generation time.
This falsifies the *cross-modal* hypothesis specifically, distinct from
falsifying fusion in general.

---

## 13. Baseline/Innovation Separation & Anti-Contamination Protocol

- **Physical separation, already established**: `src/baseline/` and
  `src/evaluation/` vs. `src/innovation/`, in place since the Phase-1
  skeleton commit.
- **One-directional import rule**: `src/innovation/**` may import from
  `src/baseline/**`, `src/evaluation/**`, `src/common/**`,
  `src/data/**`, `src/retrieval/**`, `src/generation/**`. None of those
  may ever import from `src/innovation/**`. Enforced manually before
  every innovation commit via:
  ```
  grep -rn "from src.innovation\|import src.innovation" \
    src/baseline src/evaluation src/common src/data src/retrieval src/generation
  ```
  which must return empty (no CI workflow exists yet to automate this —
  disclosed limitation, `docs/milestone_2_6_completion.md` §14).
- **No file outside `src/innovation/**`, new `tests/unit/
  test_innovation_*.py` files, and `docs/**` may be modified by any
  innovation commit.** This mirrors the discipline already used for
  this contract itself.
- **Public-API-only reuse**: innovation code calls baseline/evaluation
  code only through its existing public API — never monkeypatching.
  The one established exception is direct reuse of already-`_`-prefixed
  internal factories where Milestone 2.6's own Cell 36 dry run already
  set the precedent (e.g. `_default_f1radgraph_factory`/
  `_default_f1chexbert_factory` for Oracle-style instance scoring) —
  the same precedent applies to Innovation 5's verifier, and no further
  private-internal reuse beyond that established pattern is permitted
  without a contract update.
- **Baseline control-arm integrity**: every comparison experiment must
  run the *unmodified* baseline path (no innovation code anywhere in
  its import graph) as the control arm, in the exact same
  run/report format Milestone 2.6 already verified as `CELL 36: PASS`
  — baseline numbers used for comparison are never regenerated through
  a different code path.
- **Validation-only tuning, enforced in code**: every innovation config
  threshold/weight/bound is loaded from a `*_validation_tuned.json`
  artifact produced by a validation-split run, never an inline literal
  computed from or tuned against test-split data. This is a code-level
  discipline in addition to the manual review checklist in §19.

## 14. Branch Strategy

**RESOLVED.** Baseline work currently lives on
`claude/factmm-rag-repo-setup-o04jx3` (Milestones 2.1–2.6, open draft
PR #1 targeting `main`, not yet merged as of this contract). The
recommended, resolved strategy:

1. **Merge PR #1 into `main` first.** No innovation branch is created
   before the verified baseline (Milestones 2.1–2.6, `CELL 36: PASS`,
   921/921 tests) is on `main` — this is the point at which "baseline"
   becomes a stable, shared reference rather than an in-progress branch.
2. **Tag the merged baseline** at the merge commit as:
   ```
   baseline-v1.0
   ```
   This tag is the fixed, permanent reference point every innovation
   experiment's `baseline_commit_hash` (§16) is pinned against, and the
   fixed control-arm code path (§13) for every comparison.
3. **Create a new Innovation branch from that tag (or `main`, which is
   identical to the tag at merge time)**, e.g.:
   ```
   innovation/iterative-rag
   ```
   for Core Innovation 1, and one branch per supporting innovation
   following the same `innovation/<short-name>` convention (e.g.
   `innovation/confidence-gating`, `innovation/adaptive-retrieval-depth`,
   `innovation/evidence-aware-prompting`, `innovation/self-verification`,
   `innovation/cross-modal-fusion`).
4. Each innovation branch/PR stays scoped to exactly one module from
   §7–§12 — no combined "all innovations" branch, so each can be
   reviewed, tested, and (eventually) evaluated independently before any
   composition experiment (§21) is attempted. Every innovation branch
   is created from `main`/`baseline-v1.0` directly, never from another
   in-progress innovation branch, to avoid cross-innovation coupling.
5. This preserves strict baseline/innovation separation (§13) at the
   branch level, not just the directory level: `main`/`baseline-v1.0`
   never receives a commit that imports from `src/innovation/**`, and
   every innovation branch's comparison experiments run against the
   exact code at `baseline-v1.0`, never a moving target.
6. Implementation does not begin on any branch until this contract is
   approved, matching every prior milestone's audit-before-code
   discipline.

## 15. Experiment Naming & Versioning

Extends `EvaluationRunConfig.run_id` (already free-text, Milestone 2.6)
with a fixed convention:

```
{innovation_id}-{variant}-{split}-v{n}
```

Examples: `innov1-gap-triggered-val-v1`,
`innov1-baseline-control-test-v1`, `innov3-adaptive-k-val-v2`.

- `innovation_id` ∈ `{innov1, innov2, innov3, innov4, innov5, innov6}`
  (or `baseline` for the unmodified control arm).
- `variant` names the specific ablation arm (e.g. `gap-triggered`,
  `forced-2-round`, `fixed-k3`, `text-only-fusion`).
- `split` ∈ `{val, test}` — never omitted, so a run's data provenance
  is unambiguous at a glance.
- `v{n}` bumps on any config, prompt-template, or checkpoint change —
  never overwriting a prior run's artifacts (append-only, matching
  Cells 30–36's atomic-write-never-overwrite discipline).

## 16. Reproducibility Metadata

Every innovation run emits the same fields `EvaluationRunResult`'s
summary JSON already emits (`package_versions`, `hardware`,
`git_commit`, `generated_timestamp_utc`, Milestone 2.6, unmodified),
plus innovation-specific additions:

- `innovation_id` — which module/arm produced this run.
- `innovation_config_hash` — sha256 of the frozen config dataclass's
  serialized form, so any config drift between runs is immediately
  detectable.
- `baseline_commit_hash` and `baseline_tag` — the exact baseline commit
  (and, once §14's `baseline-v1.0` tag exists, the tag name) the
  comparison control arm was run against. Until PR #1 merges and
  `baseline-v1.0` is tagged, pinned to the pre-merge commit
  `1dac8e80ba9ca052eddac2653917ef626f422f8c` (Milestone 2.6's
  completion commit); after tagging, pinned to `baseline-v1.0` and its
  resolved commit hash. Any later baseline change requires an explicit,
  disclosed pin update in this contract — never silent drift.

## 17. Comparison Tables (Reserved Shape, Not Populated)

No comparison numbers exist yet — **no improvement is claimed anywhere
in this contract.** The shape is reserved for implementation to
generate later, via `src/innovation/evaluation/innovation_metrics.py`
(extending its existing stub with a `comparison_table.py`-equivalent
generator):

For each innovation, on both validation and held-out test splits:

| Arm | ROUGE-L | BLEU-4 | BERTScore | F1RadGraph | F1CheXbert (dataset) | F1CheXbert (instance) | MRR | Recall@K | NDCG@K | Mean diff (95% CI) | p-value | Compute cost |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | — | — | — | — | — | — | — | — | — | — | — | — |
| innovation arm 1 | — | — | — | — | — | — | — | — | — | — | — | — |
| ... | | | | | | | | | | | | |

This table is produced only after real experiments run, never
populated speculatively.

## 18. Bootstrap Significance Plan

Every innovation-vs-baseline comparison reuses
`paired_bootstrap_significance` (Milestone 2.6, `src/evaluation/
significance.py`) **unmodified**:

- Paired per-example scores from the *same* query set — never a
  different sample between arms.
- `num_bootstrap_samples >= 10000` (the config default) for any
  reported, non-dry-run result.
- `confidence_level = 0.95` default.
- A fixed seed recorded in the run's reproducibility metadata (§16) for
  exact reproducibility.
- **Multiple-comparison disclosure**: when an innovation is compared
  against baseline across many metrics/ablation arms simultaneously,
  the report must disclose the total number of comparisons made and
  must never present a single favorable p-value in isolation without
  that context. Whether to apply a formal correction (e.g. Bonferroni)
  is deferred — see §21 — not decided in this contract.

## 19. Error-Analysis Protocol

For every innovation, after any real experiment:

1. Manually review the N lowest-scoring queries under the innovation
   arm **vs. the same queries under baseline** (a paired diff, never
   independent top/bottom lists compiled separately).
2. Review every query where verification/gap-detection/gating fired,
   spot-checking whether the triggering rationale was sound.
3. Aggregate every disclosed failure-mode occurrence (classified
   WARNINGs) by category, matching Milestone 2.6's own
   compatibility-report discipline.

No error-analysis finding is used to retroactively cherry-pick metrics
or thresholds on the test split. Any adjustment triggered by error
analysis is applied only via re-tuning on the validation split,
followed by a fresh, single test-split run — never iterative
test-split peeking.

## 20. Known Design Conflicts With Current Baseline

**20.1 — Task framing (RESOLVED).** An earlier draft of this contract
flagged a mismatch between a "Medical Question Answering" thesis
framing and the verified baseline's radiology-report-generation scope.
**This is resolved as of §3: the project continues as Radiology Report
Generation (Path A selected, Path B deferred/out of scope).** No
QA-pair data, QA schema, or QA-specific metric is introduced anywhere
in this contract; every innovation in §7–§12 reuses the baseline's
report-generation metrics unmodified. See §3 for the full decision,
rationale, and the historical record (§3.5) of why the QA framing was
considered and deferred.

**20.2 — Oracle's single-modality scope.** The Oracle module (Milestone
2.6, reused unmodified by Innovation 5) is defined and validated over a
same-modality (text-report) corpus only, per the paper's own Appendix
A.3 definition. If Innovation 6's cross-modal evidence is later fed into
an Oracle-style comparison, Oracle's own definition would need
re-auditing — this contract flags, but does not resolve, that future
extension.

**20.3 — `factual_reranking/reranker.py` (original Innovation B) is
intentionally excluded.** Core Innovation 1's gap-detection step and
Supporting Innovation 6's fusion step both have natural overlap with
reranking-style logic, but this contract deliberately does not fold
reranking in, to stay scoped exactly to the six modules requested. Not
a conflict requiring resolution now — flagged as a likely near-term
extension.

## 21. Open Questions (`UNKNOWN`)

- What is the target dataset/scale for the first real (non-synthetic)
  innovation experiment — presumably the same MIMIC-CXR-derived corpus
  the baseline was verified against (per §3's resolved task framing),
  but the exact split/subset size is not yet decided.
- Should any of these six modules ever use a *learned* (trained)
  component (k-selector, gate, verifier), or must all six remain
  rule-based/calibrated-threshold only in this phase, given no training
  infrastructure beyond `RetrieverTrainer` (Milestone 2.4) exists for
  this class of small auxiliary model?
- Should formal multiple-comparison correction (e.g. Bonferroni) be
  applied across the full innovation matrix, or reported
  disclosed-but-uncorrected per comparison (§18)?
- What is the actual compute budget/timeline for real (GPU)
  experiments? This bounds how ambitious the initial ablation matrix
  can be — deferred to a future Phase 5 experiment protocol, consistent
  with `docs/innovation_proposals.md`'s own existing note on this.
- Should Core Innovation 1 be validated standalone first, or built to
  compose with all five supporting innovations simultaneously (a "full
  system" arm) from the outset?
