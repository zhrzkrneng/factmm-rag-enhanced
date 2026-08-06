# Phase 3 — Innovation Implementation Roadmap

> **PLANNING DOCUMENT — NOT EXECUTABLE CODE.** This is the master
> implementation roadmap for Phase 3, written on `innovation/
> iterative-rag` after `baseline-v1.0` was tagged and PR #1 merged to
> `main`. It sequences the six innovations already fully specified in
> `docs/phase_3_innovation_contract.md` (research hypothesis through
> falsification criterion, per innovation) into an implementation order,
> a cumulative experiment plan, a risk register, and a commit plan. No
> baseline implementation, no test, and no evaluation code is modified
> by this document. No Innovation code is implemented by this document.
> No improvement is claimed anywhere below — every "expected gain" is a
> hypothesized direction pending real, not-yet-run experiments.

---

## 1. Current Project State

- **`baseline-v1.0`** is an annotated tag on `main` at the PR #1 merge
  commit (`83d1d1c07e2a62e71a8e8dcc1a4686a1f311294f`), message
  "Verified FactMM-RAG baseline through Milestone 2.6." It is the fixed,
  permanent reference point for every innovation experiment's control
  arm and reproducibility metadata (per `docs/phase_3_innovation_
  contract.md` §14/§16).
- **Milestones 2.1–2.6 are complete** on `main`: data pipeline (2.1),
  RadGraph/CheXbert annotation (2.2), fact-aware pair mining (2.3),
  retriever baseline (2.4, plus the real Hugging Face adapter
  compatibility dry run, 2.4G), generator baseline (2.5, plus its real
  compatibility dry run, 2.5G), and the evaluation baseline (2.6).
- **921/921 tests passing** — confirmed both at the PR #1 merge and
  freshly re-confirmed on this branch (`innovation/iterative-rag`)
  before writing this roadmap, with zero warnings, zero errors, zero
  failures, working tree clean.
- **`CELL 36: PASS`** — the final Milestone 2.6 real-evaluation dry run,
  independently verified in the user's own Colab environment (not just
  in-sandbox): real `rouge==1.0.1`, `evaluate==0.4.6`,
  `bert-score==0.3.13`, `pytrec_eval==0.5`, plus already-real
  `radgraph==0.0.9`/`f1chexbert==0.0.2`. No mock/fallback path used in
  the final verification. See `docs/milestone_2_6_completion.md`.
- **The evaluation baseline is complete and real**: ROUGE-L, BLEU-4,
  BERTScore, F1RadGraph, F1CheXbert (dataset- and instance-level), MRR,
  Recall@K, NDCG@K, the `EvaluationRunner` orchestrator, the Oracle
  upper-bound baseline, and paired bootstrap significance testing — all
  real, all reused unmodified by every innovation in this roadmap.
- **The baseline is now frozen.** This roadmap, and every innovation it
  sequences, is additive-only under `src/innovation/**` (per
  `docs/phase_3_innovation_contract.md` §13). No file under
  `src/baseline/`, `src/evaluation/`, `src/common/`, `src/data/`,
  `src/retrieval/`, `src/generation/`, or `tests/` is touched by any
  step in this roadmap.
- **Current working branch**: `innovation/iterative-rag`, branched from
  `main`/`baseline-v1.0`, per the resolved branch strategy
  (`docs/phase_3_innovation_contract.md` §14).

## 2. Phase 3 Goal

The Phase 3 objective is:

> **Iterative Multimodal Retrieval-Augmented Generation for Radiology
> Report Generation**

**Not** Medical Question Answering. This is a resolved decision (Path A
selected, Path B deferred/out of scope), recorded in full — decision,
rationale, terminology rules, and the historical record of why the QA
framing was considered and rejected — in `docs/phase_3_innovation_
contract.md` §3. Every innovation in this roadmap targets the same
task the verified baseline already implements and evaluates: given an
image (+ optional prior text and retrieved evidence), produce a
generated radiology report, scored against a reference report using the
baseline's own real, unmodified metrics (§6 below). No QA dataset, QA
schema, or QA-specific metric appears anywhere in this roadmap.

## 3. Innovation Order

Implementation proceeds in exactly this order:

1. **Innovation 1 — Iterative Evidence Refinement** (core)
2. **Innovation 2 — Confidence-Aware Retrieval**
3. **Innovation 3 — Adaptive Retrieval Depth**
4. **Innovation 4 — Evidence-Aware Prompt Construction**
5. **Innovation 5 — Self-Verification** (and bounded correction)
6. **Innovation 6 — Cross-Modal Evidence Fusion**

### Why this order minimizes risk

- **Innovation 1 first, to establish the pipeline skeleton once.**
  Innovation 1's design (`docs/phase_3_innovation_contract.md` §7)
  already defines explicit, optional plug-in points for every
  supporting innovation: a gate before round 1 (Innovation 2), a
  round-sizing signal (Innovation 3), a prompt-construction strategy
  for the final radiology report (Innovation 4), a post-generation
  verification/correction step (Innovation 5), and a fusion strategy
  (Innovation 6). Building the core loop first — with the simplest
  possible default at every plug-in point (e.g. always-retrieve,
  fixed-k, baseline raw-concatenation prompting, no verification,
  text-only fusion) — produces one working, real, deterministic,
  fully-tested pipeline early. Every subsequent innovation then only has
  to prove a single swap-in component works, never new integration
  surface — directly minimizing integration risk for everything that
  follows.
- **Innovations 2 and 3 next, because they are the lowest-risk,
  fastest-to-validate components.** Both are pure, RNG-free functions
  of already-computed retrieval scores — no model forward pass, no new
  external dependency, trivially deterministic, and each affects only
  the *first* retrieval step (upstream of everything else). Validating
  them early means any regression they might cause surfaces immediately,
  before more complex components are layered on top, and neither
  requires Innovation 1's second round to already exist to be tested
  standalone.
- **Innovation 4 next, because it changes what the generator sees.**
  It depends only on an evidence object (either the baseline's raw
  retrieved text or a structured object from Innovation 1/6) and a
  documented, tested fallback to baseline prompting on malformed
  input (`docs/phase_3_innovation_contract.md` §10) — so it does not
  require Innovation 6 to exist first, only to be *compatible* with it
  later. It is placed before Innovation 5 because Innovation 5 verifies
  *whatever* the current prompting strategy produces — verifying after
  the prompting strategy is settled avoids re-validating Innovation 5
  against a moving target.
- **Innovation 5 next, because it is low-additional-risk but
  conceptually downstream of generation.** It reuses the exact same
  real `radgraph`/`f1chexbert` instance scorers Oracle already uses
  (Milestone 2.6, unmodified) — no new external dependency — but it
  only makes sense to verify a final radiology report once the
  pipeline producing that report (Innovations 1–4) is already in place.
- **Innovation 6 last, because it is the highest-risk, most speculative
  component.** It introduces an entirely new signal source
  (image-derived evidence) fused against text-derived RadGraph/CheXbert
  annotations, with cross-modal attribution correctness that must be
  validated from scratch — and the contract's own risk analysis
  (`docs/phase_3_innovation_contract.md` §12) already flags that its
  contribution "may prove negligible or net-negative." Implementing it
  last means every cheaper, better-understood, already-verified
  component (Innovations 1–5) is available as a stable comparison
  baseline, so if Innovation 6 falsifies its own hypothesis, that
  finding is isolated and does not block or contaminate the rest of the
  roadmap.

## 4. Per-Innovation Plan

Each block below summarizes, without code, what
`docs/phase_3_innovation_contract.md` already specifies in full for
that innovation (research hypothesis through falsification criterion) —
this roadmap adds the *implementation planning* view: affected modules,
expected new configs/tests, ablations, and stopping criteria, framed for
sequencing rather than specification.

### Innovation 1 — Iterative Evidence Refinement

- **Hypothesis**: a second, gap-targeted retrieval round — triggered
  only when a detected evidence gap exceeds a threshold — and fused
  evidence before the final radiology report improves F1RadGraph/
  F1CheXbert over the single-shot baseline, holding ROUGE-L/BLEU-4/
  BERTScore steady, at bounded extra cost. Full detail: contract §7.
- **Expected gain (hypothesized direction, not claimed)**: F1RadGraph/
  F1CheXbert improvement on the gap-triggered query subset; retrieval
  metrics (Recall@K/NDCG@K/MRR) reported as a supporting diagnostic,
  not a primary target.
- **Affected modules**: new only — `src/innovation/iterative_
  refinement/{config.py, pipeline.py, gap_detector.py, fusion.py}`.
  Calls `MultiModalRetriever`, `FaissFlatIPIndex`, `GeneratorAdapter`
  (all unmodified, read-only).
- **Expected new configs**: `IterativeRefinementConfig`,
  `IterativeRefinementRequest`, `EvidenceGapReport`,
  `IterativeRefinementResult`.
- **Expected new tests**: `tests/unit/test_iterative_refinement.py` —
  no-gap-path-matches-baseline, gap-triggered-exactly-one-second-call,
  determinism (rerun ×2), each documented failure mode forced and
  asserted graceful, fusion conflict-preservation.
- **Expected ablations**: baseline (single-shot) vs. iterative
  (gap-triggered) vs. iterative-forced (always 2 rounds).
- **Stopping criteria**: exactly 2 rounds maximum; round 2 only on
  `gap_detection_threshold` exceeded.

### Innovation 2 — Confidence-Aware Retrieval

- **Hypothesis**: a calibrated retrieval-confidence signal can gate
  low-value retrieval to non-RAG generation without significant harm
  (and possibly improvement) to ROUGE-L/BLEU-4/BERTScore/F1RadGraph/
  F1CheXbert, while reducing total retrieval calls. Full detail:
  contract §8.
- **Expected gain (hypothesized)**: non-negative paired quality delta
  on gated queries; measurable reduction in retrieval calls.
- **Affected modules**: fills existing stub —
  `src/innovation/confidence_gating/{config.py, gate.py,
  calibration.py}`.
- **Expected new configs**: `ConfidenceGatingConfig`,
  `RetrievalConfidenceDecision`.
- **Expected new tests**: `tests/unit/test_confidence_gating.py` —
  monotonicity, edge cases (empty/degenerate score distributions),
  determinism, validation-only-threshold enforcement.
- **Expected ablations**: always-retrieve vs. always-no-retrieve vs.
  confidence-gated, crossed with each candidate signal (top-1
  similarity, margin, entropy).
- **Stopping criteria**: single decision point per query, computed
  once, never retried.

### Innovation 3 — Adaptive Retrieval Depth

- **Hypothesis**: choosing `k` adaptively per query outperforms any
  fixed-`k` baseline on Recall@K/NDCG@K/MRR and downstream ROUGE-L/
  BLEU-4/BERTScore/F1RadGraph/F1CheXbert at equal or lower average `k`.
  Full detail: contract §9.
- **Expected gain (hypothesized)**: matches-or-exceeds best fixed-`k`
  arm at lower average retrieval budget.
- **Affected modules**: fills existing stub —
  `src/innovation/adaptive_retrieval/{config.py, adaptive_k.py}`.
- **Expected new configs**: `AdaptiveRetrievalConfig`,
  `AdaptiveKDecision`.
- **Expected new tests**: `tests/unit/test_adaptive_retrieval.py` — `k`
  always within `[k_min, k_max]`, monotonicity, determinism, `k_max` >
  corpus-size edge case.
- **Expected ablations**: fixed `k` ∈ {1, 2, 3, 5} vs. adaptive-`k`, at
  matched average `k`.
- **Stopping criteria**: single decision point per query; `[k_min,
  k_max]` is the enforced bound, not an iterative process.

### Innovation 4 — Evidence-Aware Prompt Construction

- **Hypothesis**: prompting from a structured, fact-annotated evidence
  representation improves F1RadGraph/F1CheXbert without degrading
  ROUGE-L/BLEU-4/BERTScore, relative to the baseline's raw
  concatenation, and the effect survives a length-matched control. Full
  detail: contract §10.
- **Expected gain (hypothesized)**: F1RadGraph/F1CheXbert improvement
  not explained by prompt length alone.
- **Affected modules**: new only —
  `src/innovation/evidence_aware_prompting/{config.py, builder.py}`.
  Output is a strict superset of the baseline `PromptResult` schema.
- **Expected new configs**: `EvidenceAwarePromptConfig`,
  `EvidenceAwarePromptResult`.
- **Expected new tests**: `tests/unit/test_evidence_aware_prompting.py`
  — schema superset compatibility, fallback-to-baseline on malformed
  evidence, determinism, template-variant coverage.
- **Expected ablations**: baseline raw-concatenation vs. each
  evidence-aware template variant vs. a length-matched
  raw-concatenation control.
- **Stopping criteria**: N/A — single deterministic construction step,
  not iterative.

### Innovation 5 — Self-Verification (and bounded correction)

- **Hypothesis**: comparing the generated radiology report against its
  own conditioning evidence (via Oracle's real instance scorers,
  unmodified) detects unsupported claims; one bounded correction attempt
  reduces the unsupported-claim rate and improves F1RadGraph/F1CheXbert
  without regressing ROUGE-L/BLEU-4/BERTScore. Full detail: contract
  §11.
- **Expected gain (hypothesized)**: reduced unsupported-claim rate
  and/or F1RadGraph/F1CheXbert improvement on the verify-and-correct
  arm.
- **Affected modules**: extends existing stub —
  `src/innovation/self_verification/{config.py, verifier.py,
  corrector.py}`.
- **Expected new configs**: `SelfVerificationConfig`,
  `VerificationReport`, `CorrectedGenerationResult`.
- **Expected new tests**: `tests/unit/test_self_verification.py` —
  supported-claim-no-correction, unsupported-claim-exactly-one-
  correction, correction-still-fails-returned-disclosed, scorer-
  unavailable-classified-skip.
- **Expected ablations**: no-verification (baseline) vs. verify-only
  vs. verify-and-correct.
- **Stopping criteria**: at most one correction attempt; a still-failing
  corrected output is returned as-is with the failure disclosed, never
  retried further.

### Innovation 6 — Cross-Modal Evidence Fusion

- **Hypothesis**: fusing image-derived evidence (retriever similarity
  components) with text-derived evidence (RadGraph/CheXbert
  annotations) improves F1RadGraph/F1CheXbert over text-only fusion
  specifically (not just over no fusion), without ROUGE-L/BLEU-4/
  BERTScore regression. Full detail: contract §12.
- **Expected gain (hypothesized)**: F1RadGraph/F1CheXbert improvement
  attributable to the image-derived signal specifically, isolated via
  the text-only-fusion control arm.
- **Affected modules**: fills + extends existing stub —
  `src/innovation/multi_report_fusion/{config.py, fusion.py,
  modality_attribution.py}`.
- **Expected new configs**: `CrossModalFusionConfig`,
  `CrossModalEvidence`.
- **Expected new tests**: `tests/unit/test_multi_report_fusion.py` —
  agreement/contradiction detection correctness, redundancy/diversity
  scoring, `modality_attribution` correctness, determinism, each
  documented failure mode.
- **Expected ablations**: no fusion (baseline) vs. text-only fusion vs.
  cross-modal fusion.
- **Stopping criteria**: N/A for a standalone call (single fusion pass
  over a fixed candidate set); Innovation 1 may invoke it up to twice,
  once per round.

## 5. Experiment Plan

A cumulative, staircase composition plan — additive to, not a
replacement for, each innovation's own standalone ablation in §4. Each
arm below is run on both the validation split and the held-out test
split, and compared against **both** the frozen `baseline-v1.0` control
arm **and** the immediately preceding staircase step, via `paired_
bootstrap_significance` (unmodified, §6/§18 of the contract):

```
Baseline                              (= baseline-v1.0, unmodified)
+ Innovation 1                        (iterative evidence refinement alone)
+ Innovation 1 + 2                    (+ confidence-aware retrieval)
+ Innovation 1 + 2 + 3                (+ adaptive retrieval depth)
+ Innovation 1 + 2 + 3 + 4            (+ evidence-aware prompting)
+ Innovation 1 + 2 + 3 + 4 + 5        (+ self-verification)
+ Innovation 1 + 2 + 3 + 4 + 5 + 6    (Full system)
```

Each staircase step is added only after its own standalone validation
(§4's per-innovation ablation) has already run and been reviewed —
resolving `docs/phase_3_innovation_contract.md` §21's open question
("standalone first, or compose from the outset") in favor of
**standalone-first, then staircase composition**, consistent with the
risk-minimizing order in §3. If any step regresses a metric
significantly relative to the previous step, that regression is
reported and investigated (§8/§19's error-analysis protocol) before the
next step is added — the staircase does not proceed past an unexplained
regression.

## 6. Evaluation Plan

Every experiment in this roadmap (§4 and §5) reuses **only** the
following real, already-verified metrics — **no new metric is
introduced anywhere in Phase 3**:

- ROUGE-L
- BLEU-4
- BERTScore
- F1RadGraph
- F1CheXbert (dataset-level and instance-level)
- Recall@K
- NDCG@K
- MRR
- Paired bootstrap significance (`paired_bootstrap_significance`, for
  every innovation-vs-baseline and step-vs-previous-step comparison)

All are Milestone 2.6's own real, `CELL 36: PASS`-verified
implementations, called through their existing public API, unmodified.

## 7. Paper Figures

Reserved figure slots — **no content generated yet**, since no real
experiment has run:

1. **System overview** — the full architecture diagram (baseline +
   six innovation modules and their composition points), extending
   `docs/phase_3_innovation_contract.md` §5.
2. **Iteration loop** — Innovation 1's round-1 → draft radiology report
   → evidence-gap detection → round-2 → fused evidence → final
   radiology report cycle.
3. **Evidence refinement** — a before/after comparison of round-1-only
   evidence vs. round-1+2 fused evidence for a representative
   gap-triggered query.
4. **Ablation summary** — the §5 staircase composition plot (metric
   value vs. cumulative innovation set).
5. **Comparison table** — the reserved shape from
   `docs/phase_3_innovation_contract.md` §17, populated only after real
   experiments run.
6. **Error analysis** — qualitative examples from the §19
   error-analysis protocol (paired lowest-scoring queries, gate/
   verification trigger spot-checks).

## 8. Risk Register

| Innovation | Technical risk | Compute risk | Reproducibility risk |
|---|---|---|---|
| 1 — Iterative Evidence Refinement | Gap detector miscalibration (over/under-triggering); round-2 retrieving near-duplicates of round 1; fusion introducing unresolved contradictions | Up to 2× retrieval + 2× generation calls per query on gap-triggered queries only | Requires pinned (greedy/seeded) generator decoding for the determinism claim to hold, same caveat as Milestone 2.5's `HFGeneratorAdapter` |
| 2 — Confidence-Aware Retrieval | Gate may exclude genuinely-hard queries retrieval would have helped; signal may not calibrate across query-difficulty distributions | Near-zero (scalar computation); net cost expected lower than baseline via saved calls | Pure function of `(scores, config)`, no RNG — low risk, but calibration-split discipline must be enforced (§13) to avoid test-split leakage |
| 3 — Adaptive Retrieval Depth | Mapping function may be miscalibrated (wrong monotonic direction); benefit may not materialize if quality is `k`-insensitive | Variable per query, bounded by `k_max`; expected lower average than fixed-`k_max` baseline | Pure function, no RNG — low risk; must report full `k` distribution, not just the mean |
| 4 — Evidence-Aware Prompt Construction | Longer/structured prompts may confuse the generator (model-dependent, unverified for this template family); only as faithful as its upstream evidence source | Negligible extra compute; token-count deltas must be measured | Deterministic templating, low risk; requires the length-matched control arm to rule out a length-only confound |
| 5 — Self-Verification | Correction may overcorrect (removing correct-but-flagged claims); reuses radgraph/f1chexbert, both compatibility-shim-gated | +1 verification pass (cheap) + up to +1 generation pass only on correction-triggered subset | **Inherits the already-open, unresolved F1CheXbert run-to-run label instability risk (`docs/risk_register.md` #15)** — this innovation cannot be more reliable than that unresolved upstream risk |
| 6 — Cross-Modal Evidence Fusion | Image-derived signal may be a much weaker/noisier proxy than text-derived RadGraph/CheXbert signal; contradiction resolution may not scale past tiny synthetic scale | Fusion itself cheap (no forward pass); cost dominated by number of candidates fused | Deterministic pure aggregation, low risk in isolation; highest integration risk of the six (newest signal source, least precedent in this project) |

## 9. Success Criteria

**Innovation 1**: on held-out validation data, the gap-triggered arm
statistically significantly outperforms `baseline-v1.0` on at least one
of F1RadGraph/F1CheXbert (bootstrap CI excludes 0) without a significant
regression on any other reused metric, at disclosed bounded cost — per
contract §7, unchanged here.

**Entire Phase 3**: the best-performing staircase arm from §5
(potentially, but not necessarily, the full system) statistically
significantly outperforms `baseline-v1.0` on at least one clinical
metric (F1RadGraph or F1CheXbert) without significant regression on
ROUGE-L, BLEU-4, BERTScore, or any retrieval metric, on the held-out
test split — with every result reproducible (deterministic reruns per
module, fixed-seed bootstrap significance), every ablation arm's result
reported regardless of outcome (no cherry-picking), strict
baseline/innovation separation maintained throughout (verified via the
§13 import-boundary check), and a completed error analysis (§19) for
every arm that ran on real data.

**Paper submission**: comparison tables (§17 of the contract, §5 here)
complete for both validation and test splits across the full staircase;
bootstrap significance reported with explicit multiple-comparison
disclosure (§18); all six figure slots (§7) populated with real
experiment output; reproducibility metadata (§16 of the contract)
attached to every reported number; any innovation whose hypothesis was
falsified is reported as such, not omitted; the full test suite
(baseline's 921 plus every innovation's own new tests) passing; and a
direct, disclosed comparison against the original FactMM-RAG paper's own
Table 1 / Appendix A.3 Oracle numbers.

## 10. Commit Plan

Following this project's established one-cohesive-unit-per-commit
discipline (Cells 30–35's own pattern), the expected future commits, in
implementation order:

1. `Implement Iterative Evidence Refinement pipeline` — Innovation 1:
   `config.py`, `pipeline.py`, `gap_detector.py`, `fusion.py`, plus
   `tests/unit/test_iterative_refinement.py`.
2. `Implement Confidence-Aware Retrieval gating` — Innovation 2: fills
   `confidence_gating/gate.py`, adds `config.py`/`calibration.py`, plus
   `tests/unit/test_confidence_gating.py`.
3. `Implement Adaptive Retrieval Depth selection` — Innovation 3: fills
   `adaptive_retrieval/adaptive_k.py`, adds `config.py`, plus
   `tests/unit/test_adaptive_retrieval.py`.
4. `Implement Evidence-Aware Prompt Construction` — Innovation 4:
   `evidence_aware_prompting/{config.py, builder.py}`, plus
   `tests/unit/test_evidence_aware_prompting.py`.
5. `Implement Self-Verification and bounded correction` — Innovation 5:
   extends `self_verification/{config.py, verifier.py, corrector.py}`,
   plus `tests/unit/test_self_verification.py`.
6. `Implement Cross-Modal Evidence Fusion` — Innovation 6: fills +
   extends `multi_report_fusion/{config.py, fusion.py,
   modality_attribution.py}`, plus `tests/unit/test_multi_report_
   fusion.py`.
7. `Run Innovation 1 standalone real-dependency dry run` and
   equivalent per-innovation dry-run commits/entries (Colab execution
   plans, §4 of the contract) for Innovations 2–6, each documented, not
   necessarily one commit per innovation depending on findings.
8. `Run cumulative staircase experiments and populate comparison
   tables` — §5/§17, after all six innovations are individually
   verified.
9. `Document Phase 3 results and error analysis` — final write-up
   commit, populating §7's figure slots and §9's success-criteria
   findings honestly, including any falsified hypotheses.

Every commit above stays scoped to exactly what this list states — no
combined "implement everything" commit, and no commit modifies any file
outside `src/innovation/**`, new `tests/unit/test_innovation_*.py`
files, and `docs/**`, per §13's anti-contamination protocol.
