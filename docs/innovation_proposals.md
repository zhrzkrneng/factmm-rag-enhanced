# Innovation Proposals

All items below are `PROPOSED_EXTENSION` — none exist in the paper (as far
as currently known, pending PDF) or in the official repository. They are
design sketches only; no code will be written until the baseline smoke test
passes (Phase 3 complete), per project rules. Implementation will live
exclusively under `src/innovation/`, never mixed into `src/baseline/`.

## A. Adaptive Multi-Report Retrieval

**Idea**: replace the official baseline's fixed top-1 (post patient/study
filter) retrieval with a variable-`k` scheme: retrieve top-N candidates,
estimate a confidence score, and choose fewer reports for high-confidence
queries and more for ambiguous ones (bounded by configurable min/max k).

**Baseline being extended**: `build_rag_dataset.py`'s single-candidate
selection (§12, §3 step 8 of `docs/paper_analysis.md`).

**Comparison**: fixed top-1 / top-2 / top-3 / top-5 vs. adaptive-k, on
retrieval metrics (Recall@K, MRR) and downstream generation metrics.

**Evaluation discipline**: `k` selection logic tuned only on validation
data; test-set `k` decisions must be reproducible from the same trained
confidence estimator, with the chosen `k` and its rationale logged per
query for auditability.

## B. Fact-Aware Multi-Objective Reranking

**Idea**: rerank retrieved candidates with a configurable weighted
combination of: image-report similarity, raw retriever score, RadGraph
factual similarity, CheXbert label agreement, negation consistency,
anatomical consistency, a redundancy penalty, and a diversity reward.

**Baseline being extended**: the retriever's raw FAISS ranking
(`knn.py` / `retrieval.py`), which uses embedding similarity alone.

**Constraint**: ground-truth test reports must never be used as reranking
signal at inference; weights are tuned only on the validation split.

## C. Diverse Multi-Report Fusion

**Idea**: when multiple reports are retrieved (depends on Innovation A),
extract factual units per report, identify agreements vs. contradictions
across candidates, drop redundant facts, and preserve clinically relevant
negations, producing a structured evidence object
(`supporting_facts`, `conflicting_facts`, `source_ids`, `confidence_scores`)
for the generator instead of raw text concatenation.

**Baseline being extended**: the official prompt template's raw single-
report string interpolation (§9 of `docs/paper_analysis.md`).

**Comparison**: raw concatenation of top-N reports vs. structured fusion,
on factual-consistency metrics (F1RadGraph, F1CheXbert) specifically, since
this innovation targets factual quality, not just fluency.

## D. Retrieval Confidence Gating

**Idea**: a gate deciding whether to retrieve at all, using signals such as
top-1 similarity, top-1-vs-top-2 margin, candidate agreement, RadGraph
consistency across candidates, or retriever calibration/entropy.

**Baseline being extended**: the official pipeline always retrieves
exactly one report (no gating exists in official code).

**Comparison**: always-retrieve vs. never-retrieve (the repo's own non-RAG
VQA path, `build_nonrag_dataset.py`) vs. confidence-gated retrieval.

## E. Uncertainty and Contradiction Detection

**Idea**: an auxiliary module flagging low-confidence retrieval, conflicting
retrieved facts, generated findings unsupported by any retrieved evidence,
contradictory positive/negative observations, and other suspicious patterns
— explicitly framed as a research-oriented warning system, not a clinical
correctness claim.

**Baseline being extended**: none directly — this is a net-new auxiliary
analysis layer on top of whatever generation path (baseline or innovation)
is active.

**Guardrail**: outputs must be labeled as research warnings; this module
must never be presented as validating or invalidating clinical accuracy.

## F. Optional Longitudinal Context

**Idea**: if prior studies for the same patient are legally and
structurally available, retrieve the patient's own prior report separately
(never mixing prior studies across different patients) and model
new/stable/improved/worsened findings relative to it.

**Status**: optional and explicitly out of scope unless the underlying
dataset access legally and structurally supports linking a patient's
studies over time — this is not part of the original baseline and will not
be attempted until Innovations A–E are complete and the user confirms
suitable data access.

## Experiment Matrix (Phase 5 preview — not run yet)

Baseline arms: no-retrieval, official top-1 retriever, reproduced
FactMM-RAG, oracle-analysis (only if methodologically valid, i.e. not used
to inflate reported numbers). The paper (now available) already defines and
reports its own Oracle upper bound precisely (Appendix A.3: argmax over the
corpus of `F1RadGraph + F1CheXbert` instance-wise score, self-excluded for
training queries) — this project reuses that exact definition rather than
inventing a different oracle procedure, so oracle numbers stay comparable
to the paper's own Table 1 Oracle row (see `docs/paper_analysis.md` §17).

Innovation ablations: adaptive-k only, reranking only, fusion only, gating
only, uncertainty-module only, and cumulative combinations up to the full
enhanced system — exact matrix to be finalized in `docs/experiment_protocol.md`
during Phase 5, after the baseline exists.

No innovation will be reported as "successful" without validation-set
tuning followed by held-out test-set confirmation, per project rules.
