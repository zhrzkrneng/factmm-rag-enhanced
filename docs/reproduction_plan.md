# Reproduction Plan

## Guiding Constraints

- Baseline before innovation. Nothing in `src/innovation/` starts until the
  baseline smoke test (Phase 3) passes end-to-end on synthetic/small data.
- No claim of "working" without an executed, observed test run.
- Every stage configurable (YAML/JSON config, not hard-coded CLI defaults).
- Patient-level splitting enforced and independently verified, not assumed
  from upstream MIMIC-CXR/CheXpert splits.
- No dataset, checkpoint, or credential ever committed.

## Milestone Sequence (Phase 2)

1. **2.1 Data pipeline** — manifest schema, patient/study-disjoint split
   verification, frontal-view handling, integrity checks (missing files,
   duplicate studies, missing patient IDs).
2. **2.2 RadGraph processing** — modular annotator interface with a real
   backend (RadGraph + CheXbert packages) and a deterministic mock backend
   for tests when GPU/model weights are unavailable.
3. **2.3 Fact-aware pair mining** — reproduce `gen_similarity.py` +
   `gen_topk_pos.py` logic with configurable thresholds
   (`chexbert_threshold`, `radgraph_threshold`, `top_k`), provenance
   recording (scores + which threshold each pair passed), and leakage
   checks (no validation/test report ever used as a training corpus
   candidate for a training query, and vice versa).
4. **2.4 Retriever** — adapt (not blindly rewrite) the official
   CLIP+T5/MARVEL architecture and training loop; preserve official
   hyperparameters as the default config; add checkpointing, embedding
   export, FAISS indexing, and MRR/Recall/NDCG tracking.
5. **2.5 Retrieval-augmented generator** — reproduce the RAG dataset
   construction (including the official patient/study leakage filter) and
   the LLaVA prompt template; provide a lightweight mock generator (e.g., a
   template-filling stub) for smoke tests so the pipeline is exercisable
   without a multi-GPU LLaVA fine-tune; document the real LLaVA path as a
   separate, larger experiment.
6. **2.6 Evaluation** — F1CheXbert, F1RadGraph, ROUGE-L, BERTScore, MRR,
   Recall@K, plus (as an explicit, labeled addition beyond the official
   code) bootstrap confidence intervals and significance testing.

Each milestone follows the same inner loop: inspect → plan → implement
smallest coherent unit → unit test → smoke test → document deviations →
summarize changed files → stop for approval.

## Smallest Viable Reproduction (Phase 3 target)

A synthetic or tiny legally-usable sample (order of 10s of records, no real
patient data unless the user supplies already-licensed access) that
exercises every pipeline stage end-to-end:

parse → mock-or-real RadGraph/CheXbert label → similarity scoring → top-k
positive mining → retriever forward pass (untrained or few-step-trained) →
FAISS index build → KNN retrieval with leakage filter → mock generation →
full metric suite (including retrieval metrics and the new CI/significance
utilities) → runtime/memory measurement.

This validates plumbing and correctness, not paper-level scores.

## Explicit Non-Goals For Now

- No full MIMIC-CXR/CheXpert-scale training run.
- No attempt to reproduce headline paper numbers before the PDF is
  available to know what those numbers are.
- No LLaVA fine-tuning at full scale in an interactive Colab session
  without the user's explicit go-ahead (compute-heavy, see
  `docs/compute_requirements.md`).

## Immediate Next Step

Await either (a) the user uploading `paper/factmm_rag.pdf`, or (b) explicit
approval to proceed to Phase 1 (architecture scaffolding) using only the
official-repository-sourced understanding documented so far, with paper
cross-checks deferred to a follow-up pass once the PDF is available.
