# FactMM-RAG Enhanced

## Project Architecture

Phase 1 scaffolding only — no model code is implemented yet. Every module
below currently contains empty Python files with docstrings describing
responsibility; see `docs/reproduction_plan.md` for the milestone order in
which they will be implemented.

```
src/
├── common/                   Cross-cutting utilities: seeding, logging,
│                              config loading/validation, dataset manifest
│                              schema, shared exceptions. No ML logic.
├── data/                     Shared dataset parsing, patient/study split
│                              validation, integrity checks, and manifest
│                              generation. Used by both baseline and
│                              innovation code.
├── evaluation/               Shared metric implementations: F1CheXbert,
│                              F1RadGraph, ROUGE-L, BERTScore, MRR/Recall/
│                              NDCG, and bootstrap CI / significance
│                              testing (our own addition beyond the paper).
├── retrieval/                Shared retrieval infrastructure: abstract
│                              retriever interface, FAISS indexing,
│                              embedding export.
├── generation/                Shared generation infrastructure: abstract
│                              generator interface, prompt templates, RAG
│                              dataset construction with patient/study
│                              leakage filtering.
├── baseline/                 Faithful reproduction of the FactMM-RAG
│   ├── radgraph/              paper. Never imports from innovation/.
│   ├── pair_mining/            radgraph/    RadGraph + CheXbert annotation
│   ├── retrieval/               pair_mining/ Factual report-pair mining
│   ├── generation/               retrieval/  MARVEL-style multimodal retriever
│   ├── evaluation/                generation/ LLaVA-based RAG generator
│   └── utils/                      evaluation/ Paper-comparison reporting
│                                     utils/      Baseline-only helpers
└── innovation/                Proposed extensions beyond the paper. Never
    ├── adaptive_retrieval/     imports from baseline/. Nothing here is
    ├── factual_reranking/      implemented until the baseline smoke test
    ├── multi_report_fusion/    passes (Phase 3).
    ├── confidence_gating/        adaptive_retrieval/  Innovation A
    ├── uncertainty/               factual_reranking/   Innovation B
    └── evaluation/                 multi_report_fusion/ Innovation C
                                      confidence_gating/   Innovation D
                                       uncertainty/          Innovation E
                                        evaluation/  Innovation-specific metrics

configs/
├── data/          Dataset paths and preprocessing options for src/data
├── baseline/      Baseline hyperparameters (thresholds, training settings)
├── innovation/    Innovation module settings
└── experiments/   Named experiment configs composing the above
```

Innovation F (optional longitudinal context) is intentionally not
scaffolded yet — it depends on prior-study data availability that has not
been confirmed (see `docs/innovation_proposals.md`).

See `docs/paper_analysis.md`, `docs/original_repository_audit.md`, and
`docs/reproduction_matrix.md` for the source material this architecture is
built against, and `docs/risk_register.md` for known paper-vs-official-code
discrepancies that the baseline modules must expose as configurable choices
rather than resolve silently.
