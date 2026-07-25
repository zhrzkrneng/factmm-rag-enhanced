"""FactMM-RAG Enhanced source package.

Layout:
    common/      Cross-cutting utilities (seeding, logging, config loading,
                 dataset manifests, shared exceptions). No ML logic.
    data/        Shared dataset parsing, patient/study split validation, and
                 integrity checks. Used by both baseline and innovation code.
    evaluation/  Shared metric implementations (retrieval + generation),
                 independent of any specific retriever/generator.
    retrieval/   Shared retrieval infrastructure: abstract retriever
                 interface, index construction, embedding export.
    generation/  Shared generation infrastructure: abstract generator
                 interface, prompt templates, RAG dataset construction.
    baseline/    Faithful reproduction of the FactMM-RAG paper, built on top
                 of the shared packages above. Never imports from innovation/.
    innovation/  Proposed extensions (adaptive retrieval, factual reranking,
                 multi-report fusion, confidence gating, uncertainty
                 detection), built on the same shared packages. Never
                 imports from baseline/.

No implementation exists yet — this package currently defines structure
only, per the project's milestone-by-milestone workflow (see CLAUDE.md).
"""
