"""Faithful reproduction of the FactMM-RAG paper.

Responsibility: implement the paper's exact pipeline — RadGraph
annotation, factual report-pair mining, the MARVEL-style multimodal
retriever, and the LLaVA-based retrieval-augmented generator — built on
top of the shared src/data, src/retrieval, src/generation, and
src/evaluation packages. Never imports from src/innovation, per
CLAUDE.md's baseline/innovation separation rule. Every implementation
choice here traces back to docs/paper_analysis.md and
docs/original_repository_audit.md, with deviations documented rather than
silently introduced.
"""
