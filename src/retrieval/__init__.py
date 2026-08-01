"""Shared retrieval infrastructure.

Responsibility: define the abstract retriever interface and shared FAISS
indexing/embedding-export utilities that both the baseline MARVEL-style
retriever (src/baseline/retrieval) and any innovation retrieval variants
(src/innovation/adaptive_retrieval, src/innovation/factual_reranking)
build on top of. Contains no concrete model architecture itself.
"""
