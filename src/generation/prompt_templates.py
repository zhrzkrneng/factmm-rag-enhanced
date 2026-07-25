"""Prompt template construction.

Responsibility: build the exact VQA (non-RAG) and RAG prompt strings
specified in the paper's Figure 5 and matched by the official
build_rag_dataset.py ("Here is a report of a related patient: ...
Generate a radiology report from this image: <image>"), plus any
structured-evidence prompt variants required by innovation modules (e.g.
multi-report fusion). No implementation yet.
"""
