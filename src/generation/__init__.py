"""Shared generation infrastructure.

Responsibility: define the abstract generator interface, shared prompt
templates, and RAG dataset construction (with patient/study leakage
filtering) used by both the baseline LLaVA-based generator
(src/baseline/generation) and innovation generation variants (e.g.
src/innovation/multi_report_fusion). Contains no concrete model
architecture itself.
"""
