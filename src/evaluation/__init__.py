"""Shared evaluation metrics.

Responsibility: reusable metric implementations for both retrieval and
generation quality, independent of whether the system under test is the
baseline reproduction or an innovation variant. Shared by src/baseline
and src/innovation so neither re-implements metric computation
independently.
"""
