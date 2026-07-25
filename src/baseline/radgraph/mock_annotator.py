"""Deterministic mock RadGraph/CheXbert annotator.

Responsibility: provide a lightweight, deterministic stand-in for the
real annotator (src/baseline/radgraph/annotator.py) so unit tests and
smoke tests can exercise the full pipeline without GPU access or
RadGraph/CheXbert model weights. Must implement the same interface as the
real annotator. No implementation yet.
"""
