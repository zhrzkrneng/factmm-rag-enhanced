"""Shared dataset handling.

Responsibility: parsing raw MIMIC-CXR/CheXpert-style files into the
project's record schema, building and validating patient-level splits,
running integrity checks, and generating dataset manifests. This package
is shared by both src/baseline and src/innovation — neither one
re-implements data loading independently. Contains no retrieval,
generation, or evaluation logic.
"""
