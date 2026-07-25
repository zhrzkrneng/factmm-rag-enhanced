"""Raw dataset parsing.

Responsibility: parse raw MIMIC-CXR/CheXpert-style source files (parallel
image-path / finding / impression files, or dataset-native formats) into
the project's schema.py record structure, selecting the frontal view per
study and concatenating finding+impression text where required by the
reproduction (per the paper's stated preprocessing). No implementation
yet.
"""
