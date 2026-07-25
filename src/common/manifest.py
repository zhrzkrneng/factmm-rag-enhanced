"""Dataset manifest schema.

Responsibility: define the typed schema for a dataset manifest — the
record of which patient IDs, study IDs, and (relative) image paths belong
to each split, and split-level record counts. Manifests never contain raw
report text or images, only identifiers and paths, so they are safe to
version-control and inspect without exposing PHI. Used by src/data to
build manifests and by src/data's split-validation logic to detect
patient/study leakage across train/valid/test. No implementation yet.
"""
