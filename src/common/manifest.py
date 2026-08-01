"""Dataset manifest schema.

Responsibility: define the typed schema for a dataset manifest — the
record of which patient IDs, study IDs, and (relative) image paths
belong to one split, plus split-level counts. Manifests never contain
raw report text or images, only identifiers and paths, so they are safe
to inspect and (in principle) version-control without exposing report
content — though per docs/data_requirements.md, even ID-only manifests
are still treated as sensitive (patient/study IDs from a credentialed
dataset) and are never actually committed.

Design note (deviation from the original Phase 1 sketch): the original
plan called for two classes, `DatasetManifest` and `SplitManifest`.
Nothing in this project currently needs a combined multi-split wrapper
— a plain `Dict[str, List[ReportRecord]]` (what
src.data.splits.PatientSplitValidator already accepts) covers that need
— so only `SplitManifest` (one manifest per split, matching the actual
`data/manifests/{split}_manifest.json` output file) is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ManifestEntry:
    """One record's identifying metadata in a manifest — no report text.

    Attributes:
        patient_id: de-identified patient identifier.
        study_id: de-identified study identifier.
        image_paths: relative image file paths for this study.
        dataset: source dataset tag (e.g. "mimic-cxr", "chexpert").
    """

    patient_id: str
    study_id: str
    image_paths: List[str]
    dataset: str


@dataclass(frozen=True)
class SplitManifest:
    """Manifest for one dataset split (e.g. "train", "valid", "test").

    Attributes:
        split_name: the split this manifest describes.
        entries: per-record identifying metadata, in source order.
    """

    split_name: str
    entries: List[ManifestEntry]

    @property
    def record_count(self) -> int:
        """Number of records (studies) in this split."""
        return len(self.entries)

    @property
    def patient_count(self) -> int:
        """Number of distinct patients in this split."""
        return len({entry.patient_id for entry in self.entries})
