"""Dataset manifest generation.

Responsibility: build a SplitManifest (schema defined in
src/common/manifest.py) from a list of ReportRecords, and serialize it
to/from JSON — recording patient IDs, study IDs, relative image paths,
and split-level counts, but never raw report text. This is the last
stage of the Milestone 2.1 pipeline: it is expected to run only after
IntegrityChecker.check_and_raise() and PatientSplitValidator.
check_and_raise() have both passed cleanly for the record set being
manifested.
"""

from __future__ import annotations

import json
import os
from typing import List

from src.common.manifest import ManifestEntry, SplitManifest
from src.data.schema import ReportRecord


class ManifestBuilder:
    """Builds and serializes SplitManifests from ReportRecords."""

    def build(
        self, split_name: str, records: List[ReportRecord]
    ) -> SplitManifest:
        """Build a SplitManifest from a list of ReportRecords.

        Only identifiers and image paths are carried over — `finding`
        and `impression` text is deliberately not read from `records`
        here, so it can never end up in the manifest.
        """
        entries = [
            ManifestEntry(
                patient_id=record.patient_id,
                study_id=record.study_id,
                image_paths=list(record.image_paths),
                dataset=record.dataset,
            )
            for record in records
        ]
        return SplitManifest(split_name=split_name, entries=entries)

    def save(self, manifest: SplitManifest, output_path: str) -> None:
        """Serialize a SplitManifest to a JSON file.

        Creates parent directories if they don't already exist.
        """
        payload = {
            "split_name": manifest.split_name,
            "record_count": manifest.record_count,
            "patient_count": manifest.patient_count,
            "entries": [
                {
                    "patient_id": entry.patient_id,
                    "study_id": entry.study_id,
                    "image_paths": entry.image_paths,
                    "dataset": entry.dataset,
                }
                for entry in manifest.entries
            ],
        }
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, input_path: str) -> SplitManifest:
        """Load a previously-saved SplitManifest from a JSON file."""
        with open(input_path, "r") as f:
            payload = json.load(f)
        entries = [
            ManifestEntry(
                patient_id=entry["patient_id"],
                study_id=entry["study_id"],
                image_paths=entry["image_paths"],
                dataset=entry["dataset"],
            )
            for entry in payload["entries"]
        ]
        return SplitManifest(
            split_name=payload["split_name"], entries=entries
        )
