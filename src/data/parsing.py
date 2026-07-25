"""Raw dataset parsing.

Responsibility: parse the official FactMM-RAG JSON schema
(`[{"image": [...], "finding": ..., "impression": ...}, ...]`, produced
by the official repo's `data/parse.py` for both MIMIC-CXR and CheXpert)
into `src.data.schema.ReportRecord` objects, extracting patient/study
identifiers from each record's image path.

Design note (deviation from the original Phase 1 sketch): the original
architecture plan called for two separate classes, `MimicCxrParser` and
`CheXpertParser`. On inspection, both datasets are distributed in the
exact same JSON schema and differ only in the patient/study directory
naming convention embedded in their image paths (MIMIC-CXR:
`pNNNNNNNN/sNNNNNNNN/`; CheXpert: `patientNNNNN/studyN/`). A single
`JsonReportParser`, parameterized by `dataset`, covers both without
duplicating identical parsing logic across two classes.
"""

from __future__ import annotations

import json
import re
from typing import List, Tuple

from src.data.schema import ReportRecord

# Patient/study ID patterns, keyed by dataset tag. Each pattern must
# capture exactly two groups: (patient_id, study_id).
_ID_PATTERNS = {
    "mimic-cxr": re.compile(r"/(p\d+)/(s\d+)/"),
    "chexpert": re.compile(r"/(patient\d+)/(study\d+)/"),
}


class JsonReportParser:
    """Parses the official FactMM-RAG JSON schema into ReportRecords.

    Args:
        dataset: which dataset's patient/study ID convention to use for
            extracting identifiers from image paths. Must be one of the
            keys in `_ID_PATTERNS` ("mimic-cxr" or "chexpert").
    """

    def __init__(self, dataset: str):
        if dataset not in _ID_PATTERNS:
            raise ValueError(
                f"Unknown dataset {dataset!r}; expected one of "
                f"{sorted(_ID_PATTERNS)}"
            )
        self.dataset = dataset
        self._id_pattern = _ID_PATTERNS[dataset]

    def parse_file(self, json_path: str) -> List[ReportRecord]:
        """Parse one JSON file into a list of ReportRecords.

        Args:
            json_path: path to a JSON file matching the official
                `[{"image": [...], "finding": ..., "impression": ...}]`
                schema.

        Returns:
            One ReportRecord per entry in the source JSON, in order.

        Raises:
            ValueError: if any record's first image path doesn't match
                this parser's dataset ID convention — this indicates a
                genuinely malformed or mismatched-dataset input, not a
                recoverable edge case, so it is raised rather than
                skipped.
        """
        with open(json_path, "r") as f:
            raw_records = json.load(f)

        records = []
        for raw in raw_records:
            image_paths = list(raw["image"])
            patient_id, study_id = self._extract_ids(image_paths[0])
            records.append(
                ReportRecord(
                    image_paths=image_paths,
                    finding=raw["finding"],
                    impression=raw["impression"],
                    patient_id=patient_id,
                    study_id=study_id,
                    dataset=self.dataset,
                )
            )
        return records

    def _extract_ids(self, image_path: str) -> Tuple[str, str]:
        """Extract (patient_id, study_id) from an image path."""
        match = self._id_pattern.search(image_path)
        if not match:
            raise ValueError(
                f"Could not extract patient/study IDs from image path "
                f"{image_path!r} using the {self.dataset!r} convention "
                f"(pattern: {self._id_pattern.pattern!r}). The path may "
                f"not follow the expected directory layout, or this may "
                f"be the wrong dataset for this parser."
            )
        return match.group(1), match.group(2)
