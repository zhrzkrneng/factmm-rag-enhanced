"""Dataset integrity checks.

Responsibility: detect missing image files, missing/empty finding or
impression text, duplicate (patient_id, study_id) pairs within a split,
and records with no identifiable image, per the integrity checks
required in docs/data_requirements.md.

Design: `IntegrityChecker.check()` collects *every* issue across the
whole record set rather than stopping at the first one, so a complete
diagnostic report can be produced (matching the
`{split}_integrity_report.json` artifact described in
docs/data_requirements.md). `IntegrityChecker.check_and_raise()` then
enforces the "raise loudly, don't silently skip bad records" rule by
raising `DataIntegrityError` if that report is non-empty.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from typing import List

from src.common.exceptions import DataIntegrityError
from src.data.schema import ReportRecord


@dataclass(frozen=True)
class IntegrityIssue:
    """One integrity violation found for one record.

    Attributes:
        kind: short machine-readable category, one of:
            "empty_finding", "empty_impression", "no_image_paths",
            "missing_image_file", "duplicate_study".
        patient_id: the affected record's patient_id.
        study_id: the affected record's study_id.
        detail: a human-readable explanation, safe to log (never
            contains report text or raw image bytes).
    """

    kind: str
    patient_id: str
    study_id: str
    detail: str


class IntegrityChecker:
    """Checks a list of ReportRecords for integrity violations.

    Args:
        image_root: filesystem directory that ReportRecord image paths
            are relative to, used to check that image files actually
            exist on disk. Required because ReportRecord itself only
            stores relative paths (see src/data/schema.py).
    """

    def __init__(self, image_root: str):
        self.image_root = image_root

    def check(self, records: List[ReportRecord]) -> List[IntegrityIssue]:
        """Return every integrity issue found across all records.

        Never raises on its own — this method's job is to produce a
        complete report. Use check_and_raise() to enforce failure.
        """
        issues: List[IntegrityIssue] = []
        seen_study_keys = set()

        for record in records:
            if not record.finding.strip():
                issues.append(IntegrityIssue(
                    kind="empty_finding",
                    patient_id=record.patient_id,
                    study_id=record.study_id,
                    detail="finding text is empty or whitespace-only",
                ))
            if not record.impression.strip():
                issues.append(IntegrityIssue(
                    kind="empty_impression",
                    patient_id=record.patient_id,
                    study_id=record.study_id,
                    detail="impression text is empty or whitespace-only",
                ))

            if not record.image_paths:
                issues.append(IntegrityIssue(
                    kind="no_image_paths",
                    patient_id=record.patient_id,
                    study_id=record.study_id,
                    detail="record has zero image paths",
                ))
            else:
                frontal_path = os.path.join(
                    self.image_root, record.frontal_image_path
                )
                if not os.path.isfile(frontal_path):
                    issues.append(IntegrityIssue(
                        kind="missing_image_file",
                        patient_id=record.patient_id,
                        study_id=record.study_id,
                        detail=f"frontal image file not found on disk "
                               f"(relative path: "
                               f"{record.frontal_image_path!r})",
                    ))

            study_key = (record.patient_id, record.study_id)
            if study_key in seen_study_keys:
                issues.append(IntegrityIssue(
                    kind="duplicate_study",
                    patient_id=record.patient_id,
                    study_id=record.study_id,
                    detail="(patient_id, study_id) pair appears more "
                           "than once in this record set",
                ))
            else:
                seen_study_keys.add(study_key)

        return issues

    def check_and_raise(self, records: List[ReportRecord]) -> None:
        """Run check() and raise DataIntegrityError if any issue exists.

        Raises:
            DataIntegrityError: with a per-kind count summary, if
                check() found one or more issues.
        """
        issues = self.check(records)
        if issues:
            counts = Counter(issue.kind for issue in issues)
            summary = ", ".join(
                f"{kind}={count}" for kind, count in sorted(counts.items())
            )
            raise DataIntegrityError(
                f"{len(issues)} integrity issue(s) found: {summary}"
            )
