"""Patient-level split validation.

Responsibility: independently verify — never assume — that no
patient_id appears in more than one of {train, valid, test}. This is a
stricter, standalone check than the official FactMM-RAG code's
RAG-construction-time filter (which only excludes same-study/
same-patient *retrieval candidates* when building generation prompts,
not a check over the split assignment itself) — see
docs/data_requirements.md.

Follows the same collect-then-raise pattern as
src/data/integrity.py's IntegrityChecker: check() reports every leaking
patient_id, check_and_raise() enforces the fail-loud rule.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.common.exceptions import PatientLeakageError
from src.data.schema import ReportRecord

_PREVIEW_LIMIT = 5


@dataclass(frozen=True)
class LeakageIssue:
    """One patient_id found in more than one split.

    Attributes:
        patient_id: the leaking patient_id.
        splits: sorted tuple of split names this patient_id appears in.
    """

    patient_id: str
    splits: Tuple[str, ...]


class PatientSplitValidator:
    """Validates that no patient_id appears in more than one split.

    A patient having multiple *studies* within the *same* split is
    normal and not flagged — only cross-split appearance is a leak.
    """

    def check(
        self, split_records: Dict[str, List[ReportRecord]]
    ) -> List[LeakageIssue]:
        """Return every patient_id found in more than one split.

        Args:
            split_records: mapping from split name (e.g. "train",
                "valid", "test") to the ReportRecords assigned to it.
        """
        patient_to_splits: Dict[str, set] = defaultdict(set)
        for split_name, records in split_records.items():
            for record in records:
                patient_to_splits[record.patient_id].add(split_name)

        issues = [
            LeakageIssue(patient_id=patient_id, splits=tuple(sorted(splits)))
            for patient_id, splits in patient_to_splits.items()
            if len(splits) > 1
        ]
        return sorted(issues, key=lambda issue: issue.patient_id)

    def check_and_raise(
        self, split_records: Dict[str, List[ReportRecord]]
    ) -> None:
        """Run check() and raise PatientLeakageError if any leak exists.

        Raises:
            PatientLeakageError: listing (up to _PREVIEW_LIMIT) leaking
                patient_ids and which splits each was found in, if
                check() found one or more.
        """
        issues = self.check(split_records)
        if issues:
            preview = ", ".join(
                f"{issue.patient_id} in {issue.splits}"
                for issue in issues[:_PREVIEW_LIMIT]
            )
            remaining = len(issues) - _PREVIEW_LIMIT
            more = f" (+{remaining} more)" if remaining > 0 else ""
            raise PatientLeakageError(
                f"{len(issues)} patient_id(s) found in more than one "
                f"split: {preview}{more}"
            )
