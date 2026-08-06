"""IU X-Ray deterministic target-report policy.

Responsibility: derive one target-report string per IU X-Ray study,
following an explicit, documented, deterministic precedence -- never
inferring or fabricating missing clinical text. This is the text every
downstream retrieval/generation adapter in this package uses as the
study's "finding"/target text.

Policy (highest precedence first):
  1. findings + impression, when both are available
  2. findings only, when impression is absent
  3. impression only, when findings is absent
  4. full_report only, when neither of the above can produce text and
     the record is otherwise valid

Branch 4 is implemented for defensive completeness and is exercised by
this module's own unit tests (synthetic fixtures), but is DISCLOSED as
practically unreachable for any record Cell 43 marked "valid": Cell 43's
own exclusion rule already requires findings is not None OR impression
is not None for validation_status="valid" -- so branches 1-3 always
suffice for a usable record. Recorded here, not hidden, per this cell's
"record which policy branch each sample used" requirement.
"""

from __future__ import annotations

import re
from typing import NamedTuple, Optional, Tuple

from src.data.iu_xray.records import IuXrayCanonicalRecord

FINDINGS_AND_IMPRESSION = "findings_and_impression"
FINDINGS_ONLY = "findings_only"
IMPRESSION_ONLY = "impression_only"
FULL_REPORT_ONLY = "full_report_only"
UNAVAILABLE = "unavailable"

POLICY_TABLE: Tuple[dict, ...] = (
    {"rank": 1, "branch": FINDINGS_AND_IMPRESSION,
     "condition": "findings is not None and impression is not None",
     "description": "findings and impression concatenated with a single "
                     "space (or just one copy if they are identical "
                     "strings -- never duplicated)"},
    {"rank": 2, "branch": FINDINGS_ONLY,
     "condition": "findings is not None and impression is None",
     "description": "findings text alone"},
    {"rank": 3, "branch": IMPRESSION_ONLY,
     "condition": "findings is None and impression is not None",
     "description": "impression text alone"},
    {"rank": 4, "branch": FULL_REPORT_ONLY,
     "condition": "findings is None and impression is None and "
                   "full_report is not None",
     "description": "full_report text (comparison+indication+findings+"
                     "impression, whichever were present) -- disclosed "
                     "as practically unreachable for any Cell-43-valid "
                     "record, since Cell 43's own validity rule already "
                     "requires findings or impression to be non-None"},
    {"rank": 5, "branch": UNAVAILABLE,
     "condition": "none of the above can produce text",
     "description": "no target report text could be derived; never "
                     "fabricated -- callers must handle this explicitly "
                     "(e.g. exclude the record), never substitute a "
                     "placeholder string"},
)


class TargetReportResult(NamedTuple):
    text: Optional[str]
    policy_branch: str


def _normalize_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized or None


def build_target_report(record: IuXrayCanonicalRecord) -> TargetReportResult:
    """Derives the deterministic target-report text for one study.

    Never invents missing clinical text. Whitespace is normalized
    deterministically (collapsed runs of whitespace, stripped ends) --
    the same normalization Cell 43's canonicalization already applied,
    reapplied here defensively rather than assumed.
    """
    findings = _normalize_whitespace(record.findings)
    impression = _normalize_whitespace(record.impression)

    if findings is not None and impression is not None:
        text = findings if findings == impression else f"{findings} {impression}"
        return TargetReportResult(text=text, policy_branch=FINDINGS_AND_IMPRESSION)
    if findings is not None:
        return TargetReportResult(text=findings, policy_branch=FINDINGS_ONLY)
    if impression is not None:
        return TargetReportResult(text=impression, policy_branch=IMPRESSION_ONLY)

    full_report = _normalize_whitespace(record.full_report)
    if full_report is not None:
        return TargetReportResult(text=full_report, policy_branch=FULL_REPORT_ONLY)

    return TargetReportResult(text=None, policy_branch=UNAVAILABLE)
