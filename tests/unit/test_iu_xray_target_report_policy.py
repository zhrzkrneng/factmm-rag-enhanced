"""Unit tests for src/data/iu_xray/target_report_policy.py.

Covers: (10) target report policy, (11) findings-only handling,
(12) impression-only handling, (13) missing optional metadata,
(19) repeated-run determinism (pure-function determinism).
"""

from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.target_report_policy import (
    FINDINGS_AND_IMPRESSION,
    FINDINGS_ONLY,
    FULL_REPORT_ONLY,
    IMPRESSION_ONLY,
    POLICY_TABLE,
    UNAVAILABLE,
    build_target_report,
)


def _record(**overrides):
    payload = dict(
        dataset_name="iu_xray", study_id="CXR1", report_id="CXR1",
        image_ids=["CXR1_IM-0001-1001"], image_paths=["/fake/CXR1_IM-0001-1001.png"],
        image_count=1, findings="Findings text.", impression="Impression text.",
        indication=None, comparison=None, full_report="Findings text. Impression text.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def test_policy_table_has_all_five_branches_in_rank_order():
    branches = [row["branch"] for row in POLICY_TABLE]
    assert branches == [FINDINGS_AND_IMPRESSION, FINDINGS_ONLY, IMPRESSION_ONLY,
                         FULL_REPORT_ONLY, UNAVAILABLE]


def test_findings_and_impression_both_present():
    record = _record(findings="The heart is normal.", impression="No acute disease.")
    result = build_target_report(record)
    assert result.text == "The heart is normal. No acute disease."
    assert result.policy_branch == FINDINGS_AND_IMPRESSION


def test_findings_and_impression_identical_not_duplicated():
    record = _record(findings="Stable.", impression="Stable.")
    result = build_target_report(record)
    assert result.text == "Stable."  # not "Stable. Stable."
    assert result.policy_branch == FINDINGS_AND_IMPRESSION


def test_findings_only_when_impression_absent():
    record = _record(findings="Lungs are clear.", impression=None)
    result = build_target_report(record)
    assert result.text == "Lungs are clear."
    assert result.policy_branch == FINDINGS_ONLY


def test_impression_only_when_findings_absent():
    record = _record(findings=None, impression="No acute cardiopulmonary process.")
    result = build_target_report(record)
    assert result.text == "No acute cardiopulmonary process."
    assert result.policy_branch == IMPRESSION_ONLY


def test_full_report_fallback_when_findings_and_impression_both_absent():
    record = _record(findings=None, impression=None, full_report="Comparison text. Indication text.")
    result = build_target_report(record)
    assert result.text == "Comparison text. Indication text."
    assert result.policy_branch == FULL_REPORT_ONLY


def test_unavailable_when_nothing_can_produce_text():
    record = _record(findings=None, impression=None, full_report=None)
    result = build_target_report(record)
    assert result.text is None
    assert result.policy_branch == UNAVAILABLE


def test_missing_optional_metadata_indication_comparison_do_not_affect_policy():
    record = _record(findings="Findings.", impression="Impression.",
                      indication=None, comparison=None)
    result = build_target_report(record)
    assert result.text == "Findings. Impression."


def test_whitespace_normalized_deterministically():
    record = _record(findings="The   heart  \n size is  normal.  ", impression="  No  disease.  ")
    result = build_target_report(record)
    assert result.text == "The heart size is normal. No disease."


def test_never_invents_missing_text():
    record = _record(findings=None, impression="Impression only.")
    result = build_target_report(record)
    assert "Findings" not in (result.text or "")
    assert result.text == "Impression only."


def test_repeated_calls_are_deterministic():
    record = _record(findings="A.", impression="B.")
    first = build_target_report(record)
    second = build_target_report(record)
    assert first == second
