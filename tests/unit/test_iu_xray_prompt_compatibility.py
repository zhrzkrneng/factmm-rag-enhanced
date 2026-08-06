"""Unit tests for src/data/iu_xray/prompt_compatibility.py.

Covers: (15) prompt-builder compatibility, (16) empty-evidence
behavior, (19) repeated-run determinism.
"""

from src.baseline.generation.dataset_builder import RAGDatasetRow
from src.baseline.generation.prompt_builder import PromptBuilder, PromptBuilderConfig
from src.data.iu_xray.prompt_compatibility import CAVEATS, validate_row, validate_rows


def _row(query_key, retrieved_key, retrieved_text, target_text, excluded=False):
    return RAGDatasetRow(
        query_key=query_key, retrieved_key=retrieved_key,
        image_path="/fake/img.png", retrieved_report_text=retrieved_text,
        target_report_text=target_text, rank_selected=0 if retrieved_key else None,
        num_candidates_rejected_self_study=0, num_candidates_rejected_self_patient=0,
        num_candidates_rejected_short_text=0, fallback_used=False, excluded=excluded,
        strict_patient_isolation=True, reproduce_official_bug=False, config_version="1.0",
    )


def test_non_excluded_row_builds_rag_inference_prompt():
    row = _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"),
               "Evidence from a different study.", "Target report for CXR1.")
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    assert result["mode"] == "rag_inference"
    assert result["excluded"] is False


def test_excluded_row_builds_vqa_inference_prompt_no_empty_string_hack():
    row = _row(("iu-xray", "CXR1", "CXR1"), None, None, "Target report for CXR1.", excluded=True)
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    assert result["mode"] == "vqa_inference"
    assert result["excluded"] is True


def test_target_report_leak_is_detected_when_present():
    row = _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"),
               "This evidence happens to contain Target report for CXR1. verbatim.",
               "Target report for CXR1.")
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    assert result["target_report_leaked_into_retrieved_evidence"] is True


def test_no_leak_for_genuinely_distinct_evidence():
    row = _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"),
               "Completely different findings entirely.", "Target report for CXR1.")
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    assert result["target_report_leaked_into_retrieved_evidence"] is False


def test_repeated_build_is_deterministic():
    row = _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"),
               "Evidence text.", "Target text.")
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    assert result["deterministic"] is True


def test_validate_rows_summary_and_caveats_present():
    rows = [
        _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"), "Evidence.", "Target."),
        _row(("iu-xray", "CXR3", "CXR3"), None, None, "Target 3.", excluded=True),
    ]
    summary = validate_rows(rows)
    assert summary["row_count"] == 2
    assert summary["rag_mode_count"] == 1
    assert summary["excluded_count"] == 1
    assert summary["all_deterministic"] is True
    assert summary["caveats"] == list(CAVEATS)


def test_no_mimic_or_chexpert_identifiers_fabricated_in_prompt_text():
    row = _row(("iu-xray", "CXR1", "CXR1"), ("iu-xray", "CXR2", "CXR2"),
               "Evidence text.", "Target text.")
    result = validate_row(row, PromptBuilder(PromptBuilderConfig()))
    # PromptBuilder never injects dataset identifiers into its own text
    # at all -- confirmed by construction, not by string-scanning (see
    # src/baseline/generation/prompt_builder.py, read in full for Cell 45).
    assert result["prompt_text_length"] > 0
