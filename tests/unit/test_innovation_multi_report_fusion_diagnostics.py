"""Unit tests for I3's diagnostics aggregation (diagnostics.py) --
Cell 50, this revision. Covers spec category (16): diagnostics
aggregation.
"""

from src.innovation.multi_report_fusion.diagnostics import aggregate_fusion_diagnostics
from src.innovation.multi_report_fusion.fusion import FusionCandidateInput, FusionConfig, select_fusion_candidates


def test_aggregate_diagnostics_empty_input_returns_neutral_zeros():
    diagnostics = aggregate_fusion_diagnostics([])
    assert diagnostics.decision_count == 0
    assert diagnostics.mean_fused_report_count == 0.0
    assert diagnostics.median_fused_report_count == 0.0
    assert diagnostics.fraction_single_report_contexts == 0.0
    assert diagnostics.fraction_multi_report_contexts == 0.0
    assert diagnostics.fraction_high_confidence == 0.0
    assert diagnostics.fraction_medium_confidence == 0.0
    assert diagnostics.fraction_low_confidence == 0.0
    assert diagnostics.candidate_shortage_rate == 0.0
    assert diagnostics.average_confidence == 0.0
    assert diagnostics.mean_evidence_blocks_per_query == 0.0
    assert diagnostics.available_candidate_count_histogram == {}


def _decision(scores):
    candidates = [FusionCandidateInput(candidate_key=f"c{i}", score=s) for i, s in enumerate(scores)]
    return select_fusion_candidates(candidates, FusionConfig())


def test_aggregate_diagnostics_counts_and_means_match_hand_computed_values():
    high = _decision([10.0, 1.0, 0.0])          # band high, selected=1
    medium = _decision([5.0, 3.0, 0.0])         # band medium, selected=2
    low = _decision([5.0, 4.5, 4.0, 3.0, 2.0])  # band low, selected=3
    decisions = [high, medium, low]

    diagnostics = aggregate_fusion_diagnostics(decisions)

    assert diagnostics.decision_count == 3
    assert diagnostics.mean_fused_report_count == (1 + 2 + 3) / 3
    assert diagnostics.median_fused_report_count == 2
    assert diagnostics.fraction_single_report_contexts == 1 / 3
    assert diagnostics.fraction_multi_report_contexts == 2 / 3
    assert diagnostics.fraction_high_confidence == 1 / 3
    assert diagnostics.fraction_medium_confidence == 1 / 3
    assert diagnostics.fraction_low_confidence == 1 / 3
    assert diagnostics.mean_evidence_blocks_per_query == diagnostics.mean_fused_report_count


def test_aggregate_diagnostics_candidate_shortage_rate():
    shortage = _decision([5.0, 5.0])  # tied -> low band, desired=3, only 2 available -> capped
    plenty = _decision([5.0, 4.5, 4.0, 3.0, 2.0])  # low band, desired=3, satisfied
    diagnostics = aggregate_fusion_diagnostics([shortage, plenty])
    assert diagnostics.candidate_shortage_rate == 0.5


def test_aggregate_diagnostics_available_candidate_count_histogram():
    d1 = _decision([5.0, 3.0])       # available=2
    d2 = _decision([5.0, 3.0, 1.0])  # available=3
    d3 = _decision([9.0, 1.0])       # available=2
    diagnostics = aggregate_fusion_diagnostics([d1, d2, d3])
    assert diagnostics.available_candidate_count_histogram == {"2": 2, "3": 1}


def test_aggregate_diagnostics_empty_candidate_decision_counts_as_zero_available():
    empty_decision = select_fusion_candidates([], FusionConfig())
    diagnostics = aggregate_fusion_diagnostics([empty_decision])
    assert diagnostics.decision_count == 1
    assert diagnostics.mean_fused_report_count == 0.0
    assert diagnostics.fraction_single_report_contexts == 0.0
    assert diagnostics.fraction_multi_report_contexts == 0.0
    assert diagnostics.available_candidate_count_histogram == {"0": 1}


def test_to_json_dict_includes_all_fields():
    diagnostics = aggregate_fusion_diagnostics([_decision([5.0, 3.0, 0.0])])
    payload = diagnostics.to_json_dict()
    for key in (
        "decision_count", "mean_fused_report_count", "median_fused_report_count",
        "fraction_single_report_contexts", "fraction_multi_report_contexts",
        "fraction_high_confidence", "fraction_medium_confidence", "fraction_low_confidence",
        "candidate_shortage_rate", "average_confidence", "mean_evidence_blocks_per_query",
        "available_candidate_count_histogram",
    ):
        assert key in payload
