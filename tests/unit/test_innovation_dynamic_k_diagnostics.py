"""Unit tests for src/innovation/adaptive_retrieval/diagnostics.py."""

import pytest

from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig, select_adaptive_k
from src.innovation.adaptive_retrieval.diagnostics import (
    aggregate_adaptive_k_diagnostics,
    verify_batch_determinism,
)

CONFIG = AdaptiveKConfig(min_k=1, max_k=5, default_k=2)


def _decision(candidates, scores):
    return select_adaptive_k(candidates, scores, CONFIG)


def test_empty_batch_returns_neutral_diagnostics():
    diag = aggregate_adaptive_k_diagnostics([])
    assert diag.decision_count == 0
    assert diag.mean_selected_k == 0.0
    assert diag.fraction_high_confidence == 0.0
    assert diag.candidate_shortage_rate == 0.0


def test_aggregation_over_mixed_confidence_batch():
    decisions = [
        _decision(["a", "b", "c"], [10.0, 1.0, 0.5]),  # high
        _decision(["a", "b", "c"], [5.0, 4.9, 4.8]),  # medium
        _decision(["a", "b", "c", "d", "e", "f"], [5.0] * 6),  # low
    ]
    diag = aggregate_adaptive_k_diagnostics(decisions)
    assert diag.decision_count == 3
    assert diag.fraction_high_confidence == pytest.approx(1 / 3)
    assert diag.fraction_medium_confidence == pytest.approx(1 / 3)
    assert diag.fraction_low_confidence == pytest.approx(1 / 3)
    assert diag.min_selected_k <= diag.median_selected_k <= diag.max_selected_k
    assert 0.0 <= diag.average_confidence <= 1.0


def test_candidate_shortage_rate_detected():
    decisions = [
        _decision(["a", "b"], [5.0, 5.0]),  # low confidence, only 2 available (shortage: wants 5)
        _decision(["a", "b", "c", "d", "e"], [5.0] * 5),  # low confidence, exactly 5 available (no shortage)
    ]
    diag = aggregate_adaptive_k_diagnostics(decisions)
    assert diag.candidate_shortage_rate == pytest.approx(0.5)


def test_mean_and_median_selected_k():
    decisions = [
        _decision(["a"], [1.0]),  # selected_k = 1 (only 1 available)
        _decision(["a", "b", "c"], [5.0, 4.9, 4.8]),  # medium -> default_k=2
    ]
    diag = aggregate_adaptive_k_diagnostics(decisions)
    assert diag.mean_selected_k == pytest.approx((1 + 2) / 2)


def test_diagnostics_to_json_dict_shape():
    decisions = [_decision(["a", "b", "c"], [10.0, 1.0, 0.5])]
    diag = aggregate_adaptive_k_diagnostics(decisions)
    payload = diag.to_json_dict()
    for key in (
        "decision_count", "mean_selected_k", "median_selected_k", "min_selected_k",
        "max_selected_k", "fraction_high_confidence", "fraction_medium_confidence",
        "fraction_low_confidence", "candidate_shortage_rate", "average_confidence",
        "deterministic_decision_count", "total_decisions_checked_for_determinism",
    ):
        assert key in payload


def test_verify_batch_determinism_all_match():
    candidates, scores = ["a", "b", "c"], [5.0, 4.0, 3.0]
    run1 = [_decision(candidates, scores) for _ in range(3)]
    run2 = [_decision(candidates, scores) for _ in range(3)]
    matched = verify_batch_determinism(run1, run2)
    assert matched == 3


def test_verify_batch_determinism_rejects_length_mismatch():
    with pytest.raises(ValueError):
        verify_batch_determinism([_decision(["a"], [1.0])], [])


def test_deterministic_decision_count_passed_through():
    decisions = [_decision(["a", "b"], [5.0, 4.0])]
    diag = aggregate_adaptive_k_diagnostics(
        decisions, deterministic_decision_count=5, total_decisions_checked_for_determinism=5,
    )
    assert diag.deterministic_decision_count == 5
    assert diag.total_decisions_checked_for_determinism == 5
