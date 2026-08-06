"""Unit tests for src/innovation/shared/confidence.py.

Covers Cell 47 spec Section 12 tests 5-8: shared confidence is bounded
[0, 1], handles ties, handles empty candidate lists, handles one
candidate.
"""

from src.innovation.shared.confidence import compute_confidence


def test_confidence_bounded_in_unit_interval_across_many_shapes():
    cases = [
        [],
        [1.0],
        [5.0, 5.0],
        [5.0, 5.0, 5.0],
        [10.0, 1.0, 0.5],
        [1.0, 0.99999],
        [-3.0, -10.0, -50.0],
        [0.0, 0.0],
        [100.0, -100.0],
    ]
    for scores in cases:
        result = compute_confidence(scores)
        assert 0.0 <= result.value <= 1.0, f"out of bounds for {scores}: {result.value}"


def test_confidence_handles_empty_candidate_list():
    result = compute_confidence([])
    assert result.value == 0.0
    assert result.candidate_count == 0
    assert result.top1_score is None
    assert result.top2_score is None
    assert result.margin is None
    assert result.spread is None
    assert result.basis == "empty_candidate_list"


def test_confidence_handles_one_candidate():
    result = compute_confidence([7.5])
    assert result.candidate_count == 1
    assert result.top1_score == 7.5
    assert result.top2_score is None
    assert result.margin is None
    assert result.spread == 0.0
    assert result.basis == "single_candidate_no_margin"
    assert result.value == 0.0


def test_confidence_handles_ties():
    result = compute_confidence([3.0, 3.0, 3.0])
    assert result.basis == "tied_scores"
    assert result.value == 0.0
    assert result.margin == 0.0
    assert result.spread == 0.0

    # A tie specifically between the top two, with other candidates
    # present and distinct, must also be handled (not just an
    # all-identical list).
    result2 = compute_confidence([5.0, 5.0, 1.0])
    assert result2.margin == 0.0
    assert result2.value == 0.0


def test_confidence_well_separated_scores_yield_high_confidence():
    result = compute_confidence([10.0, 1.0, 0.5])
    assert result.basis == "margin_based"
    assert result.value > 0.9


def test_confidence_deterministic_across_repeated_calls():
    scores = [4.2, 3.1, 3.0, 1.5]
    r1 = compute_confidence(scores)
    r2 = compute_confidence(scores)
    assert r1 == r2
