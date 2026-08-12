"""Unit tests for select_adaptive_k in
src/innovation/adaptive_retrieval/adaptive_k.py (Cell 48 real algorithm).

Covers Cell 48 spec Section D's required edge cases: empty candidate
list, one candidate, fewer candidates than selected K, tied scores,
identical score range, negative scores, unsorted input, NaN/infinite
score rejection, min_k == max_k, deterministic repeated execution.
"""

import pytest

from src.innovation.adaptive_retrieval.adaptive_k import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    AdaptiveKConfig,
    select_adaptive_k,
)

DEFAULT_CONFIG = AdaptiveKConfig(min_k=1, max_k=5, default_k=2)


def test_high_confidence_selects_min_k():
    decision = select_adaptive_k(["a", "b", "c"], [10.0, 1.0, 0.5], DEFAULT_CONFIG)
    assert decision.confidence_band == BAND_HIGH
    assert decision.selected_k == DEFAULT_CONFIG.min_k
    assert decision.selected_candidates == ("a",)


def test_low_confidence_selects_max_k_when_enough_candidates():
    candidates = ["a", "b", "c", "d", "e", "f"]
    scores = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    decision = select_adaptive_k(candidates, scores, DEFAULT_CONFIG)
    assert decision.confidence_band == BAND_LOW
    assert decision.selected_k == DEFAULT_CONFIG.max_k
    assert decision.selected_candidates == tuple(candidates[: DEFAULT_CONFIG.max_k])


def test_medium_confidence_selects_default_k():
    decision = select_adaptive_k(["a", "b", "c"], [5.0, 4.9, 4.8], DEFAULT_CONFIG)
    assert decision.confidence_band == BAND_MEDIUM
    assert decision.selected_k == DEFAULT_CONFIG.default_k


def test_true_default_config_maps_all_three_bands_to_1_3_5_independently():
    # Regression/design-refinement test: AdaptiveKConfig() with NO
    # explicit overrides -- the actual class-level defaults a caller
    # gets by just writing `AdaptiveKConfig()`. min_k=1, default_k=3,
    # max_k=5 must produce a genuine three-level policy (1 / 3 / 5),
    # not a collapsed two-level one (the pre-refinement default_k=1
    # made high and medium indistinguishable: both selected k=1).
    true_default = AdaptiveKConfig()
    assert (true_default.min_k, true_default.default_k, true_default.max_k) == (1, 3, 5)

    high = select_adaptive_k(["a", "b", "c", "d", "e", "f"], [10.0, 1.0, 0.9, 0.8, 0.7, 0.6], true_default)
    # margin=0.5, spread=0.9 -> confidence=0.5556, in [0.33, 0.66) -> medium.
    medium = select_adaptive_k(["a", "b", "c", "d", "e", "f"], [5.0, 4.5, 4.4, 4.3, 4.2, 4.1], true_default)
    low = select_adaptive_k(["a", "b", "c", "d", "e", "f"], [5.0] * 6, true_default)

    assert high.confidence_band == BAND_HIGH
    assert medium.confidence_band == BAND_MEDIUM
    assert low.confidence_band == BAND_LOW

    assert high.selected_k == 1
    assert medium.selected_k == 3
    assert low.selected_k == 5

    # All three distinct -- the exact "collapsed two-level policy" bug
    # this refinement fixes.
    assert len({high.selected_k, medium.selected_k, low.selected_k}) == 3


def test_default_config_thresholds_unchanged_by_this_refinement():
    # high_confidence_threshold/low_confidence_threshold must be exactly
    # preserved -- only default_k changed.
    config = AdaptiveKConfig()
    assert config.high_confidence_threshold == 0.66
    assert config.low_confidence_threshold == 0.33
    assert config.min_k == 1
    assert config.max_k == 5


def test_configurability_preserved_explicit_default_k_still_honored():
    # Requirement 4: users may still explicitly set another valid
    # default_k -- the new class default must not be hardcoded anywhere
    # else in the selection logic.
    custom = AdaptiveKConfig(min_k=1, max_k=5, default_k=2)
    decision = select_adaptive_k(["a", "b", "c"], [5.0, 4.9, 4.8], custom)
    assert decision.confidence_band == BAND_MEDIUM
    assert decision.selected_k == 2


def test_high_confidence_never_selects_more_than_low_confidence_same_config():
    high = select_adaptive_k(["a", "b", "c", "d", "e", "f"], [10.0, 1.0, 0.9, 0.8, 0.7, 0.6], DEFAULT_CONFIG)
    low = select_adaptive_k(["a", "b", "c", "d", "e", "f"], [5.0] * 6, DEFAULT_CONFIG)
    assert high.confidence_band == BAND_HIGH
    assert low.confidence_band == BAND_LOW
    assert high.selected_k <= low.selected_k


# ---------------------------------------------------------------------------
# Section D edge cases
# ---------------------------------------------------------------------------


def test_empty_candidate_list():
    decision = select_adaptive_k([], [], DEFAULT_CONFIG)
    assert decision.available_candidate_count == 0
    assert decision.selected_k == 0
    assert decision.selected_candidate_count == 0
    assert decision.selected_candidates == ()


def test_one_candidate():
    decision = select_adaptive_k(["a"], [3.0], DEFAULT_CONFIG)
    assert decision.available_candidate_count == 1
    assert decision.selected_k == 1
    assert decision.selected_candidates == ("a",)


def test_fewer_candidates_than_selected_k_never_fabricates():
    # Low confidence would want max_k=5, but only 2 candidates exist.
    decision = select_adaptive_k(["a", "b"], [5.0, 5.0], DEFAULT_CONFIG)
    assert decision.confidence_band == BAND_LOW
    assert decision.available_candidate_count == 2
    assert decision.selected_k == 2
    assert decision.selected_candidate_count == 2
    assert len(decision.selected_candidates) == 2
    assert "capped to" in decision.reason


def test_tied_scores():
    decision = select_adaptive_k(["a", "b", "c"], [7.0, 7.0, 7.0], DEFAULT_CONFIG)
    assert decision.confidence == 0.0
    assert decision.confidence_band == BAND_LOW


def test_identical_score_range_two_candidates():
    decision = select_adaptive_k(["a", "b"], [1.0, 1.0], DEFAULT_CONFIG)
    assert decision.confidence == 0.0
    assert decision.score_spread == 0.0


def test_negative_scores():
    decision = select_adaptive_k(["a", "b", "c"], [-1.0, -5.0, -9.0], DEFAULT_CONFIG)
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.selected_k >= DEFAULT_CONFIG.min_k


def test_unsorted_input_is_never_reordered():
    # Deliberately NOT descending -- the function must not "fix" this;
    # it must select a prefix of the given order, unchanged.
    candidates = ["a", "b", "c", "d"]
    scores = [1.0, 9.0, 3.0, 0.5]  # unsorted on purpose
    decision = select_adaptive_k(candidates, scores, DEFAULT_CONFIG)
    assert decision.selected_candidates == tuple(candidates[: decision.selected_k])
    # Confirms top1/top2 were read positionally (index 0/1), not by
    # re-sorting the scores first.
    assert decision.score_margin == pytest.approx(1.0 - 9.0)


def test_nan_score_rejected():
    with pytest.raises(ValueError):
        select_adaptive_k(["a", "b"], [float("nan"), 1.0], DEFAULT_CONFIG)


def test_infinite_score_rejected():
    with pytest.raises(ValueError):
        select_adaptive_k(["a", "b"], [float("inf"), 1.0], DEFAULT_CONFIG)
    with pytest.raises(ValueError):
        select_adaptive_k(["a", "b"], [float("-inf"), 1.0], DEFAULT_CONFIG)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        select_adaptive_k(["a", "b", "c"], [1.0, 2.0], DEFAULT_CONFIG)


def test_min_k_equals_max_k_always_returns_that_k_when_available():
    fixed_config = AdaptiveKConfig(min_k=2, max_k=2, default_k=2)
    high = select_adaptive_k(["a", "b", "c"], [10.0, 1.0, 0.5], fixed_config)
    low = select_adaptive_k(["a", "b", "c"], [5.0, 5.0, 5.0], fixed_config)
    assert high.selected_k == 2
    assert low.selected_k == 2


def test_deterministic_repeated_execution():
    candidates = ["a", "b", "c", "d", "e"]
    scores = [5.0, 4.5, 3.0, 2.9, 1.0]
    first = select_adaptive_k(candidates, scores, DEFAULT_CONFIG)
    second = select_adaptive_k(candidates, scores, DEFAULT_CONFIG)
    assert first == second


def test_selected_k_never_exceeds_max_k():
    for scores in ([5.0] * 10, [10.0, 1.0] + [0.5] * 8, list(range(10))):
        decision = select_adaptive_k([f"c{i}" for i in range(10)], [float(s) for s in scores], DEFAULT_CONFIG)
        assert decision.selected_k <= DEFAULT_CONFIG.max_k
