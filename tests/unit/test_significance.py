"""Unit tests for src/evaluation/significance.py (Milestone 2.6, Cell 35).

Scope: SignificanceConfig, SignificanceResult, paired_bootstrap_significance
-- determinism (including the "local generator, never global RNG state"
guarantee), confidence interval / p-value sanity for both identical and
clearly-different paired systems, and input validation."""

import dataclasses

import numpy as np
import pytest

from src.common.exceptions import EvaluationError
from src.evaluation.significance import (
    SignificanceConfig,
    SignificanceResult,
    paired_bootstrap_significance,
)


# ---------------------------------------------------------------------------
# SignificanceConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"method": "not_a_real_method"},
        {"num_bootstrap_samples": 0},
        {"num_bootstrap_samples": -1},
        {"num_bootstrap_samples": 1.5},
        {"num_bootstrap_samples": True},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
        {"confidence_level": 1.5},
        {"confidence_level": -0.1},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        SignificanceConfig(**overrides)


def test_config_defaults_are_valid():
    config = SignificanceConfig()
    assert config.method == "paired_bootstrap"
    assert config.num_bootstrap_samples == 10000
    assert config.confidence_level == 0.95
    assert config.seed == 42


def test_config_is_immutable():
    config = SignificanceConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7


def test_significance_result_is_immutable():
    result = SignificanceResult(
        metric_name="m", system_a_name="A", system_b_name="B",
        system_a_mean=1.0, system_b_mean=0.5, mean_difference=0.5,
        ci_low=0.1, ci_high=0.9, p_value=0.01, num_bootstrap_samples=100, seed=42,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.p_value = 0.5


# ---------------------------------------------------------------------------
# paired_bootstrap_significance -- input validation
# ---------------------------------------------------------------------------


def test_rejects_length_mismatch():
    with pytest.raises(EvaluationError, match="same length"):
        paired_bootstrap_significance(
            [1.0, 2.0], [1.0],
            metric_name="m", system_a_name="A", system_b_name="B",
            config=SignificanceConfig(),
        )


def test_rejects_empty_scores():
    with pytest.raises(EvaluationError, match="non-empty"):
        paired_bootstrap_significance(
            [], [],
            metric_name="m", system_a_name="A", system_b_name="B",
            config=SignificanceConfig(),
        )


# ---------------------------------------------------------------------------
# Correctness -- identical systems
# ---------------------------------------------------------------------------


def test_identical_systems_zero_mean_difference_and_degenerate_ci():
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    result = paired_bootstrap_significance(
        scores, scores,
        metric_name="rouge_l", system_a_name="A", system_b_name="A_copy",
        config=SignificanceConfig(num_bootstrap_samples=2000),
    )
    assert result.mean_difference == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.system_a_mean == pytest.approx(result.system_b_mean)


# ---------------------------------------------------------------------------
# Correctness -- clearly different systems
# ---------------------------------------------------------------------------


def test_clearly_different_systems_ci_excludes_zero_and_small_p_value():
    # A consistently and substantially better than B, low variance
    system_a = [0.9, 0.91, 0.89, 0.92, 0.90, 0.88, 0.93, 0.91, 0.90, 0.89]
    system_b = [0.4, 0.41, 0.39, 0.42, 0.40, 0.38, 0.43, 0.41, 0.40, 0.39]
    result = paired_bootstrap_significance(
        system_a, system_b,
        metric_name="rouge_l", system_a_name="FactMM-RAG", system_b_name="Med-MARVEL",
        config=SignificanceConfig(num_bootstrap_samples=5000),
    )
    assert result.mean_difference > 0
    assert result.ci_low > 0  # CI entirely excludes zero
    assert result.p_value < 0.05
    assert result.system_a_mean > result.system_b_mean


def test_result_fields_are_correctly_populated():
    config = SignificanceConfig(num_bootstrap_samples=500, seed=7, confidence_level=0.9)
    result = paired_bootstrap_significance(
        [1.0, 2.0, 3.0], [0.5, 1.5, 2.5],
        metric_name="bert_score", system_a_name="Sys A", system_b_name="Sys B",
        config=config,
    )
    assert result.metric_name == "bert_score"
    assert result.system_a_name == "Sys A"
    assert result.system_b_name == "Sys B"
    assert result.num_bootstrap_samples == 500
    assert result.seed == 7
    assert result.system_a_mean == pytest.approx(2.0)
    assert result.system_b_mean == pytest.approx(1.5)
    assert result.mean_difference == pytest.approx(0.5)
    assert result.ci_low <= result.ci_high


def test_wider_confidence_level_produces_wider_or_equal_interval():
    system_a = [0.9, 0.5, 0.7, 0.6, 0.8, 0.4, 0.95, 0.3, 0.85, 0.55]
    system_b = [0.4, 0.6, 0.3, 0.5, 0.2, 0.7, 0.35, 0.65, 0.25, 0.45]
    narrow = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B",
        config=SignificanceConfig(num_bootstrap_samples=5000, confidence_level=0.80, seed=1),
    )
    wide = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B",
        config=SignificanceConfig(num_bootstrap_samples=5000, confidence_level=0.99, seed=1),
    )
    assert (wide.ci_high - wide.ci_low) >= (narrow.ci_high - narrow.ci_low)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_20_repeated_calls():
    # Matches this project's own established verification discipline
    # (>=20 repetitions) from fixing test_retrieval_trainer.py's real
    # seeding-order flakiness bug earlier in this project.
    system_a = [0.8, 0.6, 0.9, 0.7, 0.5]
    system_b = [0.4, 0.5, 0.3, 0.6, 0.2]
    config = SignificanceConfig(num_bootstrap_samples=1000, seed=42)
    results = [
        paired_bootstrap_significance(
            system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config
        )
        for _ in range(20)
    ]
    assert all(result == results[0] for result in results)


def test_determinism_is_independent_of_prior_global_numpy_rng_state():
    # The central claim this cell must prove: a LOCAL Generator is used,
    # never the global np.random state -- direct carry-over of the
    # seeding-order lesson from this project's own real flakiness bug.
    system_a = [0.8, 0.6, 0.9, 0.7, 0.5]
    system_b = [0.4, 0.5, 0.3, 0.6, 0.2]
    config = SignificanceConfig(num_bootstrap_samples=1000, seed=42)

    np.random.seed(1)
    result_after_seed_1 = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config
    )

    np.random.seed(999)
    np.random.random(500)  # consume a bunch of global state
    result_after_different_global_state = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config
    )

    assert result_after_seed_1 == result_after_different_global_state


def test_different_seeds_are_independently_reproducible():
    system_a = [0.8, 0.6, 0.9, 0.7, 0.5, 0.55, 0.65]
    system_b = [0.4, 0.5, 0.3, 0.6, 0.2, 0.25, 0.35]
    config_seed_1 = SignificanceConfig(num_bootstrap_samples=1000, seed=1)
    config_seed_2 = SignificanceConfig(num_bootstrap_samples=1000, seed=2)

    result_1a = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config_seed_1
    )
    result_1b = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config_seed_1
    )
    result_2 = paired_bootstrap_significance(
        system_a, system_b, metric_name="m", system_a_name="A", system_b_name="B", config=config_seed_2
    )

    assert result_1a == result_1b  # same seed -> identical
    # different seeds are not required to differ in every field (means
    # are seed-independent), but the CI bounds (which depend on the
    # resampled distribution) are extremely unlikely to match exactly
    assert (result_1a.ci_low, result_1a.ci_high) != (result_2.ci_low, result_2.ci_high)
