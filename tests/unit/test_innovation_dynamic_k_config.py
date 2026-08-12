"""Unit tests for AdaptiveKConfig's Cell 48 extensions (confidence-band
thresholds) in src/innovation/adaptive_retrieval/adaptive_k.py.

Does not weaken or delete any Cell 47 test -- see
tests/unit/test_innovation_config.py, still passing unmodified.
"""

import pytest

from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig


def test_default_config_has_ordered_bounds_and_non_overlapping_bands():
    config = AdaptiveKConfig()
    assert config.min_k <= config.default_k <= config.max_k
    assert config.low_confidence_threshold < config.high_confidence_threshold


def test_cell47_confidence_threshold_field_still_works_unmodified():
    # Regression: Cell 47's own tests construct AdaptiveKConfig with
    # only `confidence_threshold` set -- must still validate/construct
    # correctly after Cell 48's extension.
    config = AdaptiveKConfig(confidence_threshold=0.7)
    assert config.confidence_threshold == 0.7
    with pytest.raises(ValueError):
        AdaptiveKConfig(confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        AdaptiveKConfig(confidence_threshold=1.1)


def test_overlapping_confidence_bands_rejected():
    with pytest.raises(ValueError):
        AdaptiveKConfig(low_confidence_threshold=0.5, high_confidence_threshold=0.5)
    with pytest.raises(ValueError):
        AdaptiveKConfig(low_confidence_threshold=0.7, high_confidence_threshold=0.3)


def test_out_of_range_band_thresholds_rejected():
    with pytest.raises(ValueError):
        AdaptiveKConfig(high_confidence_threshold=1.5)
    with pytest.raises(ValueError):
        AdaptiveKConfig(low_confidence_threshold=-0.1)


def test_well_formed_band_thresholds_accepted():
    config = AdaptiveKConfig(low_confidence_threshold=0.2, high_confidence_threshold=0.8)
    assert config.low_confidence_threshold == 0.2
    assert config.high_confidence_threshold == 0.8


def test_min_k_equals_max_k_is_valid():
    config = AdaptiveKConfig(min_k=3, max_k=3, default_k=3)
    assert config.min_k == config.default_k == config.max_k == 3
