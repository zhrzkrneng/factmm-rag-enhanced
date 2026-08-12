"""Unit tests for I3's configuration contract (FusionConfig) in
src/innovation/multi_report_fusion/fusion.py -- Cell 50, this revision.

Covers spec category (1): config defaults. The pre-existing
confidence_threshold/disagreement_threshold validation (Cell 47) is
already covered by tests/unit/test_innovation_config.py and is not
duplicated here -- this file focuses on the new
high_confidence_threshold/low_confidence_threshold fields Cell 50
added.
"""

import pytest

from src.innovation.experiment_contract import InnovationConfig
from src.innovation.multi_report_fusion.fusion import FusionConfig


def test_default_fusion_config_values():
    config = FusionConfig()
    assert config.max_evidence_count == 3
    assert config.context_budget_chars == 4000
    assert config.confidence_threshold == 0.5
    assert config.disagreement_threshold == 0.3
    assert config.high_confidence_threshold == 0.66
    assert config.low_confidence_threshold == 0.33


def test_default_fusion_config_is_what_innovation_config_uses_by_default():
    # InnovationConfig() with no args is E0 -- its nested `fusion` field
    # must be exactly FusionConfig()'s own defaults, matching I1/I2's
    # identical guarantee for adaptive_k/reranking.
    assert InnovationConfig().fusion == FusionConfig()


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_high_confidence_threshold_out_of_range_rejected(bad_value):
    with pytest.raises(ValueError):
        FusionConfig(high_confidence_threshold=bad_value)


@pytest.mark.parametrize("bad_value", [-0.1, 1.1])
def test_low_confidence_threshold_out_of_range_rejected(bad_value):
    with pytest.raises(ValueError):
        FusionConfig(low_confidence_threshold=bad_value)


def test_low_threshold_equal_to_high_threshold_rejected():
    with pytest.raises(ValueError):
        FusionConfig(high_confidence_threshold=0.5, low_confidence_threshold=0.5)


def test_low_threshold_greater_than_high_threshold_rejected():
    with pytest.raises(ValueError):
        FusionConfig(high_confidence_threshold=0.3, low_confidence_threshold=0.6)


def test_valid_custom_thresholds_accepted():
    config = FusionConfig(high_confidence_threshold=0.8, low_confidence_threshold=0.2)
    assert config.high_confidence_threshold == 0.8
    assert config.low_confidence_threshold == 0.2


def test_existing_budget_and_legacy_threshold_validation_unchanged():
    # Cell 47's original validation must still work unmodified after
    # Cell 50's field additions.
    with pytest.raises(ValueError):
        FusionConfig(max_evidence_count=0)
    with pytest.raises(ValueError):
        FusionConfig(context_budget_chars=0)
    with pytest.raises(ValueError):
        FusionConfig(confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        FusionConfig(disagreement_threshold=1.1)
