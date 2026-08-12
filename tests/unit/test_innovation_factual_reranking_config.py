"""Unit tests for I2's configuration defaults (RerankConfig) and its
integration into InnovationConfig -- Cell 47's contract-level tests
(test_innovation_reranker_fusion_contracts.py) already cover
RerankConfig's own validation; this file focuses on the specific
defaults required for this revision's baseline-equivalence guarantee.
"""

import pytest

from src.innovation.experiment_contract import InnovationConfig
from src.innovation.factual_reranking.reranker import RerankConfig


def test_rerank_config_defaults_are_baseline_equivalent():
    config = RerankConfig()
    assert config.alpha == 1.0
    assert config.beta == 0.0
    assert config.gamma == 0.0


def test_innovation_config_default_reranking_matches_rerank_config_default():
    innovation_config = InnovationConfig()
    assert innovation_config.reranking == RerankConfig()
    assert innovation_config.factual_reranking_enabled is False


def test_rerank_config_allows_enabling_beta_and_gamma():
    config = RerankConfig(alpha=1.0, beta=0.5, gamma=0.25)
    assert config.beta == 0.5
    assert config.gamma == 0.25


def test_rerank_config_rejects_all_zero_weights():
    with pytest.raises(ValueError):
        RerankConfig(alpha=0.0, beta=0.0, gamma=0.0)


def test_rerank_config_rejects_negative_weight():
    with pytest.raises(ValueError):
        RerankConfig(alpha=1.0, beta=-0.1, gamma=0.0)
