"""Unit tests for src/innovation/experiment_contract.py's InnovationConfig
and the per-innovation config dataclasses it composes.

Covers Cell 47 spec Section 12 tests: 1 (baseline configuration has all
innovations disabled), 2 (each innovation can be enabled independently),
9 (invalid k configuration rejected), 10 (invalid reranking weights
rejected), 11 (invalid confidence thresholds rejected), 12 (invalid
fusion budget rejected), 18 (configuration serialization deterministic).
"""

import pytest

from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig
from src.innovation.experiment_contract import InnovationConfig
from src.innovation.factual_reranking.reranker import RerankConfig
from src.innovation.multi_report_fusion.fusion import FusionConfig


def test_baseline_configuration_has_all_innovations_disabled():
    config = InnovationConfig()
    assert config.adaptive_retrieval_enabled is False
    assert config.factual_reranking_enabled is False
    assert config.confidence_gated_fusion_enabled is False


def test_each_innovation_can_be_enabled_independently():
    only_adaptive = InnovationConfig(adaptive_retrieval_enabled=True)
    assert only_adaptive.adaptive_retrieval_enabled is True
    assert only_adaptive.factual_reranking_enabled is False
    assert only_adaptive.confidence_gated_fusion_enabled is False

    only_reranking = InnovationConfig(factual_reranking_enabled=True)
    assert only_reranking.adaptive_retrieval_enabled is False
    assert only_reranking.factual_reranking_enabled is True
    assert only_reranking.confidence_gated_fusion_enabled is False

    only_fusion = InnovationConfig(confidence_gated_fusion_enabled=True)
    assert only_fusion.adaptive_retrieval_enabled is False
    assert only_fusion.factual_reranking_enabled is False
    assert only_fusion.confidence_gated_fusion_enabled is True


# ---------------------------------------------------------------------------
# Invalid k configuration rejected
# ---------------------------------------------------------------------------


def test_invalid_k_configuration_rejected():
    with pytest.raises(ValueError):
        AdaptiveKConfig(min_k=0)
    with pytest.raises(ValueError):
        AdaptiveKConfig(min_k=5, max_k=2)
    with pytest.raises(ValueError):
        AdaptiveKConfig(min_k=2, max_k=5, default_k=10)
    with pytest.raises(ValueError):
        AdaptiveKConfig(min_k=2, max_k=5, default_k=1)


def test_valid_k_configuration_accepted():
    config = AdaptiveKConfig(min_k=1, max_k=5, default_k=3)
    assert config.min_k == 1
    assert config.max_k == 5
    assert config.default_k == 3


# ---------------------------------------------------------------------------
# Invalid reranking weights rejected
# ---------------------------------------------------------------------------


def test_invalid_reranking_weights_rejected():
    with pytest.raises(ValueError):
        RerankConfig(alpha=-1.0)
    with pytest.raises(ValueError):
        RerankConfig(beta=-0.5)
    with pytest.raises(ValueError):
        RerankConfig(gamma=-0.1)
    with pytest.raises(ValueError):
        RerankConfig(alpha=0.0, beta=0.0, gamma=0.0)


def test_valid_reranking_weights_accepted():
    config = RerankConfig(alpha=0.5, beta=0.3, gamma=0.2)
    assert config.alpha == 0.5


# ---------------------------------------------------------------------------
# Invalid confidence thresholds rejected
# ---------------------------------------------------------------------------


def test_invalid_confidence_thresholds_rejected():
    with pytest.raises(ValueError):
        AdaptiveKConfig(confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        AdaptiveKConfig(confidence_threshold=1.1)
    with pytest.raises(ValueError):
        FusionConfig(confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        FusionConfig(confidence_threshold=1.1)
    with pytest.raises(ValueError):
        FusionConfig(disagreement_threshold=-0.1)
    with pytest.raises(ValueError):
        FusionConfig(disagreement_threshold=1.1)


# ---------------------------------------------------------------------------
# Invalid fusion budget rejected
# ---------------------------------------------------------------------------


def test_invalid_fusion_budget_rejected():
    with pytest.raises(ValueError):
        FusionConfig(max_evidence_count=0)
    with pytest.raises(ValueError):
        FusionConfig(context_budget_chars=0)
    with pytest.raises(ValueError):
        FusionConfig(max_evidence_count=-1)
    with pytest.raises(ValueError):
        FusionConfig(context_budget_chars=-100)


def test_valid_fusion_budget_accepted():
    config = FusionConfig(max_evidence_count=5, context_budget_chars=8000)
    assert config.max_evidence_count == 5
    assert config.context_budget_chars == 8000


# ---------------------------------------------------------------------------
# Configuration serialization deterministic
# ---------------------------------------------------------------------------


def test_configuration_serialization_deterministic():
    config = InnovationConfig(adaptive_retrieval_enabled=True, factual_reranking_enabled=True)
    json_1 = config.to_json()
    json_2 = config.to_json()
    assert json_1 == json_2

    dict_1 = config.to_json_dict()
    dict_2 = config.to_json_dict()
    assert dict_1 == dict_2


def test_configuration_serialization_round_trips_nested_configs():
    config = InnovationConfig(
        adaptive_k=AdaptiveKConfig(min_k=2, max_k=8, default_k=2),
        reranking=RerankConfig(alpha=0.7, beta=0.3),
        fusion=FusionConfig(max_evidence_count=4),
    )
    payload = config.to_json_dict()
    assert payload["adaptive_k"]["min_k"] == 2
    assert payload["adaptive_k"]["max_k"] == 8
    assert payload["reranking"]["alpha"] == 0.7
    assert payload["fusion"]["max_evidence_count"] == 4
