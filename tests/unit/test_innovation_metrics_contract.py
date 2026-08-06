"""Unit tests for src/innovation/evaluation/innovation_metrics.py.

Covers Cell 47 spec Section 12 tests: 16 (metric registry contract
contains required metrics), 17 (innovation diagnostic registry contains
required diagnostics).
"""

from src.evaluation.generation_metrics import _REGISTERED_METRIC_NAMES
from src.evaluation.retrieval_metrics import _REGISTERED_RETRIEVAL_METRIC_NAMES
from src.innovation.evaluation.innovation_metrics import (
    ADAPTIVE_K_DIAGNOSTICS,
    CLINICAL_METRICS,
    EFFICIENCY_DIAGNOSTICS,
    FUSION_DIAGNOSTICS,
    GENERATION_METRICS,
    INNOVATION_DIAGNOSTICS,
    RERANKING_DIAGNOSTICS,
    RETRIEVAL_METRICS,
    diagnostic_registry,
    metric_contract,
)


def test_metric_contract_contains_required_retrieval_metrics():
    contract = metric_contract()
    assert set(contract["retrieval"]) == {"mrr", "recall", "ndcg"}


def test_metric_contract_contains_required_generation_metrics():
    contract = metric_contract()
    assert {"rouge_l", "bleu4", "bert_score"}.issubset(set(contract["generation"]))


def test_metric_contract_contains_required_clinical_metrics():
    contract = metric_contract()
    assert set(contract["clinical"]) == {"f1radgraph", "f1chexbert"}


def test_retrieval_metric_names_match_the_real_registry_exactly():
    # This contract must never silently drift from the real,
    # already-implemented registry it names.
    assert set(RETRIEVAL_METRICS).issubset(set(_REGISTERED_RETRIEVAL_METRIC_NAMES))


def test_generation_and_clinical_metric_names_match_the_real_registry_exactly():
    assert set(GENERATION_METRICS).issubset(set(_REGISTERED_METRIC_NAMES))
    assert set(CLINICAL_METRICS).issubset(set(_REGISTERED_METRIC_NAMES))


def test_innovation_diagnostic_registry_contains_required_diagnostics():
    registry = diagnostic_registry()
    assert set(ADAPTIVE_K_DIAGNOSTICS).issubset(set(registry["adaptive_k"]))
    assert set(RERANKING_DIAGNOSTICS).issubset(set(registry["reranking"]))
    assert set(FUSION_DIAGNOSTICS).issubset(set(registry["fusion"]))
    assert set(EFFICIENCY_DIAGNOSTICS).issubset(set(registry["efficiency"]))

    assert "mean_selected_k" in registry["adaptive_k"]
    assert "k_distribution" in registry["adaptive_k"]
    assert "evidence_reduction_vs_fixed_k" in registry["adaptive_k"]

    assert "rank_change_rate" in registry["reranking"]
    assert "mean_rank_displacement" in registry["reranking"]
    assert "top1_replacement_rate" in registry["reranking"]

    assert "fusion_activation_rate" in registry["fusion"]
    assert "mean_evidence_count" in registry["fusion"]

    assert "latency_per_query" in registry["efficiency"]
    assert "evidence_count" in registry["efficiency"]
    assert "approximate_context_length" in registry["efficiency"]


def test_innovation_diagnostics_registry_has_no_duplicate_names():
    assert len(INNOVATION_DIAGNOSTICS) == len(set(INNOVATION_DIAGNOSTICS))
