"""Unit tests for src/innovation/adaptive_retrieval/pipeline_adapter.py.

Covers Cell 48 spec Section F: E0 (I1 disabled) must remain
baseline-equivalent, proven directly against the REAL, unmodified
src.baseline.generation.dataset_builder.RAGDatasetBuilder (not a
reimplementation asserted to be "equivalent" by inspection); E1 must
alter only evidence-depth selection.
"""

from src.baseline.generation.dataset_builder import RAGDatasetBuilder, RAGDatasetBuilderConfig
from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig
from src.innovation.adaptive_retrieval.pipeline_adapter import (
    select_baseline_equivalent_evidence,
    select_dynamic_k_evidence,
    select_evidence_for_query,
)
from src.innovation.experiment_contract import InnovationConfig


def _flat_record(study_id):
    return {
        "dataset": "iu-xray", "patient_id": study_id, "study_id": study_id,
        "finding": f"finding {study_id}", "image_path": f"/fake/{study_id}.png",
    }


def _keys(n):
    return [("iu-xray", f"CXR{i}", f"CXR{i}") for i in range(n)]


def test_e0_matches_real_rag_dataset_builder_exactly():
    keys = _keys(6)
    query_key = keys[0]
    corpus = {k: _flat_record(k[1]) for k in keys}
    ranking = keys  # includes self-match at rank 0, deliberately, to test filtering
    scores = [10.0 - i for i in range(len(ranking))]

    real_builder = RAGDatasetBuilder(
        corpus, corpus, {query_key: ranking}, config=RAGDatasetBuilderConfig(),
        query_split_name="train", corpus_split_name="train",
    )
    real_row = real_builder.build_row(query_key)

    adapter_result = select_baseline_equivalent_evidence(query_key, ranking, scores)

    assert adapter_result.adaptive_k_used is False
    assert adapter_result.selected_candidates == (real_row.retrieved_key,)


def test_e0_matches_real_builder_across_multiple_synthetic_queries():
    keys = _keys(10)
    corpus = {k: _flat_record(k[1]) for k in keys}
    rankings = {}
    for i, qk in enumerate(keys):
        # Each query's own ranking: itself first (self-match, must be
        # filtered), then the rest in a rotated order.
        rankings[qk] = [qk] + [k for k in keys if k != qk]

    real_builder = RAGDatasetBuilder(
        corpus, corpus, rankings, config=RAGDatasetBuilderConfig(),
        query_split_name="train", corpus_split_name="train",
    )

    for qk in keys:
        real_row = real_builder.build_row(qk)
        scores = [10.0 - i for i in range(len(rankings[qk]))]
        adapter_result = select_baseline_equivalent_evidence(qk, rankings[qk], scores)
        assert adapter_result.selected_candidates == (real_row.retrieved_key,)


def test_e1_selects_more_than_one_under_low_confidence():
    keys = _keys(6)
    query_key = keys[0]
    ranking = keys[1:]  # no self-match needed for this check
    scores = [5.0] * len(ranking)  # tied -> low confidence
    config = AdaptiveKConfig(min_k=1, max_k=4, default_k=2)

    result = select_dynamic_k_evidence(query_key, ranking, scores, config)
    assert result.adaptive_k_used is True
    assert result.decision.confidence_band == "low"
    assert len(result.selected_candidates) == 4


def test_e1_never_includes_self_match():
    keys = _keys(4)
    query_key = keys[0]
    ranking = keys  # self-match included at rank 0
    scores = [10.0, 9.0, 8.0, 7.0]
    config = AdaptiveKConfig(min_k=1, max_k=4, default_k=2)

    result = select_dynamic_k_evidence(query_key, ranking, scores, config)
    assert query_key not in result.selected_candidates


def test_select_evidence_for_query_branches_on_innovation_config():
    keys = _keys(6)
    query_key = keys[0]
    ranking = keys[1:]
    scores = [5.0] * len(ranking)

    e0_result = select_evidence_for_query(query_key, ranking, scores, InnovationConfig())
    e1_result = select_evidence_for_query(
        query_key, ranking, scores, InnovationConfig(adaptive_retrieval_enabled=True),
    )
    assert e0_result.adaptive_k_used is False
    assert len(e0_result.selected_candidates) == 1
    assert e1_result.adaptive_k_used is True
    assert len(e1_result.selected_candidates) > 1


def test_e1_preserves_candidate_ordering_never_reorders():
    keys = _keys(6)
    query_key = keys[0]
    ranking = keys[1:]
    scores = [1.0, 9.0, 3.0, 0.5, 2.0]  # deliberately unsorted
    config = AdaptiveKConfig(min_k=1, max_k=5, default_k=3)

    result = select_dynamic_k_evidence(query_key, ranking, scores, config)
    # Selected candidates must be a prefix of the (unsorted) input ranking.
    assert result.selected_candidates == tuple(ranking[: len(result.selected_candidates)])


def test_e1_does_not_alter_scores_or_query_construction():
    # The adapter must be a pure function over (query_key, ranking,
    # scores, config) -- calling it twice with identical inputs must
    # not mutate anything observable.
    keys = _keys(4)
    query_key = keys[0]
    ranking = tuple(keys[1:])
    scores = (5.0, 4.0, 3.0)
    config = AdaptiveKConfig(min_k=1, max_k=3, default_k=2)

    r1 = select_dynamic_k_evidence(query_key, ranking, scores, config)
    r2 = select_dynamic_k_evidence(query_key, ranking, scores, config)
    assert r1.selected_candidates == r2.selected_candidates
    assert r1.decision == r2.decision
    assert ranking == tuple(keys[1:])  # unchanged
    assert scores == (5.0, 4.0, 3.0)  # unchanged
