"""Unit tests for I2's diagnostics aggregation (diagnostics.py) --
Cell 48, this revision.
"""

from src.innovation.factual_reranking.diagnostics import aggregate_reranking_diagnostics
from src.innovation.factual_reranking.reranker import RerankCandidateInput, RerankConfig, rerank_candidates


def test_aggregate_diagnostics_empty_input_returns_neutral_zeros():
    diagnostics = aggregate_reranking_diagnostics([])
    assert diagnostics.decision_count == 0
    assert diagnostics.mean_compatibility_score == 0.0
    assert diagnostics.median_compatibility_score == 0.0
    assert diagnostics.mean_absolute_rank_shift == 0.0
    assert diagnostics.fraction_rank_changed == 0.0
    assert diagnostics.top1_changed_fraction == 0.0
    assert diagnostics.mean_evidence_quality_score == 0.0
    assert diagnostics.k1_neutral_count == 0
    assert diagnostics.k1_neutral_fraction == 0.0
    assert diagnostics.empty_mesh_neutral_or_zero_count == 0
    assert diagnostics.empty_mesh_neutral_or_zero_fraction == 0.0


def test_aggregate_diagnostics_empty_query_list_is_skipped_not_counted():
    diagnostics = aggregate_reranking_diagnostics([[]])
    assert diagnostics.decision_count == 0
    assert diagnostics.top1_changed_fraction == 0.0


def _query1_results(beta):
    candidates = [
        RerankCandidateInput("a", 3.0, [], "F", "I"),
        RerankCandidateInput("b", 2.0, ["cardiomegaly", "effusion"], "F", "I"),
        RerankCandidateInput("c", 1.0, ["cardiomegaly", "effusion"], "F", "I"),
    ]
    return rerank_candidates(candidates, RerankConfig(alpha=1.0, beta=beta))


def test_aggregate_diagnostics_counts_and_means_match_hand_computed_values():
    # beta=0 -> order unchanged, no rank shift, top-1 unchanged.
    results = _query1_results(beta=0.0)
    diagnostics = aggregate_reranking_diagnostics([results])

    assert diagnostics.decision_count == 3
    expected_mean_compat = sum(r.compatibility_score for r in results) / 3
    assert diagnostics.mean_compatibility_score == expected_mean_compat
    assert diagnostics.mean_absolute_rank_shift == 0.0
    assert diagnostics.fraction_rank_changed == 0.0
    assert diagnostics.top1_changed_fraction == 0.0
    assert diagnostics.mean_evidence_quality_score == 1.0  # all three have both findings+impression


def test_aggregate_diagnostics_detects_rank_shift_and_top1_change():
    # Large beta -> b/c (mesh overlap) outrank a -> top-1 changes from
    # "a" to one of b/c, and rank shift is nonzero.
    results = _query1_results(beta=5.0)
    diagnostics = aggregate_reranking_diagnostics([results])

    assert diagnostics.fraction_rank_changed > 0.0
    assert diagnostics.mean_absolute_rank_shift > 0.0
    assert diagnostics.top1_changed_fraction == 1.0  # the one query's top-1 did change


def test_aggregate_diagnostics_multiple_queries_averages_across_all():
    results_q1 = _query1_results(beta=0.0)  # top-1 unchanged
    results_q2 = _query1_results(beta=5.0)  # top-1 changed
    diagnostics = aggregate_reranking_diagnostics([results_q1, results_q2])

    assert diagnostics.decision_count == 6
    assert diagnostics.top1_changed_fraction == 0.5  # 1 of 2 queries changed


def test_aggregate_diagnostics_k1_neutral_counting():
    single = rerank_candidates(
        [RerankCandidateInput("only", 4.0, ["cardiomegaly"], "F", "I")], RerankConfig(beta=1.0)
    )
    diagnostics = aggregate_reranking_diagnostics([single])
    assert diagnostics.k1_neutral_count == 1
    assert diagnostics.k1_neutral_fraction == 1.0


def test_aggregate_diagnostics_empty_mesh_counting():
    candidates = [
        RerankCandidateInput("a", 3.0, [], None, None),
        RerankCandidateInput("b", 2.0, [], None, None),
    ]
    results = rerank_candidates(candidates, RerankConfig(beta=1.0))
    diagnostics = aggregate_reranking_diagnostics([results])
    # both candidates have empty mesh_terms and their only "other" also
    # has empty mesh_terms -> both_empty for both -> counted
    assert diagnostics.empty_mesh_neutral_or_zero_count == 2
    assert diagnostics.empty_mesh_neutral_or_zero_fraction == 1.0
    assert diagnostics.mean_evidence_quality_score == 0.0  # no findings/impression anywhere
