"""Unit tests for the real I2 algorithm (reranker.py's
compute_mesh_term_consensus_score / compute_evidence_quality_score /
rerank_candidates) -- Cell 48, this revision.

Every test uses only candidates' OWN mesh_terms/findings/impression --
no test constructs or passes anything resembling a query's target
report, matching the "no target leakage" requirement (see
test_no_target_report_field_anywhere_in_the_api below for an explicit
API-shape check of that guarantee).
"""

import inspect

from src.innovation.factual_reranking.reranker import (
    RerankCandidateInput,
    RerankConfig,
    compute_evidence_quality_score,
    compute_mesh_term_consensus_score,
    rerank_candidates,
)


# ---------------------------------------------------------------------------
# compute_mesh_term_consensus_score
# ---------------------------------------------------------------------------


def test_jaccard_exact_value_partial_overlap():
    # candidate={a,b,c}; others union={b,c,d} -> intersection={b,c} (2),
    # union={a,b,c,d} (4) -> 2/4 = 0.5
    score, basis = compute_mesh_term_consensus_score(["a", "b", "c"], [["b", "c", "d"]])
    assert score == 0.5
    assert basis == "jaccard"


def test_jaccard_exact_value_full_overlap():
    score, basis = compute_mesh_term_consensus_score(["a", "b"], [["a", "b"]])
    assert score == 1.0
    assert basis == "jaccard"


def test_jaccard_exact_value_no_overlap_both_nonempty():
    # intersection=0, union=4 -> 0.0, but both sides nonempty so basis is "jaccard"
    score, basis = compute_mesh_term_consensus_score(["a", "b"], [["c", "d"]])
    assert score == 0.0
    assert basis == "jaccard"


def test_jaccard_aggregates_union_across_multiple_other_candidates():
    # others = [{b}, {c}] -> union {b, c}; candidate={a,b} -> intersection={b} (1), union={a,b,c} (3)
    score, basis = compute_mesh_term_consensus_score(["a", "b"], [["b"], ["c"]])
    assert score == 1 / 3
    assert basis == "jaccard"


def test_both_empty_returns_neutral_half():
    score, basis = compute_mesh_term_consensus_score([], [[]])
    assert score == 0.5
    assert basis == "both_empty"


def test_both_empty_multiple_others_all_empty():
    score, basis = compute_mesh_term_consensus_score([], [[], [], []])
    assert score == 0.5
    assert basis == "both_empty"


def test_candidate_empty_consensus_nonempty_returns_zero():
    score, basis = compute_mesh_term_consensus_score([], [["a", "b"]])
    assert score == 0.0
    assert basis == "one_side_empty"


def test_candidate_nonempty_consensus_empty_returns_zero():
    score, basis = compute_mesh_term_consensus_score(["a", "b"], [[]])
    assert score == 0.0
    assert basis == "one_side_empty"


def test_k1_no_others_returns_neutral_half_regardless_of_own_terms():
    # No other candidates at all (e.g. I1 selected K=1) -- neutral,
    # even though this candidate itself has real mesh terms.
    score, basis = compute_mesh_term_consensus_score(["cardiomegaly", "effusion"], [])
    assert score == 0.5
    assert basis == "k1_no_others"


def test_k1_no_others_with_empty_own_terms_also_neutral():
    score, basis = compute_mesh_term_consensus_score([], [])
    assert score == 0.5
    assert basis == "k1_no_others"


def test_consensus_score_always_bounded_0_1():
    cases = [
        (["a"], [["a", "b", "c", "d", "e"]]),
        (["a", "b", "c", "d", "e"], [["a"]]),
        ([], []),
        ([], [["x"]]),
    ]
    for candidate_terms, others in cases:
        score, _ = compute_mesh_term_consensus_score(candidate_terms, others)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# compute_evidence_quality_score
# ---------------------------------------------------------------------------


def test_evidence_quality_both_present():
    assert compute_evidence_quality_score("Findings text.", "Impression text.") == 1.0


def test_evidence_quality_only_findings():
    assert compute_evidence_quality_score("Findings text.", None) == 0.5
    assert compute_evidence_quality_score("Findings text.", "") == 0.5
    assert compute_evidence_quality_score("Findings text.", "   ") == 0.5


def test_evidence_quality_only_impression():
    assert compute_evidence_quality_score(None, "Impression text.") == 0.5


def test_evidence_quality_neither_present():
    assert compute_evidence_quality_score(None, None) == 0.0
    assert compute_evidence_quality_score("", "") == 0.0
    assert compute_evidence_quality_score("   ", None) == 0.0


# ---------------------------------------------------------------------------
# rerank_candidates -- ordering, determinism, candidate-set preservation
# ---------------------------------------------------------------------------


def _candidates():
    return [
        RerankCandidateInput("a", 3.0, [], "F-a", "I-a"),
        RerankCandidateInput("b", 2.0, ["cardiomegaly", "effusion"], "F-b", "I-b"),
        RerankCandidateInput("c", 1.0, ["cardiomegaly", "effusion"], "F-c", "I-c"),
    ]


def test_beta_zero_preserves_retrieval_ordering():
    results = rerank_candidates(_candidates(), RerankConfig(alpha=1.0, beta=0.0, gamma=0.0))
    assert [r.candidate_key for r in results] == ["a", "b", "c"]
    assert all(r.rank_after == r.rank_before for r in results)


def test_reranking_changes_order_when_beta_positive():
    # b and c share mesh terms with each other (high consensus); a has
    # none -- with a large enough beta, b/c should outrank a despite a
    # having the highest raw retrieval_score.
    results = rerank_candidates(_candidates(), RerankConfig(alpha=1.0, beta=5.0, gamma=0.0))
    ordered_keys = [r.candidate_key for r in results]
    assert ordered_keys != ["a", "b", "c"]
    assert ordered_keys[0] in ("b", "c")


def test_candidate_set_preserved_exactly():
    inputs = _candidates()
    results = rerank_candidates(inputs, RerankConfig(alpha=1.0, beta=1.0))
    assert {r.candidate_key for r in results} == {c.candidate_key for c in inputs}
    assert len(results) == len(inputs)


def test_deterministic_repeated_execution():
    inputs = _candidates()
    config = RerankConfig(alpha=1.0, beta=2.0, gamma=1.0)
    first = rerank_candidates(inputs, config)
    second = rerank_candidates(inputs, config)
    assert first == second


def test_deterministic_tie_breaking_uses_original_rank():
    # Identical retrieval_score AND identical mesh_terms -> a true tie
    # in final_score; must break by rank_before (stable), never randomly.
    tied = [
        RerankCandidateInput("x", 5.0, ["a"], None, None),
        RerankCandidateInput("y", 5.0, ["a"], None, None),
        RerankCandidateInput("z", 5.0, ["a"], None, None),
    ]
    results = rerank_candidates(tied, RerankConfig(alpha=1.0, beta=1.0))
    assert [r.candidate_key for r in results] == ["x", "y", "z"]
    assert [r.rank_after for r in results] == [0, 1, 2]


def test_rerank_candidates_handles_empty_input():
    assert rerank_candidates([], RerankConfig()) == []


def test_rerank_candidates_handles_single_candidate():
    results = rerank_candidates([RerankCandidateInput("only", 4.0, ["x"], "F", "I")], RerankConfig(beta=1.0))
    assert len(results) == 1
    assert results[0].compatibility_score == 0.5  # k1_no_others
    assert results[0].rank_before == 0
    assert results[0].rank_after == 0


def test_no_target_report_field_anywhere_in_the_api():
    # Explicit API-shape guarantee: neither RerankCandidateInput nor
    # rerank_candidates' signature has any parameter/attribute whose
    # name suggests target-report access -- I2 physically cannot see
    # the query's target report through this interface.
    candidate_field_names = {f for f in RerankCandidateInput.__dataclass_fields__}
    forbidden_substrings = ("target", "ground_truth", "label_report", "query_report")
    for field_name in candidate_field_names:
        assert not any(s in field_name.lower() for s in forbidden_substrings), field_name

    signature_params = set(inspect.signature(rerank_candidates).parameters)
    for param_name in signature_params:
        assert not any(s in param_name.lower() for s in forbidden_substrings), param_name
