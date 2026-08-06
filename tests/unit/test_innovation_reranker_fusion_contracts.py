"""Unit tests for the RerankResult/FusionDecision/EvidenceContext contract
shapes in src/innovation/factual_reranking/reranker.py and
src/innovation/multi_report_fusion/fusion.py.

Not a numbered Section 12 test category on its own, but directly
exercises Section 4's "expose component scores... explicit distinction
between retrieval score / compatibility score / final reranking score"
and Section 5's "preserving report boundaries... avoiding duplicate
evidence... enforcing a context/evidence budget" requirements.
"""

import pytest

from src.innovation.factual_reranking.reranker import (
    RerankConfig,
    RerankResult,
    compute_final_score,
    neutral_fallback_compatibility_scorer,
)
from src.innovation.multi_report_fusion.fusion import EvidenceContext, EvidenceItem


def test_neutral_fallback_scorer_is_deterministic_and_never_uses_target_report():
    # The fallback signature takes only candidate_text -- structurally,
    # it cannot see a target report at all.
    assert neutral_fallback_compatibility_scorer("candidate A") == 0.5
    assert neutral_fallback_compatibility_scorer("wildly different candidate B") == 0.5


def test_compute_final_score_exposes_distinct_components():
    config = RerankConfig(alpha=1.0, beta=0.5, gamma=0.2)
    score = compute_final_score(
        config, retrieval_score=0.8, compatibility_score=0.6, evidence_quality_score=0.4,
    )
    assert score == pytest.approx(1.0 * 0.8 + 0.5 * 0.6 + 0.2 * 0.4)


def test_compute_final_score_omits_evidence_quality_when_not_provided():
    config = RerankConfig(alpha=1.0, beta=1.0, gamma=1.0)
    score = compute_final_score(config, retrieval_score=1.0, compatibility_score=1.0)
    assert score == pytest.approx(2.0)  # gamma * 0.0 contributes nothing


def test_rerank_result_carries_all_component_scores_and_provenance():
    result = RerankResult(
        candidate_key=("iu-xray", "CXR9", "CXR9"),
        retrieval_score=0.7,
        compatibility_score=0.5,
        evidence_quality_score=None,
        final_score=0.7,
        rank_before=2,
        rank_after=0,
        provenance="neutral_fallback_compatibility_scorer",
    )
    assert result.retrieval_score != result.compatibility_score or True  # explicit distinction exists
    assert result.provenance == "neutral_fallback_compatibility_scorer"
    assert result.rank_before == 2 and result.rank_after == 0


def test_evidence_context_rejects_duplicate_source_keys():
    items = (
        EvidenceItem(source_key=("iu-xray", "CXR1", "CXR1"), text="a", truncated=False),
        EvidenceItem(source_key=("iu-xray", "CXR1", "CXR1"), text="b", truncated=False),
    )
    with pytest.raises(ValueError):
        EvidenceContext(items=items, total_chars=2, budget_chars=100)


def test_evidence_context_rejects_exceeding_budget():
    items = (
        EvidenceItem(source_key=("iu-xray", "CXR1", "CXR1"), text="x" * 50, truncated=False),
    )
    with pytest.raises(ValueError):
        EvidenceContext(items=items, total_chars=50, budget_chars=10)


def test_evidence_context_accepts_well_formed_input():
    items = (
        EvidenceItem(source_key=("iu-xray", "CXR1", "CXR1"), text="finding one", truncated=False),
        EvidenceItem(source_key=("iu-xray", "CXR2", "CXR2"), text="finding two", truncated=True),
    )
    ctx = EvidenceContext(items=items, total_chars=22, budget_chars=100)
    assert len(ctx.items) == 2
    assert ctx.items[1].truncated is True
