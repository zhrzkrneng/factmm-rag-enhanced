"""Factual/Clinical Reranking diagnostics aggregation (Cell 48, I2,
mirroring adaptive_retrieval/diagnostics.py's architecture).

Responsibility: a simple aggregation helper over batches of
RerankResult lists (one list per query), kept separate from the
reranking algorithm itself (reranker.py) per the same discipline
Section H established for I1: expose a post-hoc aggregation helper
rather than embedding evaluation logic into the algorithm. Computes
nothing new, never re-runs reranking, never touches candidate text
directly (only the already-computed component scores and provenance
labels).

No clinical-improvement claim is made or implied by any field here --
these are purely mechanical/structural statistics about how reranking
behaved (score distributions, how much order changed), never a
retrieval-quality or clinical-accuracy measurement.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from src.innovation.factual_reranking.reranker import RerankResult


@dataclass(frozen=True)
class FactualRerankingDiagnostics:
    """Aggregate diagnostics over one or more queries' RerankResult
    lists.

    Attributes:
        decision_count: total number of per-candidate RerankResult
            entries aggregated, across all queries (0 is allowed; every
            other field falls back to a well-defined neutral value).
        mean_compatibility_score: mean of compatibility_score across
            all candidates.
        median_compatibility_score: median of compatibility_score
            across all candidates.
        mean_absolute_rank_shift: mean of |rank_after - rank_before|
            across all candidates whose rank_after is set.
        fraction_rank_changed: fraction of candidates whose rank_after
            differs from rank_before.
        top1_changed_fraction: fraction of QUERIES (not candidates)
            whose top-ranked (rank 0) candidate changed after
            reranking.
        mean_evidence_quality_score: mean of evidence_quality_score
            across all candidates that have one set.
        k1_neutral_count / k1_neutral_fraction: how many/what fraction
            of candidates got the 0.5 "k1_no_others" neutral
            compatibility score (only one candidate in that query's
            set -- no consensus possible).
        empty_mesh_neutral_or_zero_count / _fraction: how many/what
            fraction of candidates hit the "both_empty" (0.5 neutral)
            or "one_side_empty" (0.0) MeSH-emptiness edge cases.
    """

    decision_count: int
    mean_compatibility_score: float
    median_compatibility_score: float
    mean_absolute_rank_shift: float
    fraction_rank_changed: float
    top1_changed_fraction: float
    mean_evidence_quality_score: float
    k1_neutral_count: int
    k1_neutral_fraction: float
    empty_mesh_neutral_or_zero_count: int
    empty_mesh_neutral_or_zero_fraction: float

    def to_json_dict(self) -> dict:
        return {
            "decision_count": self.decision_count,
            "mean_compatibility_score": self.mean_compatibility_score,
            "median_compatibility_score": self.median_compatibility_score,
            "mean_absolute_rank_shift": self.mean_absolute_rank_shift,
            "fraction_rank_changed": self.fraction_rank_changed,
            "top1_changed_fraction": self.top1_changed_fraction,
            "mean_evidence_quality_score": self.mean_evidence_quality_score,
            "k1_neutral_count": self.k1_neutral_count,
            "k1_neutral_fraction": self.k1_neutral_fraction,
            "empty_mesh_neutral_or_zero_count": self.empty_mesh_neutral_or_zero_count,
            "empty_mesh_neutral_or_zero_fraction": self.empty_mesh_neutral_or_zero_fraction,
        }


_EMPTY_DIAGNOSTICS_KWARGS = dict(
    decision_count=0,
    mean_compatibility_score=0.0,
    median_compatibility_score=0.0,
    mean_absolute_rank_shift=0.0,
    fraction_rank_changed=0.0,
    top1_changed_fraction=0.0,
    mean_evidence_quality_score=0.0,
    k1_neutral_count=0,
    k1_neutral_fraction=0.0,
    empty_mesh_neutral_or_zero_count=0,
    empty_mesh_neutral_or_zero_fraction=0.0,
)


def aggregate_reranking_diagnostics(
    results_per_query: Sequence[Sequence[RerankResult]],
) -> FactualRerankingDiagnostics:
    """Aggregates one or more queries' RerankResult lists into summary
    diagnostics. Never raises on an empty input (returns well-defined
    zero/neutral values instead).

    Args:
        results_per_query: one inner sequence of RerankResult per
            query -- i.e. one `rerank_candidates()` call's full output
            per query, not flattened by the caller.
    """
    all_results = [result for query_results in results_per_query for result in query_results]
    n = len(all_results)
    if n == 0:
        return FactualRerankingDiagnostics(**_EMPTY_DIAGNOSTICS_KWARGS)

    compatibility_scores = [r.compatibility_score for r in all_results]
    evidence_scores = [
        r.evidence_quality_score for r in all_results if r.evidence_quality_score is not None
    ]
    rank_shifts = [
        abs(r.rank_after - r.rank_before) for r in all_results if r.rank_after is not None
    ]
    changed = [
        r for r in all_results if r.rank_after is not None and r.rank_after != r.rank_before
    ]

    query_count = 0
    top1_changed_count = 0
    for query_results in results_per_query:
        if not query_results:
            continue
        query_count += 1
        before_top1 = next((r.candidate_key for r in query_results if r.rank_before == 0), None)
        after_top1 = next((r.candidate_key for r in query_results if r.rank_after == 0), None)
        if before_top1 is not None and after_top1 is not None and before_top1 != after_top1:
            top1_changed_count += 1

    k1_neutral_count = sum(1 for r in all_results if "k1_no_others" in r.provenance)
    empty_mesh_count = sum(
        1 for r in all_results if "both_empty" in r.provenance or "one_side_empty" in r.provenance
    )

    return FactualRerankingDiagnostics(
        decision_count=n,
        mean_compatibility_score=statistics.mean(compatibility_scores),
        median_compatibility_score=statistics.median(compatibility_scores),
        mean_absolute_rank_shift=statistics.mean(rank_shifts) if rank_shifts else 0.0,
        fraction_rank_changed=len(changed) / n,
        top1_changed_fraction=(top1_changed_count / query_count) if query_count else 0.0,
        mean_evidence_quality_score=statistics.mean(evidence_scores) if evidence_scores else 0.0,
        k1_neutral_count=k1_neutral_count,
        k1_neutral_fraction=k1_neutral_count / n,
        empty_mesh_neutral_or_zero_count=empty_mesh_count,
        empty_mesh_neutral_or_zero_fraction=empty_mesh_count / n,
    )
