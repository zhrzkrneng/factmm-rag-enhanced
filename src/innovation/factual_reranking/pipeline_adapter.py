"""Factual/Clinical Reranking <-> existing RAG-pipeline adapter (Cell 48,
I2, mirroring adaptive_retrieval/pipeline_adapter.py's architecture).

Responsibility: bridge the generic, dataset-agnostic reranking algorithm
(reranker.py's `rerank_candidates`, which only ever sees
RerankCandidateInput) to real IU X-Ray data
(src.data.iu_xray.records.IuXrayCanonicalRecord) WITHOUT modifying any
baseline source file and WITHOUT ever passing target-report content
into the reranker -- only each CANDIDATE's own mesh_terms/findings/
impression, keyed by that candidate's own QueryKey.

I2 operates strictly on the candidate set it is given (e.g. I1's
already-selected evidence, or the raw retrieval ranking if I1 is
disabled) -- it never fetches additional candidates, never touches
retrieval itself, and never changes the size of the candidate set,
only its order and attached component scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.innovation.experiment_contract import InnovationConfig
from src.innovation.factual_reranking.reranker import (
    RerankCandidateInput,
    RerankConfig,
    RerankResult,
    rerank_candidates,
)


@dataclass(frozen=True)
class RerankingSelectionResult:
    """Uniform result shape for both the baseline-equivalent (I2
    disabled) and reranked (I2 enabled) paths, mirroring
    adaptive_retrieval.pipeline_adapter.EvidenceSelectionResult so a
    caller can treat both identically.

    Attributes:
        query_key: the query this reranking is for.
        selected_candidates: candidate keys in FINAL order -- exactly
            the same set as the input candidates, never adds or drops
            any; unchanged input order when I2 is disabled.
        reranking_used: False for the baseline-equivalent path (input
            order passed through verbatim), True for the real
            reranking path.
        results: the full per-candidate RerankResult list (reranked
            order), or None on the baseline-equivalent path (which
            never runs the reranking algorithm at all).
    """

    query_key: QueryKey
    selected_candidates: Tuple[QueryKey, ...]
    reranking_used: bool
    results: Optional[Tuple[RerankResult, ...]]


def _build_rerank_inputs(
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
) -> List[RerankCandidateInput]:
    """Builds the generic RerankCandidateInput list from real
    IU X-Ray records, keyed by each CANDIDATE's own key -- never the
    query's. A candidate key missing from `records_by_key` degrades to
    empty mesh_terms/None findings/impression (never fabricated,
    never an exception -- matches this project's "never raise for a
    missing optional lookup" convention where the caller has already
    validated candidate keys elsewhere, e.g. via filter_self_matches)."""
    if len(candidate_keys) != len(candidate_scores):
        raise ValueError(
            f"candidate_keys (len={len(candidate_keys)}) and candidate_scores "
            f"(len={len(candidate_scores)}) must have the same length"
        )
    inputs: List[RerankCandidateInput] = []
    for key, score in zip(candidate_keys, candidate_scores):
        record = records_by_key.get(key)
        inputs.append(
            RerankCandidateInput(
                candidate_key=key,
                retrieval_score=score,
                mesh_terms=list(record.mesh_terms) if record is not None else [],
                findings=record.findings if record is not None else None,
                impression=record.impression if record is not None else None,
            )
        )
    return inputs


def select_baseline_equivalent_ranking(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
) -> RerankingSelectionResult:
    """I2 disabled: candidate identity AND order are passed through
    completely unchanged -- byte-identical to whatever the caller
    (I1's selection, or raw retrieval) already produced. Scores are
    accepted only to keep this function's signature symmetric with the
    reranked path; they are never inspected here."""
    return RerankingSelectionResult(
        query_key=query_key,
        selected_candidates=tuple(candidate_keys),
        reranking_used=False,
        results=None,
    )


def select_reranked_evidence(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
    config: RerankConfig,
) -> RerankingSelectionResult:
    """I2 enabled: builds generic reranking inputs from real candidate
    records, runs rerank_candidates, and returns the reranked key
    order. The input candidate SET is exactly what is returned --
    rerank_candidates never adds or drops a candidate."""
    inputs = _build_rerank_inputs(candidate_keys, candidate_scores, records_by_key)
    results = rerank_candidates(inputs, config)
    ordered_keys = tuple(result.candidate_key for result in results)
    return RerankingSelectionResult(
        query_key=query_key,
        selected_candidates=ordered_keys,
        reranking_used=True,
        results=tuple(results),
    )


def select_evidence_for_query(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
    innovation_config: InnovationConfig,
) -> RerankingSelectionResult:
    """Single entry point: branches on
    innovation_config.factual_reranking_enabled so callers never
    duplicate that branch. With every innovation disabled
    (InnovationConfig(), i.e. E0), this is exactly baseline-equivalent
    (candidate order untouched)."""
    if innovation_config.factual_reranking_enabled:
        return select_reranked_evidence(
            query_key, candidate_keys, candidate_scores, records_by_key, innovation_config.reranking,
        )
    return select_baseline_equivalent_ranking(query_key, candidate_keys, candidate_scores)
