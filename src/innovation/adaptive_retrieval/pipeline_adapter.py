"""Dynamic-K <-> existing RAG-pipeline adapter (Cell 48, Section F).

Responsibility: let I1's Dynamic-K decision be consumed by the existing
retrieval/RAG pipeline WITHOUT modifying any baseline source file
(src.baseline.generation.dataset_builder.RAGDatasetBuilder itself is
never touched). This module operates entirely on the outside: it
consumes the same `knn_rankings`-shaped input
(Dict[QueryKey, List[QueryKey]], best-first) the real pipeline already
produces (see src.data.iu_xray.generation_adapter's ranking builders),
plus caller-supplied per-candidate scores, and reuses (never
reimplements) src.data.iu_xray.retrieval_adapter.filter_self_matches --
the same, already-tested self-match exclusion Cell 45's real adapters
use -- so eligibility filtering can never silently drift from the
established policy.

Baseline compatibility (Section F):
    - `select_baseline_equivalent_evidence` reproduces exactly what
      RAGDatasetBuilder.build_row already does for K=1: filter self
      matches (in this project, self-study and self-patient exclusion
      collapse to one check for IU X-Ray, per
      retrieval_adapter.SELF_MATCH_EXCLUSION_POLICY), then take the
      first eligible candidate. E0 (I1 disabled) uses this path.
    - `select_dynamic_k_evidence` filters the same way, then defers to
      select_adaptive_k for how many candidates to keep. E1 (I1
      enabled) uses this path.
    - `select_evidence_for_query` is the single entry point that
      branches on `InnovationConfig.adaptive_retrieval_enabled`, so a
      caller never has to duplicate that branch itself.

Neither path touches query construction, corpus construction, retrieval
scores, candidate ordering, the target report, the prompt template, or
the generator -- only which/how-many already-ranked, already-filtered
candidates are kept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray.retrieval_adapter import filter_self_matches
from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig, AdaptiveKDecision, select_adaptive_k
from src.innovation.experiment_contract import InnovationConfig


def _filter_eligible_with_scores(
    query_key: QueryKey,
    ranked_candidate_keys: Sequence[QueryKey],
    ranked_candidate_scores: Sequence[float],
) -> Tuple[List[QueryKey], List[float]]:
    """Applies the established self-match filter to `ranked_candidate_keys`
    (via the real, already-tested retrieval_adapter.filter_self_matches),
    then keeps `ranked_candidate_scores` aligned by index -- order and
    1:1 correspondence with the original ranking are both preserved."""
    if len(ranked_candidate_keys) != len(ranked_candidate_scores):
        raise ValueError(
            f"ranked_candidate_keys (len={len(ranked_candidate_keys)}) and "
            f"ranked_candidate_scores (len={len(ranked_candidate_scores)}) must have "
            f"the same length"
        )
    eligible_keys = filter_self_matches(query_key, list(ranked_candidate_keys))
    eligible_key_set = set(eligible_keys)
    filtered_keys: List[QueryKey] = []
    filtered_scores: List[float] = []
    for key, score in zip(ranked_candidate_keys, ranked_candidate_scores):
        if key in eligible_key_set:
            filtered_keys.append(key)
            filtered_scores.append(score)
    return filtered_keys, filtered_scores


@dataclass(frozen=True)
class EvidenceSelectionResult:
    """Uniform result shape for both the baseline-equivalent (E0) and
    Dynamic-K (E1) selection paths, so a caller can treat them
    identically regardless of which one ran.

    Attributes:
        query_key: the query this selection is for.
        selected_candidates: chosen candidate keys, best-first, never
            containing `query_key` itself or a same-patient candidate.
        adaptive_k_used: False for the baseline-equivalent path, True
            for the Dynamic-K path.
        decision: the full AdaptiveKDecision, or None on the
            baseline-equivalent path (which does not run the Dynamic-K
            algorithm at all).
    """

    query_key: QueryKey
    selected_candidates: Tuple[QueryKey, ...]
    adaptive_k_used: bool
    decision: Optional[AdaptiveKDecision]


def select_baseline_equivalent_evidence(
    query_key: QueryKey,
    ranked_candidate_keys: Sequence[QueryKey],
    ranked_candidate_scores: Sequence[float],
) -> EvidenceSelectionResult:
    """E0 path: exactly RAGDatasetBuilder.build_row's own K=1 selection
    (first eligible candidate after self-match filtering), reproduced
    here without touching that class. Scores are accepted only to keep
    this function's signature symmetric with the Dynamic-K path; they do
    not affect which candidate is chosen (RAGDatasetBuilder itself never
    uses a numeric score either -- only ranking order)."""
    eligible_keys, _ = _filter_eligible_with_scores(
        query_key, ranked_candidate_keys, ranked_candidate_scores
    )
    selected = tuple(eligible_keys[:1])
    return EvidenceSelectionResult(
        query_key=query_key, selected_candidates=selected, adaptive_k_used=False, decision=None,
    )


def select_dynamic_k_evidence(
    query_key: QueryKey,
    ranked_candidate_keys: Sequence[QueryKey],
    ranked_candidate_scores: Sequence[float],
    config: AdaptiveKConfig,
) -> EvidenceSelectionResult:
    """E1 path: the same self-match filtering as the baseline-equivalent
    path, then Dynamic-K selection (select_adaptive_k) over the filtered
    candidates."""
    eligible_keys, eligible_scores = _filter_eligible_with_scores(
        query_key, ranked_candidate_keys, ranked_candidate_scores
    )
    decision = select_adaptive_k(eligible_keys, eligible_scores, config)
    return EvidenceSelectionResult(
        query_key=query_key,
        selected_candidates=decision.selected_candidates,
        adaptive_k_used=True,
        decision=decision,
    )


def select_evidence_for_query(
    query_key: QueryKey,
    ranked_candidate_keys: Sequence[QueryKey],
    ranked_candidate_scores: Sequence[float],
    innovation_config: InnovationConfig,
) -> EvidenceSelectionResult:
    """Single entry point: branches on
    innovation_config.adaptive_retrieval_enabled so callers never
    duplicate that branch. With every innovation disabled
    (InnovationConfig(), i.e. E0), this is exactly baseline-equivalent
    -- see test_e0_matches_real_rag_dataset_builder in
    tests/unit/test_innovation_dynamic_k_pipeline_adapter.py for a
    direct, real-RAGDatasetBuilder proof."""
    if innovation_config.adaptive_retrieval_enabled:
        return select_dynamic_k_evidence(
            query_key, ranked_candidate_keys, ranked_candidate_scores, innovation_config.adaptive_k,
        )
    return select_baseline_equivalent_evidence(query_key, ranked_candidate_keys, ranked_candidate_scores)
