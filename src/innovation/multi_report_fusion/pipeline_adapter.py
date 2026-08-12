"""Confidence-Gated Multi-Report Fusion <-> existing RAG-pipeline
adapter (Cell 50, I3, mirroring adaptive_retrieval/pipeline_adapter.py
and factual_reranking/pipeline_adapter.py's architecture).

Responsibility: bridge the generic, dataset-agnostic fusion algorithm
(fusion.py's `select_fusion_candidates`/`build_evidence_context`, which
only ever see FusionCandidateInput / (key, text) pairs) to real IU
X-Ray data (src.data.iu_xray.records.IuXrayCanonicalRecord) WITHOUT
modifying any baseline source file and WITHOUT ever passing the
query's target-report content into the fusion algorithm -- only each
CANDIDATE's own study_id/findings/impression, keyed by that
candidate's own QueryKey.

Pipeline order (Section 5): Retrieval -> I1 Dynamic-K -> I2 Reranking
-> I3 Confidence-Gated Fusion -> Evidence Context. This module
consumes `candidate_keys`/`candidate_scores` exactly as they are
handed to it -- i.e. already in whatever final order I1 and/or I2
already produced (I2's post-rerank order, with I2's own final_score,
when I2 ran; the raw retrieval order/score otherwise). I3 never
reorders its input and never adds a candidate beyond what it is given
-- `effective_fusion_count = min(policy_fusion_count,
available_candidate_count)` (enforced inside
fusion.select_fusion_candidates) is the hard guarantee that I3 can
never invent or resurrect a candidate an earlier stage (I1) already
discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.innovation.experiment_contract import InnovationConfig
from src.innovation.multi_report_fusion.fusion import (
    EvidenceContext,
    FusionCandidateInput,
    FusionConfig,
    FusionDecision,
    build_evidence_context,
    select_fusion_candidates,
)


def _format_evidence_block(
    rank: int, study_id: str, findings: Optional[str], impression: Optional[str]
) -> str:
    """Deterministic, structured, no-LLM evidence text for one
    candidate (Section 5: "I3 is evidence-context fusion, not final
    report generation"). Retains candidate/study ID, rank, findings if
    available, impression if available -- the minimum fields Cell 50's
    spec requires each fused evidence block to keep."""
    findings_text = findings.strip() if findings and findings.strip() else "N/A"
    impression_text = impression.strip() if impression and impression.strip() else "N/A"
    return (
        f"[Evidence {rank + 1} | Study {study_id}]\n"
        f"Findings: {findings_text}\n"
        f"Impression: {impression_text}"
    )


def _build_evidence_source_items(
    selected_keys: Sequence[QueryKey],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
) -> Tuple[Tuple[QueryKey, str], ...]:
    """Builds (source_key, text) pairs for build_evidence_context from
    real IU X-Ray records, in the exact order given. A candidate key
    missing from `records_by_key` degrades to a "N/A" evidence block
    (never fabricated, never an exception) -- study_id falls back to
    the QueryKey's own study-id component, matching
    factual_reranking.pipeline_adapter's identical missing-record
    convention."""
    items = []
    for rank, key in enumerate(selected_keys):
        record = records_by_key.get(key)
        study_id = record.study_id if record is not None else key[2]
        findings = record.findings if record is not None else None
        impression = record.impression if record is not None else None
        items.append((key, _format_evidence_block(rank, study_id, findings, impression)))
    return tuple(items)


@dataclass(frozen=True)
class FusionSelectionResult:
    """Uniform result shape for both the baseline-equivalent (I3
    disabled) and confidence-gated-fusion (I3 enabled) paths, mirroring
    RerankingSelectionResult/EvidenceSelectionResult so a caller can
    treat both identically.

    Attributes:
        query_key: the query this fusion decision is for.
        selected_candidates: the candidate keys actually selected into
            evidence, in order -- always a PREFIX of the input
            `candidate_keys`, never containing a candidate outside
            that input.
        fusion_used: False for the baseline-equivalent path (I3
            disabled -- top-1 pass-through, matching pre-I3 single-
            report behavior exactly), True for the real
            confidence-gated fusion path.
        decision: the full FusionDecision, or None on the
            baseline-equivalent path (which does not run the fusion
            algorithm at all).
        evidence_context: the structured EvidenceContext, or None on
            the baseline-equivalent path (Section: I3 disabled must
            preserve EXISTING evidence behavior -- no new budget/
            truncation machinery is introduced when I3 is off).
    """

    query_key: QueryKey
    selected_candidates: Tuple[QueryKey, ...]
    fusion_used: bool
    decision: Optional[FusionDecision]
    evidence_context: Optional[EvidenceContext]


def select_baseline_equivalent_evidence(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
) -> FusionSelectionResult:
    """I3 disabled: exactly the pre-I3 single-report selection (first
    candidate of whatever order it is given -- I1/I2's own E0 paths
    already reproduce this same "take candidate 0" behavior one stage
    earlier). No EvidenceContext is built here -- I3 disabled means
    I3's machinery (including budget enforcement) never runs, so no
    existing behavior can be silently altered. `candidate_scores` is
    accepted only to keep this function's signature symmetric with the
    fused path; it is never inspected here."""
    selected = tuple(candidate_keys[:1])
    return FusionSelectionResult(
        query_key=query_key, selected_candidates=selected, fusion_used=False,
        decision=None, evidence_context=None,
    )


def select_fused_evidence(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
    config: FusionConfig,
) -> FusionSelectionResult:
    """I3 enabled: runs the real confidence-gated fusion decision over
    `candidate_keys` in the exact order given (post-I2 order when I2
    ran), then builds a budget-enforced EvidenceContext over exactly
    the selected prefix -- never more candidates than
    `select_fusion_candidates` decided on, never candidates outside
    the input set."""
    if len(candidate_keys) != len(candidate_scores):
        raise ValueError(
            f"candidate_keys (len={len(candidate_keys)}) and candidate_scores "
            f"(len={len(candidate_scores)}) must have the same length"
        )
    fusion_inputs = [
        FusionCandidateInput(candidate_key=key, score=score)
        for key, score in zip(candidate_keys, candidate_scores)
    ]
    decision = select_fusion_candidates(fusion_inputs, config)
    selected_keys = tuple(candidate_keys[: decision.selected_evidence_count])
    source_items = _build_evidence_source_items(selected_keys, records_by_key)
    evidence_context = build_evidence_context(source_items, config)
    return FusionSelectionResult(
        query_key=query_key, selected_candidates=selected_keys, fusion_used=True,
        decision=decision, evidence_context=evidence_context,
    )


def select_evidence_for_query(
    query_key: QueryKey,
    candidate_keys: Sequence[QueryKey],
    candidate_scores: Sequence[float],
    records_by_key: Dict[QueryKey, IuXrayCanonicalRecord],
    innovation_config: InnovationConfig,
) -> FusionSelectionResult:
    """Single entry point: branches on
    innovation_config.confidence_gated_fusion_enabled so callers never
    duplicate that branch. With every innovation disabled
    (InnovationConfig(), i.e. E0), this is exactly baseline-equivalent
    (single top-1 candidate, no EvidenceContext machinery invoked).

    Callers are responsible for supplying `candidate_keys`/
    `candidate_scores` already in the CURRENT pipeline order (Section
    5: "I3 must consume candidate ordering AFTER I2 when I2 is
    enabled") -- i.e. pass through I2's `selected_candidates` (and
    each candidate's I2 `final_score`, or the raw retrieval score if
    I2 did not run) directly, without re-deriving or re-sorting them
    here.
    """
    if innovation_config.confidence_gated_fusion_enabled:
        return select_fused_evidence(
            query_key, candidate_keys, candidate_scores, records_by_key, innovation_config.fusion,
        )
    return select_baseline_equivalent_evidence(query_key, candidate_keys, candidate_scores)
