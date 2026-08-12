"""Confidence-Gated Multi-Report Fusion diagnostics aggregation (Cell
50, I3, mirroring adaptive_retrieval/diagnostics.py and
factual_reranking/diagnostics.py's architecture).

Responsibility: a simple aggregation helper over a batch of
FusionDecision results, kept separate from the fusion algorithm itself
(fusion.py) per the same discipline established for I1/I2: expose a
post-hoc aggregation helper rather than embedding evaluation logic
into the algorithm. Pure post-hoc statistics -- computes nothing new,
never re-runs fusion, never touches candidate/evidence text.

No clinical-improvement claim is made or implied by any field here --
these are purely mechanical/structural statistics about how the
confidence-gated fusion decision behaved (how many reports were
combined, confidence-band distribution), never a retrieval-quality or
clinical-accuracy measurement.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Dict, Sequence

from src.innovation.multi_report_fusion.fusion import BAND_HIGH, BAND_LOW, BAND_MEDIUM, FusionDecision


@dataclass(frozen=True)
class MultiReportFusionDiagnostics:
    """Aggregate diagnostics over a batch of FusionDecision results.

    Attributes:
        decision_count: number of decisions aggregated (0 is allowed;
            all other fields fall back to well-defined neutral values
            -- see aggregate_fusion_diagnostics).
        mean_fused_report_count: mean of selected_evidence_count across
            all decisions.
        median_fused_report_count: median of selected_evidence_count
            across all decisions.
        fraction_single_report_contexts: fraction of decisions with
            selected_evidence_count == 1 (i.e. should_fuse is False).
        fraction_multi_report_contexts: fraction of decisions with
            selected_evidence_count > 1 (i.e. should_fuse is True).
        fraction_high_confidence / fraction_medium_confidence /
            fraction_low_confidence: confidence-band distribution
            (Cell 50's three-band policy).
        candidate_shortage_rate: fraction of decisions where the
            band's desired report count was capped down because fewer
            real candidates were available than the policy wanted
            (detected via decision.selected_evidence_count <
            desired-report-count, exposed here by the decision's own
            reason string containing "capped to" -- the same
            convention adaptive_retrieval's diagnostics uses).
        average_confidence: mean of confidence across all decisions.
        mean_evidence_blocks_per_query: mean number of evidence blocks
            (== selected_evidence_count) actually assembled per query
            -- reported as its own field per Cell 50's spec, even
            though numerically identical to mean_fused_report_count,
            since the two describe conceptually distinct things (how
            many reports were selected vs. how many evidence blocks
            ended up in the built EvidenceContext).
        available_candidate_count_histogram: maps each observed
            available_candidate_count (as a string, for JSON safety)
            to how many decisions saw that many candidates -- exposes
            I3's interaction with whatever count I1 (and I2) already
            produced upstream, without recomputing or re-deriving it.
    """

    decision_count: int
    mean_fused_report_count: float
    median_fused_report_count: float
    fraction_single_report_contexts: float
    fraction_multi_report_contexts: float
    fraction_high_confidence: float
    fraction_medium_confidence: float
    fraction_low_confidence: float
    candidate_shortage_rate: float
    average_confidence: float
    mean_evidence_blocks_per_query: float
    available_candidate_count_histogram: Dict[str, int]

    def to_json_dict(self) -> dict:
        return {
            "decision_count": self.decision_count,
            "mean_fused_report_count": self.mean_fused_report_count,
            "median_fused_report_count": self.median_fused_report_count,
            "fraction_single_report_contexts": self.fraction_single_report_contexts,
            "fraction_multi_report_contexts": self.fraction_multi_report_contexts,
            "fraction_high_confidence": self.fraction_high_confidence,
            "fraction_medium_confidence": self.fraction_medium_confidence,
            "fraction_low_confidence": self.fraction_low_confidence,
            "candidate_shortage_rate": self.candidate_shortage_rate,
            "average_confidence": self.average_confidence,
            "mean_evidence_blocks_per_query": self.mean_evidence_blocks_per_query,
            "available_candidate_count_histogram": dict(self.available_candidate_count_histogram),
        }


_EMPTY_DIAGNOSTICS_KWARGS = dict(
    decision_count=0,
    mean_fused_report_count=0.0,
    median_fused_report_count=0.0,
    fraction_single_report_contexts=0.0,
    fraction_multi_report_contexts=0.0,
    fraction_high_confidence=0.0,
    fraction_medium_confidence=0.0,
    fraction_low_confidence=0.0,
    candidate_shortage_rate=0.0,
    average_confidence=0.0,
    mean_evidence_blocks_per_query=0.0,
    available_candidate_count_histogram={},
)


def aggregate_fusion_diagnostics(
    decisions: Sequence[FusionDecision],
) -> MultiReportFusionDiagnostics:
    """Aggregates a batch of FusionDecision results into summary
    diagnostics. Never raises on an empty batch (returns well-defined
    zero/neutral values instead) -- diagnostics must be safe to compute
    even before any real decision has been made.

    Args:
        decisions: one FusionDecision per query.
    """
    n = len(decisions)
    if n == 0:
        return MultiReportFusionDiagnostics(**_EMPTY_DIAGNOSTICS_KWARGS)

    selected_counts = [d.selected_evidence_count for d in decisions]
    confidences = [d.confidence for d in decisions]
    band_counts = {BAND_HIGH: 0, BAND_MEDIUM: 0, BAND_LOW: 0}
    for d in decisions:
        band_counts[d.confidence_band] += 1

    single_report_count = sum(1 for d in decisions if d.selected_evidence_count == 1)
    multi_report_count = sum(1 for d in decisions if d.selected_evidence_count > 1)

    # A decision "had a shortage" iff its desired report count was
    # capped by candidate availability -- detectable directly from the
    # decision's own `reason` string, which select_fusion_candidates
    # always sets to mention "capped to" only in exactly that case
    # (see fusion.py's own construction; mirrors adaptive_k.py's
    # identical convention).
    shortage_count = sum(1 for d in decisions if "capped to" in d.reason)

    histogram: Dict[str, int] = {}
    for d in decisions:
        key = str(d.available_candidate_count)
        histogram[key] = histogram.get(key, 0) + 1

    return MultiReportFusionDiagnostics(
        decision_count=n,
        mean_fused_report_count=statistics.mean(selected_counts),
        median_fused_report_count=statistics.median(selected_counts),
        fraction_single_report_contexts=single_report_count / n,
        fraction_multi_report_contexts=multi_report_count / n,
        fraction_high_confidence=band_counts[BAND_HIGH] / n,
        fraction_medium_confidence=band_counts[BAND_MEDIUM] / n,
        fraction_low_confidence=band_counts[BAND_LOW] / n,
        candidate_shortage_rate=shortage_count / n,
        average_confidence=statistics.mean(confidences),
        mean_evidence_blocks_per_query=statistics.mean(selected_counts),
        available_candidate_count_histogram=histogram,
    )
