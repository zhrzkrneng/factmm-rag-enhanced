"""Dynamic-K diagnostics aggregation (Cell 48, Section H).

Responsibility: a simple aggregation helper over a batch of
AdaptiveKDecision results, kept separate from the selection algorithm
itself (adaptive_k.py) per Section H: "expose a simple aggregation
helper rather than embedding evaluation logic into the retrieval
algorithm." Pure post-hoc statistics -- computes nothing new, never
re-runs selection, never touches candidate text.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from src.innovation.adaptive_retrieval.adaptive_k import BAND_HIGH, BAND_LOW, BAND_MEDIUM, AdaptiveKDecision


@dataclass(frozen=True)
class AdaptiveKDiagnostics:
    """Aggregate diagnostics over a batch of AdaptiveKDecision results.

    Attributes:
        decision_count: number of decisions aggregated (0 is allowed;
            all other fields fall back to well-defined neutral values --
            see aggregate_adaptive_k_diagnostics).
        mean_selected_k: mean of selected_k across all decisions.
        median_selected_k: median of selected_k across all decisions.
        min_selected_k: minimum selected_k observed.
        max_selected_k: maximum selected_k observed.
        fraction_high_confidence: fraction of decisions in the "high"
            band.
        fraction_medium_confidence: fraction of decisions in the
            "medium" band.
        fraction_low_confidence: fraction of decisions in the "low"
            band.
        candidate_shortage_rate: fraction of decisions where
            selected_candidate_count < the band's raw policy k (i.e.
            fewer real candidates were available than the policy
            wanted) -- detected here via selected_k <
            available_candidate_count's own implied policy_k, exposed
            simply as selected_k < max(min_k-equivalent...); computed
            directly from each decision's own `reason` string never
            being needed -- see implementation: a decision "had a
            shortage" iff its available_candidate_count was the
            limiting factor, i.e. selected_candidate_count ==
            available_candidate_count AND available_candidate_count is
            less than what an unconstrained policy could have chosen
            for that band. This module infers that purely from the
            already-computed fields (confidence_band, selected_k,
            available_candidate_count), never by re-running the policy.
        average_confidence: mean of confidence across all decisions.
        deterministic_decision_count: how many decisions were verified
            deterministic (see verify_batch_determinism) -- 0 if that
            verification was not run.
        total_decisions_checked_for_determinism: denominator for the
            above (0 if not run).
    """

    decision_count: int
    mean_selected_k: float
    median_selected_k: float
    min_selected_k: int
    max_selected_k: int
    fraction_high_confidence: float
    fraction_medium_confidence: float
    fraction_low_confidence: float
    candidate_shortage_rate: float
    average_confidence: float
    deterministic_decision_count: int
    total_decisions_checked_for_determinism: int

    def to_json_dict(self) -> dict:
        return {
            "decision_count": self.decision_count,
            "mean_selected_k": self.mean_selected_k,
            "median_selected_k": self.median_selected_k,
            "min_selected_k": self.min_selected_k,
            "max_selected_k": self.max_selected_k,
            "fraction_high_confidence": self.fraction_high_confidence,
            "fraction_medium_confidence": self.fraction_medium_confidence,
            "fraction_low_confidence": self.fraction_low_confidence,
            "candidate_shortage_rate": self.candidate_shortage_rate,
            "average_confidence": self.average_confidence,
            "deterministic_decision_count": self.deterministic_decision_count,
            "total_decisions_checked_for_determinism": self.total_decisions_checked_for_determinism,
        }


def aggregate_adaptive_k_diagnostics(
    decisions: Sequence[AdaptiveKDecision],
    *,
    deterministic_decision_count: int = 0,
    total_decisions_checked_for_determinism: int = 0,
) -> AdaptiveKDiagnostics:
    """Aggregates a batch of AdaptiveKDecision results into summary
    diagnostics. Never raises on an empty batch (returns well-defined
    zero/neutral values instead) -- diagnostics must be safe to compute
    even before any real decision has been made.

    Args:
        decisions: the decisions to aggregate.
        deterministic_decision_count: how many decisions were confirmed
            deterministic by the caller (e.g. via a repeated-run
            comparison) -- passed through, not recomputed here.
        total_decisions_checked_for_determinism: denominator for the
            above.
    """
    n = len(decisions)
    if n == 0:
        return AdaptiveKDiagnostics(
            decision_count=0, mean_selected_k=0.0, median_selected_k=0.0,
            min_selected_k=0, max_selected_k=0, fraction_high_confidence=0.0,
            fraction_medium_confidence=0.0, fraction_low_confidence=0.0,
            candidate_shortage_rate=0.0, average_confidence=0.0,
            deterministic_decision_count=deterministic_decision_count,
            total_decisions_checked_for_determinism=total_decisions_checked_for_determinism,
        )

    selected_ks = [d.selected_k for d in decisions]
    confidences = [d.confidence for d in decisions]
    band_counts = {BAND_HIGH: 0, BAND_MEDIUM: 0, BAND_LOW: 0}
    for d in decisions:
        band_counts[d.confidence_band] += 1

    # A decision "had a shortage" iff its selected_candidate_count was
    # capped by availability rather than by the band's own policy --
    # detectable directly from the decision's own `reason` string,
    # which select_adaptive_k always sets to mention "capped to" only
    # in exactly that case (see adaptive_k.py's own construction).
    shortage_count = sum(1 for d in decisions if "capped to" in d.reason)

    return AdaptiveKDiagnostics(
        decision_count=n,
        mean_selected_k=statistics.mean(selected_ks),
        median_selected_k=statistics.median(selected_ks),
        min_selected_k=min(selected_ks),
        max_selected_k=max(selected_ks),
        fraction_high_confidence=band_counts[BAND_HIGH] / n,
        fraction_medium_confidence=band_counts[BAND_MEDIUM] / n,
        fraction_low_confidence=band_counts[BAND_LOW] / n,
        candidate_shortage_rate=shortage_count / n,
        average_confidence=statistics.mean(confidences),
        deterministic_decision_count=deterministic_decision_count,
        total_decisions_checked_for_determinism=total_decisions_checked_for_determinism,
    )


def verify_batch_determinism(
    decisions_first_run: Sequence[AdaptiveKDecision],
    decisions_second_run: Sequence[AdaptiveKDecision],
) -> int:
    """Returns how many of the two equal-length decision batches are
    identical pairwise -- the caller is expected to have produced
    `decisions_second_run` by re-running the exact same selection over
    the exact same input as `decisions_first_run`.

    Raises:
        ValueError: if the two batches have different lengths.
    """
    if len(decisions_first_run) != len(decisions_second_run):
        raise ValueError(
            f"decisions_first_run (len={len(decisions_first_run)}) and "
            f"decisions_second_run (len={len(decisions_second_run)}) must have the "
            f"same length to compare pairwise"
        )
    return sum(1 for a, b in zip(decisions_first_run, decisions_second_run) if a == b)
