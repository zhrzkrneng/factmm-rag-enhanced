"""Adaptive-k selection: contract (Cell 47) + real algorithm (Cell 48).

Cell 47 established AdaptiveKConfig/AdaptiveKDecision as a design
contract, explicitly deferring the selection algorithm. This module
(Cell 48, Milestone 3.0 Innovation I1) now implements that algorithm for
real: a deterministic, confidence-band-based mapping from an ordered
candidate list + scores to a selected_k and the actual selected
candidates -- no learned parameters, no retrieval, CPU-only.

Formula (deterministic, documented rather than hidden in code):

    1. Reuse src.innovation.shared.confidence.compute_confidence on the
       caller-supplied, already-ranked candidate scores -- this module
       never computes its own confidence signal (Cell 48 spec Section
       B: "Do not create a second confidence formula").
    2. Classify the resulting confidence.value into one of three
       non-overlapping bands, using AdaptiveKConfig's two thresholds:
           value >= high_confidence_threshold        -> "high"
           low_confidence_threshold <= value < high...-> "medium"
           value <  low_confidence_threshold          -> "low"
    3. Map each band to a policy k:
           "high"   -> config.min_k    (small k: confident retrieval needs little evidence)
           "medium" -> config.default_k (intermediate k)
           "low"    -> config.max_k    (large k: ambiguous retrieval benefits from more evidence)
       This reuses AdaptiveKConfig.default_k for the medium band rather
       than adding a fourth k parameter -- default_k's own contract
       already guarantees min_k <= default_k <= max_k, so the three
       band-to-k values are always monotonic by construction.
    4. Cap the policy k at the number of candidates actually available
       AFTER any caller-side eligibility filtering (e.g. self-match
       removal -- see pipeline_adapter.py): selected_k =
       min(policy_k, available_candidate_count). This is a hard
       structural invariant, not a best-effort check -- Cell 48 spec
       Section D: "Do not silently fabricate candidates to satisfy K."
       Consequently selected_k is always <= max_k, but is only
       guaranteed >= min_k when at least min_k real candidates exist;
       with fewer, the honest (capped) count is returned instead.
    5. selected_candidates = the first selected_k entries of the
       caller-supplied candidate list, in the exact order given --
       this function never reorders or re-sorts its input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence, Tuple

from src.innovation.shared.confidence import compute_confidence

BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"
VALID_BANDS = (BAND_HIGH, BAND_MEDIUM, BAND_LOW)


@dataclass(frozen=True)
class AdaptiveKConfig:
    """I1's configuration. Every bound is validated eagerly (Section 12,
    test 9: "invalid k configuration rejected") rather than discovered
    later at decision time.

    Attributes:
        min_k: smallest number of reports I1 may ever select. Must be
            >= 1.
        max_k: largest number of reports I1 may ever select. Must be
            >= min_k.
        default_k: the k used for medium-confidence queries, and (per
            Cell 48) more generally whenever a clean high/low
            determination is not made. Must satisfy
            min_k <= default_k <= max_k.
        confidence_threshold: legacy single-threshold field, retained
            unmodified from Cell 47 for backward compatibility with
            existing tests/contracts. Must be in [0, 1]. Not used by
            Cell 48's three-band policy (see high_confidence_threshold/
            low_confidence_threshold below).
        margin_threshold: a raw (unnormalized) score-margin threshold,
            reserved for a future secondary signal. Must be >= 0. Not
            used by Cell 48's policy (confidence already incorporates
            margin, normalized by spread -- see
            src.innovation.shared.confidence).
        high_confidence_threshold: the confidence value at or above
            which the "high" band applies (policy k = min_k). Must be
            in [0, 1].
        low_confidence_threshold: the confidence value below which the
            "low" band applies (policy k = max_k); confidence in
            [low_confidence_threshold, high_confidence_threshold)
            is the "medium" band (policy k = default_k). Must be in
            [0, 1] and strictly less than high_confidence_threshold,
            so the three bands are always non-overlapping and together
            cover all of [0, 1].
    """

    min_k: int = 1
    max_k: int = 5
    default_k: int = 3
    confidence_threshold: float = 0.5
    margin_threshold: float = 0.0
    high_confidence_threshold: float = 0.66
    low_confidence_threshold: float = 0.33

    def __post_init__(self) -> None:
        if not isinstance(self.min_k, int) or isinstance(self.min_k, bool) or self.min_k < 1:
            raise ValueError(f"min_k must be a positive integer, got {self.min_k!r}")
        if not isinstance(self.max_k, int) or isinstance(self.max_k, bool) or self.max_k < self.min_k:
            raise ValueError(f"max_k must be an integer >= min_k ({self.min_k}), got {self.max_k!r}")
        if (
            not isinstance(self.default_k, int)
            or isinstance(self.default_k, bool)
            or not (self.min_k <= self.default_k <= self.max_k)
        ):
            raise ValueError(
                f"default_k must be an integer in [min_k, max_k] "
                f"([{self.min_k}, {self.max_k}]), got {self.default_k!r}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold!r}"
            )
        if self.margin_threshold < 0.0:
            raise ValueError(f"margin_threshold must be >= 0.0, got {self.margin_threshold!r}")
        if not (0.0 <= self.high_confidence_threshold <= 1.0):
            raise ValueError(
                f"high_confidence_threshold must be in [0.0, 1.0], got {self.high_confidence_threshold!r}"
            )
        if not (0.0 <= self.low_confidence_threshold <= 1.0):
            raise ValueError(
                f"low_confidence_threshold must be in [0.0, 1.0], got {self.low_confidence_threshold!r}"
            )
        if not (self.low_confidence_threshold < self.high_confidence_threshold):
            raise ValueError(
                f"low_confidence_threshold ({self.low_confidence_threshold!r}) must be "
                f"strictly less than high_confidence_threshold "
                f"({self.high_confidence_threshold!r}) -- the three confidence bands "
                f"(low/medium/high) must be non-overlapping"
            )


class AdaptiveKDecision(NamedTuple):
    """One query's real adaptive-k decision (Cell 48).

    Attributes:
        selected_k: the final, availability-capped candidate count --
            always equal to len(selected_candidates) and always
            <= max_k; only guaranteed >= min_k when at least min_k
            candidates were actually available (see module docstring
            step 4 -- never fabricated to reach min_k).
        confidence: the shared-confidence value (Section 6) this
            decision was based on, in [0, 1].
        confidence_band: one of "high"/"medium"/"low" (see module
            docstring step 2).
        score_margin: the raw top1-vs-top2 margin (see
            src.innovation.shared.confidence.ConfidenceResult.margin),
            or None if fewer than 2 candidates were available.
        score_spread: the raw candidate-score spread (see
            ConfidenceResult.spread), or None iff zero candidates.
        available_candidate_count: how many candidates were passed in
            (after any caller-side eligibility filtering).
        selected_candidate_count: == selected_k == len(selected_candidates)
            -- reported as its own field per Cell 48 spec Section E's
            explicit output-contract requirement.
        selected_candidates: the chosen candidates, in the exact input
            order (this module never reorders or re-sorts).
        reason: a short, human-readable explanation of why this k was
            selected (for per-query auditability, per
            docs/innovation_proposals.md Innovation A's evaluation
            discipline), including the raw policy k before any
            availability capping was applied.
    """

    selected_k: int
    confidence: float
    confidence_band: str
    score_margin: Optional[float]
    score_spread: Optional[float]
    available_candidate_count: int
    selected_candidate_count: int
    selected_candidates: Tuple[Any, ...]
    reason: str


def _classify_band(confidence: float, config: AdaptiveKConfig) -> str:
    if confidence >= config.high_confidence_threshold:
        return BAND_HIGH
    if confidence < config.low_confidence_threshold:
        return BAND_LOW
    return BAND_MEDIUM


def _policy_k_for_band(band: str, config: AdaptiveKConfig) -> int:
    return {
        BAND_HIGH: config.min_k,
        BAND_MEDIUM: config.default_k,
        BAND_LOW: config.max_k,
    }[band]


def _validate_scores(candidate_scores: Sequence[float]) -> None:
    for score in candidate_scores:
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"candidate score must be a real number, got {score!r}")
        if score != score:  # NaN check without importing math -- NaN != NaN.
            raise ValueError("candidate score must not be NaN")
        if score in (float("inf"), float("-inf")):
            raise ValueError("candidate score must not be infinite")


def select_adaptive_k(
    candidate_ids: Sequence[Any],
    candidate_scores: Sequence[float],
    config: AdaptiveKConfig,
) -> AdaptiveKDecision:
    """Deterministic Dynamic-K selection over an already-ranked candidate
    list. See module docstring for the exact formula.

    Does not perform retrieval, does not reorder `candidate_ids`, does
    not fabricate candidates to reach a target k. Generic over candidate
    identity type (QueryKey, str, int, ...) -- this function only ever
    sees ids + scores, never candidate text or a target report.

    Args:
        candidate_ids: candidate identifiers, already ranked
            best-first by the caller.
        candidate_scores: parallel retrieval scores (same length,
            same order as candidate_ids).
        config: validated AdaptiveKConfig.

    Raises:
        ValueError: if `candidate_ids` and `candidate_scores` have
            different lengths, or if any score is non-numeric, NaN, or
            infinite (Section D: "NaN / infinite score rejection") --
            never silently coerced or dropped.
    """
    if len(candidate_ids) != len(candidate_scores):
        raise ValueError(
            f"candidate_ids (len={len(candidate_ids)}) and candidate_scores "
            f"(len={len(candidate_scores)}) must have the same length"
        )
    _validate_scores(candidate_scores)

    available = len(candidate_ids)
    confidence_result = compute_confidence(candidate_scores)
    band = _classify_band(confidence_result.value, config)
    policy_k = _policy_k_for_band(band, config)
    selected_k = min(policy_k, available)
    selected_candidates = tuple(candidate_ids[:selected_k])

    if selected_k < policy_k:
        reason = (
            f"confidence={confidence_result.value:.4f} (band={band!r}) -> policy k="
            f"{policy_k}, capped to {selected_k} because only {available} candidate(s) "
            f"were available (never fabricated)"
        )
    else:
        reason = f"confidence={confidence_result.value:.4f} (band={band!r}) -> selected k={selected_k}"

    return AdaptiveKDecision(
        selected_k=selected_k,
        confidence=confidence_result.value,
        confidence_band=band,
        score_margin=confidence_result.margin,
        score_spread=confidence_result.spread,
        available_candidate_count=available,
        selected_candidate_count=len(selected_candidates),
        selected_candidates=selected_candidates,
        reason=reason,
    )
