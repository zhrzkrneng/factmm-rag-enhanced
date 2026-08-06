"""Adaptive-k selection contract (Cell 47, Milestone 3.0, Section 3).

Responsibility (Cell 47 scope -- contract only): the configuration and
decision-result dataclasses for I1 (Adaptive Retrieval / Dynamic-K).
The actual selection algorithm (a function that consumes ranked
candidate scores and returns an AdaptiveKDecision) is intentionally NOT
implemented here -- Cell 47's spec Section 10 is explicit: "Do NOT yet
implement the full Adaptive-K... algorithm[]. Those belong to Cells
48-50. Cell 47 should establish interfaces/configuration/contracts
sufficient for those cells." This module is that contract.

Documented candidate principle (Section 3), to be implemented by Cell 48
against src.innovation.shared.confidence.compute_confidence's margin/
spread signals -- not hidden in undocumented code:

    High-confidence retrieval (large normalized top1-vs-top2 margin
    relative to this query's own candidate-score spread) -> small k
    (closer to min_k).

    Ambiguous / flat score distribution (small or zero margin -- see
    confidence.py's "tied_scores" basis) -> larger k (closer to max_k).

    A future selection function is expected to interpolate between
    min_k and max_k as a function of
    src.innovation.shared.confidence.ConfidenceResult.value, gated by
    `confidence_threshold`/`margin_threshold` below, and must itself
    satisfy every requirement in Section 3: deterministic, CPU-only, no
    learned parameters, bounded min_k <= k <= max_k, safe with fewer
    than max_k candidates, safe with ties, safe with empty rankings,
    and never reordering the input ranking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional


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
        default_k: the k used when a decision cannot otherwise be made
            (e.g. by a caller wanting a baseline-equivalent constant-k
            fallback). Must satisfy min_k <= default_k <= max_k.
        confidence_threshold: the shared-confidence value (Section 6)
            at or above which a future selection algorithm should prefer
            small k. Must be in [0, 1].
        margin_threshold: a raw (unnormalized) score-margin threshold a
            future selection algorithm may use as a secondary signal
            alongside confidence_threshold. Must be >= 0.
    """

    min_k: int = 1
    max_k: int = 5
    default_k: int = 1
    confidence_threshold: float = 0.5
    margin_threshold: float = 0.0

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


class AdaptiveKDecision(NamedTuple):
    """One query's adaptive-k decision (produced by Cell 48's selection
    algorithm, not by this module).

    Attributes:
        selected_k: bounded min_k <= selected_k <= max_k.
        confidence: the shared-confidence value (Section 6) this
            decision was based on, in [0, 1].
        score_margin: the raw top1-vs-top2 margin (see
            src.innovation.shared.confidence.ConfidenceResult.margin),
            or None if fewer than 2 candidates were available.
        score_spread: the raw candidate-score spread (see
            ConfidenceResult.spread), or None iff zero candidates.
        reason: a short, human-readable explanation of why this k was
            selected (for per-query auditability, per
            docs/innovation_proposals.md Innovation A's evaluation
            discipline).
    """

    selected_k: int
    confidence: float
    score_margin: Optional[float]
    score_spread: Optional[float]
    reason: str
