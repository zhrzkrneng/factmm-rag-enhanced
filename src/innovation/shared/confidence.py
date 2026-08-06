"""Shared retrieval/evidence confidence contract (Cell 47, Section 6).

Responsibility: one normalized confidence representation in [0, 1],
computed only from ranked retrieval candidate scores, that I1 (Adaptive
Retrieval / Dynamic-K), I2 (Factual/Clinical Reranking), and I3
(Confidence-Gated Multi-Report Fusion) all consume -- so no innovation
invents its own unrelated definition of "confidence" (a real, non-mock
requirement of Milestone 3.0's contract; see Cell 47's spec Section 6).

Formula (deterministic, no learned parameters):

    Given `scores`, assumed already ranked descending by the caller
    (this module does not re-sort -- callers own their ranking order,
    matching Cell 47's "preserve original ranking order" requirement for
    the innovations built on top of this module):

        n = len(scores)
        if n == 0:
            confidence = 0.0            (no evidence at all)
        else:
            top1 = scores[0]
            top2 = scores[1] if n >= 2 else None
            raw_margin = (top1 - top2) if top2 is not None else None
            raw_spread = (max(scores) - min(scores)) if n >= 2 else (0.0 if n == 1 else None)

            denom = raw_spread if (raw_spread is not None and raw_spread > 0) else 1.0
            confidence = clamp((raw_margin or 0.0) / denom, 0.0, 1.0)

    In words: confidence is the top-1-vs-top-2 margin, normalized by
    this candidate set's own observed score range (min-max), clamped to
    [0, 1]. Normalizing by the set's own range -- rather than assuming
    any fixed absolute scale -- is what makes this usable across
    retrievers whose raw score semantics differ (cosine similarity, dot
    product, BM25, etc.); this module makes no assumption about what a
    raw score "means" beyond "higher ranks higher."

Which signals contribute:
    - the top-1 score, relative to the top-2 score (the "margin")
    - the observed score range across all supplied candidates (the
      "spread"), used only as the margin's normalizer

Which signals do NOT contribute (deliberately):
    - the absolute magnitude of any individual score (only relative
      structure within the given candidate set is used, since raw
      retrieval-score scales are not comparable across retrievers)
    - anything about the candidates' text content (this module only
      ever sees numeric scores)

How missing signals are handled (never an exception -- always a
defined, documented result):
    - zero candidates: confidence = 0.0 (`basis="empty_candidate_list"`)
      -- the documented minimum, since there is no evidence at all.
    - exactly one candidate: there is no second candidate to compute a
      margin against, so `margin=None` and the margin's contribution to
      confidence is treated as 0.0 (`basis="single_candidate_no_margin"`)
      -- this deliberately does NOT assume high confidence merely from
      the absence of competing candidates; the absence of comparative
      evidence is not itself evidence of a clear winner.
    - all candidates tied (raw_spread == 0.0): the normalizer falls back
      to 1.0 and the margin is 0.0 by construction, so confidence = 0.0
      (`basis="tied_scores"`) -- a flat score distribution is treated as
      the most ambiguous case, consistent with Cell 47's stated
      candidate principle ("ambiguous / flat score distribution -> lower
      confidence -> larger k").

What confidence DOES mean:
    - a purely structural, deterministic summary of how much the
      top-ranked candidate stands out from the runner-up, relative to
      this query's own observed candidate-score spread.

What confidence DOES NOT mean (Cell 47 spec, Section 6):
    - it is NOT calibrated clinical certainty
    - it is NOT a probability that any diagnosis is correct
    - it is NOT a probability that a generated report will be medically
      correct
    - it says nothing about the retrieved text's factual content at all
      -- only about the numeric shape of the ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class ConfidenceResult:
    """One query's shared confidence result. See module docstring for the
    exact formula and what each field does/does not mean.

    Attributes:
        value: confidence in [0, 1], always defined (never None).
        candidate_count: len(scores) as passed in.
        top1_score: scores[0], or None iff candidate_count == 0.
        top2_score: scores[1], or None iff candidate_count < 2.
        margin: raw (unnormalized) top1 - top2, or None iff
            candidate_count < 2 -- distinct from `value`, which is the
            normalized, clamped confidence derived from this margin.
        spread: raw (unnormalized) max(scores) - min(scores), or None
            iff candidate_count == 0; 0.0 iff candidate_count == 1 (a
            single point has zero spread by definition).
        basis: short machine-readable label for which documented branch
            produced this result (see module docstring) -- always one
            of "empty_candidate_list", "single_candidate_no_margin",
            "tied_scores", or "margin_based".
    """

    value: float
    candidate_count: int
    top1_score: Optional[float]
    top2_score: Optional[float]
    margin: Optional[float]
    spread: Optional[float]
    basis: str


_VALID_BASES = (
    "empty_candidate_list",
    "single_candidate_no_margin",
    "tied_scores",
    "margin_based",
)


def compute_confidence(scores: Sequence[float]) -> ConfidenceResult:
    """Computes the shared confidence result for one query's ranked
    candidate scores. See module docstring for the full formula.

    Never raises -- every input shape (empty, single-candidate, tied,
    well-separated) has defined, documented behavior (Cell 47 spec,
    Section 12, tests 6-8: "confidence handles ties/empty/one
    candidate").

    Args:
        scores: candidate scores, already ranked descending by the
            caller (this function does not re-sort).
    """
    n = len(scores)
    if n == 0:
        return ConfidenceResult(
            value=0.0, candidate_count=0, top1_score=None, top2_score=None,
            margin=None, spread=None, basis="empty_candidate_list",
        )

    top1 = float(scores[0])
    if n == 1:
        return ConfidenceResult(
            value=0.0, candidate_count=1, top1_score=top1, top2_score=None,
            margin=None, spread=0.0, basis="single_candidate_no_margin",
        )

    top2 = float(scores[1])
    raw_margin = top1 - top2
    raw_spread = float(max(scores)) - float(min(scores))

    if raw_spread <= 0.0:
        return ConfidenceResult(
            value=0.0, candidate_count=n, top1_score=top1, top2_score=top2,
            margin=raw_margin, spread=raw_spread, basis="tied_scores",
        )

    normalized = raw_margin / raw_spread
    clamped = max(0.0, min(1.0, normalized))
    return ConfidenceResult(
        value=clamped, candidate_count=n, top1_score=top1, top2_score=top2,
        margin=raw_margin, spread=raw_spread, basis="margin_based",
    )
