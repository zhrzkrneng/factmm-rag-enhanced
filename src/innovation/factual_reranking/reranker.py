"""Factual/clinical reranking contract (Cell 47, Milestone 3.0, Section 4).

Responsibility (Cell 47 scope -- contract only): the configuration and
per-candidate result dataclasses for I2 (Factual/Clinical Reranking).
The actual reranking algorithm (a function that scores and reorders a
candidate list) is intentionally NOT implemented here -- Cell 47's spec
Section 10 defers "the full... reranker... algorithm[]" to Cells 48-50.

Signal audit (Section 4: "First inspect what signals are actually
available... If no reliable structured clinical signal currently
exists, implement a pluggable scoring interface and a deterministic
lightweight fallback instead of pretending that a clinical model
exists"):

    Inspected: src/baseline/radgraph/{annotator,mock_annotator}.py
    provide RadGraph/CheXbert-style annotation, but exclusively for
    training-time pair mining (src/baseline/pair_mining/), never wired
    to any inference-time candidate-scoring path. No structured
    clinical compatibility signal is currently available at reranking
    time without either (a) invoking a real clinical annotation model
    per candidate at inference (an expensive-GPU-inference operation
    explicitly forbidden by Cell 47) or (b) reaching for the query's
    own target report as a comparison signal (explicitly forbidden --
    that would be target-report leakage, Section 4's "no access to the
    ground-truth target report during reranking").

    Conclusion: CompatibilityScorer below is a pluggable Protocol, and
    `neutral_fallback_compatibility_scorer` is the deterministic
    lightweight fallback Section 4 calls for. It returns a constant
    0.5 ("no information either way") for every candidate, never
    inventing a clinical judgment. Because RerankConfig.beta defaults
    to 0.0, this fallback has zero effect on final_score unless a
    caller explicitly opts in by raising beta -- at which point they
    are expected to supply a real scorer (Cell 48/49's job, once a
    genuine structured signal is confirmed available) via the pluggable
    interface, not rely on the neutral placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from src.baseline.pair_mining.mining import QueryKey


@dataclass(frozen=True)
class RerankConfig:
    """I2's configurable weights for:

        final_score = alpha * retrieval_relevance
                     + beta * factual_or_clinical_compatibility
                     + gamma * evidence_quality

    (Section 4's composite-score form.) All three weights are
    independently configurable; `gamma` is optional (defaults to 0.0,
    i.e. evidence_quality is not used unless a caller opts in).

    Attributes:
        alpha: weight on the original retrieval relevance score. Must
            be >= 0.0.
        beta: weight on factual/clinical compatibility. Must be >= 0.0.
        gamma: weight on evidence quality (optional signal). Must be
            >= 0.0.

    Validated eagerly (Section 12, test 10: "invalid reranking weights
    rejected"): all weights must be non-negative, and at least one must
    be strictly positive (an all-zero weight set could never produce a
    meaningful final_score ordering).
    """

    alpha: float = 1.0
    beta: float = 0.0
    gamma: float = 0.0

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0.0:
                raise ValueError(f"{name} must be a non-negative number, got {value!r}")
        if self.alpha + self.beta + self.gamma <= 0.0:
            raise ValueError(
                "at least one of alpha/beta/gamma must be > 0.0 -- an "
                "all-zero weight set cannot produce a meaningful ranking"
            )


class CompatibilityScorer(Protocol):
    """Pluggable factual/clinical compatibility scoring interface.

    A real implementation (Cell 48/49, once a genuine structured signal
    is confirmed available and is safe to compute without target-report
    access) takes a retrieved candidate's own text and returns a score
    in [0, 1]. Must never be given the query's target report.
    """

    def __call__(self, candidate_text: str) -> float:
        ...


def neutral_fallback_compatibility_scorer(candidate_text: str) -> float:
    """Deterministic lightweight fallback (Section 4): returns a
    constant 0.5 for every candidate, regardless of its text. See
    module docstring's signal audit for why this exists instead of a
    fabricated clinical heuristic."""
    return 0.5


@dataclass(frozen=True)
class RerankResult:
    """One candidate's auditable reranking result (Section 4: "expose
    component scores"..."Return an auditable result per candidate").

    Attributes:
        candidate_key: the reranked candidate's QueryKey.
        retrieval_score: the original (pre-rerank) retrieval score,
            preserved verbatim for provenance.
        compatibility_score: the factual/clinical compatibility score
            in [0, 1] (from a CompatibilityScorer -- real or the
            neutral fallback).
        evidence_quality_score: the optional evidence-quality score in
            [0, 1], or None if not computed (gamma == 0.0 callers may
            omit this entirely).
        final_score: alpha*retrieval_score + beta*compatibility_score +
            gamma*(evidence_quality_score or 0.0) -- the value ranking
            is actually sorted by.
        rank_before: the candidate's 0-indexed rank before reranking.
        rank_after: the candidate's 0-indexed rank after reranking, or
            None if not yet assigned (e.g. a single candidate's score
            was computed but the full list has not been re-sorted yet).
        provenance: which compatibility scorer produced
            compatibility_score (e.g. "neutral_fallback_compatibility_scorer"
            vs. a real scorer's own identifying name) -- always present,
            never silently omitted.
    """

    candidate_key: QueryKey
    retrieval_score: float
    compatibility_score: float
    evidence_quality_score: Optional[float]
    final_score: float
    rank_before: int
    rank_after: Optional[int]
    provenance: str


def compute_final_score(
    config: RerankConfig,
    *,
    retrieval_score: float,
    compatibility_score: float,
    evidence_quality_score: Optional[float] = None,
) -> float:
    """Applies RerankConfig's weighted-sum formula. Pure, deterministic,
    stable (no tie-breaking randomness) -- ties in final_score are the
    caller's responsibility to break by a stable secondary key (e.g.
    original rank_before), per Section 4's "stable tie-breaking"
    requirement; this function only computes the score, it does not
    sort."""
    evidence_component = evidence_quality_score if evidence_quality_score is not None else 0.0
    return (
        config.alpha * retrieval_score
        + config.beta * compatibility_score
        + config.gamma * evidence_component
    )
