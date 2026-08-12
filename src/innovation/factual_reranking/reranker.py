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
from typing import Any, List, Optional, Protocol, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Cell 48 (this revision): the real I2 algorithm -- MeSH-term consensus
# reranking, on top of the contract above. Reuses compute_final_score
# and RerankResult unchanged; adds no new fields to either. Generic
# over candidate identity (RerankCandidateInput.candidate_key), never
# depends on any specific dataset's record type directly -- the real
# IU X-Ray bridge lives in pipeline_adapter.py, mirroring
# adaptive_retrieval's own algorithm/pipeline-adapter separation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RerankCandidateInput:
    """One candidate's input to rerank_candidates -- generic, decoupled
    from any specific dataset's record type (see pipeline_adapter.py for
    the IU X-Ray-specific bridge).

    Attributes:
        candidate_key: opaque candidate identifier (e.g. QueryKey).
        retrieval_score: this candidate's original (pre-rerank)
            retrieval score.
        mesh_terms: this candidate's OWN MeSH terms (real, structured,
            extracted from that candidate's own report at Cell 43
            canonicalization time) -- never the query's or target's.
        findings: this candidate's own Findings text, or None.
        impression: this candidate's own Impression text, or None.
    """

    candidate_key: Any
    retrieval_score: float
    mesh_terms: Sequence[str]
    findings: Optional[str]
    impression: Optional[str]


def compute_mesh_term_consensus_score(
    candidate_mesh_terms: Sequence[str],
    other_candidates_mesh_terms: Sequence[Sequence[str]],
) -> Tuple[float, str]:
    """The primary I2 compatibility signal: Jaccard similarity between
    one candidate's MeSH terms and the union of MeSH terms across the
    OTHER candidates in the same selected set -- never the query's
    target report (no target leakage: this function does not accept
    and has never seen target-report content).

    Deterministic, pure set arithmetic, bounded [0, 1]. Edge cases
    (matching the required specification exactly):
        - `other_candidates_mesh_terms` empty (this is the only
          candidate in the set, e.g. K=1 from I1) -> 0.5, basis
          "k1_no_others": there is no consensus to compare against at
          all, distinct from a real disagreement.
        - candidate's own terms empty AND the union of others' terms
          also empty -> 0.5, basis "both_empty": no information either
          way, not evidence of disagreement.
        - exactly one side empty (candidate empty / consensus
          non-empty, or the reverse) -> 0.0, basis "one_side_empty":
          falls out of the Jaccard formula itself (intersection is
          always empty when one side is empty), not a special case.
        - both non-empty -> real Jaccard similarity, basis "jaccard".

    Returns:
        (score, basis) -- score in [0, 1]; basis is a short,
        machine-readable label naming which branch produced it (used
        by diagnostics to count neutral/edge cases without re-deriving
        them from raw mesh_terms).
    """
    if not other_candidates_mesh_terms:
        return 0.5, "k1_no_others"

    candidate_set = set(candidate_mesh_terms)
    others_union = set()
    for terms in other_candidates_mesh_terms:
        others_union.update(terms)

    if not candidate_set and not others_union:
        return 0.5, "both_empty"

    union = candidate_set | others_union
    intersection = candidate_set & others_union
    basis = "jaccard" if candidate_set and others_union else "one_side_empty"
    return len(intersection) / len(union), basis


def compute_evidence_quality_score(findings: Optional[str], impression: Optional[str]) -> float:
    """Purely structural evidence-quality score, computed only from a
    candidate's OWN report text -- no external model call, no target
    access:

        both findings and impression present (non-empty after
            stripping whitespace) -> 1.0
        exactly one present                                -> 0.5
        neither present                                    -> 0.0
    """
    has_findings = bool(findings and findings.strip())
    has_impression = bool(impression and impression.strip())
    if has_findings and has_impression:
        return 1.0
    if has_findings or has_impression:
        return 0.5
    return 0.0


def rerank_candidates(
    candidates: Sequence[RerankCandidateInput],
    config: RerankConfig,
) -> List[RerankResult]:
    """Reranks `candidates` using the shared Cell 47 scoring contract
    (compute_final_score) with the real MeSH-term consensus
    compatibility signal and the structural evidence-quality signal
    above.

    Preserves the candidate SET exactly -- returns exactly one
    RerankResult per input candidate, never adding or dropping any
    (Section requirement: "I2 may reorder/reweight candidates but must
    not introduce new candidates"). Deterministic: candidates are
    sorted by (-final_score, rank_before) -- ties in final_score are
    broken by original rank ascending, never randomly, so a caller
    re-running this with identical inputs always gets an identical
    order.

    With `config.beta == 0.0` and `config.gamma == 0.0` (RerankConfig's
    own defaults), final_score reduces to `config.alpha *
    retrieval_score` for every candidate -- sorting by that reproduces
    the original retrieval-score order exactly, so `rank_after ==
    rank_before` for every candidate whenever the input was already
    retrieval-score-ordered. This is why I2 disabled (beta=gamma=0) is
    baseline-equivalent by construction, not by a separate code path.
    """
    n = len(candidates)
    all_mesh_terms = [list(c.mesh_terms) for c in candidates]

    scored: List[RerankResult] = []
    for idx, candidate in enumerate(candidates):
        others_mesh_terms = [all_mesh_terms[j] for j in range(n) if j != idx]
        compatibility_score, basis = compute_mesh_term_consensus_score(
            candidate.mesh_terms, others_mesh_terms
        )
        evidence_quality_score = compute_evidence_quality_score(candidate.findings, candidate.impression)
        final_score = compute_final_score(
            config,
            retrieval_score=candidate.retrieval_score,
            compatibility_score=compatibility_score,
            evidence_quality_score=evidence_quality_score,
        )
        scored.append(
            RerankResult(
                candidate_key=candidate.candidate_key,
                retrieval_score=candidate.retrieval_score,
                compatibility_score=compatibility_score,
                evidence_quality_score=evidence_quality_score,
                final_score=final_score,
                rank_before=idx,
                rank_after=None,
                provenance=f"mesh_term_consensus_jaccard:{basis}",
            )
        )

    ordered = sorted(scored, key=lambda r: (-r.final_score, r.rank_before))
    return [
        RerankResult(
            candidate_key=r.candidate_key,
            retrieval_score=r.retrieval_score,
            compatibility_score=r.compatibility_score,
            evidence_quality_score=r.evidence_quality_score,
            final_score=r.final_score,
            rank_before=r.rank_before,
            rank_after=new_rank,
            provenance=r.provenance,
        )
        for new_rank, r in enumerate(ordered)
    ]
