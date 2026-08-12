"""Confidence-gated multi-report fusion contract (Cell 47, Milestone 3.0,
Section 5).

Responsibility (Cell 47 scope -- contract only): the configuration,
gating-decision, and structured-evidence-context dataclasses for I3
(Confidence-Gated Multi-Report Fusion). The actual fusion algorithm (a
function that decides should_fuse and assembles an EvidenceContext from
real candidates) is intentionally NOT implemented here -- Cell 47's spec
Section 10 defers "the full... fusion algorithm[]" to Cells 48-50.

Expected behavior (Section 5, to be implemented by Cell 50 against
src.innovation.shared.confidence.compute_confidence's output -- not
hidden in undocumented code):

    HIGH confidence (large normalized top1-vs-top2 margin) -> use
    minimal evidence (should_fuse=False, or a small
    selected_evidence_count).

    LOW confidence / ambiguous retrieval (small or zero margin) ->
    permit multi-report fusion (should_fuse=True, up to
    FusionConfig.max_evidence_count).

Note: this narrows/refines src/innovation/confidence_gating/gate.py's
original, broader "Innovation D: retrieve at all, yes/no" scope (see
cell47_existing_innovation_audit.json) into the specific gate Section 5
actually asks for -- a should_fuse decision over already-retrieved
evidence, not a retrieve-vs-not-retrieve decision. gate.py itself is
left untouched by Cell 47 (classified UNUSED for this cell -- its
broader scope is not one of the three required innovations, and Section
5's contract is fully expressed here instead, to avoid two files
partially implementing the same decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.innovation.shared.confidence import compute_confidence


@dataclass(frozen=True)
class FusionConfig:
    """I3's configuration.

    Attributes:
        max_evidence_count: the largest number of retrieved reports
            fusion may ever combine. Must be >= 1.
        context_budget_chars: the evidence context's total character
            budget (Section 5: "enforcing a context/evidence budget").
            Must be >= 1.
        confidence_threshold: the shared-confidence value (Section 6)
            at or above which a future fusion algorithm should prefer
            NOT fusing (i.e. use minimal evidence). Must be in [0, 1].
        disagreement_threshold: the disagreement/dispersion value at or
            above which a future fusion algorithm should prefer fusing
            (candidates disagree enough that more evidence is useful).
            Must be in [0, 1]. Reserved for a future secondary signal --
            not used by Cell 50's three-band policy below (mirrors
            AdaptiveKConfig.margin_threshold's own "reserved, not used
            by the three-band policy" precedent).
        high_confidence_threshold: the shared-confidence value (Section
            6) at or above which Cell 50's band policy applies "high"
            (desired_reports=1, i.e. minimal evidence). Must be in
            [0, 1].
        low_confidence_threshold: the confidence value below which the
            band policy applies "low" (desired_reports=3, capped by
            max_evidence_count); confidence in
            [low_confidence_threshold, high_confidence_threshold) is
            "medium" (desired_reports=2). Must be in [0, 1] and
            strictly less than high_confidence_threshold, so the three
            bands are always non-overlapping and together cover all of
            [0, 1] -- mirrors AdaptiveKConfig's identical band-threshold
            contract exactly (see adaptive_k.py).

    Validated eagerly (Section 12, test 12: "invalid fusion budget
    rejected").
    """

    max_evidence_count: int = 3
    context_budget_chars: int = 4000
    confidence_threshold: float = 0.5
    disagreement_threshold: float = 0.3
    high_confidence_threshold: float = 0.66
    low_confidence_threshold: float = 0.33

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_evidence_count, int)
            or isinstance(self.max_evidence_count, bool)
            or self.max_evidence_count < 1
        ):
            raise ValueError(
                f"max_evidence_count must be a positive integer, got {self.max_evidence_count!r}"
            )
        if (
            not isinstance(self.context_budget_chars, int)
            or isinstance(self.context_budget_chars, bool)
            or self.context_budget_chars < 1
        ):
            raise ValueError(
                f"context_budget_chars must be a positive integer, got {self.context_budget_chars!r}"
            )
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold!r}"
            )
        if not (0.0 <= self.disagreement_threshold <= 1.0):
            raise ValueError(
                f"disagreement_threshold must be in [0.0, 1.0], got {self.disagreement_threshold!r}"
            )
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


class FusionDecision(NamedTuple):
    """One query's confidence-gated fusion decision (produced by Cell
    50's fusion algorithm, not by this module).

    Attributes:
        should_fuse: whether multiple retrieved reports should be used
            together for this query.
        selected_evidence_count: how many reports were selected (1 iff
            should_fuse is False; up to FusionConfig.max_evidence_count
            iff True).
        confidence: the shared-confidence value (Section 6) this
            decision was based on, in [0, 1].
        confidence_band: one of "high"/"medium"/"low" (Cell 50's
            three-band policy -- see FusionConfig.high_confidence_threshold
            / low_confidence_threshold).
        disagreement_score: a measure of disagreement/dispersion among
            retrieved evidence, or None if not computed by this
            decision. Cell 50's policy is confidence-band-only and
            deliberately does not invent a second, independent
            disagreement formula (Section 6: reuse
            compute_confidence, do not create a second confidence-like
            signal) -- always None here.
        available_candidate_count: how many candidates were passed in
            (after any upstream I1/I2 filtering/reordering) -- reported
            as its own field (mirroring AdaptiveKDecision's identical
            field) so diagnostics can measure I3's interaction with
            I1-selected K without parsing `reason`.
        reason: a short, human-readable explanation of the decision
            (for per-query auditability).
    """

    should_fuse: bool
    selected_evidence_count: int
    confidence: float
    confidence_band: str
    disagreement_score: Optional[float]
    available_candidate_count: int
    reason: str


@dataclass(frozen=True)
class EvidenceItem:
    """One retrieved report's contribution to a structured evidence
    context (Section 5: "preserving report boundaries, preserving
    provenance").

    Attributes:
        source_key: the retrieved candidate's QueryKey -- never the
            query's own key, and never the target report (Section 5:
            "never including the target report").
        text: this candidate's own text, verbatim (never mutated).
        truncated: whether this item's text was truncated to fit the
            context budget -- if True, the truncation is being recorded
            here, not silently applied (Section 5: "never silently
            truncating without recording it").
    """

    source_key: QueryKey
    text: str
    truncated: bool


@dataclass(frozen=True)
class EvidenceContext:
    """The structured, budget-enforced evidence context assembled by
    fusion (Section 5).

    Attributes:
        items: the fused evidence, in selection order, one item per
            distinct source (Section 5: "avoiding duplicate evidence" --
            no two items may share a source_key).
        total_chars: sum of len(item.text) for item in items, after any
            truncation.
        budget_chars: the FusionConfig.context_budget_chars this
            context was built against.
    """

    items: Tuple[EvidenceItem, ...]
    total_chars: int
    budget_chars: int

    def __post_init__(self) -> None:
        source_keys = [item.source_key for item in self.items]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError(
                f"EvidenceContext must not contain duplicate source_key "
                f"entries, got {source_keys!r}"
            )
        if self.total_chars > self.budget_chars:
            raise ValueError(
                f"EvidenceContext.total_chars ({self.total_chars}) must not "
                f"exceed budget_chars ({self.budget_chars}) -- the budget "
                f"must be enforced before construction, not after"
            )


# ---------------------------------------------------------------------------
# Cell 50 (this revision): the real I3 algorithm -- confidence-gated
# multi-report fusion, on top of the contract above. Reuses
# FusionDecision/EvidenceContext/EvidenceItem unchanged (only
# FusionDecision gained the new confidence_band field, added above).
# Reuses src.innovation.shared.confidence.compute_confidence verbatim
# -- no second confidence formula is invented anywhere in this module.
# Generic over candidate identity (FusionCandidateInput.candidate_key),
# never depends on any specific dataset's record type directly -- the
# real IU X-Ray bridge lives in pipeline_adapter.py, mirroring
# adaptive_retrieval's and factual_reranking's own
# algorithm/pipeline-adapter separation.
# ---------------------------------------------------------------------------

BAND_HIGH = "high"
BAND_MEDIUM = "medium"
BAND_LOW = "low"
VALID_BANDS = (BAND_HIGH, BAND_MEDIUM, BAND_LOW)

# Hardcoded per Cell 50's specified initial fusion policy: high
# confidence -> 1 report (minimal evidence), medium -> 2, low -> 3 --
# always bounded by FusionConfig.max_evidence_count (see
# _desired_reports_for_band) and always further capped by however many
# candidates are actually available (see select_fusion_candidates).
_DESIRED_REPORTS_BY_BAND = {BAND_HIGH: 1, BAND_MEDIUM: 2, BAND_LOW: 3}


@dataclass(frozen=True)
class FusionCandidateInput:
    """One candidate's input to select_fusion_candidates -- generic,
    decoupled from any specific dataset's record type (see
    pipeline_adapter.py for the IU X-Ray-specific bridge).

    Attributes:
        candidate_key: opaque candidate identifier (e.g. QueryKey).
        score: this candidate's ranking score in the CURRENT candidate
            ordering -- I2's final_score when I2 ran before I3, or the
            raw retrieval score otherwise (the caller/pipeline_adapter
            decides which, matching whichever ordering
            `candidate_keys` is already sorted by). Used only to
            compute the shared confidence signal; select_fusion_candidates
            never re-sorts candidates by this score -- fusion always
            selects a PREFIX of the order it is given, never reorders
            it (Section 5: fusion is evidence-context assembly, not a
            second reranking step).
    """

    candidate_key: Any
    score: float


def _classify_band(confidence: float, config: FusionConfig) -> str:
    if confidence >= config.high_confidence_threshold:
        return BAND_HIGH
    if confidence < config.low_confidence_threshold:
        return BAND_LOW
    return BAND_MEDIUM


def _desired_reports_for_band(band: str, config: FusionConfig) -> int:
    return min(_DESIRED_REPORTS_BY_BAND[band], config.max_evidence_count)


def select_fusion_candidates(
    candidates: Sequence[FusionCandidateInput],
    config: FusionConfig,
) -> FusionDecision:
    """Deterministic confidence-gated fusion decision over an
    already-ranked candidate list (already-ranked by whatever upstream
    stage produced it -- raw retrieval, I1 Dynamic-K, or I2 Reranking).

    Formula (Section 5/6, Cell 50's initial policy):
        1. Reuse src.innovation.shared.confidence.compute_confidence on
           the candidates' own scores, in the order given -- this
           module never computes its own confidence signal.
        2. Classify confidence.value into "high"/"medium"/"low" via
           FusionConfig's two thresholds (mirrors AdaptiveKConfig's
           identical band contract).
        3. Map each band to a desired report count (1/2/3 by default,
           see _DESIRED_REPORTS_BY_BAND), bounded by
           config.max_evidence_count.
        4. Cap the desired count at how many candidates are actually
           available: effective_fusion_count =
           min(desired_reports, available_candidate_count). This is a
           hard structural invariant -- I3 must NEVER invent or
           recover candidates an earlier stage (I1) discarded. If only
           one candidate is available, I3 gracefully fuses one
           candidate only (should_fuse=False).

    Does not build evidence text and does not touch candidate content
    -- see build_evidence_context for that. Generic over candidate
    identity; never sees a target report.
    """
    available = len(candidates)
    scores = [c.score for c in candidates]
    confidence_result = compute_confidence(scores)
    band = _classify_band(confidence_result.value, config)
    desired = _desired_reports_for_band(band, config)
    selected_count = min(desired, available)

    if selected_count < desired:
        reason = (
            f"confidence={confidence_result.value:.4f} (band={band!r}) -> desired="
            f"{desired}, capped to {selected_count} because only {available} "
            f"candidate(s) were available (never resurrected/fabricated)"
        )
    else:
        reason = (
            f"confidence={confidence_result.value:.4f} (band={band!r}) -> selected="
            f"{selected_count}"
        )

    return FusionDecision(
        should_fuse=selected_count > 1,
        selected_evidence_count=selected_count,
        confidence=confidence_result.value,
        confidence_band=band,
        disagreement_score=None,
        available_candidate_count=available,
        reason=reason,
    )


def build_evidence_context(
    selected_items: Sequence[Tuple[Any, str]],
    config: FusionConfig,
) -> EvidenceContext:
    """Assembles a budget-enforced EvidenceContext from already-selected
    (source_key, text) pairs, in the exact order given (Section 5:
    "preserving report boundaries" -- one EvidenceItem per source,
    never merged/concatenated into a single fabricated block).

    Budget enforcement (Section 5: "enforcing a context/evidence
    budget", "never silently truncating without recording it"):
    `config.context_budget_chars` is divided evenly across the
    selected items (integer floor division); any item whose text
    exceeds its own share is truncated to exactly that share and
    marked `truncated=True`. This guarantees
    `total_chars <= budget_chars` by construction (sum of n shares of
    floor(budget/n) never exceeds budget), so EvidenceContext's own
    invariant check never fails here.

    Never invents evidence text and never merges two candidates'
    text into one item -- each source keeps its own EvidenceItem,
    deliberately avoiding "silently merging contradictory statements
    into fabricated consensus" (Section 5).
    """
    n = len(selected_items)
    if n == 0:
        return EvidenceContext(items=(), total_chars=0, budget_chars=config.context_budget_chars)

    per_item_budget = config.context_budget_chars // n
    items: List[EvidenceItem] = []
    for source_key, text in selected_items:
        if len(text) > per_item_budget:
            items.append(EvidenceItem(source_key=source_key, text=text[:per_item_budget], truncated=True))
        else:
            items.append(EvidenceItem(source_key=source_key, text=text, truncated=False))

    total_chars = sum(len(item.text) for item in items)
    return EvidenceContext(items=tuple(items), total_chars=total_chars, budget_chars=config.context_budget_chars)
