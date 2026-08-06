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
from typing import NamedTuple, Optional, Tuple

from src.baseline.pair_mining.mining import QueryKey


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
            Must be in [0, 1].

    Validated eagerly (Section 12, test 12: "invalid fusion budget
    rejected").
    """

    max_evidence_count: int = 3
    context_budget_chars: int = 4000
    confidence_threshold: float = 0.5
    disagreement_threshold: float = 0.3

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
        disagreement_score: a measure of disagreement/dispersion among
            retrieved evidence, or None if not computed by this
            decision (e.g. when should_fuse is trivially False from
            confidence alone).
        reason: a short, human-readable explanation of the decision
            (for per-query auditability).
    """

    should_fuse: bool
    selected_evidence_count: int
    confidence: float
    disagreement_score: Optional[float]
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
