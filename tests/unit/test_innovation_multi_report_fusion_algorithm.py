"""Unit tests for the real I3 algorithm (fusion.py's
select_fusion_candidates / build_evidence_context) -- Cell 50, this
revision.

Every test builds FusionCandidateInput from bare (key, score) pairs
only -- no test constructs or passes anything resembling a query's
target report, matching the "I3 never sees the target report" design
principle already established for I1/I2.
"""

from src.innovation.multi_report_fusion.fusion import (
    BAND_HIGH,
    BAND_LOW,
    BAND_MEDIUM,
    EvidenceContext,
    EvidenceItem,
    FusionCandidateInput,
    FusionConfig,
    build_evidence_context,
    select_fusion_candidates,
)


def _candidates(scores):
    return [FusionCandidateInput(candidate_key=f"c{i}", score=s) for i, s in enumerate(scores)]


# ---------------------------------------------------------------------------
# (2) high-confidence -> one report
# ---------------------------------------------------------------------------


def test_high_confidence_selects_one_report():
    # top1=10, top2=1, spread=10 -> confidence=0.9 -> high band.
    candidates = _candidates([10.0, 1.0, 0.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.confidence_band == BAND_HIGH
    assert decision.selected_evidence_count == 1
    assert decision.should_fuse is False


# ---------------------------------------------------------------------------
# (3) medium-confidence -> two reports
# ---------------------------------------------------------------------------


def test_medium_confidence_selects_two_reports():
    # top1=5, top2=3, spread=5 -> confidence=0.4 -> medium band.
    candidates = _candidates([5.0, 3.0, 0.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.confidence_band == BAND_MEDIUM
    assert decision.selected_evidence_count == 2
    assert decision.should_fuse is True


# ---------------------------------------------------------------------------
# (4) low-confidence -> three reports
# ---------------------------------------------------------------------------


def test_low_confidence_selects_three_reports():
    # top1=5, top2=4.5, spread=3 -> confidence=0.1667 -> low band.
    candidates = _candidates([5.0, 4.5, 4.0, 3.0, 2.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.confidence_band == BAND_LOW
    assert decision.selected_evidence_count == 3
    assert decision.should_fuse is True


# ---------------------------------------------------------------------------
# (5) candidate shortage
# ---------------------------------------------------------------------------


def test_candidate_shortage_caps_below_desired_count():
    # Tied scores -> confidence=0.0 -> low band (desired=3), but only 2
    # candidates exist -> capped.
    candidates = _candidates([5.0, 5.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.confidence_band == BAND_LOW
    assert decision.selected_evidence_count == 2  # desired=3, capped to 2
    assert "capped to" in decision.reason
    assert decision.available_candidate_count == 2


# ---------------------------------------------------------------------------
# (6) empty candidate input
# ---------------------------------------------------------------------------


def test_empty_candidate_input_selects_zero_reports():
    decision = select_fusion_candidates([], FusionConfig())
    assert decision.selected_evidence_count == 0
    assert decision.should_fuse is False
    assert decision.available_candidate_count == 0
    assert decision.confidence == 0.0


def test_build_evidence_context_handles_empty_selection():
    ctx = build_evidence_context([], FusionConfig())
    assert ctx.items == ()
    assert ctx.total_chars == 0


# ---------------------------------------------------------------------------
# (7) one candidate
# ---------------------------------------------------------------------------


def test_single_candidate_gracefully_fuses_one_report_only():
    candidates = _candidates([7.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.selected_evidence_count == 1
    assert decision.should_fuse is False
    assert decision.available_candidate_count == 1


# ---------------------------------------------------------------------------
# (8) tied-score confidence behavior
# ---------------------------------------------------------------------------


def test_tied_scores_treated_as_low_confidence_and_fuse_all_available():
    candidates = _candidates([5.0, 5.0, 5.0])
    decision = select_fusion_candidates(candidates, FusionConfig())
    assert decision.confidence == 0.0
    assert decision.confidence_band == BAND_LOW
    assert decision.selected_evidence_count == 3
    assert decision.should_fuse is True


# ---------------------------------------------------------------------------
# (9) deterministic output
# ---------------------------------------------------------------------------


def test_deterministic_repeated_execution():
    candidates = _candidates([5.0, 3.0, 1.0])
    config = FusionConfig()
    first = select_fusion_candidates(candidates, config)
    second = select_fusion_candidates(candidates, config)
    assert first == second


def test_build_evidence_context_deterministic_repeated_execution():
    items = (("k1", "some text"), ("k2", "other text"))
    config = FusionConfig()
    first = build_evidence_context(items, config)
    second = build_evidence_context(items, config)
    assert first == second


# ---------------------------------------------------------------------------
# (10) candidate order preservation
# ---------------------------------------------------------------------------


def test_build_evidence_context_preserves_input_order():
    items = (("c0", "first"), ("c1", "second"), ("c2", "third"))
    ctx = build_evidence_context(items, FusionConfig())
    assert [item.source_key for item in ctx.items] == ["c0", "c1", "c2"]


def test_select_fusion_candidates_never_reorders_scores_only_selects_prefix():
    # Even though c1's score is higher than c0's, select_fusion_candidates
    # must not resort -- confidence is computed from the given order,
    # and the caller (pipeline_adapter) is responsible for selecting a
    # prefix of `candidates` in that same order.
    candidates = [
        FusionCandidateInput(candidate_key="c0", score=1.0),
        FusionCandidateInput(candidate_key="c1", score=99.0),
    ]
    decision = select_fusion_candidates(candidates, FusionConfig())
    # Confidence is computed from scores[0] vs scores[1] as given
    # (1.0 vs 99.0), not from a re-sorted order.
    assert decision.confidence == 0.0  # top1(1.0) < top2(99.0) -> margin clamped to 0


# ---------------------------------------------------------------------------
# Evidence-context budget enforcement
# ---------------------------------------------------------------------------


def test_build_evidence_context_truncates_and_records_when_over_budget():
    items = (("c0", "x" * 100), ("c1", "y" * 100))
    ctx = build_evidence_context(items, FusionConfig(context_budget_chars=50))
    assert ctx.total_chars <= 50
    assert all(item.truncated for item in ctx.items)


def test_build_evidence_context_does_not_truncate_when_under_budget():
    items = (("c0", "short"), ("c1", "text"))
    ctx = build_evidence_context(items, FusionConfig(context_budget_chars=4000))
    assert not any(item.truncated for item in ctx.items)
    assert ctx.total_chars == len("short") + len("text")
