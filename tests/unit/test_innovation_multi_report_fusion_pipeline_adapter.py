"""Unit tests for I3's pipeline adapter (pipeline_adapter.py) against
real-shaped IU X-Ray canonical records (src.data.iu_xray.records.
IuXrayCanonicalRecord) -- Cell 50, this revision.

Every record built here uses the exact real field names/types Cell 43
canonicalization produces (see records.py's CANONICAL_SCHEMA_FIELDS),
never a simplified ad hoc stand-in schema.
"""

from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.innovation.experiment_contract import InnovationConfig
from src.innovation.factual_reranking.reranker import RerankConfig
from src.innovation.multi_report_fusion.fusion import FusionConfig
from src.innovation.multi_report_fusion.pipeline_adapter import (
    select_baseline_equivalent_evidence,
    select_evidence_for_query,
    select_fused_evidence,
)

QUERY = ("iu_xray", "pQuery", "sQuery")
Q_A = ("iu_xray", "pA", "sA")
Q_B = ("iu_xray", "pB", "sB")
Q_C = ("iu_xray", "pC", "sC")


def _record(study_id, findings="Findings text.", impression="Impression text."):
    return IuXrayCanonicalRecord(
        dataset_name="iu_xray",
        study_id=study_id,
        report_id=study_id,
        image_ids=[f"{study_id}-img"],
        image_paths=[f"/data/{study_id}.png"],
        image_count=1,
        findings=findings,
        impression=impression,
        indication=None,
        comparison=None,
        full_report=f"{findings} {impression}".strip(),
        mesh_terms=[],
        labels=[],
        source_metadata={},
        source_split="train",
        validation_status="valid",
        exclusion_reason=None,
    )


def _records_by_key():
    return {
        Q_A: _record("sA", "Findings A.", "Impression A."),
        Q_B: _record("sB", "Findings B.", "Impression B."),
        Q_C: _record("sC", "Findings C.", "Impression C."),
    }


# ---------------------------------------------------------------------------
# (11) candidate provenance preservation
# ---------------------------------------------------------------------------


def test_fused_evidence_preserves_candidate_id_rank_findings_impression():
    # Low-confidence scores -> fuse all three.
    # top1=5.0, top2=4.9, spread=5.0-3.0=2.0 -> confidence=0.05 -> low
    # band -> desired=3, satisfied by all 3 available candidates.
    result = select_fused_evidence(
        QUERY, [Q_A, Q_B, Q_C], [5.0, 4.9, 3.0], _records_by_key(), FusionConfig(),
    )
    assert result.fusion_used is True
    assert result.evidence_context is not None
    assert len(result.evidence_context.items) == 3
    for rank, (key, item) in enumerate(zip([Q_A, Q_B, Q_C], result.evidence_context.items)):
        assert item.source_key == key
        assert f"Evidence {rank + 1}" in item.text
        assert key[2] in item.text  # study_id
        assert "Findings" in item.text
        assert "Impression" in item.text


def test_fused_evidence_missing_record_degrades_to_na_not_exception():
    partial_records = {Q_A: _record("sA")}  # Q_B/Q_C missing
    result = select_fused_evidence(
        QUERY, [Q_A, Q_B, Q_C], [5.0, 4.9, 3.0], partial_records, FusionConfig(),
    )
    assert len(result.evidence_context.items) == 3
    missing_item = result.evidence_context.items[1]
    assert "N/A" in missing_item.text
    assert "sB" in missing_item.text  # study_id falls back to the QueryKey itself


# ---------------------------------------------------------------------------
# (12) disabled I3 baseline equivalence
# ---------------------------------------------------------------------------


def test_baseline_equivalent_path_selects_only_top1_no_context_built():
    result = select_baseline_equivalent_evidence(QUERY, [Q_A, Q_B, Q_C], [5.0, 4.9, 4.8])
    assert result.fusion_used is False
    assert result.decision is None
    assert result.evidence_context is None
    assert result.selected_candidates == (Q_A,)


def test_select_evidence_for_query_disabled_is_baseline_equivalent():
    config = InnovationConfig(confidence_gated_fusion_enabled=False)
    result = select_evidence_for_query(QUERY, [Q_A, Q_B, Q_C], [5.0, 4.9, 4.8], _records_by_key(), config)
    assert result.fusion_used is False
    assert result.selected_candidates == (Q_A,)
    assert result.evidence_context is None


def test_default_innovation_config_is_e0_baseline_equivalent_for_fusion():
    config = InnovationConfig()
    result = select_evidence_for_query(QUERY, [Q_A, Q_B, Q_C], [5.0, 4.9, 4.8], _records_by_key(), config)
    assert result.fusion_used is False
    assert result.selected_candidates == (Q_A,)


# ---------------------------------------------------------------------------
# (13) interaction with I1 K=1
# ---------------------------------------------------------------------------


def test_fusion_enabled_with_only_one_i1_selected_candidate_fuses_one():
    # Simulates I1 having already narrowed the candidate set to K=1.
    config = InnovationConfig(confidence_gated_fusion_enabled=True, fusion=FusionConfig())
    result = select_evidence_for_query(QUERY, [Q_A], [5.0], _records_by_key(), config)
    assert result.fusion_used is True
    assert result.selected_candidates == (Q_A,)
    assert result.decision.selected_evidence_count == 1
    assert result.decision.should_fuse is False
    assert len(result.evidence_context.items) == 1


# ---------------------------------------------------------------------------
# (14) interaction with I2 reordered candidates
# ---------------------------------------------------------------------------


def test_fusion_respects_i2_reordered_candidate_sequence():
    # Simulate I2 having reordered candidates to [Q_C, Q_A, Q_B] (not
    # the original retrieval order) with I2's own final_score attached.
    reordered_keys = [Q_C, Q_A, Q_B]
    reordered_scores = [9.0, 8.9, 8.8]  # low-confidence gap -> fuse multiple
    config = InnovationConfig(confidence_gated_fusion_enabled=True, fusion=FusionConfig())
    result = select_evidence_for_query(QUERY, reordered_keys, reordered_scores, _records_by_key(), config)
    assert result.fusion_used is True
    # Selection must be a PREFIX of the given (I2-reordered) order, not
    # re-sorted back to Q_A/Q_B/Q_C.
    assert result.selected_candidates[0] == Q_C
    assert list(result.selected_candidates) == reordered_keys[: len(result.selected_candidates)]


# ---------------------------------------------------------------------------
# (15) no candidate resurrection
# ---------------------------------------------------------------------------


def test_fusion_never_selects_a_candidate_outside_the_input_set():
    # Only 2 candidates supplied (e.g. I1 discarded the third), tied
    # scores -> low confidence -> desired=3, but availability caps it.
    result = select_fused_evidence(
        QUERY, [Q_A, Q_B], [5.0, 5.0], _records_by_key(), FusionConfig(),
    )
    assert set(result.selected_candidates).issubset({Q_A, Q_B})
    assert Q_C not in result.selected_candidates
    assert result.decision.selected_evidence_count <= 2
    assert "capped to" in result.decision.reason
