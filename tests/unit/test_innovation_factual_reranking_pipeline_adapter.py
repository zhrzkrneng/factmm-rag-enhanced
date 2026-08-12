"""Unit tests for I2's pipeline adapter (pipeline_adapter.py) against
real-shaped IU X-Ray canonical records (src.data.iu_xray.records.
IuXrayCanonicalRecord) -- Cell 48, this revision.

Every record built here uses the exact real field names/types Cell 43
canonicalization produces (see records.py's CANONICAL_SCHEMA_FIELDS),
never a simplified ad hoc stand-in schema.
"""

from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.innovation.experiment_contract import InnovationConfig
from src.innovation.factual_reranking.pipeline_adapter import (
    select_baseline_equivalent_ranking,
    select_evidence_for_query,
    select_reranked_evidence,
)
from src.innovation.factual_reranking.reranker import RerankConfig

Q_A = ("iu_xray", "pA", "sA")
Q_B = ("iu_xray", "pB", "sB")
Q_C = ("iu_xray", "pC", "sC")
QUERY = ("iu_xray", "pQuery", "sQuery")


def _record(study_id, mesh_terms, findings="Findings.", impression="Impression."):
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
        mesh_terms=mesh_terms,
        labels=[],
        source_metadata={},
        source_split="train",
        validation_status="valid",
        exclusion_reason=None,
    )


def _records_by_key():
    return {
        Q_A: _record("sA", ["cardiomegaly", "effusion"]),
        Q_B: _record("sB", ["cardiomegaly"]),
        Q_C: _record("sC", []),
    }


def test_pipeline_adapter_reranks_real_iu_xray_records():
    result = select_reranked_evidence(
        QUERY,
        [Q_A, Q_B, Q_C],
        [1.0, 2.0, 3.0],
        _records_by_key(),
        RerankConfig(alpha=1.0, beta=5.0),
    )
    assert result.reranking_used is True
    assert set(result.selected_candidates) == {Q_A, Q_B, Q_C}
    assert len(result.results) == 3
    # A and B share "cardiomegaly" with each other; C has no mesh
    # terms at all -- A/B should outrank C despite C's higher raw
    # retrieval_score.
    assert result.selected_candidates[-1] == Q_C


def test_pipeline_adapter_missing_record_degrades_gracefully():
    partial_records = {Q_A: _record("sA", ["x"])}  # Q_B/Q_C missing
    result = select_reranked_evidence(
        QUERY, [Q_A, Q_B, Q_C], [3.0, 2.0, 1.0], partial_records, RerankConfig(beta=1.0),
    )
    assert set(result.selected_candidates) == {Q_A, Q_B, Q_C}


def test_baseline_equivalent_ranking_preserves_order_and_identity():
    result = select_baseline_equivalent_ranking(QUERY, [Q_A, Q_B, Q_C], [1.0, 2.0, 3.0])
    assert result.reranking_used is False
    assert result.results is None
    assert result.selected_candidates == (Q_A, Q_B, Q_C)


def test_select_evidence_for_query_disabled_is_baseline_equivalent():
    config = InnovationConfig(factual_reranking_enabled=False)
    result = select_evidence_for_query(QUERY, [Q_A, Q_B, Q_C], [3.0, 2.0, 1.0], _records_by_key(), config)
    assert result.reranking_used is False
    assert result.selected_candidates == (Q_A, Q_B, Q_C)


def test_select_evidence_for_query_enabled_uses_reranking():
    config = InnovationConfig(
        factual_reranking_enabled=True,
        reranking=RerankConfig(alpha=1.0, beta=5.0),
    )
    result = select_evidence_for_query(QUERY, [Q_A, Q_B, Q_C], [1.0, 2.0, 3.0], _records_by_key(), config)
    assert result.reranking_used is True
    assert set(result.selected_candidates) == {Q_A, Q_B, Q_C}


def test_default_innovation_config_is_e0_baseline_equivalent():
    # InnovationConfig() with no args -- E0 -- must be exactly
    # baseline-equivalent for I2, matching I1's own E0 guarantee.
    config = InnovationConfig()
    result = select_evidence_for_query(QUERY, [Q_A, Q_B, Q_C], [3.0, 2.0, 1.0], _records_by_key(), config)
    assert result.reranking_used is False
    assert result.selected_candidates == (Q_A, Q_B, Q_C)


def test_pipeline_adapter_never_receives_target_report():
    # This test's own construction is the guarantee: `QUERY`'s
    # canonical record (if it even existed) is never looked up or
    # passed anywhere in these calls -- only the CANDIDATES' records
    # (Q_A/Q_B/Q_C) are ever supplied. records_by_key intentionally
    # never contains an entry for QUERY itself.
    records = _records_by_key()
    assert QUERY not in records
    result = select_reranked_evidence(QUERY, [Q_A, Q_B], [1.0, 2.0], records, RerankConfig(beta=1.0))
    assert result.reranking_used is True
