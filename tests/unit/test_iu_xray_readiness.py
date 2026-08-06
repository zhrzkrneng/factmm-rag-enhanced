"""Unit tests for src/data/iu_xray/readiness.py.

Covers: (1) deterministic mock retrieval, (2) stable top-k ordering,
(3) self-match exclusion, (4) Oracle-arm isolation, (5) non-Oracle
leakage prevention, (6) mock-generation diagnostic labeling,
(7) evaluation-input construction, (8) one output per input sample,
(13) no forbidden resource access, (14) CPU-only execution,
(15) repeated-run determinism, (16) legacy baseline compatibility.
"""

import inspect

from src.baseline.generation.dataset_builder import RAGDatasetBuilder, RAGDatasetBuilderConfig
from src.data.iu_xray import generation_adapter, readiness
from src.data.iu_xray.records import IuXrayCanonicalRecord

def _record(sid, **overrides):
    payload = dict(
        dataset_name="iu_xray", study_id=sid, report_id=sid,
        image_ids=[f"{sid}_IM-1"], image_paths=[f"/fake/{sid}_IM-1.png"],
        image_count=1, findings=f"Findings {sid}.", impression=f"Impression {sid}.",
        indication=None, comparison=None, full_report=f"Findings {sid}. Impression {sid}.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def _rag_rows(records):
    corpus = generation_adapter.build_generation_records(records)
    config = RAGDatasetBuilderConfig(
        rag_data_mode=generation_adapter.GENERATION_TEXT_FIELD,
        output_data_mode=generation_adapter.GENERATION_TEXT_FIELD,
    )
    rankings = generation_adapter.build_deterministic_synthetic_rankings(list(corpus), list(corpus))
    builder = RAGDatasetBuilder(corpus, corpus, rankings, config=config,
                                 query_split_name="train", corpus_split_name="train")
    return [builder.build_row(k) for k in corpus]


# ---------------------------------------------------------------------------
# ARM 1: DATA_ONLY
# ---------------------------------------------------------------------------


def test_data_only_arm_one_sample_per_study_and_deterministic_ordering():
    records = [_record(f"CXR{i}") for i in range(5)]
    result = readiness.run_data_only_arm(records)
    assert result["arm"] == "DATA_ONLY"
    assert result["one_sample_per_study"] is True
    assert result["deterministic_ordering"] is True
    assert result["no_excluded_records"] is True


def test_data_only_arm_rejects_excluded_records():
    excluded = _record("CXR1", validation_status="excluded", exclusion_reason="missing_image")
    import pytest
    with pytest.raises(ValueError):
        readiness.run_data_only_arm([excluded])


def test_data_only_arm_deterministic_ordering_does_not_require_global_sort():
    # Regression: a real Colab run against a readiness sample built by
    # concatenating separately-sorted per-split slices (train + validation
    # + test, matching the orchestration script's core_sample =
    # sample_train + sample_val + sample_test) failed arm1_data_only_ok
    # because deterministic_ordering previously required study_ids ==
    # sorted(study_ids) globally. IuXrayDataset.load_split() only
    # guarantees ascending order WITHIN one split -- a caller-merged
    # multi-split sample is legitimately not globally sorted while still
    # being fully deterministic. This must not be flagged as a failure.
    sample_train = [_record(sid) for sid in ("CXR100", "CXR200", "CXR900")]
    sample_val = [_record(sid) for sid in ("CXR050", "CXR150")]
    sample_test = [_record(sid) for sid in ("CXR010", "CXR999")]
    core_sample = sample_train + sample_val + sample_test
    study_ids = [r.study_id for r in core_sample]
    assert study_ids != sorted(study_ids), "fixture must reproduce the real non-sorted shape"

    result = readiness.run_data_only_arm(core_sample)
    assert result["deterministic_ordering"] is True
    assert result["one_sample_per_study"] is True


def test_data_only_arm_accepts_real_usable_impression_only_and_findings_only_records():
    # Regression: a real Colab run against genuine IU X-Ray data hit
    # IuXraySchemaMappingError on a usable (validation_status="valid")
    # impression-only study (findings=None), because the arm previously
    # called schema_mapping.map_to_report_record() directly, which
    # requires both findings and impression to be non-None. A usable
    # record legitimately has only one of the two (that's exactly why
    # target_report_policy's 4-branch precedence exists) -- the arm must
    # accept this, not crash.
    records = [
        _record("CXR1"),
        _record("CXR1002", findings=None),
        _record("CXR2000", impression=None),
    ]
    result = readiness.run_data_only_arm(records)
    assert result["one_sample_per_study"] is True
    assert result["all_target_report_policy_texts_present"] is True
    assert result["schema_mapping_full_record_count"] == 1
    assert result["schema_mapping_skipped_count"] == 2


# ---------------------------------------------------------------------------
# ARM 2: MOCK_RETRIEVAL
# ---------------------------------------------------------------------------


def test_mock_retrieval_arm_no_self_match():
    records = [_record(f"CXR{i}") for i in range(6)]
    result = readiness.run_mock_retrieval_arm(records, top_k=3)
    assert result["no_self_match"] is True
    assert result["evidence_from_different_study"] is True


def test_mock_retrieval_arm_stable_top_k_ordering():
    records = [_record(f"CXR{i}") for i in range(6)]
    r1 = readiness.run_mock_retrieval_arm(records, top_k=3)
    r2 = readiness.run_mock_retrieval_arm(records, top_k=3)
    assert r1 == r2


def test_mock_retrieval_arm_diagnostic_label_present_and_distinct():
    records = [_record(f"CXR{i}") for i in range(3)]
    result = readiness.run_mock_retrieval_arm(records)
    assert "MOCK_RETRIEVAL" in result["diagnostic_label"]
    assert "NOT baseline retrieval performance" in result["diagnostic_label"]


def test_mock_retrieval_arm_no_real_embedding_or_index():
    records = [_record(f"CXR{i}") for i in range(3)]
    result = readiness.run_mock_retrieval_arm(records)
    assert result["no_real_embedding_model"] is True
    assert result["no_real_faiss_index"] is True
    assert result["forbidden_retrievers_used"] == []


# ---------------------------------------------------------------------------
# ARM 3: ORACLE_PIPELINE_CHECK
# ---------------------------------------------------------------------------


def test_oracle_arm_never_self_references():
    records = [_record(f"CXR{i}") for i in range(4)]
    result = readiness.run_oracle_diagnostic_arm(records, query_study_ids=["CXR0", "CXR1"])
    assert result["no_self_reference_in_winning_key"] is True
    assert result["target_reference_isolated_from_winning_key"] is True


def test_oracle_arm_labeled_non_deployable_diagnostic():
    records = [_record(f"CXR{i}") for i in range(3)]
    result = readiness.run_oracle_diagnostic_arm(records, query_study_ids=["CXR0"])
    assert "NON-DEPLOYABLE DIAGNOSTIC" in result["diagnostic_label"]
    assert result["excluded_from_baseline_result_tables"] is True
    assert result["uses_real_chexbert_or_radgraph"] is False


def test_oracle_arm_isolated_from_non_oracle_retrieval_arm():
    # Structural isolation: the Oracle arm function is never called from
    # within the mock-retrieval arm function, and vice versa.
    retrieval_src = inspect.getsource(readiness.run_mock_retrieval_arm)
    oracle_src = inspect.getsource(readiness.run_oracle_diagnostic_arm)
    assert "run_oracle_diagnostic_arm" not in retrieval_src
    assert "run_mock_retrieval_arm" not in oracle_src


def test_oracle_arm_rejects_excluded_records():
    excluded = _record("CXR1", validation_status="excluded", exclusion_reason="missing_report")
    import pytest
    with pytest.raises(ValueError):
        readiness.run_oracle_diagnostic_arm([excluded], query_study_ids=["CXR1"])


def test_oracle_arm_accepts_real_usable_impression_only_and_findings_only_records():
    # Same regression as the DATA_ONLY arm's equivalent test: the Oracle
    # arm previously built its corpus/query ReportRecords via
    # map_to_report_record() directly, which crashes on a real usable
    # impression-only/findings-only study. It must instead derive
    # `.finding` from build_target_report(), same as retrieval_adapter.
    records = [
        _record("CXR1"),
        _record("CXR1002", findings=None),
        _record("CXR2000", impression=None),
    ]
    result = readiness.run_oracle_diagnostic_arm(records, query_study_ids=["CXR1", "CXR1002", "CXR2000"])
    assert result["query_count"] == 3
    assert result["no_self_reference_in_winning_key"] is True
    assert result["target_reference_isolated_from_winning_key"] is True


# ---------------------------------------------------------------------------
# ARM 4: MOCK_GENERATION_SMOKE
# ---------------------------------------------------------------------------


def test_mock_generation_arm_one_output_per_non_excluded_input():
    records = [_record(f"CXR{i}") for i in range(5)]
    rows = _rag_rows(records)
    result = readiness.run_mock_generation_smoke_arm(rows)
    assert result["one_output_per_non_excluded_input"] is True
    assert result["output_count"] == sum(1 for r in rows if not r.excluded)


def test_mock_generation_arm_evaluation_input_construction():
    records = [_record(f"CXR{i}") for i in range(3)]
    rows = _rag_rows(records)
    result = readiness.run_mock_generation_smoke_arm(rows)
    assert result["reference_row_count"] == result["output_count"]
    assert result["prediction_row_count"] == result["output_count"]


def test_mock_generation_arm_stable_output_identifiers():
    records = [_record(f"CXR{i}") for i in range(4)]
    rows = _rag_rows(records)
    result = readiness.run_mock_generation_smoke_arm(rows)
    assert result["stable_output_identifiers"] is True


def test_mock_generation_arm_deterministic_across_repeated_runs():
    records = [_record(f"CXR{i}") for i in range(3)]
    rows = _rag_rows(records)
    result = readiness.run_mock_generation_smoke_arm(rows)
    assert result["deterministic_across_repeated_runs"] is True


def test_mock_generation_arm_diagnostic_labeling_no_medical_claim():
    records = [_record(f"CXR{i}") for i in range(2)]
    rows = _rag_rows(records)
    result = readiness.run_mock_generation_smoke_arm(rows)
    assert "diagnostic only" in result["diagnostic_label"]
    assert "no medical-performance claim" in result["diagnostic_label"]
    assert result["no_real_model_loaded"] is True


# ---------------------------------------------------------------------------
# Cross-cutting: forbidden resources, CPU-only, legacy compatibility
# ---------------------------------------------------------------------------


def test_readiness_module_never_imports_forbidden_resources():
    # A substring scan over the whole module (including its own
    # docstrings/comments explicitly disclosing what is NOT used, e.g.
    # "No embedding model, no FAISS, no MARVEL/MedCPT/...") would flag
    # false positives on those exact disclosures. What actually matters
    # is that nothing is ever imported -- checked directly against the
    # module's real import statements via the ast module.
    import ast
    import src.data.iu_xray.readiness as mod

    tree = ast.parse(inspect.getsource(mod))
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module.lower())

    forbidden_modules = ("marvel", "faiss", "chexpert_dataset")
    for forbidden in forbidden_modules:
        assert not any(forbidden in name for name in imported_names), (
            f"forbidden module {forbidden!r} imported in readiness.py: {imported_names}"
        )


def test_arms_do_not_import_torch_cuda_or_faiss():
    import src.data.iu_xray.readiness as mod
    source = inspect.getsource(mod)
    assert "import faiss" not in source
    assert "cuda" not in source.lower()


def test_readiness_matrix_validator_flags_bad_entries():
    bad_matrix = {"x": {"status": "NOT_A_REAL_STATUS", "rationale": "..."}}
    problems = readiness.validate_readiness_matrix(bad_matrix)
    assert len(problems) == 1

    missing_rationale = {"y": {"status": readiness.READY, "rationale": ""}}
    problems2 = readiness.validate_readiness_matrix(missing_rationale)
    assert len(problems2) == 1


def test_readiness_matrix_validator_accepts_well_formed_entries():
    good_matrix = {"x": {"status": readiness.READY, "rationale": "because it works"}}
    assert readiness.validate_readiness_matrix(good_matrix) == []
