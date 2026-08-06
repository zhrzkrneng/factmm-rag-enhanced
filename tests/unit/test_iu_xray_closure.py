"""Unit tests for src/data/iu_xray/closure.py.

Covers: (9) readiness-status validation, (10) no mock result promoted
to scientific baseline, (11) closure claim correctness, (16) legacy
baseline compatibility (real mock-registry/significance calls against
this project's own unmodified evaluation modules).
"""

from src.data.iu_xray import closure, readiness


def test_component_readiness_matrix_is_well_formed():
    problems = readiness.validate_readiness_matrix(closure.COMPONENT_READINESS_MATRIX)
    assert problems == []


def test_readiness_totals_sum_to_component_count():
    totals = closure.readiness_totals()
    assert sum(totals.values()) == len(closure.COMPONENT_READINESS_MATRIX)


def test_marvel_removed_mimic_chexpert_deferred_in_matrix():
    matrix = closure.COMPONENT_READINESS_MATRIX
    assert matrix["marvel_warm_start_checkpoint"]["status"] == readiness.REMOVED
    assert matrix["mimic_cxr_dataset"]["status"] == readiness.DEFERRED
    assert matrix["chexpert_dataset"]["status"] == readiness.DEFERRED
    assert matrix["replacement_retriever_selection"]["status"] == readiness.DEFERRED
    assert matrix["vicuna_llava_access"]["status"] == readiness.DEFERRED


def test_model_dependent_components_never_marked_ready_merely_for_interface_existence():
    matrix = closure.COMPONENT_READINESS_MATRIX
    for component in ("embedding_generation", "vector_index", "real_generator_inference",
                       "real_baseline_result_table"):
        assert matrix[component]["status"] != readiness.READY, (
            f"{component} must not be READY -- no model/embedding work has occurred"
        )


def test_metric_readiness_smoke_runs_against_real_mock_registries():
    result = closure.run_metric_readiness_smoke()
    assert result["retrieval_metrics_smoke_ok"] is True
    assert result["generation_metrics_smoke_ok"] is True
    assert result["significance_smoke_ok"] is True
    assert set(result["retrieval_metrics_tested"]) == {"mrr", "recall", "ndcg"}


def test_metric_readiness_never_marks_clinical_metrics_fully_ready():
    result = closure.run_metric_readiness_smoke()
    assert result["metric_status"]["f1_radgraph"] == readiness.PARTIALLY_READY
    assert result["metric_status"]["f1_chexbert"] == readiness.PARTIALLY_READY
    assert result["values_are_mock_never_scientific"] is True


def test_metric_readiness_includes_iu_xray_clinical_caveats():
    result = closure.run_metric_readiness_smoke()
    assert len(result["clinical_metric_caveats"]) >= 2
    assert any("MIMIC-CXR" in c for c in result["clinical_metric_caveats"])


def test_bootstrap_significance_marked_ready_no_model_dependency():
    assert closure.run_metric_readiness_smoke()["metric_status"]["bootstrap_significance"] == readiness.READY


def test_must_not_claim_list_matches_instruction_exactly():
    assert closure.MUST_NOT_CLAIM == (
        "exact FactMM-RAG reproduction",
        "direct comparability to MIMIC-CXR/CheXpert paper scores",
        "real retriever execution",
        "real Vicuna/LLaVA generation",
        "scientific performance from mock outputs",
        "completed baseline result table",
    )


def test_may_claim_list_matches_instruction_exactly():
    assert closure.MAY_CLAIM == (
        "architecture-faithful adaptation inspired by FactMM-RAG",
        "official public IU X-Ray acquisition and deterministic preprocessing",
        "validated loader and schema adaptation",
        "validated retrieval/generation/prompt interfaces",
        "CPU-only end-to-end readiness",
        "readiness to implement the selected retriever and generator",
        "controlled future baseline-versus-innovation design",
    )


def test_no_overlap_between_may_and_must_not_claim():
    assert set(closure.MAY_CLAIM).isdisjoint(set(closure.MUST_NOT_CLAIM))


def test_cell_audit_covers_cells_40_through_46():
    for n in range(40, 47):
        assert f"cell_{n}" in closure.CELL_AUDIT
        assert closure.CELL_AUDIT[f"cell_{n}"]


def test_readiness_totals_deterministic_across_repeated_calls():
    assert closure.readiness_totals() == closure.readiness_totals()
