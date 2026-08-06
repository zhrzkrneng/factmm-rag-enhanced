"""Unit tests for src/innovation/experiment_contract.py's E0-E7 matrix.

Covers Cell 47 spec Section 12 tests: 3 (all eight experiment arms are
representable), 4 (experiment arm IDs are deterministic), 15 (experiment
matrix contains exactly E0-E7).
"""

from src.innovation.experiment_contract import (
    EXPERIMENT_MATRIX,
    REQUIRED_ARM_IDS,
    REQUIRED_STATISTICAL_COMPARISONS,
    InnovationConfig,
    compute_arm_id,
    experiment_matrix_to_json,
)


def test_experiment_matrix_contains_exactly_e0_through_e7():
    arm_ids = tuple(arm.arm_id for arm in EXPERIMENT_MATRIX)
    assert arm_ids == REQUIRED_ARM_IDS
    assert len(EXPERIMENT_MATRIX) == 8


def test_all_eight_experiment_arms_are_representable():
    # Every one of the 2^3 = 8 switch combinations must map to exactly
    # one arm, with no duplicates and no gaps.
    seen_switch_combos = set()
    for arm in EXPERIMENT_MATRIX:
        combo = (
            arm.config.adaptive_retrieval_enabled,
            arm.config.factual_reranking_enabled,
            arm.config.confidence_gated_fusion_enabled,
        )
        assert combo not in seen_switch_combos, f"duplicate switch combo {combo}"
        seen_switch_combos.add(combo)
    assert len(seen_switch_combos) == 8


def test_experiment_arm_ids_are_deterministic():
    for arm in EXPERIMENT_MATRIX:
        recomputed = compute_arm_id(arm.config)
        assert recomputed == arm.arm_id

    # Recomputing on a freshly constructed, structurally-identical
    # config (not the same object) must yield the same arm id.
    fresh = InnovationConfig(adaptive_retrieval_enabled=True, confidence_gated_fusion_enabled=True)
    assert compute_arm_id(fresh) == "E5"


def test_experiment_arm_ids_do_not_depend_on_nested_config_values():
    from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig

    a = InnovationConfig(adaptive_retrieval_enabled=True, adaptive_k=AdaptiveKConfig(min_k=1, max_k=3))
    b = InnovationConfig(adaptive_retrieval_enabled=True, adaptive_k=AdaptiveKConfig(min_k=2, max_k=9, default_k=2))
    assert compute_arm_id(a) == compute_arm_id(b) == "E1"


def test_specific_arm_switch_mapping_matches_spec_section_7():
    by_id = {arm.arm_id: arm.config for arm in EXPERIMENT_MATRIX}
    assert (by_id["E0"].adaptive_retrieval_enabled, by_id["E0"].factual_reranking_enabled, by_id["E0"].confidence_gated_fusion_enabled) == (False, False, False)
    assert (by_id["E1"].adaptive_retrieval_enabled, by_id["E1"].factual_reranking_enabled, by_id["E1"].confidence_gated_fusion_enabled) == (True, False, False)
    assert (by_id["E2"].adaptive_retrieval_enabled, by_id["E2"].factual_reranking_enabled, by_id["E2"].confidence_gated_fusion_enabled) == (False, True, False)
    assert (by_id["E3"].adaptive_retrieval_enabled, by_id["E3"].factual_reranking_enabled, by_id["E3"].confidence_gated_fusion_enabled) == (False, False, True)
    assert (by_id["E4"].adaptive_retrieval_enabled, by_id["E4"].factual_reranking_enabled, by_id["E4"].confidence_gated_fusion_enabled) == (True, True, False)
    assert (by_id["E5"].adaptive_retrieval_enabled, by_id["E5"].factual_reranking_enabled, by_id["E5"].confidence_gated_fusion_enabled) == (True, False, True)
    assert (by_id["E6"].adaptive_retrieval_enabled, by_id["E6"].factual_reranking_enabled, by_id["E6"].confidence_gated_fusion_enabled) == (False, True, True)
    assert (by_id["E7"].adaptive_retrieval_enabled, by_id["E7"].factual_reranking_enabled, by_id["E7"].confidence_gated_fusion_enabled) == (True, True, True)


def test_experiment_matrix_json_serialization_contains_all_arms():
    payload = experiment_matrix_to_json()
    assert len(payload) == 8
    assert {row["arm_id"] for row in payload} == set(REQUIRED_ARM_IDS)


def test_required_statistical_comparisons_reference_only_real_arm_ids():
    valid_ids = {arm.arm_id for arm in EXPERIMENT_MATRIX}
    for a, b in REQUIRED_STATISTICAL_COMPARISONS:
        assert a in valid_ids
        assert b in valid_ids
    assert ("E0", "E1") in REQUIRED_STATISTICAL_COMPARISONS
    assert ("E0", "E2") in REQUIRED_STATISTICAL_COMPARISONS
    assert ("E0", "E3") in REQUIRED_STATISTICAL_COMPARISONS
    assert ("E0", "E7") in REQUIRED_STATISTICAL_COMPARISONS
