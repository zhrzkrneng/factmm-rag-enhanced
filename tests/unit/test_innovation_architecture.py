"""Unit tests for src/innovation/architecture.py.

Covers Cell 47 spec Section 12 tests: 13 (baseline behavior contract
preserved), 14 (no target-report field is part of innovation inference
input).
"""

import dataclasses

import pytest

from src.innovation.architecture import (
    PIPELINE_STAGES,
    STAGE_ADAPTIVE_RETRIEVAL,
    STAGE_CONFIDENCE_GATED_FUSION,
    STAGE_FACTUAL_RERANKING,
    InnovationInferenceContext,
    active_stages,
    describe_architecture,
    inference_context_field_names,
)
from src.innovation.experiment_contract import EXPERIMENT_MATRIX, InnovationConfig


def test_baseline_behavior_contract_preserved_when_all_disabled():
    stages = active_stages(InnovationConfig())
    assert STAGE_ADAPTIVE_RETRIEVAL not in stages
    assert STAGE_FACTUAL_RERANKING not in stages
    assert STAGE_CONFIDENCE_GATED_FUSION not in stages
    # Every non-innovation stage must still be present, in original order.
    assert stages == tuple(s for s in PIPELINE_STAGES if s not in (
        STAGE_ADAPTIVE_RETRIEVAL, STAGE_FACTUAL_RERANKING, STAGE_CONFIDENCE_GATED_FUSION,
    ))


def test_each_innovation_stage_independently_ablatable():
    only_i1 = active_stages(InnovationConfig(adaptive_retrieval_enabled=True))
    assert STAGE_ADAPTIVE_RETRIEVAL in only_i1
    assert STAGE_FACTUAL_RERANKING not in only_i1
    assert STAGE_CONFIDENCE_GATED_FUSION not in only_i1

    only_i2 = active_stages(InnovationConfig(factual_reranking_enabled=True))
    assert STAGE_FACTUAL_RERANKING in only_i2
    assert STAGE_ADAPTIVE_RETRIEVAL not in only_i2

    only_i3 = active_stages(InnovationConfig(confidence_gated_fusion_enabled=True))
    assert STAGE_CONFIDENCE_GATED_FUSION in only_i3
    assert STAGE_ADAPTIVE_RETRIEVAL not in only_i3


def test_all_experiment_arms_produce_a_valid_stage_ordering():
    for arm in EXPERIMENT_MATRIX:
        stages = active_stages(arm.config)
        # Original pipeline order must always be preserved -- a stage
        # present in `stages` must appear in the same relative order as
        # in PIPELINE_STAGES.
        filtered_full_order = [s for s in PIPELINE_STAGES if s in stages]
        assert list(stages) == filtered_full_order


def test_describe_architecture_json_shape():
    payload = describe_architecture()
    assert payload["pipeline_stages"] == list(PIPELINE_STAGES)
    assert payload["independently_switchable"] is True
    assert STAGE_ADAPTIVE_RETRIEVAL not in payload["baseline_stages_when_all_disabled"]


# ---------------------------------------------------------------------------
# No target-report field in innovation inference input
# ---------------------------------------------------------------------------


def test_inference_context_has_no_target_report_field():
    field_names = inference_context_field_names()
    for name in field_names:
        lowered = name.lower()
        assert "target" not in lowered, f"target-report-shaped field found: {name}"
        assert "reference" not in lowered, f"target-report-shaped field found: {name}"
        assert "ground_truth" not in lowered, f"target-report-shaped field found: {name}"


def test_inference_context_field_set_is_the_documented_four_fields():
    assert inference_context_field_names() == (
        "query_key", "candidate_keys", "candidate_scores", "candidate_texts",
    )


def test_inference_context_construction_rejects_unknown_target_kwarg():
    with pytest.raises(TypeError):
        InnovationInferenceContext(
            query_key=("iu-xray", "CXR1", "CXR1"),
            candidate_keys=(("iu-xray", "CXR2", "CXR2"),),
            candidate_scores=(0.9,),
            candidate_texts=("some candidate text",),
            target_report_text="this must not be a valid field",  # type: ignore[call-arg]
        )


def test_inference_context_constructs_with_only_documented_fields():
    ctx = InnovationInferenceContext(
        query_key=("iu-xray", "CXR1", "CXR1"),
        candidate_keys=(("iu-xray", "CXR2", "CXR2"), ("iu-xray", "CXR3", "CXR3")),
        candidate_scores=(0.9, 0.4),
        candidate_texts=("candidate two text", "candidate three text"),
    )
    assert ctx.query_key == ("iu-xray", "CXR1", "CXR1")
    assert len(ctx.candidate_keys) == 2
    assert dataclasses.fields(ctx)
