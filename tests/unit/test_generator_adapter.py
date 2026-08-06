"""Unit tests for src/baseline/generation/adapter.py."""

import dataclasses

import pytest
import torch

from src.baseline.generation.adapter import (
    GenerationMetadata,
    GeneratorAdapter,
    GeneratorResult,
    PreparedGeneratorInputs,
)


def _metadata(**overrides):
    defaults = dict(
        adapter_kind="mock",
        model_checkpoint="ckpt",
        resolved_revision=None,
        base_lm_name="lm",
        vision_tower_name="vision",
        lora_checkpoint=None,
        conv_mode="vicuna_v1",
        git_commit=None,
        torch_version=None,
        transformers_version=None,
        hardware="mock",
        generation_timestamp_utc="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return GenerationMetadata(**defaults)


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------


def test_generator_adapter_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        GeneratorAdapter()


def test_incomplete_subclass_missing_a_method_cannot_be_instantiated():
    class IncompleteAdapter(GeneratorAdapter):
        def prepare_inputs(self, image_path, prompt_text):
            raise NotImplementedError

        def generate(self, query_key, image_path, prompt, generation_config):
            raise NotImplementedError

        # generate_batch intentionally not implemented

    with pytest.raises(TypeError):
        IncompleteAdapter()


def test_complete_subclass_can_be_instantiated():
    class CompleteAdapter(GeneratorAdapter):
        def prepare_inputs(self, image_path, prompt_text):
            raise NotImplementedError

        def generate(self, query_key, image_path, prompt, generation_config):
            raise NotImplementedError

        def generate_batch(self, requests, generation_config):
            raise NotImplementedError

    CompleteAdapter()  # must not raise


# ---------------------------------------------------------------------------
# PreparedGeneratorInputs
# ---------------------------------------------------------------------------


def test_prepared_generator_inputs_is_immutable():
    prepared = PreparedGeneratorInputs(
        pixel_values=torch.zeros(1), input_ids=torch.zeros(1, dtype=torch.long),
        attention_mask=torch.zeros(1, dtype=torch.long),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.pixel_values = torch.zeros(2)


def test_prepared_generator_inputs_holds_the_given_tensors():
    pixel_values = torch.rand(3, 4, 4)
    input_ids = torch.tensor([[1, 2, 3]])
    attention_mask = torch.ones_like(input_ids)
    prepared = PreparedGeneratorInputs(
        pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
    )
    assert torch.equal(prepared.pixel_values, pixel_values)
    assert torch.equal(prepared.input_ids, input_ids)
    assert torch.equal(prepared.attention_mask, attention_mask)


# ---------------------------------------------------------------------------
# GenerationMetadata
# ---------------------------------------------------------------------------


def test_generation_metadata_is_immutable():
    metadata = _metadata()
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.hardware = "cuda:0"


def test_generation_metadata_accepts_valid_hf_and_mock_kinds():
    _metadata(adapter_kind="hf")  # must not raise
    _metadata(adapter_kind="mock")  # must not raise


def test_generation_metadata_rejects_invalid_adapter_kind():
    with pytest.raises(ValueError):
        _metadata(adapter_kind="bogus")


@pytest.mark.parametrize(
    "field_name",
    ["model_checkpoint", "base_lm_name", "vision_tower_name", "conv_mode", "hardware", "generation_timestamp_utc"],
)
def test_generation_metadata_rejects_empty_required_string_fields(field_name):
    with pytest.raises(ValueError):
        _metadata(**{field_name: ""})


def test_generation_metadata_allows_optional_fields_to_be_none():
    metadata = _metadata(
        resolved_revision=None, lora_checkpoint=None, git_commit=None,
        torch_version=None, transformers_version=None,
    )
    assert metadata.resolved_revision is None
    assert metadata.lora_checkpoint is None
    assert metadata.git_commit is None


def test_generation_metadata_allows_optional_fields_to_be_populated():
    metadata = _metadata(
        resolved_revision="abc123", lora_checkpoint="/lora", git_commit="deadbeef",
        torch_version="2.1.0", transformers_version="4.40.0",
    )
    assert metadata.resolved_revision == "abc123"
    assert metadata.lora_checkpoint == "/lora"
    assert metadata.git_commit == "deadbeef"


# ---------------------------------------------------------------------------
# GeneratorResult -- error/result exclusivity
# ---------------------------------------------------------------------------


def test_generator_result_accepts_success_with_no_error():
    result = GeneratorResult(
        query_key=("mimic-cxr", "p1", "s1"),
        generated_report_text="a report",
        generation_metadata=_metadata(),
        error=None,
    )
    assert result.generated_report_text == "a report"
    assert result.error is None


def test_generator_result_accepts_failure_with_no_text():
    result = GeneratorResult(
        query_key=("mimic-cxr", "p1", "s1"),
        generated_report_text=None,
        generation_metadata=_metadata(),
        error="something went wrong",
    )
    assert result.generated_report_text is None
    assert result.error == "something went wrong"


def test_generator_result_rejects_both_text_and_error_set():
    with pytest.raises(ValueError):
        GeneratorResult(
            query_key=("mimic-cxr", "p1", "s1"),
            generated_report_text="a report",
            generation_metadata=_metadata(),
            error="also an error",
        )


def test_generator_result_rejects_neither_text_nor_error_set():
    with pytest.raises(ValueError):
        GeneratorResult(
            query_key=("mimic-cxr", "p1", "s1"),
            generated_report_text=None,
            generation_metadata=_metadata(),
            error=None,
        )


def test_generator_result_is_immutable():
    result = GeneratorResult(
        query_key=("mimic-cxr", "p1", "s1"),
        generated_report_text="a report",
        generation_metadata=_metadata(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.generated_report_text = "different"


def test_generator_result_error_defaults_to_none():
    result = GeneratorResult(
        query_key=("mimic-cxr", "p1", "s1"),
        generated_report_text="a report",
        generation_metadata=_metadata(),
    )
    assert result.error is None
