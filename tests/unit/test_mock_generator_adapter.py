"""Unit tests for src/baseline/generation/mock_adapter.py."""

import dataclasses
import inspect
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.baseline.generation.adapter import GeneratorResult
from src.baseline.generation.mock_adapter import MockGeneratorAdapter


@dataclasses.dataclass(frozen=True)
class _FakeGenerationConfig:
    """Lightweight stand-in for the future GenerationConfig
    (Milestone 2.5's generator.py, not yet implemented). Only the
    `seed` attribute MockGeneratorAdapter's contract explicitly
    requires (docs/milestone_2_5_generator_contract.md SS6, item 5's
    "thread GenerationConfig.seed through behavior") is modeled here.
    """

    seed: int = 42


Q1 = ("mimic-cxr", "p1", "s1")
Q2 = ("mimic-cxr", "p2", "s2")


def _adapter():
    return MockGeneratorAdapter()


# ---------------------------------------------------------------------------
# prepare_inputs -- separately testable, structural output
# ---------------------------------------------------------------------------


def test_prepare_inputs_returns_expected_tensor_shapes_and_dtypes():
    adapter = _adapter()
    prepared = adapter.prepare_inputs("/img.png", "a prompt with words")

    assert prepared.pixel_values.shape == (3, 4, 4)
    assert prepared.pixel_values.dtype == torch.float32
    assert prepared.input_ids.dim() == 2
    assert prepared.input_ids.shape[0] == 1
    assert prepared.input_ids.shape[1] == len("a prompt with words".split())
    assert prepared.attention_mask.shape == prepared.input_ids.shape
    assert torch.all(prepared.attention_mask == 1)


def test_prepare_inputs_is_deterministic_for_same_input():
    adapter = _adapter()
    first = adapter.prepare_inputs("/img.png", "a prompt")
    second = adapter.prepare_inputs("/img.png", "a prompt")

    assert torch.equal(first.pixel_values, second.pixel_values)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.attention_mask, second.attention_mask)


def test_prepare_inputs_differs_for_different_prompt():
    adapter = _adapter()
    first = adapter.prepare_inputs("/img.png", "prompt one")
    second = adapter.prepare_inputs("/img.png", "prompt two entirely different")

    assert not torch.equal(first.pixel_values, second.pixel_values)


def test_prepare_inputs_rejects_empty_image_path():
    with pytest.raises(ValueError):
        _adapter().prepare_inputs("", "a prompt")


def test_prepare_inputs_rejects_empty_prompt_text():
    with pytest.raises(ValueError):
        _adapter().prepare_inputs("/img.png", "")


# ---------------------------------------------------------------------------
# generate() -- deterministic output, seed threading
# ---------------------------------------------------------------------------


def test_generate_same_input_and_seed_is_byte_identical():
    adapter = _adapter()
    config = _FakeGenerationConfig(seed=42)

    first = adapter.generate(Q1, "/img.png", "a prompt", config)
    second = adapter.generate(Q1, "/img.png", "a prompt", config)

    assert first.generated_report_text == second.generated_report_text
    assert first.query_key == second.query_key
    assert first.error == second.error
    # generation_timestamp_utc legitimately varies between calls -- every
    # other metadata field must not.
    meta_a = dataclasses.asdict(first.generation_metadata)
    meta_b = dataclasses.asdict(second.generation_metadata)
    del meta_a["generation_timestamp_utc"]
    del meta_b["generation_timestamp_utc"]
    assert meta_a == meta_b


def test_generate_different_seed_changes_output():
    adapter = _adapter()
    result_a = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig(seed=1))
    result_b = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig(seed=2))

    assert result_a.generated_report_text != result_b.generated_report_text


def test_generate_different_prompt_changes_output_for_same_seed():
    adapter = _adapter()
    result_a = adapter.generate(Q1, "/img.png", "prompt one", _FakeGenerationConfig(seed=42))
    result_b = adapter.generate(Q1, "/img.png", "prompt two", _FakeGenerationConfig(seed=42))

    assert result_a.generated_report_text != result_b.generated_report_text


def test_generate_rejects_non_int_seed():
    adapter = _adapter()
    with pytest.raises(ValueError):
        adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig(seed="not-an-int"))


def test_generate_returns_generator_result_with_no_error():
    adapter = _adapter()
    result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig())
    assert isinstance(result, GeneratorResult)
    assert result.error is None
    assert result.generated_report_text is not None


# ---------------------------------------------------------------------------
# generate() must internally use prepare_inputs()
# ---------------------------------------------------------------------------


def test_generate_invokes_prepare_inputs():
    adapter = _adapter()
    with patch.object(adapter, "prepare_inputs", wraps=adapter.prepare_inputs) as spy:
        adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig())
    spy.assert_called_once_with("/img.png", "a prompt")


def test_generate_output_actually_depends_on_prepare_inputs_result():
    # Not just "called" -- the prepared tensors' content genuinely
    # influences the generated text (different prompt -> different
    # prepare_inputs() output -> different generated text), already
    # exercised by test_generate_different_prompt_changes_output_for_same_seed;
    # this test additionally confirms the coupling is via prepare_inputs
    # specifically, by forcing it to return a fixed, wrong-shaped result
    # and observing the generated text changes accordingly.
    adapter = _adapter()
    from src.baseline.generation.adapter import PreparedGeneratorInputs

    real_result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig(seed=42))

    forced = PreparedGeneratorInputs(
        pixel_values=torch.zeros(3, 4, 4),
        input_ids=torch.tensor([[999999]]),
        attention_mask=torch.tensor([[1]]),
    )
    with patch.object(adapter, "prepare_inputs", return_value=forced):
        forced_result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig(seed=42))

    assert real_result.generated_report_text != forced_result.generated_report_text


# ---------------------------------------------------------------------------
# generate_batch -- ordering, empty batch
# ---------------------------------------------------------------------------


def test_generate_batch_preserves_request_order():
    adapter = _adapter()
    requests = [
        (Q2, "/img2.png", "prompt two"),
        (Q1, "/img1.png", "prompt one"),
    ]
    results = adapter.generate_batch(requests, _FakeGenerationConfig())

    assert [r.query_key for r in results] == [Q2, Q1]


def test_generate_batch_matches_individual_generate_calls():
    adapter = _adapter()
    config = _FakeGenerationConfig(seed=7)
    requests = [(Q1, "/img1.png", "prompt one"), (Q2, "/img2.png", "prompt two")]

    batch_results = adapter.generate_batch(requests, config)
    individual_results = [adapter.generate(*req, config) for req in requests]

    assert [r.generated_report_text for r in batch_results] == [
        r.generated_report_text for r in individual_results
    ]


def test_generate_batch_empty_returns_empty_list():
    adapter = _adapter()
    results = adapter.generate_batch([], _FakeGenerationConfig())
    assert results == []


# ---------------------------------------------------------------------------
# Invalid query key
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    [
        None,
        "not-a-tuple",
        ("mimic-cxr", "p1"),  # only 2 elements
        ("mimic-cxr", "p1", "s1", "extra"),  # 4 elements
        ("mimic-cxr", "", "s1"),  # empty component
        ("mimic-cxr", "p1", 123),  # non-string component
    ],
)
def test_generate_rejects_invalid_query_key(bad_key):
    adapter = _adapter()
    with pytest.raises(ValueError):
        adapter.generate(bad_key, "/img.png", "a prompt", _FakeGenerationConfig())


def test_generate_batch_rejects_invalid_query_key_in_any_request():
    adapter = _adapter()
    requests = [(Q1, "/img1.png", "prompt one"), (("bad",), "/img2.png", "prompt two")]
    with pytest.raises(ValueError):
        adapter.generate_batch(requests, _FakeGenerationConfig())


# ---------------------------------------------------------------------------
# Metadata completeness
# ---------------------------------------------------------------------------


def test_generate_metadata_has_mock_placeholders():
    adapter = _adapter()
    result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig())
    metadata = result.generation_metadata

    assert metadata.adapter_kind == "mock"
    assert metadata.resolved_revision is None
    assert metadata.hardware == "mock"
    assert metadata.torch_version is None
    assert metadata.transformers_version is None
    assert metadata.model_checkpoint
    assert metadata.base_lm_name
    assert metadata.vision_tower_name
    assert metadata.conv_mode == "vicuna_v1"
    assert metadata.generation_timestamp_utc


def test_generate_metadata_git_commit_matches_actual_repo_head():
    adapter = _adapter()
    result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig())

    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=Path(__file__).resolve().parent,
    ).stdout.strip()
    if actual_head:
        assert result.generation_metadata.git_commit == actual_head


def test_conv_mode_is_configurable_and_reflected_in_metadata():
    adapter = MockGeneratorAdapter(conv_mode="custom_mode")
    result = adapter.generate(Q1, "/img.png", "a prompt", _FakeGenerationConfig())
    assert result.generation_metadata.conv_mode == "custom_mode"


def test_mock_generator_adapter_rejects_empty_conv_mode():
    with pytest.raises(ValueError):
        MockGeneratorAdapter(conv_mode="")


# ---------------------------------------------------------------------------
# No ML dependency (static source scan)
# ---------------------------------------------------------------------------


def test_mock_adapter_module_has_no_transformers_import():
    import ast

    import src.baseline.generation.mock_adapter as mock_adapter_module

    source = inspect.getsource(mock_adapter_module)
    tree = ast.parse(source)
    imported_root_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_root_modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_root_modules.add(node.module.split(".")[0])

    assert "transformers" not in imported_root_modules

    from_pretrained_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]
    assert from_pretrained_calls == []


def test_mock_adapter_is_a_generator_adapter():
    from src.baseline.generation.adapter import GeneratorAdapter

    assert issubclass(MockGeneratorAdapter, GeneratorAdapter)
