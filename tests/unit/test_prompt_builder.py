"""Unit tests for src/baseline/generation/prompt_builder.py."""

import dataclasses

import pytest

from src.baseline.generation.prompt_builder import (
    PromptBuilder,
    PromptBuilderConfig,
    PromptMode,
    PromptResult,
)

RAG_TRAIN_EXPECTED = (
    'Here is a report of a related patient: "a related report"\n'
    "Generate a radiology report from this image:<image>"
)
RAG_INFERENCE_EXPECTED = (
    'Here is a report of a related patient: "a related report"\n'
    "Generate a radiology report from this image:"
)
VQA_TRAIN_EXPECTED = "Generate a radiology report from this image:<image>"
VQA_INFERENCE_EXPECTED = "\nGenerate a radiology report from this image:"


def _default_builder(**overrides):
    return PromptBuilder(PromptBuilderConfig(**overrides))


# ---------------------------------------------------------------------------
# Exact byte-for-byte official strings
# ---------------------------------------------------------------------------


def test_rag_train_matches_exact_official_string():
    builder = _default_builder()
    result = builder.build(
        PromptMode.RAG_TRAIN, retrieved_report="a related report", target_report="target"
    )
    assert result.text == RAG_TRAIN_EXPECTED


def test_rag_inference_matches_exact_official_string():
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="a related report")
    assert result.text == RAG_INFERENCE_EXPECTED


def test_vqa_train_matches_exact_official_string():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_TRAIN, target_report="target")
    assert result.text == VQA_TRAIN_EXPECTED


def test_vqa_inference_matches_exact_official_string():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_INFERENCE)
    assert result.text == VQA_INFERENCE_EXPECTED


def test_exact_newline_placement_rag_train():
    builder = _default_builder()
    result = builder.build(
        PromptMode.RAG_TRAIN, retrieved_report="report text", target_report="target"
    )
    # Exactly one newline, immediately after the closing quote, none elsewhere.
    assert result.text.count("\n") == 1
    quote_close_index = result.text.index('"report text"') + len('"report text"')
    assert result.text[quote_close_index] == "\n"


def test_exact_newline_placement_vqa_inference():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_INFERENCE)
    assert result.text.startswith("\n")
    assert result.text.count("\n") == 1


def test_vqa_train_has_no_leading_newline_unlike_vqa_inference():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_TRAIN, target_report="target")
    assert not result.text.startswith("\n")


def test_no_separator_before_image_token():
    builder = _default_builder()
    result = builder.build(
        PromptMode.RAG_TRAIN, retrieved_report="report text", target_report="target"
    )
    assert "image:<image>" in result.text
    assert "image: <image>" not in result.text
    assert "image:\n<image>" not in result.text


def test_rag_inference_has_no_image_token():
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="report text")
    assert "<image>" not in result.text


# ---------------------------------------------------------------------------
# Quote placement / verbatim preservation
# ---------------------------------------------------------------------------


def test_retrieved_report_preserved_verbatim():
    weird_text = '  Odd   spacing, MiXeD case, "nested quotes" inside.  '
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_TRAIN, retrieved_report=weird_text, target_report="t")
    assert weird_text in result.text


def test_target_report_preserved_verbatim():
    weird_target = "  Target with   odd\tspacing and CAPS.  "
    builder = _default_builder()
    result = builder.build(
        PromptMode.RAG_TRAIN, retrieved_report="report", target_report=weird_target
    )
    assert result.conversations[1]["value"] == weird_target


def test_quote_characters_immediately_surround_retrieved_report():
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="XYZ")
    assert '"XYZ"' in result.text


def test_custom_quote_char_reflected_in_output():
    builder = _default_builder(retrieved_report_quote_char="'")
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="XYZ")
    assert "'XYZ'" in result.text
    assert '"XYZ"' not in result.text


def test_custom_image_token_reflected_in_output():
    builder = _default_builder(image_token="[IMG]")
    result = builder.build(PromptMode.VQA_TRAIN, target_report="t")
    assert result.text == "Generate a radiology report from this image:[IMG]"


# ---------------------------------------------------------------------------
# Purity / determinism
# ---------------------------------------------------------------------------


def test_build_is_pure_and_deterministic():
    builder = _default_builder()
    first = builder.build(PromptMode.RAG_TRAIN, retrieved_report="report", target_report="target")
    second = builder.build(PromptMode.RAG_TRAIN, retrieved_report="report", target_report="target")
    assert first == second


def test_build_does_not_mutate_config():
    config = PromptBuilderConfig()
    builder = PromptBuilder(config)
    builder.build(PromptMode.VQA_INFERENCE)
    assert builder._config == config


# ---------------------------------------------------------------------------
# Argument requirement / prohibition rules
# ---------------------------------------------------------------------------


def test_rag_train_requires_retrieved_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_TRAIN, target_report="target")


def test_rag_inference_requires_retrieved_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_INFERENCE)


def test_vqa_train_forbids_retrieved_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.VQA_TRAIN, retrieved_report="not allowed", target_report="t")


def test_vqa_inference_forbids_retrieved_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.VQA_INFERENCE, retrieved_report="not allowed")


def test_rag_train_requires_target_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_TRAIN, retrieved_report="report")


def test_vqa_train_requires_target_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.VQA_TRAIN)


def test_rag_inference_forbids_target_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_INFERENCE, retrieved_report="report", target_report="not allowed")


def test_vqa_inference_forbids_target_report():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build(PromptMode.VQA_INFERENCE, target_report="not allowed")


def test_invalid_mode_type_rejected():
    builder = _default_builder()
    with pytest.raises(ValueError):
        builder.build("rag_train")  # not a PromptMode


# ---------------------------------------------------------------------------
# Empty / whitespace / length validation
# ---------------------------------------------------------------------------


def test_empty_retrieved_report_rejected_when_configured():
    builder = _default_builder(reject_empty_retrieved_report=True)
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_INFERENCE, retrieved_report="")


def test_whitespace_only_retrieved_report_rejected_when_configured():
    builder = _default_builder(reject_empty_retrieved_report=True)
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_INFERENCE, retrieved_report="   \t  ")


def test_empty_retrieved_report_allowed_when_reject_disabled():
    builder = _default_builder(reject_empty_retrieved_report=False)
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="")
    assert '""' in result.text


def test_max_retrieved_report_length_overflow_raises():
    builder = _default_builder(max_retrieved_report_length=5)
    with pytest.raises(ValueError):
        builder.build(PromptMode.RAG_INFERENCE, retrieved_report="this is too long")


def test_max_retrieved_report_length_at_exact_limit_is_allowed():
    builder = _default_builder(max_retrieved_report_length=5)
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="12345")
    assert "12345" in result.text


def test_max_retrieved_report_length_none_allows_any_length():
    builder = _default_builder(max_retrieved_report_length=None)
    long_text = "x" * 10_000
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report=long_text)
    assert long_text in result.text


def test_retrieved_report_is_never_truncated():
    builder = _default_builder()
    long_text = "word " * 500
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report=long_text)
    assert long_text in result.text


# ---------------------------------------------------------------------------
# prompt_version propagation
# ---------------------------------------------------------------------------


def test_prompt_version_propagates_to_result():
    builder = _default_builder(prompt_version="custom_v7")
    result = builder.build(PromptMode.VQA_INFERENCE)
    assert result.prompt_version == "custom_v7"


def test_default_prompt_version_is_official_v1():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_INFERENCE)
    assert result.prompt_version == "official_v1"


# ---------------------------------------------------------------------------
# Conversation structure
# ---------------------------------------------------------------------------


def test_rag_train_conversation_structure():
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_TRAIN, retrieved_report="report", target_report="target report")
    assert result.conversations == [
        {"from": "human", "value": RAG_TRAIN_EXPECTED.replace("a related report", "report")},
        {"from": "gpt", "value": "target report"},
    ]


def test_vqa_train_conversation_structure():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_TRAIN, target_report="target report")
    assert result.conversations == [
        {"from": "human", "value": VQA_TRAIN_EXPECTED},
        {"from": "gpt", "value": "target report"},
    ]


def test_rag_inference_conversations_is_none():
    builder = _default_builder()
    result = builder.build(PromptMode.RAG_INFERENCE, retrieved_report="report")
    assert result.conversations is None


def test_vqa_inference_conversations_is_none():
    builder = _default_builder()
    result = builder.build(PromptMode.VQA_INFERENCE)
    assert result.conversations is None


# ---------------------------------------------------------------------------
# PromptResult immutability
# ---------------------------------------------------------------------------


def test_prompt_result_is_immutable():
    result = PromptResult(text="x", conversations=None, prompt_version="v1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.text = "y"


def test_prompt_builder_config_is_immutable():
    config = PromptBuilderConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.image_token = "[IMG]"


# ---------------------------------------------------------------------------
# PromptBuilderConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"prompt_version": ""},
        {"image_token": ""},
        {"retrieved_report_quote_char": ""},
        {"retrieved_report_quote_char": "''"},
        {"reject_empty_retrieved_report": "yes"},
        {"max_retrieved_report_length": 0},
        {"max_retrieved_report_length": -1},
        {"max_retrieved_report_length": 1.5},
        {"max_retrieved_report_length": True},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        PromptBuilderConfig(**overrides)


def test_config_defaults_are_valid():
    PromptBuilderConfig()  # must not raise
