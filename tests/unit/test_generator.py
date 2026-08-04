"""Unit tests for src/baseline/generation/generator.py."""

import dataclasses
import json
from unittest.mock import MagicMock, call

import pytest

from src.baseline.generation.adapter import GenerationMetadata, GeneratorAdapter, GeneratorResult
from src.baseline.generation.dataset_builder import RAGDatasetRow
from src.baseline.generation.generator import (
    GenerationConfig,
    GenerationRequest,
    GeneratorConfig,
    LLaVAGenerator,
    _generator_result_to_jsonl_row,
)
from src.baseline.generation.mock_adapter import MockGeneratorAdapter
from src.baseline.generation.prompt_builder import PromptBuilder, PromptBuilderConfig, PromptMode, PromptResult

Q1 = ("mimic-cxr", "p1", "s1")
Q2 = ("mimic-cxr", "p2", "s2")


def _metadata(**overrides):
    defaults = dict(
        adapter_kind="mock", model_checkpoint="ckpt", resolved_revision=None,
        base_lm_name="lm", vision_tower_name="vision", lora_checkpoint=None,
        conv_mode="vicuna_v1", git_commit=None, torch_version=None,
        transformers_version=None, hardware="mock",
        generation_timestamp_utc="2026-01-01T00:00:00Z",
    )
    defaults.update(overrides)
    return GenerationMetadata(**defaults)


def _real_generator(generation_config=None):
    return LLaVAGenerator(
        MockGeneratorAdapter(),
        PromptBuilder(PromptBuilderConfig()),
        generation_config or GenerationConfig(),
    )


# ---------------------------------------------------------------------------
# GeneratorConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"llava_checkpoint": ""},
        {"base_lm_name": ""},
        {"vision_tower_name": ""},
        {"adapter_kind": "bogus"},
        {"lora_enabled": "yes"},
        {"lora_enabled": True, "lora_checkpoint": None},
        {"lora_enabled": False, "lora_checkpoint": "/some/lora"},
        {"conv_mode": ""},
        {"tokenizer_name": ""},
    ],
)
def test_generator_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        GeneratorConfig(**overrides)


def test_generator_config_defaults_are_valid():
    GeneratorConfig()  # must not raise


def test_generator_config_allows_lora_enabled_with_checkpoint():
    config = GeneratorConfig(lora_enabled=True, lora_checkpoint="/some/lora")
    assert config.lora_enabled is True
    assert config.lora_checkpoint == "/some/lora"


def test_generator_config_allows_tokenizer_name_none():
    config = GeneratorConfig(tokenizer_name=None)
    assert config.tokenizer_name is None


def test_generator_config_is_immutable():
    config = GeneratorConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.adapter_kind = "hf"


# ---------------------------------------------------------------------------
# GenerationConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"temperature": -0.1},
        {"temperature": "hot"},
        {"temperature": True},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_p": -0.5},
        {"top_k": 0},
        {"top_k": -1},
        {"top_k": 1.5},
        {"max_new_tokens": 0},
        {"max_new_tokens": -1},
        {"max_new_tokens": 1.5},
        {"num_beams": 0},
        {"num_beams": -1},
        {"seed": "not-an-int"},
        {"seed": True},
        {"deterministic": "yes"},
    ],
)
def test_generation_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        GenerationConfig(**overrides)


def test_generation_config_defaults_are_valid():
    GenerationConfig()  # must not raise


def test_generation_config_deterministic_true_requires_temperature_zero():
    with pytest.raises(ValueError):
        GenerationConfig(deterministic=True, temperature=0.7)


def test_generation_config_deterministic_true_forbids_top_p():
    with pytest.raises(ValueError):
        GenerationConfig(deterministic=True, temperature=0.0, top_p=0.9)


def test_generation_config_deterministic_true_forbids_top_k():
    with pytest.raises(ValueError):
        GenerationConfig(deterministic=True, temperature=0.0, top_k=50)


def test_generation_config_deterministic_false_requires_positive_temperature():
    with pytest.raises(ValueError):
        GenerationConfig(deterministic=False, temperature=0.0)


def test_generation_config_deterministic_false_with_sampling_params_is_valid():
    config = GenerationConfig(deterministic=False, temperature=0.7, top_p=0.9, top_k=50)
    assert config.temperature == 0.7
    assert config.top_p == 0.9
    assert config.top_k == 50


def test_generation_config_is_immutable():
    config = GenerationConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7


def test_generation_config_seed_always_present_even_at_temperature_zero():
    config = GenerationConfig()
    assert config.seed == 42
    assert config.temperature == 0.0


# ---------------------------------------------------------------------------
# LLaVAGenerator construction validation
# ---------------------------------------------------------------------------


def test_llava_generator_rejects_non_adapter():
    with pytest.raises(ValueError):
        LLaVAGenerator("not-an-adapter", PromptBuilder(PromptBuilderConfig()), GenerationConfig())


def test_llava_generator_rejects_non_prompt_builder():
    with pytest.raises(ValueError):
        LLaVAGenerator(MockGeneratorAdapter(), "not-a-prompt-builder", GenerationConfig())


def test_llava_generator_rejects_non_generation_config():
    with pytest.raises(ValueError):
        LLaVAGenerator(MockGeneratorAdapter(), PromptBuilder(PromptBuilderConfig()), "not-a-config")


def test_llava_generator_accepts_valid_collaborators():
    _real_generator()  # must not raise


# ---------------------------------------------------------------------------
# generate_one -- mode/prompt validation (incompatible PromptMode,
# missing retrieved evidence)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("train_mode", [PromptMode.RAG_TRAIN, PromptMode.VQA_TRAIN])
def test_generate_one_rejects_train_modes(train_mode):
    generator = _real_generator()
    with pytest.raises(ValueError):
        generator.generate_one(Q1, image_path="/img.png", mode=train_mode)


def test_generate_one_rejects_invalid_mode_type():
    generator = _real_generator()
    with pytest.raises(ValueError):
        generator.generate_one(Q1, image_path="/img.png", mode="rag_inference")


def test_generate_one_rag_inference_requires_retrieved_report():
    generator = _real_generator()
    with pytest.raises(ValueError):
        generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.RAG_INFERENCE)


def test_generate_one_vqa_inference_forbids_retrieved_report():
    generator = _real_generator()
    with pytest.raises(ValueError):
        generator.generate_one(
            Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE, retrieved_report="not allowed"
        )


# ---------------------------------------------------------------------------
# generate_one -- integration with real PromptBuilder + real MockGeneratorAdapter
# ---------------------------------------------------------------------------


def test_generate_one_rag_inference_succeeds_with_retrieved_report():
    generator = _real_generator()
    result = generator.generate_one(
        Q1, image_path="/img.png", mode=PromptMode.RAG_INFERENCE, retrieved_report="a related report"
    )
    assert result.query_key == Q1
    assert result.generated_report_text is not None
    assert result.error is None


def test_generate_one_vqa_inference_succeeds_without_retrieved_report():
    generator = _real_generator()
    result = generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)
    assert result.query_key == Q1
    assert result.generated_report_text is not None


def test_generate_one_is_deterministic_for_same_seed():
    config = GenerationConfig(seed=42)
    generator_a = _real_generator(config)
    generator_b = _real_generator(config)

    result_a = generator_a.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)
    result_b = generator_b.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)

    assert result_a.generated_report_text == result_b.generated_report_text


# ---------------------------------------------------------------------------
# prompt forwarding / PromptBuilder invocation / adapter invocation (mocked collaborators)
# ---------------------------------------------------------------------------


def _mock_prompt_builder(text="built prompt text"):
    builder = MagicMock(spec=PromptBuilder)
    builder.build.return_value = PromptResult(text=text, conversations=None, prompt_version="official_v1")
    return builder


def _mock_adapter(query_key=Q1):
    adapter = MagicMock(spec=GeneratorAdapter)
    adapter.generate.return_value = GeneratorResult(
        query_key=query_key, generated_report_text="mocked text", generation_metadata=_metadata()
    )
    return adapter


def test_generate_one_invokes_prompt_builder_with_correct_arguments():
    prompt_builder = _mock_prompt_builder()
    adapter = _mock_adapter()
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.RAG_INFERENCE, retrieved_report="ev")

    prompt_builder.build.assert_called_once_with(PromptMode.RAG_INFERENCE, retrieved_report="ev")


def test_generate_one_forwards_prompt_builder_text_to_adapter():
    prompt_builder = _mock_prompt_builder(text="THE EXACT PROMPT TEXT")
    adapter = _mock_adapter()
    generation_config = GenerationConfig()
    generator = LLaVAGenerator(adapter, prompt_builder, generation_config)

    generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)

    adapter.generate.assert_called_once_with(Q1, "/img.png", "THE EXACT PROMPT TEXT", generation_config)


def test_generate_one_returns_adapters_result_unmodified():
    prompt_builder = _mock_prompt_builder()
    expected = GeneratorResult(query_key=Q1, generated_report_text="exact text", generation_metadata=_metadata())
    adapter = MagicMock(spec=GeneratorAdapter)
    adapter.generate.return_value = expected
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    result = generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)

    assert result is expected


# ---------------------------------------------------------------------------
# Malformed GeneratorResult rejection
# ---------------------------------------------------------------------------


def test_generate_one_rejects_result_with_mismatched_query_key():
    prompt_builder = _mock_prompt_builder()
    adapter = _mock_adapter(query_key=Q2)  # adapter returns the WRONG query_key
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    with pytest.raises(ValueError):
        generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)


def test_generate_one_rejects_non_generator_result_from_adapter():
    prompt_builder = _mock_prompt_builder()
    adapter = MagicMock(spec=GeneratorAdapter)
    adapter.generate.return_value = "not a GeneratorResult"
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    with pytest.raises(ValueError):
        generator.generate_one(Q1, image_path="/img.png", mode=PromptMode.VQA_INFERENCE)


# ---------------------------------------------------------------------------
# generate_many -- ordering, duplicate keys, batch behavior, empty batch
# ---------------------------------------------------------------------------


def test_generate_many_preserves_request_order():
    generator = _real_generator()
    requests = [
        GenerationRequest(query_key=Q2, image_path="/img2.png", mode=PromptMode.VQA_INFERENCE),
        GenerationRequest(query_key=Q1, image_path="/img1.png", mode=PromptMode.VQA_INFERENCE),
    ]
    results = generator.generate_many(requests)
    assert [r.query_key for r in results] == [Q2, Q1]


def test_generate_many_empty_returns_empty_list():
    generator = _real_generator()
    assert generator.generate_many([]) == []


def test_generate_many_rejects_duplicate_query_keys():
    generator = _real_generator()
    requests = [
        GenerationRequest(query_key=Q1, image_path="/img1.png", mode=PromptMode.VQA_INFERENCE),
        GenerationRequest(query_key=Q1, image_path="/img1-dup.png", mode=PromptMode.VQA_INFERENCE),
    ]
    with pytest.raises(ValueError):
        generator.generate_many(requests)


def test_generate_many_uses_a_single_batched_adapter_call():
    prompt_builder = _mock_prompt_builder()
    adapter = MagicMock(spec=GeneratorAdapter)
    adapter.generate_batch.return_value = [
        GeneratorResult(query_key=Q1, generated_report_text="t1", generation_metadata=_metadata()),
        GeneratorResult(query_key=Q2, generated_report_text="t2", generation_metadata=_metadata()),
    ]
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    requests = [
        GenerationRequest(query_key=Q1, image_path="/img1.png", mode=PromptMode.VQA_INFERENCE),
        GenerationRequest(query_key=Q2, image_path="/img2.png", mode=PromptMode.VQA_INFERENCE),
    ]
    results = generator.generate_many(requests)

    adapter.generate_batch.assert_called_once()
    adapter.generate.assert_not_called()
    assert [r.query_key for r in results] == [Q1, Q2]


def test_generate_many_rejects_result_count_mismatch():
    prompt_builder = _mock_prompt_builder()
    adapter = MagicMock(spec=GeneratorAdapter)
    adapter.generate_batch.return_value = [
        GeneratorResult(query_key=Q1, generated_report_text="t1", generation_metadata=_metadata()),
    ]  # only 1 result for 2 requests
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    requests = [
        GenerationRequest(query_key=Q1, image_path="/img1.png", mode=PromptMode.VQA_INFERENCE),
        GenerationRequest(query_key=Q2, image_path="/img2.png", mode=PromptMode.VQA_INFERENCE),
    ]
    with pytest.raises(ValueError):
        generator.generate_many(requests)


def test_generate_many_matches_generate_one_output_with_real_adapter():
    config = GenerationConfig(seed=99)
    generator_batch = _real_generator(config)
    generator_single = _real_generator(config)

    requests = [
        GenerationRequest(query_key=Q1, image_path="/img1.png", mode=PromptMode.VQA_INFERENCE),
        GenerationRequest(query_key=Q2, image_path="/img2.png", mode=PromptMode.RAG_INFERENCE, retrieved_report="ev"),
    ]
    batch_results = generator_batch.generate_many(requests)
    individual_results = [
        generator_single.generate_one(
            r.query_key, image_path=r.image_path, mode=r.mode, retrieved_report=r.retrieved_report
        )
        for r in requests
    ]

    assert [r.generated_report_text for r in batch_results] == [
        r.generated_report_text for r in individual_results
    ]


# ---------------------------------------------------------------------------
# generate_from_rag_row / generate_many_from_rag_rows (RAGDatasetBuilder orchestration)
# ---------------------------------------------------------------------------


def _rag_row(**overrides):
    defaults = dict(
        query_key=Q1, retrieved_key=("mimic-cxr", "p9", "s9"), image_path="/img.png",
        retrieved_report_text="retrieved text", target_report_text="target text",
        rank_selected=0, num_candidates_rejected_self_study=0,
        num_candidates_rejected_self_patient=0, num_candidates_rejected_short_text=0,
        fallback_used=False, excluded=False, strict_patient_isolation=True,
        reproduce_official_bug=False, config_version="1.0",
    )
    defaults.update(overrides)
    return RAGDatasetRow(**defaults)


def test_generate_from_rag_row_uses_rag_inference_when_not_excluded():
    prompt_builder = _mock_prompt_builder()
    adapter = _mock_adapter()
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    generator.generate_from_rag_row(_rag_row())

    prompt_builder.build.assert_called_once_with(PromptMode.RAG_INFERENCE, retrieved_report="retrieved text")


def test_generate_from_rag_row_uses_vqa_inference_when_excluded():
    prompt_builder = _mock_prompt_builder()
    adapter = _mock_adapter()
    generator = LLaVAGenerator(adapter, prompt_builder, GenerationConfig())

    excluded_row = _rag_row(
        retrieved_key=None, retrieved_report_text=None, rank_selected=None, excluded=True,
    )
    generator.generate_from_rag_row(excluded_row)

    prompt_builder.build.assert_called_once_with(PromptMode.VQA_INFERENCE, retrieved_report=None)


def test_generate_many_from_rag_rows_preserves_order_and_mixes_modes():
    generator = _real_generator()
    rows = [
        _rag_row(query_key=Q2, retrieved_key=None, retrieved_report_text=None, rank_selected=None, excluded=True),
        _rag_row(query_key=Q1),
    ]
    results = generator.generate_many_from_rag_rows(rows)
    assert [r.query_key for r in results] == [Q2, Q1]


# ---------------------------------------------------------------------------
# _generator_result_to_jsonl_row -- pure mapping function
# ---------------------------------------------------------------------------


def _row_kwargs(**overrides):
    result = GeneratorResult(
        query_key=Q1, generated_report_text="the report", generation_metadata=_metadata(),
    )
    defaults = dict(
        result=result,
        image_path="/img.png",
        retrieved_evidence_key=("mimic-cxr", "p9", "s9"),
        prompt_version="official_v1",
        paper_version="factmm_rag_naacl2025",
        implementation_version="1.0",
        generation_config=GenerationConfig(),
    )
    defaults.update(overrides)
    return defaults


def test_generator_result_to_jsonl_row_success_shape():
    row = _generator_result_to_jsonl_row(**_row_kwargs())

    assert row["query_key"] == list(Q1)
    assert row["image_path"] == "/img.png"
    assert row["retrieved_evidence_key"] == ["mimic-cxr", "p9", "s9"]
    assert row["prompt_version"] == "official_v1"
    assert row["paper_version"] == "factmm_rag_naacl2025"
    assert row["implementation_version"] == "1.0"
    assert row["generated_report_text"] == "the report"
    assert row["error"] is None
    assert row["generation_config"] == {
        "temperature": 0.0, "top_p": None, "top_k": None, "max_new_tokens": 256,
        "num_beams": 1, "seed": 42, "deterministic": True,
    }
    assert row["generation_metadata"] == dataclasses.asdict(_metadata())


def test_generator_result_to_jsonl_row_error_path_shape():
    error_result = GeneratorResult(
        query_key=Q1, generated_report_text=None, generation_metadata=_metadata(), error="model failed"
    )
    row = _generator_result_to_jsonl_row(**_row_kwargs(result=error_result))

    assert row["generated_report_text"] is None
    assert row["error"] == "model failed"


def test_generator_result_to_jsonl_row_retrieved_evidence_key_none():
    row = _generator_result_to_jsonl_row(**_row_kwargs(retrieved_evidence_key=None))
    assert row["retrieved_evidence_key"] is None


def test_generator_result_to_jsonl_row_is_pure_and_deterministic():
    kwargs = _row_kwargs()
    first = _generator_result_to_jsonl_row(**kwargs)
    second = _generator_result_to_jsonl_row(**kwargs)
    assert first == second


def test_generator_result_to_jsonl_row_is_json_serializable_and_resume_safe():
    row = _generator_result_to_jsonl_row(**_row_kwargs())
    line = json.dumps(row, ensure_ascii=False)
    reloaded = json.loads(line)

    assert reloaded == row
    # A resume loader must be able to recover the exact QueryKey tuple
    # from the serialized list -- the same round-trip pattern used by
    # every other resumable JSONL producer in this project.
    assert tuple(reloaded["query_key"]) == Q1


def test_generator_result_to_jsonl_row_does_not_mutate_inputs():
    kwargs = _row_kwargs()
    config_before = dataclasses.replace(kwargs["generation_config"])
    _generator_result_to_jsonl_row(**kwargs)
    assert kwargs["generation_config"] == config_before
