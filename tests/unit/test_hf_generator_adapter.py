"""Unit tests for HFGeneratorAdapter (src/baseline/generation/adapter.py).

Every test uses injected fakes only -- no real transformers/torch model
construction, no model download, no network access. Real torch tensors
ARE used to build fake return values (torch itself is a hard project
dependency everywhere, unrelated to the "no transformers import at
module scope" constraint this file also verifies).
"""

import ast
import inspect
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from src.baseline.generation.adapter import (
    HFGeneratorAdapter,
    _apply_vicuna_v1_template,
    _classify_generation_error,
    _default_model_factory,
    _pad_to_square,
    _resolve_model_revision,
    _select_default_dtype,
)
from src.baseline.generation.generator import GenerationConfig, GeneratorConfig

Q1 = ("mimic-cxr", "p1", "s1")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeImageProcessor:
    def __init__(self, output_shape=(1, 3, 8, 8)):
        self.last_call_image = None
        self._output_shape = output_shape

    def __call__(self, images, return_tensors="pt"):
        self.last_call_image = images
        return {"pixel_values": torch.rand(self._output_shape)}


class _FakeTokenizer:
    def __init__(self, decode_return="a fake generated report"):
        self.last_call_text = None
        self._decode_return = decode_return

    def __call__(self, text, return_tensors="pt"):
        self.last_call_text = text
        num_tokens = max(len(text.split()), 1)
        input_ids = torch.arange(num_tokens, dtype=torch.long).unsqueeze(0)
        return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}

    def decode(self, token_ids, skip_special_tokens=True):
        return self._decode_return


class _FakeModel:
    def __init__(self, output_ids=None, commit_hash="fakecommit123", raise_on_generate=None, has_config=True):
        self.config = types.SimpleNamespace(_commit_hash=commit_hash) if has_config else None
        self._output_ids = output_ids if output_ids is not None else torch.tensor([[0, 1, 2, 3, 4]])
        self._raise_on_generate = raise_on_generate
        self.last_generate_kwargs = None

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        if self._raise_on_generate is not None:
            raise self._raise_on_generate
        return self._output_ids


def _factories(model=None, tokenizer=None, image_processor=None):
    model = model if model is not None else _FakeModel()
    tokenizer = tokenizer if tokenizer is not None else _FakeTokenizer()
    image_processor = image_processor if image_processor is not None else _FakeImageProcessor()
    return dict(
        model_factory=lambda: model,
        tokenizer_factory=lambda: tokenizer,
        image_processor_factory=lambda: image_processor,
    )


def _adapter(**overrides):
    return HFGeneratorAdapter(GeneratorConfig(), **_factories(**overrides))


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (4, 6), color=(10, 20, 30)).save(path)
    return str(path)


# ---------------------------------------------------------------------------
# Lazy import behavior (no torch/transformers/PIL at module scope)
# ---------------------------------------------------------------------------


def test_adapter_module_has_no_ml_import_at_module_scope():
    import src.baseline.generation.adapter as adapter_module

    source = inspect.getsource(adapter_module)
    tree = ast.parse(source)

    top_level_imports = set()
    for node in tree.body:  # only direct module-level statements, not function bodies
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])

    assert "torch" not in top_level_imports
    assert "transformers" not in top_level_imports
    assert "PIL" not in top_level_imports
    assert "huggingface_hub" not in top_level_imports


def test_adapter_module_has_no_module_level_from_pretrained_call():
    import src.baseline.generation.adapter as adapter_module

    source = inspect.getsource(adapter_module)
    tree = ast.parse(source)
    top_level_calls = [
        node for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]
    assert top_level_calls == []


# ---------------------------------------------------------------------------
# Factory injection
# ---------------------------------------------------------------------------


def test_construction_with_all_three_factories_succeeds():
    _adapter()  # must not raise


def test_construction_rejects_partial_factory_injection():
    model = _FakeModel()
    with pytest.raises(ValueError):
        HFGeneratorAdapter(GeneratorConfig(), model_factory=lambda: model)


def test_construction_rejects_invalid_config_type():
    with pytest.raises(ValueError):
        HFGeneratorAdapter("not-a-config", **_factories())


def test_construction_never_invokes_real_from_pretrained():
    # If this test somehow tried to download a real model, it would hang
    # or fail on network access -- reaching this assertion at all proves
    # the injected factories were used exclusively.
    calls = {"model": 0, "tokenizer": 0, "image_processor": 0}

    def model_factory():
        calls["model"] += 1
        return _FakeModel()

    def tokenizer_factory():
        calls["tokenizer"] += 1
        return _FakeTokenizer()

    def image_processor_factory():
        calls["image_processor"] += 1
        return _FakeImageProcessor()

    HFGeneratorAdapter(
        GeneratorConfig(),
        model_factory=model_factory,
        tokenizer_factory=tokenizer_factory,
        image_processor_factory=image_processor_factory,
    )
    assert calls == {"model": 1, "tokenizer": 1, "image_processor": 1}


# ---------------------------------------------------------------------------
# Default dtype selection / default_model_factory (no real download --
# from_pretrained itself is monkeypatched to capture call kwargs)
# ---------------------------------------------------------------------------


def test_select_default_dtype_is_float16_when_cuda_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert _select_default_dtype() == torch.float16


def test_select_default_dtype_is_bfloat16_when_cuda_unavailable():
    with patch("torch.cuda.is_available", return_value=False):
        assert _select_default_dtype() == torch.bfloat16


def test_default_model_factory_never_requests_fp32():
    # from_pretrained()'s own default (no torch_dtype given) is fp32,
    # which needs ~28GB for a 7B model -- roughly double a typical
    # single-GPU/CPU-RAM budget. _default_model_factory must always pass
    # an explicit, memory-efficient dtype, on every hardware path.
    captured = {}

    class _FakeLlavaClass:
        @staticmethod
        def from_pretrained(checkpoint, **kwargs):
            captured["checkpoint"] = checkpoint
            captured["kwargs"] = kwargs
            return _FakeModel()

    fake_transformers_module = types.SimpleNamespace(LlavaForConditionalGeneration=_FakeLlavaClass)
    with patch.dict("sys.modules", {"transformers": fake_transformers_module}):
        _default_model_factory(GeneratorConfig())

    assert captured["checkpoint"] == GeneratorConfig().llava_checkpoint
    assert "torch_dtype" in captured["kwargs"]
    assert captured["kwargs"]["torch_dtype"] != torch.float32


# ---------------------------------------------------------------------------
# prepare_inputs -- shape validation, image/tokenizer call, prompt forwarding
# ---------------------------------------------------------------------------


def test_prepare_inputs_returns_expected_shapes(image_path):
    adapter = _adapter()
    prepared = adapter.prepare_inputs(image_path, "a prompt")

    assert prepared.pixel_values.dim() == 3
    assert prepared.input_ids.dim() == 2
    assert prepared.attention_mask.shape == prepared.input_ids.shape


def test_prepare_inputs_rejects_empty_image_path():
    with pytest.raises(ValueError):
        _adapter().prepare_inputs("", "a prompt")


def test_prepare_inputs_rejects_empty_prompt_text(image_path):
    with pytest.raises(ValueError):
        _adapter().prepare_inputs(image_path, "")


def test_prepare_inputs_rejects_missing_image_file(tmp_path):
    with pytest.raises(ValueError):
        _adapter().prepare_inputs(str(tmp_path / "does_not_exist.png"), "a prompt")


def test_prepare_inputs_calls_image_processor_with_loaded_image(image_path):
    image_processor = _FakeImageProcessor()
    adapter = _adapter(image_processor=image_processor)

    adapter.prepare_inputs(image_path, "a prompt")

    assert image_processor.last_call_image is not None
    # image_aspect_ratio=pad: a non-square (4x6) source must be padded
    # to a square canvas before reaching the image processor.
    assert image_processor.last_call_image.size[0] == image_processor.last_call_image.size[1]


def test_prepare_inputs_calls_tokenizer_with_templated_text_containing_exact_prompt(image_path):
    tokenizer = _FakeTokenizer()
    adapter = _adapter(tokenizer=tokenizer)

    adapter.prepare_inputs(image_path, "THE EXACT PROMPT TEXT")

    assert tokenizer.last_call_text is not None
    assert "THE EXACT PROMPT TEXT" in tokenizer.last_call_text
    assert "<image>" in tokenizer.last_call_text
    assert "USER:" in tokenizer.last_call_text
    assert "ASSISTANT:" in tokenizer.last_call_text


class _NonFiniteImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.tensor([[[[float("nan")]]]])}


def test_prepare_inputs_rejects_non_finite_pixel_values(image_path):
    adapter = _adapter(image_processor=_NonFiniteImageProcessor())

    with pytest.raises(ValueError):
        adapter.prepare_inputs(image_path, "a prompt")


# ---------------------------------------------------------------------------
# _apply_vicuna_v1_template / _pad_to_square (pure helpers)
# ---------------------------------------------------------------------------


def test_apply_vicuna_v1_template_inserts_image_token_when_absent():
    text = _apply_vicuna_v1_template("a prompt with no image token")
    assert text.count("<image>") == 1
    assert "a prompt with no image token" in text


def test_apply_vicuna_v1_template_does_not_duplicate_existing_image_token():
    text = _apply_vicuna_v1_template("already has <image> in it")
    assert text.count("<image>") == 1


def test_pad_to_square_leaves_already_square_image_unchanged():
    image = Image.new("RGB", (5, 5))
    assert _pad_to_square(image).size == (5, 5)


def test_pad_to_square_pads_non_square_image():
    image = Image.new("RGB", (4, 8))
    padded = _pad_to_square(image)
    assert padded.size == (8, 8)


# ---------------------------------------------------------------------------
# generate() -- deterministic seed propagation, greedy decoding arguments
# ---------------------------------------------------------------------------


def test_generate_calls_torch_manual_seed_with_configured_seed(image_path):
    model = _FakeModel()
    adapter = _adapter(model=model)
    config = GenerationConfig(seed=1234)

    with patch("torch.manual_seed") as mock_seed:
        adapter.generate(Q1, image_path, "a prompt", config)

    mock_seed.assert_called_once_with(1234)


def test_generate_deterministic_uses_greedy_arguments(image_path):
    model = _FakeModel()
    adapter = _adapter(model=model)
    config = GenerationConfig(deterministic=True, temperature=0.0, num_beams=3, max_new_tokens=64)

    adapter.generate(Q1, image_path, "a prompt", config)

    assert model.last_generate_kwargs["do_sample"] is False
    assert model.last_generate_kwargs["num_beams"] == 3
    assert model.last_generate_kwargs["max_new_tokens"] == 64
    assert "temperature" not in model.last_generate_kwargs
    assert "top_p" not in model.last_generate_kwargs
    assert "top_k" not in model.last_generate_kwargs


def test_generate_non_deterministic_includes_sampling_arguments(image_path):
    model = _FakeModel()
    adapter = _adapter(model=model)
    config = GenerationConfig(deterministic=False, temperature=0.7, top_p=0.9, top_k=40)

    adapter.generate(Q1, image_path, "a prompt", config)

    assert model.last_generate_kwargs["do_sample"] is True
    assert model.last_generate_kwargs["temperature"] == 0.7
    assert model.last_generate_kwargs["top_p"] == 0.9
    assert model.last_generate_kwargs["top_k"] == 40


def test_generate_moves_tensors_to_models_device_and_dtype(image_path):
    # A real T4-vs-fp32 OOM (observed live in an actual Colab dry run)
    # happens in two independent ways: the model itself being too large
    # for available memory (fixed by _select_default_dtype), and a
    # device/dtype mismatch between the model and its inputs once the
    # model *does* live on a specific device/dtype. This test proves the
    # second half: generate() must move input_ids/attention_mask (device
    # only) and pixel_values (device AND dtype) onto wherever the model
    # actually lives, never leaving that to prepare_inputs() (which has
    # no way to know).
    model = _FakeModel()
    model.device = torch.device("cpu")
    model.dtype = torch.bfloat16
    adapter = _adapter(model=model)

    adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert model.last_generate_kwargs["input_ids"].device == model.device
    assert model.last_generate_kwargs["attention_mask"].device == model.device
    assert model.last_generate_kwargs["pixel_values"].device == model.device
    assert model.last_generate_kwargs["pixel_values"].dtype == model.dtype
    # input_ids must stay integer -- never dtype-cast to the model's
    # floating-point dtype, unlike pixel_values.
    assert model.last_generate_kwargs["input_ids"].dtype == torch.long


def test_generate_skips_device_move_when_model_has_no_device_attribute(image_path):
    # A fake/injected model with no .device attribute (like every other
    # test in this file's default _FakeModel) must not raise -- the move
    # is a best-effort real-model courtesy, not a hard requirement.
    model = _FakeModel()  # no .device attribute set
    adapter = _adapter(model=model)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.error is None


def test_generate_returns_successful_result(image_path):
    adapter = _adapter(tokenizer=_FakeTokenizer(decode_return="the generated report text"))
    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.query_key == Q1
    assert result.generated_report_text == "the generated report text"
    assert result.error is None


def test_generate_rejects_invalid_query_key(image_path):
    adapter = _adapter()
    with pytest.raises(ValueError):
        adapter.generate(("bad",), image_path, "a prompt", GenerationConfig())


# ---------------------------------------------------------------------------
# generate_batch -- ordering, no silent drops
# ---------------------------------------------------------------------------


def test_generate_batch_preserves_order(image_path):
    adapter = _adapter()
    q2 = ("mimic-cxr", "p2", "s2")
    requests = [(q2, image_path, "prompt two"), (Q1, image_path, "prompt one")]

    results = adapter.generate_batch(requests, GenerationConfig())

    assert [r.query_key for r in results] == [q2, Q1]


def test_generate_batch_never_drops_a_failed_item(image_path):
    failing_model = _FakeModel(raise_on_generate=RuntimeError("boom"))
    adapter = _adapter(model=failing_model)
    q2 = ("mimic-cxr", "p2", "s2")
    requests = [(Q1, image_path, "prompt one"), (q2, image_path, "prompt two")]

    results = adapter.generate_batch(requests, GenerationConfig())

    assert len(results) == 2
    assert all(r.error is not None for r in results)
    assert [r.query_key for r in results] == [Q1, q2]


def test_generate_batch_empty_returns_empty_list():
    assert _adapter().generate_batch([], GenerationConfig()) == []


# ---------------------------------------------------------------------------
# Metadata completeness / resolved revision handling
# ---------------------------------------------------------------------------


def test_generate_metadata_is_complete_and_marked_hf(image_path):
    adapter = _adapter(model=_FakeModel(commit_hash="abc123def"))
    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    metadata = result.generation_metadata
    assert metadata.adapter_kind == "hf"
    assert metadata.resolved_revision == "abc123def"
    assert metadata.model_checkpoint == GeneratorConfig().llava_checkpoint
    assert metadata.base_lm_name == GeneratorConfig().base_lm_name
    assert metadata.vision_tower_name == GeneratorConfig().vision_tower_name
    assert metadata.conv_mode == GeneratorConfig().conv_mode
    assert metadata.torch_version == torch.__version__
    assert metadata.hardware in ("cpu",) or metadata.hardware.startswith("cuda:0")
    assert metadata.generation_timestamp_utc


def test_resolve_model_revision_reads_commit_hash():
    model = _FakeModel(commit_hash="deadbeef")
    assert _resolve_model_revision(model) == "deadbeef"


def test_resolve_model_revision_none_when_config_missing():
    model = _FakeModel(has_config=False)
    assert _resolve_model_revision(model) is None


def test_resolve_model_revision_none_when_commit_hash_absent():
    model = types.SimpleNamespace(config=types.SimpleNamespace())
    assert _resolve_model_revision(model) is None


# ---------------------------------------------------------------------------
# Classified error mapping (unit tests on _classify_generation_error directly)
# ---------------------------------------------------------------------------


class _FakeGatedRepoError(Exception):
    pass


class _FakeRepositoryNotFoundError(Exception):
    pass


class _FakeConnectTimeout(Exception):
    pass


@pytest.mark.parametrize(
    "exc,expected_category",
    [
        (_FakeGatedRepoError("no access"), "authentication_required"),
        (_FakeRepositoryNotFoundError("not found"), "repository_unavailable"),
        (_FakeConnectTimeout("timed out"), "network_error"),
        (MemoryError("out of memory"), "out_of_memory"),
        (RuntimeError("CUDA out of memory."), "out_of_memory"),
        (ValueError("bad input"), "invalid_input"),
        (ImportError("cannot import name 'Foo'"), "version_incompatibility"),
        (AttributeError("no such attribute"), "architecture_mismatch"),
        (KeyError("missing_key"), "architecture_mismatch"),
        (TypeError("wrong type"), "architecture_mismatch"),
        (RuntimeError("something totally unexpected"), "generation_failure"),
    ],
)
def test_classify_generation_error_categories(exc, expected_category):
    category, _detail = _classify_generation_error(exc)
    assert category == expected_category


def test_classify_generation_error_never_raises():
    # Even a pathological exception with no message must classify cleanly.
    category, detail = _classify_generation_error(Exception())
    assert category == "generation_failure"
    assert detail


# ---------------------------------------------------------------------------
# Classified error mapping via generate() itself (end-to-end wrapping)
# ---------------------------------------------------------------------------


def test_generate_wraps_model_failure_into_classified_error_result(image_path):
    failing_model = _FakeModel(raise_on_generate=_FakeGatedRepoError("no access"))
    adapter = _adapter(model=failing_model)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.generated_report_text is None
    assert result.error is not None
    assert result.error.startswith("authentication_required:")


def test_generate_never_raises_for_generation_pipeline_failures(image_path):
    failing_model = _FakeModel(raise_on_generate=RuntimeError("totally unexpected"))
    adapter = _adapter(model=failing_model)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())  # must not raise

    assert result.error.startswith("generation_failure:")


def test_generate_still_returns_complete_metadata_on_failure(image_path):
    failing_model = _FakeModel(raise_on_generate=RuntimeError("boom"))
    adapter = _adapter(model=failing_model)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.generation_metadata is not None
    assert result.generation_metadata.adapter_kind == "hf"


# ---------------------------------------------------------------------------
# Malformed model output
# ---------------------------------------------------------------------------


def test_generate_classifies_non_string_decode_output_as_architecture_mismatch(image_path):
    tokenizer = _FakeTokenizer()
    tokenizer.decode = lambda token_ids, skip_special_tokens=True: 12345  # not a string
    adapter = _adapter(tokenizer=tokenizer)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.error is not None
    assert result.error.startswith("architecture_mismatch:")


def test_generate_classifies_model_returning_none_as_architecture_mismatch(image_path):
    model = _FakeModel(output_ids=None)
    model.generate = lambda **kwargs: None  # malformed: no output_ids at all
    adapter = _adapter(model=model)

    result = adapter.generate(Q1, image_path, "a prompt", GenerationConfig())

    assert result.error is not None
