"""Generator adapter interface, shared generation-result types, and the
real HFGeneratorAdapter (Milestone 2.5).

Responsibility: define the abstract seam between `LLaVAGenerator`
(`generator.py`) and any concrete generation backend -- real
(`HFGeneratorAdapter`, this module) or fake (`MockGeneratorAdapter`,
`mock_adapter.py`). Mirrors `MultiModalRetriever`'s dependency-injection
discipline (see `src/baseline/retrieval/model.py`), but as a
fully-constructed object handed to the orchestrator, not raw factories
consumed internally.

Built from docs/milestone_2_5_generator_contract.md SS8, with disclosed,
necessary refinements:

1. (Cell 27) SS8.1's `generate`/`generate_batch` signatures as written
   have no way to supply the `query_key` that `GeneratorResult.query_key`
   (SS8.4) requires. This module adds an explicit `query_key: QueryKey`
   parameter to `generate()` and widens `generate_batch`'s per-request
   tuple to `Tuple[QueryKey, str, str]` (query_key, image_path, prompt).

2. (Cell 29, this cell) `torch` is no longer imported at module scope
   (Cell 27 imported it there for PreparedGeneratorInputs' field type
   annotations only -- `from __future__ import annotations` already
   makes every annotation a lazy, unevaluated string, so the import was
   never actually required at runtime). Removing it lets this module
   satisfy "no torch/transformers import at module scope," a real,
   test-proven requirement for HFGeneratorAdapter's lazy-import
   discipline -- every real torch/transformers/PIL/huggingface_hub call
   below happens inside a function body, never at import time.

3. HFGeneratorAdapter's vicuna_v1 conversation template
   (`_apply_vicuna_v1_template`) is a disclosed best-effort
   REASONABLE_INFERENCE, not a verified byte-exact reproduction: per the
   contract SS2, `llava/conversation.py` is not vendored in this repo
   and was not independently re-read against the pinned LLaVA commit.
   The widely-documented public Vicuna v1.1 turn structure (system
   prompt + "USER: ... ASSISTANT:") is used here, not a fabricated
   claim of exactness.

4. Error classification (`_classify_generation_error`) is Milestone
   2.4G's own `_classify_hf_error` reused verbatim for its five original
   categories (authentication_required, repository_unavailable,
   network_error, version_incompatibility, architecture_mismatch), with
   three additions this cell's generation-time (not just load-time)
   failures need: out_of_memory, invalid_input, and generation_failure
   (replacing 2.4G's generic "unknown_error" fallback name, which is not
   one of this cell's specified categories).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Literal, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.retrieval.index import _resolve_git_commit

_ADAPTER_KINDS: Tuple[str, ...] = ("hf", "mock")


@dataclass(frozen=True)
class PreparedGeneratorInputs:
    """Model-ready tensors produced by `GeneratorAdapter.prepare_inputs()`.

    The sole seam that turns a raw (image_path, prompt_text) pair into
    fully model-ready tensors -- real image preprocessing and real
    tokenization/conversation-template application for
    `HFGeneratorAdapter`; deterministic, structurally-shaped fakes for
    `MockGeneratorAdapter`. Intentionally not validated beyond dataclass
    field typing -- shape/dtype conventions are adapter-specific and
    verified by each adapter's own tests, not enforced generically here.

    pixel_values is unbatched (C, H, W) -- adding the batch dimension
    before a real model call is generate()'s job, not prepare_inputs()'s
    (matching MockGeneratorAdapter's own established shape convention).
    """

    pixel_values: "torch.Tensor"
    input_ids: "torch.Tensor"
    attention_mask: "torch.Tensor"


@dataclass(frozen=True)
class GenerationMetadata:
    """Model/runtime provenance for one generation call.

    Scoped strictly to provenance -- never duplicates GenerationConfig's
    decoding parameters (temperature, seed, ...), which are referenced
    alongside this metadata in the output row, not copied into it.
    Identical field shape across every adapter kind: HFGeneratorAdapter
    populates real values, MockGeneratorAdapter populates deterministic
    placeholders (adapter_kind="mock", resolved_revision=None,
    hardware="mock", torch_version=None, transformers_version=None) --
    no adapter-dependent field is ever silently missing from the shape.
    """

    adapter_kind: Literal["hf", "mock"]
    model_checkpoint: str
    resolved_revision: Optional[str]
    base_lm_name: str
    vision_tower_name: str
    lora_checkpoint: Optional[str]
    conv_mode: str
    git_commit: Optional[str]
    torch_version: Optional[str]
    transformers_version: Optional[str]
    hardware: str
    generation_timestamp_utc: str

    def __post_init__(self) -> None:
        if self.adapter_kind not in _ADAPTER_KINDS:
            raise ValueError(
                f"adapter_kind must be one of {_ADAPTER_KINDS}, got {self.adapter_kind!r}"
            )
        for field_name in (
            "model_checkpoint", "base_lm_name", "vision_tower_name",
            "conv_mode", "hardware", "generation_timestamp_utc",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class GeneratorResult:
    """The generation outcome for one query -- nothing else in the
    pipeline constructs one ad hoc.

    Exactly one of generated_report_text/error is set: a successful
    generation carries report text and no error; a failed one carries
    an error message and no report text. Never both, never neither --
    there is no third, ambiguous state.
    """

    query_key: QueryKey
    generated_report_text: Optional[str]
    generation_metadata: GenerationMetadata
    error: Optional[str] = None

    def __post_init__(self) -> None:
        text_set = self.generated_report_text is not None
        error_set = self.error is not None
        if text_set and error_set:
            raise ValueError(
                "generated_report_text and error must not both be set -- a "
                "GeneratorResult is either a success or a failure, never both"
            )
        if not text_set and not error_set:
            raise ValueError(
                "at least one of generated_report_text or error must be set -- "
                "a GeneratorResult cannot represent neither a success nor a failure"
            )


class GeneratorAdapter(abc.ABC):
    """Abstract seam between LLaVAGenerator and a concrete generation backend.

    See module docstring for the disclosed query_key signature
    refinement relative to the contract's SS8.1 text.
    """

    @abc.abstractmethod
    def prepare_inputs(self, image_path: str, prompt_text: str) -> PreparedGeneratorInputs:
        """Turns a raw (image_path, prompt_text) pair into model-ready tensors.

        Exposed as its own separately-testable abstract method (not a
        private helper) so a test can assert correct shapes/placeholder
        alignment without invoking any generation logic, and so
        orchestration tests can spy on this being called with the exact
        prompt PromptBuilder.build() produced.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def generate(
        self,
        query_key: QueryKey,
        image_path: str,
        prompt: str,
        generation_config,
    ) -> GeneratorResult:
        """Generates one report. Must call prepare_inputs() internally
        as its first step (see module docstring)."""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_batch(
        self,
        requests: Sequence[Tuple[QueryKey, str, str]],
        generation_config,
    ) -> List[GeneratorResult]:
        """Generates one report per (query_key, image_path, prompt)
        request, preserving request order in the returned list."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HFGeneratorAdapter -- real adapter, lazy-imported dependencies only
# ---------------------------------------------------------------------------

_IMAGE_TOKEN = "<image>"

# REASONABLE_INFERENCE, not byte-verified against the pinned LLaVA commit
# -- see module docstring, refinement 3.
_VICUNA_V1_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_valid_query_key(query_key) -> bool:
    return (
        isinstance(query_key, tuple)
        and len(query_key) == 3
        and all(isinstance(part, str) and part for part in query_key)
    )


def _apply_vicuna_v1_template(prompt_text: str, *, image_token: str = _IMAGE_TOKEN) -> str:
    """Best-effort vicuna_v1 formatting (see module docstring, refinement 3).

    RAG_INFERENCE/VQA_INFERENCE prompt text (PromptBuilder, SS5.3) never
    contains an image token literal -- inserting one here, not in the
    prompt text, is this adapter's own documented responsibility (SS8.1).
    """
    user_turn = prompt_text if image_token in prompt_text else f"{image_token}\n{prompt_text}"
    return f"{_VICUNA_V1_SYSTEM_PROMPT} USER: {user_turn} ASSISTANT:"


def _pad_to_square(image):
    """Pads a PIL RGB image to a square canvas (image_aspect_ratio=pad).

    Background color is CLIP's own per-channel mean (REASONABLE_INFERENCE,
    matching the official LLaVA `expand2square` convention of padding with
    the image processor's mean rather than black) -- never a plain
    center-crop of a non-square source image, which would silently
    discard image content.
    """
    width, height = image.size
    if width == height:
        return image
    from PIL import Image

    size = max(width, height)
    background_color = tuple(round(c * 255) for c in (0.48145466, 0.4578275, 0.40821073))
    padded = Image.new("RGB", (size, size), background_color)
    padded.paste(image, ((size - width) // 2, (size - height) // 2))
    return padded


def _resolve_model_revision(model) -> Optional[str]:
    """Best-effort real HF revision hash from an already-loaded model.

    transformers' PreTrainedModel.from_pretrained sets config._commit_hash
    when loading from the Hub in recent versions -- read defensively via
    getattr chains, never raising, returning None (matching
    MockGeneratorAdapter's resolved_revision=None convention) when
    unavailable (e.g. an injected test fake with no such attribute).
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    return getattr(config, "_commit_hash", None)


def _classify_generation_error(exc: Exception) -> Tuple[str, str]:
    """Classifies exc into one of 8 categories -- never a single generic
    bucket. See module docstring, refinement 4.
    """
    seen = []
    cur = exc
    depth = 0
    while cur is not None and depth < 6:
        seen.append(cur)
        cur = getattr(cur, "__cause__", None)
        depth += 1

    type_names = {type(e).__module__ + "." + type(e).__name__ for e in seen}
    messages = " | ".join(str(e) for e in seen).lower()

    def has_type(substr: str) -> bool:
        return any(substr in name for name in type_names)

    # Type-specific checks run BEFORE generic message-content sniffing --
    # see Milestone 2.4G's own classifier for why (a network-layer
    # failure can carry text identical to an auth failure).
    if has_type("GatedRepoError") or has_type("LocalTokenNotFoundError"):
        return "authentication_required", f"{type(exc).__name__}: {exc}"

    if has_type("RepositoryNotFoundError") or has_type("RevisionNotFoundError"):
        return "repository_unavailable", f"{type(exc).__name__}: {exc}"

    if (
        has_type("ProxyError")
        or has_type("ConnectError")
        or has_type("ConnectTimeout")
        or has_type("ConnectionError")
        or has_type("TimeoutException")
        or has_type("Timeout")
    ):
        return "network_error", f"{type(exc).__name__}: {exc}"

    if (
        has_type("OutOfMemoryError")
        or isinstance(exc, MemoryError)
        or "out of memory" in messages
        or "cuda out of memory" in messages
    ):
        return "out_of_memory", f"{type(exc).__name__}: {exc}"

    # This project's own convention (PromptBuilder, RAGDatasetBuilder,
    # GeneratorConfig, GenerationConfig, ...): ValueError == the caller
    # passed something invalid -- a bad image path, a non-finite tensor,
    # a malformed prompt/config. Checked before the generic HF-error
    # message sniffing below so a caller mistake is never relabeled as
    # an infrastructure problem.
    if isinstance(exc, ValueError):
        return "invalid_input", f"{type(exc).__name__}: {exc}"

    if any(
        tok in messages
        for tok in ("401", "403", "authentication", "access token", "gated repo", "is gated")
    ):
        return "authentication_required", f"{type(exc).__name__}: {exc}"

    if "404" in messages and ("config.json" in messages or "not a valid model identifier" in messages):
        return "repository_unavailable", f"{type(exc).__name__}: {exc}"

    if any(
        tok in messages
        for tok in (
            "couldn't connect",
            "name or service not known",
            "max retries exceeded",
            "temporary failure in name resolution",
            "connection refused",
        )
    ):
        return "network_error", f"{type(exc).__name__}: {exc}"

    if isinstance(exc, ImportError) or any(
        tok in messages
        for tok in (
            "unexpected keyword argument", "not supported for this version",
            "requires a newer version", "cannot import name",
        )
    ):
        return "version_incompatibility", f"{type(exc).__name__}: {exc}"

    if isinstance(exc, (AttributeError, KeyError, TypeError)):
        return "architecture_mismatch", f"{type(exc).__name__}: {exc}"

    return "generation_failure", f"{type(exc).__name__}: {exc}"


def _select_default_dtype():
    """fp16 on GPU, bf16 on CPU -- never the from_pretrained() default of
    fp32, which needs ~28GB for a 7B model (roughly double what a single
    consumer/Colab GPU typically has, and what real CPU-only Colab RAM
    typically has). fp16 matmul on CPU is often unsupported or very slow
    (REASONABLE_INFERENCE, general PyTorch knowledge), so bf16 -- itself
    reasonably supported on modern CPUs -- is used there instead;
    real hardware discovers the actual dtype it received via
    GenerationMetadata (not itself dtype-typed, but the resolved model
    can be introspected by the caller if needed).
    """
    import torch

    return torch.float16 if torch.cuda.is_available() else torch.bfloat16


def _default_model_factory(config):
    from transformers import LlavaForConditionalGeneration

    return LlavaForConditionalGeneration.from_pretrained(
        config.llava_checkpoint, torch_dtype=_select_default_dtype()
    )


def _default_tokenizer_factory(config):
    from transformers import AutoTokenizer

    tokenizer_name = config.tokenizer_name or config.base_lm_name
    return AutoTokenizer.from_pretrained(tokenizer_name)


def _default_image_processor_factory(config):
    from transformers import CLIPImageProcessor

    return CLIPImageProcessor.from_pretrained(config.vision_tower_name)


class HFGeneratorAdapter(GeneratorAdapter):
    """Real adapter: constructs the real vision tower/tokenizer/base LM
    per GeneratorConfig, via the same all-factories-or-none injection
    pattern as MultiModalRetriever -- real transformers/torch/PIL calls
    are only ever attempted when factories are omitted, and only inside
    method bodies, never at module import time.
    """

    def __init__(
        self,
        config,
        *,
        model_factory: Optional[Callable[[], object]] = None,
        tokenizer_factory: Optional[Callable[[], object]] = None,
        image_processor_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        # Lazy, deferred import -- GeneratorConfig lives in generator.py,
        # which itself imports GeneratorAdapter/GeneratorResult from this
        # module; a module-level import here would be circular. See
        # module docstring, refinement 2, for why this is safe: by the
        # time __init__ actually runs, both modules are fully loaded.
        from src.baseline.generation.generator import GeneratorConfig

        if not isinstance(config, GeneratorConfig):
            raise ValueError(f"config must be a GeneratorConfig instance, got {type(config)!r}")

        factories = (model_factory, tokenizer_factory, image_processor_factory)
        provided = [f is not None for f in factories]
        if any(provided) and not all(provided):
            raise ValueError(
                "model_factory, tokenizer_factory, and image_processor_factory "
                "must all be provided together, or all omitted -- partial "
                "injection is a test-only escape hatch, not general DI."
            )

        if model_factory is None:
            model_factory = lambda: _default_model_factory(config)
            tokenizer_factory = lambda: _default_tokenizer_factory(config)
            image_processor_factory = lambda: _default_image_processor_factory(config)

        self._config = config
        self._model = model_factory()
        self._tokenizer = tokenizer_factory()
        self._image_processor = image_processor_factory()
        self._resolved_revision = _resolve_model_revision(self._model)

    def prepare_inputs(self, image_path: str, prompt_text: str) -> PreparedGeneratorInputs:
        if not image_path:
            raise ValueError("image_path must be a non-empty string")
        if not prompt_text:
            raise ValueError("prompt_text must be a non-empty string")

        import torch
        from PIL import Image, UnidentifiedImageError

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Failed to load image at {image_path!r}: {exc}") from exc

        padded_image = _pad_to_square(image)  # image_aspect_ratio=pad

        image_inputs = self._image_processor(images=padded_image, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]
        if pixel_values.dim() == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        if pixel_values.dim() != 3:
            raise ValueError(
                f"pixel_values must be rank-3 (C, H, W) after unbatching, got "
                f"shape {tuple(pixel_values.shape)}"
            )
        if not torch.isfinite(pixel_values).all():
            raise ValueError("image preprocessing produced non-finite pixel_values")

        templated_text = _apply_vicuna_v1_template(prompt_text, image_token=_IMAGE_TOKEN)
        token_inputs = self._tokenizer(templated_text, return_tensors="pt")
        input_ids = token_inputs["input_ids"]
        attention_mask = token_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be rank-2 (1, L), got shape {tuple(input_ids.shape)}")

        return PreparedGeneratorInputs(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    def _build_metadata(self) -> GenerationMetadata:
        import torch

        try:
            import transformers as _transformers

            transformers_version = getattr(_transformers, "__version__", None)
        except ImportError:
            transformers_version = None

        hardware = "cpu"
        if torch.cuda.is_available():
            try:
                hardware = f"cuda:0 ({torch.cuda.get_device_name(0)})"
            except Exception:  # noqa: BLE001 -- metadata must never fail generation
                hardware = "cuda:0"

        return GenerationMetadata(
            adapter_kind="hf",
            model_checkpoint=self._config.llava_checkpoint,
            resolved_revision=self._resolved_revision,
            base_lm_name=self._config.base_lm_name,
            vision_tower_name=self._config.vision_tower_name,
            lora_checkpoint=self._config.lora_checkpoint,
            conv_mode=self._config.conv_mode,
            git_commit=_resolve_git_commit(),
            torch_version=torch.__version__,
            transformers_version=transformers_version,
            hardware=hardware,
            generation_timestamp_utc=_utc_now_iso(),
        )

    def generate(
        self,
        query_key: QueryKey,
        image_path: str,
        prompt: str,
        generation_config,
    ) -> GeneratorResult:
        if not _is_valid_query_key(query_key):
            raise ValueError(
                f"query_key must be a (dataset, patient_id, study_id) tuple of "
                f"non-empty strings, got {query_key!r}"
            )

        try:
            import torch

            seed = getattr(generation_config, "seed", None)
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"generation_config.seed must be an int, got {seed!r}")
            torch.manual_seed(seed)

            prepared = self.prepare_inputs(image_path, prompt)

            gen_kwargs = dict(
                input_ids=prepared.input_ids,
                attention_mask=prepared.attention_mask,
                pixel_values=prepared.pixel_values.unsqueeze(0),
                max_new_tokens=generation_config.max_new_tokens,
                num_beams=generation_config.num_beams,
                do_sample=not generation_config.deterministic,
            )
            if not generation_config.deterministic:
                gen_kwargs["temperature"] = generation_config.temperature
                if generation_config.top_p is not None:
                    gen_kwargs["top_p"] = generation_config.top_p
                if generation_config.top_k is not None:
                    gen_kwargs["top_k"] = generation_config.top_k

            # prepare_inputs() has no knowledge of where the model actually
            # lives (CPU, a specific CUDA device) or what dtype it was
            # loaded in (see _select_default_dtype) -- moving tensors here,
            # right before the real forward pass, is generate()'s job, not
            # prepare_inputs()'s. input_ids/attention_mask move device only
            # (never dtype-cast -- they must stay integer); pixel_values
            # moves both device and dtype (the vision tower's own weights
            # are in the model's dtype, e.g. fp16, and pixel_values must
            # match or the forward pass raises a dtype-mismatch error).
            model_device = getattr(self._model, "device", None)
            model_dtype = getattr(self._model, "dtype", None)
            if model_device is not None:
                gen_kwargs["input_ids"] = gen_kwargs["input_ids"].to(model_device)
                gen_kwargs["attention_mask"] = gen_kwargs["attention_mask"].to(model_device)
                pixel_values_on_device = gen_kwargs["pixel_values"].to(model_device)
                if model_dtype is not None:
                    pixel_values_on_device = pixel_values_on_device.to(model_dtype)
                gen_kwargs["pixel_values"] = pixel_values_on_device

            with torch.no_grad():
                output_ids = self._model.generate(**gen_kwargs)

            new_token_ids = output_ids[0][prepared.input_ids.shape[-1]:]
            generated_text = self._tokenizer.decode(new_token_ids, skip_special_tokens=True)
            if not isinstance(generated_text, str):
                raise TypeError(
                    f"Decoded generation output is not a string: {type(generated_text)!r}"
                )
            generated_text = generated_text.strip()

            return GeneratorResult(
                query_key=query_key,
                generated_report_text=generated_text,
                generation_metadata=self._build_metadata(),
                error=None,
            )
        except Exception as exc:  # noqa: BLE001 -- classified, never left to propagate raw
            category, detail = _classify_generation_error(exc)
            return GeneratorResult(
                query_key=query_key,
                generated_report_text=None,
                generation_metadata=self._build_metadata(),
                error=f"{category}: {detail}",
            )

    def generate_batch(
        self,
        requests: Sequence[Tuple[QueryKey, str, str]],
        generation_config,
    ) -> List[GeneratorResult]:
        # generate() never raises for generation-pipeline failures (they
        # become an error-carrying GeneratorResult instead) -- so every
        # request always yields exactly one result, in order, and a
        # failed item is never silently dropped from the returned list.
        return [
            self.generate(query_key, image_path, prompt, generation_config)
            for query_key, image_path, prompt in requests
        ]
