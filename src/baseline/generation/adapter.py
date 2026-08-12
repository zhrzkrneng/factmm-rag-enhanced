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

5. FIXED (real bug, found via a real Colab dry run against the actual
   liuhaotian/llava-v1.5-7b checkpoint): `prepare_inputs()` tokenized
   the literal "<image>" substring with a plain base-LM tokenizer that
   has no such special token, so it always decomposed into ordinary
   sub-word tokens -- never the model's real `image_token_index` -- and
   `generate()` failed every time with "Image features and image
   tokens do not match: tokens: 0, features N". `prepare_inputs()` now
   resolves the model's real `image_token_index` and the exact number
   of image feature positions its vision tower produces (from
   `config.vision_config.image_size`/`patch_size`), and splices that
   many copies of the real token id in place of the placeholder
   (`_resolve_image_token_expansion`/`_expand_image_placeholder`). Only
   takes effect when the loaded model's config actually exposes that
   metadata -- every real Llava checkpoint does; the pre-refinement,
   single-placeholder-token behavior is unchanged for anything that
   doesn't (e.g. this module's own injected test doubles).

6. FIXED (real bug, found via a real Colab dry run: nearly all
   language-model/vision-tower/multimodal-projector weights reported
   "newly initialized", then `generate()` failed with a raw
   `IndexError: index out of range in self`): `liuhaotian/llava-v1.5-7b`
   is the ORIGINAL LLaVA repository's checkpoint, packaged for that
   repo's own model code -- its parameter names and config.json shape
   do not match transformers' native `LlavaForConditionalGeneration`,
   so loading it there leaves almost every core weight randomly
   initialized, and the (also essentially-default) `image_token_index`
   ends up landing exactly one past a randomly-shaped embedding table's
   bounds. `_default_model_factory`/`_default_tokenizer_factory`/
   `_default_image_processor_factory` now resolve
   `llava-hf/llava-1.5-7b-hf` (the transformers-native conversion) for
   ACTUAL weight/tokenizer/image-processor loading whenever
   `config.llava_checkpoint` is left at that original-repo default --
   disclosed explicitly (printed), never silent, and never overriding
   an explicitly-chosen non-default checkpoint. `config.llava_checkpoint`
   itself is untouched everywhere else in this project (provenance
   records, Milestone 2.8 resource discovery, etc. still name the
   official paper/repo checkpoint). Two new hard gates were added: (a)
   `_check_checkpoint_architecture_match` inspects
   `from_pretrained(..., output_loading_info=True)`'s `missing_keys` and
   raises `CheckpointArchitectureMismatchError` at construction time,
   before any generation is attempted, if a substantial fraction of
   core weights were not actually loaded from the checkpoint; (b)
   `generate()` now validates every input token id is within the
   model's real embedding-table bounds and raises
   `VocabRangeMismatchError` before ever calling the real
   `model.generate()` if not. A new `diagnose()` method reports the
   exact pre-generation values (tokenizer length, model embedding rows,
   input_ids min/max, image token id/count, pixel_values shape, model
   dtype/device) without performing generation, for explicit,
   before-the-fact auditing.
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


class CheckpointArchitectureMismatchError(RuntimeError):
    """Raised at HFGeneratorAdapter construction time (see refinement 6)
    when a loaded checkpoint's weights don't actually match the
    LlavaForConditionalGeneration architecture it was loaded into -- a
    substantial fraction of core weights (language model, vision
    tower, or multimodal projector) were reported "newly initialized"
    (i.e. missing from the checkpoint) rather than actually loaded.
    Never subclasses ValueError -- must be classified as its own
    "checkpoint_architecture_mismatch" category, not folded into the
    generic "invalid_input" bucket.
    """


class VocabRangeMismatchError(RuntimeError):
    """Raised by generate() (see refinement 6) when prepared input_ids
    contain a token id outside the model's real embedding-table bounds
    -- e.g. a tokenizer pulled from a different checkpoint family than
    the model. Raised BEFORE the real model.generate() is ever called.
    Classified as its own "vocab_range_mismatch" category.
    """


class ImageTokenMismatchError(RuntimeError):
    """Raised by _expand_image_placeholder (see refinement 5) when the
    templated prompt text doesn't contain exactly one "<image>"
    placeholder, or the spliced input_ids don't end up with the exact
    expected image-token count. Classified as its own
    "image_token_mismatch" category, distinct from a general
    invalid_input caller mistake.
    """


@dataclass(frozen=True)
class PreGenerationDiagnostics:
    """Exact, explicit pre-generation values (refinement 6, requirement
    10): computed by HFGeneratorAdapter.diagnose() without ever calling
    the real model.generate() -- for auditing what generate() is about
    to do, before it does it.
    """

    tokenizer_length: Optional[int]
    model_embedding_rows: Optional[int]
    input_ids_min: int
    input_ids_max: int
    image_token_id: Optional[int]
    image_token_count: int
    pixel_values_shape: Tuple[int, ...]
    model_dtype: Optional[str]
    model_device: Optional[str]
    within_vocab_range: bool


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


def _resolve_image_token_expansion(model) -> Optional[Tuple[int, int]]:
    """Resolves (image_token_id, num_image_tokens) from a loaded model's
    config, or None if that metadata isn't present (see refinement 5).

    Every real LlavaForConditionalGeneration checkpoint's config exposes
    both `image_token_index` (the vocabulary id the model scatters vision
    features into) and `vision_config.image_size`/`patch_size` (which
    together determine exactly how many image feature positions the
    vision tower produces per image, e.g. (336 // 14) ** 2 == 576 for
    CLIP-ViT-L-336). Returning None here (e.g. an injected test double
    whose fake config has neither field) signals prepare_inputs() to
    keep the pre-refinement single-placeholder-token behavior -- this
    never happens for a real checkpoint, only for metadata that
    genuinely doesn't describe a Llava vision/language pairing.
    """
    config = getattr(model, "config", None)
    image_token_id = getattr(config, "image_token_index", None) if config is not None else None
    vision_config = getattr(config, "vision_config", None) if config is not None else None
    if image_token_id is None or vision_config is None:
        return None

    image_size = getattr(vision_config, "image_size", None)
    patch_size = getattr(vision_config, "patch_size", None)
    if not image_size or not patch_size:
        return None

    num_patches = (image_size // patch_size) ** 2
    strategy = getattr(config, "vision_feature_select_strategy", "default")
    num_image_tokens = num_patches + 1 if strategy == "full" else num_patches
    return image_token_id, num_image_tokens


def _expand_image_placeholder(tokenizer, templated_text: str, image_token_id: int, num_image_tokens: int):
    """Replaces the single, human-readable "<image>" substring in
    `templated_text` with `num_image_tokens` copies of the model's real
    `image_token_id`, tokenizing the text on either side separately.

    This is necessary (see refinement 5) because the plain base-LM
    tokenizer used here (loaded from `base_lm_name`/`tokenizer_name`,
    never the Llava checkpoint's own paired processor) has no "<image>"
    special token at all -- tokenizing the literal substring produces
    ordinary sub-word tokens that never equal `image_token_id`, so the
    model's forward pass finds zero image placeholder positions to
    scatter its real, non-empty vision features into.
    """
    import torch

    occurrences = templated_text.count(_IMAGE_TOKEN)
    if occurrences != 1:
        raise ImageTokenMismatchError(
            f"templated_text must contain exactly one {_IMAGE_TOKEN!r} "
            f"placeholder to expand, found {occurrences}"
        )
    pre_text, _, post_text = templated_text.partition(_IMAGE_TOKEN)

    pre_ids = list(tokenizer(pre_text, add_special_tokens=True)["input_ids"])
    post_ids = list(tokenizer(post_text, add_special_tokens=False)["input_ids"])

    spliced = pre_ids + [image_token_id] * num_image_tokens + post_ids
    input_ids = torch.tensor([spliced], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    actual = int((input_ids == image_token_id).sum())
    if actual != num_image_tokens:
        raise ImageTokenMismatchError(
            f"Internal error expanding the image placeholder: expected "
            f"{num_image_tokens} occurrence(s) of image_token_id="
            f"{image_token_id} in the spliced input_ids, got {actual}."
        )
    return input_ids, attention_mask


_ORIGINAL_LLAVA_CHECKPOINT = "liuhaotian/llava-v1.5-7b"
_HF_NATIVE_LLAVA_CHECKPOINT = "llava-hf/llava-1.5-7b-hf"

_CORE_WEIGHT_PREFIXES: Tuple[str, ...] = ("language_model.", "vision_tower.", "multi_modal_projector.")


def _resolve_hf_native_checkpoint(config) -> str:
    """Resolves the checkpoint identifier to actually load for the
    model/tokenizer/image processor (see refinement 6).

    Substitutes `llava-hf/llava-1.5-7b-hf` (the transformers-native
    conversion) for `config.llava_checkpoint` ONLY when it is still the
    original-repo default (`liuhaotian/llava-v1.5-7b`) -- the original
    repo's checkpoint is packaged for that repo's own model code, not
    transformers' LlavaForConditionalGeneration, and loading it there
    leaves nearly every core weight randomly initialized (see
    _check_checkpoint_architecture_match). config.llava_checkpoint
    itself is never mutated -- this substitution is local to how
    HFGeneratorAdapter's own factories actually load weights; an
    explicitly-chosen non-default checkpoint is always respected as-is.
    """
    if config.llava_checkpoint == _ORIGINAL_LLAVA_CHECKPOINT:
        print(
            f"[HFGeneratorAdapter] NOTE: disclosed compatibility substitution -- "
            f"{_ORIGINAL_LLAVA_CHECKPOINT!r} is the original LLaVA repository's "
            f"checkpoint, incompatible with transformers' native "
            f"LlavaForConditionalGeneration (its weight names/config don't "
            f"match, leaving core weights randomly initialized if loaded "
            f"there). Using the transformers-native conversion "
            f"{_HF_NATIVE_LLAVA_CHECKPOINT!r} instead for actual model/"
            f"tokenizer/image-processor loading. config.llava_checkpoint "
            f"itself is unchanged everywhere else in this project."
        )
        return _HF_NATIVE_LLAVA_CHECKPOINT
    return config.llava_checkpoint


def _check_checkpoint_architecture_match(model, loading_info, *, max_missing_fraction: float = 0.05) -> None:
    """Raises CheckpointArchitectureMismatchError (refinement 6,
    requirement 6) if a substantial fraction of core weights (language
    model, vision tower, or multimodal projector) were reported missing
    from the checkpoint -- i.e. randomly/newly initialized rather than
    actually loaded -- per `from_pretrained(..., output_loading_info=True)`'s
    own `missing_keys` list. This is the exact, structured signal transformers
    itself provides for "checkpoint doesn't actually match this architecture";
    never inferred from parsing the human-readable warning text.
    """
    missing_keys = list((loading_info or {}).get("missing_keys") or [])
    if not missing_keys:
        return

    total_params = sum(1 for _ in model.named_parameters())
    if total_params == 0:
        return

    missing_fraction = len(missing_keys) / total_params
    missing_core = [key for key in missing_keys if any(prefix in key for prefix in _CORE_WEIGHT_PREFIXES)]

    if missing_fraction > max_missing_fraction and missing_core:
        raise CheckpointArchitectureMismatchError(
            f"{len(missing_keys)}/{total_params} parameters "
            f"({missing_fraction:.1%}) were NOT found in the checkpoint and "
            f"were newly/randomly initialized instead, including "
            f"{len(missing_core)} core weight(s) (language model / vision "
            f"tower / multimodal projector), e.g. {missing_core[:3]}. This "
            f"means the checkpoint's parameter names do not actually match "
            f"the LlavaForConditionalGeneration architecture it was loaded "
            f"into -- refusing to attempt generation against substantially "
            f"random weights."
        )


def _resolve_embedding_row_count(model) -> Optional[int]:
    """Best-effort real embedding-table row count -- the true bound
    input_ids must respect, more reliable than a possibly-stale
    `config.vocab_size` field. Returns None (never raises) if `model`
    doesn't expose `get_input_embeddings()` (e.g. an injected test
    double) -- refinement 6's vocab-range guard is then simply skipped,
    matching the same fallback discipline as refinement 5.
    """
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if get_input_embeddings is None:
        return None
    try:
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        shape = getattr(weight, "shape", None) if weight is not None else None
        if not shape:
            return None
        return int(shape[0])
    except Exception:  # noqa: BLE001 -- diagnostics must never crash generate() itself
        return None


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
    """Classifies exc into one of 11 categories -- never a single generic
    bucket. See module docstring, refinements 4 and 6.
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

    # Refinement 6's own three exception types are checked first and
    # explicitly, by isinstance -- ahead of every other check, including
    # the generic ValueError bucket below, so they are never folded
    # into invalid_input/generation_failure. Distinguishes checkpoint
    # incompatibility (A) / vocab mismatch (B) / image-token expansion
    # mismatch (C) from an ordinary caller mistake or a genuine
    # generation-time failure (D).
    if isinstance(exc, CheckpointArchitectureMismatchError):
        return "checkpoint_architecture_mismatch", f"{type(exc).__name__}: {exc}"

    if isinstance(exc, VocabRangeMismatchError):
        return "vocab_range_mismatch", f"{type(exc).__name__}: {exc}"

    if isinstance(exc, ImageTokenMismatchError):
        return "image_token_mismatch", f"{type(exc).__name__}: {exc}"

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

    checkpoint = _resolve_hf_native_checkpoint(config)
    model, loading_info = LlavaForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=_select_default_dtype(), output_loading_info=True
    )
    _check_checkpoint_architecture_match(model, loading_info)
    return model


def _default_tokenizer_factory(config):
    from transformers import AutoTokenizer

    checkpoint = _resolve_hf_native_checkpoint(config)
    if checkpoint != config.llava_checkpoint:
        # Substituted checkpoint -- use ITS bundled tokenizer, not
        # config.tokenizer_name/base_lm_name (which target the
        # ORIGINAL checkpoint's family and would reintroduce the exact
        # vocab/embedding-table mismatch this refinement fixes;
        # requirement 8: tokenizer/processor/model must all come from
        # the same compatible checkpoint family).
        return AutoTokenizer.from_pretrained(checkpoint)
    tokenizer_name = config.tokenizer_name or config.base_lm_name
    return AutoTokenizer.from_pretrained(tokenizer_name)


def _default_image_processor_factory(config):
    from transformers import CLIPImageProcessor

    checkpoint = _resolve_hf_native_checkpoint(config)
    if checkpoint != config.llava_checkpoint:
        return CLIPImageProcessor.from_pretrained(checkpoint)

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

        # Refinement 5 (real bug, found via a real Colab dry run against
        # the actual liuhaotian/llava-v1.5-7b checkpoint): the tokenizer
        # above has no "<image>" special token, so the naive input_ids
        # built from it contain zero occurrences of the model's real
        # image_token_index while the vision tower still produces a full
        # set of real image features -- LlavaForConditionalGeneration's
        # forward pass then raises "Image features and image tokens do
        # not match: tokens: 0, features N". When the loaded model's
        # config exposes enough metadata to compute the real expected
        # image token count (every real Llava checkpoint does), replace
        # the single placeholder with that many copies of the real
        # image_token_index instead. Falls back to the input_ids already
        # built above when that metadata isn't present (e.g. an injected
        # test double) -- never a silent behavior change for a real
        # checkpoint, only for a model that genuinely isn't describable
        # this way.
        expansion = _resolve_image_token_expansion(self._model)
        if expansion is not None:
            image_token_id, num_image_tokens = expansion
            input_ids, attention_mask = _expand_image_placeholder(
                self._tokenizer, templated_text, image_token_id, num_image_tokens
            )

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

    def diagnose(self, image_path: str, prompt_text: str) -> PreGenerationDiagnostics:
        """Computes the exact pre-generation diagnostic values
        (refinement 6, requirement 10) -- tokenizer length, model
        embedding-table row count, input_ids min/max, image token
        id/count, pixel_values shape, model dtype/device -- WITHOUT
        ever calling the real model.generate(). Intended for explicit,
        before-the-fact auditing by a caller (e.g. a dry-run script)
        immediately before it calls generate() itself; generate()
        performs the same vocab-range check on its own and raises
        VocabRangeMismatchError if it fails, so calling diagnose()
        first is optional, never required for correctness.
        """
        prepared = self.prepare_inputs(image_path, prompt_text)

        expansion = _resolve_image_token_expansion(self._model)
        image_token_id = expansion[0] if expansion is not None else None
        image_token_count = (
            int((prepared.input_ids == image_token_id).sum()) if image_token_id is not None else 0
        )

        tokenizer_length = None
        if hasattr(self._tokenizer, "__len__"):
            try:
                tokenizer_length = len(self._tokenizer)
            except Exception:  # noqa: BLE001 -- diagnostics must never crash
                tokenizer_length = None

        embedding_rows = _resolve_embedding_row_count(self._model)
        input_ids_min = int(prepared.input_ids.min())
        input_ids_max = int(prepared.input_ids.max())
        within_vocab_range = embedding_rows is None or (0 <= input_ids_min and input_ids_max < embedding_rows)

        return PreGenerationDiagnostics(
            tokenizer_length=tokenizer_length,
            model_embedding_rows=embedding_rows,
            input_ids_min=input_ids_min,
            input_ids_max=input_ids_max,
            image_token_id=image_token_id,
            image_token_count=image_token_count,
            pixel_values_shape=tuple(prepared.pixel_values.shape),
            model_dtype=str(getattr(self._model, "dtype", None)),
            model_device=str(getattr(self._model, "device", None)),
            within_vocab_range=within_vocab_range,
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

            # Refinement 6, requirement 11: STOP before calling the real
            # model.generate() if any input token id falls outside the
            # model's actual embedding-table bounds -- the exact
            # failure mode that previously surfaced as a raw, opaque
            # "IndexError: index out of range in self" deep inside the
            # real forward pass instead of a clear, classified error.
            embedding_rows = _resolve_embedding_row_count(self._model)
            if embedding_rows is not None:
                min_id = int(prepared.input_ids.min())
                max_id = int(prepared.input_ids.max())
                if max_id >= embedding_rows or min_id < 0:
                    raise VocabRangeMismatchError(
                        f"input_ids contain token id(s) outside the model's "
                        f"embedding table range [0, {embedding_rows}): "
                        f"min={min_id}, max={max_id}. This means the "
                        f"tokenizer/processor used to build input_ids is not "
                        f"from the same checkpoint family as the loaded "
                        f"model."
                    )

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
