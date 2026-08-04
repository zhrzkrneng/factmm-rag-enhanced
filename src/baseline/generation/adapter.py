"""Generator adapter interface and shared generation-result types (Milestone 2.5).

Responsibility: define the abstract seam between `LLaVAGenerator` (the
future thin orchestrator, not implemented in this cell) and any
concrete generation backend -- real (`HFGeneratorAdapter`, deferred to
a later cell once `GeneratorConfig` exists) or fake
(`MockGeneratorAdapter`, `mock_adapter.py`). Mirrors
`MultiModalRetriever`'s dependency-injection discipline (see
`src/baseline/retrieval/model.py`), but as a fully-constructed object
handed to the orchestrator, not raw factories consumed internally.

Built from docs/milestone_2_5_generator_contract.md SS8, with one
disclosed, necessary refinement: SS8.1's `generate`/`generate_batch`
signatures as written (`generate(image_path, prompt, generation_config)`,
`generate_batch(requests: Sequence[Tuple[str, str]], ...)`) have no way
to supply the `query_key` that `GeneratorResult.query_key` (SS8.4)
requires. This module adds an explicit `query_key: QueryKey` parameter
to `generate()` and widens `generate_batch`'s per-request tuple to
`Tuple[QueryKey, str, str]` (query_key, image_path, prompt) --
otherwise identical to the contract. Everything else (three
abstractmethods, `PreparedGeneratorInputs`, `GenerationMetadata`,
`GeneratorResult`, `HFGeneratorAdapter` deferred) matches SS8 exactly.

`HFGeneratorAdapter` is intentionally NOT defined in this module yet:
its constructor signature (SS8.5) takes `config: GeneratorConfig`, and
`GeneratorConfig` itself belongs to `generator.py` (SS7), not yet
implemented. Building a real/skeleton `HFGeneratorAdapter` against a
type that doesn't exist would either fabricate `GeneratorConfig`
prematurely (pre-empting a design decision that belongs to the
`generator.py` cell) or leave it duck-typed and half-formed -- both
worse than deferring the whole class to the cell where
`GeneratorConfig` is actually introduced. Only `GeneratorAdapter`'s
abstract interface is defined here, which is sufficient for
`MockGeneratorAdapter` (mock_adapter.py) to implement fully.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import torch

from src.baseline.pair_mining.mining import QueryKey

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
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


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
