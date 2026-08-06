"""LLaVAGenerator orchestrator, GeneratorConfig, GenerationConfig (Milestone 2.5).

Responsibility: a thin orchestrator that turns a query (image + optional
retrieved evidence) into a GeneratorResult by (1) building the exact
prompt text via PromptBuilder, (2) calling a caller-supplied
GeneratorAdapter, (3) sanity-checking the adapter's response. It never
constructs an adapter itself (mirrors src/baseline/retrieval/model.py's
dependency-injection discipline -- see
docs/milestone_2_5_generator_contract.md SS3) and never knows adapter
internals: MockGeneratorAdapter and the future HFGeneratorAdapter are
interchangeable here, both accessed only through the GeneratorAdapter
interface (src/baseline/generation/adapter.py). Contains no prompt
string literals (those live in prompt_builder.py) and no dataset
filtering logic (that lives in dataset_builder.py).

Built strictly from docs/milestone_2_5_generator_contract.md SS6 and
SS8.4, with two disclosed, necessary refinements:

1. GeneratorConfig.llava_checkpoint (SS6) is written as `= ...` in the
   contract, which is not valid between two already-defaulted dataclass
   fields (config_version, then llava_checkpoint, then base_lm_name --
   Python requires every field after the first defaulted one to also
   have a default). A concrete placeholder default is used instead:
   the public base LLaVA-1.5 checkpoint (`liuhaotian/llava-v1.5-7b`,
   REASONABLE_INFERENCE, matching the contract's own convention for the
   projector artifact identity) -- NOT this project's own eventual
   RAG-fine-tuned checkpoint, which is environment-specific and must be
   supplied explicitly once real training produces one.

2. `_generator_result_to_jsonl_row` (SS8.4) omits `image_path` from its
   parameter list, but SS10's own output schema example includes an
   `image_path` field, and GeneratorResult (adapter.py) does not carry
   an image path. An `image_path: str` keyword parameter is added --
   the same kind of necessary signature refinement already disclosed in
   adapter.py's module docstring for GeneratorResult.query_key.

GenerationConfig.deterministic is this project's explicit stand-in for
transformers' `do_sample` flag (SS6 does not define a literal
`do_sample` field): deterministic=True means do_sample=False (greedy or
beam search only -- temperature must be exactly 0, top_p/top_k must be
None, since sampling parameters are meaningless outside sampling mode);
deterministic=False means do_sample=True (temperature must be > 0).
The two are validated for mutual compatibility exactly as a real
do_sample flag would require, matching decoding semantics without
inventing an undocumented field.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from src.baseline.generation.adapter import GeneratorAdapter, GeneratorResult
from src.baseline.generation.dataset_builder import RAGDatasetRow
from src.baseline.generation.prompt_builder import PromptBuilder, PromptMode
from src.baseline.pair_mining.mining import QueryKey

_ADAPTER_KINDS: tuple = ("hf", "mock")
_INFERENCE_MODES: tuple = (PromptMode.RAG_INFERENCE, PromptMode.VQA_INFERENCE)


@dataclass(frozen=True)
class GeneratorConfig:
    """Adapter construction config -- what to load, never how to decode.

    Consumed by HFGeneratorAdapter.__init__ / MockGeneratorAdapter.__init__
    (not by LLaVAGenerator directly -- LLaVAGenerator is handed an
    already-constructed adapter, see module docstring and SS3).
    """

    config_version: str = "1.0"
    llava_checkpoint: str = "liuhaotian/llava-v1.5-7b"  # see module docstring, refinement 1
    base_lm_name: str = "lmsys/vicuna-7b-v1.5"
    vision_tower_name: str = "openai/clip-vit-large-patch14-336"
    adapter_kind: Literal["hf", "mock"] = "mock"
    lora_enabled: bool = False
    lora_checkpoint: Optional[str] = None
    conv_mode: str = "vicuna_v1"
    tokenizer_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.llava_checkpoint:
            raise ValueError("llava_checkpoint must be a non-empty string")
        if not self.base_lm_name:
            raise ValueError("base_lm_name must be a non-empty string")
        if not self.vision_tower_name:
            raise ValueError("vision_tower_name must be a non-empty string")
        if self.adapter_kind not in _ADAPTER_KINDS:
            raise ValueError(
                f"adapter_kind must be one of {_ADAPTER_KINDS}, got {self.adapter_kind!r}"
            )
        if not isinstance(self.lora_enabled, bool):
            raise ValueError("lora_enabled must be a bool")
        if self.lora_enabled and not self.lora_checkpoint:
            raise ValueError("lora_checkpoint is required when lora_enabled=True")
        if not self.lora_enabled and self.lora_checkpoint is not None:
            raise ValueError("lora_checkpoint is forbidden when lora_enabled=False")
        if not self.conv_mode:
            raise ValueError("conv_mode must be a non-empty string")
        if self.tokenizer_name is not None and not self.tokenizer_name:
            raise ValueError("tokenizer_name must be a non-empty string or None")


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding-time configuration -- how to decode, never what to load."""

    config_version: str = "1.0"
    temperature: float = 0.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 256
    num_beams: int = 1
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)):
            raise ValueError(f"temperature must be a number, got {self.temperature!r}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature!r}")
        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                raise ValueError(f"top_p must be a number or None, got {self.top_p!r}")
            if not (0 < self.top_p <= 1):
                raise ValueError(f"top_p must be in (0, 1] or None, got {self.top_p!r}")
        if self.top_k is not None:
            if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
                raise ValueError(f"top_k must be a positive integer or None, got {self.top_k!r}")
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens < 1
        ):
            raise ValueError(
                f"max_new_tokens must be a positive integer, got {self.max_new_tokens!r}"
            )
        if (
            not isinstance(self.num_beams, int)
            or isinstance(self.num_beams, bool)
            or self.num_beams < 1
        ):
            raise ValueError(f"num_beams must be a positive integer, got {self.num_beams!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not isinstance(self.deterministic, bool):
            raise ValueError("deterministic must be a bool")

        # do_sample compatibility (see module docstring): deterministic is
        # this project's explicit stand-in for transformers' do_sample=False.
        if self.deterministic:
            if self.temperature != 0:
                raise ValueError(
                    "deterministic=True (do_sample=False) requires temperature==0, "
                    f"got {self.temperature!r} -- the two must agree"
                )
            if self.top_p is not None or self.top_k is not None:
                raise ValueError(
                    "deterministic=True (do_sample=False) forbids top_p/top_k -- "
                    "sampling parameters are meaningless outside sampling mode"
                )
        else:
            if self.temperature <= 0:
                raise ValueError(
                    "deterministic=False (do_sample=True) requires temperature > 0, "
                    f"got {self.temperature!r} -- greedy decoding contradicts sampling"
                )


@dataclass(frozen=True)
class GenerationRequest:
    """One generate_many() request: a query, its image, and (optionally)
    its single retrieved-evidence report text."""

    query_key: QueryKey
    image_path: str
    mode: PromptMode
    retrieved_report: Optional[str] = None


class LLaVAGenerator:
    """Thin orchestrator: PromptBuilder -> GeneratorAdapter -> GeneratorResult.

    Never constructs an adapter itself -- adapter is a fully-built
    GeneratorAdapter instance (MockGeneratorAdapter today, a future
    HFGeneratorAdapter later), handed in at construction time.
    generation_config is decoding-time only (see GenerationConfig);
    GeneratorConfig is never passed here -- it is the adapter's own
    construction config, consumed before this class ever sees the
    adapter.
    """

    def __init__(
        self,
        adapter: GeneratorAdapter,
        prompt_builder: PromptBuilder,
        generation_config: GenerationConfig,
    ) -> None:
        if not isinstance(adapter, GeneratorAdapter):
            raise ValueError(
                f"adapter must be a GeneratorAdapter instance, got {type(adapter)!r}"
            )
        if not isinstance(prompt_builder, PromptBuilder):
            raise ValueError(
                f"prompt_builder must be a PromptBuilder instance, got {type(prompt_builder)!r}"
            )
        if not isinstance(generation_config, GenerationConfig):
            raise ValueError(
                f"generation_config must be a GenerationConfig instance, got "
                f"{type(generation_config)!r}"
            )
        self._adapter = adapter
        self._prompt_builder = prompt_builder
        self._generation_config = generation_config

    def _build_prompt_text(self, mode: PromptMode, retrieved_report: Optional[str]) -> str:
        if not isinstance(mode, PromptMode) or mode not in _INFERENCE_MODES:
            raise ValueError(
                f"LLaVAGenerator only generates in inference modes "
                f"{_INFERENCE_MODES}, got {mode!r} -- *_TRAIN modes build SFT "
                f"training conversations, not real-time generation"
            )
        requires_retrieved = mode == PromptMode.RAG_INFERENCE
        if requires_retrieved and retrieved_report is None:
            raise ValueError(
                f"mode={mode!r} requires retrieved_report (missing retrieved "
                f"evidence) -- pass a RAGDatasetRow with excluded=False and a "
                f"retrieved_key, or use PromptMode.VQA_INFERENCE for an "
                f"excluded/no-evidence row"
            )
        if not requires_retrieved and retrieved_report is not None:
            raise ValueError(f"retrieved_report is forbidden for mode={mode!r}")

        prompt_result = self._prompt_builder.build(mode, retrieved_report=retrieved_report)
        return prompt_result.text

    def _validate_result(self, expected_query_key: QueryKey, result: GeneratorResult) -> None:
        if not isinstance(result, GeneratorResult):
            raise ValueError(
                f"adapter returned a non-GeneratorResult object for "
                f"{expected_query_key}: {result!r}"
            )
        if result.query_key != expected_query_key:
            raise ValueError(
                f"adapter returned a GeneratorResult for query_key "
                f"{result.query_key} but {expected_query_key} was requested -- "
                f"malformed adapter result"
            )
        if result.generation_metadata is None:
            raise ValueError(
                f"GeneratorResult for {expected_query_key} is missing generation_metadata"
            )

    def generate_one(
        self,
        query_key: QueryKey,
        *,
        image_path: str,
        mode: PromptMode,
        retrieved_report: Optional[str] = None,
    ) -> GeneratorResult:
        """Builds the prompt via PromptBuilder, calls adapter.generate(),
        and sanity-checks the returned GeneratorResult before returning it."""
        prompt_text = self._build_prompt_text(mode, retrieved_report)
        result = self._adapter.generate(query_key, image_path, prompt_text, self._generation_config)
        self._validate_result(query_key, result)
        return result

    def generate_many(self, requests: Sequence[GenerationRequest]) -> List[GeneratorResult]:
        """Builds prompts for every request, then makes one
        adapter.generate_batch() call -- preserving request order in the
        returned list.

        Raises ValueError immediately, before calling the adapter, if
        requests contains a duplicate query_key.
        """
        if not requests:
            return []

        query_keys = [request.query_key for request in requests]
        counts = Counter(query_keys)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                f"generate_many received duplicate query_key(s): {duplicates}"
            )

        batch_requests = [
            (request.query_key, request.image_path, self._build_prompt_text(request.mode, request.retrieved_report))
            for request in requests
        ]
        results = self._adapter.generate_batch(batch_requests, self._generation_config)

        if len(results) != len(requests):
            raise ValueError(
                f"adapter.generate_batch returned {len(results)} results for "
                f"{len(requests)} requests -- result count must match request count"
            )
        for request, result in zip(requests, results):
            self._validate_result(request.query_key, result)
        return results

    def generate_from_rag_row(self, row: RAGDatasetRow) -> GeneratorResult:
        """Orchestrates a single RAGDatasetBuilder output row: RAG_INFERENCE
        with the row's retrieved evidence when not excluded, VQA_INFERENCE
        (no evidence) when the row was excluded (SS4's strict-isolation
        exclusion, or any other reason build_row() found nothing to retrieve)."""
        if row.excluded:
            return self.generate_one(
                row.query_key, image_path=row.image_path, mode=PromptMode.VQA_INFERENCE
            )
        return self.generate_one(
            row.query_key,
            image_path=row.image_path,
            mode=PromptMode.RAG_INFERENCE,
            retrieved_report=row.retrieved_report_text,
        )

    def generate_many_from_rag_rows(self, rows: Sequence[RAGDatasetRow]) -> List[GeneratorResult]:
        """generate_many(), built from a sequence of RAGDatasetBuilder rows."""
        requests = [
            GenerationRequest(
                query_key=row.query_key,
                image_path=row.image_path,
                mode=PromptMode.VQA_INFERENCE if row.excluded else PromptMode.RAG_INFERENCE,
                retrieved_report=None if row.excluded else row.retrieved_report_text,
            )
            for row in rows
        ]
        return self.generate_many(requests)


def _generator_result_to_jsonl_row(
    result: GeneratorResult,
    *,
    image_path: str,
    retrieved_evidence_key: Optional[QueryKey],
    prompt_version: str,
    paper_version: str,
    implementation_version: str,
    generation_config: GenerationConfig,
) -> dict:
    """Pure, deterministic mapping: GeneratorResult + surrounding
    per-sample context -> the SS10 output JSONL row shape.

    Nothing else in the pipeline constructs an output row ad hoc --
    every caller (a future resumable write loop, a synthetic smoke
    test) goes through this one function. Given the same arguments, it
    always returns a byte-identical (via json.dumps) dict; no I/O, no
    randomness, no hidden state.
    """
    return {
        "query_key": list(result.query_key),
        "image_path": image_path,
        "retrieved_evidence_key": (
            list(retrieved_evidence_key) if retrieved_evidence_key is not None else None
        ),
        "prompt_version": prompt_version,
        "paper_version": paper_version,
        "implementation_version": implementation_version,
        "generated_report_text": result.generated_report_text,
        "generation_config": {
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "max_new_tokens": generation_config.max_new_tokens,
            "num_beams": generation_config.num_beams,
            "seed": generation_config.seed,
            "deterministic": generation_config.deterministic,
        },
        "generation_metadata": dataclasses.asdict(result.generation_metadata),
        "error": result.error,
    }
