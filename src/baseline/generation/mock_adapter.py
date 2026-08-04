"""Deterministic, template-filling fake GeneratorAdapter (Milestone 2.5).

No `transformers` import, no `.from_pretrained(...)` call, no model
download, no network access -- ever. Used by every unit test and every
synthetic smoke test until Cell 29's real HFGeneratorAdapter dry run
(docs/milestone_2_5_generator_contract.md SS8.6, SS12).

`torch` is used to construct the `PreparedGeneratorInputs` tensors
`prepare_inputs()` must return (its fields are contractually typed as
`torch.Tensor`, SS8.2) -- torch is a plain tensor container library
already a hard project dependency since Milestone 2.4, not an "ML
model" in the sense this module avoids. What this module genuinely
never does is load or run a real neural network: no CLIP/T5/LLaVA
construction, no forward pass, no checkpoint of any kind.

Determinism is seeded exclusively via hashlib.sha256 over the exact
input strings (image_path/prompt_text/seed), never Python's built-in
hash() -- that function is per-process-randomized unless
PYTHONHASHSEED is fixed (see the FakeImageProcessor bug fixed earlier
in this project, tests/unit/test_retrieval_dataset.py), which would
silently break the "same input + same seed = byte-identical output"
requirement across separate processes/runs.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import List, Sequence, Tuple

import torch

from src.baseline.generation.adapter import (
    GenerationMetadata,
    GeneratorAdapter,
    GeneratorResult,
    PreparedGeneratorInputs,
)
from src.baseline.pair_mining.mining import QueryKey
from src.retrieval.index import _resolve_git_commit

_MOCK_TENSOR_DIM = 4  # arbitrary small fixed size -- structural placeholder only,
                      # never claims to be a real image/token count


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _deterministic_int(*parts: str) -> int:
    """sha256-based deterministic int, never Python's randomized hash()."""
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _is_valid_query_key(query_key) -> bool:
    return (
        isinstance(query_key, tuple)
        and len(query_key) == 3
        and all(isinstance(part, str) and part for part in query_key)
    )


class MockGeneratorAdapter(GeneratorAdapter):
    """Deterministic fake adapter: no ML dependency, no randomness beyond
    the caller-supplied GenerationConfig.seed.

    prepare_inputs() is seeded purely from (image_path, prompt_text) --
    it has no seed parameter (mirrors real preprocessing, which is not
    a function of the decoding seed). generate()/generate_batch() are
    additionally seeded from generation_config.seed, so the same
    (query_key, image_path, prompt, seed) always produces byte-identical
    output, and a different seed changes the generated text.
    """

    def __init__(self, *, conv_mode: str = "vicuna_v1") -> None:
        if not conv_mode:
            raise ValueError("conv_mode must be a non-empty string")
        self._conv_mode = conv_mode

    def prepare_inputs(self, image_path: str, prompt_text: str) -> PreparedGeneratorInputs:
        if not image_path:
            raise ValueError("image_path must be a non-empty string")
        if not prompt_text:
            raise ValueError("prompt_text must be a non-empty string")

        tensor_seed = _deterministic_int("prepare_inputs", image_path, prompt_text) % (2**31)
        generator = torch.Generator().manual_seed(tensor_seed)
        pixel_values = torch.rand(
            (3, _MOCK_TENSOR_DIM, _MOCK_TENSOR_DIM), generator=generator
        )

        tokens = prompt_text.split()
        token_ids = [_deterministic_int("token", tok) % 32000 for tok in tokens]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return PreparedGeneratorInputs(
            pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask
        )

    def _build_metadata(self) -> GenerationMetadata:
        return GenerationMetadata(
            adapter_kind="mock",
            model_checkpoint="mock-checkpoint",
            resolved_revision=None,
            base_lm_name="mock-base-lm",
            vision_tower_name="mock-vision-tower",
            lora_checkpoint=None,
            conv_mode=self._conv_mode,
            git_commit=_resolve_git_commit(),
            torch_version=None,
            transformers_version=None,
            hardware="mock",
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

        seed = getattr(generation_config, "seed", None)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError(f"generation_config.seed must be an int, got {seed!r}")

        prepared = self.prepare_inputs(image_path, prompt)
        prepared_signature = str(int(prepared.input_ids.sum().item()))

        text_seed = _deterministic_int("generate", prepared_signature, str(seed))
        generated_text = f"[MOCK REPORT seed={seed} sig={text_seed % 10**8:08d}]"

        return GeneratorResult(
            query_key=query_key,
            generated_report_text=generated_text,
            generation_metadata=self._build_metadata(),
            error=None,
        )

    def generate_batch(
        self,
        requests: Sequence[Tuple[QueryKey, str, str]],
        generation_config,
    ) -> List[GeneratorResult]:
        return [
            self.generate(query_key, image_path, prompt, generation_config)
            for query_key, image_path, prompt in requests
        ]
