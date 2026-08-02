"""PromptBuilder core -- exact official LLaVA prompt string construction.

Responsibility: reproduce the four literal prompt string shapes fixed by
Milestone 2.5's design contract
(docs/milestone_2_5_generator_contract.md, section 5.3) exactly --
byte-for-byte, including whitespace -- matching the official FactMM-RAG
build_rag_dataset.py / build_nonrag_dataset.py f-strings, NOT the
paper's Figure 5 visual line layout (those differ; see the contract's
audit findings for why the figure's line breaks are a rendering choice,
not literal training-string content).

PromptBuilder.build() is pure and deterministic: no I/O, no randomness,
no hidden state beyond the immutable PromptBuilderConfig it was
constructed with. The four literal template shapes are internal
constants, not configurable text -- exposing them as mutable config
would risk silent drift from the official strings this module exists to
reproduce exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

_INSTRUCTION = "Generate a radiology report from this image:"


class PromptMode(Enum):
    RAG_TRAIN = "rag_train"
    RAG_INFERENCE = "rag_inference"
    VQA_TRAIN = "vqa_train"
    VQA_INFERENCE = "vqa_inference"


_MODES_REQUIRING_RETRIEVED_REPORT = (PromptMode.RAG_TRAIN, PromptMode.RAG_INFERENCE)
_MODES_REQUIRING_TARGET_REPORT = (PromptMode.RAG_TRAIN, PromptMode.VQA_TRAIN)


@dataclass(frozen=True)
class PromptBuilderConfig:
    config_version: str = "1.0"
    prompt_version: str = "official_v1"
    image_token: str = "<image>"
    retrieved_report_quote_char: str = '"'
    reject_empty_retrieved_report: bool = True
    max_retrieved_report_length: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.prompt_version:
            raise ValueError("prompt_version must be a non-empty string")
        if not self.image_token:
            raise ValueError("image_token must be a non-empty string")
        if len(self.retrieved_report_quote_char) != 1:
            raise ValueError(
                f"retrieved_report_quote_char must be exactly one character, "
                f"got {self.retrieved_report_quote_char!r}"
            )
        if not isinstance(self.reject_empty_retrieved_report, bool):
            raise ValueError("reject_empty_retrieved_report must be a bool")
        if self.max_retrieved_report_length is not None:
            invalid = (
                not isinstance(self.max_retrieved_report_length, int)
                or isinstance(self.max_retrieved_report_length, bool)
                or self.max_retrieved_report_length <= 0
            )
            if invalid:
                raise ValueError(
                    f"max_retrieved_report_length must be a positive integer or "
                    f"None, got {self.max_retrieved_report_length!r}"
                )


@dataclass(frozen=True)
class PromptResult:
    text: str
    conversations: Optional[List[dict]]
    prompt_version: str


class PromptBuilder:
    """Builds exact official prompt strings for a fixed PromptBuilderConfig.

    build() is pure: identical (mode, retrieved_report, target_report)
    arguments always produce a byte-identical PromptResult, with no
    side effects and no dependence on anything beyond the config this
    instance was constructed with.
    """

    def __init__(self, config: PromptBuilderConfig) -> None:
        self._config = config

    def build(
        self,
        mode: PromptMode,
        *,
        retrieved_report: Optional[str] = None,
        target_report: Optional[str] = None,
    ) -> PromptResult:
        """Builds a PromptResult for the given mode.

        Raises:
            ValueError: mode is not a PromptMode; retrieved_report is
                missing for a RAG_* mode or supplied for a VQA_* mode;
                target_report is missing for a *_TRAIN mode or supplied
                for a *_INFERENCE mode; retrieved_report is empty/
                whitespace-only while config.reject_empty_retrieved_report
                is True; or retrieved_report exceeds
                config.max_retrieved_report_length.
        """
        if not isinstance(mode, PromptMode):
            raise ValueError(f"mode must be a PromptMode, got {mode!r}")

        requires_retrieved = mode in _MODES_REQUIRING_RETRIEVED_REPORT
        requires_target = mode in _MODES_REQUIRING_TARGET_REPORT

        if requires_retrieved:
            if retrieved_report is None:
                raise ValueError(f"retrieved_report is required for mode {mode!r}")
            if self._config.reject_empty_retrieved_report and not retrieved_report.strip():
                raise ValueError(
                    f"retrieved_report must not be empty/whitespace-only for mode {mode!r}"
                )
            if (
                self._config.max_retrieved_report_length is not None
                and len(retrieved_report) > self._config.max_retrieved_report_length
            ):
                raise ValueError(
                    f"retrieved_report length ({len(retrieved_report)}) exceeds "
                    f"max_retrieved_report_length "
                    f"({self._config.max_retrieved_report_length}) -- never silently truncated"
                )
        elif retrieved_report is not None:
            raise ValueError(f"retrieved_report is forbidden for mode {mode!r}")

        if requires_target:
            if target_report is None:
                raise ValueError(f"target_report is required for mode {mode!r}")
        elif target_report is not None:
            raise ValueError(f"target_report is forbidden for mode {mode!r}")

        quote = self._config.retrieved_report_quote_char
        image_token = self._config.image_token

        if mode == PromptMode.RAG_TRAIN:
            text = (
                f"Here is a report of a related patient: {quote}{retrieved_report}{quote}"
                f"\n{_INSTRUCTION}{image_token}"
            )
            conversations = [
                {"from": "human", "value": text},
                {"from": "gpt", "value": target_report},
            ]
        elif mode == PromptMode.RAG_INFERENCE:
            text = (
                f"Here is a report of a related patient: {quote}{retrieved_report}{quote}"
                f"\n{_INSTRUCTION}"
            )
            conversations = None
        elif mode == PromptMode.VQA_TRAIN:
            text = f"{_INSTRUCTION}{image_token}"
            conversations = [
                {"from": "human", "value": text},
                {"from": "gpt", "value": target_report},
            ]
        else:  # PromptMode.VQA_INFERENCE
            text = f"\n{_INSTRUCTION}"
            conversations = None

        return PromptResult(
            text=text, conversations=conversations, prompt_version=self._config.prompt_version
        )
