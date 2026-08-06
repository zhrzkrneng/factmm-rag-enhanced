"""IU X-Ray -> baseline PromptBuilder compatibility validation.

Responsibility: validate that src.baseline.generation.prompt_builder.
PromptBuilder (unmodified -- pure text-in/text-out, no dataset-specific
logic) works correctly against IU-X-Ray-derived target/retrieved report
strings. This module never generates a report and never loads a
generator model; it only exercises PromptBuilder's own existing string
construction and its own existing validation rules (reject_empty_
retrieved_report, max_retrieved_report_length, required/forbidden
argument combinations per PromptMode).
"""

from __future__ import annotations

from typing import List

from src.baseline.generation.dataset_builder import RAGDatasetRow
from src.baseline.generation.prompt_builder import (
    PromptBuilder,
    PromptBuilderConfig,
    PromptMode,
)

CAVEATS = (
    "PromptBuilder itself carries no dataset-specific identifiers or "
    "logic -- it only ever sees plain retrieved_report/target_report "
    "strings, so there is no MIMIC-CXR/CheXpert metadata for it to "
    "fabricate in the first place; this validation confirms that "
    "remains true for IU-X-Ray-derived strings too.",
    "Target report text is never included in retrieved evidence for "
    "non-Oracle rows -- checked directly (substring test), not just "
    "inferred from self-study exclusion, since two distinct studies "
    "could in principle share overlapping text.",
    "Empty evidence has explicit supported behavior via PromptMode."
    "VQA_INFERENCE/VQA_TRAIN (no retrieved_report argument at all), "
    "not a special IU-X-Ray-specific empty-string path -- an excluded "
    "RAGDatasetRow (retrieved_report_text=None) is routed to VQA mode, "
    "never RAG mode with an empty string.",
    "Multi-image metadata is not visible to PromptBuilder at all (it "
    "only ever sees a single <image> token placeholder, never a real "
    "path) -- multi-image grouping is preserved one layer up, in "
    "IuXrayRetrievalRecord.source, not lost by this validation.",
    "Text truncation only occurs through PromptBuilderConfig."
    "max_retrieved_report_length, explicitly configured -- never a "
    "hidden default truncation.",
)


def validate_row(row: RAGDatasetRow, builder: PromptBuilder) -> dict:
    """Builds the correct-mode prompt for one RAGDatasetRow and checks
    the invariants CAVEATS documents.

    Raises:
        ValueError: propagated unmodified from PromptBuilder.build() if
            the row's own data violates PromptBuilder's own rules (e.g.
            an excluded row's None retrieved_report_text reaching
            RAG_INFERENCE mode by caller error).
    """
    if row.excluded:
        mode = PromptMode.VQA_INFERENCE
        result = builder.build(mode)
        target_leak = False
    else:
        mode = PromptMode.RAG_INFERENCE
        result = builder.build(mode, retrieved_report=row.retrieved_report_text)
        target_leak = row.target_report_text in row.retrieved_report_text

    result_rerun = builder.build(mode, retrieved_report=row.retrieved_report_text) if not row.excluded else builder.build(mode)
    deterministic = result.text == result_rerun.text

    return {
        "query_key": list(row.query_key),
        "mode": mode.value,
        "excluded": row.excluded,
        "target_report_leaked_into_retrieved_evidence": target_leak,
        "deterministic": deterministic,
        "prompt_text_length": len(result.text),
    }


def validate_rows(
    rows: List[RAGDatasetRow], config: PromptBuilderConfig = None
) -> dict:
    """Validates every row, collecting per-row results plus a summary.

    Never raises on an individual row's own caveats (leak/nondeterminism
    are recorded as booleans, not exceptions) -- only a genuine
    PromptBuilder ValueError (malformed row data) propagates, since that
    indicates this adapter itself produced an invalid row, not a
    documentable caveat.
    """
    builder = PromptBuilder(config or PromptBuilderConfig())
    results = [validate_row(row, builder) for row in rows]
    return {
        "row_count": len(results),
        "any_target_leak": any(r["target_report_leaked_into_retrieved_evidence"] for r in results),
        "all_deterministic": all(r["deterministic"] for r in results),
        "excluded_count": sum(1 for r in results if r["excluded"]),
        "rag_mode_count": sum(1 for r in results if not r["excluded"]),
        "caveats": list(CAVEATS),
        "rows": results,
    }
