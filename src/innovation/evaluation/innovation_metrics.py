"""Innovation-specific metrics contract (Cell 47, Milestone 3.0, Section 8).

Responsibility: name, not reimplement, the retrieval/generation/clinical
metrics every experiment arm (E0-E7) must report on, reusing this
project's own real, already-implemented metric registries
(src.evaluation.retrieval_metrics.build_retrieval_metric_registry,
src.evaluation.generation_metrics.build_metric_registry) -- plus the
innovation-specific diagnostics (Section 8: adaptive-k, reranking,
fusion, and efficiency diagnostics) not needed by the baseline at all.

Cell 47 does not compute or report any of these values (Section 8: "Do
not report scientific performance in Cell 47. This cell defines the
contract only.").
"""

from __future__ import annotations

from typing import Dict, Tuple

# Exactly the real registry keys from src.evaluation.retrieval_metrics
# and src.evaluation.generation_metrics (see
# _REGISTERED_RETRIEVAL_METRIC_NAMES / _REGISTERED_METRIC_NAMES) --
# reused verbatim so this contract can never silently drift from the
# real, already-implemented metric registries.
RETRIEVAL_METRICS: Tuple[str, ...] = ("mrr", "recall", "ndcg")
GENERATION_METRICS: Tuple[str, ...] = ("rouge_l", "bleu4", "bert_score")
CLINICAL_METRICS: Tuple[str, ...] = ("f1radgraph", "f1chexbert")

ADAPTIVE_K_DIAGNOSTICS: Tuple[str, ...] = (
    "mean_selected_k",
    "k_distribution",
    "evidence_reduction_vs_fixed_k",
)
RERANKING_DIAGNOSTICS: Tuple[str, ...] = (
    "rank_change_rate",
    "mean_rank_displacement",
    "top1_replacement_rate",
)
FUSION_DIAGNOSTICS: Tuple[str, ...] = (
    "fusion_activation_rate",
    "mean_evidence_count",
    "evidence_context_reduction",
)
EFFICIENCY_DIAGNOSTICS: Tuple[str, ...] = (
    "latency_per_query",
    "evidence_count",
    "approximate_context_length",
)

INNOVATION_DIAGNOSTICS: Tuple[str, ...] = (
    ADAPTIVE_K_DIAGNOSTICS + RERANKING_DIAGNOSTICS + FUSION_DIAGNOSTICS + EFFICIENCY_DIAGNOSTICS
)


def metric_contract() -> Dict[str, Tuple[str, ...]]:
    """JSON-ready metric contract, separated into the three categories
    Section 8 requires (retrieval / generation / clinical)."""
    return {
        "retrieval": RETRIEVAL_METRICS,
        "generation": GENERATION_METRICS,
        "clinical": CLINICAL_METRICS,
    }


def diagnostic_registry() -> Dict[str, Tuple[str, ...]]:
    """JSON-ready innovation-specific diagnostic registry, grouped by
    which innovation (or cross-cutting efficiency concern) each
    diagnostic belongs to."""
    return {
        "adaptive_k": ADAPTIVE_K_DIAGNOSTICS,
        "reranking": RERANKING_DIAGNOSTICS,
        "fusion": FUSION_DIAGNOSTICS,
        "efficiency": EFFICIENCY_DIAGNOSTICS,
    }
