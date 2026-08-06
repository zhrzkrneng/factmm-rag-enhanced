"""Milestone 3.0 (Cell 47) -- innovation pipeline architecture.

Responsibility: the ordered pipeline-stage contract (Cell 47 spec,
Section 2) and the exact-shape "what I1/I2/I3 may see at inference time"
contract (Section 12, test 14: "no target-report field is part of
innovation inference input"). Design/contract only -- no retrieval,
reranking, or fusion algorithm is implemented here (those belong to
Cells 48-50).

Import direction: this module may import from src.baseline.** (QueryKey)
per docs/phase_3_innovation_contract.md Section 13's one-directional
rule ("src/innovation/** may import from src/baseline/**... none of
those may ever import from src/innovation/**"). See
cell47_report.md's design-caveats section for why this project follows
that contract's version of the rule rather than this package's own
(narrower, pre-Milestone-2.8-pivot) __init__.py docstring, which predates
the real, load-bearing baseline interfaces now living under
src/baseline/.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.innovation.experiment_contract import InnovationConfig

# ---------------------------------------------------------------------------
# Pipeline stages (Cell 47 spec, Section 2)
# ---------------------------------------------------------------------------

STAGE_QUERY = "query"
STAGE_INITIAL_RETRIEVAL = "initial_retrieval"
STAGE_ADAPTIVE_RETRIEVAL = "adaptive_retrieval_dynamic_k"  # I1
STAGE_FACTUAL_RERANKING = "factual_clinical_reranking"  # I2
STAGE_CONFIDENCE_GATED_FUSION = "confidence_gated_multi_report_fusion"  # I3
STAGE_EVIDENCE_CONTEXT = "evidence_context"
STAGE_PROMPT_BUILDER = "prompt_builder"
STAGE_GENERATOR = "generator"
STAGE_GENERATED_REPORT = "generated_report"

PIPELINE_STAGES: Tuple[str, ...] = (
    STAGE_QUERY,
    STAGE_INITIAL_RETRIEVAL,
    STAGE_ADAPTIVE_RETRIEVAL,
    STAGE_FACTUAL_RERANKING,
    STAGE_CONFIDENCE_GATED_FUSION,
    STAGE_EVIDENCE_CONTEXT,
    STAGE_PROMPT_BUILDER,
    STAGE_GENERATOR,
    STAGE_GENERATED_REPORT,
)

# Which InnovationConfig flag gates which optional stage -- the three
# innovation stages are the only ones any switch ever removes; every
# other stage is always present (baseline or innovation alike), which is
# what "the three innovations must remain independently switchable"
# (Section 2) actually requires structurally.
_STAGE_GATE_FLAG = {
    STAGE_ADAPTIVE_RETRIEVAL: "adaptive_retrieval_enabled",
    STAGE_FACTUAL_RERANKING: "factual_reranking_enabled",
    STAGE_CONFIDENCE_GATED_FUSION: "confidence_gated_fusion_enabled",
}


def active_stages(config: InnovationConfig) -> Tuple[str, ...]:
    """Returns the ordered stages active for `config`, skipping any of
    I1/I2/I3 whose switch is disabled. All non-innovation stages are
    always present.

    With every switch disabled (InnovationConfig()'s defaults, i.e. E0),
    this returns exactly the baseline's own stage sequence -- no I1/I2/I3
    stage name appears -- which is the structural form of "baseline
    behavior contract preserved" (Section 12, test 13).
    """
    result = []
    for stage in PIPELINE_STAGES:
        gate_flag = _STAGE_GATE_FLAG.get(stage)
        if gate_flag is not None and not getattr(config, gate_flag):
            continue
        result.append(stage)
    return tuple(result)


def describe_architecture() -> dict:
    """JSON-ready description of the pipeline architecture, for
    cell47_innovation_architecture.json."""
    return {
        "pipeline_stages": list(PIPELINE_STAGES),
        "innovation_stages": {
            "I1_adaptive_retrieval_dynamic_k": STAGE_ADAPTIVE_RETRIEVAL,
            "I2_factual_clinical_reranking": STAGE_FACTUAL_RERANKING,
            "I3_confidence_gated_multi_report_fusion": STAGE_CONFIDENCE_GATED_FUSION,
        },
        "stage_gate_flags": dict(_STAGE_GATE_FLAG),
        "baseline_stages_when_all_disabled": list(active_stages(InnovationConfig())),
        "independently_switchable": True,
        "note": "Design/contract only -- no retrieval, reranking, or "
                "fusion algorithm is implemented by this module. See "
                "src/innovation/adaptive_retrieval/adaptive_k.py, "
                "src/innovation/factual_reranking/reranker.py, and "
                "src/innovation/multi_report_fusion/fusion.py for the "
                "per-innovation config/result contracts (Cells 48-50 "
                "implement the actual decision algorithms).",
    }


# ---------------------------------------------------------------------------
# Inference-input contract (Section 12, test 14)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InnovationInferenceContext:
    """Everything I1/I2/I3 may see at inference time for one query.

    Deliberately excludes any target/ground-truth report field: I1
    (which k to use), I2 (how to rerank), and I3 (whether/how much to
    fuse) must never have access to the answer they are helping retrieve
    evidence for. `candidate_texts` holds the retrieved CANDIDATES' own
    text (i.e. corpus evidence, exactly like retrieval_adapter's corpus
    records) -- never the query's target report.

    This dataclass's field set is itself the contract: any future
    innovation code that needs more information than this must add a
    field here explicitly (and any reviewer/test can immediately see
    whether a target-report-shaped field was added), rather than an
    innovation function silently reaching for a target report through
    some other channel.
    """

    query_key: QueryKey
    candidate_keys: Tuple[QueryKey, ...]
    candidate_scores: Tuple[float, ...]
    candidate_texts: Tuple[str, ...]


def inference_context_field_names() -> Tuple[str, ...]:
    """Returns InnovationInferenceContext's field names, for tests that
    assert no target-report-shaped field is present."""
    return tuple(f.name for f in fields(InnovationInferenceContext))
