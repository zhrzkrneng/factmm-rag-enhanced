"""Milestone 3.0 (Cell 47) -- innovation configuration + experiment matrix.

Responsibility: the single innovation configuration contract (Section
11) with independent enable/disable switches for I1/I2/I3, and the
required E0-E7 experiment matrix (Section 7). Design/contract only -- no
retrieval, reranking, fusion, or generation is run here.

All defaults reproduce baseline behavior when every innovation is
disabled (Section 11) -- `InnovationConfig()` with no arguments is
exactly E0.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple

from src.innovation.adaptive_retrieval.adaptive_k import AdaptiveKConfig
from src.innovation.factual_reranking.reranker import RerankConfig
from src.innovation.multi_report_fusion.fusion import FusionConfig

# Project-wide deterministic seed convention, matching
# src.baseline.generation.generator.GenerationConfig's default (seed=42)
# and src.evaluation.significance.SignificanceConfig's default (seed=42)
# -- Section 7's "identical random seed" requirement reuses the same
# established convention rather than inventing a new one.
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class InnovationConfig:
    """The single innovation configuration contract (Section 11).

    Attributes:
        adaptive_retrieval_enabled: I1 switch.
        factual_reranking_enabled: I2 switch.
        confidence_gated_fusion_enabled: I3 switch.
        adaptive_k: I1's AdaptiveKConfig (validated on construction).
        reranking: I2's RerankConfig (validated on construction).
        fusion: I3's FusionConfig (validated on construction).
        random_seed: the single seed shared by every experiment arm
            (Section 7: "identical random seed").

    All three switches default to False and every nested config defaults
    to its own baseline-reproducing defaults, so `InnovationConfig()`
    with no arguments is E0 (Section 11: "All defaults must reproduce
    baseline behavior when all innovations are disabled").
    """

    adaptive_retrieval_enabled: bool = False
    factual_reranking_enabled: bool = False
    confidence_gated_fusion_enabled: bool = False
    adaptive_k: AdaptiveKConfig = field(default_factory=AdaptiveKConfig)
    reranking: RerankConfig = field(default_factory=RerankConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    random_seed: int = DEFAULT_RANDOM_SEED

    def to_json_dict(self) -> dict:
        """Deterministic serialization (Section 12, test 18): the same
        InnovationConfig always serializes to the same dict -- field
        order is fixed by dataclasses.asdict's declaration order, and no
        field value is derived from wall-clock time, randomness, or
        object identity."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), sort_keys=True)


# ---------------------------------------------------------------------------
# Experiment matrix: E0-E7 (Section 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentArm:
    """One arm of the required E0-E7 ablation matrix.

    Attributes:
        arm_id: "E0".."E7", deterministic given the three switches (see
            compute_arm_id).
        description: short human-readable description.
        config: the InnovationConfig this arm evaluates.
    """

    arm_id: str
    description: str
    config: InnovationConfig


_ARM_ID_BY_SWITCHES: Dict[Tuple[bool, bool, bool], str] = {
    (False, False, False): "E0",
    (True, False, False): "E1",
    (False, True, False): "E2",
    (False, False, True): "E3",
    (True, True, False): "E4",
    (True, False, True): "E5",
    (False, True, True): "E6",
    (True, True, True): "E7",
}

_ARM_DESCRIPTION_BY_ID: Dict[str, str] = {
    "E0": "baseline (no innovation enabled)",
    "E1": "Adaptive-K only",
    "E2": "Reranking only",
    "E3": "Confidence-Gated Fusion only",
    "E4": "Adaptive-K + Reranking",
    "E5": "Adaptive-K + Fusion",
    "E6": "Reranking + Fusion",
    "E7": "Adaptive-K + Reranking + Fusion",
}


def compute_arm_id(config: InnovationConfig) -> str:
    """Deterministic arm-ID lookup from an InnovationConfig's three
    switches (Section 12, test 4: "experiment arm IDs are deterministic")
    -- a pure function of
    (adaptive_retrieval_enabled, factual_reranking_enabled,
    confidence_gated_fusion_enabled); nested config values (thresholds,
    weights, budgets) never affect the arm ID."""
    key = (
        config.adaptive_retrieval_enabled,
        config.factual_reranking_enabled,
        config.confidence_gated_fusion_enabled,
    )
    return _ARM_ID_BY_SWITCHES[key]


def _build_experiment_matrix() -> Tuple[ExperimentArm, ...]:
    arms = []
    for switches, arm_id in _ARM_ID_BY_SWITCHES.items():
        adaptive_on, reranking_on, fusion_on = switches
        config = InnovationConfig(
            adaptive_retrieval_enabled=adaptive_on,
            factual_reranking_enabled=reranking_on,
            confidence_gated_fusion_enabled=fusion_on,
        )
        arms.append(ExperimentArm(arm_id=arm_id, description=_ARM_DESCRIPTION_BY_ID[arm_id], config=config))
    # Sorted by arm_id ("E0".."E7") for a deterministic, human-readable order.
    return tuple(sorted(arms, key=lambda arm: arm.arm_id))


EXPERIMENT_MATRIX: Tuple[ExperimentArm, ...] = _build_experiment_matrix()

REQUIRED_ARM_IDS: Tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7")

# Section 9: primary statistical comparisons, preserving (not
# reimplementing) src.evaluation.significance.paired_bootstrap_significance.
REQUIRED_STATISTICAL_COMPARISONS: Tuple[Tuple[str, str], ...] = (
    ("E0", "E1"),
    ("E0", "E2"),
    ("E0", "E3"),
    ("E0", "E7"),
)


def experiment_matrix_to_json() -> list:
    """JSON-ready experiment matrix, for cell47_experiment_matrix.json."""
    return [
        {"arm_id": arm.arm_id, "description": arm.description, "config": arm.config.to_json_dict()}
        for arm in EXPERIMENT_MATRIX
    ]
