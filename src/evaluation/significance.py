"""Bootstrap confidence intervals and significance testing (Milestone 2.6,
Cell 35).

Responsibility: this project's own addition beyond the original paper
(PROPOSED_EXTENSION -- the paper reports only a p-value with no stated
test, and no confidence intervals at all; contract §2, §7.4,
docs/paper_analysis.md §15). Provides paired bootstrap confidence
intervals and a documented significance test (paired_bootstrap_significance)
for comparing two systems' per-example metric scores -- e.g. two
GenerationExampleScore.f1_radgraph tuples, or two RetrievalMetricComputation
per-query breakdowns, from Cells 30-34's own already-implemented output
shapes.

Deterministic given config.seed via a LOCAL numpy.random.Generator
instance (numpy.random.default_rng(config.seed)) -- NEVER the global
np.random/torch RNG state (contract §13). This is a direct, disclosed
carry-over of the exact lesson learned fixing
test_synthetic_end_to_end_smoke_lifecycle's real flakiness bug earlier
in this project (seeding-order hazard: constructing something that
consumes global RNG state before seeding makes "identical seed =>
identical result" false). The Generator is constructed fresh inside
paired_bootstrap_significance from config.seed on every call, never
reused or mutated across calls, so two calls with the same seed always
produce byte-identical results regardless of call order or prior global
RNG state -- verified directly by this cell's own repeated-call
determinism tests.

numpy is a hard, always-installed project dependency (already imported
at module scope throughout src/retrieval/*), not a "metric dependency"
in the sense Cells 31-33 guard against with lazy loading -- imported
normally here, matching existing project precedent.

No resume machinery here, unlike src.baseline.evaluation.oracle (this
same cell's other half): a bootstrap significance computation is a
single fast, in-memory operation (vectorized numpy resampling, no
external library call, no per-row I/O) -- contract §12's own
resumability distinction applies only to genuinely expensive,
interruption-prone work like Oracle construction, not to this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from src.common.exceptions import EvaluationError

_SIGNIFICANCE_METHODS: tuple = ("paired_bootstrap",)


@dataclass(frozen=True)
class SignificanceConfig:
    """Contract §7.4."""

    config_version: str = "1.0"
    method: Literal["paired_bootstrap"] = "paired_bootstrap"
    num_bootstrap_samples: int = 10000
    confidence_level: float = 0.95
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if self.method not in _SIGNIFICANCE_METHODS:
            raise ValueError(f"method must be one of {_SIGNIFICANCE_METHODS}, got {self.method!r}")
        if (
            not isinstance(self.num_bootstrap_samples, int)
            or isinstance(self.num_bootstrap_samples, bool)
            or self.num_bootstrap_samples < 1
        ):
            raise ValueError(
                f"num_bootstrap_samples must be a positive integer, got {self.num_bootstrap_samples!r}"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")


@dataclass(frozen=True)
class SignificanceResult:
    """Contract §7.6."""

    metric_name: str
    system_a_name: str
    system_b_name: str
    system_a_mean: float
    system_b_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    num_bootstrap_samples: int
    seed: int


def paired_bootstrap_significance(
    system_a_scores: Sequence[float],
    system_b_scores: Sequence[float],
    *,
    metric_name: str,
    system_a_name: str,
    system_b_name: str,
    config: SignificanceConfig,
) -> SignificanceResult:
    """Contract §8. Paired bootstrap over per-example score differences:
    resamples (with replacement) the SAME index for both systems on
    every draw -- preserving the pairing -- computes each resample's
    mean difference, and derives a confidence interval from the
    empirical percentiles of that distribution plus a two-sided p-value
    from the proportion of resamples that cross zero relative to the
    observed direction. A standard paired-bootstrap formulation
    (Efron & Tibshirani); PROPOSED_EXTENSION, not a reproduction of any
    specific procedure the paper names (it names none, contract §2).

    Raises:
        EvaluationError: system_a_scores/system_b_scores differ in
            length, or either is empty.
    """
    if len(system_a_scores) != len(system_b_scores):
        raise EvaluationError(
            f"paired_bootstrap_significance: system_a_scores and system_b_scores must "
            f"be the same length, got {len(system_a_scores)} vs {len(system_b_scores)}"
        )
    if len(system_a_scores) == 0:
        raise EvaluationError("paired_bootstrap_significance: scores must be non-empty")

    a = np.asarray(system_a_scores, dtype=np.float64)
    b = np.asarray(system_b_scores, dtype=np.float64)
    diffs = a - b
    n = diffs.shape[0]

    rng = np.random.default_rng(config.seed)
    resample_indices = rng.integers(0, n, size=(config.num_bootstrap_samples, n))
    bootstrap_means = diffs[resample_indices].mean(axis=1)

    alpha = 1.0 - config.confidence_level
    ci_low = float(np.quantile(bootstrap_means, alpha / 2))
    ci_high = float(np.quantile(bootstrap_means, 1.0 - alpha / 2))

    observed_diff = float(diffs.mean())
    if observed_diff >= 0:
        p_one_sided = float(np.mean(bootstrap_means <= 0))
    else:
        p_one_sided = float(np.mean(bootstrap_means >= 0))
    p_value = min(1.0, 2.0 * p_one_sided)

    return SignificanceResult(
        metric_name=metric_name,
        system_a_name=system_a_name,
        system_b_name=system_b_name,
        system_a_mean=float(a.mean()),
        system_b_mean=float(b.mean()),
        mean_difference=observed_diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        num_bootstrap_samples=config.num_bootstrap_samples,
        seed=config.seed,
    )
