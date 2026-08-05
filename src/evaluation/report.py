"""Evaluation orchestration and reporting (Milestone 2.6, Cell 34).

Responsibility: unify generation, clinical, and retrieval metric
execution behind one deterministic entry point (EvaluationRunner), and
write its results to disk as one machine-readable JSON summary plus one
per-sample JSONL file -- alongside seed, config, code commit, package
versions, and hardware/runtime metadata, per CLAUDE.md's
experiment-recording rules. Comparison tables against the paper's
reported Table 1 numbers remain src.baseline.evaluation.comparison's
own, separate responsibility (contract §6, §15) -- not built here.
Bootstrap significance (contract §7.4) is not populated by this cell
either; EvaluationRunResult carries no significance field at all (the
contract's own EvaluationReport §7.6 does, as an Optional this cell
would leave permanently None) -- deferred cleanly to whichever future
cell implements it, without this cell inventing a placeholder for
something not yet designed to be run.

Built strictly from docs/milestone_2_6_evaluation_contract.md §7.5
(EvaluationRunConfig), §7.6 (EvaluationReport), §9 (Evaluation
Pipelines), §12 (Resume Behavior), §15 (Output Reports), plus this
cell's own explicit requirements, which extend beyond the contract's
original simpler §7.6 sketch (per-sample rows, an explicit
generation/clinical namespace split, structured per-metric failure
accounting) -- EvaluationRunResult is a disclosed, necessary
elaboration of §7.6's EvaluationReport concept, not a literal
implementation of that exact dataclass shape.

Disclosed field-ordering fix: contract §7.5 writes EvaluationRunConfig
with `config_version: str = "1.0"` BEFORE `run_id: str` (no default) --
invalid Python dataclass syntax (a non-default field cannot follow a
defaulted one), the same class of issue already disclosed and fixed for
GeneratorConfig.llava_checkpoint in the Milestone 2.5 contract. Fixed
here by reordering: run_id/system_name/dataset_name (all required,
caller-supplied) first, config_version/split/seed (all defaulted)
after -- field semantics unchanged, only their declaration order.

Namespace separation (this cell's own requirement #5): Cells 31/32's
generation_metrics.build_metric_registry() returns ONE combined dict
covering both generation metrics (rouge_l, bleu4, bert_score) and
clinical metrics (f1radgraph, f1chexbert, f1chexbert_instance) -- there
is no separate "clinical registry" anywhere upstream.
split_generation_and_clinical_registries() below introduces the
generation/clinical split by a fixed metric-name classification (the
same six names those two cells already established), purely for this
orchestrator's own reporting namespace -- it does not change Cell
31/32's registry shape at all, and EvaluationRunner itself never
constructs a registry: generation_metric_registry,
clinical_metric_registry, and retrieval_metric_registry are ALWAYS
injected (never real, never hard-coded) -- this module contains no
reference to radgraph/f1chexbert/rouge/evaluate/bert_score/pytrec_eval
anywhere, not even indirectly; it only ever calls .compute() on
whatever GenerationMetricWrapper/RetrievalMetricWrapper instances it is
handed. Reading installed package *metadata* for provenance
(_detect_package_versions, via importlib.metadata) does not import or
execute any of those packages, so it is not a "hard-coded dependency"
in the sense this requirement guards against.

Resume-safety here means CRASH-SAFE ATOMIC WRITE (temp-file-then-
os.replace, so a crash mid-write never leaves a partially-written file
on disk), NOT skip-already-done resumability -- the exact same
distinction Cell 30's own write_generation_load_summary_atomic already
established, and consistent with contract §12: a single evaluation
run's metric computation is comparatively cheap (a handful of
corpus-level library calls, not thousands of individual per-row API
calls) and is NOT itself resumable, matching official evaluation.py/
evaluate_retriever.py's own single-pass, no-checkpointing behavior.

Requirement #4 ("do not silently drop failed samples") is satisfied at
two levels: sample-level failures are carried through unchanged from
GenerationLoadResult.summary (Cell 30) into
EvaluationRunResult.num_generation_samples_failed_upstream/
num_generation_samples_dropped_malformed -- never recomputed or
silently ignored; metric-level failures (a registry entry's .compute()
raising) are caught per metric, recorded as a MetricFailure, and do NOT
abort the rest of the run -- one failing metric never prevents the
others (or the report) from being produced.

EvaluationRunner.run() requires GenerationLoadResult.query_keys to
contain no None entries (i.e. it was built with
GenerationMetricsConfig.pairing_mode="by_query_key", Cell 30's strict
default) -- positional_official_reproduction mode is rejected here,
since per-sample rows and this cell's own requirement #3 ("preserve
strict query_key alignment") both depend on real, individually
addressable keys.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import EvaluationError
from src.evaluation.generation_metrics import GenerationLoadResult, GenerationMetricWrapper
from src.evaluation.retrieval_metrics import RetrievalMetricWrapper


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _write_atomic(path: Path, content: str) -> None:
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(str(tmp_path), str(path))


# ---------------------------------------------------------------------------
# Run metadata (provenance only -- reads installed-package metadata, never
# imports/executes any metric library)
# ---------------------------------------------------------------------------

_RELEVANT_PACKAGES: Tuple[str, ...] = (
    "torch",
    "transformers",
    "rouge",
    "evaluate",
    "bert-score",
    "radgraph",
    "f1chexbert",
    "pytrec_eval",
)


def _detect_package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package_name in _RELEVANT_PACKAGES:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _detect_hardware() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda ({torch.cuda.get_device_name(0)})"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Generation/clinical namespace split (this cell's own reporting
# convention -- see module docstring)
# ---------------------------------------------------------------------------

_CLINICAL_METRIC_NAMES: FrozenSet[str] = frozenset({"f1radgraph", "f1chexbert", "f1chexbert_instance"})


def split_generation_and_clinical_registries(
    combined_registry: Dict[str, GenerationMetricWrapper],
) -> Tuple[Dict[str, GenerationMetricWrapper], Dict[str, GenerationMetricWrapper]]:
    """Splits a combined registry (e.g. from
    generation_metrics.build_metric_registry / build_mock_metric_registry)
    into (generation_registry, clinical_registry) by the fixed metric-name
    classification already established across Cells 31/32. Does not
    modify or reconstruct the input registry's wrapper instances --
    purely a dict partition."""
    generation_registry: Dict[str, GenerationMetricWrapper] = {}
    clinical_registry: Dict[str, GenerationMetricWrapper] = {}
    for name, wrapper in combined_registry.items():
        if name in _CLINICAL_METRIC_NAMES:
            clinical_registry[name] = wrapper
        else:
            generation_registry[name] = wrapper
    return generation_registry, clinical_registry


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationRunConfig:
    """Contract §7.5, field order fixed (see module docstring)."""

    run_id: str
    system_name: str
    dataset_name: str  # intent: Literal["mimic-cxr", "chexpert"], enforced in __post_init__
    config_version: str = "1.0"
    split: str = "test"  # intent: Literal["train", "valid", "test"]
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be a non-empty string")
        if not self.system_name:
            raise ValueError("system_name must be a non-empty string")
        if self.dataset_name not in ("mimic-cxr", "chexpert"):
            raise ValueError(
                f"dataset_name must be 'mimic-cxr' or 'chexpert', got {self.dataset_name!r}"
            )
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if self.split not in ("train", "valid", "test"):
            raise ValueError(f"split must be 'train', 'valid', or 'test', got {self.split!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricFailure:
    metric_name: str
    namespace: str  # "generation" | "clinical" | "retrieval"
    error_message: str


@dataclass(frozen=True)
class PerSampleResultRow:
    query_key: QueryKey
    generation_scores: Dict[str, Optional[float]]
    clinical_scores: Dict[str, Optional[float]]


@dataclass(frozen=True)
class EvaluationRunResult:
    run_config: EvaluationRunConfig
    generation_corpus_scores: Dict[str, float]
    clinical_corpus_scores: Dict[str, float]
    retrieval_scores: Dict[str, Dict[int, float]]
    per_sample_rows: Tuple[PerSampleResultRow, ...]
    failures: Tuple[MetricFailure, ...]
    num_generation_samples_scored: Optional[int]
    num_generation_samples_failed_upstream: Optional[int]
    num_generation_samples_dropped_malformed: Optional[int]
    package_versions: Dict[str, Optional[str]]
    hardware: str
    git_commit: Optional[str]
    generated_timestamp_utc: str


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EvaluationRunner:
    """Unifies generation, clinical, and retrieval metric execution
    behind one deterministic entry point. All three registries are
    ALWAYS injected via __init__ -- this class never constructs a
    registry itself (requirements #1/#2)."""

    def __init__(
        self,
        *,
        generation_metric_registry: Dict[str, GenerationMetricWrapper],
        clinical_metric_registry: Dict[str, GenerationMetricWrapper],
        retrieval_metric_registry: Dict[str, RetrievalMetricWrapper],
    ) -> None:
        overlap = set(generation_metric_registry) & set(clinical_metric_registry)
        if overlap:
            raise EvaluationError(
                f"EvaluationRunner: the same metric name(s) {sorted(overlap)} appear in "
                f"both generation_metric_registry and clinical_metric_registry -- each "
                f"metric must belong to exactly one namespace"
            )
        self._generation_registry = generation_metric_registry
        self._clinical_registry = clinical_metric_registry
        self._retrieval_registry = retrieval_metric_registry

    def run(
        self,
        run_config: EvaluationRunConfig,
        *,
        generation_load_result: Optional[GenerationLoadResult] = None,
        retrieval_positives: Optional[Dict[QueryKey, Sequence[QueryKey]]] = None,
        retrieval_ranked: Optional[Dict[QueryKey, Sequence[QueryKey]]] = None,
    ) -> EvaluationRunResult:
        """Runs every registered metric, in deterministic (sorted-by-name)
        order within each namespace, against whichever of
        generation_load_result / (retrieval_positives + retrieval_ranked)
        is provided -- at least one must be. A metric whose .compute()
        raises is recorded as a MetricFailure and does not abort the run;
        every other metric, and the report itself, is still produced.
        """
        # Checked BEFORE the "at least one input provided" check below,
        # deliberately: a caller who supplies exactly one of
        # retrieval_positives/retrieval_ranked clearly intended a
        # retrieval run and made a mistake completing the pair -- that
        # specific mistake must be reported, not masked by a generic
        # "provide something" message that would fire first if checked
        # in the other order (both conditions can be simultaneously
        # true when generation_load_result is also None).
        if (retrieval_positives is None) != (retrieval_ranked is None):
            raise EvaluationError(
                "EvaluationRunner.run: retrieval_positives and retrieval_ranked must "
                "both be provided or both omitted"
            )
        if generation_load_result is None and retrieval_positives is None:
            raise EvaluationError(
                "EvaluationRunner.run: at least one of generation_load_result or "
                "(retrieval_positives + retrieval_ranked) must be provided"
            )

        generation_corpus_scores: Dict[str, float] = {}
        clinical_corpus_scores: Dict[str, float] = {}
        retrieval_scores: Dict[str, Dict[int, float]] = {}
        failures: List[MetricFailure] = []
        per_metric_example_scores: Dict[str, Dict[QueryKey, float]] = {}

        num_generation_samples_scored: Optional[int] = None
        num_generation_samples_failed_upstream: Optional[int] = None
        num_generation_samples_dropped_malformed: Optional[int] = None

        if generation_load_result is not None:
            if any(query_key is None for query_key in generation_load_result.query_keys):
                raise EvaluationError(
                    "EvaluationRunner.run: generation_load_result must use strict "
                    "query_key-based pairing (GenerationMetricsConfig.pairing_mode="
                    "'by_query_key') -- positional_official_reproduction mode is not "
                    "supported here, since per-sample rows require real query keys"
                )
            num_generation_samples_scored = generation_load_result.summary.num_paired
            num_generation_samples_failed_upstream = (
                generation_load_result.summary.num_failed_generations
            )
            num_generation_samples_dropped_malformed = (
                generation_load_result.summary.num_dropped_malformed_hypothesis
            )

            for namespace, registry, scores_out in (
                ("generation", self._generation_registry, generation_corpus_scores),
                ("clinical", self._clinical_registry, clinical_corpus_scores),
            ):
                for name in sorted(registry):
                    wrapper = registry[name]
                    try:
                        computation = wrapper.compute(
                            generation_load_result.hyps, generation_load_result.refs
                        )
                    except Exception as exc:
                        failures.append(
                            MetricFailure(metric_name=name, namespace=namespace, error_message=str(exc))
                        )
                        continue
                    scores_out[name] = computation.corpus_score
                    if computation.per_example_scores is not None:
                        per_metric_example_scores[name] = dict(
                            zip(generation_load_result.query_keys, computation.per_example_scores)
                        )

        if retrieval_positives is not None:
            for name in sorted(self._retrieval_registry):
                wrapper = self._retrieval_registry[name]
                try:
                    computation = wrapper.compute(retrieval_positives, retrieval_ranked)
                except Exception as exc:
                    failures.append(
                        MetricFailure(metric_name=name, namespace="retrieval", error_message=str(exc))
                    )
                    continue
                retrieval_scores[name] = dict(computation.scores_at_k)

        per_sample_rows: Tuple[PerSampleResultRow, ...] = ()
        if generation_load_result is not None:
            rows: List[PerSampleResultRow] = []
            for query_key in generation_load_result.query_keys:
                generation_row_scores = {
                    name: per_metric_example_scores.get(name, {}).get(query_key)
                    for name in sorted(self._generation_registry)
                }
                clinical_row_scores = {
                    name: per_metric_example_scores.get(name, {}).get(query_key)
                    for name in sorted(self._clinical_registry)
                }
                rows.append(
                    PerSampleResultRow(
                        query_key=query_key,
                        generation_scores=generation_row_scores,
                        clinical_scores=clinical_row_scores,
                    )
                )
            per_sample_rows = tuple(rows)

        return EvaluationRunResult(
            run_config=run_config,
            generation_corpus_scores=generation_corpus_scores,
            clinical_corpus_scores=clinical_corpus_scores,
            retrieval_scores=retrieval_scores,
            per_sample_rows=per_sample_rows,
            failures=tuple(failures),
            num_generation_samples_scored=num_generation_samples_scored,
            num_generation_samples_failed_upstream=num_generation_samples_failed_upstream,
            num_generation_samples_dropped_malformed=num_generation_samples_dropped_malformed,
            package_versions=_detect_package_versions(),
            hardware=_detect_hardware(),
            git_commit=_resolve_git_commit(),
            generated_timestamp_utc=_utc_now_iso(),
        )


# ---------------------------------------------------------------------------
# Resume-safe (crash-safe atomic write, not skip-already-done) report writers
# ---------------------------------------------------------------------------


def write_evaluation_summary_json_atomic(path: Path, result: EvaluationRunResult) -> None:
    """Writes the corpus-level summary only -- excludes per_sample_rows,
    which belongs in its own JSONL file (write_per_sample_rows_jsonl_atomic).
    Atomic: written to a temp file, then os.replace'd into place -- a
    crash mid-write never leaves a truncated/corrupt file at `path`."""
    payload = {
        "run_config": asdict(result.run_config),
        "generation_corpus_scores": result.generation_corpus_scores,
        "clinical_corpus_scores": result.clinical_corpus_scores,
        "retrieval_scores": {
            name: {str(k): v for k, v in scores.items()}
            for name, scores in result.retrieval_scores.items()
        },
        "failures": [asdict(failure) for failure in result.failures],
        "num_generation_samples_scored": result.num_generation_samples_scored,
        "num_generation_samples_failed_upstream": result.num_generation_samples_failed_upstream,
        "num_generation_samples_dropped_malformed": result.num_generation_samples_dropped_malformed,
        "package_versions": result.package_versions,
        "hardware": result.hardware,
        "git_commit": result.git_commit,
        "generated_timestamp_utc": result.generated_timestamp_utc,
    }
    _write_atomic(Path(path), json.dumps(payload, ensure_ascii=False, indent=2))


def write_per_sample_rows_jsonl_atomic(path: Path, rows: Sequence[PerSampleResultRow]) -> None:
    """One JSON object per line, one line per query_key. Atomic: all rows
    are serialized in memory first, then the whole file is written to a
    temp path and os.replace'd into place -- either every row is visible
    at `path`, or (on a crash before the replace) none of this write's
    rows are, never a partial file."""
    lines = [
        json.dumps(
            {
                "query_key": list(row.query_key),
                "generation_scores": row.generation_scores,
                "clinical_scores": row.clinical_scores,
            },
            ensure_ascii=False,
        )
        for row in rows
    ]
    content = "\n".join(lines) + ("\n" if lines else "")
    _write_atomic(Path(path), content)
