"""Retrieval quality metrics (Milestone 2.6, Cell 33).

Responsibility: compute Mean Reciprocal Rank (MRR), Recall@K, and NDCG@K
for a trained retriever's rankings against known positive/relevant
documents, matching the official FactMM-RAG evaluate_retriever.py
definitions (contract §3.2, §4, §8).

Cell 33 scope only: config, dataclasses, the two contract-specified
metric functions, a uniform RetrievalMetricWrapper interface + registry
integration, and mock wrappers. The evaluate_retrieval() orchestrator
that would consume a real FaissFlatIPIndex + real query embeddings
(contract §9.2) is explicitly out of scope here -- a later cell, along
with Oracle evaluation, bootstrap significance, and the JSON report
writer.

MRR is preserved EXACTLY as the official repository's own hand-written
compute_mrr (contract §3.2) -- reimplemented with this project's
QueryKey typing instead of the official pickle format's integer pids,
never delegated to pytrec_eval or any other library. Recall@K/NDCG@K
ARE delegated to pytrec_eval (contract §4), lazily imported, mirroring
official evaluate_retriever.py's own single
`pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut', 'recall'})` call and
subsequent `result[k]["recall_100"]`/`result[k]["ndcg_cut_100"]`-style
extraction -- verbatim official code, already audited in the contract
(§3.2), not guessed at. pytrec_eval is added to requirements.txt in
this same cell, the deferred dependency contract §18 flagged.

Both `compute_mrr_at_k` and `compute_recall_ndcg_at_k` exist as
standalone, directly-testable functions matching contract §8's exact
signatures -- unlike Cells 31/32, which consolidated their bare-function
specs directly into wrapper-class compute() bodies, this cell keeps the
free functions distinctly named and separately unit-tested, precisely
because this cell's own explicit requirement is to make the official
MRR algorithm's preservation structurally obvious and independently
auditable, not folded invisibly into class internals. MRRMetric /
RecallAtKMetric / NdcgAtKMetric are thin registry-integration wrappers
around these same functions, binding RetrievalMetricsConfig.k_values at
construction (never passed per compute() call) -- the same
"config bound once, uniform compute() signature thereafter" discipline
already established for GenerationMetricWrapper (Cell 31).

Both functions share one strict validation pass
(_validate_retrieval_inputs) enforcing this cell's own explicit
requirements -- deliberately STRICTER than official code, which
silently tolerates a positives-having query absent from the ranked
dict (treating it as a zero contributor) and never validates
duplicate/malformed keys at all:
  - every QueryKey (both dict keys and every element of every
    positives/ranked sequence) must be a well-formed 3-tuple of
    non-empty strings;
  - positives and ranked must cover the EXACT SAME query set (neither
    a superset nor a subset of the other);
  - every positives[query_key] must be non-empty, with no duplicate
    keys;
  - every ranked[query_key] must contain no duplicate retrieved keys.
No opt-in "reproduce official's silent gap-tolerance" mode is offered
here, unlike Cell 30's pairing_mode -- this cell's own requirements are
unconditionally about strictness, not about controlled A/B
reproduction of a specific official gap.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import EvaluationError
from src.evaluation.generation_metrics import _classify_metric_dependency_error

# Reused rather than duplicated: this is a ~40-line, genuinely
# general-purpose (not generation-specific) exception classifier, unlike
# the trivial 2-3 line helpers (_utc_now_iso, _write_meta_atomic) this
# project deliberately re-copies into every module that needs them --
# duplicating this one would risk real drift between two copies of the
# same six-category taxonomy.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalMetricsConfig:
    """Contract §7.2. chexbert_threshold/radgraph_threshold are purely
    descriptive provenance (which upstream mining run produced the
    positives being scored) -- matching official evaluate_retriever.py's
    own --chexbert_threshold/--radgraph_threshold CLI flags, which the
    script itself never uses for filtering (§3.2); not consumed by any
    function in this cell."""

    config_version: str = "1.0"
    k_values: Tuple[int, ...] = (100, 200, 500, 1000)
    chexbert_threshold: float = 1.0
    radgraph_threshold: float = 0.4

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.k_values:
            raise ValueError("k_values must be non-empty")
        for k in self.k_values:
            if not isinstance(k, int) or isinstance(k, bool) or k < 1:
                raise ValueError(f"k_values must contain only positive integers, got {k!r}")
        if len(set(self.k_values)) != len(self.k_values):
            raise ValueError(f"k_values must not contain duplicates, got {self.k_values!r}")
        if list(self.k_values) != sorted(self.k_values):
            raise ValueError(f"k_values must be strictly increasing, got {self.k_values!r}")
        if self.chexbert_threshold < 0:
            raise ValueError("chexbert_threshold must be non-negative")
        if self.radgraph_threshold < 0:
            raise ValueError("radgraph_threshold must be non-negative")

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "k_values": list(self.k_values),
            "chexbert_threshold": self.chexbert_threshold,
            "radgraph_threshold": self.radgraph_threshold,
        }


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _is_well_formed_query_key(key: object) -> bool:
    return (
        isinstance(key, tuple)
        and len(key) == 3
        and all(isinstance(part, str) and part for part in key)
    )


def _validate_retrieval_inputs(
    positives: Dict[QueryKey, Sequence[QueryKey]],
    ranked: Dict[QueryKey, Sequence[QueryKey]],
    *,
    metric_name: str,
) -> None:
    """Contract'd requirements: strict query-key validation, ranking
    validation, duplicate detection (see module docstring for the full
    itemized list). Raises EvaluationError naming exactly what's wrong
    -- never silently coerces, drops, or pads a malformed input."""
    if not positives:
        raise EvaluationError(f"{metric_name}: positives must be non-empty")
    if not ranked:
        raise EvaluationError(f"{metric_name}: ranked must be non-empty")

    for key in positives:
        if not _is_well_formed_query_key(key):
            raise EvaluationError(f"{metric_name}: malformed query_key in positives: {key!r}")
    for key in ranked:
        if not _is_well_formed_query_key(key):
            raise EvaluationError(f"{metric_name}: malformed query_key in ranked: {key!r}")

    positives_keys: Set[QueryKey] = set(positives)
    ranked_keys: Set[QueryKey] = set(ranked)
    if positives_keys != ranked_keys:
        missing_from_ranked = sorted(positives_keys - ranked_keys)
        extra_in_ranked = sorted(ranked_keys - positives_keys)
        raise EvaluationError(
            f"{metric_name}: positives and ranked must cover the exact same query "
            f"set -- {len(missing_from_ranked)} key(s) in positives missing from "
            f"ranked (e.g. {missing_from_ranked[:3]}), {len(extra_in_ranked)} key(s) "
            f"in ranked not present in positives (e.g. {extra_in_ranked[:3]})"
        )

    for query_key, positive_seq in positives.items():
        if len(positive_seq) == 0:
            raise EvaluationError(
                f"{metric_name}: positives[{query_key}] must be non-empty -- a "
                f"zero-positive query must be dropped by the caller before "
                f"reaching this function, never passed through (matches official "
                f"code's own pre-filtering, contract §3.2)"
            )
        for candidate in positive_seq:
            if not _is_well_formed_query_key(candidate):
                raise EvaluationError(
                    f"{metric_name}: malformed candidate key in positives[{query_key}]: "
                    f"{candidate!r}"
                )
        if len(set(positive_seq)) != len(positive_seq):
            raise EvaluationError(f"{metric_name}: positives[{query_key}] contains duplicate keys")

    for query_key, ranked_seq in ranked.items():
        for candidate in ranked_seq:
            if not _is_well_formed_query_key(candidate):
                raise EvaluationError(
                    f"{metric_name}: malformed candidate key in ranked[{query_key}]: "
                    f"{candidate!r}"
                )
        if len(set(ranked_seq)) != len(ranked_seq):
            raise EvaluationError(
                f"{metric_name}: ranked[{query_key}] contains duplicate retrieved keys"
            )


def _validate_k(k: int, *, metric_name: str) -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise EvaluationError(f"{metric_name}: k must be a positive integer, got {k!r}")


# ---------------------------------------------------------------------------
# compute_mrr_at_k -- the preserved official hand-written algorithm
# ---------------------------------------------------------------------------


def compute_mrr_at_k(
    positives: Dict[QueryKey, Sequence[QueryKey]],
    ranked: Dict[QueryKey, Sequence[QueryKey]],
    k: int,
) -> float:
    """Own reimplementation of official evaluate_retriever.py::compute_mrr
    (contract §3.2), ported from integer pids/pickle format to this
    project's QueryKey typing -- NEVER delegated to pytrec_eval or any
    other library (contract §4 classifies MRR as "implemented inside
    the repository" in the official code itself, and it stays that way
    here; this cell's own requirements #1/#2 explicitly forbid silently
    replacing it). Same semantics: first-hit-in-top-k against a query's
    (possibly multi-element) positive set, independently computed per k
    -- call this once per k value, exactly matching official code's own
    four independent compute_mrr(..., MaxMRRRank) calls at
    k=100/200/500/1000, never derived from a larger-k result.

    Official code pads every ranked list to a fixed 1000 slots (with a
    sentinel placeholder pid) before indexing candidate_pid[i] for
    i in range(MaxMRRRank) -- an implementation detail of its own
    upstream FAISS-search materialization, not a semantic requirement.
    This reimplementation instead slices ranked[query_key][:k], which
    behaves identically for any real ranking (a short real ranking
    simply yields no match beyond its own length, exactly like hitting
    the sentinel padding would) without depending on a magic
    sentinel-padding convention.

    Raises:
        EvaluationError: any of _validate_retrieval_inputs's checks
            fail, or k is not a positive integer.
    """
    _validate_retrieval_inputs(positives, ranked, metric_name="mrr")
    _validate_k(k, metric_name="mrr")

    total = 0.0
    for query_key, ranked_seq in ranked.items():
        positive_set = set(positives[query_key])
        for rank_index, candidate_key in enumerate(ranked_seq[:k]):
            if candidate_key in positive_set:
                total += 1.0 / (rank_index + 1)
                break
    return total / len(positives)


# ---------------------------------------------------------------------------
# compute_recall_ndcg_at_k -- delegated to pytrec_eval
# ---------------------------------------------------------------------------


def _build_query_key_index(
    positives: Dict[QueryKey, Sequence[QueryKey]], ranked: Dict[QueryKey, Sequence[QueryKey]]
) -> Dict[QueryKey, str]:
    """Assigns every QueryKey appearing anywhere (as a query or as a
    candidate, in either dict) a unique string id, for pytrec_eval's
    own string-keyed qrel/run dicts. An index-based encoding (not e.g.
    "|".join(key)) is used deliberately -- it cannot collide regardless
    of what characters a dataset/patient/study_id string happens to
    contain, unlike a delimiter-joined string would."""
    all_keys: Set[QueryKey] = set(positives) | set(ranked)
    for seq in positives.values():
        all_keys.update(seq)
    for seq in ranked.values():
        all_keys.update(seq)
    return {key: str(i) for i, key in enumerate(sorted(all_keys))}


def _default_pytrec_eval_factory(qrel: dict, measures: set) -> object:
    import pytrec_eval  # lazy: only imported when actually constructing, never at module scope

    return pytrec_eval.RelevanceEvaluator(qrel, measures)


def compute_recall_ndcg_at_k(
    positives: Dict[QueryKey, Sequence[QueryKey]],
    ranked: Dict[QueryKey, Sequence[QueryKey]],
    k_values: Sequence[int],
    *,
    pytrec_eval_factory: Optional[Callable[[dict, set], object]] = None,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Delegates to pytrec_eval.RelevanceEvaluator (contract §3.2, §4),
    in-process, mirroring official evaluate_retriever.py's own single
    `pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut', 'recall'})` call
    verbatim -- the bare family names, not per-k measure strings; k=100/
    200/500/1000 are among trec_eval's own default cutoff points for
    both families, which is exactly why official code can extract
    result[k]["recall_100"]/result[k]["ndcg_cut_100"] afterward without
    ever passing an explicit cutoff list. Returns (recall_at_k, ndcg_at_k),
    each a corpus-level mean across all queries in positives -- matching
    official EvalDevQuery's own `recall_100 / eval_query_cnt` averaging.

    Raises:
        EvaluationError: any of _validate_retrieval_inputs's checks
            fail, any k in k_values is not a positive integer, or the
            injected/real pytrec_eval call raises (classified via
            _classify_metric_dependency_error and chained).
    """
    _validate_retrieval_inputs(positives, ranked, metric_name="recall_ndcg")
    if not k_values:
        raise EvaluationError("recall_ndcg: k_values must be non-empty")
    for k in k_values:
        _validate_k(k, metric_name="recall_ndcg")

    key_index = _build_query_key_index(positives, ranked)
    qrel = {
        key_index[query_key]: {key_index[pos]: 1 for pos in positive_seq}
        for query_key, positive_seq in positives.items()
    }
    run = {
        key_index[query_key]: {
            key_index[candidate]: -(rank + 1) for rank, candidate in enumerate(ranked_seq)
        }
        for query_key, ranked_seq in ranked.items()
    }

    factory = pytrec_eval_factory or _default_pytrec_eval_factory
    try:
        evaluator = factory(qrel, {"ndcg_cut", "recall"})
        per_query_result = evaluator.evaluate(run)
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        category, detail = _classify_metric_dependency_error(exc)
        raise EvaluationError(f"recall_ndcg: {category}: {detail}") from exc

    try:
        num_queries = len(per_query_result)
        if num_queries == 0:
            raise ValueError("pytrec_eval returned an empty per-query result set")
        recall_at_k: Dict[int, float] = {}
        ndcg_at_k: Dict[int, float] = {}
        for k in k_values:
            recall_at_k[k] = (
                sum(q[f"recall_{k}"] for q in per_query_result.values()) / num_queries
            )
            ndcg_at_k[k] = (
                sum(q[f"ndcg_cut_{k}"] for q in per_query_result.values()) / num_queries
            )
    except Exception as exc:
        if isinstance(exc, EvaluationError):
            raise
        category, detail = _classify_metric_dependency_error(exc)
        raise EvaluationError(f"recall_ndcg: {category}: {detail}") from exc

    return recall_at_k, ndcg_at_k


# ---------------------------------------------------------------------------
# Uniform registry-integration interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalMetricComputation:
    """The one uniform return shape every RetrievalMetricWrapper.compute()
    produces -- mirrors MetricComputation's role for generation metrics
    (Cell 31), adapted to the retrieval domain's per-k (not per-example)
    granularity."""

    metric_name: str
    scores_at_k: Tuple[Tuple[int, float], ...]  # sorted by k, immutable (a plain dict is not frozen-safe)


def _scores_at_k_from_dict(scores: Dict[int, float]) -> Tuple[Tuple[int, float], ...]:
    return tuple(sorted(scores.items()))


class RetrievalMetricWrapper(abc.ABC):
    """One uniform API for every retrieval metric: compute(positives, ranked)
    -> RetrievalMetricComputation, nothing else. k_values (and, for
    RecallAtKMetric/NdcgAtKMetric, any injected pytrec_eval_factory) are
    bound once at construction via RetrievalMetricsConfig, never passed
    per call -- the same discipline already established for
    GenerationMetricWrapper (Cell 31)."""

    name: str

    @abc.abstractmethod
    def compute(
        self,
        positives: Dict[QueryKey, Sequence[QueryKey]],
        ranked: Dict[QueryKey, Sequence[QueryKey]],
    ) -> RetrievalMetricComputation:
        ...


class MRRMetric(RetrievalMetricWrapper):
    name = "mrr"

    def __init__(self, config: RetrievalMetricsConfig) -> None:
        self._config = config

    def compute(
        self,
        positives: Dict[QueryKey, Sequence[QueryKey]],
        ranked: Dict[QueryKey, Sequence[QueryKey]],
    ) -> RetrievalMetricComputation:
        scores = {k: compute_mrr_at_k(positives, ranked, k) for k in self._config.k_values}
        return RetrievalMetricComputation(
            metric_name=self.name, scores_at_k=_scores_at_k_from_dict(scores)
        )


class RecallAtKMetric(RetrievalMetricWrapper):
    """Calls compute_recall_ndcg_at_k and keeps only the recall half.
    NdcgAtKMetric (below) independently calls the same function for its
    own half -- a deliberate, disclosed trade-off: pytrec_eval's
    underlying computation runs twice if both registry entries are used
    together, in exchange for each registry entry staying genuinely
    independent (no shared hidden state between them). A future
    orchestrator that wants to avoid the duplicate call can call
    compute_recall_ndcg_at_k directly instead of through the registry."""

    name = "recall"

    def __init__(
        self, config: RetrievalMetricsConfig, *, pytrec_eval_factory: Optional[Callable[[dict, set], object]] = None
    ) -> None:
        self._config = config
        self._pytrec_eval_factory = pytrec_eval_factory

    def compute(
        self,
        positives: Dict[QueryKey, Sequence[QueryKey]],
        ranked: Dict[QueryKey, Sequence[QueryKey]],
    ) -> RetrievalMetricComputation:
        recall_at_k, _ = compute_recall_ndcg_at_k(
            positives, ranked, self._config.k_values, pytrec_eval_factory=self._pytrec_eval_factory
        )
        return RetrievalMetricComputation(
            metric_name=self.name, scores_at_k=_scores_at_k_from_dict(recall_at_k)
        )


class NdcgAtKMetric(RetrievalMetricWrapper):
    """See RecallAtKMetric's own docstring for the shared-computation
    trade-off note."""

    name = "ndcg"

    def __init__(
        self, config: RetrievalMetricsConfig, *, pytrec_eval_factory: Optional[Callable[[dict, set], object]] = None
    ) -> None:
        self._config = config
        self._pytrec_eval_factory = pytrec_eval_factory

    def compute(
        self,
        positives: Dict[QueryKey, Sequence[QueryKey]],
        ranked: Dict[QueryKey, Sequence[QueryKey]],
    ) -> RetrievalMetricComputation:
        _, ndcg_at_k = compute_recall_ndcg_at_k(
            positives, ranked, self._config.k_values, pytrec_eval_factory=self._pytrec_eval_factory
        )
        return RetrievalMetricComputation(
            metric_name=self.name, scores_at_k=_scores_at_k_from_dict(ndcg_at_k)
        )


def _hit_rate_score_fn(
    positives: Dict[QueryKey, Sequence[QueryKey]], ranked: Dict[QueryKey, Sequence[QueryKey]], k: int
) -> float:
    """Deterministic, no-library mock score: fraction of queries with at
    least one positive appearing in ranked[:k]. Pure function of the
    inputs -- no randomness anywhere."""
    hits = 0
    for query_key, positive_seq in positives.items():
        positive_set = set(positive_seq)
        if any(candidate in positive_set for candidate in ranked[query_key][:k]):
            hits += 1
    return hits / len(positives)


class MockRetrievalMetricWrapper(RetrievalMetricWrapper):
    """Deterministic, no-library fake conforming to the exact same
    RetrievalMetricWrapper interface as MRRMetric/RecallAtKMetric/
    NdcgAtKMetric -- for unit-testing any code that consumes a retrieval
    metric (or the registry) generically, without pytrec_eval at all.
    Scores a pure hit-rate@k by default (see _hit_rate_score_fn); a
    custom score_fn(positives, ranked, k) -> float may be injected for
    tests that need specific, controlled values instead."""

    def __init__(
        self,
        config: RetrievalMetricsConfig,
        name: str = "mock_retrieval_metric",
        *,
        score_fn: Optional[Callable[[Dict, Dict, int], float]] = None,
    ) -> None:
        self._config = config
        self.name = name
        self._score_fn = score_fn or _hit_rate_score_fn

    def compute(
        self,
        positives: Dict[QueryKey, Sequence[QueryKey]],
        ranked: Dict[QueryKey, Sequence[QueryKey]],
    ) -> RetrievalMetricComputation:
        _validate_retrieval_inputs(positives, ranked, metric_name=self.name)
        scores = {k: self._score_fn(positives, ranked, k) for k in self._config.k_values}
        return RetrievalMetricComputation(
            metric_name=self.name, scores_at_k=_scores_at_k_from_dict(scores)
        )


# ---------------------------------------------------------------------------
# Registry -- explicit, freshly-constructed, never a shared mutable
# module-level dict (same discipline as build_metric_registry, Cell 31).
# ---------------------------------------------------------------------------

_REGISTERED_RETRIEVAL_METRIC_NAMES: Tuple[str, ...] = ("mrr", "recall", "ndcg")


def build_retrieval_metric_registry(
    config: RetrievalMetricsConfig,
    *,
    pytrec_eval_factory: Optional[Callable[[dict, set], object]] = None,
) -> Dict[str, RetrievalMetricWrapper]:
    """Builds the real (lazy, for recall/ndcg) retrieval-metric registry:
    mrr, recall, ndcg -- a separate registry from
    generation_metrics.build_metric_registry (different domain, different
    compute() signature: (positives, ranked), not (hyps, refs))."""
    return {
        "mrr": MRRMetric(config),
        "recall": RecallAtKMetric(config, pytrec_eval_factory=pytrec_eval_factory),
        "ndcg": NdcgAtKMetric(config, pytrec_eval_factory=pytrec_eval_factory),
    }


def build_mock_retrieval_metric_registry(
    config: RetrievalMetricsConfig,
    names: Sequence[str] = _REGISTERED_RETRIEVAL_METRIC_NAMES,
) -> Dict[str, RetrievalMetricWrapper]:
    """Builds an all-mock registry with the same key shape as
    build_retrieval_metric_registry's default output -- for exercising
    any registry-consuming code without pytrec_eval installed at all."""
    return {name: MockRetrievalMetricWrapper(config, name=name) for name in names}
