"""Unit tests for src/evaluation/retrieval_metrics.py (Milestone 2.6, Cell 33).

Scope: RetrievalMetricsConfig, _validate_retrieval_inputs (strict
query-key/ranking/duplicate validation), compute_mrr_at_k (the preserved
official hand-written algorithm -- cross-checked directly against a
literal port of official evaluate_retriever.py::compute_mrr),
compute_recall_ndcg_at_k (delegated to pytrec_eval, exercised only via
injected fakes), the uniform RetrievalMetricWrapper interface, the mock
wrapper, and registry integration."""

import ast
import dataclasses
import inspect

import pytest

import src.evaluation.retrieval_metrics as rm
from src.evaluation.retrieval_metrics import (
    MockRetrievalMetricWrapper,
    MRRMetric,
    NdcgAtKMetric,
    RecallAtKMetric,
    RetrievalMetricComputation,
    RetrievalMetricsConfig,
    RetrievalMetricWrapper,
    build_mock_retrieval_metric_registry,
    build_retrieval_metric_registry,
    compute_mrr_at_k,
    compute_recall_ndcg_at_k,
)
from src.common.exceptions import EvaluationError


def qk(name: str):
    return ("mimic-cxr", name, name)


# ---------------------------------------------------------------------------
# Literal port of official evaluate_retriever.py::compute_mrr (contract
# §3.2), operating on the ORIGINAL integer-pid/list shape, used purely
# to cross-check compute_mrr_at_k's output against the unmodified
# official algorithm -- this is the strongest possible evidence that
# the official implementation was preserved, not silently altered.
# ---------------------------------------------------------------------------


def _official_compute_mrr(qids_to_relevant_passageids, qids_to_ranked_candidate_passages, MaxMRRRank):
    MRR = 0
    for qid in qids_to_ranked_candidate_passages:
        if qid in qids_to_relevant_passageids:
            target_pid = qids_to_relevant_passageids[qid]
            candidate_pid = qids_to_ranked_candidate_passages[qid]
            for i in range(0, MaxMRRRank):
                if i >= len(candidate_pid):
                    break  # official code relies on 1000-slot zero-padding instead; a
                    # bounds check here is the equivalent, not a behavior change
                if candidate_pid[i] in target_pid:
                    MRR += 1 / (i + 1)
                    break
    return MRR / len(qids_to_relevant_passageids)


# ---------------------------------------------------------------------------
# Module-scope import discipline
# ---------------------------------------------------------------------------


def test_module_never_imports_pytrec_eval_at_module_scope():
    source = inspect.getsource(rm)
    tree = ast.parse(source)
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "pytrec_eval" not in top_level_imports


# ---------------------------------------------------------------------------
# RetrievalMetricsConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"k_values": ()},
        {"k_values": (100, 0)},
        {"k_values": (100, -5)},
        {"k_values": (100, 1.5)},
        {"k_values": (100, True)},
        {"k_values": (100, 100)},
        {"k_values": (1000, 100)},  # not strictly increasing
        {"chexbert_threshold": -0.1},
        {"radgraph_threshold": -0.1},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        RetrievalMetricsConfig(**overrides)


def test_config_defaults_are_valid():
    config = RetrievalMetricsConfig()
    assert config.k_values == (100, 200, 500, 1000)
    assert config.chexbert_threshold == 1.0
    assert config.radgraph_threshold == 0.4


def test_config_is_immutable():
    config = RetrievalMetricsConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.k_values = (1,)


def test_config_as_actual_used_dict_round_trips_all_fields():
    config = RetrievalMetricsConfig()
    actual = config.as_actual_used_dict()
    assert actual["k_values"] == [100, 200, 500, 1000]
    assert set(actual) == {f.name for f in dataclasses.fields(RetrievalMetricsConfig)}


# ---------------------------------------------------------------------------
# _validate_retrieval_inputs (exercised through compute_mrr_at_k)
# ---------------------------------------------------------------------------


def test_rejects_empty_positives():
    with pytest.raises(EvaluationError, match="positives must be non-empty"):
        compute_mrr_at_k({}, {}, 100)


def test_rejects_malformed_query_key_in_positives():
    with pytest.raises(EvaluationError, match="malformed query_key in positives"):
        compute_mrr_at_k({"not-a-tuple": (qk("p1"),)}, {qk("q1"): (qk("p1"),)}, 100)


def test_rejects_malformed_query_key_wrong_length():
    with pytest.raises(EvaluationError, match="malformed query_key"):
        compute_mrr_at_k({("a", "b"): (qk("p1"),)}, {("a", "b"): (qk("p1"),)}, 100)


def test_rejects_malformed_query_key_empty_part():
    with pytest.raises(EvaluationError, match="malformed query_key"):
        compute_mrr_at_k({("a", "", "c"): (qk("p1"),)}, {("a", "", "c"): (qk("p1"),)}, 100)


def test_rejects_positives_ranked_key_set_mismatch_missing_from_ranked():
    positives = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="exact same query"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_positives_ranked_key_set_mismatch_extra_in_ranked():
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="exact same query"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_empty_positive_sequence():
    positives = {qk("q1"): ()}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="must be non-empty"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_malformed_candidate_in_positives_sequence():
    positives = {qk("q1"): (("bad",),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="malformed candidate key in positives"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_malformed_candidate_in_ranked_sequence():
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (("bad",),)}
    with pytest.raises(EvaluationError, match="malformed candidate key in ranked"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_duplicate_keys_in_positives_sequence():
    positives = {qk("q1"): (qk("p1"), qk("p1"))}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="positives\\[.*\\] contains duplicate"):
        compute_mrr_at_k(positives, ranked, 100)


def test_rejects_duplicate_retrieved_keys_in_ranked_sequence():
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p1"))}
    with pytest.raises(EvaluationError, match="ranked\\[.*\\] contains duplicate retrieved"):
        compute_mrr_at_k(positives, ranked, 100)


# ---------------------------------------------------------------------------
# compute_mrr_at_k -- hand-computed correctness
# ---------------------------------------------------------------------------


def test_mrr_perfect_match_at_rank_zero_is_one():
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"))}
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(1.0)


def test_mrr_match_at_rank_two_gives_one_third():
    positives = {qk("q1"): (qk("p3"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(1 / 3)


def test_mrr_no_match_in_ranking_contributes_zero():
    positives = {qk("q1"): (qk("p9"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"))}
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(0.0)


def test_mrr_match_beyond_k_is_not_counted():
    positives = {qk("q1"): (qk("p3"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}
    assert compute_mrr_at_k(positives, ranked, 2) == pytest.approx(0.0)  # p3 is at index 2, outside top-2


def test_mrr_matches_first_of_multiple_positives():
    positives = {qk("q1"): (qk("p2"), qk("p3"))}
    ranked = {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}
    # first hit in ranking order is p2 at index 1 -> reciprocal rank 1/2
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(0.5)


def test_mrr_averages_across_multiple_queries():
    positives = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p9"),)}
    ranked = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p1"),)}
    # q1: hit at rank 0 -> 1.0; q2: no hit -> 0.0; mean = 0.5
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(0.5)


def test_mrr_is_computed_independently_per_k_not_derived():
    positives = {qk("q1"): (qk("p3"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}
    assert compute_mrr_at_k(positives, ranked, 1) == pytest.approx(0.0)
    assert compute_mrr_at_k(positives, ranked, 2) == pytest.approx(0.0)
    assert compute_mrr_at_k(positives, ranked, 3) == pytest.approx(1 / 3)


def test_mrr_short_ranking_below_k_behaves_like_no_match():
    positives = {qk("q1"): (qk("p9"),)}
    ranked = {qk("q1"): (qk("p1"),)}  # only 1 candidate, k=100 requested
    assert compute_mrr_at_k(positives, ranked, 100) == pytest.approx(0.0)


@pytest.mark.parametrize("bad_k", [0, -1, 1.5, True])
def test_mrr_rejects_invalid_k(bad_k):
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="k must be a positive integer"):
        compute_mrr_at_k(positives, ranked, bad_k)


def test_mrr_is_deterministic_across_repeated_calls():
    positives = {qk("q1"): (qk("p2"),), qk("q2"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2")), qk("q2"): (qk("p1"),)}
    first = compute_mrr_at_k(positives, ranked, 100)
    second = compute_mrr_at_k(positives, ranked, 100)
    assert first == second


# ---------------------------------------------------------------------------
# compute_mrr_at_k cross-checked against a literal port of the official
# algorithm -- the central evidence for "preserve the official
# hand-written implementation, do not silently replace it."
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "positives, ranked, k",
    [
        ({qk("q1"): (qk("p1"),)}, {qk("q1"): (qk("p1"), qk("p2"))}, 100),
        ({qk("q1"): (qk("p3"),)}, {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}, 100),
        ({qk("q1"): (qk("p3"),)}, {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}, 2),
        (
            {qk("q1"): (qk("p1"),), qk("q2"): (qk("p9"),), qk("q3"): (qk("p2"), qk("p5"))},
            {
                qk("q1"): (qk("p1"), qk("p2")),
                qk("q2"): (qk("p1"), qk("p2")),
                qk("q3"): (qk("p1"), qk("p5"), qk("p2")),
            },
            100,
        ),
        (
            {qk("q1"): (qk("p2"), qk("p3"))},
            {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))},
            1,
        ),
    ],
)
def test_compute_mrr_at_k_matches_literal_official_port(positives, ranked, k):
    ours = compute_mrr_at_k(positives, ranked, k)

    official_positives = {key: list(val) for key, val in positives.items()}
    official_ranked = {key: list(val) for key, val in ranked.items()}
    official = _official_compute_mrr(official_positives, official_ranked, k)

    assert ours == pytest.approx(official)


# ---------------------------------------------------------------------------
# compute_recall_ndcg_at_k -- delegated to pytrec_eval, injected fakes only
# ---------------------------------------------------------------------------


class _FakePytrecEvaluator:
    def __init__(self, per_query_result):
        self._per_query_result = per_query_result
        self.evaluate_calls = []
        self.constructed_with = None

    def evaluate(self, run):
        self.evaluate_calls.append(run)
        return self._per_query_result


class _RaisingPytrecEvaluator:
    def __init__(self, exc):
        self._exc = exc

    def evaluate(self, run):
        raise self._exc


def _make_factory(evaluator, capture=None):
    def factory(qrel, measures):
        if capture is not None:
            capture["qrel"] = qrel
            capture["measures"] = measures
        return evaluator

    return factory


def test_recall_ndcg_computes_means_across_queries():
    per_query_result = {
        "0": {"recall_100": 0.5, "ndcg_cut_100": 0.6},
        "1": {"recall_100": 1.0, "ndcg_cut_100": 0.8},
    }
    fake = _FakePytrecEvaluator(per_query_result)
    positives = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p1"),)}

    recall_at_k, ndcg_at_k = compute_recall_ndcg_at_k(
        positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake)
    )
    assert recall_at_k == {100: pytest.approx(0.75)}
    assert ndcg_at_k == {100: pytest.approx(0.7)}


def test_recall_ndcg_passes_bare_family_names_as_measures():
    per_query_result = {"0": {"recall_100": 1.0, "ndcg_cut_100": 1.0}}
    fake = _FakePytrecEvaluator(per_query_result)
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    capture = {}

    compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake, capture))
    assert capture["measures"] == {"ndcg_cut", "recall"}


def test_recall_ndcg_qrel_and_run_shape():
    per_query_result = {"0": {"recall_100": 1.0, "ndcg_cut_100": 1.0}}
    fake = _FakePytrecEvaluator(per_query_result)
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p2"), qk("p1"))}  # p1 at rank index 1
    capture = {}

    compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake, capture))

    qrel = capture["qrel"]
    assert len(qrel) == 1
    (query_id, positive_dict), = qrel.items()
    assert positive_dict == {list(positive_dict)[0]: 1}

    # run is passed to evaluate(), not the factory -- inspect it directly
    run = fake.evaluate_calls[0]
    assert len(run) == 1
    (run_qid, candidate_scores), = run.items()
    assert run_qid == query_id
    # p1 is ranked at index 1 -> score -(1+1) = -2; p2 is at index 0 -> score -1
    assert sorted(candidate_scores.values()) == [-2, -1]


def test_recall_ndcg_multiple_k_values_computed_independently():
    per_query_result = {"0": {"recall_100": 0.4, "ndcg_cut_100": 0.5, "recall_200": 0.9, "ndcg_cut_200": 0.95}}
    fake = _FakePytrecEvaluator(per_query_result)
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}

    recall_at_k, ndcg_at_k = compute_recall_ndcg_at_k(
        positives, ranked, (100, 200), pytrec_eval_factory=_make_factory(fake)
    )
    assert recall_at_k == {100: pytest.approx(0.4), 200: pytest.approx(0.9)}
    assert ndcg_at_k == {100: pytest.approx(0.5), 200: pytest.approx(0.95)}


def test_recall_ndcg_rejects_empty_k_values():
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="k_values must be non-empty"):
        compute_recall_ndcg_at_k(positives, ranked, (), pytrec_eval_factory=lambda q, m: None)


@pytest.mark.parametrize("bad_k", [0, -5, 1.5])
def test_recall_ndcg_rejects_invalid_k_in_k_values(bad_k):
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="k must be a positive integer"):
        compute_recall_ndcg_at_k(positives, ranked, (bad_k,), pytrec_eval_factory=lambda q, m: None)


def test_recall_ndcg_construction_failure_is_classified_and_wrapped():
    def raising_factory(qrel, measures):
        raise ImportError("no module named pytrec_eval")

    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="missing_dependency"):
        compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=raising_factory)


def test_recall_ndcg_evaluate_failure_is_classified_and_wrapped():
    fake = _RaisingPytrecEvaluator(RuntimeError("internal trec_eval error"))
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="computation_failed"):
        compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake))


def test_recall_ndcg_rejects_empty_per_query_result():
    fake = _FakePytrecEvaluator({})
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError):
        compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake))


def test_recall_ndcg_missing_measure_key_is_classified_and_wrapped():
    per_query_result = {"0": {"recall_100": 1.0}}  # missing ndcg_cut_100
    fake = _FakePytrecEvaluator(per_query_result)
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    with pytest.raises(EvaluationError, match="invalid_input"):
        compute_recall_ndcg_at_k(positives, ranked, (100,), pytrec_eval_factory=_make_factory(fake))


def test_default_pytrec_eval_factory_lazily_imports_and_constructs(monkeypatch):
    import sys

    constructed_args = {}

    class _FakePytrecEvalModule:
        class RelevanceEvaluator:
            def __init__(self, qrel, measures):
                constructed_args["qrel"] = qrel
                constructed_args["measures"] = measures

    monkeypatch.setitem(sys.modules, "pytrec_eval", _FakePytrecEvalModule())

    rm._default_pytrec_eval_factory({"0": {"1": 1}}, {"recall", "ndcg_cut"})
    assert constructed_args["qrel"] == {"0": {"1": 1}}
    assert constructed_args["measures"] == {"recall", "ndcg_cut"}


# ---------------------------------------------------------------------------
# _build_query_key_index
# ---------------------------------------------------------------------------


def test_build_query_key_index_covers_all_keys_and_is_injective():
    positives = {qk("q1"): (qk("p1"), qk("p2"))}
    ranked = {qk("q1"): (qk("p2"), qk("p3"))}
    index = rm._build_query_key_index(positives, ranked)
    expected_keys = {qk("q1"), qk("p1"), qk("p2"), qk("p3")}
    assert set(index) == expected_keys
    assert len(set(index.values())) == len(index)  # injective: no two keys share an id


def test_build_query_key_index_is_deterministic():
    positives = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p2"),)}
    ranked = {qk("q1"): (qk("p1"),), qk("q2"): (qk("p2"),)}
    first = rm._build_query_key_index(positives, ranked)
    second = rm._build_query_key_index(positives, ranked)
    assert first == second


# ---------------------------------------------------------------------------
# Uniform RetrievalMetricWrapper interface
# ---------------------------------------------------------------------------


def test_retrieval_metric_wrapper_is_abstract():
    with pytest.raises(TypeError):
        RetrievalMetricWrapper()


def test_retrieval_metric_computation_is_immutable():
    computation = RetrievalMetricComputation(metric_name="mrr", scores_at_k=((100, 0.5),))
    with pytest.raises(dataclasses.FrozenInstanceError):
        computation.metric_name = "other"


def test_all_four_retrieval_wrappers_share_identical_compute_signature():
    for cls in (MRRMetric, RecallAtKMetric, NdcgAtKMetric, MockRetrievalMetricWrapper):
        sig = inspect.signature(cls.compute)
        assert list(sig.parameters) == ["self", "positives", "ranked"]
        assert issubclass(cls, RetrievalMetricWrapper)


def test_mrr_metric_wrapper_computes_scores_at_every_configured_k():
    config = RetrievalMetricsConfig(k_values=(1, 3))
    metric = MRRMetric(config)
    positives = {qk("q1"): (qk("p3"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2"), qk("p3"))}
    result = metric.compute(positives, ranked)
    assert result.metric_name == "mrr"
    assert dict(result.scores_at_k) == {1: pytest.approx(0.0), 3: pytest.approx(1 / 3)}


def test_recall_and_ndcg_metric_wrappers_use_injected_factory():
    per_query_result = {"0": {"recall_100": 0.5, "ndcg_cut_100": 0.6}}
    fake = _FakePytrecEvaluator(per_query_result)
    config = RetrievalMetricsConfig(k_values=(100,))
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}

    recall_metric = RecallAtKMetric(config, pytrec_eval_factory=_make_factory(fake))
    ndcg_metric = NdcgAtKMetric(config, pytrec_eval_factory=_make_factory(fake))

    recall_result = recall_metric.compute(positives, ranked)
    ndcg_result = ndcg_metric.compute(positives, ranked)

    assert recall_result.metric_name == "recall"
    assert dict(recall_result.scores_at_k) == {100: pytest.approx(0.5)}
    assert ndcg_result.metric_name == "ndcg"
    assert dict(ndcg_result.scores_at_k) == {100: pytest.approx(0.6)}


# ---------------------------------------------------------------------------
# MockRetrievalMetricWrapper
# ---------------------------------------------------------------------------


def test_mock_retrieval_metric_hit_rate_default():
    config = RetrievalMetricsConfig(k_values=(1, 2))
    mock = MockRetrievalMetricWrapper(config, name="mock")
    positives = {qk("q1"): (qk("p2"),), qk("q2"): (qk("p9"),)}
    ranked = {qk("q1"): (qk("p1"), qk("p2")), qk("q2"): (qk("p1"), qk("p2"))}
    result = mock.compute(positives, ranked)
    # k=1: only p1 is in top-1 for both queries -> neither hits -> 0.0
    # k=2: q1 hits (p2 present in top-2), q2 never hits -> 0.5
    assert dict(result.scores_at_k) == {1: pytest.approx(0.0), 2: pytest.approx(0.5)}


def test_mock_retrieval_metric_is_deterministic():
    config = RetrievalMetricsConfig(k_values=(100,))
    mock = MockRetrievalMetricWrapper(config, name="mock")
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    first = mock.compute(positives, ranked)
    second = mock.compute(positives, ranked)
    assert first == second


def test_mock_retrieval_metric_accepts_custom_score_fn():
    config = RetrievalMetricsConfig(k_values=(100,))
    mock = MockRetrievalMetricWrapper(config, name="custom", score_fn=lambda p, r, k: 0.42)
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    result = mock.compute(positives, ranked)
    assert dict(result.scores_at_k) == {100: pytest.approx(0.42)}


def test_mock_retrieval_metric_validates_inputs_too():
    config = RetrievalMetricsConfig(k_values=(100,))
    mock = MockRetrievalMetricWrapper(config)
    with pytest.raises(EvaluationError):
        mock.compute({}, {})


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_build_retrieval_metric_registry_has_all_three_metrics():
    registry = build_retrieval_metric_registry(RetrievalMetricsConfig())
    assert set(registry) == {"mrr", "recall", "ndcg"}
    assert isinstance(registry["mrr"], MRRMetric)
    assert isinstance(registry["recall"], RecallAtKMetric)
    assert isinstance(registry["ndcg"], NdcgAtKMetric)


def test_build_retrieval_metric_registry_returns_independent_instances_across_calls():
    config = RetrievalMetricsConfig()
    registry_one = build_retrieval_metric_registry(config)
    registry_two = build_retrieval_metric_registry(config)
    assert registry_one["mrr"] is not registry_two["mrr"]
    assert registry_one["recall"] is not registry_two["recall"]


def test_build_retrieval_metric_registry_propagates_injected_factory():
    per_query_result = {"0": {"recall_100": 1.0, "ndcg_cut_100": 1.0}}
    fake = _FakePytrecEvaluator(per_query_result)
    registry = build_retrieval_metric_registry(
        RetrievalMetricsConfig(k_values=(100,)), pytrec_eval_factory=_make_factory(fake)
    )
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    result = registry["recall"].compute(positives, ranked)
    assert dict(result.scores_at_k) == {100: pytest.approx(1.0)}


def test_build_mock_retrieval_metric_registry_covers_all_three_metrics():
    mock_registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig())
    assert set(mock_registry) == {"mrr", "recall", "ndcg"}
    for name, wrapper in mock_registry.items():
        assert isinstance(wrapper, MockRetrievalMetricWrapper)
        assert wrapper.name == name


def test_build_mock_retrieval_metric_registry_is_usable_without_pytrec_eval():
    registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(k_values=(100,)))
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}
    for wrapper in registry.values():
        result = wrapper.compute(positives, ranked)
        assert dict(result.scores_at_k) == {100: pytest.approx(1.0)}


def test_build_mock_retrieval_metric_registry_accepts_custom_names():
    registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(), names=("only_this",))
    assert set(registry) == {"only_this"}
