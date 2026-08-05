"""Unit tests for src/evaluation/report.py (Milestone 2.6, Cell 34).

Scope: EvaluationRunConfig, split_generation_and_clinical_registries,
EvaluationRunner (deterministic ordering, per-metric failure handling,
per-sample row construction, sample-level failure propagation), and the
atomic JSON/JSONL report writers. Every metric is exercised only via
mock registries (Cell 30/31/32/33's own MockGenerationMetricWrapper /
MockRetrievalMetricWrapper, or small local fakes) -- no real metric
library is imported or invoked anywhere in this file."""

import dataclasses
import json

import pytest

from src.common.exceptions import EvaluationError
from src.evaluation.generation_metrics import (
    GenerationMetricsConfig,
    build_mock_metric_registry,
    load_generation_pairs,
)
from src.evaluation.generation_metrics import GenerationMetricWrapper, MetricComputation
from src.evaluation.retrieval_metrics import (
    RetrievalMetricsConfig,
    RetrievalMetricWrapper,
    RetrievalMetricComputation,
    build_mock_retrieval_metric_registry,
)
from src.evaluation.report import (
    EvaluationRunConfig,
    EvaluationRunner,
    EvaluationRunResult,
    MetricFailure,
    PerSampleResultRow,
    split_generation_and_clinical_registries,
    write_evaluation_summary_json_atomic,
    write_per_sample_rows_jsonl_atomic,
)


def qk(name: str):
    return ("mimic-cxr", name, name)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _ref_row(name, finding):
    return {"query_key": list(qk(name)), "finding": finding}


def _pred_row(name, retrieved_finding=None, error=None):
    obj = {"query_key": list(qk(name))}
    if error is not None:
        obj["error"] = error
    else:
        obj["retrieved_finding"] = [retrieved_finding]
    return obj


def _make_generation_load_result(tmp_path, ref_rows, pred_rows, config=None):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, ref_rows)
    _write_jsonl(pred_path, pred_rows)
    return load_generation_pairs(ref_path, pred_path, config or GenerationMetricsConfig())


# ---------------------------------------------------------------------------
# Small local fakes (order tracking / failure injection / no-per-example)
# ---------------------------------------------------------------------------


class _OrderTrackingGenerationMetric(GenerationMetricWrapper):
    def __init__(self, name, execution_log):
        self.name = name
        self._log = execution_log

    def compute(self, hyps, refs):
        self._log.append(self.name)
        return MetricComputation(
            metric_name=self.name, corpus_score=1.0, per_example_scores=tuple(1.0 for _ in hyps)
        )


class _RaisingGenerationMetric(GenerationMetricWrapper):
    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    def compute(self, hyps, refs):
        raise self._exc


class _NoPerExampleGenerationMetric(GenerationMetricWrapper):
    """Mirrors Bleu4Metric/DatasetF1ChexbertMetric's own real shape:
    corpus-level only, per_example_scores always None."""

    def __init__(self, name, corpus_score):
        self.name = name
        self._corpus_score = corpus_score

    def compute(self, hyps, refs):
        return MetricComputation(metric_name=self.name, corpus_score=self._corpus_score, per_example_scores=None)


class _OrderTrackingRetrievalMetric(RetrievalMetricWrapper):
    def __init__(self, name, execution_log):
        self.name = name
        self._log = execution_log

    def compute(self, positives, ranked):
        self._log.append(self.name)
        return RetrievalMetricComputation(metric_name=self.name, scores_at_k=((100, 1.0),))


class _RaisingRetrievalMetric(RetrievalMetricWrapper):
    def __init__(self, name, exc):
        self.name = name
        self._exc = exc

    def compute(self, positives, ranked):
        raise self._exc


# ---------------------------------------------------------------------------
# EvaluationRunConfig
# ---------------------------------------------------------------------------


def test_evaluation_run_config_field_order_is_valid_python():
    # The mere existence of this dataclass without a TypeError at import
    # time is itself evidence the field-ordering fix (module docstring)
    # is correct -- this test just makes the check explicit.
    config = EvaluationRunConfig(run_id="r1", system_name="FactMM-RAG", dataset_name="mimic-cxr")
    assert config.config_version == "1.0"
    assert config.split == "test"
    assert config.seed == 42


@pytest.mark.parametrize(
    "overrides",
    [
        {"run_id": ""},
        {"system_name": ""},
        {"dataset_name": "not-a-real-dataset"},
        {"config_version": ""},
        {"split": "not-a-real-split"},
        {"seed": 1.5},
        {"seed": True},
    ],
)
def test_evaluation_run_config_rejects_invalid_values(overrides):
    base = {"run_id": "r1", "system_name": "FactMM-RAG", "dataset_name": "mimic-cxr"}
    base.update(overrides)
    with pytest.raises(ValueError):
        EvaluationRunConfig(**base)


def test_evaluation_run_config_is_immutable():
    config = EvaluationRunConfig(run_id="r1", system_name="FactMM-RAG", dataset_name="mimic-cxr")
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.seed = 7


# ---------------------------------------------------------------------------
# split_generation_and_clinical_registries
# ---------------------------------------------------------------------------


def test_split_registries_matches_established_classification():
    combined = build_mock_metric_registry()
    generation_registry, clinical_registry = split_generation_and_clinical_registries(combined)
    assert set(generation_registry) == {"rouge_l", "bleu4", "bert_score"}
    assert set(clinical_registry) == {"f1radgraph", "f1chexbert", "f1chexbert_instance"}
    assert set(generation_registry) | set(clinical_registry) == set(combined)


def test_split_registries_is_a_pure_partition_not_a_reconstruction():
    combined = build_mock_metric_registry()
    generation_registry, clinical_registry = split_generation_and_clinical_registries(combined)
    assert generation_registry["rouge_l"] is combined["rouge_l"]
    assert clinical_registry["f1radgraph"] is combined["f1radgraph"]


def test_split_registries_handles_empty_input():
    generation_registry, clinical_registry = split_generation_and_clinical_registries({})
    assert generation_registry == {}
    assert clinical_registry == {}


# ---------------------------------------------------------------------------
# EvaluationRunner construction
# ---------------------------------------------------------------------------


def test_runner_rejects_overlapping_generation_and_clinical_registry_names():
    execution_log = []
    shared_metric = _OrderTrackingGenerationMetric("shared", execution_log)
    with pytest.raises(EvaluationError, match="both generation_metric_registry and clinical_metric_registry"):
        EvaluationRunner(
            generation_metric_registry={"shared": shared_metric},
            clinical_metric_registry={"shared": shared_metric},
            retrieval_metric_registry={},
        )


def test_runner_construction_succeeds_with_disjoint_registries():
    EvaluationRunner(
        generation_metric_registry=split_generation_and_clinical_registries(build_mock_metric_registry())[0],
        clinical_metric_registry=split_generation_and_clinical_registries(build_mock_metric_registry())[1],
        retrieval_metric_registry=build_mock_retrieval_metric_registry(RetrievalMetricsConfig()),
    )  # must not raise


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- input validation
# ---------------------------------------------------------------------------


def _make_runner(generation_registry=None, clinical_registry=None, retrieval_registry=None):
    return EvaluationRunner(
        generation_metric_registry=generation_registry or {},
        clinical_metric_registry=clinical_registry or {},
        retrieval_metric_registry=retrieval_registry or {},
    )


def test_run_rejects_no_inputs_at_all():
    runner = _make_runner()
    with pytest.raises(EvaluationError, match="at least one of"):
        runner.run(EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"))


def test_run_rejects_partial_retrieval_inputs():
    runner = _make_runner()
    with pytest.raises(EvaluationError, match="both be provided or both omitted"):
        runner.run(
            EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
            retrieval_positives={qk("q1"): (qk("p1"),)},
        )


def test_run_rejects_generation_load_result_from_positional_mode(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    ref_path.write_text(json.dumps({"finding": "a"}) + "\n")
    pred_path.write_text(json.dumps({"retrieved_finding": ["a"]}) + "\n")
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    load_result = load_generation_pairs(ref_path, pred_path, config)

    runner = _make_runner()
    with pytest.raises(EvaluationError, match="strict query_key-based pairing"):
        runner.run(
            EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
            generation_load_result=load_result,
        )


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- deterministic ordering
# ---------------------------------------------------------------------------


def test_run_executes_generation_and_clinical_metrics_in_sorted_order(tmp_path):
    execution_log = []
    generation_registry = {
        "zeta": _OrderTrackingGenerationMetric("zeta", execution_log),
        "alpha": _OrderTrackingGenerationMetric("alpha", execution_log),
    }
    clinical_registry = {
        "yankee": _OrderTrackingGenerationMetric("yankee", execution_log),
        "bravo": _OrderTrackingGenerationMetric("bravo", execution_log),
    }
    runner = _make_runner(generation_registry, clinical_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref text")], [_pred_row("q1", "hyp text")]
    )
    runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert execution_log == ["alpha", "zeta", "bravo", "yankee"]


def test_run_executes_retrieval_metrics_in_sorted_order():
    execution_log = []
    retrieval_registry = {
        "zeta": _OrderTrackingRetrievalMetric("zeta", execution_log),
        "alpha": _OrderTrackingRetrievalMetric("alpha", execution_log),
    }
    runner = _make_runner(retrieval_registry=retrieval_registry)
    runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    assert execution_log == ["alpha", "zeta"]


def test_run_is_deterministic_across_repeated_calls(tmp_path):
    generation_registry, clinical_registry = split_generation_and_clinical_registries(
        build_mock_metric_registry()
    )
    retrieval_registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(k_values=(100,)))
    runner = _make_runner(generation_registry, clinical_registry, retrieval_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref text")], [_pred_row("q1", "hyp text")]
    )
    run_config = EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr")
    positives = {qk("q1"): (qk("p1"),)}
    ranked = {qk("q1"): (qk("p1"),)}

    first = runner.run(run_config, generation_load_result=load_result, retrieval_positives=positives, retrieval_ranked=ranked)
    second = runner.run(run_config, generation_load_result=load_result, retrieval_positives=positives, retrieval_ranked=ranked)

    assert first.generation_corpus_scores == second.generation_corpus_scores
    assert first.clinical_corpus_scores == second.clinical_corpus_scores
    assert first.retrieval_scores == second.retrieval_scores
    assert first.per_sample_rows == second.per_sample_rows


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- generation/retrieval-only runs
# ---------------------------------------------------------------------------


def test_run_generation_only_leaves_retrieval_empty(tmp_path):
    generation_registry, clinical_registry = split_generation_and_clinical_registries(
        build_mock_metric_registry()
    )
    runner = _make_runner(generation_registry, clinical_registry, retrieval_registry={"mrr": _OrderTrackingRetrievalMetric("mrr", [])})
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref text")], [_pred_row("q1", "hyp text")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.retrieval_scores == {}
    assert result.generation_corpus_scores
    assert result.clinical_corpus_scores


def test_run_retrieval_only_leaves_generation_and_per_sample_rows_empty():
    retrieval_registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(k_values=(100,)))
    runner = _make_runner(retrieval_registry=retrieval_registry)
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    assert result.generation_corpus_scores == {}
    assert result.clinical_corpus_scores == {}
    assert result.per_sample_rows == ()
    assert result.num_generation_samples_scored is None
    assert result.retrieval_scores


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- per-sample rows
# ---------------------------------------------------------------------------


def test_run_per_sample_rows_carry_per_example_scores(tmp_path):
    generation_registry = {"with_examples": _OrderTrackingGenerationMetric("with_examples", [])}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path,
        [_ref_row("q1", "ref one"), _ref_row("q2", "ref two")],
        [_pred_row("q1", "hyp one"), _pred_row("q2", "hyp two")],
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert len(result.per_sample_rows) == 2
    for row in result.per_sample_rows:
        assert row.generation_scores == {"with_examples": 1.0}
        assert row.clinical_scores == {}
        assert isinstance(row, PerSampleResultRow)


def test_run_per_sample_rows_are_none_for_metrics_without_per_example_decomposition(tmp_path):
    generation_registry = {"corpus_only": _NoPerExampleGenerationMetric("corpus_only", 0.42)}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.generation_corpus_scores == {"corpus_only": pytest.approx(0.42)}
    assert result.per_sample_rows[0].generation_scores == {"corpus_only": None}


def test_run_per_sample_rows_ordered_same_as_generation_load_result_query_keys(tmp_path):
    generation_registry = {"m": _OrderTrackingGenerationMetric("m", [])}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path,
        [_ref_row("q2", "ref two"), _ref_row("q1", "ref one")],  # file order reversed
        [_pred_row("q2", "hyp two"), _pred_row("q1", "hyp one")],
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    # Cell 30's loader already sorts by query_key -- the orchestrator must
    # not silently reorder relative to that.
    assert [row.query_key for row in result.per_sample_rows] == list(load_result.query_keys)
    assert result.per_sample_rows[0].query_key == qk("q1")
    assert result.per_sample_rows[1].query_key == qk("q2")


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- metric-level failure handling (never aborts)
# ---------------------------------------------------------------------------


def test_run_records_generation_metric_failure_and_continues(tmp_path):
    generation_registry = {
        "good": _OrderTrackingGenerationMetric("good", []),
        "bad": _RaisingGenerationMetric("bad", ValueError("simulated failure")),
    }
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert "good" in result.generation_corpus_scores
    assert "bad" not in result.generation_corpus_scores
    assert len(result.failures) == 1
    failure = result.failures[0]
    assert failure.metric_name == "bad"
    assert failure.namespace == "generation"
    assert "simulated failure" in failure.error_message


def test_run_records_clinical_metric_failure_with_correct_namespace(tmp_path):
    clinical_registry = {"bad_clinical": _RaisingGenerationMetric("bad_clinical", RuntimeError("boom"))}
    runner = _make_runner(clinical_registry=clinical_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.failures[0].namespace == "clinical"


def test_run_records_retrieval_metric_failure_and_continues():
    retrieval_registry = {
        "good": _OrderTrackingRetrievalMetric("good", []),
        "bad": _RaisingRetrievalMetric("bad", ValueError("simulated failure")),
    }
    runner = _make_runner(retrieval_registry=retrieval_registry)
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    assert "good" in result.retrieval_scores
    assert "bad" not in result.retrieval_scores
    assert result.failures[0].namespace == "retrieval"
    assert result.failures[0].metric_name == "bad"


def test_run_with_no_failures_has_empty_failures_tuple(tmp_path):
    generation_registry, clinical_registry = split_generation_and_clinical_registries(
        build_mock_metric_registry()
    )
    runner = _make_runner(generation_registry, clinical_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.failures == ()


# ---------------------------------------------------------------------------
# EvaluationRunner.run -- sample-level failure propagation (never silently
# dropped, per requirement #4)
# ---------------------------------------------------------------------------


def test_run_propagates_upstream_generation_failures_from_loader(tmp_path):
    generation_registry = {"m": _OrderTrackingGenerationMetric("m", [])}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path,
        [_ref_row("q1", "ref one"), _ref_row("q2", "ref two")],
        [_pred_row("q1", "hyp one"), _pred_row("q2", error="generation timed out")],
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.num_generation_samples_scored == 1
    assert result.num_generation_samples_failed_upstream == 1
    # the failed sample is not silently absent from the count, even though
    # it correctly has no per-sample row (nothing was ever scored for it)
    assert len(result.per_sample_rows) == 1


def test_run_propagates_dropped_malformed_hypothesis_count_from_loader(tmp_path):
    generation_registry = {"m": _OrderTrackingGenerationMetric("m", [])}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path,
        [_ref_row("q1", "ref one"), _ref_row("q2", "ref two")],
        [_pred_row("q1", "hyp one"), _pred_row("q2", retrieved_finding="")],
        config=GenerationMetricsConfig(malformed_hypothesis_filter_enabled=True),
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    assert result.num_generation_samples_dropped_malformed == 1


# ---------------------------------------------------------------------------
# EvaluationRunResult -- run metadata / immutability
# ---------------------------------------------------------------------------


def test_run_result_carries_run_metadata(tmp_path):
    runner = _make_runner()
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    assert isinstance(result.package_versions, dict)
    assert "torch" in result.package_versions
    assert isinstance(result.hardware, str)
    assert result.git_commit is None or isinstance(result.git_commit, str)
    assert result.generated_timestamp_utc.endswith("Z")


def test_run_result_is_immutable():
    runner = _make_runner()
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.hardware = "gpu"


def test_metric_failure_and_per_sample_row_are_immutable():
    failure = MetricFailure(metric_name="m", namespace="generation", error_message="x")
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.metric_name = "y"
    row = PerSampleResultRow(query_key=qk("q1"), generation_scores={}, clinical_scores={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.query_key = qk("q2")


# ---------------------------------------------------------------------------
# Atomic report writers
# ---------------------------------------------------------------------------


def test_write_evaluation_summary_json_atomic_writes_valid_json_excluding_per_sample_rows(tmp_path):
    generation_registry, clinical_registry = split_generation_and_clinical_registries(
        build_mock_metric_registry()
    )
    retrieval_registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(k_values=(100,)))
    runner = _make_runner(generation_registry, clinical_registry, retrieval_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="FactMM-RAG", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
        retrieval_positives={qk("p1"): (qk("c1"),)},
        retrieval_ranked={qk("p1"): (qk("c1"),)},
    )

    summary_path = tmp_path / "summary.json"
    write_evaluation_summary_json_atomic(summary_path, result)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "per_sample_rows" not in payload
    assert payload["run_config"]["run_id"] == "r1"
    assert payload["generation_corpus_scores"]
    assert payload["clinical_corpus_scores"]
    assert payload["retrieval_scores"]["mrr"]["100"] == pytest.approx(1.0)  # int k -> str key
    assert payload["failures"] == []
    assert payload["num_generation_samples_scored"] == 1


def test_write_evaluation_summary_json_atomic_serializes_failures(tmp_path):
    generation_registry = {"bad": _RaisingGenerationMetric("bad", ValueError("boom"))}
    runner = _make_runner(generation_registry)
    load_result = _make_generation_load_result(
        tmp_path, [_ref_row("q1", "ref one")], [_pred_row("q1", "hyp one")]
    )
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        generation_load_result=load_result,
    )
    summary_path = tmp_path / "summary.json"
    write_evaluation_summary_json_atomic(summary_path, result)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["failures"] == [{"metric_name": "bad", "namespace": "generation", "error_message": "boom"}]


def test_write_evaluation_summary_json_atomic_leaves_no_tmp_file(tmp_path):
    runner = _make_runner()
    result = runner.run(
        EvaluationRunConfig(run_id="r1", system_name="s", dataset_name="mimic-cxr"),
        retrieval_positives={qk("q1"): (qk("p1"),)},
        retrieval_ranked={qk("q1"): (qk("p1"),)},
    )
    summary_path = tmp_path / "summary.json"
    write_evaluation_summary_json_atomic(summary_path, result)
    assert list(tmp_path.glob("summary.json.tmp*")) == []


def test_write_per_sample_rows_jsonl_atomic_writes_one_line_per_row(tmp_path):
    rows = (
        PerSampleResultRow(query_key=qk("q1"), generation_scores={"m": 0.5}, clinical_scores={"c": None}),
        PerSampleResultRow(query_key=qk("q2"), generation_scores={"m": 0.9}, clinical_scores={"c": 0.1}),
    )
    rows_path = tmp_path / "rows.jsonl"
    write_per_sample_rows_jsonl_atomic(rows_path, rows)
    lines = rows_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["query_key"] == list(qk("q1"))
    assert first["generation_scores"] == {"m": 0.5}
    assert first["clinical_scores"] == {"c": None}


def test_write_per_sample_rows_jsonl_atomic_handles_empty_rows(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    write_per_sample_rows_jsonl_atomic(rows_path, ())
    assert rows_path.read_text(encoding="utf-8") == ""


def test_write_per_sample_rows_jsonl_atomic_leaves_no_tmp_file(tmp_path):
    rows = (PerSampleResultRow(query_key=qk("q1"), generation_scores={}, clinical_scores={}),)
    rows_path = tmp_path / "rows.jsonl"
    write_per_sample_rows_jsonl_atomic(rows_path, rows)
    assert list(tmp_path.glob("rows.jsonl.tmp*")) == []
