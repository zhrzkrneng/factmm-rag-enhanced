"""Unit tests for src/evaluation/generation_metrics.py (Milestone 2.6, Cell 30).

Scope: GenerationMetricsConfig, the input/output dataclasses, and
load_generation_pairs() -- strict query_key alignment, duplicate/missing-key
validation, malformed-row handling, deterministic ordering, and
resume-safe load metadata. No metric computation is exercised here (none
exists yet)."""

import dataclasses
import json

import pytest

from src.evaluation.generation_metrics import (
    GenerationExampleScore,
    GenerationLoadResult,
    GenerationLoadSummary,
    GenerationMetricsConfig,
    GenerationMetricsResult,
    PredictionRow,
    ReferenceRow,
    load_generation_pairs,
    load_prediction_rows,
    load_reference_rows,
    write_generation_load_summary_atomic,
)
from src.common.exceptions import EvaluationError


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _ref_row(dataset, patient_id, study_id, finding):
    return {"query_key": [dataset, patient_id, study_id], "finding": finding}


def _pred_row(dataset, patient_id, study_id, retrieved_finding=None, error=None):
    obj = {"query_key": [dataset, patient_id, study_id]}
    if error is not None:
        obj["error"] = error
    else:
        obj["retrieved_finding"] = [retrieved_finding]
    return obj


# ---------------------------------------------------------------------------
# GenerationMetricsConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"radgraph_reward_level": ""},
        {"radgraph_model_type": ""},
        {"chexbert_device": ""},
        {"bert_score_model_type": ""},
        {"bert_score_num_layers": 0},
        {"bert_score_num_layers": -1},
        {"bert_score_num_layers": 1.5},
        {"bert_score_num_layers": True},
        {"bert_score_batch_size": 0},
        {"bert_score_batch_size": True},
        {"bert_score_rescale_with_baseline": "yes"},
        {"bleu_enabled": 1},
        {"pairing_mode": "something_else"},
        {"malformed_hypothesis_filter_enabled": "no"},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        GenerationMetricsConfig(**overrides)


def test_config_defaults_are_valid():
    config = GenerationMetricsConfig()
    assert config.pairing_mode == "by_query_key"
    assert config.malformed_hypothesis_filter_enabled is False


def test_config_is_immutable():
    config = GenerationMetricsConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.bleu_enabled = False


def test_config_as_actual_used_dict_round_trips_all_fields():
    config = GenerationMetricsConfig()
    actual = config.as_actual_used_dict()
    assert actual["pairing_mode"] == "by_query_key"
    assert actual["malformed_hypothesis_filter_enabled"] is False
    assert set(actual) == {f.name for f in dataclasses.fields(GenerationMetricsConfig)}


# ---------------------------------------------------------------------------
# Dataclass immutability / shape
# ---------------------------------------------------------------------------


def test_reference_row_is_immutable():
    row = ReferenceRow(query_key=("mimic-cxr", "p1", "s1"), finding="text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.finding = "other"


def test_prediction_row_is_immutable():
    row = PredictionRow(query_key=("mimic-cxr", "p1", "s1"), retrieved_finding="text")
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.retrieved_finding = "other"


def test_generation_example_score_and_result_are_immutable():
    score = GenerationExampleScore(
        query_key=("mimic-cxr", "p1", "s1"),
        f1_radgraph=0.5,
        f1_chexbert_instance=0.4,
        rouge_l=0.3,
        bert_score=0.6,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        score.f1_radgraph = 1.0

    result = GenerationMetricsResult(
        config_version="1.0",
        num_examples_scored=1,
        num_examples_dropped=0,
        f1_radgraph=0.5,
        f1_chexbert=0.4,
        rouge_l=0.3,
        bleu4=None,
        bert_score=0.6,
        per_example=(score,),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.f1_radgraph = 1.0


# ---------------------------------------------------------------------------
# load_reference_rows / load_prediction_rows -- JSONL schema validation
# ---------------------------------------------------------------------------


def test_load_reference_rows_happy_path(tmp_path):
    path = tmp_path / "ref.jsonl"
    _write_jsonl(path, [_ref_row("mimic-cxr", "p1", "s1", "finding one")])
    rows = load_reference_rows(path, require_key=True)
    assert rows == [ReferenceRow(query_key=("mimic-cxr", "p1", "s1"), finding="finding one")]


def test_load_reference_rows_skips_blank_lines(tmp_path):
    path = tmp_path / "ref.jsonl"
    path.write_text(
        json.dumps(_ref_row("mimic-cxr", "p1", "s1", "f")) + "\n\n   \n", encoding="utf-8"
    )
    rows = load_reference_rows(path, require_key=True)
    assert len(rows) == 1


def test_load_reference_rows_missing_finding_field_raises(tmp_path):
    path = tmp_path / "ref.jsonl"
    _write_jsonl(path, [{"query_key": ["mimic-cxr", "p1", "s1"]}])
    with pytest.raises(EvaluationError, match="missing required field 'finding'"):
        load_reference_rows(path, require_key=True)


def test_load_reference_rows_wrong_type_finding_raises(tmp_path):
    path = tmp_path / "ref.jsonl"
    _write_jsonl(path, [{"query_key": ["mimic-cxr", "p1", "s1"], "finding": 123}])
    with pytest.raises(EvaluationError, match="'finding' must be a string"):
        load_reference_rows(path, require_key=True)


def test_load_reference_rows_malformed_json_raises_with_line_number(tmp_path):
    path = tmp_path / "ref.jsonl"
    path.write_text('{"finding": "ok"\n', encoding="utf-8")
    with pytest.raises(EvaluationError, match="line 1"):
        load_reference_rows(path, require_key=False)


def test_load_reference_rows_non_object_line_raises(tmp_path):
    path = tmp_path / "ref.jsonl"
    path.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(EvaluationError, match="expected a JSON object"):
        load_reference_rows(path, require_key=False)


@pytest.mark.parametrize(
    "bad_key",
    [
        "not-a-list",
        ["only", "two"],
        ["a", "b", "c", "d"],
        ["a", "", "c"],
        ["a", 1, "c"],
        None,
    ],
)
def test_load_reference_rows_rejects_malformed_query_key(tmp_path, bad_key):
    path = tmp_path / "ref.jsonl"
    _write_jsonl(path, [{"query_key": bad_key, "finding": "f"}])
    with pytest.raises(EvaluationError, match="query_key"):
        load_reference_rows(path, require_key=True)


def test_load_reference_rows_ignores_query_key_when_not_required(tmp_path):
    path = tmp_path / "ref.jsonl"
    _write_jsonl(path, [{"finding": "f"}])  # no query_key at all
    rows = load_reference_rows(path, require_key=False)
    assert rows == [ReferenceRow(query_key=None, finding="f")]


def test_load_prediction_rows_happy_path_success(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp text")])
    rows = load_prediction_rows(path, require_key=True)
    assert rows == [
        PredictionRow(query_key=("mimic-cxr", "p1", "s1"), retrieved_finding="hyp text", error=None)
    ]


def test_load_prediction_rows_happy_path_error(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(path, [_pred_row("mimic-cxr", "p1", "s1", error="generation timed out")])
    rows = load_prediction_rows(path, require_key=True)
    assert rows == [
        PredictionRow(query_key=("mimic-cxr", "p1", "s1"), retrieved_finding=None, error="generation timed out")
    ]


def test_load_prediction_rows_rejects_both_finding_and_error(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(
        path,
        [{"query_key": ["mimic-cxr", "p1", "s1"], "retrieved_finding": ["hyp"], "error": "oops"}],
    )
    with pytest.raises(EvaluationError, match="got both"):
        load_prediction_rows(path, require_key=True)


def test_load_prediction_rows_rejects_neither_finding_nor_error(tmp_path):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(path, [{"query_key": ["mimic-cxr", "p1", "s1"]}])
    with pytest.raises(EvaluationError, match="got neither"):
        load_prediction_rows(path, require_key=True)


@pytest.mark.parametrize("bad_finding", ["not-a-list", [], [123], [None]])
def test_load_prediction_rows_rejects_malformed_retrieved_finding(tmp_path, bad_finding):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(
        path, [{"query_key": ["mimic-cxr", "p1", "s1"], "retrieved_finding": bad_finding}]
    )
    with pytest.raises(EvaluationError, match="retrieved_finding"):
        load_prediction_rows(path, require_key=True)


@pytest.mark.parametrize("bad_error", ["", 123, True])
def test_load_prediction_rows_rejects_malformed_error(tmp_path, bad_error):
    path = tmp_path / "pred.jsonl"
    _write_jsonl(path, [{"query_key": ["mimic-cxr", "p1", "s1"], "error": bad_error}])
    with pytest.raises(EvaluationError, match="'error' must be a non-empty string"):
        load_prediction_rows(path, require_key=True)


# ---------------------------------------------------------------------------
# load_generation_pairs -- by_query_key mode (default)
# ---------------------------------------------------------------------------


def test_load_generation_pairs_happy_path_by_query_key(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p2", "s2", "ref two"),
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
        ],
    )
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one"),
            _pred_row("mimic-cxr", "p2", "s2", retrieved_finding="hyp two"),
        ],
    )
    config = GenerationMetricsConfig()
    result = load_generation_pairs(ref_path, pred_path, config)

    assert isinstance(result, GenerationLoadResult)
    # deterministic ordering: sorted by query_key, independent of file order
    assert result.query_keys == (("mimic-cxr", "p1", "s1"), ("mimic-cxr", "p2", "s2"))
    assert result.hyps == ("hyp one", "hyp two")
    assert result.refs == ("ref one", "ref two")
    assert result.summary.num_paired == 2
    assert result.summary.num_reference_rows == 2
    assert result.summary.num_prediction_rows == 2
    assert result.summary.num_failed_generations == 0
    assert result.summary.num_dropped_malformed_hypothesis == 0
    assert result.summary.pairing_mode == "by_query_key"
    assert result.summary.git_commit is None or isinstance(result.summary.git_commit, str)


def test_load_generation_pairs_ordering_independent_of_file_order(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p3", "s3", "ref three"),
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
            _ref_row("mimic-cxr", "p2", "s2", "ref two"),
        ],
    )
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p2", "s2", retrieved_finding="hyp two"),
            _pred_row("mimic-cxr", "p3", "s3", retrieved_finding="hyp three"),
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one"),
        ],
    )
    result = load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())
    assert result.query_keys == (
        ("mimic-cxr", "p1", "s1"),
        ("mimic-cxr", "p2", "s2"),
        ("mimic-cxr", "p3", "s3"),
    )
    assert result.hyps == ("hyp one", "hyp two", "hyp three")
    assert result.refs == ("ref one", "ref two", "ref three")


def test_load_generation_pairs_rejects_duplicate_reference_key(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
            _ref_row("mimic-cxr", "p1", "s1", "ref one again"),
        ],
    )
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp")])
    with pytest.raises(EvaluationError, match="[Dd]uplicate query_key"):
        load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())


def test_load_generation_pairs_rejects_duplicate_prediction_key(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp"),
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp again"),
        ],
    )
    with pytest.raises(EvaluationError, match="[Dd]uplicate query_key"):
        load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())


def test_load_generation_pairs_rejects_prediction_with_no_matching_reference(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp"),
            _pred_row("mimic-cxr", "pX", "sX", retrieved_finding="orphan"),
        ],
    )
    with pytest.raises(EvaluationError, match="no matching reference"):
        load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())


def test_load_generation_pairs_rejects_reference_with_no_matching_prediction(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
            _ref_row("mimic-cxr", "p2", "s2", "ref two"),
        ],
    )
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp")])
    with pytest.raises(EvaluationError, match="no matching prediction row"):
        load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())


def test_load_generation_pairs_keeps_failed_generation_visible_not_dropped(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
            _ref_row("mimic-cxr", "p2", "s2", "ref two"),
        ],
    )
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one"),
            _pred_row("mimic-cxr", "p2", "s2", error="OOM during generation"),
        ],
    )
    result = load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())
    assert result.summary.num_paired == 1
    assert result.summary.num_failed_generations == 1
    assert result.summary.failed_query_keys == (("mimic-cxr", "p2", "s2"),)
    # the failed query never silently vanishes -- it's counted, named, and
    # simply excluded from the scored hyps/refs (nothing to score)
    assert result.query_keys == (("mimic-cxr", "p1", "s1"),)
    assert result.hyps == ("hyp one",)
    assert result.refs == ("ref one",)


def test_load_generation_pairs_rejects_blank_hypothesis_by_default(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="   ")])
    with pytest.raises(EvaluationError, match="blank"):
        load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())


def test_load_generation_pairs_official_filter_drops_only_empty_string_silently(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(
        ref_path,
        [
            _ref_row("mimic-cxr", "p1", "s1", "ref one"),
            _ref_row("mimic-cxr", "p2", "s2", "ref two"),
        ],
    )
    _write_jsonl(
        pred_path,
        [
            _pred_row("mimic-cxr", "p1", "s1", retrieved_finding=""),
            _pred_row("mimic-cxr", "p2", "s2", retrieved_finding="real hyp"),
        ],
    )
    config = GenerationMetricsConfig(malformed_hypothesis_filter_enabled=True)
    result = load_generation_pairs(ref_path, pred_path, config)
    assert result.summary.num_dropped_malformed_hypothesis == 1
    assert result.summary.dropped_malformed_hypothesis_keys == (("mimic-cxr", "p1", "s1"),)
    assert result.query_keys == (("mimic-cxr", "p2", "s2"),)
    assert result.hyps == ("real hyp",)


def test_load_generation_pairs_official_filter_does_not_catch_whitespace_only(tmp_path):
    # Documented official-code quirk (contract §18): a purely whitespace
    # hyp is NOT caught by the official filter, only a literally empty
    # string (or a string made only of "." characters) is.
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="   ")])
    config = GenerationMetricsConfig(malformed_hypothesis_filter_enabled=True)
    result = load_generation_pairs(ref_path, pred_path, config)
    assert result.summary.num_dropped_malformed_hypothesis == 0
    assert result.hyps == ("   ",)


# ---------------------------------------------------------------------------
# load_generation_pairs -- positional_official_reproduction mode (opt-in)
# ---------------------------------------------------------------------------


def test_load_generation_pairs_positional_mode_happy_path(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [{"finding": "ref one"}, {"finding": "ref two"}])
    _write_jsonl(
        pred_path,
        [{"retrieved_finding": ["hyp one"]}, {"retrieved_finding": ["hyp two"]}],
    )
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    result = load_generation_pairs(ref_path, pred_path, config)
    assert result.query_keys == (None, None)
    assert result.hyps == ("hyp one", "hyp two")
    assert result.refs == ("ref one", "ref two")
    assert result.summary.pairing_mode == "positional_official_reproduction"


def test_load_generation_pairs_positional_mode_rejects_length_mismatch(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [{"finding": "ref one"}, {"finding": "ref two"}])
    _write_jsonl(pred_path, [{"retrieved_finding": ["hyp one"]}])
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    with pytest.raises(EvaluationError, match="equal length"):
        load_generation_pairs(ref_path, pred_path, config)


def test_load_generation_pairs_positional_mode_ignores_query_key_field(tmp_path):
    # A query_key field present in the file is simply never read/validated
    # under positional mode -- it is neither required nor forbidden.
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one")])
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    result = load_generation_pairs(ref_path, pred_path, config)
    assert result.query_keys == (None,)
    assert result.hyps == ("hyp one",)


def test_load_generation_pairs_positional_mode_tracks_failed_generation_by_index(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [{"finding": "ref one"}, {"finding": "ref two"}])
    _write_jsonl(
        pred_path,
        [{"error": "failed"}, {"retrieved_finding": ["hyp two"]}],
    )
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    result = load_generation_pairs(ref_path, pred_path, config)
    assert result.summary.num_failed_generations == 1
    assert result.summary.failed_query_keys == (0,)
    assert result.hyps == ("hyp two",)
    assert result.refs == ("ref two",)


# ---------------------------------------------------------------------------
# resume-safe metadata sidecar
# ---------------------------------------------------------------------------


def test_write_generation_load_summary_atomic_round_trips(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one")])
    result = load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())

    meta_path = tmp_path / "load.meta.json"
    write_generation_load_summary_atomic(meta_path, result.summary)
    assert meta_path.exists()

    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["num_paired"] == 1
    assert payload["pairing_mode"] == "by_query_key"
    assert payload["failed_query_keys"] == []
    assert payload["config_version"] == "1.0"


def test_write_generation_load_summary_atomic_serializes_positional_indices_as_ints(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [{"finding": "ref one"}])
    _write_jsonl(pred_path, [{"error": "failed"}])
    config = GenerationMetricsConfig(pairing_mode="positional_official_reproduction")
    result = load_generation_pairs(ref_path, pred_path, config)

    meta_path = tmp_path / "load.meta.json"
    write_generation_load_summary_atomic(meta_path, result.summary)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["failed_query_keys"] == [0]


def test_write_generation_load_summary_atomic_is_atomic_no_tmp_file_left(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one")])
    result = load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())

    meta_path = tmp_path / "load.meta.json"
    write_generation_load_summary_atomic(meta_path, result.summary)
    leftover_tmp_files = list(tmp_path.glob("load.meta.json.tmp*"))
    assert leftover_tmp_files == []


# ---------------------------------------------------------------------------
# GenerationLoadSummary is immutable
# ---------------------------------------------------------------------------


def test_generation_load_summary_is_immutable(tmp_path):
    ref_path = tmp_path / "ref.jsonl"
    pred_path = tmp_path / "pred.jsonl"
    _write_jsonl(ref_path, [_ref_row("mimic-cxr", "p1", "s1", "ref one")])
    _write_jsonl(pred_path, [_pred_row("mimic-cxr", "p1", "s1", retrieved_finding="hyp one")])
    result = load_generation_pairs(ref_path, pred_path, GenerationMetricsConfig())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.summary.num_paired = 999
