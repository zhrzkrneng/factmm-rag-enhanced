"""Unit tests for src/baseline/evaluation/oracle.py (Milestone 2.6, Cell 35).

Scope: OracleConfig, compute_oracle_score, OracleEvaluator.build_row
(argmax correctness, self-exclusion, tie-breaking, top-k tracking),
OracleEvaluator.build_all (resume/atomic-write/continue_on_error,
matching the established PairMiner/RadGraphAnnotator/RAGDatasetBuilder/
LLaVAGenerator pattern), and oracle_result_row_to_pred_row_dict's
structural compatibility with Cell 30's real load_prediction_rows
parser. Every scorer is a small, deterministic local fake -- no real
CheXbert/RadGraph instance scoring anywhere in this file."""

import dataclasses
import json

import pytest

from src.baseline.evaluation.oracle import (
    OracleBuildSummary,
    OracleConfig,
    OracleEvaluator,
    OracleResultRow,
    compute_oracle_score,
    oracle_result_row_to_pred_row_dict,
)
from src.common.exceptions import EvaluationError
from src.data.schema import ReportRecord
from src.evaluation.generation_metrics import load_prediction_rows


def qk(name):
    return ("mimic-cxr", name, name)


def rec(name, finding):
    return ReportRecord(
        image_paths=[], finding=finding, impression="", patient_id=name, study_id=name, dataset="mimic-cxr"
    )


def _zero_scorer(a, b):
    return 0.0


# ---------------------------------------------------------------------------
# OracleConfig
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"corpus_scope": ""},
        {"exclude_self": "yes"},
        {"top_k_candidates_considered": 0},
        {"top_k_candidates_considered": -1},
        {"top_k_candidates_considered": 1.5},
        {"top_k_candidates_considered": True},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        OracleConfig(**overrides)


def test_config_defaults_are_valid():
    config = OracleConfig()
    assert config.exclude_self is True
    assert config.top_k_candidates_considered == 30
    assert config.corpus_scope == "train_only"


def test_config_is_immutable():
    config = OracleConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.exclude_self = False


def test_config_as_actual_used_dict_round_trips_all_fields():
    config = OracleConfig()
    actual = config.as_actual_used_dict()
    assert set(actual) == {f.name for f in dataclasses.fields(OracleConfig)}


def test_oracle_result_row_and_summary_are_immutable():
    row = OracleResultRow(
        query_key=qk("q1"), winning_key=qk("c1"), winning_score=1.0,
        winning_finding_text="text", excluded=False, num_candidates_considered=1,
        top_k_keys=(qk("c1"),), top_k_scores=(1.0,), config_version="1.0",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.winning_score = 2.0
    summary = OracleBuildSummary(processed=1, skipped_already_done=0, failed=0, excluded=0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        summary.processed = 2


# ---------------------------------------------------------------------------
# compute_oracle_score
# ---------------------------------------------------------------------------


def test_compute_oracle_score_is_a_plain_sum():
    assert compute_oracle_score(0.4, 0.3) == pytest.approx(0.7)
    assert compute_oracle_score(0.0, 0.0) == pytest.approx(0.0)
    assert compute_oracle_score(1.0, 1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# OracleEvaluator construction
# ---------------------------------------------------------------------------


def test_evaluator_rejects_empty_query_records():
    with pytest.raises(EvaluationError, match="query_records must be non-empty"):
        OracleEvaluator(
            {}, {qk("c1"): rec("c1", "x")}, config=OracleConfig(),
            chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
        )


def test_evaluator_rejects_empty_corpus_records():
    with pytest.raises(EvaluationError, match="corpus_records must be non-empty"):
        OracleEvaluator(
            {qk("q1"): rec("q1", "x")}, {}, config=OracleConfig(),
            chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
        )


# ---------------------------------------------------------------------------
# build_row -- argmax correctness
# ---------------------------------------------------------------------------


def test_build_row_picks_highest_scoring_candidate():
    query_records = {qk("q1"): rec("q1", "query text")}
    corpus_records = {
        qk("c1"): rec("c1", "low score candidate"),
        qk("c2"): rec("c2", "high score candidate"),
        qk("c3"): rec("c3", "medium score candidate"),
    }

    def chex(a, b):
        return {"low score candidate": 0.1, "high score candidate": 0.9, "medium score candidate": 0.5}[b]

    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(),
        chexbert_instance_scorer=chex, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.winning_key == qk("c2")
    assert row.winning_score == pytest.approx(0.9)
    assert row.winning_finding_text == "high score candidate"
    assert row.excluded is False
    assert row.num_candidates_considered == 3


def test_build_row_score_is_sum_of_both_scorers():
    query_records = {qk("q1"): rec("q1", "q")}
    corpus_records = {qk("c1"): rec("c1", "c")}
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(),
        chexbert_instance_scorer=lambda a, b: 0.3, radgraph_instance_scorer=lambda a, b: 0.4,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.winning_score == pytest.approx(0.7)


def test_build_row_raises_for_unknown_query_key():
    query_records = {qk("q1"): rec("q1", "q")}
    corpus_records = {qk("c1"): rec("c1", "c")}
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(),
        chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
    )
    with pytest.raises(EvaluationError, match="no matching query record"):
        evaluator.build_row(qk("unknown"))


# ---------------------------------------------------------------------------
# build_row -- self-exclusion
# ---------------------------------------------------------------------------


def test_build_row_excludes_self_by_default_even_if_it_would_win():
    query_records = {qk("q1"): rec("q1", "identical text")}
    corpus_records = {
        qk("q1"): rec("q1", "identical text"),  # self -- would score highest (perfect match)
        qk("c1"): rec("c1", "different text"),
    }

    def chex(a, b):
        return 1.0 if a == b else 0.1

    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=True),
        chexbert_instance_scorer=chex, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.winning_key == qk("c1")  # not q1, despite q1 scoring higher
    assert row.num_candidates_considered == 1


def test_build_row_allows_self_when_exclude_self_is_false():
    query_records = {qk("q1"): rec("q1", "identical text")}
    corpus_records = {
        qk("q1"): rec("q1", "identical text"),
        qk("c1"): rec("c1", "different text"),
    }

    def chex(a, b):
        return 1.0 if a == b else 0.1

    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=False),
        chexbert_instance_scorer=chex, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.winning_key == qk("q1")
    assert row.num_candidates_considered == 2


def test_build_row_excluded_when_no_eligible_candidates():
    query_records = {qk("q1"): rec("q1", "q")}
    corpus_records = {qk("q1"): rec("q1", "q")}  # only self in corpus
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=True),
        chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.excluded is True
    assert row.winning_key is None
    assert row.winning_score is None
    assert row.winning_finding_text is None
    assert row.num_candidates_considered == 0
    assert row.top_k_keys == ()
    assert row.top_k_scores == ()


# ---------------------------------------------------------------------------
# build_row -- tie-breaking (contract §13: lowest corpus-iteration index)
# ---------------------------------------------------------------------------


def test_build_row_tie_breaks_to_lowest_iteration_order_index():
    query_records = {qk("q1"): rec("q1", "q")}
    # dict insertion order is preserved in Python 3.7+ -- c1 appears
    # before c2, both score identically
    corpus_records = {qk("c1"): rec("c1", "tied"), qk("c2"): rec("c2", "tied")}
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(),
        chexbert_instance_scorer=lambda a, b: 0.5, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert row.winning_key == qk("c1")  # first in iteration order wins the tie

    # reversed insertion order -> c2 should now win, proving this is
    # genuinely order-driven, not a hidden key-based tiebreak
    corpus_records_reversed = {qk("c2"): rec("c2", "tied"), qk("c1"): rec("c1", "tied")}
    evaluator_reversed = OracleEvaluator(
        query_records, corpus_records_reversed, config=OracleConfig(),
        chexbert_instance_scorer=lambda a, b: 0.5, radgraph_instance_scorer=_zero_scorer,
    )
    row_reversed = evaluator_reversed.build_row(qk("q1"))
    assert row_reversed.winning_key == qk("c2")


# ---------------------------------------------------------------------------
# build_row -- top-k tracking
# ---------------------------------------------------------------------------


def test_build_row_top_k_is_truncated_and_sorted_descending():
    query_records = {qk("q1"): rec("q1", "q")}
    corpus_records = {qk(f"c{i}"): rec(f"c{i}", f"c{i}") for i in range(5)}
    scores = {"c0": 0.1, "c1": 0.5, "c2": 0.9, "c3": 0.3, "c4": 0.7}

    def chex(a, b):
        return scores[b]

    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(top_k_candidates_considered=3),
        chexbert_instance_scorer=chex, radgraph_instance_scorer=_zero_scorer,
    )
    row = evaluator.build_row(qk("q1"))
    assert len(row.top_k_keys) == 3
    assert row.top_k_scores == (pytest.approx(0.9), pytest.approx(0.7), pytest.approx(0.5))
    assert row.top_k_keys == (qk("c2"), qk("c4"), qk("c1"))
    assert row.num_candidates_considered == 5  # all 5 scored, only top 3 retained for debugging


def test_build_row_is_deterministic():
    query_records = {qk("q1"): rec("q1", "q")}
    corpus_records = {qk("c1"): rec("c1", "c1"), qk("c2"): rec("c2", "c2")}
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(),
        chexbert_instance_scorer=lambda a, b: len(b) * 0.01, radgraph_instance_scorer=_zero_scorer,
    )
    first = evaluator.build_row(qk("q1"))
    second = evaluator.build_row(qk("q1"))
    assert first == second


# ---------------------------------------------------------------------------
# oracle_result_row_to_pred_row_dict
# ---------------------------------------------------------------------------


def test_pred_row_dict_shape_for_normal_row():
    row = OracleResultRow(
        query_key=qk("q1"), winning_key=qk("c1"), winning_score=1.0,
        winning_finding_text="the winning text", excluded=False, num_candidates_considered=1,
        top_k_keys=(qk("c1"),), top_k_scores=(1.0,), config_version="1.0",
    )
    assert oracle_result_row_to_pred_row_dict(row) == {
        "query_key": ["mimic-cxr", "q1", "q1"],
        "retrieved_finding": ["the winning text"],
    }


def test_pred_row_dict_shape_for_excluded_row():
    row = OracleResultRow(
        query_key=qk("q1"), winning_key=None, winning_score=None, winning_finding_text=None,
        excluded=True, num_candidates_considered=0, top_k_keys=(), top_k_scores=(), config_version="1.0",
    )
    result = oracle_result_row_to_pred_row_dict(row)
    assert result["query_key"] == ["mimic-cxr", "q1", "q1"]
    assert "error" in result
    assert "retrieved_finding" not in result


def test_pred_row_dict_is_structurally_compatible_with_cell_30_real_parser(tmp_path):
    query_records = {qk("q1"): rec("q1", "q1 text"), qk("q2"): rec("q2", "q2 text")}
    corpus_records = {qk("q1"): rec("q1", "q1 text"), qk("c1"): rec("c1", "c1 text")}
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=True),
        chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
    )
    row_q1 = evaluator.build_row(qk("q1"))  # finds c1
    row_q2 = evaluator.build_row(qk("q2"))  # q2 not in corpus at all -> excluded is False, finds q1 or c1

    pred_path = tmp_path / "pred.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(oracle_result_row_to_pred_row_dict(row_q1)) + "\n")
        handle.write(json.dumps(oracle_result_row_to_pred_row_dict(row_q2)) + "\n")

    parsed = load_prediction_rows(pred_path, require_key=True)
    assert len(parsed) == 2
    assert parsed[0].query_key == qk("q1")
    assert parsed[0].error is None
    assert parsed[0].retrieved_finding is not None


def test_pred_row_dict_excluded_row_is_structurally_compatible_with_cell_30_real_parser(tmp_path):
    row = OracleResultRow(
        query_key=qk("q1"), winning_key=None, winning_score=None, winning_finding_text=None,
        excluded=True, num_candidates_considered=0, top_k_keys=(), top_k_scores=(), config_version="1.0",
    )
    pred_path = tmp_path / "pred.jsonl"
    pred_path.write_text(json.dumps(oracle_result_row_to_pred_row_dict(row)) + "\n", encoding="utf-8")
    parsed = load_prediction_rows(pred_path, require_key=True)
    assert parsed[0].error is not None
    assert parsed[0].retrieved_finding is None


# ---------------------------------------------------------------------------
# build_all -- basic + resume + atomic pattern
# ---------------------------------------------------------------------------


def _make_evaluator(query_records=None, corpus_records=None, config=None, chex=None, radg=None):
    query_records = query_records or {qk("q1"): rec("q1", "q1 text"), qk("q2"): rec("q2", "q2 text")}
    corpus_records = corpus_records or {qk("c1"): rec("c1", "c1 text"), qk("c2"): rec("c2", "c2 text")}
    return OracleEvaluator(
        query_records, corpus_records, config=config or OracleConfig(),
        chexbert_instance_scorer=chex or (lambda a, b: 0.5),
        radgraph_instance_scorer=radg or (lambda a, b: 0.3),
    )


def test_build_all_writes_one_row_per_query(tmp_path):
    evaluator = _make_evaluator()
    output_path = tmp_path / "oracle.jsonl"
    summary = evaluator.build_all([qk("q1"), qk("q2")], output_path=output_path)
    assert summary == OracleBuildSummary(processed=2, skipped_already_done=0, failed=0, excluded=0)
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_build_all_writes_meta_json_with_completed_status(tmp_path):
    evaluator = _make_evaluator()
    output_path = tmp_path / "oracle.jsonl"
    evaluator.build_all([qk("q1")], output_path=output_path)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_status"] == "completed"
    assert meta["summary"]["processed"] == 1


def test_build_all_resumes_and_skips_already_completed(tmp_path):
    evaluator = _make_evaluator()
    output_path = tmp_path / "oracle.jsonl"
    evaluator.build_all([qk("q1"), qk("q2")], output_path=output_path)

    second_summary = evaluator.build_all([qk("q1"), qk("q2")], output_path=output_path)
    assert second_summary == OracleBuildSummary(processed=0, skipped_already_done=2, failed=0, excluded=0)
    # no duplicate lines written
    lines = output_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_build_all_rejects_resume_under_incompatible_config(tmp_path):
    output_path = tmp_path / "oracle.jsonl"
    evaluator_1 = _make_evaluator(config=OracleConfig(exclude_self=True))
    evaluator_1.build_all([qk("q1")], output_path=output_path)

    evaluator_2 = _make_evaluator(config=OracleConfig(exclude_self=False))
    with pytest.raises(EvaluationError, match="different"):
        evaluator_2.build_all([qk("q2")], output_path=output_path)


def test_build_all_rejects_malformed_existing_output(tmp_path):
    output_path = tmp_path / "oracle.jsonl"
    output_path.write_text("not valid json\n", encoding="utf-8")
    evaluator = _make_evaluator()
    with pytest.raises(EvaluationError, match="Malformed existing output"):
        evaluator.build_all([qk("q1")], output_path=output_path)


def test_build_all_rejects_duplicate_key_in_existing_output(tmp_path):
    output_path = tmp_path / "oracle.jsonl"
    row_dict = {
        "dataset": "mimic-cxr", "patient_id": "q1", "study_id": "q1",
        "winning_key": ["mimic-cxr", "c1", "c1"], "winning_score": 1.0,
        "winning_finding_text": "text", "excluded": False,
        "num_candidates_considered": 1, "top_k_keys": [["mimic-cxr", "c1", "c1"]],
        "top_k_scores": [1.0], "config_version": "1.0",
    }
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(row_dict) + "\n")
        handle.write(json.dumps(row_dict) + "\n")
    evaluator = _make_evaluator()
    with pytest.raises(EvaluationError, match="Duplicate query key"):
        evaluator.build_all([qk("q1")], output_path=output_path)


def test_build_all_continue_on_error_requires_errors_path(tmp_path):
    evaluator = _make_evaluator()
    output_path = tmp_path / "oracle.jsonl"
    with pytest.raises(ValueError, match="errors_path is required"):
        evaluator.build_all([qk("q1")], output_path=output_path, continue_on_error=True)
    assert not output_path.exists()  # raised before touching output_path


def test_build_all_fail_fast_marks_meta_failed_and_reraises(tmp_path):
    def raising_chex(a, b):
        raise RuntimeError("simulated scorer failure")

    evaluator = _make_evaluator(chex=raising_chex)
    output_path = tmp_path / "oracle.jsonl"
    with pytest.raises(RuntimeError, match="simulated scorer failure"):
        evaluator.build_all([qk("q1")], output_path=output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_status"] == "failed"
    assert meta["last_error"]["error_type"] == "RuntimeError"


def test_build_all_continue_on_error_logs_and_continues(tmp_path):
    call_count = {"n": 0}

    def sometimes_raising_chex(a, b):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated failure on first query")
        return 0.5

    evaluator = _make_evaluator(chex=sometimes_raising_chex)
    output_path = tmp_path / "oracle.jsonl"
    errors_path = tmp_path / "errors.jsonl"
    summary = evaluator.build_all(
        [qk("q1"), qk("q2")], output_path=output_path, errors_path=errors_path, continue_on_error=True
    )
    assert summary.failed == 1
    assert summary.processed == 1
    error_lines = errors_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(error_lines) == 1
    error_payload = json.loads(error_lines[0])
    assert error_payload["error_type"] == "RuntimeError"


def test_build_all_tracks_excluded_count_in_summary(tmp_path):
    query_records = {qk("q1"): rec("q1", "q1")}
    corpus_records = {qk("q1"): rec("q1", "q1")}  # only self -> excluded
    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=True),
        chexbert_instance_scorer=_zero_scorer, radgraph_instance_scorer=_zero_scorer,
    )
    output_path = tmp_path / "oracle.jsonl"
    summary = evaluator.build_all([qk("q1")], output_path=output_path)
    assert summary.excluded == 1
    assert summary.processed == 1
