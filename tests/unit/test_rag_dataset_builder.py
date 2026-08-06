"""Unit tests for src/baseline/generation/dataset_builder.py."""

import dataclasses
import json

import pytest

from src.baseline.generation.dataset_builder import (
    RAGDatasetBuilder,
    RAGDatasetBuilderConfig,
    RAGDatasetRow,
)
from src.common.exceptions import PatientLeakageError, RAGDatasetError


def _record(dataset, patient_id, study_id, finding, image_path=None, **extra):
    row = {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "finding": finding,
        "image_path": image_path if image_path is not None else f"/images/{dataset}/{study_id}.png",
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# RAGDatasetBuilderConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"rag_data_mode": ""},
        {"output_data_mode": ""},
        {"strict_patient_isolation": "yes"},
        {"reproduce_official_bug": "no"},
        {"test_short": 1},
        {"min_short_words": 0},
        {"min_short_words": -1},
        {"min_short_words": 1.5},
        {"min_short_words": True},
        {"corpus_scope": ""},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        RAGDatasetBuilderConfig(**overrides)


def test_config_defaults_are_valid():
    RAGDatasetBuilderConfig()  # must not raise


def test_config_rejects_both_isolation_flags_true():
    with pytest.raises(ValueError):
        RAGDatasetBuilderConfig(strict_patient_isolation=True, reproduce_official_bug=True)


def test_config_rejects_reproduce_official_bug_with_default_strict_isolation():
    # reproduce_official_bug=True alone, without explicitly disabling
    # strict_patient_isolation, must also raise -- the default True
    # is not silently overridden.
    with pytest.raises(ValueError):
        RAGDatasetBuilderConfig(reproduce_official_bug=True)


def test_config_allows_reproduce_official_bug_with_isolation_explicitly_off():
    config = RAGDatasetBuilderConfig(strict_patient_isolation=False, reproduce_official_bug=True)
    assert config.reproduce_official_bug is True
    assert config.strict_patient_isolation is False


def test_config_is_immutable():
    config = RAGDatasetBuilderConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.test_short = True


def test_rag_dataset_row_is_immutable():
    row = RAGDatasetRow(
        query_key=("mimic-cxr", "p1", "s1"),
        retrieved_key=None,
        image_path="/img.png",
        retrieved_report_text=None,
        target_report_text="t",
        rank_selected=None,
        num_candidates_rejected_self_study=0,
        num_candidates_rejected_self_patient=0,
        num_candidates_rejected_short_text=0,
        fallback_used=False,
        excluded=True,
        strict_patient_isolation=True,
        reproduce_official_bug=False,
        config_version="1.0",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.excluded = False


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_construction_rejects_query_record_missing_image_path():
    query_records = {("mimic-cxr", "p1", "s1"): {"dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1", "finding": "f"}}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, {}, {}, config=RAGDatasetBuilderConfig())


def test_construction_rejects_query_record_missing_output_data_mode_field():
    query_records = {("mimic-cxr", "p1", "s1"): {"dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1", "image_path": "/i.png"}}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, {}, {}, config=RAGDatasetBuilderConfig())


def test_construction_rejects_corpus_record_missing_rag_data_mode_field():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("mimic-cxr", "p2", "s2"): {"dataset": "mimic-cxr", "patient_id": "p2", "study_id": "s2"}}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, corpus_records, {}, config=RAGDatasetBuilderConfig())


def test_construction_rejects_ranking_referencing_unknown_query():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("mimic-cxr", "p2", "s2"): _record("mimic-cxr", "p2", "s2", "c finding")}
    knn_rankings = {("mimic-cxr", "p9", "s9"): [("mimic-cxr", "p2", "s2")]}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, corpus_records, knn_rankings, config=RAGDatasetBuilderConfig())


def test_construction_rejects_ranking_referencing_unknown_candidate():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("mimic-cxr", "p2", "s2"): _record("mimic-cxr", "p2", "s2", "c finding")}
    knn_rankings = {("mimic-cxr", "p1", "s1"): [("mimic-cxr", "p9", "s9")]}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, corpus_records, knn_rankings, config=RAGDatasetBuilderConfig())


def test_construction_rejects_cross_dataset_candidate_in_ranking():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("chexpert", "p2", "s2"): _record("chexpert", "p2", "s2", "c finding")}
    knn_rankings = {("mimic-cxr", "p1", "s1"): [("chexpert", "p2", "s2")]}
    with pytest.raises(RAGDatasetError):
        RAGDatasetBuilder(query_records, corpus_records, knn_rankings, config=RAGDatasetBuilderConfig())


def test_construction_preflight_rejects_cross_split_patient_leak():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    # Same patient_id p1 leaking into the (supposedly disjoint) train corpus.
    corpus_records = {("mimic-cxr", "p1", "s2"): _record("mimic-cxr", "p1", "s2", "c finding")}
    with pytest.raises(PatientLeakageError):
        RAGDatasetBuilder(
            query_records, corpus_records, {},
            config=RAGDatasetBuilderConfig(),
            query_split_name="test",
            corpus_split_name="train",
        )


def test_construction_preflight_skipped_when_splits_equal():
    # query_split_name == corpus_split_name (train-mode building):
    # overlapping patients between query/corpus sets is expected and
    # must NOT raise -- row-level filters handle it instead.
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("mimic-cxr", "p1", "s2"): _record("mimic-cxr", "p1", "s2", "c finding")}
    builder = RAGDatasetBuilder(
        query_records, corpus_records, {},
        config=RAGDatasetBuilderConfig(),
        query_split_name="train",
        corpus_split_name="train",
    )
    assert builder is not None


def test_construction_preflight_passes_for_disjoint_patients():
    query_records = {("mimic-cxr", "p1", "s1"): _record("mimic-cxr", "p1", "s1", "q finding")}
    corpus_records = {("mimic-cxr", "p2", "s2"): _record("mimic-cxr", "p2", "s2", "c finding")}
    builder = RAGDatasetBuilder(
        query_records, corpus_records, {},
        config=RAGDatasetBuilderConfig(),
        query_split_name="test",
        corpus_split_name="train",
    )
    assert builder is not None


# ---------------------------------------------------------------------------
# build_row -- selection / rejection semantics
# ---------------------------------------------------------------------------


def _default_builder(query_records, corpus_records, knn_rankings, **config_overrides):
    return RAGDatasetBuilder(
        query_records, corpus_records, knn_rankings,
        config=RAGDatasetBuilderConfig(**config_overrides),
    )


def test_build_row_selects_first_clean_candidate():
    q = ("mimic-cxr", "p1", "s1")
    c1 = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {c1: _record(*c1, "retrieved evidence text")}
    knn_rankings = {q: [c1]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    row = builder.build_row(q)

    assert row.excluded is False
    assert row.retrieved_key == c1
    assert row.rank_selected == 0
    assert row.retrieved_report_text == "retrieved evidence text"
    assert row.target_report_text == "target text"
    assert row.image_path == query_records[q]["image_path"]
    assert row.fallback_used is False
    assert row.num_candidates_rejected_self_study == 0
    assert row.num_candidates_rejected_self_patient == 0
    assert row.num_candidates_rejected_short_text == 0


def test_build_row_rejects_self_study_even_at_rank_0():
    q = ("mimic-cxr", "p1", "s1")
    clean = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {q: _record(*q, "self text"), clean: _record(*clean, "clean text")}
    # Query's own record appears at rank 0 of its own ranking (a real
    # KNN search legitimately returns the query itself as its own
    # nearest neighbor).
    knn_rankings = {q: [q, clean]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    row = builder.build_row(q)

    assert row.retrieved_key == clean
    assert row.rank_selected == 1
    assert row.num_candidates_rejected_self_study == 1
    assert row.num_candidates_rejected_self_patient == 0


def test_build_row_rejects_self_patient_different_study_even_at_rank_0():
    q = ("mimic-cxr", "p1", "s1")
    same_patient = ("mimic-cxr", "p1", "s2")
    clean = ("mimic-cxr", "p2", "s3")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {
        same_patient: _record(*same_patient, "same patient text"),
        clean: _record(*clean, "clean text"),
    }
    knn_rankings = {q: [same_patient, clean]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    row = builder.build_row(q)

    assert row.retrieved_key == clean
    assert row.rank_selected == 1
    assert row.num_candidates_rejected_self_study == 0
    assert row.num_candidates_rejected_self_patient == 1


def test_build_row_short_text_filter_only_applied_when_test_short_true():
    q = ("mimic-cxr", "p1", "s1")
    short_candidate = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {short_candidate: _record(*short_candidate, "one two")}  # 2 words < 5
    knn_rankings = {q: [short_candidate]}

    builder_off = _default_builder(query_records, corpus_records, knn_rankings, test_short=False)
    row_off = builder_off.build_row(q)
    assert row_off.retrieved_key == short_candidate
    assert row_off.num_candidates_rejected_short_text == 0

    builder_on = _default_builder(query_records, corpus_records, knn_rankings, test_short=True)
    row_on = builder_on.build_row(q)
    assert row_on.excluded is True
    assert row_on.num_candidates_rejected_short_text == 1


def test_build_row_short_text_boundary_exactly_min_words_passes():
    q = ("mimic-cxr", "p1", "s1")
    candidate = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {candidate: _record(*candidate, "one two three four five")}  # exactly 5 words
    knn_rankings = {q: [candidate]}
    builder = _default_builder(query_records, corpus_records, knn_rankings, test_short=True, min_short_words=5)

    row = builder.build_row(q)

    assert row.excluded is False
    assert row.retrieved_key == candidate
    assert row.num_candidates_rejected_short_text == 0


def test_build_row_short_text_boundary_one_under_min_words_rejected():
    q = ("mimic-cxr", "p1", "s1")
    candidate = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {candidate: _record(*candidate, "one two three four")}  # 4 words
    knn_rankings = {q: [candidate]}
    builder = _default_builder(query_records, corpus_records, knn_rankings, test_short=True, min_short_words=5)

    row = builder.build_row(q)

    assert row.excluded is True
    assert row.num_candidates_rejected_short_text == 1


def test_build_row_no_candidates_at_all_is_excluded_under_strict_isolation():
    q = ("mimic-cxr", "p1", "s1")
    query_records = {q: _record(*q, "target text")}
    builder = _default_builder(query_records, {}, {})

    row = builder.build_row(q)

    assert row.excluded is True
    assert row.retrieved_key is None
    assert row.retrieved_report_text is None
    assert row.rank_selected is None
    assert row.fallback_used is False


def test_build_row_exhaustion_excluded_under_strict_isolation_never_leaks():
    q = ("mimic-cxr", "p1", "s1")
    only_self_patient = ("mimic-cxr", "p1", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {only_self_patient: _record(*only_self_patient, "leaky text")}
    knn_rankings = {q: [only_self_patient]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    row = builder.build_row(q)

    assert row.excluded is True
    assert row.retrieved_key is None
    assert row.fallback_used is False
    assert row.num_candidates_rejected_self_patient == 1


def test_build_row_exhaustion_falls_back_to_rank_0_under_reproduce_official_bug():
    q = ("mimic-cxr", "p1", "s1")
    only_self_patient = ("mimic-cxr", "p1", "s2")
    other_leaky = ("mimic-cxr", "p1", "s3")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {
        only_self_patient: _record(*only_self_patient, "leaky text rank0"),
        other_leaky: _record(*other_leaky, "leaky text rank1"),
    }
    knn_rankings = {q: [only_self_patient, other_leaky]}
    builder = _default_builder(
        query_records, corpus_records, knn_rankings,
        strict_patient_isolation=False, reproduce_official_bug=True,
    )

    row = builder.build_row(q)

    assert row.excluded is False
    assert row.fallback_used is True
    assert row.retrieved_key == only_self_patient  # rank-0 candidate, regardless of leakage
    assert row.rank_selected == 0
    assert row.retrieved_report_text == "leaky text rank0"
    assert row.num_candidates_rejected_self_patient == 2  # both were rejected, then fell back anyway


def test_build_row_official_bug_mode_still_prefers_a_clean_candidate_when_one_exists():
    # reproduce_official_bug only changes exhaustion behavior -- a
    # clean candidate earlier in the ranking is still preferred over
    # ever falling back, exactly like the official code (which only
    # falls back when the loop finds nothing).
    q = ("mimic-cxr", "p1", "s1")
    leaky = ("mimic-cxr", "p1", "s2")
    clean = ("mimic-cxr", "p2", "s3")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {leaky: _record(*leaky, "leaky"), clean: _record(*clean, "clean")}
    knn_rankings = {q: [leaky, clean]}
    builder = _default_builder(
        query_records, corpus_records, knn_rankings,
        strict_patient_isolation=False, reproduce_official_bug=True,
    )

    row = builder.build_row(q)

    assert row.excluded is False
    assert row.fallback_used is False
    assert row.retrieved_key == clean
    assert row.rank_selected == 1


def test_build_row_strict_and_official_bug_modes_never_conflated_on_same_input():
    q = ("mimic-cxr", "p1", "s1")
    only_self_patient = ("mimic-cxr", "p1", "s2")
    query_records = {q: _record(*q, "target text")}
    corpus_records = {only_self_patient: _record(*only_self_patient, "leaky text")}
    knn_rankings = {q: [only_self_patient]}

    strict_builder = _default_builder(query_records, corpus_records, knn_rankings)
    bug_builder = _default_builder(
        query_records, corpus_records, knn_rankings,
        strict_patient_isolation=False, reproduce_official_bug=True,
    )

    strict_row = strict_builder.build_row(q)
    bug_row = bug_builder.build_row(q)

    assert strict_row.excluded is True and strict_row.fallback_used is False
    assert bug_row.excluded is False and bug_row.fallback_used is True


def test_build_row_rejects_unknown_query_key():
    builder = _default_builder({}, {}, {})
    with pytest.raises(RAGDatasetError):
        builder.build_row(("mimic-cxr", "p9", "s9"))


def test_build_row_uses_configured_rag_data_mode_and_output_data_mode():
    q = ("mimic-cxr", "p1", "s1")
    c = ("mimic-cxr", "p2", "s2")
    query_records = {q: _record(*q, "finding text", impression="impression text")}
    corpus_records = {c: _record(*c, "finding text c", impression="impression text c")}
    knn_rankings = {q: [c]}
    builder = _default_builder(
        query_records, corpus_records, knn_rankings,
        rag_data_mode="impression", output_data_mode="impression",
    )

    row = builder.build_row(q)

    assert row.retrieved_report_text == "impression text c"
    assert row.target_report_text == "impression text"


# ---------------------------------------------------------------------------
# build_all -- resume / atomic / continue_on_error behavior
# ---------------------------------------------------------------------------


def test_build_all_writes_one_row_per_query_and_summary_counts(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    q2 = ("mimic-cxr", "p2", "s2")
    clean = ("mimic-cxr", "p3", "s3")
    query_records = {q1: _record(*q1, "t1"), q2: _record(*q2, "t2")}
    corpus_records = {clean: _record(*clean, "c")}
    knn_rankings = {q1: [clean], q2: []}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    output_path = tmp_path / "rag_dataset.jsonl"
    summary = builder.build_all([q1, q2], output_path=output_path)

    assert summary.processed == 2
    assert summary.skipped_already_done == 0
    assert summary.failed == 0
    assert summary.excluded == 1  # q2 has no candidates -> excluded
    assert summary.fallback_used == 0

    lines = [json.loads(ln) for ln in output_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    for field in ("dataset", "patient_id", "study_id", "retrieved_key", "image_path",
                  "retrieved_report_text", "target_report_text", "rank_selected",
                  "num_candidates_rejected_self_study", "num_candidates_rejected_self_patient",
                  "num_candidates_rejected_short_text", "fallback_used", "excluded",
                  "strict_patient_isolation", "reproduce_official_bug", "config_version"):
        assert field in lines[0]


def test_build_all_resume_skips_already_completed_queries(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    q2 = ("mimic-cxr", "p2", "s2")
    clean = ("mimic-cxr", "p3", "s3")
    query_records = {q1: _record(*q1, "t1"), q2: _record(*q2, "t2")}
    corpus_records = {clean: _record(*clean, "c")}
    knn_rankings = {q1: [clean], q2: [clean]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    output_path = tmp_path / "rag_dataset.jsonl"
    first = builder.build_all([q1, q2], output_path=output_path)
    assert first.processed == 2
    assert first.skipped_already_done == 0

    second = builder.build_all([q1, q2], output_path=output_path)
    assert second.processed == 0
    assert second.skipped_already_done == 2

    lines = [ln for ln in output_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2  # no duplicate lines written


def test_build_all_rejects_resume_with_incompatible_config(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    clean = ("mimic-cxr", "p3", "s3")
    query_records = {q1: _record(*q1, "t1")}
    corpus_records = {clean: _record(*clean, "c")}
    knn_rankings = {q1: [clean]}

    builder_a = _default_builder(query_records, corpus_records, knn_rankings, min_short_words=5)
    output_path = tmp_path / "rag_dataset.jsonl"
    builder_a.build_all([q1], output_path=output_path)

    builder_b = _default_builder(query_records, corpus_records, knn_rankings, min_short_words=7)
    with pytest.raises(RAGDatasetError):
        builder_b.build_all([q1], output_path=output_path)


def test_build_all_continue_on_error_requires_errors_path(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    query_records = {q1: _record(*q1, "t1")}
    builder = _default_builder(query_records, {}, {})
    with pytest.raises(ValueError):
        builder.build_all([q1], output_path=tmp_path / "out.jsonl", continue_on_error=True)


def test_build_all_meta_records_run_status_and_defaults(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    clean = ("mimic-cxr", "p3", "s3")
    query_records = {q1: _record(*q1, "t1")}
    corpus_records = {clean: _record(*clean, "c")}
    knn_rankings = {q1: [clean]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    output_path = tmp_path / "rag_dataset.jsonl"
    builder.build_all([q1], output_path=output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["run_status"] == "completed"
    assert meta["query_count"] == 1
    assert meta["corpus_size"] == 1
    assert meta["defaults"]["actual_used"] == RAGDatasetBuilderConfig().as_actual_used_dict()
    assert meta["defaults"]["paper"]["min_short_length_unit"] == "characters"
    assert meta["defaults"]["official_repository"]["min_short_length_unit"] == "words"
    assert meta["summary"] == {
        "processed": 1, "skipped_already_done": 0, "failed": 0, "excluded": 0, "fallback_used": 0,
    }


def test_load_completed_rejects_malformed_json(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    query_records = {q1: _record(*q1, "t1")}
    builder = _default_builder(query_records, {}, {})
    output_path = tmp_path / "rag_dataset.jsonl"
    output_path.write_text("not json\n", encoding="utf-8")

    with pytest.raises(RAGDatasetError):
        builder.build_all([q1], output_path=output_path)


def test_load_completed_rejects_excluded_row_with_retrieved_key_present(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    query_records = {q1: _record(*q1, "t1")}
    builder = _default_builder(query_records, {}, {})
    output_path = tmp_path / "rag_dataset.jsonl"
    bad_row = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "retrieved_key": ["mimic-cxr", "p2", "s2"], "image_path": "/i.png",
        "retrieved_report_text": "x", "target_report_text": "t1", "rank_selected": 0,
        "num_candidates_rejected_self_study": 0, "num_candidates_rejected_self_patient": 0,
        "num_candidates_rejected_short_text": 0, "fallback_used": False,
        "excluded": True,  # inconsistent: excluded=True but retrieved_key present
        "strict_patient_isolation": True, "reproduce_official_bug": False, "config_version": "1.0",
    }
    output_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    with pytest.raises(RAGDatasetError):
        builder.build_all([q1], output_path=output_path)


def test_load_completed_rejects_fallback_used_incompatible_with_current_config(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    c1 = ("mimic-cxr", "p1", "s2")
    query_records = {q1: _record(*q1, "t1")}
    corpus_records = {c1: _record(*c1, "leaky")}
    output_path = tmp_path / "rag_dataset.jsonl"
    row = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "retrieved_key": ["mimic-cxr", "p1", "s2"], "image_path": "/i.png",
        "retrieved_report_text": "leaky", "target_report_text": "t1", "rank_selected": 0,
        "num_candidates_rejected_self_study": 0, "num_candidates_rejected_self_patient": 0,
        "num_candidates_rejected_short_text": 0, "fallback_used": True,
        "excluded": False,
        "strict_patient_isolation": False, "reproduce_official_bug": True, "config_version": "1.0",
    }
    output_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    # Current builder uses the default strict-isolation config (never
    # produces fallback_used=True) -- but since meta.json doesn't exist
    # yet, the config-compatibility gate is bypassed and this exercises
    # the per-line fallback_used/config consistency check directly.
    builder = _default_builder(query_records, corpus_records, {q1: [c1]})
    with pytest.raises(RAGDatasetError):
        builder.build_all([q1], output_path=output_path)


def test_build_all_continue_on_error_logs_and_continues(tmp_path):
    q1 = ("mimic-cxr", "p1", "s1")
    q2 = ("mimic-cxr", "p2", "s2")
    clean = ("mimic-cxr", "p3", "s3")
    query_records = {q1: _record(*q1, "t1"), q2: _record(*q2, "t2")}
    corpus_records = {clean: _record(*clean, "c")}
    knn_rankings = {q1: [clean], q2: [clean]}
    builder = _default_builder(query_records, corpus_records, knn_rankings)

    output_path = tmp_path / "rag_dataset.jsonl"
    errors_path = tmp_path / "errors.jsonl"

    # Pre-seed output with a row for q1 under an inconsistent shape so
    # the resume-time per-line validation raises for q1 specifically,
    # while q2 (not yet in output) proceeds normally.
    bad_row = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "retrieved_key": None, "image_path": "/i.png",
        "retrieved_report_text": None, "target_report_text": "t1", "rank_selected": None,
        "num_candidates_rejected_self_study": 0, "num_candidates_rejected_self_patient": 0,
        "num_candidates_rejected_short_text": 0, "fallback_used": True,  # inconsistent with excluded=True
        "excluded": True,
        "strict_patient_isolation": True, "reproduce_official_bug": False, "config_version": "1.0",
    }
    output_path.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")

    # The malformed pre-existing line is discovered during the resume
    # scan itself (before the per-query loop even starts), so it must
    # raise regardless of continue_on_error -- continue_on_error only
    # governs exceptions raised while building a *new* row.
    with pytest.raises(RAGDatasetError):
        builder.build_all([q1, q2], output_path=output_path, errors_path=errors_path, continue_on_error=True)
