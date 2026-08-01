"""Unit tests for src/baseline/pair_mining/mining.py."""

import json

import pytest

from src.baseline.pair_mining.mining import PairMiner, PairMiningConfig
from src.baseline.pair_mining.similarity import radgraph_similarity
from src.common.exceptions import PairMiningError


def _entity(tokens, label, relations=None):
    return {"tokens": tokens, "label": label, "relations": relations or []}


def _record(dataset, patient_id, study_id, entities, labels_5):
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "finding": "synthetic",
        "entities": entities,
        "chexbert_labels_14": [0] * 14,
        "chexbert_labels_5": labels_5,
    }


# --- PairMiningConfig validation ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chex_threshold": 1.5},
        {"chex_threshold": -0.1},
        {"radg_threshold": -0.1},
        {"radg_threshold": 1.1},
        {"top_k": 0},
        {"top_k": -1},
        {"threshold_comparison": "=="},
        {"config_version": ""},
    ],
)
def test_pair_mining_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PairMiningConfig(**kwargs)


def test_pair_mining_config_defaults_are_valid():
    PairMiningConfig()  # must not raise


# --- threshold comparison semantics (">=" vs ">") ---


def test_threshold_chex_boundary_inclusive_vs_exclusive():
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    # Exact label match: chexbert_similarity == 1.0 == chex_threshold.
    candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate]

    config_ge = PairMiningConfig(chex_threshold=1.0, radg_threshold=0.0, threshold_comparison=">=")
    result_ge = PairMiner(corpus, config=config_ge).mine_query(query)
    assert result_ge.positive_keys == [("mimic-cxr", "p2", "s2")]

    config_gt = PairMiningConfig(chex_threshold=1.0, radg_threshold=0.0, threshold_comparison=">")
    result_gt = PairMiner(corpus, config=config_gt).mine_query(query)
    assert result_gt.positive_keys == []


def test_threshold_radgraph_boundary_inclusive_vs_exclusive():
    query_entities = {f"e{i}": _entity(f"f{i}", "L") for i in range(1, 6)}
    candidate_entities = {
        "e1": _entity("f1", "L"),
        "e2": _entity("f2", "L"),
        "e3": _entity("g1", "L"),
        "e4": _entity("g2", "L"),
        "e5": _entity("g3", "L"),
    }
    # intersection = 2 facts, |a| = |b| = 5 -> precision = recall = 0.4 -> F1 = 0.4
    radg_sim_at_boundary = radgraph_similarity(query_entities, candidate_entities)
    assert radg_sim_at_boundary == pytest.approx(0.4)

    query = _record("mimic-cxr", "p1", "s1", query_entities, [1, 0, 0, 0, 0])
    candidate = _record("mimic-cxr", "p2", "s2", candidate_entities, [1, 0, 0, 0, 0])
    corpus = [query, candidate]

    # Threshold against the exact computed float (not the literal 0.4,
    # which can differ from the computed value by a floating-point
    # rounding epsilon and make the ">" boundary check flaky).
    config_ge = PairMiningConfig(
        chex_threshold=0.0, radg_threshold=radg_sim_at_boundary, threshold_comparison=">="
    )
    result_ge = PairMiner(corpus, config=config_ge).mine_query(query)
    assert result_ge.positive_keys == [("mimic-cxr", "p2", "s2")]

    config_gt = PairMiningConfig(
        chex_threshold=0.0, radg_threshold=radg_sim_at_boundary, threshold_comparison=">"
    )
    result_gt = PairMiner(corpus, config=config_gt).mine_query(query)
    assert result_gt.positive_keys == []


# --- deterministic ranking ---


def test_deterministic_tie_break_and_shuffled_corpus_equivalence():
    query_entities = {"1": _entity("x", "L")}
    query = _record("mimic-cxr", "p0", "s0", query_entities, [1, 0, 0, 0, 0])
    candidate_c = _record("mimic-cxr", "pC", "sC", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    candidate_a = _record("mimic-cxr", "pA", "sA", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    candidate_b = _record("mimic-cxr", "pB", "sB", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate_c, candidate_a, candidate_b]
    config = PairMiningConfig(top_k=2)
    expected = [("mimic-cxr", "pA", "sA"), ("mimic-cxr", "pB", "sB")]

    result_in_order = PairMiner(corpus, config=config).mine_query(query)
    assert result_in_order.positive_keys == expected

    shuffled_corpus = [query, candidate_b, candidate_a, candidate_c]
    result_shuffled = PairMiner(shuffled_corpus, config=config).mine_query(query)
    assert result_shuffled.positive_keys == expected
    assert result_shuffled.scores == result_in_order.scores


# --- exact 5-label filtering ---


def test_exact_5_label_filtering_requires_full_vector_match():
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 1, 0, 0, 0])
    # Shares 4 of 5 positions (chex_sim=0.8) -- must NOT qualify at default chex_threshold=1.0
    near_match = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    exact_match = _record("mimic-cxr", "p3", "s3", {"1": _entity("x", "L")}, [1, 1, 0, 0, 0])
    corpus = [query, near_match, exact_match]

    result = PairMiner(corpus, config=PairMiningConfig()).mine_query(query)
    assert result.positive_keys == [("mimic-cxr", "p3", "s3")]


# --- bad-sample behavior ---


def test_bad_sample_flagged_when_self_radgraph_similarity_not_one():
    query = _record("mimic-cxr", "p1", "s1", {}, [1, 0, 0, 0, 0])  # empty entities
    candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate]

    result = PairMiner(corpus, config=PairMiningConfig()).mine_query(query)
    assert result.flagged_bad_sample is True
    assert result.positive_keys == []
    assert result.scores == []
    assert result.num_qualifying_candidates == 0
    assert result.num_selected == 0
    assert result.num_candidates_considered == 2


# --- fewer than k qualifying candidates ---


def test_fewer_than_k_qualifying_candidates_returns_partial_list():
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    only_candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, only_candidate]

    result = PairMiner(corpus, config=PairMiningConfig(top_k=3)).mine_query(query)
    assert result.num_selected == 1
    assert len(result.positive_keys) == 1


# --- cross-dataset exclusion ---


def test_cross_dataset_candidates_are_never_selected():
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    chexpert_twin = _record("chexpert", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, chexpert_twin]

    result = PairMiner(corpus, config=PairMiningConfig()).mine_query(query)
    assert result.positive_keys == []
    assert result.num_candidates_considered == 1


# --- same-patient default inclusion / optional exclusion ---


def test_same_patient_different_study_included_by_default():
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    same_patient_other_study = _record("mimic-cxr", "p1", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, same_patient_other_study]

    default_result = PairMiner(corpus, config=PairMiningConfig()).mine_query(query)
    assert default_result.positive_keys == [("mimic-cxr", "p1", "s2")]

    strict_config = PairMiningConfig(exclude_same_patient=True)
    strict_result = PairMiner(corpus, config=strict_config).mine_query(query)
    assert strict_result.positive_keys == []


# --- resume validation ---


def test_resume_skips_already_completed_queries(tmp_path):
    query1 = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    query2 = _record("mimic-cxr", "p2", "s2", {"1": _entity("y", "L")}, [0, 1, 0, 0, 0])
    candidate = _record("mimic-cxr", "p3", "s3", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query1, query2, candidate]
    output_path = tmp_path / "pairs.jsonl"
    config = PairMiningConfig()

    first = PairMiner(corpus, config=config).mine_all([query1, query2], output_path=output_path)
    assert first.processed == 2
    assert first.skipped_already_done == 0

    second = PairMiner(corpus, config=config).mine_all([query1, query2], output_path=output_path)
    assert second.processed == 0
    assert second.skipped_already_done == 2

    lines = [ln for ln in output_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_continue_on_error_requires_errors_path(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    miner = PairMiner([query], config=PairMiningConfig())
    with pytest.raises(ValueError):
        miner.mine_all([query], output_path=tmp_path / "pairs.jsonl", continue_on_error=True)


def test_resume_rejects_referenced_positive_key_not_in_corpus(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate]
    output_path = tmp_path / "pairs.jsonl"

    stale_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p999", "s999"]],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0}],
        "num_candidates_considered": 2, "num_qualifying_candidates": 1,
        "num_selected": 1, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(stale_line) + "\n", encoding="utf-8")

    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=PairMiningConfig()).mine_all([query], output_path=output_path)


def test_resume_rejects_positive_keys_scores_length_mismatch(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p2", "s2"]],
        "scores": [],
        "num_candidates_considered": 2, "num_qualifying_candidates": 1,
        "num_selected": 1, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=PairMiningConfig()).mine_all([query], output_path=output_path)


def test_resume_rejects_num_selected_mismatch(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    candidate = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, candidate]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p2", "s2"]],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0}],
        "num_candidates_considered": 2, "num_qualifying_candidates": 1,
        "num_selected": 2,
        "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=PairMiningConfig()).mine_all([query], output_path=output_path)


def test_resume_rejects_num_selected_exceeding_top_k(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    c1 = _record("mimic-cxr", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    c2 = _record("mimic-cxr", "p3", "s3", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, c1, c2]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p2", "s2"], ["mimic-cxr", "p3", "s3"]],
        "scores": [
            {"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0},
            {"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0},
        ],
        "num_candidates_considered": 3, "num_qualifying_candidates": 2,
        "num_selected": 2, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=PairMiningConfig(top_k=1)).mine_all([query], output_path=output_path)


def test_resume_rejects_cross_dataset_positive_key(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    chexpert_candidate = _record("chexpert", "p2", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, chexpert_candidate]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["chexpert", "p2", "s2"]],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0}],
        "num_candidates_considered": 1, "num_qualifying_candidates": 1,
        "num_selected": 1, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=PairMiningConfig()).mine_all([query], output_path=output_path)


def test_resume_rejects_self_key_when_exclude_self_true(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p1", "s1"]],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0}],
        "num_candidates_considered": 1, "num_qualifying_candidates": 1,
        "num_selected": 1, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    config = PairMiningConfig(exclude_self=True)
    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=config).mine_all([query], output_path=output_path)


def test_resume_rejects_same_patient_key_when_exclude_same_patient_true(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    same_patient = _record("mimic-cxr", "p1", "s2", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query, same_patient]
    output_path = tmp_path / "pairs.jsonl"

    bad_line = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "positive_keys": [["mimic-cxr", "p1", "s2"]],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 2.0}],
        "num_candidates_considered": 2, "num_qualifying_candidates": 1,
        "num_selected": 1, "flagged_bad_sample": False,
    }
    output_path.write_text(json.dumps(bad_line) + "\n", encoding="utf-8")

    config = PairMiningConfig(exclude_self=False, exclude_same_patient=True)
    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=config).mine_all([query], output_path=output_path)


def test_rejects_resume_with_incompatible_metadata_config(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    corpus = [query]
    output_path = tmp_path / "pairs.jsonl"
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")

    old_config = PairMiningConfig(top_k=5)
    PairMiner(corpus, config=old_config).mine_all([query], output_path=output_path, meta_path=meta_path)

    new_config = PairMiningConfig(top_k=3)
    with pytest.raises(PairMiningError):
        PairMiner(corpus, config=new_config).mine_all([query], output_path=output_path, meta_path=meta_path)


# --- metadata provenance ---


def test_metadata_records_paper_official_and_actual_defaults(tmp_path):
    query = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    output_path = tmp_path / "pairs.jsonl"
    config = PairMiningConfig(top_k=3, threshold_comparison=">=")
    PairMiner([query], config=config).mine_all([query], output_path=output_path)

    meta = json.loads(output_path.with_suffix(output_path.suffix + ".meta.json").read_text(encoding="utf-8"))
    assert meta["defaults"]["paper"] == {"top_k": 2, "threshold_comparison": ">"}
    assert meta["defaults"]["official_repository"]["top_k"] == 3
    assert meta["defaults"]["official_repository"]["threshold_comparison"] == ">="
    assert meta["defaults"]["actual_used"]["top_k"] == 3
    assert meta["defaults"]["actual_used"]["threshold_comparison"] == ">="
    assert meta["defaults"]["actual_used"]["config_version"] == "1.0"


def test_metadata_records_query_count_and_tie_break_rule(tmp_path):
    query1 = _record("mimic-cxr", "p1", "s1", {"1": _entity("x", "L")}, [1, 0, 0, 0, 0])
    query2 = _record("mimic-cxr", "p2", "s2", {"1": _entity("y", "L")}, [0, 1, 0, 0, 0])
    corpus = [query1, query2]
    output_path = tmp_path / "pairs.jsonl"

    PairMiner(corpus, config=PairMiningConfig(), corpus_scope="unit_test_corpus").mine_all(
        [query1, query2], output_path=output_path
    )

    meta = json.loads(output_path.with_suffix(output_path.suffix + ".meta.json").read_text(encoding="utf-8"))
    assert meta["query_count"] == 2
    assert meta["corpus_size"] == 2
    assert meta["corpus_scope"] == "unit_test_corpus"
    assert "combined_score" in meta["deterministic_tie_breaking_rule"]
    assert "argpartition" in meta["deterministic_tie_breaking_rule"]
