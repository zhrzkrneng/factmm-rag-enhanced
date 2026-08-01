"""Unit tests for src/baseline/pair_mining/similarity.py."""

import pytest

from src.baseline.pair_mining.similarity import (
    chexbert_similarity,
    combined_score,
    radgraph_similarity,
)


def _entity(tokens, label, relations=None):
    return {"tokens": tokens, "label": label, "relations": relations or []}


# --- chexbert_similarity ---


def test_chexbert_similarity_identical_labels_is_one():
    assert chexbert_similarity([1, 0, 1, 0, 0], [1, 0, 1, 0, 0]) == 1.0


def test_chexbert_similarity_all_different_is_zero():
    assert chexbert_similarity([1, 1, 1, 1, 1], [0, 0, 0, 0, 0]) == 0.0


def test_chexbert_similarity_partial_match():
    # 3 of 5 positions agree (indices 0, 1, 4)
    assert chexbert_similarity([1, 0, 1, 0, 1], [1, 0, 0, 1, 1]) == pytest.approx(3 / 5)


def test_chexbert_similarity_rejects_mismatched_length():
    with pytest.raises(ValueError):
        chexbert_similarity([1, 0], [1, 0, 1])


def test_chexbert_similarity_rejects_empty_vectors():
    with pytest.raises(ValueError):
        chexbert_similarity([], [])


# --- radgraph_similarity ---


def test_radgraph_similarity_identical_entities_is_one():
    entities = {"1": _entity("cardiomegaly", "OBS-DP")}
    assert radgraph_similarity(entities, entities) == 1.0


def test_radgraph_similarity_disjoint_entities_is_zero():
    a = {"1": _entity("cardiomegaly", "OBS-DP")}
    b = {"1": _entity("pneumonia", "OBS-DA")}
    assert radgraph_similarity(a, b) == 0.0


def test_radgraph_similarity_both_empty_is_zero_not_one():
    assert radgraph_similarity({}, {}) == 0.0


def test_radgraph_similarity_one_side_empty_is_zero():
    entities = {"1": _entity("cardiomegaly", "OBS-DP")}
    assert radgraph_similarity({}, entities) == 0.0
    assert radgraph_similarity(entities, {}) == 0.0


def test_radgraph_similarity_duplicate_facts_collapse_as_a_set():
    # Two entities with an identical (tokens, label, no-relations) fact
    # -- must not count as two facts relative to a single occurrence.
    duplicated = {
        "1": _entity("bilateral", "OBS-DP"),
        "2": _entity("bilateral", "OBS-DP"),
    }
    single = {"1": _entity("bilateral", "OBS-DP")}
    assert radgraph_similarity(duplicated, single) == 1.0
    assert radgraph_similarity(single, duplicated) == 1.0


def test_radgraph_similarity_encodes_relation_presence_not_relation_content():
    # Same tokens/label, both entities have *some* relation, but the
    # relation type/target differ -- official metric ignores that.
    a = {
        "1": _entity("acute", "OBS-DA", relations=[["modify", "2"]]),
        "2": _entity("process", "OBS-DA"),
    }
    b = {
        "1": _entity("acute", "OBS-DA", relations=[["located_at", "3"]]),
        "2": _entity("process", "OBS-DA"),
    }
    assert radgraph_similarity(a, b) == 1.0


def test_radgraph_similarity_relation_presence_vs_absence_is_a_different_fact():
    a = {"1": _entity("acute", "OBS-DA", relations=[["modify", "2"]])}
    b = {"1": _entity("acute", "OBS-DA")}  # same tokens/label, no relation
    assert radgraph_similarity(a, b) == 0.0


def test_radgraph_similarity_is_symmetric():
    a = {
        "1": _entity("cardiomegaly", "OBS-DP"),
        "2": _entity("mild", "OBS-DP", relations=[["modify", "1"]]),
    }
    b = {"1": _entity("cardiomegaly", "OBS-DP")}
    assert radgraph_similarity(a, b) == radgraph_similarity(b, a)


def test_radgraph_similarity_partial_overlap_matches_f1_formula():
    a = {"1": _entity("x", "L"), "2": _entity("y", "L")}
    b = {"1": _entity("x", "L"), "2": _entity("z", "L")}
    # intersection = {("x", "L")}; |a| = |b| = 2 -> precision = recall = 0.5 -> F1 = 0.5
    assert radgraph_similarity(a, b) == pytest.approx(0.5)


# --- combined_score ---


def test_combined_score_is_the_plain_sum():
    assert combined_score(1.0, 0.4) == pytest.approx(1.4)
    assert combined_score(0.0, 0.0) == 0.0
    assert combined_score(1.0, 1.0) == 2.0


def test_combined_score_matches_official_tensor_addition_pattern():
    # tensor = chexbert + radgraph, elementwise -- verified here for a
    # handful of representative (chex, radg) pairs.
    for chex, radg in [(1.0, 0.4), (0.0, 1.0), (0.6, 0.6)]:
        assert combined_score(chex, radg) == chex + radg
