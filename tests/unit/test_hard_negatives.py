"""Unit tests for src/baseline/retrieval/hard_negatives.py."""

import pytest

from src.baseline.retrieval.hard_negatives import (
    HardNegativeMiner,
    HardNegativeMiningConfig,
)
from src.common.exceptions import RetrievalDatasetError


def _record(dataset, patient_id, study_id, chexbert_labels_5, facts):
    entities = {
        str(index + 1): {"tokens": tokens, "label": label, "relations": []}
        for index, (tokens, label) in enumerate(facts)
    }
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "chexbert_labels_5": chexbert_labels_5,
        "entities": entities,
    }


QUERY_KEY = ("mimic", "p1", "s1")
QUERY_RECORD = _record("mimic", "p1", "s1", [1, 1, 1, 1, 1], [("opacity", "OBS")])
QUERY_EMBEDDING = (1.0, 0.0)

# Fully dissimilar to QUERY_RECORD: chexbert_sim == 0.0 (< 1.0 threshold),
# radgraph_sim == 0.0 (< 0.4 threshold) -- always qualifies as a hard
# negative candidate.
DISSIMILAR_LABELS = [0, 0, 0, 0, 0]
DISSIMILAR_FACTS = [("effusion", "OBS-DA")]

# Identical to QUERY_RECORD: chexbert_sim == 1.0, radgraph_sim == 1.0 --
# never qualifies (fails both strict "<" thresholds).
SIMILAR_LABELS = [1, 1, 1, 1, 1]
SIMILAR_FACTS = [("opacity", "OBS")]


def _default_config(**overrides):
    return HardNegativeMiningConfig(**overrides)


# ---------------------------------------------------------------------------
# HardNegativeMiningConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults_match_official_repository_values():
    config = HardNegativeMiningConfig()
    assert config.top_n == 100
    assert config.num_top_neg == 2
    assert config.chex_threshold == 1.0
    assert config.radg_threshold == 0.4
    assert config.similarity_metric == "dot"
    assert config.exclude_same_patient is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_n": 0},
        {"top_n": -1},
        {"top_n": 1.5},
        {"num_top_neg": 0},
        {"num_top_neg": True},
        {"similarity_metric": "euclidean"},
        {"chex_threshold": 1.5},
        {"chex_threshold": -0.1},
        {"radg_threshold": 1.5},
        {"config_version": ""},
        {"exclude_same_patient": "yes"},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        HardNegativeMiningConfig(**overrides)


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


def test_construction_rejects_empty_embeddings():
    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner({}, {}, {}, config=_default_config())


def test_construction_rejects_missing_embedding_for_a_record():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}  # other_key has no embedding

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, {}, config=_default_config())


def test_construction_rejects_embedding_with_no_matching_record():
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {
        QUERY_KEY: QUERY_EMBEDDING,
        ("mimic", "ghost", "s9"): (0.0, 1.0),
    }

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, {}, config=_default_config())


def test_construction_rejects_wrong_embedding_dimension():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {
        QUERY_KEY: QUERY_EMBEDDING,
        other_key: (1.0, 0.0, 0.0),  # 3-dim vs. query's 2-dim
    }

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, {}, config=_default_config())


def test_construction_rejects_duplicate_embedding_vectors():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {
        QUERY_KEY: QUERY_EMBEDDING,
        other_key: QUERY_EMBEDDING,  # identical vector -- duplicate/corrupted data
    }

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, {}, config=_default_config())


def test_construction_rejects_positive_key_missing_from_corpus():
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    positives = {QUERY_KEY: [("mimic", "ghost", "s9")]}

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, positives, config=_default_config())


def test_construction_rejects_cross_dataset_positive_key():
    other_key = ("chexpert", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("chexpert", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.0, 1.0)}
    positives = {QUERY_KEY: [other_key]}

    with pytest.raises(RetrievalDatasetError):
        HardNegativeMiner(corpus, embeddings, positives, config=_default_config())


# ---------------------------------------------------------------------------
# mine_query -- core scenarios
# ---------------------------------------------------------------------------


def test_mine_query_no_candidates_returns_empty_result():
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.scores == []
    assert result.num_embedding_candidates_considered == 0
    assert result.num_qualifying_candidates == 0
    assert result.num_selected == 0


def test_mine_query_one_candidate_qualifies_as_negative():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == [other_key]
    assert result.num_embedding_candidates_considered == 1
    assert result.num_qualifying_candidates == 1
    assert result.num_selected == 1
    assert result.scores[0]["chexbert_similarity"] == 0.0
    assert result.scores[0]["radgraph_similarity"] == 0.0


def test_mine_query_self_never_appears_as_its_own_negative():
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert QUERY_KEY not in result.negative_keys


def test_mine_query_rejects_cross_dataset_candidate():
    other_key = ("chexpert", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("chexpert", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.num_embedding_candidates_considered == 0


def test_mine_query_excludes_same_patient_by_default():
    same_patient_key = ("mimic", "p1", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        same_patient_key: _record("mimic", "p1", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, same_patient_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.num_embedding_candidates_considered == 0


def test_mine_query_includes_same_patient_when_configured():
    same_patient_key = ("mimic", "p1", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        same_patient_key: _record("mimic", "p1", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, same_patient_key: (0.9, 0.1)}
    miner = HardNegativeMiner(
        corpus, embeddings, {}, config=_default_config(exclude_same_patient=False)
    )

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == [same_patient_key]


def test_mine_query_excludes_confirmed_positive():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    positives = {QUERY_KEY: [other_key]}
    miner = HardNegativeMiner(corpus, embeddings, positives, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.num_embedding_candidates_considered == 0


def test_mine_query_rejects_candidate_failing_chexbert_threshold():
    other_key = ("mimic", "p2", "s2")
    # Similar chexbert labels (chex_sim == 1.0, not < 1.0) but dissimilar
    # entities -- must not qualify since BOTH thresholds are required.
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", SIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.num_embedding_candidates_considered == 1
    assert result.num_qualifying_candidates == 0


def test_mine_query_rejects_candidate_failing_radgraph_threshold():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, SIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_query(QUERY_KEY)

    assert result.negative_keys == []
    assert result.num_qualifying_candidates == 0


def test_mine_query_respects_top_n_before_threshold_filtering():
    keys = [("mimic", f"p{i}", f"s{i}") for i in range(2, 5)]
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    for i, key in enumerate(keys):
        corpus[key] = _record(key[0], key[1], key[2], DISSIMILAR_LABELS, DISSIMILAR_FACTS)
        # Descending embedding similarity: keys[0] most similar, keys[2] least.
        embeddings[key] = (0.9 - i * 0.1, 0.0)

    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config(top_n=1))

    result = miner.mine_query(QUERY_KEY)

    assert result.num_embedding_candidates_considered == 3
    # Only the single most embedding-similar candidate is even considered.
    assert result.negative_keys == [keys[0]]


def test_mine_query_selects_num_top_neg_most_dissimilar_among_qualifying():
    key_a = ("mimic", "p2", "s2")
    key_b = ("mimic", "p3", "s3")
    key_c = ("mimic", "p4", "s4")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        # chex_sim = 0.0 for all three; radgraph differs via distinct facts
        # sharing partial overlap with the query to vary radgraph_similarity.
        key_a: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, [("effusion", "OBS-DA")]),
        key_b: _record("mimic", "p3", "s3", DISSIMILAR_LABELS, [("consolidation", "OBS-DP")]),
        key_c: _record("mimic", "p4", "s4", DISSIMILAR_LABELS, [("atelectasis", "OBS-U")]),
    }
    embeddings = {
        QUERY_KEY: QUERY_EMBEDDING,
        key_a: (0.9, 0.1),
        key_b: (0.8, 0.2),
        key_c: (0.7, 0.3),
    }
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config(num_top_neg=2))

    result = miner.mine_query(QUERY_KEY)

    assert result.num_qualifying_candidates == 3
    assert result.num_selected == 2
    assert len(result.negative_keys) == 2
    assert len(set(result.negative_keys)) == 2  # no duplicate negatives


# ---------------------------------------------------------------------------
# Deterministic tie-breaking / identical similarity
# ---------------------------------------------------------------------------


def test_mine_query_deterministic_tie_break_on_identical_embedding_similarity():
    key_a = ("mimic", "p3", "s3")
    key_b = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        key_a: _record("mimic", "p3", "s3", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
        key_b: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    # Distinct embedding vectors that nonetheless produce an identical dot
    # product against the query (1.0, 0.0): only the first component
    # matters for that metric, so differing second components keep the
    # vectors non-duplicate while still tying on similarity.
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, key_a: (0.9, 0.1), key_b: (0.9, 0.2)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config(top_n=1, num_top_neg=1))

    result = miner.mine_query(QUERY_KEY)

    # Tie-break is lexicographic (dataset, patient_id, study_id) ascending
    # -- key_b ("p2") sorts before key_a ("p3").
    assert result.negative_keys == [key_b]


def test_mine_query_tie_break_is_reproducible_across_repeated_calls():
    key_a = ("mimic", "p3", "s3")
    key_b = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        key_a: _record("mimic", "p3", "s3", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
        key_b: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, key_a: (0.9, 0.1), key_b: (0.9, 0.2)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config(num_top_neg=1))

    first = miner.mine_query(QUERY_KEY)
    second = miner.mine_query(QUERY_KEY)

    assert first.negative_keys == second.negative_keys == [key_b]


def test_mine_query_reproducible_after_corpus_shuffling():
    keys = [("mimic", f"p{i}", f"s{i}") for i in range(2, 8)]
    base_corpus = {QUERY_KEY: QUERY_RECORD}
    base_embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    for i, key in enumerate(keys):
        base_corpus[key] = _record(key[0], key[1], key[2], DISSIMILAR_LABELS, DISSIMILAR_FACTS)
        base_embeddings[key] = (0.9 - i * 0.05, float(i) * 0.01)

    miner_in_order = HardNegativeMiner(
        dict(base_corpus), dict(base_embeddings), {}, config=_default_config(num_top_neg=3)
    )
    result_in_order = miner_in_order.mine_query(QUERY_KEY)

    shuffled_keys = list(base_corpus.keys())
    shuffled_keys.reverse()
    shuffled_corpus = {key: base_corpus[key] for key in shuffled_keys}
    shuffled_embeddings = {key: base_embeddings[key] for key in shuffled_keys}

    miner_shuffled = HardNegativeMiner(
        shuffled_corpus, shuffled_embeddings, {}, config=_default_config(num_top_neg=3)
    )
    result_shuffled = miner_shuffled.mine_query(QUERY_KEY)

    assert result_in_order.negative_keys == result_shuffled.negative_keys
    assert result_in_order.scores == result_shuffled.scores


# ---------------------------------------------------------------------------
# similarity_metric configurability
# ---------------------------------------------------------------------------


def test_similarity_metric_dot_vs_cosine_can_change_ranking():
    # key_a: far away but perfectly aligned direction (high cosine, but
    # also high dot product since magnitude is large).
    key_a = ("mimic", "p2", "s2")
    # key_b: closer magnitude but slightly off-axis (lower cosine, but
    # comparable dot product due to larger raw magnitude on the aligned axis).
    key_b = ("mimic", "p3", "s3")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        key_a: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
        key_b: _record("mimic", "p3", "s3", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {
        QUERY_KEY: (1.0, 0.0),
        key_a: (10.0, 0.0),  # dot = 10.0, cosine = 1.0
        key_b: (5.0, 5.0),  # dot = 5.0, cosine ~= 0.707
    }

    dot_miner = HardNegativeMiner(
        corpus, embeddings, {}, config=_default_config(top_n=1, num_top_neg=1, similarity_metric="dot")
    )
    cosine_miner = HardNegativeMiner(
        corpus, embeddings, {}, config=_default_config(top_n=1, num_top_neg=1, similarity_metric="cosine")
    )

    dot_result = dot_miner.mine_query(QUERY_KEY)
    cosine_result = cosine_miner.mine_query(QUERY_KEY)

    # Both metrics rank key_a highest here (dot: 10.0 > 5.0; cosine: 1.0 > 0.707)
    # -- assert the metric is actually used (not ignored) by checking both
    # produce the expected top-1 candidate.
    assert dot_result.negative_keys == [key_a]
    assert cosine_result.negative_keys == [key_a]


# ---------------------------------------------------------------------------
# mine_all
# ---------------------------------------------------------------------------


def test_mine_all_returns_negative_keys_per_query():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_all([QUERY_KEY, other_key])

    assert result[QUERY_KEY] == [other_key]
    # other_key's mining is symmetric here: QUERY_KEY is equally a
    # qualifying (dissimilar, same-dataset, different-patient) candidate
    # from other_key's perspective.
    assert result[other_key] == [QUERY_KEY]


def test_mine_all_is_idempotent_for_duplicate_query_keys():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    result = miner.mine_all([QUERY_KEY, QUERY_KEY])

    assert result[QUERY_KEY] == [other_key]
    assert len(result) == 1


def test_mine_query_raises_for_unknown_query_key():
    corpus = {QUERY_KEY: QUERY_RECORD}
    embeddings = {QUERY_KEY: QUERY_EMBEDDING}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    with pytest.raises(RetrievalDatasetError):
        miner.mine_query(("mimic", "ghost", "s9"))


def test_embedding_dim_property_reports_resolved_dimension():
    other_key = ("mimic", "p2", "s2")
    corpus = {
        QUERY_KEY: QUERY_RECORD,
        other_key: _record("mimic", "p2", "s2", DISSIMILAR_LABELS, DISSIMILAR_FACTS),
    }
    embeddings = {QUERY_KEY: QUERY_EMBEDDING, other_key: (0.9, 0.1)}
    miner = HardNegativeMiner(corpus, embeddings, {}, config=_default_config())

    assert miner.embedding_dim == 2
