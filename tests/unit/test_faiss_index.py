"""Unit tests for src/retrieval/index.py (FaissFlatIPIndex)."""

import json

import numpy as np
import pytest

from src.common.exceptions import RetrievalIndexError
from src.retrieval.index import FaissFlatIPIndex


def _unit(vector):
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def _keys(*names):
    return [("mimic", name, name) for name in names]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construction_rejects_invalid_embedding_dim():
    with pytest.raises(ValueError):
        FaissFlatIPIndex(0, normalized=True)
    with pytest.raises(ValueError):
        FaissFlatIPIndex(-1, normalized=True)


def test_new_index_has_zero_vectors():
    index = FaissFlatIPIndex(4, normalized=False)
    assert index.num_vectors == 0
    assert index.embedding_dim == 4


def test_similarity_type_reflects_normalized_flag():
    assert FaissFlatIPIndex(4, normalized=True).similarity_type == "cosine"
    assert FaissFlatIPIndex(4, normalized=False).similarity_type == "inner_product"


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------


def test_add_vectors_increments_num_vectors_and_preserves_key_order():
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keys = _keys("a", "b")

    index.add(embeddings, keys)

    assert index.num_vectors == 2


def test_add_rejects_wrong_dimension():
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    with pytest.raises(RetrievalIndexError):
        index.add(embeddings, _keys("a"))


def test_add_rejects_length_mismatch():
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(RetrievalIndexError):
        index.add(embeddings, _keys("a"))


def test_add_rejects_non_finite_embeddings():
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[float("nan"), 0.0]], dtype=np.float32)

    with pytest.raises(RetrievalIndexError):
        index.add(embeddings, _keys("a"))


def test_add_rejects_duplicate_key_within_call():
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    keys = _keys("a", "a")

    with pytest.raises(RetrievalIndexError):
        index.add(embeddings, keys)


def test_add_rejects_duplicate_key_across_calls():
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32), _keys("a"))

    with pytest.raises(RetrievalIndexError):
        index.add(np.array([[0.0, 1.0]], dtype=np.float32), _keys("a"))


def test_add_rejects_non_unit_norm_when_normalized_true():
    index = FaissFlatIPIndex(2, normalized=True)
    embeddings = np.array([[2.0, 0.0]], dtype=np.float32)  # norm == 2.0, not 1.0

    with pytest.raises(RetrievalIndexError):
        index.add(embeddings, _keys("a"))


def test_add_accepts_unit_norm_when_normalized_true():
    index = FaissFlatIPIndex(2, normalized=True)
    embeddings = np.stack([_unit([1.0, 0.5]), _unit([0.2, 1.0])])

    index.add(embeddings, _keys("a", "b"))

    assert index.num_vectors == 2


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------


def test_search_returns_top_k_in_similarity_order():
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(
        np.array([[1.0, 0.0], [0.5, 0.0], [0.0, 1.0]], dtype=np.float32),
        _keys("high", "mid", "low"),
    )

    scores, result_keys = index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=2)

    assert scores.shape == (1, 2)
    assert result_keys[0][0] == ("mimic", "high", "high")
    assert result_keys[0][1] == ("mimic", "mid", "mid")


def test_search_k_greater_than_corpus_size_clips_without_error():
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32), _keys("a", "b"))

    scores, result_keys = index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=100)

    assert scores.shape == (1, 2)
    assert len(result_keys[0]) == 2


def test_search_on_empty_index_returns_empty_results():
    index = FaissFlatIPIndex(2, normalized=False)

    scores, result_keys = index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=5)

    assert scores.shape == (1, 0)
    assert result_keys == [[]]


def test_search_rejects_invalid_top_k():
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32), _keys("a"))

    with pytest.raises(ValueError):
        index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=0)
    with pytest.raises(ValueError):
        index.search(np.array([[1.0, 0.0]], dtype=np.float32), top_k=-1)


def test_search_rejects_wrong_query_dimension():
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32), _keys("a"))

    with pytest.raises(RetrievalIndexError):
        index.search(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), top_k=1)


def test_search_rejects_non_unit_norm_query_when_normalized_true():
    index = FaissFlatIPIndex(2, normalized=True)
    index.add(np.stack([_unit([1.0, 0.0])]), _keys("a"))

    with pytest.raises(RetrievalIndexError):
        index.search(np.array([[2.0, 0.0]], dtype=np.float32), top_k=1)


def test_search_is_independent_of_insertion_order():
    in_order = FaissFlatIPIndex(2, normalized=False)
    in_order.add(
        np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32),
        _keys("a", "b", "c"),
    )

    shuffled = FaissFlatIPIndex(2, normalized=False)
    shuffled.add(np.array([[0.0, 1.0]], dtype=np.float32), _keys("c"))
    shuffled.add(np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32), _keys("a", "b"))

    query = np.array([[1.0, 0.0]], dtype=np.float32)
    scores_in_order, keys_in_order = in_order.search(query, top_k=3)
    scores_shuffled, keys_shuffled = shuffled.search(query, top_k=3)

    assert sorted(keys_in_order[0]) == sorted(keys_shuffled[0])
    np.testing.assert_allclose(sorted(scores_in_order[0]), sorted(scores_shuffled[0]))


# ---------------------------------------------------------------------------
# save() / load()
# ---------------------------------------------------------------------------


def test_save_creates_expected_files(tmp_path):
    index = FaissFlatIPIndex(2, normalized=True, config_version="1.0")
    index.add(np.stack([_unit([1.0, 0.0]), _unit([0.0, 1.0])]), _keys("a", "b"))

    directory = tmp_path / "my_index"
    index.save(directory)

    assert (directory / "index.faiss").exists()
    assert (directory / "metadata.json").exists()


def test_metadata_contains_required_fields(tmp_path):
    index = FaissFlatIPIndex(3, normalized=True, config_version="2.0")
    index.add(np.stack([_unit([1.0, 0.0, 0.0])]), _keys("a"))

    directory = tmp_path / "idx"
    index.save(directory)
    payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    for field in (
        "config_version",
        "embedding_dim",
        "normalized",
        "similarity_type",
        "num_vectors",
        "build_timestamp",
        "git_commit",
        "index_type",
        "faiss_version",
    ):
        assert field in payload

    assert payload["config_version"] == "2.0"
    assert payload["embedding_dim"] == 3
    assert payload["normalized"] is True
    assert payload["similarity_type"] == "cosine"
    assert payload["num_vectors"] == 1
    assert payload["index_type"] == "FlatIP"


def test_load_reconstructs_identical_search_results(tmp_path):
    index = FaissFlatIPIndex(2, normalized=False)
    embeddings = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    keys = _keys("a", "b", "c")
    index.add(embeddings, keys)

    directory = tmp_path / "idx"
    index.save(directory)

    reloaded = FaissFlatIPIndex.load(directory)

    query = np.array([[1.0, 0.0]], dtype=np.float32)
    original_scores, original_keys = index.search(query, top_k=3)
    reloaded_scores, reloaded_keys = reloaded.search(query, top_k=3)

    np.testing.assert_allclose(original_scores, reloaded_scores)
    assert original_keys == reloaded_keys
    assert reloaded.num_vectors == index.num_vectors
    assert reloaded.embedding_dim == index.embedding_dim
    assert reloaded.normalized == index.normalized


def test_load_rejects_missing_metadata(tmp_path):
    directory = tmp_path / "idx"
    directory.mkdir()
    (directory / "index.faiss").write_bytes(b"not a real index")

    with pytest.raises(RetrievalIndexError):
        FaissFlatIPIndex.load(directory)


def test_load_rejects_missing_index_file(tmp_path):
    directory = tmp_path / "idx"
    directory.mkdir()
    (directory / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RetrievalIndexError):
        FaissFlatIPIndex.load(directory)


def test_load_rejects_malformed_metadata_json(tmp_path):
    index = FaissFlatIPIndex(2, normalized=False)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32), _keys("a"))
    directory = tmp_path / "idx"
    index.save(directory)

    (directory / "metadata.json").write_text("not json", encoding="utf-8")

    with pytest.raises(RetrievalIndexError):
        FaissFlatIPIndex.load(directory)


# ---------------------------------------------------------------------------
# Part 4 -- Synthetic smoke test
# ---------------------------------------------------------------------------


def test_synthetic_smoke_full_lifecycle(tmp_path):
    rng = np.random.default_rng(42)
    raw = rng.normal(size=(20, 8)).astype(np.float32)
    normalized_embeddings = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    keys = _keys(*[f"study{i}" for i in range(20)])

    index = FaissFlatIPIndex(8, normalized=True, config_version="1.0")
    index.add(normalized_embeddings, keys)
    assert index.num_vectors == 20

    query = normalized_embeddings[:3]
    scores, result_keys = index.search(query, top_k=5)
    assert scores.shape == (3, 5)
    assert len(result_keys) == 3
    # Each query's own vector (identical to a corpus row) should be its own
    # closest match, with cosine similarity ~1.0.
    for i in range(3):
        assert result_keys[i][0] == keys[i]
        assert scores[i][0] == pytest.approx(1.0, abs=1e-4)

    directory = tmp_path / "smoke_index"
    index.save(directory)
    reloaded = FaissFlatIPIndex.load(directory)

    reloaded_scores, reloaded_keys = reloaded.search(query, top_k=5)
    np.testing.assert_allclose(scores, reloaded_scores, atol=1e-5)
    assert result_keys == reloaded_keys

    # Deterministic behavior: repeating the same search twice on the same
    # (reloaded) index gives bit-identical results.
    again_scores, again_keys = reloaded.search(query, top_k=5)
    np.testing.assert_array_equal(reloaded_scores, again_scores)
    assert reloaded_keys == again_keys

    # Shuffled insertion order produces identical retrieval.
    shuffled_order = list(range(20))
    rng.shuffle(shuffled_order)
    shuffled_index = FaissFlatIPIndex(8, normalized=True, config_version="1.0")
    shuffled_index.add(
        normalized_embeddings[shuffled_order],
        [keys[i] for i in shuffled_order],
    )
    shuffled_scores, shuffled_keys = shuffled_index.search(query, top_k=5)

    np.testing.assert_allclose(sorted(scores[0]), sorted(shuffled_scores[0]), atol=1e-5)
    assert sorted(result_keys[0]) == sorted(shuffled_keys[0])
