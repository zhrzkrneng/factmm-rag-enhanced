"""Unit tests for src/baseline/retrieval/dataset.py."""

import json
from pathlib import Path

import pytest
import torch

from src.baseline.retrieval.dataset import (
    RetrievalCollator,
    RetrievalTrainingRow,
    RetrieverTrainingDataset,
    load_annotated_records,
    load_pair_mining_pairs,
)
from src.common.exceptions import RetrievalDatasetError


def _write_jsonl(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def _record(dataset, patient_id, study_id, finding, image_path=None):
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "finding": finding,
        "image_path": image_path if image_path is not None else f"/images/{dataset}/{study_id}.png",
    }


def _pair_row(dataset, patient_id, study_id, positive_keys):
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "positive_keys": [list(k) for k in positive_keys],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 1.0}] * len(positive_keys),
        "num_candidates_considered": 10,
        "num_qualifying_candidates": len(positive_keys),
        "num_selected": len(positive_keys),
        "flagged_bad_sample": False,
    }


# ---------------------------------------------------------------------------
# Fake test doubles (image_processor / tokenizer)
# ---------------------------------------------------------------------------


class FakeImageProcessor:
    """Deterministic callable(List[str]) -> Tensor[N, 3, H, W]."""

    def __init__(self, channels: int = 3, height: int = 4, width: int = 4):
        self._shape = (channels, height, width)

    def __call__(self, paths):
        tensors = []
        for path in paths:
            seed = abs(hash(path)) % (2**31)
            generator = torch.Generator().manual_seed(seed)
            tensors.append(torch.rand(self._shape, generator=generator))
        return torch.stack(tensors, dim=0)


class FakeTokenizer:
    """Deterministic callable matching the HF batched-call convention."""

    def __call__(self, texts, *, padding=True, truncation=False, max_length=None, return_tensors="pt"):
        token_lists = [text.split() for text in texts]
        if truncation and max_length is not None:
            token_lists = [tokens[:max_length] for tokens in token_lists]

        vocab = {}

        def token_id(token):
            return vocab.setdefault(token, len(vocab) + 1)

        id_lists = [[token_id(tok) for tok in tokens] for tokens in token_lists]
        max_len = max((len(ids) for ids in id_lists), default=0)

        input_ids = torch.zeros((len(texts), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.long)
        for i, ids in enumerate(id_lists):
            length = len(ids)
            if length:
                input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# load_annotated_records
# ---------------------------------------------------------------------------


def test_load_annotated_records_valid(tmp_path):
    path = tmp_path / "annotations.jsonl"
    _write_jsonl(
        path,
        [
            _record("mimic", "p1", "s1", "finding one"),
            _record("mimic", "p2", "s2", "finding two"),
        ],
    )

    records = load_annotated_records(path)

    assert set(records.keys()) == {("mimic", "p1", "s1"), ("mimic", "p2", "s2")}
    assert records[("mimic", "p1", "s1")]["finding"] == "finding one"


def test_load_annotated_records_rejects_malformed_json(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text('{"dataset": "mimic", not valid json\n', encoding="utf-8")

    with pytest.raises(RetrievalDatasetError):
        load_annotated_records(path)


def test_load_annotated_records_rejects_missing_required_field(tmp_path):
    path = tmp_path / "annotations.jsonl"
    bad = _record("mimic", "p1", "s1", "finding one")
    del bad["image_path"]
    _write_jsonl(path, [bad])

    with pytest.raises(RetrievalDatasetError):
        load_annotated_records(path)


def test_load_annotated_records_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "annotations.jsonl"
    _write_jsonl(
        path,
        [
            _record("mimic", "p1", "s1", "finding one"),
            _record("mimic", "p1", "s1", "finding one again"),
        ],
    )

    with pytest.raises(RetrievalDatasetError):
        load_annotated_records(path)


# ---------------------------------------------------------------------------
# load_pair_mining_pairs
# ---------------------------------------------------------------------------


def test_load_pair_mining_pairs_valid(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2")]),
            _pair_row("mimic", "p2", "s2", [("mimic", "p1", "s1")]),
        ],
    )

    pairs = load_pair_mining_pairs(path)

    assert pairs[("mimic", "p1", "s1")] == [("mimic", "p2", "s2")]
    assert pairs[("mimic", "p2", "s2")] == [("mimic", "p1", "s1")]


def test_load_pair_mining_pairs_rejects_malformed_json(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("not json at all\n", encoding="utf-8")

    with pytest.raises(RetrievalDatasetError):
        load_pair_mining_pairs(path)


def test_load_pair_mining_pairs_rejects_duplicate_query_keys(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_row("mimic", "p1", "s1", []),
            _pair_row("mimic", "p1", "s1", []),
        ],
    )

    with pytest.raises(RetrievalDatasetError):
        load_pair_mining_pairs(path)


def test_load_pair_mining_pairs_rejects_cross_dataset_positive(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_row("mimic", "p1", "s1", [("chexpert", "p2", "s2")]),
        ],
    )

    with pytest.raises(RetrievalDatasetError):
        load_pair_mining_pairs(path)


def test_load_pair_mining_pairs_preserves_deterministic_file_order(tmp_path):
    path = tmp_path / "pairs.jsonl"
    _write_jsonl(
        path,
        [
            _pair_row("mimic", "p3", "s3", []),
            _pair_row("mimic", "p1", "s1", []),
            _pair_row("mimic", "p2", "s2", []),
        ],
    )

    pairs = load_pair_mining_pairs(path)

    assert list(pairs.keys()) == [
        ("mimic", "p3", "s3"),
        ("mimic", "p1", "s1"),
        ("mimic", "p2", "s2"),
    ]


# ---------------------------------------------------------------------------
# RetrieverTrainingDataset
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path, *, pair_rows, records=None, hard_negatives_by_key=None):
    records_by_key = records
    if records_by_key is None:
        records_by_key = {
            ("mimic", "p1", "s1"): _record("mimic", "p1", "s1", "finding one"),
            ("mimic", "p2", "s2"): _record("mimic", "p2", "s2", "finding two"),
            ("mimic", "p3", "s3"): _record("mimic", "p3", "s3", "finding three longer text here"),
        }
    pairs_path = tmp_path / "pairs.jsonl"
    _write_jsonl(pairs_path, pair_rows)
    return RetrieverTrainingDataset(
        records_by_key, pairs_path, hard_negatives_by_key=hard_negatives_by_key
    )


def test_dataset_flattens_one_row_per_query_positive(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        pair_rows=[
            _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2"), ("mimic", "p3", "s3")]),
        ],
    )

    assert len(dataset) == 2
    row0 = dataset[0]
    row1 = dataset[1]
    assert row0["query_key"] == ("mimic", "p1", "s1")
    assert row0["positive_key"] == ("mimic", "p2", "s2")
    assert row1["positive_key"] == ("mimic", "p3", "s3")


def test_dataset_rejects_missing_referenced_positive(tmp_path):
    with pytest.raises(RetrievalDatasetError):
        _make_dataset(
            tmp_path,
            pair_rows=[
                _pair_row("mimic", "p1", "s1", [("mimic", "p9", "s9")]),
            ],
        )


def test_dataset_rejects_query_key_missing_from_records(tmp_path):
    records_by_key = {
        ("mimic", "p2", "s2"): _record("mimic", "p2", "s2", "finding two"),
    }
    with pytest.raises(RetrievalDatasetError):
        _make_dataset(
            tmp_path,
            records=records_by_key,
            pair_rows=[
                _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2")]),
            ],
        )


def test_dataset_preserves_deterministic_ordering(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        pair_rows=[
            _pair_row("mimic", "p3", "s3", [("mimic", "p1", "s1")]),
            _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2"), ("mimic", "p3", "s3")]),
        ],
    )

    query_keys_in_order = [dataset[i]["query_key"] for i in range(len(dataset))]
    assert query_keys_in_order == [
        ("mimic", "p3", "s3"),
        ("mimic", "p1", "s1"),
        ("mimic", "p1", "s1"),
    ]


def test_dataset_supports_optional_hard_negatives_without_generating_them(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        pair_rows=[
            _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2")]),
        ],
        hard_negatives_by_key={("mimic", "p1", "s1"): [("mimic", "p3", "s3")]},
    )

    row = dataset[0]
    assert row["negative_keys"] == (("mimic", "p3", "s3"),)
    assert row["negative_texts"] == ("finding three longer text here",)


def test_dataset_rejects_hard_negative_missing_from_records(tmp_path):
    with pytest.raises(RetrievalDatasetError):
        _make_dataset(
            tmp_path,
            pair_rows=[
                _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2")]),
            ],
            hard_negatives_by_key={("mimic", "p1", "s1"): [("mimic", "p9", "s9")]},
        )


def test_dataset_rejects_cross_dataset_hard_negative(tmp_path):
    records_by_key = {
        ("mimic", "p1", "s1"): _record("mimic", "p1", "s1", "finding one"),
        ("mimic", "p2", "s2"): _record("mimic", "p2", "s2", "finding two"),
        ("chexpert", "p9", "s9"): _record("chexpert", "p9", "s9", "finding nine"),
    }
    with pytest.raises(RetrievalDatasetError):
        _make_dataset(
            tmp_path,
            records=records_by_key,
            pair_rows=[
                _pair_row("mimic", "p1", "s1", [("mimic", "p2", "s2")]),
            ],
            hard_negatives_by_key={("mimic", "p1", "s1"): [("chexpert", "p9", "s9")]},
        )


def test_dataset_flagged_bad_sample_query_contributes_no_rows(tmp_path):
    dataset = _make_dataset(
        tmp_path,
        pair_rows=[
            _pair_row("mimic", "p1", "s1", []),
        ],
    )

    assert len(dataset) == 0


# ---------------------------------------------------------------------------
# RetrievalCollator
# ---------------------------------------------------------------------------


def _rows(*specs):
    """specs: list of (query_key, positive_key, positive_text, negative_texts)."""
    rows = []
    for query_key, positive_key, positive_text, negative_texts in specs:
        rows.append(
            {
                "query_key": query_key,
                "query_image_path": f"/images/{'/'.join(query_key)}.png",
                "positive_key": positive_key,
                "positive_image_path": f"/images/{'/'.join(positive_key)}.png",
                "positive_text": positive_text,
                "negative_keys": tuple(("mimic", "neg", str(i)) for i in range(len(negative_texts))),
                "negative_image_paths": tuple(
                    f"/images/neg/{i}.png" for i in range(len(negative_texts))
                ),
                "negative_texts": tuple(negative_texts),
            }
        )
    return rows


def test_collator_rejects_empty_batch():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer())

    with pytest.raises(ValueError):
        collator([])


def test_collator_stage1_tensor_shapes():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="dpr")
    batch = _rows(
        (("mimic", "q1", "s1"), ("mimic", "p1", "s1"), "short report", ()),
        (("mimic", "q2", "s2"), ("mimic", "p2", "s2"), "a much longer radiology report here", ()),
    )

    output = collator(batch)

    assert output["query_image_inputs"].shape == (2, 3, 4, 4)
    assert output["positive_image_inputs"].shape == (2, 3, 4, 4)
    assert output["positive_text_input_ids"].shape[0] == 2
    assert output["positive_text_attention_mask"].shape == output["positive_text_input_ids"].shape
    assert torch.equal(output["targets"], torch.tensor([0, 1], dtype=torch.long))


def test_collator_stage1_supports_variable_length_reports():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="dpr")
    batch = _rows(
        (("mimic", "q1", "s1"), ("mimic", "p1", "s1"), "one", ()),
        (("mimic", "q2", "s2"), ("mimic", "p2", "s2"), "one two three four five", ()),
    )

    output = collator(batch)

    max_len = output["positive_text_input_ids"].shape[1]
    assert max_len == 5
    # Shorter report's row is zero-padded and masked out beyond its length.
    assert output["positive_text_attention_mask"][0].sum().item() == 1
    assert output["positive_text_attention_mask"][1].sum().item() == 5


def test_collator_stage2_tensor_shapes_and_mask():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="hard_negative")
    batch = _rows(
        (("mimic", "q1", "s1"), ("mimic", "p1", "s1"), "positive one", ["negative one a", "negative one b"]),
        (("mimic", "q2", "s2"), ("mimic", "p2", "s2"), "positive two", ["negative two a"]),
    )

    output = collator(batch)

    # max_negatives = 2 -> num_slots = 3
    assert output["candidate_image_inputs"].shape == (2, 3, 3, 4, 4)
    assert output["candidate_text_input_ids"].shape[:2] == (2, 3)
    assert output["candidate_text_attention_mask"].shape == output["candidate_text_input_ids"].shape
    assert output["candidate_mask"].shape == (2, 3)

    # Row 0 has 2 real negatives -> all 3 slots real.
    assert output["candidate_mask"][0].tolist() == [True, True, True]
    # Row 1 has only 1 real negative -> slot 2 is padding.
    assert output["candidate_mask"][1].tolist() == [True, True, False]

    assert torch.equal(output["targets"], torch.zeros(2, dtype=torch.long))


def test_collator_stage2_with_no_negatives_in_batch():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="hard_negative")
    batch = _rows(
        (("mimic", "q1", "s1"), ("mimic", "p1", "s1"), "positive one", []),
        (("mimic", "q2", "s2"), ("mimic", "p2", "s2"), "positive two", []),
    )

    output = collator(batch)

    assert output["candidate_image_inputs"].shape == (2, 1, 3, 4, 4)
    assert output["candidate_mask"].tolist() == [[True], [True]]


def test_collator_batch_is_reproducible():
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="dpr")
    batch = _rows(
        (("mimic", "q1", "s1"), ("mimic", "p1", "s1"), "some report text", ()),
        (("mimic", "q2", "s2"), ("mimic", "p2", "s2"), "another report", ()),
    )

    output_a = collator(batch)
    output_b = collator(batch)

    assert torch.equal(output_a["query_image_inputs"], output_b["query_image_inputs"])
    assert torch.equal(output_a["positive_image_inputs"], output_b["positive_image_inputs"])
    assert torch.equal(output_a["positive_text_input_ids"], output_b["positive_text_input_ids"])
    assert torch.equal(output_a["positive_text_attention_mask"], output_b["positive_text_attention_mask"])


def test_collator_rejects_invalid_training_stage():
    with pytest.raises(ValueError):
        RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="bogus")


def test_retrieval_training_row_is_frozen_dataclass():
    row = RetrievalTrainingRow(
        query_key=("mimic", "p1", "s1"),
        positive_key=("mimic", "p2", "s2"),
        negative_keys=(),
    )
    assert row.query_key == ("mimic", "p1", "s1")
    with pytest.raises(Exception):
        row.query_key = ("mimic", "p9", "s9")
