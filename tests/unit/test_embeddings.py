"""Unit tests for src/retrieval/embeddings.py."""

import numpy as np
import pytest
import torch

from src.common.exceptions import RetrievalIndexError
from src.retrieval.embeddings import export_embeddings


class FakeImageModel:
    """encode_images_only: mean-pools each image over spatial dims."""

    def encode_images_only(self, pixel_values):
        return pixel_values.mean(dim=(2, 3))


class FakeTextModel:
    """encode_text_only: deterministic function of masked token ids."""

    def encode_text_only(self, text_inputs):
        input_ids = text_inputs["input_ids"].float()
        mask = text_inputs["attention_mask"].float()
        summed = (input_ids * mask).sum(dim=1, keepdim=True)
        return summed.repeat(1, 3)


class FakeNaNImageModel:
    def encode_images_only(self, pixel_values):
        out = pixel_values.mean(dim=(2, 3))
        out[0, 0] = float("nan")
        return out


class FakeInfImageModel:
    def encode_images_only(self, pixel_values):
        out = pixel_values.mean(dim=(2, 3))
        out[0, 0] = float("inf")
        return out


def _image_tensor(fill_value, shape=(3, 4, 4)):
    return torch.full(shape, float(fill_value))


def _text_tensor(token_ids):
    ids = torch.tensor(token_ids, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ---------------------------------------------------------------------------
# Basic valid export
# ---------------------------------------------------------------------------


def test_export_embeddings_image_mode_shapes_and_keys():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.1)),
        (("mimic", "p2", "s2"), _image_tensor(0.2)),
        (("mimic", "p3", "s3"), _image_tensor(0.3)),
    ]

    embeddings, keys = export_embeddings(FakeImageModel(), records, mode="image", normalize=False)

    assert embeddings.shape == (3, 3)
    assert keys == [key for key, _ in records]
    assert embeddings.dtype == np.float32


def test_export_embeddings_text_mode_shapes_and_keys():
    records = [
        (("mimic", "p1", "s1"), _text_tensor([1, 2, 3])),
        (("mimic", "p2", "s2"), _text_tensor([4, 5])),
    ]

    embeddings, keys = export_embeddings(FakeTextModel(), records, mode="text", normalize=False)

    assert embeddings.shape == (2, 3)
    assert keys == [("mimic", "p1", "s1"), ("mimic", "p2", "s2")]


def test_export_embeddings_preserves_deterministic_input_order():
    records = [
        (("mimic", "p3", "s3"), _image_tensor(0.3)),
        (("mimic", "p1", "s1"), _image_tensor(0.1)),
        (("mimic", "p2", "s2"), _image_tensor(0.2)),
    ]

    _embeddings, keys = export_embeddings(FakeImageModel(), records, mode="image", normalize=False)

    assert keys == [
        ("mimic", "p3", "s3"),
        ("mimic", "p1", "s1"),
        ("mimic", "p2", "s2"),
    ]


def test_export_embeddings_variable_length_text_is_padded():
    records = [
        (("mimic", "p1", "s1"), _text_tensor([1])),
        (("mimic", "p2", "s2"), _text_tensor([1, 1, 1, 1, 1])),
    ]

    embeddings, _keys = export_embeddings(FakeTextModel(), records, mode="text", normalize=False)

    # Row 0 padded with zeros beyond length 1 -- should not corrupt its own sum.
    assert embeddings[0, 0] == pytest.approx(1.0)
    assert embeddings[1, 0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_export_embeddings_normalize_true_produces_unit_norm():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.5)),
        (("mimic", "p2", "s2"), _image_tensor(2.0)),
    ]

    embeddings, _keys = export_embeddings(FakeImageModel(), records, mode="image", normalize=True)

    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_export_embeddings_normalize_false_leaves_raw_scale():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.5)),
    ]

    embeddings, _keys = export_embeddings(FakeImageModel(), records, mode="image", normalize=False)

    # FakeImageModel mean-pools a constant-0.5 image -> raw embedding is [0.5, 0.5, 0.5], norm != 1.
    assert not np.isclose(np.linalg.norm(embeddings[0]), 1.0)


# ---------------------------------------------------------------------------
# Validation / rejection
# ---------------------------------------------------------------------------


def test_export_embeddings_rejects_empty_corpus():
    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeImageModel(), [], mode="image")


def test_export_embeddings_rejects_duplicate_keys():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.1)),
        (("mimic", "p1", "s1"), _image_tensor(0.2)),
    ]

    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeImageModel(), records, mode="image")


def test_export_embeddings_rejects_inconsistent_image_dimensions():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.1, shape=(3, 4, 4))),
        (("mimic", "p2", "s2"), _image_tensor(0.1, shape=(3, 8, 8))),
    ]

    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeImageModel(), records, mode="image")


def test_export_embeddings_rejects_nan_input():
    records = [
        (("mimic", "p1", "s1"), torch.tensor([[[float("nan")]], [[1.0]], [[1.0]]])),
    ]

    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeImageModel(), records, mode="image")


def test_export_embeddings_rejects_nan_output():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.1)),
        (("mimic", "p2", "s2"), _image_tensor(0.2)),
    ]

    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeNaNImageModel(), records, mode="image")


def test_export_embeddings_rejects_inf_output():
    records = [
        (("mimic", "p1", "s1"), _image_tensor(0.1)),
        (("mimic", "p2", "s2"), _image_tensor(0.2)),
    ]

    with pytest.raises(RetrievalIndexError):
        export_embeddings(FakeInfImageModel(), records, mode="image")


def test_export_embeddings_rejects_invalid_mode():
    records = [(("mimic", "p1", "s1"), _image_tensor(0.1))]

    with pytest.raises(ValueError):
        export_embeddings(FakeImageModel(), records, mode="joint")
