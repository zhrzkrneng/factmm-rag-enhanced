"""Embedding export utilities.

Responsibility: run a trained (or fake, injected) retriever over an
ordered corpus/query set and return a stable, deterministically-ordered
embedding array, mirroring the official FactMM-RAG
DPR/ANCE gen_embeddings.py scripts -- which always L2-normalize both
image and text embeddings before persisting them
(`F.normalize(embeddings, dim=-1)`, unconditionally, in both
gen_img_embeddings and gen_txt_embeddings).

Two modes only, matching what MultiModalRetriever's own audited
encode_* methods support for corpus/query export (see
src.baseline.retrieval.model's module docstring): "image" dispatches to
encode_images_only (QUERY encoding, paper Eq. 3); "text" dispatches to
encode_text_only -- the official gen_embeddings.py's own candidate
corpus representation at inference time (a confirmed train/inference
representation mismatch relative to encode_images_with_text, which is
what training actually uses for candidates -- see
docs/risk_register.md). No joint image+text export mode is offered
here, since the official gen_embeddings.py scripts never export one.

`model` is duck-typed, not required to be a real MultiModalRetriever:
any object exposing `encode_images_only(pixel_values)` /
`encode_text_only(text_inputs)` returning a 2-D tensor works, so tests
can inject a lightweight fake and never need real CLIP/T5 inference.

`records` is a Sequence of (QueryKey, model_input) pairs -- deliberately
a list of pairs, not a Dict, so a caller-supplied duplicate key is a
real, observable condition this module can reject rather than one a
dict would have silently collapsed. Output key order is exactly the
input order; this module never sorts or reorders.

The `normalize` parameter is independent of whatever a real
MultiModalRetriever's own config.normalize_embeddings already did --
re-normalizing an already-unit-norm vector is idempotent, so this is
safe to leave on even when the model itself normalizes internally, and
is the only way to get normalized output from a bare duck-typed fake
that does not normalize on its own.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import RetrievalIndexError

_MODES: Tuple[str, ...] = ("image", "text")


def _validate_keys(records: Sequence[Tuple[QueryKey, Any]]) -> None:
    if not records:
        raise RetrievalIndexError("export_embeddings received an empty corpus")

    seen = set()
    for key, _ in records:
        if key in seen:
            raise RetrievalIndexError(f"Duplicate key {key} found in export_embeddings records")
        seen.add(key)


def _stack_image_inputs(records: Sequence[Tuple[QueryKey, torch.Tensor]]) -> torch.Tensor:
    first_shape = records[0][1].shape
    for key, tensor in records:
        if tensor.shape != first_shape:
            raise RetrievalIndexError(
                f"Inconsistent image input shape for key {key}: expected "
                f"{tuple(first_shape)}, got {tuple(tensor.shape)}"
            )
        if not torch.isfinite(tensor).all():
            raise RetrievalIndexError(f"Non-finite (NaN/Inf) image input value for key {key}")
    return torch.stack([tensor for _, tensor in records], dim=0)


def _pad_and_stack_text_inputs(records: Sequence[Tuple[QueryKey, Dict[str, torch.Tensor]]]) -> dict:
    for key, text_input in records:
        if "input_ids" not in text_input or "attention_mask" not in text_input:
            raise RetrievalIndexError(
                f"Text input for key {key} missing 'input_ids'/'attention_mask'"
            )
        if text_input["input_ids"].shape != text_input["attention_mask"].shape:
            raise RetrievalIndexError(f"input_ids/attention_mask shape mismatch for key {key}")

    max_len = max(text_input["input_ids"].shape[0] for _, text_input in records)
    batch_size = len(records)

    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for row, (_, text_input) in enumerate(records):
        length = text_input["input_ids"].shape[0]
        input_ids[row, :length] = text_input["input_ids"]
        attention_mask[row, :length] = text_input["attention_mask"]

    return {"input_ids": input_ids, "attention_mask": attention_mask}


def export_embeddings(
    model: Any,
    records: Sequence[Tuple[QueryKey, Any]],
    *,
    mode: Literal["image", "text"],
    normalize: bool = True,
) -> Tuple[np.ndarray, List[QueryKey]]:
    """Encodes `records` in one batched call and returns (embeddings, keys).

    Args:
        model: exposes encode_images_only(pixel_values) for mode="image"
            or encode_text_only(text_inputs) for mode="text"; either
            returns a 2-D [N, embedding_dim] tensor.
        records: ordered (QueryKey, model_input) pairs. For mode="image",
            model_input is a per-record pixel_values tensor of uniform
            shape. For mode="text", model_input is a per-record
            {"input_ids": Tensor[L], "attention_mask": Tensor[L]} dict
            (variable length -- padded here to the batch's max length).
        mode: "image" or "text" -- never a joint mode (see module
            docstring).
        normalize: L2-normalize the returned embeddings along the last
            dimension. Independent of the model's own normalization.

    Returns:
        (embeddings, keys): embeddings is a [N, embedding_dim] float32
        numpy array; keys is a list of QueryKey in exactly `records`'
        input order.

    Raises:
        RetrievalIndexError: records is empty, contains a duplicate key,
            has inconsistent per-record image shapes, or the resulting
            embeddings contain a non-finite (NaN/Inf) value.
        ValueError: mode is not "image" or "text".
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")

    _validate_keys(records)
    keys = [key for key, _ in records]

    with torch.no_grad():
        if mode == "image":
            batch = _stack_image_inputs(records)
            raw_embeddings = model.encode_images_only(batch)
        else:
            batch = _pad_and_stack_text_inputs(records)
            raw_embeddings = model.encode_text_only(batch)

    if not torch.isfinite(raw_embeddings).all():
        raise RetrievalIndexError(
            f"export_embeddings produced non-finite (NaN/Inf) embedding values in mode={mode!r}"
        )

    if normalize:
        raw_embeddings = F.normalize(raw_embeddings, dim=-1)

    embeddings = raw_embeddings.detach().cpu().numpy().astype(np.float32)
    return embeddings, keys
