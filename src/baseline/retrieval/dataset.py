"""Retriever training dataset and collator.

Responsibility: turn RadGraphAnnotator-shaped annotated records plus a
PairMiner-shaped pair-mining JSONL into flattened, per-(query, positive)
training rows, and batch them into the Stage 1 (DPR) or Stage 2
(hard-negative/ANCE) tensor schemas approved for MultiModalRetriever.

Disclosed limitation: real RadGraphAnnotator output
(src.baseline.radgraph.annotator.REQUIRED_SUCCESS_FIELDS) does not
include an image path -- only dataset/patient_id/study_id/finding/
entities/chexbert_labels. Joining a real image path in from manifest/
ReportRecord data is not automated anywhere in this project yet.
load_annotated_records therefore *requires* an `image_path` field on
every line; producing a JSONL file that actually has one (by joining
annotation output with manifest data) is left to the caller, not
silently assumed or synthesized here.

Hard negatives are supported but never generated: RetrieverTrainingDataset
accepts an optional, already-computed `hard_negatives_by_key` mapping
(query key -> list of negative keys). Actually mining those negatives is
src.baseline.retrieval.hard_negatives's job (still an unimplemented
Phase-1 stub) -- entirely out of scope here.

Negative keys are validated against the same same-dataset invariant
PairMiner itself enforces on positives (mine_query only ever considers
same-dataset candidates) -- a REASONABLE_INFERENCE that the ANCE hard-
negative corpus is scoped identically, since hard_negatives.py has not
been implemented/audited yet to confirm this independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.baseline.pair_mining.mining import REQUIRED_OUTPUT_FIELDS, QueryKey
from src.common.exceptions import RetrievalDatasetError

REQUIRED_RECORD_FIELDS: Tuple[str, ...] = (
    "dataset",
    "patient_id",
    "study_id",
    "finding",
    "image_path",
)

_TRAINING_STAGES: Tuple[str, ...] = ("dpr", "hard_negative")


def load_annotated_records(path: Path) -> Dict[QueryKey, dict]:
    """Loads a JSONL file of annotated records keyed by composite ID.

    Each non-blank line must be valid JSON containing every field in
    REQUIRED_RECORD_FIELDS (see module docstring re: image_path). A
    malformed line or a duplicate (dataset, patient_id, study_id) key
    raises RetrievalDatasetError naming the file and line number --
    never silently skipped or overwritten.
    """
    path = Path(path)
    records: Dict[QueryKey, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalDatasetError(
                    f"Malformed annotated record at {path} line {line_no}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_RECORD_FIELDS if field not in obj]
            if missing:
                raise RetrievalDatasetError(
                    f"Annotated record at {path} line {line_no} missing required "
                    f"fields: {missing}"
                )

            key: QueryKey = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if key in records:
                raise RetrievalDatasetError(
                    f"Duplicate key {key} found in annotated records at {path} "
                    f"line {line_no}"
                )
            records[key] = obj
    return records


def load_pair_mining_pairs(path: Path) -> Dict[QueryKey, List[QueryKey]]:
    """Loads a PairMiner-produced JSONL file, in file order.

    Validates every line against REQUIRED_OUTPUT_FIELDS (reused from
    src.baseline.pair_mining.mining for schema consistency), rejects a
    duplicate query key, and rejects any positive key whose dataset
    differs from its query's dataset (the cross-dataset invariant
    PairMiner itself enforces at mining time). Existence of referenced
    positive keys within a candidate corpus is NOT checked here -- that
    requires records_by_key and is RetrieverTrainingDataset's job.
    """
    path = Path(path)
    pairs: Dict[QueryKey, List[QueryKey]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RetrievalDatasetError(
                    f"Malformed pair-mining output at {path} line {line_no}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in obj]
            if missing:
                raise RetrievalDatasetError(
                    f"Pair-mining output at {path} line {line_no} missing required "
                    f"fields: {missing}"
                )

            query_key: QueryKey = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if query_key in pairs:
                raise RetrievalDatasetError(
                    f"Duplicate query key {query_key} found in pair-mining output "
                    f"at {path} line {line_no}"
                )

            positive_keys: List[QueryKey] = []
            for raw_key in obj["positive_keys"]:
                positive_key: QueryKey = tuple(raw_key)
                if positive_key[0] != query_key[0]:
                    raise RetrievalDatasetError(
                        f"Pair-mining output at {path} line {line_no}: positive key "
                        f"{positive_key} violates the cross-dataset invariant "
                        f"(query dataset={query_key[0]!r})"
                    )
                positive_keys.append(positive_key)
            pairs[query_key] = positive_keys
    return pairs


@dataclass(frozen=True)
class RetrievalTrainingRow:
    """One flattened (query, single positive) training example.

    negative_keys is empty for Stage 1 (DPR) or whenever no hard
    negatives were supplied for this query -- never populated with a
    synthesized/guessed value.
    """

    query_key: QueryKey
    positive_key: QueryKey
    negative_keys: Tuple[QueryKey, ...]


class RetrieverTrainingDataset(Dataset):
    """Flattens (query, pair-mining positives[, hard negatives]) into
    one row per (query, single positive), matching the official
    MedDataset's own flattening -- a query with N positives contributes
    N rows, each carrying that query's full negative_keys set.

    Row order is exactly the pair-mining JSONL's file order for
    queries, and each query's own positive_keys list order for its
    rows -- fully deterministic given deterministic input files; no
    shuffling is performed here (that is a DataLoader/sampler concern).
    """

    def __init__(
        self,
        records_by_key: Dict[QueryKey, dict],
        pairs_jsonl_path: Path,
        *,
        hard_negatives_by_key: Optional[Dict[QueryKey, List[QueryKey]]] = None,
    ) -> None:
        self._records_by_key = records_by_key
        pairs = load_pair_mining_pairs(Path(pairs_jsonl_path))

        rows: List[RetrievalTrainingRow] = []
        for query_key, positive_keys in pairs.items():
            if query_key not in records_by_key:
                raise RetrievalDatasetError(
                    f"Query key {query_key} referenced in pair-mining output has no "
                    f"matching annotated record"
                )

            negative_keys: Tuple[QueryKey, ...] = ()
            if hard_negatives_by_key is not None and query_key in hard_negatives_by_key:
                negative_keys = tuple(hard_negatives_by_key[query_key])
                for negative_key in negative_keys:
                    if negative_key not in records_by_key:
                        raise RetrievalDatasetError(
                            f"Hard negative key {negative_key} for query {query_key} "
                            f"has no matching annotated record"
                        )
                    if negative_key[0] != query_key[0]:
                        raise RetrievalDatasetError(
                            f"Hard negative key {negative_key} violates the "
                            f"cross-dataset invariant (query dataset="
                            f"{query_key[0]!r})"
                        )

            for positive_key in positive_keys:
                if positive_key not in records_by_key:
                    raise RetrievalDatasetError(
                        f"Positive key {positive_key} referenced by query "
                        f"{query_key} has no matching annotated record"
                    )
                rows.append(
                    RetrievalTrainingRow(
                        query_key=query_key,
                        positive_key=positive_key,
                        negative_keys=negative_keys,
                    )
                )

        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict:
        row = self._rows[index]
        query_record = self._records_by_key[row.query_key]
        positive_record = self._records_by_key[row.positive_key]
        negative_records = [self._records_by_key[key] for key in row.negative_keys]

        return {
            "query_key": row.query_key,
            "query_image_path": query_record["image_path"],
            "positive_key": row.positive_key,
            "positive_image_path": positive_record["image_path"],
            "positive_text": positive_record["finding"],
            "negative_keys": row.negative_keys,
            "negative_image_paths": tuple(record["image_path"] for record in negative_records),
            "negative_texts": tuple(record["finding"] for record in negative_records),
        }


class RetrievalCollator:
    """Batches RetrieverTrainingDataset rows into the approved Stage 1 /
    Stage 2 tensor schemas.

    image_processor: callable(List[str] paths) -> Tensor[N, 3, H, W].
    A custom abstraction, not literally CLIPProcessor-compatible -- a
    real adapter is deferred to a later cell.
    tokenizer: callable matching the HF batched-call convention --
    tokenizer(List[str], padding=..., truncation=..., max_length=...,
    return_tensors="pt") -> {"input_ids": Tensor[N, L], "attention_mask":
    Tensor[N, L]}.

    training_stage is fixed per collator instance (never inferred per
    batch), matching RetrieverConfig.training_stage's own contract.
    """

    def __init__(
        self,
        image_processor: Callable[[List[str]], torch.Tensor],
        tokenizer: Callable[..., dict],
        *,
        training_stage: Literal["dpr", "hard_negative"] = "dpr",
        max_text_length: Optional[int] = None,
    ) -> None:
        if training_stage not in _TRAINING_STAGES:
            raise ValueError(
                f"training_stage must be one of {_TRAINING_STAGES}, got {training_stage!r}"
            )
        if max_text_length is not None and max_text_length <= 0:
            raise ValueError(
                f"max_text_length must be a positive int or None, got {max_text_length!r}"
            )
        self._image_processor = image_processor
        self._tokenizer = tokenizer
        self._training_stage = training_stage
        self._max_text_length = max_text_length

    def _tokenize(self, texts: List[str]) -> dict:
        kwargs = {"padding": True, "truncation": self._max_text_length is not None, "return_tensors": "pt"}
        if self._max_text_length is not None:
            kwargs["max_length"] = self._max_text_length
        return self._tokenizer(texts, **kwargs)

    def __call__(self, batch: List[dict]) -> dict:
        if not batch:
            raise ValueError("RetrievalCollator received an empty batch")
        if self._training_stage == "dpr":
            return self._collate_stage1(batch)
        return self._collate_stage2(batch)

    def _collate_stage1(self, batch: List[dict]) -> dict:
        batch_size = len(batch)
        query_image_inputs = self._image_processor([row["query_image_path"] for row in batch])
        positive_image_inputs = self._image_processor([row["positive_image_path"] for row in batch])
        text_inputs = self._tokenize([row["positive_text"] for row in batch])

        return {
            "query_image_inputs": query_image_inputs,
            "positive_image_inputs": positive_image_inputs,
            "positive_text_input_ids": text_inputs["input_ids"],
            "positive_text_attention_mask": text_inputs["attention_mask"],
            "targets": torch.arange(batch_size, dtype=torch.long),
        }

    def _collate_stage2(self, batch: List[dict]) -> dict:
        batch_size = len(batch)
        max_negatives = max((len(row["negative_keys"]) for row in batch), default=0)
        num_slots = 1 + max_negatives

        query_image_inputs = self._image_processor([row["query_image_path"] for row in batch])

        candidate_image_paths: List[str] = []
        candidate_texts: List[str] = []
        candidate_mask = torch.zeros((batch_size, num_slots), dtype=torch.bool)

        for row_index, row in enumerate(batch):
            candidate_image_paths.append(row["positive_image_path"])
            candidate_texts.append(row["positive_text"])
            candidate_mask[row_index, 0] = True

            for slot in range(1, num_slots):
                negative_index = slot - 1
                if negative_index < len(row["negative_image_paths"]):
                    candidate_image_paths.append(row["negative_image_paths"][negative_index])
                    candidate_texts.append(row["negative_texts"][negative_index])
                    candidate_mask[row_index, slot] = True
                else:
                    # Inert filler (this row's own positive) so the
                    # image_processor/tokenizer always receive a valid
                    # input; candidate_mask marks the slot as padding.
                    candidate_image_paths.append(row["positive_image_path"])
                    candidate_texts.append(row["positive_text"])
                    candidate_mask[row_index, slot] = False

        candidate_image_inputs = self._image_processor(candidate_image_paths)
        candidate_text_inputs = self._tokenize(candidate_texts)

        candidate_image_inputs = candidate_image_inputs.view(
            batch_size, num_slots, *candidate_image_inputs.shape[1:]
        )
        candidate_text_input_ids = candidate_text_inputs["input_ids"].view(batch_size, num_slots, -1)
        candidate_text_attention_mask = candidate_text_inputs["attention_mask"].view(
            batch_size, num_slots, -1
        )

        return {
            "query_image_inputs": query_image_inputs,
            "candidate_image_inputs": candidate_image_inputs,
            "candidate_text_input_ids": candidate_text_input_ids,
            "candidate_text_attention_mask": candidate_text_attention_mask,
            "candidate_mask": candidate_mask,
            "targets": torch.zeros(batch_size, dtype=torch.long),
        }
