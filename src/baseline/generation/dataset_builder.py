"""RAG dataset construction (Milestone 2.5).

Responsibility: build one row per query -- an image, its single
retrieved-evidence report text, and its own target report text -- by
walking a query's precomputed KNN ranking (the shape
src.retrieval.index.FaissFlatIPIndex.search() itself produces per
query, List[QueryKey] best-first; this module accepts that ranking
already computed and keyed by query, and never touches FAISS or
embeddings itself) and keeping the first candidate that is not the
same study, not the same patient, and (only if config.test_short=True)
not under config.min_short_words words.

Matches the official FactMM-RAG build_rag_dataset.py's candidate
filtering (see docs/milestone_2_5_generator_contract.md SS2, SS9), with
one explicit, disclosed, opt-in deviation: on ranking exhaustion the
official code silently falls back to the rank-0 candidate regardless
of patient/study leakage. This module makes that choice explicit and
non-default via RAGDatasetBuilderConfig.reproduce_official_bug (see
SS4) -- the safe default (strict_patient_isolation=True) excludes the
query instead of ever returning a leaking row.

Resume/atomic-write/continue_on_error behavior follows the exact
established pattern from src.baseline.pair_mining.mining.PairMiner and
src.baseline.radgraph.annotator.RadGraphAnnotator -- the fourth
implementation of this same pattern in this project, not a new one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import RAGDatasetError
from src.data.schema import ReportRecord
from src.data.splits import PatientSplitValidator

REQUIRED_OUTPUT_FIELDS: Tuple[str, ...] = (
    "dataset",
    "patient_id",
    "study_id",
    "retrieved_key",
    "image_path",
    "retrieved_report_text",
    "target_report_text",
    "rank_selected",
    "num_candidates_rejected_self_study",
    "num_candidates_rejected_self_patient",
    "num_candidates_rejected_short_text",
    "fallback_used",
    "excluded",
    "strict_patient_isolation",
    "reproduce_official_bug",
    "config_version",
)

# Paper Appendix A.2 (PAPER_EXPLICIT): "we filter out ... malformed
# reports ... specified by being less than 5 characters", worded as
# unconditional -- recorded verbatim, never silently conflated with
# the actual official-code behavior below.
_PAPER_DEFAULTS = {
    "min_short_length": 5,
    "min_short_length_unit": "characters",
    "filter_always_applied": True,
}

# build_rag_dataset.py / build_nonrag_dataset.py shipped defaults
# (OFFICIAL_REPOSITORY): --test_short is opt-in (action="store_true",
# default False); the actual check is len(text.split()) < 5 -- words,
# not characters. See docs/milestone_2_5_generator_contract.md SS2.
_OFFICIAL_REPOSITORY_DEFAULTS = {
    "test_short": False,
    "min_short_words": 5,
    "min_short_length_unit": "words",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_meta_atomic(meta_path: Path, payload: dict) -> None:
    tmp_path = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(meta_path))


@dataclass(frozen=True)
class RAGDatasetBuilderConfig:
    config_version: str = "1.0"
    rag_data_mode: str = "finding"
    output_data_mode: str = "finding"
    strict_patient_isolation: bool = True
    reproduce_official_bug: bool = False
    test_short: bool = False
    min_short_words: int = 5
    corpus_scope: str = "train_only"

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.rag_data_mode:
            raise ValueError("rag_data_mode must be a non-empty string")
        if not self.output_data_mode:
            raise ValueError("output_data_mode must be a non-empty string")
        if not isinstance(self.strict_patient_isolation, bool):
            raise ValueError("strict_patient_isolation must be a bool")
        if not isinstance(self.reproduce_official_bug, bool):
            raise ValueError("reproduce_official_bug must be a bool")
        if not isinstance(self.test_short, bool):
            raise ValueError("test_short must be a bool")
        if (
            not isinstance(self.min_short_words, int)
            or isinstance(self.min_short_words, bool)
            or self.min_short_words < 1
        ):
            raise ValueError(
                f"min_short_words must be a positive integer, got {self.min_short_words!r}"
            )
        if not self.corpus_scope:
            raise ValueError("corpus_scope must be a non-empty string")
        if self.strict_patient_isolation and self.reproduce_official_bug:
            raise ValueError(
                "strict_patient_isolation and reproduce_official_bug are mutually "
                "exclusive -- reproduce_official_bug=True requires explicitly also "
                "passing strict_patient_isolation=False; the two are never silently "
                "reconciled or stacked"
            )

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "rag_data_mode": self.rag_data_mode,
            "output_data_mode": self.output_data_mode,
            "strict_patient_isolation": self.strict_patient_isolation,
            "reproduce_official_bug": self.reproduce_official_bug,
            "test_short": self.test_short,
            "min_short_words": self.min_short_words,
            "corpus_scope": self.corpus_scope,
        }


@dataclass(frozen=True)
class RAGDatasetRow:
    query_key: QueryKey
    retrieved_key: Optional[QueryKey]
    image_path: str
    retrieved_report_text: Optional[str]
    target_report_text: str
    rank_selected: Optional[int]
    num_candidates_rejected_self_study: int
    num_candidates_rejected_self_patient: int
    num_candidates_rejected_short_text: int
    fallback_used: bool
    excluded: bool
    strict_patient_isolation: bool
    reproduce_official_bug: bool
    config_version: str


@dataclass(frozen=True)
class RAGDatasetBuildSummary:
    processed: int
    skipped_already_done: int
    failed: int
    excluded: int
    fallback_used: int


def _row_to_dict(row: RAGDatasetRow) -> dict:
    return {
        "dataset": row.query_key[0],
        "patient_id": row.query_key[1],
        "study_id": row.query_key[2],
        "retrieved_key": list(row.retrieved_key) if row.retrieved_key is not None else None,
        "image_path": row.image_path,
        "retrieved_report_text": row.retrieved_report_text,
        "target_report_text": row.target_report_text,
        "rank_selected": row.rank_selected,
        "num_candidates_rejected_self_study": row.num_candidates_rejected_self_study,
        "num_candidates_rejected_self_patient": row.num_candidates_rejected_self_patient,
        "num_candidates_rejected_short_text": row.num_candidates_rejected_short_text,
        "fallback_used": row.fallback_used,
        "excluded": row.excluded,
        "strict_patient_isolation": row.strict_patient_isolation,
        "reproduce_official_bug": row.reproduce_official_bug,
        "config_version": row.config_version,
    }


def _to_report_record(key: QueryKey, record: dict) -> ReportRecord:
    image_path = record.get("image_path")
    return ReportRecord(
        image_paths=[image_path] if image_path is not None else [],
        finding=record.get("finding", ""),
        impression=record.get("impression", ""),
        patient_id=key[1],
        study_id=key[2],
        dataset=key[0],
    )


def _check_meta_compatibility(meta_path: Path, config: RAGDatasetBuilderConfig) -> None:
    """Rejects resuming against metadata written under a different config.

    A missing meta_path is fine (first run). An existing meta_path is
    only usable if its recorded `defaults.actual_used` exactly matches
    the current run's config -- including config_version. Any mismatch
    raises RAGDatasetError rather than silently resuming under an
    incompatible contract.
    """
    if not meta_path.exists():
        return
    try:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RAGDatasetError(f"Malformed existing metadata at {meta_path}: {exc}") from exc

    existing_actual = existing_meta.get("defaults", {}).get("actual_used")
    if existing_actual is None:
        raise RAGDatasetError(
            f"Existing metadata at {meta_path} is missing defaults.actual_used -- "
            f"cannot verify configuration compatibility, refusing to resume"
        )
    current_actual = config.as_actual_used_dict()
    if existing_actual != current_actual:
        raise RAGDatasetError(
            f"Existing metadata at {meta_path} was produced under a different "
            f"configuration ({existing_actual}) than the current run "
            f"({current_actual}) -- refusing to silently resume under an "
            f"incompatible config"
        )


def _load_completed_rag_dataset_keys(
    output_path: Path,
    *,
    corpus_keys: Set[QueryKey],
    config: RAGDatasetBuilderConfig,
) -> Set[QueryKey]:
    """Validates any existing output_path and returns its completed query keys.

    Beyond well-formed JSON, all required fields present, and no
    duplicate query key, each line is additionally validated for:

      - excluded rows carry no retrieved_key/retrieved_report_text/
        rank_selected/fallback_used
      - non-excluded rows carry all three, the referenced retrieved_key
        is a member of the current corpus, and shares the query's
        dataset (cross-dataset invariant)
      - a non-excluded row's retrieved_key does not share the query's
        patient_id when config.strict_patient_isolation is True
      - fallback_used=True never appears unless
        config.reproduce_official_bug is True

    Any violation raises RAGDatasetError naming the file, line number,
    and specific problem -- existing output is never silently trusted
    or silently treated as stale-but-harmless.
    """
    if not output_path.exists():
        return set()

    completed: Set[QueryKey] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RAGDatasetError(
                    f"Malformed existing output at {output_path} line {line_no}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in obj]
            if missing:
                raise RAGDatasetError(
                    f"Existing output at {output_path} line {line_no} missing "
                    f"required fields: {missing}"
                )

            query_key: QueryKey = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if query_key in completed:
                raise RAGDatasetError(
                    f"Duplicate query key {query_key} found in existing output "
                    f"at {output_path} line {line_no}"
                )

            excluded = obj["excluded"]
            retrieved_key_raw = obj["retrieved_key"]
            fallback_used = obj["fallback_used"]

            if excluded:
                if (
                    retrieved_key_raw is not None
                    or obj["retrieved_report_text"] is not None
                    or obj["rank_selected"] is not None
                    or fallback_used
                ):
                    raise RAGDatasetError(
                        f"{output_path} line {line_no}: excluded=True but carries a "
                        f"retrieved_key/retrieved_report_text/rank_selected/"
                        f"fallback_used inconsistent with exclusion"
                    )
            else:
                if (
                    retrieved_key_raw is None
                    or obj["retrieved_report_text"] is None
                    or obj["rank_selected"] is None
                ):
                    raise RAGDatasetError(
                        f"{output_path} line {line_no}: excluded=False but is missing "
                        f"retrieved_key/retrieved_report_text/rank_selected"
                    )
                retrieved_key: QueryKey = tuple(retrieved_key_raw)
                if retrieved_key not in corpus_keys:
                    raise RAGDatasetError(
                        f"{output_path} line {line_no}: referenced retrieved_key "
                        f"{retrieved_key} not found in the current corpus"
                    )
                if retrieved_key[0] != query_key[0]:
                    raise RAGDatasetError(
                        f"{output_path} line {line_no}: retrieved_key {retrieved_key} "
                        f"violates the cross-dataset invariant (query dataset="
                        f"{query_key[0]!r})"
                    )
                if config.strict_patient_isolation and retrieved_key[1] == query_key[1]:
                    raise RAGDatasetError(
                        f"{output_path} line {line_no}: retrieved_key {retrieved_key} "
                        f"shares the query's patient but "
                        f"config.strict_patient_isolation=True -- output is stale "
                        f"relative to this run's configuration"
                    )

            if fallback_used and not config.reproduce_official_bug:
                raise RAGDatasetError(
                    f"{output_path} line {line_no}: fallback_used=True but "
                    f"config.reproduce_official_bug=False -- output is stale "
                    f"relative to this run's configuration"
                )

            completed.add(query_key)
    return completed


class RAGDatasetBuilder:
    """Builds one RAGDatasetRow per query from a precomputed KNN ranking.

    query_records and corpus_records are plain dicts matching
    src.baseline.retrieval.dataset.load_annotated_records's output
    schema (dataset, patient_id, study_id, finding, image_path, ...).
    knn_rankings maps each query key to its own ranked candidate list
    (best first) -- the shape src.retrieval.index.FaissFlatIPIndex.
    search() produces per query, computed by the caller; this class
    never touches FAISS/embeddings itself.

    query_split_name/corpus_split_name are purely descriptive labels
    used only for the one-time pre-flight patient-isolation check (see
    module docstring and contract SS9.3): when they differ (e.g.
    valid/test queries against the always-train-only corpus), a
    src.data.splits.PatientSplitValidator check confirms the two
    record sets share no patient_id before any row is built, raising
    PatientLeakageError otherwise. When they are equal (train-mode
    building, corpus == query split), the check is skipped -- overlap
    is expected there and is handled per-row by the self-study/
    self-patient filters instead, never by a split-level check.

    Construction eagerly validates that every candidate key referenced
    by any ranking exists in corpus_records and shares its query's
    dataset (the cross-dataset invariant enforced project-wide), and
    that every query/corpus record carries the configured
    rag_data_mode/output_data_mode/image_path fields.
    """

    def __init__(
        self,
        query_records: Dict[QueryKey, dict],
        corpus_records: Dict[QueryKey, dict],
        knn_rankings: Dict[QueryKey, List[QueryKey]],
        *,
        config: RAGDatasetBuilderConfig,
        query_split_name: str = "train",
        corpus_split_name: str = "train",
    ) -> None:
        for query_key, record in query_records.items():
            if "image_path" not in record:
                raise RAGDatasetError(
                    f"Query record {query_key} is missing required field 'image_path'"
                )
            if config.output_data_mode not in record:
                raise RAGDatasetError(
                    f"Query record {query_key} is missing configured output_data_mode "
                    f"field {config.output_data_mode!r}"
                )

        for candidate_key, record in corpus_records.items():
            if config.rag_data_mode not in record:
                raise RAGDatasetError(
                    f"Corpus record {candidate_key} is missing configured rag_data_mode "
                    f"field {config.rag_data_mode!r}"
                )

        for query_key, ranking in knn_rankings.items():
            if query_key not in query_records:
                raise RAGDatasetError(
                    f"knn_rankings references query key {query_key} with no matching "
                    f"query record"
                )
            for candidate_key in ranking:
                if candidate_key not in corpus_records:
                    raise RAGDatasetError(
                        f"knn_rankings for query {query_key} references candidate "
                        f"{candidate_key} with no matching corpus record"
                    )
                if candidate_key[0] != query_key[0]:
                    raise RAGDatasetError(
                        f"knn_rankings for query {query_key} references candidate "
                        f"{candidate_key} violating the cross-dataset invariant "
                        f"(query dataset={query_key[0]!r})"
                    )

        if query_split_name != corpus_split_name:
            split_records = {
                query_split_name: [_to_report_record(k, r) for k, r in query_records.items()],
                corpus_split_name: [_to_report_record(k, r) for k, r in corpus_records.items()],
            }
            PatientSplitValidator().check_and_raise(split_records)

        self._query_records = query_records
        self._corpus_records = corpus_records
        self._knn_rankings = knn_rankings
        self._config = config
        self._query_split_name = query_split_name
        self._corpus_split_name = corpus_split_name

    def build_row(self, query_key: QueryKey) -> RAGDatasetRow:
        if query_key not in self._query_records:
            raise RAGDatasetError(f"Query key {query_key} has no matching query record")

        query_record = self._query_records[query_key]
        candidates = self._knn_rankings.get(query_key, [])

        num_rejected_self_study = 0
        num_rejected_self_patient = 0
        num_rejected_short_text = 0

        selected_key: Optional[QueryKey] = None
        selected_rank: Optional[int] = None

        for rank, candidate_key in enumerate(candidates):
            if candidate_key == query_key:
                num_rejected_self_study += 1
                continue
            if candidate_key[1] == query_key[1]:
                num_rejected_self_patient += 1
                continue
            if self._config.test_short:
                candidate_text = self._corpus_records[candidate_key][self._config.rag_data_mode]
                if len(candidate_text.split()) < self._config.min_short_words:
                    num_rejected_short_text += 1
                    continue
            selected_key = candidate_key
            selected_rank = rank
            break

        target_text = query_record[self._config.output_data_mode]

        if selected_key is not None:
            retrieved_record = self._corpus_records[selected_key]
            return RAGDatasetRow(
                query_key=query_key,
                retrieved_key=selected_key,
                image_path=query_record["image_path"],
                retrieved_report_text=retrieved_record[self._config.rag_data_mode],
                target_report_text=target_text,
                rank_selected=selected_rank,
                num_candidates_rejected_self_study=num_rejected_self_study,
                num_candidates_rejected_self_patient=num_rejected_self_patient,
                num_candidates_rejected_short_text=num_rejected_short_text,
                fallback_used=False,
                excluded=False,
                strict_patient_isolation=self._config.strict_patient_isolation,
                reproduce_official_bug=self._config.reproduce_official_bug,
                config_version=self._config.config_version,
            )

        if self._config.reproduce_official_bug and candidates:
            fallback_key = candidates[0]
            fallback_record = self._corpus_records[fallback_key]
            return RAGDatasetRow(
                query_key=query_key,
                retrieved_key=fallback_key,
                image_path=query_record["image_path"],
                retrieved_report_text=fallback_record[self._config.rag_data_mode],
                target_report_text=target_text,
                rank_selected=0,
                num_candidates_rejected_self_study=num_rejected_self_study,
                num_candidates_rejected_self_patient=num_rejected_self_patient,
                num_candidates_rejected_short_text=num_rejected_short_text,
                fallback_used=True,
                excluded=False,
                strict_patient_isolation=self._config.strict_patient_isolation,
                reproduce_official_bug=self._config.reproduce_official_bug,
                config_version=self._config.config_version,
            )

        return RAGDatasetRow(
            query_key=query_key,
            retrieved_key=None,
            image_path=query_record["image_path"],
            retrieved_report_text=None,
            target_report_text=target_text,
            rank_selected=None,
            num_candidates_rejected_self_study=num_rejected_self_study,
            num_candidates_rejected_self_patient=num_rejected_self_patient,
            num_candidates_rejected_short_text=num_rejected_short_text,
            fallback_used=False,
            excluded=True,
            strict_patient_isolation=self._config.strict_patient_isolation,
            reproduce_official_bug=self._config.reproduce_official_bug,
            config_version=self._config.config_version,
        )

    def build_all(
        self,
        query_keys: Iterable[QueryKey],
        *,
        output_path: Path,
        errors_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        input_path: Optional[str] = None,
        continue_on_error: bool = False,
    ) -> RAGDatasetBuildSummary:
        """Builds a row for each of query_keys, writing one JSONL line each.

        Raises ValueError immediately, before touching output_path or
        meta_path, if continue_on_error=True and errors_path is None.

        meta_path defaults to
        output_path.with_suffix(output_path.suffix + ".meta.json") when
        omitted.

        Before any processing: rejects (RAGDatasetError) an existing
        meta_path recorded under an incompatible config, then validates
        any existing output_path (see _load_completed_rag_dataset_keys)
        to build the resume set.

        A query key already present as a valid completed row is
        skipped. Each new query's row is serialized in one write call
        and flushed immediately (crash-safe). In fail-fast mode
        (continue_on_error=False, the default) any exception updates
        the sidecar metadata to run_status="failed" with the error
        recorded, then re-raises unchanged -- the row is never marked
        completed. In continue_on_error mode the exception is logged to
        errors_path instead and the loop continues.
        """
        output_path = Path(output_path)
        if continue_on_error and errors_path is None:
            raise ValueError("errors_path is required when continue_on_error=True")
        errors_path = Path(errors_path) if errors_path is not None else None
        meta_path = (
            Path(meta_path)
            if meta_path is not None
            else output_path.with_suffix(output_path.suffix + ".meta.json")
        )
        query_keys = list(query_keys)

        _check_meta_compatibility(meta_path, self._config)
        completed_keys = _load_completed_rag_dataset_keys(
            output_path, corpus_keys=set(self._corpus_records), config=self._config
        )

        meta = {
            "run_status": "running",
            "run_started_at_utc": _utc_now_iso(),
            "run_finished_at_utc": None,
            "input_path": input_path if input_path is not None else "unspecified (records provided in-memory)",
            "output_path": str(output_path),
            "errors_path": str(errors_path) if errors_path is not None else None,
            "query_split_name": self._query_split_name,
            "corpus_split_name": self._corpus_split_name,
            "corpus_size": len(self._corpus_records),
            "query_count": len(query_keys),
            "defaults": {
                "paper": dict(_PAPER_DEFAULTS),
                "official_repository": dict(_OFFICIAL_REPOSITORY_DEFAULTS),
                "actual_used": self._config.as_actual_used_dict(),
            },
            "continue_on_error": continue_on_error,
            "summary": {
                "processed": 0, "skipped_already_done": 0, "failed": 0,
                "excluded": 0, "fallback_used": 0,
            },
            "last_error": None,
        }
        _write_meta_atomic(meta_path, meta)

        processed = 0
        skipped = 0
        failed = 0
        excluded_count = 0
        fallback_count = 0

        errors_handle = (
            errors_path.open("a", encoding="utf-8")
            if (continue_on_error and errors_path is not None)
            else None
        )
        try:
            with output_path.open("a", encoding="utf-8") as output_handle:
                for query_key in query_keys:
                    if query_key in completed_keys:
                        skipped += 1
                        continue
                    try:
                        row = self.build_row(query_key)
                        line = json.dumps(_row_to_dict(row), ensure_ascii=False) + "\n"
                        output_handle.write(line)
                        output_handle.flush()
                        completed_keys.add(query_key)
                        processed += 1
                        if row.excluded:
                            excluded_count += 1
                        if row.fallback_used:
                            fallback_count += 1
                    except Exception as exc:
                        if not continue_on_error:
                            meta["run_status"] = "failed"
                            meta["run_finished_at_utc"] = _utc_now_iso()
                            meta["summary"] = {
                                "processed": processed,
                                "skipped_already_done": skipped,
                                "failed": failed,
                                "excluded": excluded_count,
                                "fallback_used": fallback_count,
                            }
                            meta["last_error"] = {
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                            _write_meta_atomic(meta_path, meta)
                            raise
                        error_payload = {
                            "dataset": query_key[0],
                            "patient_id": query_key[1],
                            "study_id": query_key[2],
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                        errors_handle.write(json.dumps(error_payload, ensure_ascii=False) + "\n")
                        errors_handle.flush()
                        failed += 1
        finally:
            if errors_handle is not None:
                errors_handle.close()

        meta["run_status"] = "completed"
        meta["run_finished_at_utc"] = _utc_now_iso()
        meta["summary"] = {
            "processed": processed,
            "skipped_already_done": skipped,
            "failed": failed,
            "excluded": excluded_count,
            "fallback_used": fallback_count,
        }
        _write_meta_atomic(meta_path, meta)

        return RAGDatasetBuildSummary(
            processed=processed,
            skipped_already_done=skipped,
            failed=failed,
            excluded=excluded_count,
            fallback_used=fallback_count,
        )
