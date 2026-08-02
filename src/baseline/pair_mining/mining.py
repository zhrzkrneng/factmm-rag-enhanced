"""Fact-aware positive report pair mining.

Responsibility: implement the paper's Eq. 1/2 factual-similarity mining
procedure over annotated records (the JSONL schema produced by
src.baseline.radgraph.annotator.RadGraphAnnotator) -- restrict
candidates to the query's own dataset, optionally exclude self and/or
same-patient candidates, score the remainder by
src.baseline.pair_mining.similarity, keep only candidates passing both
threshold masks, rank the survivors by a fully deterministic tie-break
order, and keep the top-k as positive pairs. Every threshold/top_k/
comparison-mode choice is a configurable, recorded value -- never a
silently hard-coded assumption (see docs/risk_register.md #1b, #1c,
and the new threshold-comparison and tie-breaking rows this milestone
adds).

Streaming, not matrix-based: one query is scored against the corpus at
a time, only qualifying candidates and the final top-k are retained,
and everything else is discarded immediately -- an n-by-n similarity
matrix is never materialized (see docs/risk_register.md #6 on why that
would be infeasible at real scale).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from src.baseline.pair_mining.similarity import chexbert_similarity, combined_score, radgraph_similarity
from src.common.exceptions import PairMiningError

QueryKey = Tuple[str, str, str]

REQUIRED_OUTPUT_FIELDS: Tuple[str, ...] = (
    "dataset",
    "patient_id",
    "study_id",
    "positive_keys",
    "scores",
    "num_candidates_considered",
    "num_qualifying_candidates",
    "num_selected",
    "flagged_bad_sample",
)

REQUIRED_SCORE_FIELDS: Tuple[str, ...] = (
    "chexbert_similarity",
    "radgraph_similarity",
    "combined_score",
)

# Paper Implementation Details: "we ... use the top 2 factual report
# pairs for each query" (PAPER_EXPLICIT). Recorded here, never invented,
# never silently substituted for the actual default below.
_PAPER_DEFAULTS = {
    "top_k": 2,
    "threshold_comparison": ">",
}

# data/factual_mining/{build_pos_train,build_pos_valid}/gen_topk_pos.sh
# shipped defaults (OFFICIAL_REPOSITORY): chex_thresh=1.0, top_k=3,
# radg_thresh=0.4; masks are `>=` in gen_topk_pos.py, not `>`.
_OFFICIAL_REPOSITORY_DEFAULTS = {
    "top_k": 3,
    "chex_threshold": 1.0,
    "radg_threshold": 0.4,
    "threshold_comparison": ">=",
    "exclude_self": True,
    "exclude_same_patient": False,
}

# Static description of the ranking rule (see PairMiner.mine_query) --
# recorded verbatim in run metadata, not derived from data.
_DETERMINISTIC_TIE_BREAKING_RULE = (
    "(-combined_score, dataset, patient_id, study_id) ascending: highest "
    "combined_score first, ties broken by lexicographic candidate ID. "
    "Never np.argpartition or any other implementation-defined/unstable "
    "ordering."
)


@dataclass(frozen=True)
class PairMiningConfig:
    config_version: str = "1.0"
    chex_threshold: float = 1.0
    radg_threshold: float = 0.4
    top_k: int = 3
    threshold_comparison: str = ">="  # ">=" or ">"
    exclude_self: bool = True
    exclude_same_patient: bool = False

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not (0.0 <= self.chex_threshold <= 1.0):
            raise ValueError(f"chex_threshold must be in [0.0, 1.0], got {self.chex_threshold!r}")
        if not (0.0 <= self.radg_threshold <= 1.0):
            raise ValueError(f"radg_threshold must be in [0.0, 1.0], got {self.radg_threshold!r}")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {self.top_k!r}")
        if self.threshold_comparison not in (">=", ">"):
            raise ValueError(
                f"threshold_comparison must be '>=' or '>', got {self.threshold_comparison!r}"
            )

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "top_k": self.top_k,
            "chex_threshold": self.chex_threshold,
            "radg_threshold": self.radg_threshold,
            "threshold_comparison": self.threshold_comparison,
            "exclude_self": self.exclude_self,
            "exclude_same_patient": self.exclude_same_patient,
        }


@dataclass(frozen=True)
class QueryMiningResult:
    query_key: QueryKey
    positive_keys: List[QueryKey]
    scores: List[Dict[str, float]]  # parallel to positive_keys
    num_candidates_considered: int
    num_qualifying_candidates: int
    num_selected: int
    flagged_bad_sample: bool


@dataclass(frozen=True)
class PairMiningRunSummary:
    processed: int
    skipped_already_done: int
    failed: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_meta_atomic(meta_path: Path, payload: dict) -> None:
    tmp_path = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(meta_path))


def _build_defaults_block(config: PairMiningConfig) -> dict:
    return {
        "paper": dict(_PAPER_DEFAULTS),
        "official_repository": dict(_OFFICIAL_REPOSITORY_DEFAULTS),
        "actual_used": config.as_actual_used_dict(),
    }


def _record_key(record: dict) -> QueryKey:
    return (record["dataset"], record["patient_id"], record["study_id"])


def _check_meta_compatibility(meta_path: Path, config: PairMiningConfig) -> None:
    """Rejects resuming against metadata written under a different config.

    A missing meta_path is fine (first run). An existing meta_path is
    only usable if its recorded `defaults.actual_used` exactly matches
    the current run's config -- including config_version. Any
    mismatch raises PairMiningError rather than silently resuming
    under an incompatible contract.
    """
    if not meta_path.exists():
        return
    try:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PairMiningError(f"Malformed existing metadata at {meta_path}: {exc}") from exc

    existing_actual = existing_meta.get("defaults", {}).get("actual_used")
    if existing_actual is None:
        raise PairMiningError(
            f"Existing metadata at {meta_path} is missing defaults.actual_used -- "
            f"cannot verify configuration compatibility, refusing to resume"
        )
    current_actual = config.as_actual_used_dict()
    if existing_actual != current_actual:
        raise PairMiningError(
            f"Existing metadata at {meta_path} was produced under a different "
            f"configuration ({existing_actual}) than the current run "
            f"({current_actual}) -- refusing to silently resume under an "
            f"incompatible config_version/config"
        )


def _load_completed_pair_mining_keys(
    output_path: Path,
    *,
    corpus_keys: Set[QueryKey],
    config: PairMiningConfig,
) -> Set[QueryKey]:
    """Validates any existing output_path and returns its completed query keys.

    Beyond the base checks shared with the annotator's resume pattern
    (well-formed JSON, all required fields present, no duplicate query
    key), each line is additionally validated for:

      - positive_keys and scores are both lists of equal length
      - num_selected == len(positive_keys)
      - num_selected <= config.top_k
      - every positive key is a member of the current candidate corpus
      - every positive key shares the query's dataset (cross-dataset
        invariant)
      - no positive key equals the query's own key when
        config.exclude_self is True
      - no positive key shares the query's patient_id when
        config.exclude_same_patient is True

    Any violation raises PairMiningError naming the file, line number,
    and specific problem -- existing output is never silently trusted
    or silently skipped as stale.
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
                raise PairMiningError(
                    f"Malformed existing output at {output_path} line {line_no}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in obj]
            if missing:
                raise PairMiningError(
                    f"Existing output at {output_path} line {line_no} "
                    f"missing required fields: {missing}"
                )

            query_key: QueryKey = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if query_key in completed:
                raise PairMiningError(
                    f"Duplicate query key {query_key} found in existing output "
                    f"at {output_path} line {line_no}"
                )

            positive_keys_raw = obj["positive_keys"]
            scores_raw = obj["scores"]
            if not isinstance(positive_keys_raw, list):
                raise PairMiningError(
                    f"{output_path} line {line_no}: positive_keys is not a list"
                )
            if not isinstance(scores_raw, list):
                raise PairMiningError(f"{output_path} line {line_no}: scores is not a list")
            if len(positive_keys_raw) != len(scores_raw):
                raise PairMiningError(
                    f"{output_path} line {line_no}: positive_keys "
                    f"({len(positive_keys_raw)}) and scores ({len(scores_raw)}) "
                    f"have different lengths"
                )

            num_selected = obj["num_selected"]
            if num_selected != len(positive_keys_raw):
                raise PairMiningError(
                    f"{output_path} line {line_no}: num_selected ({num_selected}) "
                    f"!= len(positive_keys) ({len(positive_keys_raw)})"
                )
            if num_selected > config.top_k:
                raise PairMiningError(
                    f"{output_path} line {line_no}: num_selected ({num_selected}) "
                    f"exceeds current config.top_k ({config.top_k}) -- output is "
                    f"stale relative to this run's configuration"
                )

            for raw_key in positive_keys_raw:
                positive_key: QueryKey = tuple(raw_key)
                if positive_key not in corpus_keys:
                    raise PairMiningError(
                        f"{output_path} line {line_no}: referenced positive key "
                        f"{positive_key} not found in the current candidate corpus"
                    )
                if positive_key[0] != query_key[0]:
                    raise PairMiningError(
                        f"{output_path} line {line_no}: positive key {positive_key} "
                        f"violates the cross-dataset invariant "
                        f"(query dataset={query_key[0]!r})"
                    )
                if config.exclude_self and positive_key == query_key:
                    raise PairMiningError(
                        f"{output_path} line {line_no}: self key present in "
                        f"positive_keys but config.exclude_self=True -- output is "
                        f"stale relative to this run's configuration"
                    )
                if config.exclude_same_patient and positive_key[1] == query_key[1]:
                    raise PairMiningError(
                        f"{output_path} line {line_no}: same-patient key "
                        f"{positive_key} present in positive_keys but "
                        f"config.exclude_same_patient=True -- output is stale "
                        f"relative to this run's configuration"
                    )

            completed.add(query_key)
    return completed


class PairMiner:
    """Mines top-k positive pairs for each query against a fixed corpus.

    corpus_records and query_records are plain dicts matching
    RadGraphAnnotator's JSONL output schema (dataset, patient_id,
    study_id, entities, chexbert_labels_5, ...). corpus_scope is a
    caller-supplied, purely descriptive label (e.g. "train_only")
    recorded in run metadata -- this class does not itself derive
    which split a record belongs to; scoping the corpus to the correct
    split (e.g. always train, per the paper's single retrieval index)
    is the caller's responsibility.
    """

    def __init__(
        self,
        corpus_records: List[dict],
        *,
        config: PairMiningConfig,
        corpus_scope: str = "train_only",
    ) -> None:
        self._corpus_records = list(corpus_records)
        self._config = config
        self._corpus_scope = corpus_scope
        self._corpus_keys: Set[QueryKey] = {_record_key(r) for r in self._corpus_records}

    def _passes_threshold(self, value: float, threshold: float) -> bool:
        if self._config.threshold_comparison == ">=":
            return value >= threshold
        return value > threshold

    def mine_query(self, query_record: dict) -> QueryMiningResult:
        query_key = _record_key(query_record)
        query_entities = query_record["entities"]
        query_labels_5 = query_record["chexbert_labels_5"]

        # Bad-sample check first (cheap: only needs the query's own
        # entities) -- an empty/degenerate entity set self-scores 0.0,
        # not 1.0, per radgraph_similarity's documented empty-set rule.
        self_similarity = radgraph_similarity(query_entities, query_entities)
        flagged_bad_sample = not math.isclose(self_similarity, 1.0, abs_tol=1e-9)

        same_dataset_candidates = [
            c for c in self._corpus_records if c["dataset"] == query_record["dataset"]
        ]
        num_candidates_considered = len(same_dataset_candidates)

        if flagged_bad_sample:
            return QueryMiningResult(
                query_key=query_key,
                positive_keys=[],
                scores=[],
                num_candidates_considered=num_candidates_considered,
                num_qualifying_candidates=0,
                num_selected=0,
                flagged_bad_sample=True,
            )

        qualifying: List[Tuple[QueryKey, float, float]] = []
        for candidate in same_dataset_candidates:
            candidate_key = _record_key(candidate)
            if self._config.exclude_self and candidate_key == query_key:
                continue
            if self._config.exclude_same_patient and candidate["patient_id"] == query_record["patient_id"]:
                continue

            chex_sim = chexbert_similarity(query_labels_5, candidate["chexbert_labels_5"])
            if not self._passes_threshold(chex_sim, self._config.chex_threshold):
                continue

            radg_sim = radgraph_similarity(query_entities, candidate["entities"])
            if not self._passes_threshold(radg_sim, self._config.radg_threshold):
                continue

            qualifying.append((candidate_key, chex_sim, radg_sim))

        num_qualifying_candidates = len(qualifying)

        # Deterministic ranking: (-combined_score, dataset, patient_id,
        # study_id) ascending -- highest score first, ties broken by
        # lexicographic candidate ID. Never np.argpartition or any
        # other implementation-defined/unstable ordering (see
        # docs/risk_register.md's new tie-breaking row).
        qualifying.sort(
            key=lambda item: (
                -combined_score(item[1], item[2]),
                item[0][0],
                item[0][1],
                item[0][2],
            )
        )

        selected = qualifying[: self._config.top_k]
        positive_keys = [item[0] for item in selected]
        scores = [
            {
                "chexbert_similarity": item[1],
                "radgraph_similarity": item[2],
                "combined_score": combined_score(item[1], item[2]),
            }
            for item in selected
        ]

        return QueryMiningResult(
            query_key=query_key,
            positive_keys=positive_keys,
            scores=scores,
            num_candidates_considered=num_candidates_considered,
            num_qualifying_candidates=num_qualifying_candidates,
            num_selected=len(positive_keys),
            flagged_bad_sample=False,
        )

    def mine_all(
        self,
        query_records: Iterable[dict],
        *,
        output_path: Path,
        errors_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        input_path: Optional[str] = None,
        continue_on_error: bool = False,
    ) -> PairMiningRunSummary:
        """Mines an iterable of query records, writing one JSONL line each.

        Raises ValueError immediately, before touching output_path or
        meta_path, if continue_on_error=True and errors_path is None.

        meta_path defaults to
        output_path.with_suffix(output_path.suffix + ".meta.json") when
        omitted.

        Before any processing: rejects (PairMiningError) an existing
        meta_path recorded under an incompatible config/config_version,
        then validates any existing output_path (see
        _load_completed_pair_mining_keys) to build the resume set.

        A query record already present as a valid completed key is
        skipped. Each new query's result is serialized in one write
        call and flushed immediately (crash-safe). In fail-fast mode
        (continue_on_error=False, the default) any exception updates
        the sidecar metadata to run_status="failed" with the error
        recorded, then re-raises unchanged -- the record is never
        marked completed. In continue_on_error mode the exception is
        logged to errors_path instead and the loop continues.
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
        # Materialized once (not left as a lazy iterable) so query_count
        # can be recorded in metadata without consuming the iterable
        # before the main loop uses it.
        query_records = list(query_records)

        _check_meta_compatibility(meta_path, self._config)
        completed_keys = _load_completed_pair_mining_keys(
            output_path, corpus_keys=self._corpus_keys, config=self._config
        )

        meta = {
            "run_status": "running",
            "run_started_at_utc": _utc_now_iso(),
            "run_finished_at_utc": None,
            "input_path": input_path if input_path is not None else "unspecified (records provided in-memory)",
            "output_path": str(output_path),
            "errors_path": str(errors_path) if errors_path is not None else None,
            "corpus_scope": self._corpus_scope,
            "corpus_size": len(self._corpus_records),
            "query_count": len(query_records),
            "deterministic_tie_breaking_rule": _DETERMINISTIC_TIE_BREAKING_RULE,
            "defaults": _build_defaults_block(self._config),
            "continue_on_error": continue_on_error,
            "summary": {"processed": 0, "skipped_already_done": 0, "failed": 0},
            "last_error": None,
        }
        _write_meta_atomic(meta_path, meta)

        processed = 0
        skipped = 0
        failed = 0

        errors_handle = (
            errors_path.open("a", encoding="utf-8")
            if (continue_on_error and errors_path is not None)
            else None
        )
        try:
            with output_path.open("a", encoding="utf-8") as output_handle:
                for record in query_records:
                    key = _record_key(record)
                    if key in completed_keys:
                        skipped += 1
                        continue
                    try:
                        result = self.mine_query(record)
                        payload = {
                            "dataset": result.query_key[0],
                            "patient_id": result.query_key[1],
                            "study_id": result.query_key[2],
                            "positive_keys": [list(pk) for pk in result.positive_keys],
                            "scores": result.scores,
                            "num_candidates_considered": result.num_candidates_considered,
                            "num_qualifying_candidates": result.num_qualifying_candidates,
                            "num_selected": result.num_selected,
                            "flagged_bad_sample": result.flagged_bad_sample,
                        }
                        line = json.dumps(payload, ensure_ascii=False) + "\n"
                        output_handle.write(line)
                        output_handle.flush()
                        completed_keys.add(key)
                        processed += 1
                    except Exception as exc:
                        if not continue_on_error:
                            meta["run_status"] = "failed"
                            meta["run_finished_at_utc"] = _utc_now_iso()
                            meta["summary"] = {
                                "processed": processed,
                                "skipped_already_done": skipped,
                                "failed": failed,
                            }
                            meta["last_error"] = {
                                "error_type": type(exc).__name__,
                                "error_message": str(exc),
                            }
                            _write_meta_atomic(meta_path, meta)
                            raise
                        error_payload = {
                            "dataset": record["dataset"],
                            "patient_id": record["patient_id"],
                            "study_id": record["study_id"],
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
        meta["summary"] = {"processed": processed, "skipped_already_done": skipped, "failed": failed}
        _write_meta_atomic(meta_path, meta)

        return PairMiningRunSummary(processed=processed, skipped_already_done=skipped, failed=failed)
