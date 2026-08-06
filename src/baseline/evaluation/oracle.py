"""Oracle baseline construction (Milestone 2.6, Cell 35).

Responsibility: the small-scale, brute-force Oracle mechanism defined
exactly per the paper's Appendix A.3 and confirmed against the official
code (contract §2, §3.3, §10): for each query, argmax over
s(qi, dj) = F1CheXbert_instance(qi, dj) + F1RadGraph_instance(qi, dj)
across corpus candidates (excluding self per config.exclude_self), then
take the winning candidate's finding text VERBATIM as the "prediction"
-- zero generation, matching official knn_index_to_evaluation_file.py's
own mechanism exactly.

This is explicitly NOT the official large-scale pipeline (contract
§3.3, §18): the real repository computes a full O(N x N) pairwise
score matrix via gen_similarity.py, a 64-shard SLURM array job (hours
of compute per shard), before gen_topk_pos.py/knn_ideal.py select the
per-query top-k. This cell implements ONLY the small-scale version --
a direct per-query loop over corpus_records, scoring each candidate
on-the-fly via injected scorer callbacks (chexbert_instance_scorer,
radgraph_instance_scorer) -- no precomputed matrix, no SLURM, no
large-scale construction of any kind. Real, full-corpus Oracle
construction remains genuinely infeasible in this environment (§18)
and is not attempted here or anywhere in this project.

Kept clearly separated from the normal evaluation pipeline
(requirement #2): OracleEvaluator has no dependency on, and is not
depended on by, src.evaluation.report.EvaluationRunner (Cell 34) --
the two are entirely independent code paths. The one, deliberate,
one-way bridge between them is oracle_result_row_to_pred_row_dict(),
which converts an OracleResultRow into the exact pred_rows JSONL shape
Cell 30's load_generation_pairs/PredictionRow already consumes
(contract §9.1, §11.1) -- so an Oracle build's output CAN be fed into
the same generation-metrics scoring pipeline as any real generator's
output, without the two implementations knowing anything about each
other beyond that one shared schema. This bridge is structurally
tested (the converted dict's shape is verified against Cell 30's own
parser) but never invoked end-to-end here -- no real evaluation run is
performed by this cell.

Resume/atomic-write/continue_on_error behavior follows the exact
established pattern from src.baseline.pair_mining.mining.PairMiner,
src.baseline.radgraph.annotator.RadGraphAnnotator,
src.baseline.generation.dataset_builder.RAGDatasetBuilder, and
src.baseline.generation.generator.LLaVAGenerator -- the fifth
implementation of this same pattern in this project (contract §12),
not a new one. Deterministic given fixed query_records/corpus_records/
scorer callbacks (contract §13): ties in s(qi, dj) are broken by lowest
corpus-iteration index (Python's sorted() reverse=True remains stable
for equal keys -- verified directly by this cell's own tie-breaking
test), never left to incidental dict/argmax ordering.

Mock oracle pipeline (this cell's own requirement): OracleEvaluator's
scorer callbacks are dependency-injected from the start (matching
RAGDatasetBuilder's/PairMiner's own established DI discipline) -- every
unit test in this cell constructs a real OracleEvaluator with small,
deterministic FAKE scorer functions, never a real CheXbert/RadGraph
instance-scoring call. No separate "MockOracleEvaluator" class exists;
none is needed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import EvaluationError
from src.data.schema import ReportRecord

REQUIRED_OUTPUT_FIELDS: Tuple[str, ...] = (
    "dataset",
    "patient_id",
    "study_id",
    "winning_key",
    "winning_score",
    "winning_finding_text",
    "excluded",
    "num_candidates_considered",
    "top_k_keys",
    "top_k_scores",
    "config_version",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_meta_atomic(meta_path: Path, payload: dict) -> None:
    tmp_path = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(meta_path))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleConfig:
    """Contract §7.3."""

    config_version: str = "1.0"
    corpus_scope: str = "train_only"
    exclude_self: bool = True
    top_k_candidates_considered: int = 30

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.corpus_scope:
            raise ValueError("corpus_scope must be a non-empty string")
        if not isinstance(self.exclude_self, bool):
            raise ValueError("exclude_self must be a bool")
        if (
            not isinstance(self.top_k_candidates_considered, int)
            or isinstance(self.top_k_candidates_considered, bool)
            or self.top_k_candidates_considered < 1
        ):
            raise ValueError(
                f"top_k_candidates_considered must be a positive integer, got "
                f"{self.top_k_candidates_considered!r}"
            )

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "corpus_scope": self.corpus_scope,
            "exclude_self": self.exclude_self,
            "top_k_candidates_considered": self.top_k_candidates_considered,
        }


# ---------------------------------------------------------------------------
# s(qi, dj) -- contract §2 Appendix A.3 / §10
# ---------------------------------------------------------------------------


def compute_oracle_score(chex_instance_score: float, radg_instance_score: float) -> float:
    """s(qi, dj) = F1CheXbert_instance + F1RadGraph_instance (contract §2
    Appendix A.3, verbatim gen_topk_pos.py's own `tensor = chexbert +
    radgraph`, §3.3) -- NOT Equation 1's normalized-overlap formula, a
    distinct computation confirmed independently from both the paper
    text and the official code."""
    return chex_instance_score + radg_instance_score


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OracleResultRow:
    query_key: QueryKey
    winning_key: Optional[QueryKey]  # None iff excluded (no eligible candidate)
    winning_score: Optional[float]
    winning_finding_text: Optional[str]  # the verbatim corpus text used as the "prediction"
    excluded: bool
    num_candidates_considered: int
    top_k_keys: Tuple[QueryKey, ...]  # up to config.top_k_candidates_considered, sorted desc by score
    top_k_scores: Tuple[float, ...]
    config_version: str


@dataclass(frozen=True)
class OracleBuildSummary:
    processed: int
    skipped_already_done: int
    failed: int
    excluded: int


def oracle_result_row_to_pred_row_dict(row: OracleResultRow) -> dict:
    """Converts an OracleResultRow into the exact pred_rows JSONL schema
    Cell 30's load_generation_pairs/PredictionRow consumes (contract
    §9.1, §11.1) -- see module docstring. An excluded row (no eligible
    candidate) is represented via the 'error' field, exactly matching
    how Cell 30 already represents a legitimate upstream "nothing to
    score" outcome -- never silently omitted."""
    obj: dict = {"query_key": list(row.query_key)}
    if row.excluded:
        obj["error"] = "oracle_excluded_no_eligible_candidate"
    else:
        obj["retrieved_finding"] = [row.winning_finding_text]
    return obj


def _row_to_dict(row: OracleResultRow) -> dict:
    return {
        "dataset": row.query_key[0],
        "patient_id": row.query_key[1],
        "study_id": row.query_key[2],
        "winning_key": list(row.winning_key) if row.winning_key is not None else None,
        "winning_score": row.winning_score,
        "winning_finding_text": row.winning_finding_text,
        "excluded": row.excluded,
        "num_candidates_considered": row.num_candidates_considered,
        "top_k_keys": [list(k) for k in row.top_k_keys],
        "top_k_scores": list(row.top_k_scores),
        "config_version": row.config_version,
    }


def _check_meta_compatibility(meta_path: Path, config: OracleConfig) -> None:
    """Rejects resuming against metadata written under a different config
    -- same pattern as RAGDatasetBuilder's own _check_meta_compatibility."""
    if not meta_path.exists():
        return
    try:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"Malformed existing metadata at {meta_path}: {exc}") from exc

    existing_actual = existing_meta.get("defaults", {}).get("actual_used")
    if existing_actual is None:
        raise EvaluationError(
            f"Existing metadata at {meta_path} is missing defaults.actual_used -- "
            f"cannot verify configuration compatibility, refusing to resume"
        )
    current_actual = config.as_actual_used_dict()
    if existing_actual != current_actual:
        raise EvaluationError(
            f"Existing metadata at {meta_path} was produced under a different "
            f"configuration ({existing_actual}) than the current run "
            f"({current_actual}) -- refusing to silently resume under an "
            f"incompatible config"
        )


def _load_completed_oracle_keys(output_path: Path) -> Set[QueryKey]:
    """Validates any existing output_path and returns its completed query
    keys -- same pattern as RAGDatasetBuilder's own
    _load_completed_rag_dataset_keys, adapted to OracleResultRow's own
    required fields."""
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
                raise EvaluationError(
                    f"Malformed existing output at {output_path} line {line_no}: {exc}"
                ) from exc

            missing = [field for field in REQUIRED_OUTPUT_FIELDS if field not in obj]
            if missing:
                raise EvaluationError(
                    f"Existing output at {output_path} line {line_no} missing "
                    f"required fields: {missing}"
                )

            query_key: QueryKey = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if query_key in completed:
                raise EvaluationError(
                    f"Duplicate query key {query_key} found in existing output "
                    f"at {output_path} line {line_no}"
                )

            excluded = obj["excluded"]
            if excluded:
                if (
                    obj["winning_key"] is not None
                    or obj["winning_score"] is not None
                    or obj["winning_finding_text"] is not None
                ):
                    raise EvaluationError(
                        f"{output_path} line {line_no}: excluded=True but carries a "
                        f"winning_key/winning_score/winning_finding_text inconsistent "
                        f"with exclusion"
                    )
            else:
                if (
                    obj["winning_key"] is None
                    or obj["winning_score"] is None
                    or obj["winning_finding_text"] is None
                ):
                    raise EvaluationError(
                        f"{output_path} line {line_no}: excluded=False but is missing "
                        f"winning_key/winning_score/winning_finding_text"
                    )

            completed.add(query_key)
    return completed


# ---------------------------------------------------------------------------
# OracleEvaluator
# ---------------------------------------------------------------------------


class OracleEvaluator:
    """Builds one OracleResultRow per query via a direct, small-scale,
    per-candidate scoring loop -- see module docstring. Never touches
    FAISS/embeddings/a precomputed matrix; the scorer callbacks are the
    entire seam between this class and real chexbert/radgraph instance
    scoring (dependency injection, mirroring RAGDatasetBuilder's own
    discipline)."""

    def __init__(
        self,
        query_records: Dict[QueryKey, ReportRecord],
        corpus_records: Dict[QueryKey, ReportRecord],
        *,
        config: OracleConfig,
        chexbert_instance_scorer: Callable[[str, str], float],
        radgraph_instance_scorer: Callable[[str, str], float],
    ) -> None:
        if not query_records:
            raise EvaluationError("OracleEvaluator: query_records must be non-empty")
        if not corpus_records:
            raise EvaluationError("OracleEvaluator: corpus_records must be non-empty")
        self._query_records = query_records
        self._corpus_records = corpus_records
        self._config = config
        self._chexbert_instance_scorer = chexbert_instance_scorer
        self._radgraph_instance_scorer = radgraph_instance_scorer

    def build_row(self, query_key: QueryKey) -> OracleResultRow:
        if query_key not in self._query_records:
            raise EvaluationError(f"Query key {query_key} has no matching query record")
        query_record = self._query_records[query_key]

        scored: List[Tuple[QueryKey, float]] = []
        for candidate_key, candidate_record in self._corpus_records.items():
            if self._config.exclude_self and candidate_key == query_key:
                continue
            chex_score = self._chexbert_instance_scorer(query_record.finding, candidate_record.finding)
            radg_score = self._radgraph_instance_scorer(query_record.finding, candidate_record.finding)
            scored.append((candidate_key, compute_oracle_score(chex_score, radg_score)))

        if not scored:
            return OracleResultRow(
                query_key=query_key,
                winning_key=None,
                winning_score=None,
                winning_finding_text=None,
                excluded=True,
                num_candidates_considered=0,
                top_k_keys=(),
                top_k_scores=(),
                config_version=self._config.config_version,
            )

        # Stable sort: Python's sorted(..., reverse=True) remains stable
        # for equal keys (documented CPython guarantee), so ties resolve
        # to the LOWEST corpus-iteration index -- contract §13's own
        # documented tie-breaking rule, verified directly by this cell's
        # own tie-breaking test, not left to incidental ordering.
        scored_sorted = sorted(scored, key=lambda item: item[1], reverse=True)
        top_k = scored_sorted[: self._config.top_k_candidates_considered]
        winning_key, winning_score = scored_sorted[0]

        return OracleResultRow(
            query_key=query_key,
            winning_key=winning_key,
            winning_score=winning_score,
            winning_finding_text=self._corpus_records[winning_key].finding,
            excluded=False,
            num_candidates_considered=len(scored),
            top_k_keys=tuple(key for key, _ in top_k),
            top_k_scores=tuple(score for _, score in top_k),
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
    ) -> OracleBuildSummary:
        """Builds a row for each of query_keys, writing one JSONL line
        each. Resume/atomic-write/continue_on_error behavior identical to
        RAGDatasetBuilder.build_all (see that class's own docstring) --
        the fifth implementation of this pattern in this project, not a
        new one. Raises ValueError immediately, before touching
        output_path or meta_path, if continue_on_error=True and
        errors_path is None."""
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
        completed_keys = _load_completed_oracle_keys(output_path)

        meta = {
            "run_status": "running",
            "run_started_at_utc": _utc_now_iso(),
            "run_finished_at_utc": None,
            "input_path": input_path if input_path is not None else "unspecified (records provided in-memory)",
            "output_path": str(output_path),
            "errors_path": str(errors_path) if errors_path is not None else None,
            "corpus_size": len(self._corpus_records),
            "query_count": len(query_keys),
            "defaults": {"actual_used": self._config.as_actual_used_dict()},
            "continue_on_error": continue_on_error,
            "summary": {"processed": 0, "skipped_already_done": 0, "failed": 0, "excluded": 0},
            "last_error": None,
        }
        _write_meta_atomic(meta_path, meta)

        processed = 0
        skipped = 0
        failed = 0
        excluded_count = 0

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
                    except Exception as exc:
                        if not continue_on_error:
                            meta["run_status"] = "failed"
                            meta["run_finished_at_utc"] = _utc_now_iso()
                            meta["summary"] = {
                                "processed": processed,
                                "skipped_already_done": skipped,
                                "failed": failed,
                                "excluded": excluded_count,
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
        }
        _write_meta_atomic(meta_path, meta)

        return OracleBuildSummary(
            processed=processed, skipped_already_done=skipped, failed=failed, excluded=excluded_count
        )
