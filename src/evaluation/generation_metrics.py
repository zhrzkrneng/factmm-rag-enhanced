"""Generation quality metrics (Milestone 2.6).

Responsibility: compute F1CheXbert (micro-averaged F1 over the 5-class
observation subset), F1RadGraph (partial/RG_ER reward), ROUGE-L, BLEU-4,
and BERTScore between generated and reference reports, matching the
exact definitions used in the official FactMM-RAG evaluation script and
the paper (see docs/paper_analysis.md §15 and
docs/milestone_2_6_evaluation_contract.md).

Cell 30 implemented the config, input/output dataclasses, and the
reference/prediction loader (query_key alignment, duplicate/missing-key
validation, malformed-row handling, deterministic ordering, resume-safe
load metadata) -- everything §9.1/§11.1 of the contract needs before any
metric can be computed.

Cell 31 (this addition) implements three of the contract's §8 metric
wrappers -- ROUGE-L, BLEU-4, BERTScore -- behind one uniform
GenerationMetricWrapper interface, plus an explicit (never globally
mutable) metric registry and deterministic mock wrappers for testing
any future code that consumes the registry generically.
F1RadGraph/F1CheXbert (contract §8's other two wrappers), the
evaluate_generation() orchestrator that calls all five together,
retrieval metrics, bootstrap significance, and Oracle evaluation are
all explicitly out of scope here -- later cells. No external metric
library (radgraph, f1chexbert) is imported by this module; rouge,
evaluate, and bert_score ARE now real (lazy) dependencies of this
module's default factories -- never imported at module scope, only
inside a factory function invoked no earlier than a wrapper's first
real compute() call, and never at all when a fake factory is injected
(every unit test in this project injects fakes for these three; real
invocation is reserved for a dedicated Colab dry-run cell, matching
HFGeneratorAdapter's and RadGraphAnnotator's own established
discipline). Added to requirements.txt in this same change: rouge,
evaluate, bert-score (matching the precedent set when Pillow was added
alongside HFGeneratorAdapter in Milestone 2.5) -- pytrec_eval remains
deferred to its own future retrieval-metrics cell.

Pairing is strict query_key-based by default
(GenerationMetricsConfig.pairing_mode="by_query_key"): both reference
and prediction JSONL rows carry an explicit query_key, duplicates are
rejected, a prediction with no matching reference (or a reference with
no matching prediction row) is rejected, and pairing order is always
the sorted query_key order -- independent of on-disk row order. The
official evaluation.py's own behavior (raw positional zip, no keys, a
silent per-example malformed-hypothesis filter, see contract §3.1/§18)
is reproduced exactly, but only opt-in, via
pairing_mode="positional_official_reproduction".

A prediction row with a non-null "error" (mirroring
src.baseline.generation.generator's GeneratorResult.error, Milestone
2.5) represents a legitimate upstream generation failure, not a schema
violation -- it is never silently dropped: it is excluded from scoring
(there is no generated text to score) but always counted and named in
GenerationLoadSummary.failed_query_keys.
"""

from __future__ import annotations

import abc
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import EvaluationError

_PAIRING_MODES: Tuple[str, ...] = ("by_query_key", "positional_official_reproduction")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _write_meta_atomic(meta_path: Path, payload: dict) -> None:
    tmp_path = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(meta_path))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationMetricsConfig:
    """Contract §7.1. Metric-specific fields (radgraph_reward_level,
    chexbert_device, bert_score_*, bleu_enabled) are validated here but
    not yet consumed by any code in this cell -- Cell 31's metric
    wrappers are what actually read them."""

    config_version: str = "1.0"
    radgraph_reward_level: str = "partial"
    radgraph_model_type: str = "radgraph"
    chexbert_device: str = "cpu"
    bert_score_model_type: str = "distilbert-base-uncased"
    bert_score_num_layers: int = 5
    bert_score_batch_size: int = 64
    bert_score_rescale_with_baseline: bool = True
    bleu_enabled: bool = True
    pairing_mode: Literal["by_query_key", "positional_official_reproduction"] = "by_query_key"
    malformed_hypothesis_filter_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not self.radgraph_reward_level:
            raise ValueError("radgraph_reward_level must be a non-empty string")
        if not self.radgraph_model_type:
            raise ValueError("radgraph_model_type must be a non-empty string")
        if not self.chexbert_device:
            raise ValueError("chexbert_device must be a non-empty string")
        if not self.bert_score_model_type:
            raise ValueError("bert_score_model_type must be a non-empty string")
        if (
            not isinstance(self.bert_score_num_layers, int)
            or isinstance(self.bert_score_num_layers, bool)
            or self.bert_score_num_layers < 1
        ):
            raise ValueError(
                f"bert_score_num_layers must be a positive integer, got {self.bert_score_num_layers!r}"
            )
        if (
            not isinstance(self.bert_score_batch_size, int)
            or isinstance(self.bert_score_batch_size, bool)
            or self.bert_score_batch_size < 1
        ):
            raise ValueError(
                f"bert_score_batch_size must be a positive integer, got {self.bert_score_batch_size!r}"
            )
        if not isinstance(self.bert_score_rescale_with_baseline, bool):
            raise ValueError("bert_score_rescale_with_baseline must be a bool")
        if not isinstance(self.bleu_enabled, bool):
            raise ValueError("bleu_enabled must be a bool")
        if self.pairing_mode not in _PAIRING_MODES:
            raise ValueError(f"pairing_mode must be one of {_PAIRING_MODES}, got {self.pairing_mode!r}")
        if not isinstance(self.malformed_hypothesis_filter_enabled, bool):
            raise ValueError("malformed_hypothesis_filter_enabled must be a bool")

    def as_actual_used_dict(self) -> dict:
        return {
            "config_version": self.config_version,
            "radgraph_reward_level": self.radgraph_reward_level,
            "radgraph_model_type": self.radgraph_model_type,
            "chexbert_device": self.chexbert_device,
            "bert_score_model_type": self.bert_score_model_type,
            "bert_score_num_layers": self.bert_score_num_layers,
            "bert_score_batch_size": self.bert_score_batch_size,
            "bert_score_rescale_with_baseline": self.bert_score_rescale_with_baseline,
            "bleu_enabled": self.bleu_enabled,
            "pairing_mode": self.pairing_mode,
            "malformed_hypothesis_filter_enabled": self.malformed_hypothesis_filter_enabled,
        }


# ---------------------------------------------------------------------------
# Input row dataclasses (contract §11.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceRow:
    query_key: Optional[QueryKey]  # None only under positional_official_reproduction mode
    finding: str


@dataclass(frozen=True)
class PredictionRow:
    query_key: Optional[QueryKey]  # None only under positional_official_reproduction mode
    retrieved_finding: Optional[str]  # None iff error is set
    error: Optional[str] = None  # a legitimate upstream generation failure, never silently dropped


# ---------------------------------------------------------------------------
# Output dataclasses (contract §7.6) -- schema only, not yet populated by
# any metric computation in this cell.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationExampleScore:
    query_key: QueryKey
    f1_radgraph: float
    f1_chexbert_instance: float
    rouge_l: float
    bert_score: float


@dataclass(frozen=True)
class GenerationMetricsResult:
    config_version: str
    num_examples_scored: int
    num_examples_dropped: int
    f1_radgraph: float
    f1_chexbert: float
    rouge_l: float
    bleu4: Optional[float]
    bert_score: float
    per_example: Optional[Tuple[GenerationExampleScore, ...]]


# ---------------------------------------------------------------------------
# Load result / resume-safe metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationLoadSummary:
    config_version: str
    ref_path: str
    pred_path: str
    pairing_mode: str
    num_reference_rows: int
    num_prediction_rows: int
    num_paired: int
    num_failed_generations: int
    num_dropped_malformed_hypothesis: int
    failed_query_keys: Tuple[Union[QueryKey, int], ...]
    dropped_malformed_hypothesis_keys: Tuple[Union[QueryKey, int], ...]
    loaded_timestamp_utc: str
    git_commit: Optional[str]


@dataclass(frozen=True)
class GenerationLoadResult:
    query_keys: Tuple[Optional[QueryKey], ...]  # aligned 1:1 with hyps/refs
    hyps: Tuple[str, ...]
    refs: Tuple[str, ...]
    summary: GenerationLoadSummary


def write_generation_load_summary_atomic(meta_path: Path, summary: GenerationLoadSummary) -> None:
    """Writes summary as an atomic JSON sidecar, same shape/safety
    discipline as every other run-metadata sidecar in this project
    (RadGraphAnnotator/PairMiner/RAGDatasetBuilder/LLaVAGenerator's own
    meta.json files) -- reusable by later cells (e.g. report.py,
    Oracle's own resumable pipeline). This does NOT imply
    load_generation_pairs() itself is resumable: per contract §12, a
    one-shot scoring load is re-run in full every time, never resumed
    from a partial state -- this sidecar exists purely for provenance,
    not for skip-already-done logic."""
    def _serialize_identifier(identifier: Union[QueryKey, int]) -> object:
        # QueryKey (by_query_key mode) -> a 3-element list; positional mode
        # has no query_key at all, so a bare 0-based row index is recorded
        # instead -- never coerced into a fake 3-tuple.
        return list(identifier) if isinstance(identifier, tuple) else identifier

    payload = {
        "config_version": summary.config_version,
        "ref_path": summary.ref_path,
        "pred_path": summary.pred_path,
        "pairing_mode": summary.pairing_mode,
        "num_reference_rows": summary.num_reference_rows,
        "num_prediction_rows": summary.num_prediction_rows,
        "num_paired": summary.num_paired,
        "num_failed_generations": summary.num_failed_generations,
        "num_dropped_malformed_hypothesis": summary.num_dropped_malformed_hypothesis,
        "failed_query_keys": [_serialize_identifier(k) for k in summary.failed_query_keys],
        "dropped_malformed_hypothesis_keys": [
            _serialize_identifier(k) for k in summary.dropped_malformed_hypothesis_keys
        ],
        "loaded_timestamp_utc": summary.loaded_timestamp_utc,
        "git_commit": summary.git_commit,
    }
    _write_meta_atomic(Path(meta_path), payload)


# ---------------------------------------------------------------------------
# Malformed-hypothesis checks
# ---------------------------------------------------------------------------


def _is_blank(text: str) -> bool:
    """Strict-default malformed-hypothesis check: True iff text is
    empty or whitespace-only after stripping. Deliberately stricter
    than the official filter below -- a disclosed improvement, not a
    reproduction of official behavior (see contract §18)."""
    return len(text.strip()) == 0


def _official_malformed_hypothesis_check(text: str) -> bool:
    """Verbatim port of official evaluation.py's own filter (contract
    §3.1): True iff splitting on "." yields zero fragments whose *raw*
    (pre-normalization) length is greater than zero. Faithfully
    reproduces the official filter's own quirk -- a purely
    whitespace-only hyp (e.g. "  ") is NOT caught by this check (only a
    literally empty string, or a string made only of "." characters,
    is), because the official code's own `if len(_) > 0` guard runs
    before whitespace normalization. Used only under
    pairing_mode="positional_official_reproduction" +
    malformed_hypothesis_filter_enabled=True, for controlled A/B
    reproduction -- never the project's own default."""
    fragments = [" ".join(frag.split()) for frag in text.split(".") if len(frag) > 0]
    return len(fragments) == 0


# ---------------------------------------------------------------------------
# JSONL row parsing
# ---------------------------------------------------------------------------


def _parse_query_key(obj: dict, *, path: Path, line_no: int) -> QueryKey:
    raw = obj.get("query_key")
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or not all(isinstance(part, str) and part for part in raw)
    ):
        raise EvaluationError(
            f"{path} line {line_no}: 'query_key' must be a list of exactly 3 "
            f"non-empty strings, got {raw!r}"
        )
    return (raw[0], raw[1], raw[2])


def _parse_reference_line(
    obj: dict, *, path: Path, line_no: int, require_key: bool
) -> ReferenceRow:
    if "finding" not in obj:
        raise EvaluationError(f"{path} line {line_no}: missing required field 'finding'")
    finding = obj["finding"]
    if not isinstance(finding, str):
        raise EvaluationError(
            f"{path} line {line_no}: 'finding' must be a string, got {type(finding).__name__}"
        )
    query_key = _parse_query_key(obj, path=path, line_no=line_no) if require_key else None
    return ReferenceRow(query_key=query_key, finding=finding)


def _parse_prediction_line(
    obj: dict, *, path: Path, line_no: int, require_key: bool
) -> PredictionRow:
    has_finding = "retrieved_finding" in obj and obj["retrieved_finding"] is not None
    has_error = "error" in obj and obj["error"] is not None

    if has_finding and has_error:
        raise EvaluationError(
            f"{path} line {line_no}: a prediction row must carry exactly one of "
            f"'retrieved_finding' or 'error', got both"
        )
    if not has_finding and not has_error:
        raise EvaluationError(
            f"{path} line {line_no}: a prediction row must carry exactly one of "
            f"'retrieved_finding' or 'error', got neither"
        )

    retrieved_finding: Optional[str] = None
    error: Optional[str] = None

    if has_finding:
        raw = obj["retrieved_finding"]
        if not isinstance(raw, list) or len(raw) < 1 or not isinstance(raw[0], str):
            raise EvaluationError(
                f"{path} line {line_no}: 'retrieved_finding' must be a non-empty "
                f"list whose first element is a string, got {raw!r}"
            )
        retrieved_finding = raw[0]
    else:
        raw_error = obj["error"]
        if not isinstance(raw_error, str) or not raw_error:
            raise EvaluationError(
                f"{path} line {line_no}: 'error' must be a non-empty string, got {raw_error!r}"
            )
        error = raw_error

    query_key = _parse_query_key(obj, path=path, line_no=line_no) if require_key else None
    return PredictionRow(query_key=query_key, retrieved_finding=retrieved_finding, error=error)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path} line {line_no}: malformed JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise EvaluationError(
                    f"{path} line {line_no}: expected a JSON object per line, got {type(obj).__name__}"
                )
            yield line_no, obj


def load_reference_rows(path: Path, *, require_key: bool) -> List[ReferenceRow]:
    """Parses a JSONL reference file, one ReferenceRow per non-blank
    line, in file order. require_key=False (positional mode) never
    reads/validates a 'query_key' field even if present."""
    path = Path(path)
    rows: List[ReferenceRow] = []
    for line_no, obj in _iter_jsonl(path):
        rows.append(_parse_reference_line(obj, path=path, line_no=line_no, require_key=require_key))
    return rows


def load_prediction_rows(path: Path, *, require_key: bool) -> List[PredictionRow]:
    """Parses a JSONL prediction file, one PredictionRow per non-blank
    line, in file order. require_key=False (positional mode) never
    reads/validates a 'query_key' field even if present."""
    path = Path(path)
    rows: List[PredictionRow] = []
    for line_no, obj in _iter_jsonl(path):
        rows.append(_parse_prediction_line(obj, path=path, line_no=line_no, require_key=require_key))
    return rows


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _index_by_query_key(
    rows, *, path: Path, kind: str
) -> Dict[QueryKey, object]:
    indexed: Dict[QueryKey, object] = {}
    for row in rows:
        assert row.query_key is not None  # guaranteed by require_key=True at parse time
        if row.query_key in indexed:
            raise EvaluationError(
                f"{path}: duplicate query_key {row.query_key} found among {kind} rows"
            )
        indexed[row.query_key] = row
    return indexed


def _load_by_query_key(
    ref_path: Path, pred_path: Path, config: GenerationMetricsConfig
) -> GenerationLoadResult:
    ref_rows = load_reference_rows(ref_path, require_key=True)
    pred_rows = load_prediction_rows(pred_path, require_key=True)

    ref_by_key = _index_by_query_key(ref_rows, path=ref_path, kind="reference")
    pred_by_key = _index_by_query_key(pred_rows, path=pred_path, kind="prediction")

    pred_only = set(pred_by_key) - set(ref_by_key)
    if pred_only:
        raise EvaluationError(
            f"{pred_path}: {len(pred_only)} prediction query_key(s) have no matching "
            f"reference in {ref_path} (e.g. {sorted(pred_only)[0]}) -- every prediction "
            f"must reference a real query"
        )
    ref_only = set(ref_by_key) - set(pred_by_key)
    if ref_only:
        raise EvaluationError(
            f"{ref_path}: {len(ref_only)} reference query_key(s) have no matching "
            f"prediction row in {pred_path} (e.g. {sorted(ref_only)[0]}) -- every "
            f"reference must have a corresponding prediction row (successful or "
            f"failed-with-error), never silently absent"
        )

    query_keys: List[QueryKey] = []
    hyps: List[str] = []
    refs: List[str] = []
    failed_keys: List[QueryKey] = []
    dropped_keys: List[QueryKey] = []

    for query_key in sorted(ref_by_key.keys()):
        ref_row: ReferenceRow = ref_by_key[query_key]
        pred_row: PredictionRow = pred_by_key[query_key]

        if pred_row.error is not None:
            failed_keys.append(query_key)
            continue

        hyp_text = pred_row.retrieved_finding
        assert hyp_text is not None

        if config.malformed_hypothesis_filter_enabled:
            if _official_malformed_hypothesis_check(hyp_text):
                dropped_keys.append(query_key)
                continue
        else:
            if _is_blank(hyp_text):
                raise EvaluationError(
                    f"{pred_path}: prediction for query_key {query_key} has a blank/"
                    f"whitespace-only retrieved_finding -- rejected under the strict "
                    f"default (malformed_hypothesis_filter_enabled=False); set it to "
                    f"True to reproduce official evaluation.py's own silent-drop "
                    f"behavior instead"
                )

        query_keys.append(query_key)
        hyps.append(hyp_text)
        refs.append(ref_row.finding)

    summary = GenerationLoadSummary(
        config_version=config.config_version,
        ref_path=str(ref_path),
        pred_path=str(pred_path),
        pairing_mode=config.pairing_mode,
        num_reference_rows=len(ref_rows),
        num_prediction_rows=len(pred_rows),
        num_paired=len(query_keys),
        num_failed_generations=len(failed_keys),
        num_dropped_malformed_hypothesis=len(dropped_keys),
        failed_query_keys=tuple(failed_keys),
        dropped_malformed_hypothesis_keys=tuple(dropped_keys),
        loaded_timestamp_utc=_utc_now_iso(),
        git_commit=_resolve_git_commit(),
    )
    return GenerationLoadResult(
        query_keys=tuple(query_keys), hyps=tuple(hyps), refs=tuple(refs), summary=summary
    )


def _load_positional_official_reproduction(
    ref_path: Path, pred_path: Path, config: GenerationMetricsConfig
) -> GenerationLoadResult:
    ref_rows = load_reference_rows(ref_path, require_key=False)
    pred_rows = load_prediction_rows(pred_path, require_key=False)

    if len(ref_rows) != len(pred_rows):
        raise EvaluationError(
            f"{ref_path} has {len(ref_rows)} row(s) but {pred_path} has "
            f"{len(pred_rows)} row(s) -- positional pairing requires equal length, "
            f"matching official evaluation.py's own `assert len(hyps) == len(refs)`"
        )

    hyps: List[str] = []
    refs: List[str] = []
    failed_keys: List[int] = []
    dropped_keys: List[int] = []

    for position, (ref_row, pred_row) in enumerate(zip(ref_rows, pred_rows)):
        if pred_row.error is not None:
            failed_keys.append(position)  # no query_key available positionally -- 0-based row index instead
            continue

        hyp_text = pred_row.retrieved_finding
        assert hyp_text is not None

        if config.malformed_hypothesis_filter_enabled and _official_malformed_hypothesis_check(hyp_text):
            dropped_keys.append(position)
            continue

        hyps.append(hyp_text)
        refs.append(ref_row.finding)

    summary = GenerationLoadSummary(
        config_version=config.config_version,
        ref_path=str(ref_path),
        pred_path=str(pred_path),
        pairing_mode=config.pairing_mode,
        num_reference_rows=len(ref_rows),
        num_prediction_rows=len(pred_rows),
        num_paired=len(hyps),
        num_failed_generations=len(failed_keys),
        num_dropped_malformed_hypothesis=len(dropped_keys),
        failed_query_keys=tuple(failed_keys),
        dropped_malformed_hypothesis_keys=tuple(dropped_keys),
        loaded_timestamp_utc=_utc_now_iso(),
        git_commit=_resolve_git_commit(),
    )
    return GenerationLoadResult(
        query_keys=tuple(None for _ in hyps), hyps=tuple(hyps), refs=tuple(refs), summary=summary
    )


def load_generation_pairs(
    ref_path: Path, pred_path: Path, config: GenerationMetricsConfig
) -> GenerationLoadResult:
    """Loads and aligns reference/prediction JSONL files per
    config.pairing_mode (contract §9.1, §11.1):

    - "by_query_key" (default): both files must carry an explicit
      query_key; duplicates, a prediction with no matching reference,
      and a reference with no matching prediction row all raise
      EvaluationError. Pairing order is the sorted query_key order,
      independent of on-disk row order (deterministic regardless of how
      either file was written).
    - "positional_official_reproduction": no keys read at all; rows are
      paired by raw file order (zip), requiring equal file lengths --
      an exact port of official evaluation.py's own pairing (contract
      §3.1), opt-in only.

    In both modes, a prediction row with a non-null 'error' is a
    legitimate upstream generation failure: excluded from hyps/refs
    (nothing to score) but always counted, never silently absent from
    the returned GenerationLoadSummary.
    """
    ref_path = Path(ref_path)
    pred_path = Path(pred_path)
    if config.pairing_mode == "by_query_key":
        return _load_by_query_key(ref_path, pred_path, config)
    if config.pairing_mode == "positional_official_reproduction":
        return _load_positional_official_reproduction(ref_path, pred_path, config)
    raise EvaluationError(f"Unknown pairing_mode {config.pairing_mode!r}")  # unreachable: config validates this


# ---------------------------------------------------------------------------
# Metric wrapper interface (Cell 31)
# ---------------------------------------------------------------------------


def _validate_hyps_refs(metric_name: str, hyps: Sequence[str], refs: Sequence[str]) -> None:
    if len(hyps) != len(refs):
        raise EvaluationError(
            f"{metric_name}: hyps and refs must be the same length, got "
            f"{len(hyps)} vs {len(refs)}"
        )
    if len(hyps) == 0:
        raise EvaluationError(f"{metric_name}: cannot compute over zero examples")


def _classify_metric_dependency_error(exc: Exception) -> Tuple[str, str]:
    """Classifies exc into one of a fixed set of categories -- never a
    single generic bucket. Same walk-the-__cause__-chain +
    type-name-substring approach as
    src.baseline.generation.adapter._classify_generation_error (contract
    §14: "External library raises during a metric call -> EvaluationError,
    chained"), adapted to the exception types rouge/evaluate/bert_score
    (and their own huggingface_hub-based downloads, for BERTScore's
    checkpoint) actually raise."""
    seen = []
    cur: Optional[BaseException] = exc
    depth = 0
    while cur is not None and depth < 6:
        seen.append(cur)
        cur = getattr(cur, "__cause__", None)
        depth += 1

    type_names = {type(e).__module__ + "." + type(e).__name__ for e in seen}
    messages = " | ".join(str(e) for e in seen).lower()

    def has_type(substr: str) -> bool:
        return any(substr in name for name in type_names)

    if isinstance(exc, (ImportError, ModuleNotFoundError)) or has_type("ImportError"):
        return "missing_dependency", f"{type(exc).__name__}: {exc}"

    if has_type("GatedRepoError") or has_type("LocalTokenNotFoundError"):
        return "authentication_required", f"{type(exc).__name__}: {exc}"

    if has_type("RepositoryNotFoundError") or has_type("RevisionNotFoundError"):
        return "checkpoint_unavailable", f"{type(exc).__name__}: {exc}"

    if (
        has_type("ProxyError")
        or has_type("ConnectError")
        or has_type("ConnectTimeout")
        or has_type("ConnectionError")
        or has_type("TimeoutException")
        or has_type("Timeout")
        or "connection" in messages
    ):
        return "network_error", f"{type(exc).__name__}: {exc}"

    if isinstance(exc, (ValueError, IndexError, KeyError)):
        return "invalid_input", f"{type(exc).__name__}: {exc}"

    return "computation_failed", f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class MetricComputation:
    """The one uniform return shape every GenerationMetricWrapper.compute()
    produces. per_example_scores is None only for metrics with no
    per-example decomposition (BLEU-4 -- contract §7.6: the official
    evaluate.load("bleu") API is corpus-level-only, no per-example
    breakdown exists to report)."""

    metric_name: str
    corpus_score: float
    per_example_scores: Optional[Tuple[float, ...]]


class GenerationMetricWrapper(abc.ABC):
    """One uniform API for every generation metric: compute(hyps, refs) ->
    MetricComputation, nothing else. Per-metric configuration (which
    checkpoint, which device, ...) is bound once at construction time,
    never passed per-call -- so every concrete wrapper's compute()
    signature is identical regardless of what it wraps.

    No hidden global state: a wrapper instance's only mutable state is
    its own lazily-constructed, instance-scoped library handle (never a
    module-level shared object) -- see _LazyLibraryMixin.
    """

    name: str

    @abc.abstractmethod
    def compute(self, hyps: Sequence[str], refs: Sequence[str]) -> MetricComputation:
        ...


class _LazyLibraryMixin:
    """Shared lazy-construction-and-cache behavior: the real (or
    injected-fake) library object is never constructed until the first
    real compute() call, and is cached on the instance (not a module
    global) for reuse across subsequent calls -- matching official
    evaluation.py's own single-construction-per-run pattern, and
    HFGeneratorAdapter's established lazy-factory discipline."""

    _factory: Callable[[], object]
    _instance: Optional[object]

    def _get_instance(self, *, metric_name: str) -> object:
        if self._instance is None:
            try:
                self._instance = self._factory()
            except Exception as exc:
                if isinstance(exc, EvaluationError):
                    raise
                category, detail = _classify_metric_dependency_error(exc)
                raise EvaluationError(
                    f"{metric_name}: failed to construct dependency: {category}: {detail}"
                ) from exc
        return self._instance


class RougeLMetric(_LazyLibraryMixin, GenerationMetricWrapper):
    """Wraps the `rouge` PyPI package's Rouge().get_scores() (contract
    §3.1, §4, §18 -- NOT Google's rouge-score / HF evaluate's built-in
    "rouge", which compute ROUGE-L differently; never substituted).
    get_scores(..., avg=False) is called once to obtain per-example
    scores, and the corpus score is this project's own mean of those
    per-example scores -- exactly what the library's own avg=True mode
    does internally, so this avoids a second, redundant library call."""

    name = "rouge_l"

    def __init__(self, *, rouge_factory: Optional[Callable[[], object]] = None) -> None:
        self._factory = rouge_factory or _default_rouge_factory
        self._instance: Optional[object] = None

    def compute(self, hyps: Sequence[str], refs: Sequence[str]) -> MetricComputation:
        _validate_hyps_refs(self.name, hyps, refs)
        rouge = self._get_instance(metric_name=self.name)
        try:
            per_example_raw = rouge.get_scores(list(hyps), list(refs), avg=False)
            per_example_scores = tuple(item["rouge-l"]["f"] for item in per_example_raw)
        except Exception as exc:
            if isinstance(exc, EvaluationError):
                raise
            category, detail = _classify_metric_dependency_error(exc)
            raise EvaluationError(f"{self.name}: {category}: {detail}") from exc
        corpus_score = sum(per_example_scores) / len(per_example_scores)
        return MetricComputation(
            metric_name=self.name, corpus_score=corpus_score, per_example_scores=per_example_scores
        )


class Bleu4Metric(_LazyLibraryMixin, GenerationMetricWrapper):
    """Wraps HuggingFace `evaluate.load("bleu")` (contract §3.1, §4).
    Corpus-level only -- results['precisions'][3] is the 4-gram
    precision component the official code reports as "BLEU-4", NOT the
    geometric-mean BLEU score. No per-example decomposition exists in
    this library's API (contract §7.6), so per_example_scores is always
    None here -- never approximated or faked."""

    name = "bleu4"

    def __init__(self, *, bleu_factory: Optional[Callable[[], object]] = None) -> None:
        self._factory = bleu_factory or _default_bleu_factory
        self._instance: Optional[object] = None

    def compute(self, hyps: Sequence[str], refs: Sequence[str]) -> MetricComputation:
        _validate_hyps_refs(self.name, hyps, refs)
        bleu = self._get_instance(metric_name=self.name)
        try:
            results = bleu.compute(predictions=list(hyps), references=[[r] for r in refs])
            corpus_score = float(results["precisions"][3])
        except Exception as exc:
            if isinstance(exc, EvaluationError):
                raise
            category, detail = _classify_metric_dependency_error(exc)
            raise EvaluationError(f"{self.name}: {category}: {detail}") from exc
        return MetricComputation(metric_name=self.name, corpus_score=corpus_score, per_example_scores=None)


class BertScoreMetric(_LazyLibraryMixin, GenerationMetricWrapper):
    """Wraps `bert_score.BERTScorer` (contract §3.1, §4), constructed
    from GenerationMetricsConfig's bert_score_* fields. device reuses
    config.chexbert_device -- not a naming mistake: the official
    evaluation.py passes one single shared `--device` CLI flag to BOTH
    F1CheXbert and BERTScorer (contract §3.1); reusing the same config
    field here is MORE faithful to that shared-device behavior than
    introducing a second, independently-settable device field would be."""

    name = "bert_score"

    def __init__(
        self,
        config: GenerationMetricsConfig,
        *,
        bert_scorer_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        self._config = config
        self._factory = bert_scorer_factory or (lambda: _default_bert_scorer_factory(config))
        self._instance: Optional[object] = None

    def compute(self, hyps: Sequence[str], refs: Sequence[str]) -> MetricComputation:
        _validate_hyps_refs(self.name, hyps, refs)
        scorer = self._get_instance(metric_name=self.name)
        try:
            _, _, f = scorer.score(cands=list(hyps), refs=list(refs))
            per_example_scores = tuple(float(x) for x in f.tolist())
        except Exception as exc:
            if isinstance(exc, EvaluationError):
                raise
            category, detail = _classify_metric_dependency_error(exc)
            raise EvaluationError(f"{self.name}: {category}: {detail}") from exc
        corpus_score = sum(per_example_scores) / len(per_example_scores)
        return MetricComputation(
            metric_name=self.name, corpus_score=corpus_score, per_example_scores=per_example_scores
        )


def _default_rouge_factory() -> object:
    from rouge import Rouge  # lazy: only imported when actually constructing, never at module scope

    return Rouge()


def _default_bleu_factory() -> object:
    import evaluate  # lazy: only imported when actually constructing, never at module scope

    return evaluate.load("bleu")


def _default_bert_scorer_factory(config: GenerationMetricsConfig) -> object:
    from bert_score import BERTScorer  # lazy: only imported when actually constructing

    return BERTScorer(
        model_type=config.bert_score_model_type,
        num_layers=config.bert_score_num_layers,
        batch_size=config.bert_score_batch_size,
        nthreads=4,
        all_layers=False,
        idf=False,
        device=config.chexbert_device,
        lang="en",
        rescale_with_baseline=config.bert_score_rescale_with_baseline,
        baseline_path=None,
    )


def _word_overlap_ratio(hyp: str, ref: str) -> float:
    """Pure, deterministic Jaccard word overlap -- MockGenerationMetricWrapper's
    scoring function. No randomness anywhere: same (hyp, ref) always
    produces the exact same score, on any run, in any order."""
    hyp_words = set(hyp.lower().split())
    ref_words = set(ref.lower().split())
    if not hyp_words or not ref_words:
        return 0.0
    return len(hyp_words & ref_words) / len(hyp_words | ref_words)


class MockGenerationMetricWrapper(GenerationMetricWrapper):
    """Deterministic, no-ML-dependency fake conforming to the exact same
    GenerationMetricWrapper interface as the three real wrappers above --
    for unit-testing any code that consumes a metric (or the registry)
    generically, without invoking rouge/evaluate/bert_score at all.
    Scores a pure word-overlap ratio (see _word_overlap_ratio); a custom
    score_fn may be injected for tests that need specific, controlled
    values instead."""

    def __init__(
        self,
        name: str = "mock_metric",
        *,
        score_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.name = name
        self._score_fn = score_fn or _word_overlap_ratio

    def compute(self, hyps: Sequence[str], refs: Sequence[str]) -> MetricComputation:
        _validate_hyps_refs(self.name, hyps, refs)
        per_example_scores = tuple(self._score_fn(h, r) for h, r in zip(hyps, refs))
        corpus_score = sum(per_example_scores) / len(per_example_scores)
        return MetricComputation(
            metric_name=self.name, corpus_score=corpus_score, per_example_scores=per_example_scores
        )


# ---------------------------------------------------------------------------
# Metric registry -- explicit, freshly-constructed, never a shared
# mutable module-level dict (no hidden global state: two calls return
# fully independent wrapper instances with independent lazy-construction
# caches).
# ---------------------------------------------------------------------------

_REGISTERED_METRIC_NAMES: Tuple[str, ...] = ("rouge_l", "bleu4", "bert_score")


def build_metric_registry(
    config: GenerationMetricsConfig,
    *,
    rouge_factory: Optional[Callable[[], object]] = None,
    bleu_factory: Optional[Callable[[], object]] = None,
    bert_scorer_factory: Optional[Callable[[], object]] = None,
) -> Dict[str, GenerationMetricWrapper]:
    """Builds the real (lazy) generation-metric registry for this cell's
    scope: rouge_l and bert_score are always present; bleu4 is present
    iff config.bleu_enabled (contract §7.1 -- bleu_enabled exists
    precisely so BLEU-4's OFFICIAL_REPOSITORY-only, non-paper-reported
    status, contract §2/§4/§18, can be turned off without touching call
    sites). F1RadGraph/F1CheXbert are not registered here -- a later
    cell's scope."""
    registry: Dict[str, GenerationMetricWrapper] = {
        "rouge_l": RougeLMetric(rouge_factory=rouge_factory),
        "bert_score": BertScoreMetric(config, bert_scorer_factory=bert_scorer_factory),
    }
    if config.bleu_enabled:
        registry["bleu4"] = Bleu4Metric(bleu_factory=bleu_factory)
    return registry


def build_mock_metric_registry(
    names: Sequence[str] = _REGISTERED_METRIC_NAMES,
) -> Dict[str, GenerationMetricWrapper]:
    """Builds an all-mock registry with the same key shape as
    build_metric_registry's default (bleu_enabled=True) output -- for
    exercising any registry-consuming code (e.g. a future
    evaluate_generation()) without any real metric library at all."""
    return {name: MockGenerationMetricWrapper(name=name) for name in names}
