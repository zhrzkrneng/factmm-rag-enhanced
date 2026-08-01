"""RadGraph + CheXbert annotation pipeline.

Responsibility: wrap the RadGraph and CheXbert annotation process behind
a modular interface, matching the official FactMM-RAG data/label.py:
only ``ReportRecord.finding`` is annotated (never ``impression`` or the
concatenated form), RadGraph is called with a single-item list and its
``entities`` dict is kept verbatim, and F1CheXbert's 14-class label
vector is reduced to the paper's 5-class subset via fixed indices
(``CHEXBERT_5_INDICES``) -- exactly matching the official
``np.isin``-derived index selection, without depending on numpy.

Compatibility concerns (Python-version/API shims, checkpoint cache
placement) live entirely in ``src.baseline.radgraph.compat`` and are
applied via a single ``compat.patch_all(...)`` call in
``RadGraphAnnotator.__init__`` -- this module contains no shim logic of
its own and never imports ``radgraph``/``f1chexbert`` at module level,
only inside the default factories, so that unit tests (which inject
fake factories) never trigger a real import.

Scope reminder (same as Cell 14): this module verifies execution
compatibility and produces well-formed output of the expected shape --
it does not verify clinical or metric correctness. Correctness/quality
validation belongs to Milestone 2.3 (pair-mining validation) and
Milestone 2.6 (evaluation).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Set, Tuple

from src.baseline.radgraph import compat
from src.common.exceptions import AnnotationError
from src.data.schema import ReportRecord

# Exact upstream ordering (reference/original_repository/FactMM-RAG/data/label.py) --
# f1chexbert.get_label() returns a bare list with no field names, so this
# ordering must never be silently assumed to be different.
CHEXBERT_14_CLASSES: Tuple[str, ...] = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
)
# Matches official code's `target_names_5_index = np.where(np.isin(target_names, target_names_5))[0]`
# for target_names_5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"],
# expressed as fixed indices instead of a numpy dependency.
CHEXBERT_5_INDICES: Tuple[int, ...] = (1, 4, 5, 7, 9)
CHEXBERT_5_CLASSES: Tuple[str, ...] = tuple(CHEXBERT_14_CLASSES[i] for i in CHEXBERT_5_INDICES)
CHEXBERT_LABEL_DOMAIN = frozenset({0, 1})

REQUIRED_SUCCESS_FIELDS: Tuple[str, ...] = (
    "dataset",
    "patient_id",
    "study_id",
    "finding",
    "entities",
    "chexbert_labels_14",
    "chexbert_labels_5",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_meta_atomic(meta_path: Path, payload: dict) -> None:
    tmp_path = meta_path.with_name(meta_path.name + f".tmp{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp_path), str(meta_path))


def _load_completed_keys(output_path: Path) -> Set[Tuple[str, str, str]]:
    """Validate any existing output_path and return its completed keys.

    Blank lines (e.g. a trailing newline) are skipped. Any non-blank
    line that fails to parse as JSON, or is missing a required success
    field, raises AnnotationError naming the file and line number --
    existing output is never silently ignored. A (dataset, patient_id,
    study_id) key seen twice among valid lines also raises.
    """
    if not output_path.exists():
        return set()

    completed: Set[Tuple[str, str, str]] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise AnnotationError(
                    f"Malformed existing output at {output_path} line {line_no}: {exc}"
                ) from exc
            missing = [field for field in REQUIRED_SUCCESS_FIELDS if field not in obj]
            if missing:
                raise AnnotationError(
                    f"Existing output at {output_path} line {line_no} "
                    f"missing required fields: {missing}"
                )
            key = (obj["dataset"], obj["patient_id"], obj["study_id"])
            if key in completed:
                raise AnnotationError(
                    f"Duplicate completed key {key} found in existing "
                    f"output at {output_path} line {line_no}"
                )
            completed.add(key)
    return completed


def _default_radgraph_factory():
    from radgraph import RadGraph

    return RadGraph()


def _default_chexbert_factory():
    from f1chexbert import F1CheXbert

    return F1CheXbert()


@dataclass(frozen=True)
class AnnotationRunSummary:
    processed: int
    skipped_already_done: int
    failed: int


class RadGraphAnnotator:
    """Constructs RadGraph/F1CheXbert once per process and annotates
    ReportRecords one at a time, with resumable, crash-safe JSONL output.
    """

    def __init__(
        self,
        *,
        radgraph_factory: Optional[Callable[[], object]] = None,
        chexbert_factory: Optional[Callable[[], object]] = None,
        allow_experimental_dataset_field: bool = True,
    ) -> None:
        """
        Args:
            radgraph_factory: injectable for testing. Must be provided
                together with chexbert_factory, or not at all -- partial
                injection is a test-only escape hatch, not general DI.
                When both are omitted, this calls
                compat.patch_all(allow_experimental_dataset_field=...)
                and compat.preplace_all_checkpoints() before constructing
                the real RadGraph()/F1CheXbert() -- guaranteeing no
                radgraph import can happen before the pre-import shims
                are applied.
            chexbert_factory: see radgraph_factory.
            allow_experimental_dataset_field: forwarded to
                compat.patch_all(); ignored when factories are injected.
        """
        if (radgraph_factory is None) != (chexbert_factory is None):
            raise ValueError(
                "radgraph_factory and chexbert_factory must both be "
                "provided or both omitted -- partial injection is a "
                "test-only escape hatch, not general DI."
            )

        self._allow_experimental_dataset_field = allow_experimental_dataset_field

        if radgraph_factory is None:
            compat.patch_all(allow_experimental_dataset_field=allow_experimental_dataset_field)
            self._checkpoint_results: Optional[Dict[str, compat.CheckpointPlacementResult]] = (
                compat.preplace_all_checkpoints()
            )
            radgraph_factory = _default_radgraph_factory
            chexbert_factory = _default_chexbert_factory
        else:
            self._checkpoint_results = None

        self._radgraph = radgraph_factory()
        self._chexbert = chexbert_factory()

    def annotate_record(self, record: ReportRecord) -> dict:
        """Returns the success schema (see REQUIRED_SUCCESS_FIELDS).

        Raises:
            AnnotationError: empty/whitespace-only finding text, or a
                malformed CheXbert output vector (wrong length or
                out-of-domain values).
            Exception: whatever radgraph/f1chexbert themselves raise on
                a real inference failure -- not wrapped or reclassified.
        """
        finding = record.finding
        if finding is None or not finding.strip():
            raise AnnotationError(
                f"Empty or whitespace-only finding text for "
                f"(dataset={record.dataset!r}, patient_id={record.patient_id!r}, "
                f"study_id={record.study_id!r})"
            )

        annotations = self._radgraph([finding])
        entities = annotations["0"]["entities"]

        chexbert_vector = self._chexbert.get_label(finding)
        if not isinstance(chexbert_vector, (list, tuple)):
            raise AnnotationError(
                f"CheXbert output is not list-like: {type(chexbert_vector).__name__}"
            )
        if len(chexbert_vector) != len(CHEXBERT_14_CLASSES):
            raise AnnotationError(
                f"CheXbert output has {len(chexbert_vector)} elements, "
                f"expected {len(CHEXBERT_14_CLASSES)}"
            )
        if any(value not in CHEXBERT_LABEL_DOMAIN for value in chexbert_vector):
            raise AnnotationError(
                f"CheXbert output contains values outside "
                f"{sorted(CHEXBERT_LABEL_DOMAIN)}: {list(chexbert_vector)}"
            )

        chexbert_labels_14 = list(chexbert_vector)
        chexbert_labels_5 = [chexbert_labels_14[i] for i in CHEXBERT_5_INDICES]

        return {
            "dataset": record.dataset,
            "patient_id": record.patient_id,
            "study_id": record.study_id,
            "finding": finding,
            "entities": entities,
            "chexbert_labels_14": chexbert_labels_14,
            "chexbert_labels_5": chexbert_labels_5,
        }

    def _serialize_checkpoints(self) -> Optional[dict]:
        if self._checkpoint_results is None:
            return None
        return {
            filename: {
                "repo_id": compat.CHECKPOINT_REPO_ID,
                "destination_path": result.destination_path,
                "resolved_revision": result.resolved_revision,
                "revision_source": result.revision_source,
            }
            for filename, result in self._checkpoint_results.items()
        }

    def annotate_records(
        self,
        records: Iterable[ReportRecord],
        *,
        output_path: Path,
        errors_path: Optional[Path] = None,
        meta_path: Optional[Path] = None,
        input_path: Optional[str] = None,
        continue_on_error: bool = False,
    ) -> AnnotationRunSummary:
        """Annotates an iterable of ReportRecord, never a manifest object.

        Raises ValueError immediately, before touching output_path or
        processing any record, if continue_on_error=True and
        errors_path is None.

        meta_path defaults to
        output_path.with_suffix(output_path.suffix + ".meta.json") when
        omitted.

        Any existing output_path is validated first (see
        _load_completed_keys) to build the resume set -- malformed or
        duplicate-keyed existing output raises rather than being
        silently ignored, and a record whose key is already validly
        completed is skipped.

        A success record is only written after annotate_record() has
        returned (i.e. after both model outputs and all schema
        validations have completed), serialized in one write call and
        flushed immediately. If continue_on_error is False (default),
        any exception from annotate_record() propagates immediately --
        the sidecar metadata is updated to run_status="failed" first,
        but the record itself is never marked completed and the
        original exception is re-raised unchanged. If True, the
        exception is logged to errors_path (same write/flush pattern)
        and the loop continues; the record is still never marked
        completed, so a later rerun will retry it.
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

        completed_keys = _load_completed_keys(output_path)

        meta = {
            "run_status": "running",
            "run_started_at_utc": _utc_now_iso(),
            "run_finished_at_utc": None,
            "input_path": input_path if input_path is not None else "unspecified (records provided in-memory)",
            "output_path": str(output_path),
            "errors_path": str(errors_path) if errors_path is not None else None,
            "environment": asdict(compat.detect_environment()),
            "allow_experimental_dataset_field": self._allow_experimental_dataset_field,
            "chexbert_14_classes": list(CHEXBERT_14_CLASSES),
            "chexbert_5_indices": list(CHEXBERT_5_INDICES),
            "chexbert_5_classes": list(CHEXBERT_5_CLASSES),
            "checkpoints": self._serialize_checkpoints(),
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
                for record in records:
                    key = (record.dataset, record.patient_id, record.study_id)
                    if key in completed_keys:
                        skipped += 1
                        continue
                    try:
                        payload = self.annotate_record(record)
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
                            "dataset": record.dataset,
                            "patient_id": record.patient_id,
                            "study_id": record.study_id,
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

        return AnnotationRunSummary(processed=processed, skipped_already_done=skipped, failed=failed)
