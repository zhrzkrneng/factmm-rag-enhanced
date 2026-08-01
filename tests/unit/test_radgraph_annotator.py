"""Unit tests for src/baseline/radgraph/annotator.py.

These tests exercise only RadGraphAnnotator's own logic (schema
validation, resume/skip, error handling, sidecar metadata lifecycle),
using lightweight fake stand-ins for the real radgraph/f1chexbert
objects, injected via the constructor's optional factory parameters.
They deliberately do NOT install or import the real heavy ML packages,
and do NOT run any part of the actual RadGraph/CheXbert model -- that
verification belongs to the real Colab smoke test, not this suite.
"""

import json

import pytest

from src.baseline.radgraph.annotator import (
    CHEXBERT_5_CLASSES,
    CHEXBERT_5_INDICES,
    CHEXBERT_14_CLASSES,
    REQUIRED_SUCCESS_FIELDS,
    RadGraphAnnotator,
)
from src.common.exceptions import AnnotationError
from src.data.schema import ReportRecord


class _FakeRadGraph:
    def __init__(self, entities=None):
        self.entities = entities if entities is not None else {
            "1": {"tokens": "acute", "label": "OBS-DA", "relations": []}
        }
        self.calls = []

    def __call__(self, report_list):
        self.calls.append(report_list)
        return {"0": {"text": report_list[0], "entities": self.entities}}


class _FakeCheXbert:
    def __init__(self, vector=None):
        self.vector = vector if vector is not None else [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.calls = []

    def get_label(self, text):
        self.calls.append(text)
        return self.vector


def _make_annotator(entities=None, vector=None):
    return RadGraphAnnotator(
        radgraph_factory=lambda: _FakeRadGraph(entities),
        chexbert_factory=lambda: _FakeCheXbert(vector),
    )


def _make_record(patient_id="p1", study_id="s1", finding="Mild cardiomegaly.", dataset="mimic-cxr"):
    return ReportRecord(
        image_paths=[f"{patient_id}/{study_id}/frontal.jpg"],
        finding=finding,
        impression="",
        patient_id=patient_id,
        study_id=study_id,
        dataset=dataset,
    )


# --- Constructor: factory injection contract ------------------------------


def test_constructor_rejects_partial_factory_injection():
    with pytest.raises(ValueError):
        RadGraphAnnotator(radgraph_factory=lambda: _FakeRadGraph())


# --- annotate_record: schema and validation --------------------------------


def test_annotate_record_returns_exact_schema_with_fake_models():
    fake_entities = {"1": {"tokens": "acute", "label": "OBS-DA", "relations": []}}
    fake_vector = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    annotator = _make_annotator(entities=fake_entities, vector=fake_vector)
    record = _make_record(finding="Mild cardiomegaly.")

    result = annotator.annotate_record(record)

    assert set(result.keys()) == set(REQUIRED_SUCCESS_FIELDS)
    assert result["dataset"] == record.dataset
    assert result["patient_id"] == record.patient_id
    assert result["study_id"] == record.study_id
    assert result["finding"] == "Mild cardiomegaly."
    assert result["entities"] == fake_entities
    assert result["chexbert_labels_14"] == fake_vector
    assert result["chexbert_labels_5"] == [fake_vector[i] for i in CHEXBERT_5_INDICES]


def test_annotate_record_rejects_empty_finding():
    annotator = _make_annotator()
    with pytest.raises(AnnotationError):
        annotator.annotate_record(_make_record(finding=""))


def test_annotate_record_rejects_whitespace_only_finding():
    annotator = _make_annotator()
    with pytest.raises(AnnotationError):
        annotator.annotate_record(_make_record(finding="   "))


def test_chexbert_vector_rejects_wrong_length():
    annotator = _make_annotator(vector=[0] * 13)
    with pytest.raises(AnnotationError):
        annotator.annotate_record(_make_record())


def test_chexbert_vector_rejects_out_of_domain_values():
    vector = [0] * 14
    vector[0] = 2
    annotator = _make_annotator(vector=vector)
    with pytest.raises(AnnotationError):
        annotator.annotate_record(_make_record())


def test_chexbert_5_subset_matches_named_indices():
    assert CHEXBERT_5_INDICES == (1, 4, 5, 7, 9)
    assert CHEXBERT_5_CLASSES == (
        "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion",
    )
    for idx, name in zip(CHEXBERT_5_INDICES, CHEXBERT_5_CLASSES):
        assert CHEXBERT_14_CLASSES[idx] == name


def test_chexbert_5_subset_extracted_correctly_from_valid_vector():
    vector = [0] * 14
    for idx in CHEXBERT_5_INDICES:
        vector[idx] = 1
    annotator = _make_annotator(vector=vector)
    result = annotator.annotate_record(_make_record())
    assert result["chexbert_labels_5"] == [1, 1, 1, 1, 1]


def test_annotate_records_method_name_used_consistently():
    assert hasattr(RadGraphAnnotator, "annotate_records")
    assert not hasattr(RadGraphAnnotator, "annotate_manifest")


# --- annotate_records: errors_path / meta_path defaults --------------------


def test_annotate_records_continue_on_error_without_errors_path_raises_before_processing(tmp_path):
    output_path = tmp_path / "out.jsonl"
    annotator = _make_annotator()
    with pytest.raises(ValueError):
        annotator.annotate_records([_make_record()], output_path=output_path, continue_on_error=True)
    assert not output_path.exists()


def test_annotate_records_default_meta_path_derived_deterministically(tmp_path):
    output_path = tmp_path / "annotations.jsonl"
    annotator = _make_annotator()
    annotator.annotate_records([_make_record()], output_path=output_path)
    expected_meta_path = tmp_path / "annotations.jsonl.meta.json"
    assert expected_meta_path.is_file()


# --- annotate_records: sidecar metadata lifecycle --------------------------


def test_annotate_records_writes_meta_running_before_iteration_begins(tmp_path):
    output_path = tmp_path / "out.jsonl"
    meta_path = tmp_path / "custom.meta.json"
    observed = {}

    class ObservingRadGraph(_FakeRadGraph):
        def __call__(self, report_list):
            if not observed:
                meta = json.loads(meta_path.read_text())
                observed["run_status"] = meta["run_status"]
                observed["summary"] = meta["summary"]
                observed["run_finished_at_utc"] = meta["run_finished_at_utc"]
            return super().__call__(report_list)

    annotator = RadGraphAnnotator(
        radgraph_factory=lambda: ObservingRadGraph(),
        chexbert_factory=lambda: _FakeCheXbert(),
    )
    annotator.annotate_records([_make_record()], output_path=output_path, meta_path=meta_path)

    assert observed["run_status"] == "running"
    assert observed["summary"] == {"processed": 0, "skipped_already_done": 0, "failed": 0}
    assert observed["run_finished_at_utc"] is None


def test_annotate_records_meta_becomes_completed_after_success(tmp_path):
    output_path = tmp_path / "out.jsonl"
    annotator = _make_annotator()
    records = [
        _make_record(patient_id="p1", study_id="s1"),
        _make_record(patient_id="p2", study_id="s2"),
    ]
    summary = annotator.annotate_records(records, output_path=output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["run_status"] == "completed"
    assert meta["run_finished_at_utc"] is not None
    assert meta["summary"] == {"processed": 2, "skipped_already_done": 0, "failed": 0}
    assert summary.processed == 2


def test_annotate_records_meta_becomes_failed_and_reraises_original_exception_in_fail_fast_mode(tmp_path):
    output_path = tmp_path / "out.jsonl"
    annotator = _make_annotator()

    with pytest.raises(AnnotationError):
        annotator.annotate_records([_make_record(finding="")], output_path=output_path)

    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = json.loads(meta_path.read_text())
    assert meta["run_status"] == "failed"
    assert meta["run_finished_at_utc"] is not None
    assert meta["last_error"]["error_type"] == "AnnotationError"


# --- annotate_records: resume / existing-output validation -----------------


def test_annotate_records_resume_skips_already_completed_keys(tmp_path):
    output_path = tmp_path / "out.jsonl"
    existing_payload = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "finding": "Prior finding.", "entities": {},
        "chexbert_labels_14": [0] * 14, "chexbert_labels_5": [0] * 5,
    }
    output_path.write_text(json.dumps(existing_payload) + "\n")

    annotator = _make_annotator()
    records = [
        _make_record(patient_id="p1", study_id="s1"),
        _make_record(patient_id="p2", study_id="s2"),
    ]
    summary = annotator.annotate_records(records, output_path=output_path)

    assert summary.processed == 1
    assert summary.skipped_already_done == 1


def test_annotate_records_rejects_malformed_existing_output(tmp_path):
    output_path = tmp_path / "out.jsonl"
    output_path.write_text("{not valid json\n")
    annotator = _make_annotator()
    with pytest.raises(AnnotationError):
        annotator.annotate_records([_make_record()], output_path=output_path)


def test_annotate_records_rejects_duplicate_completed_keys_in_existing_output(tmp_path):
    output_path = tmp_path / "out.jsonl"
    payload = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "finding": "x", "entities": {},
        "chexbert_labels_14": [0] * 14, "chexbert_labels_5": [0] * 5,
    }
    line = json.dumps(payload) + "\n"
    output_path.write_text(line + line)

    annotator = _make_annotator()
    with pytest.raises(AnnotationError):
        annotator.annotate_records([_make_record()], output_path=output_path)


def test_annotate_records_blank_lines_in_existing_output_are_ignored(tmp_path):
    output_path = tmp_path / "out.jsonl"
    payload = {
        "dataset": "mimic-cxr", "patient_id": "p1", "study_id": "s1",
        "finding": "x", "entities": {},
        "chexbert_labels_14": [0] * 14, "chexbert_labels_5": [0] * 5,
    }
    output_path.write_text(json.dumps(payload) + "\n\n")

    annotator = _make_annotator()
    records = [_make_record(patient_id="p1", study_id="s1")]
    summary = annotator.annotate_records(records, output_path=output_path)
    assert summary.skipped_already_done == 1
    assert summary.processed == 0


# --- annotate_records: continue_on_error behavior --------------------------


def test_annotate_records_continue_on_error_logs_and_continues_with_correct_counts(tmp_path):
    output_path = tmp_path / "out.jsonl"
    errors_path = tmp_path / "out.errors.jsonl"
    annotator = _make_annotator()
    records = [
        _make_record(patient_id="p1", study_id="s1", finding="Valid finding one."),
        _make_record(patient_id="p2", study_id="s2", finding=""),
        _make_record(patient_id="p3", study_id="s3", finding="Valid finding two."),
    ]

    summary = annotator.annotate_records(
        records, output_path=output_path, errors_path=errors_path, continue_on_error=True
    )

    assert summary.processed == 2
    assert summary.failed == 1
    assert summary.skipped_already_done == 0

    error_lines = errors_path.read_text().strip().splitlines()
    assert len(error_lines) == 1
    error_obj = json.loads(error_lines[0])
    assert error_obj["patient_id"] == "p2"
    assert error_obj["study_id"] == "s2"
    assert error_obj["error_type"] == "AnnotationError"


def test_annotate_records_never_marks_a_failed_record_as_completed_on_resume(tmp_path):
    output_path = tmp_path / "out.jsonl"
    errors_path = tmp_path / "out.errors.jsonl"
    annotator = _make_annotator()
    failing_record = _make_record(patient_id="p2", study_id="s2", finding="")

    summary1 = annotator.annotate_records(
        [failing_record], output_path=output_path, errors_path=errors_path, continue_on_error=True
    )
    assert summary1.failed == 1
    assert summary1.processed == 0

    summary2 = annotator.annotate_records(
        [failing_record], output_path=output_path, errors_path=errors_path, continue_on_error=True
    )
    assert summary2.failed == 1
    assert summary2.skipped_already_done == 0
