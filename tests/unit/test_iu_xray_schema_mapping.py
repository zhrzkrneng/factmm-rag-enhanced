"""Unit tests for src/data/iu_xray/schema_mapping.py.

Covers: (12) baseline schema mapping, (13) unsupported baseline field
handling.
"""

from pathlib import Path

import pytest

from src.data.iu_xray.exceptions import IuXraySchemaMappingError
from src.data.iu_xray.paths import DATASET_ROOT_TOKEN, resolve_path
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.schema_mapping import (
    IU_XRAY_DATASET_TAG,
    MAPPING_TABLE,
    map_records,
    map_to_report_record,
    to_mapping_table_json,
)


def _record(**overrides):
    payload = dict(
        dataset_name="iu_xray", study_id="CXR1", report_id="CXR1",
        image_ids=["CXR1_IM-0001-3001"],
        image_paths=[f"{DATASET_ROOT_TOKEN}/extracted/CXR1_IM-0001-3001.png"],
        image_count=1, findings="Findings text.", impression="Impression text.",
        indication="Indication text.", comparison="Comparison text.",
        full_report="Comparison text. Indication text. Findings text. Impression text.",
        mesh_terms=["Cardiomegaly"], labels=[], source_metadata={"source_file": "1.xml"},
        source_split=None, validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def test_mapping_table_json_matches_source_table():
    assert to_mapping_table_json() == [dict(row) for row in MAPPING_TABLE]
    # every row has the four required columns
    for row in to_mapping_table_json():
        assert set(row) == {"iu_xray_field", "baseline_field", "transformation", "loss_or_caveat"}


def test_map_to_report_record_direct_fields():
    storage_root = Path("/resolved/storage/root")
    record = _record()
    mapped = map_to_report_record(record, storage_root, resolve_path)

    assert mapped.finding == "Findings text."
    assert mapped.impression == "Impression text."
    assert mapped.study_id == "CXR1"
    assert mapped.dataset == IU_XRAY_DATASET_TAG


def test_map_to_report_record_uses_study_id_as_patient_id():
    storage_root = Path("/resolved/storage/root")
    record = _record(study_id="CXR42")
    mapped = map_to_report_record(record, storage_root, resolve_path)
    assert mapped.patient_id == "CXR42" == mapped.study_id


def test_map_to_report_record_resolves_token_image_paths():
    storage_root = Path("/resolved/storage/root")
    record = _record(image_paths=[f"{DATASET_ROOT_TOKEN}/extracted/x.png"])
    mapped = map_to_report_record(record, storage_root, resolve_path)
    assert mapped.image_paths == ["/resolved/storage/root/extracted/x.png"]


def test_map_to_report_record_drops_iu_xray_only_sections():
    # indication/comparison/mesh_terms/labels are real values here, and
    # the mapper must not error or silently coerce them -- they're just
    # not part of ReportRecord (see MAPPING_TABLE).
    storage_root = Path("/resolved/storage/root")
    record = _record(indication="Cough.", comparison="Prior film.")
    mapped = map_to_report_record(record, storage_root, resolve_path)
    assert not hasattr(mapped, "indication")
    assert not hasattr(mapped, "comparison")


def test_map_to_report_record_raises_when_findings_missing():
    storage_root = Path("/resolved/storage/root")
    record = _record(findings=None)
    with pytest.raises(IuXraySchemaMappingError, match="findings"):
        map_to_report_record(record, storage_root, resolve_path)


def test_map_to_report_record_raises_when_impression_missing():
    storage_root = Path("/resolved/storage/root")
    record = _record(impression=None)
    with pytest.raises(IuXraySchemaMappingError, match="impression"):
        map_to_report_record(record, storage_root, resolve_path)


def test_map_to_report_record_never_fabricates_missing_text():
    # A record missing both findings and impression must still raise,
    # never fall back to full_report or any invented text.
    storage_root = Path("/resolved/storage/root")
    record = _record(findings=None, impression=None,
                      full_report="Comparison text. Indication text.")
    with pytest.raises(IuXraySchemaMappingError):
        map_to_report_record(record, storage_root, resolve_path)


def test_map_records_collects_skips_without_raising():
    storage_root = Path("/resolved/storage/root")
    good = _record(study_id="CXR1")
    incompatible = _record(study_id="CXR2", impression=None)

    mapped, skipped = map_records([good, incompatible], storage_root, resolve_path)

    assert len(mapped) == 1
    assert mapped[0].study_id == "CXR1"
    assert len(skipped) == 1
    assert skipped[0].study_id == "CXR2"
    assert "impression" in skipped[0].reason
