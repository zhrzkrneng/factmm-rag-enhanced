"""Unit tests for src/data/iu_xray/records.py.

Covers: (8) multi-image grouping, (9) missing optional text fields.
"""

import dataclasses

import pytest

from src.data.iu_xray.records import CANONICAL_SCHEMA_FIELDS, IuXrayCanonicalRecord


def _payload(**overrides):
    defaults = dict(
        dataset_name="iu_xray",
        study_id="CXR1",
        report_id="CXR1",
        image_ids=["CXR1_IM-0001-3001", "CXR1_IM-0001-4001"],
        image_paths=["/data/iu_xray/extracted/CXR1_IM-0001-3001.png",
                      "/data/iu_xray/extracted/CXR1_IM-0001-4001.png"],
        image_count=2,
        findings="The heart size is normal.",
        impression="No acute disease.",
        indication="Cough.",
        comparison="None.",
        full_report="None. Cough. The heart size is normal. No acute disease.",
        mesh_terms=["Cardiomegaly"],
        labels=[],
        source_metadata={"source_file": "1.xml"},
        source_split=None,
        validation_status="valid",
        exclusion_reason=None,
    )
    defaults.update(overrides)
    return defaults


def test_schema_fields_constant_matches_dataclass_field_order():
    field_names = tuple(f.name for f in dataclasses.fields(IuXrayCanonicalRecord))
    assert field_names == CANONICAL_SCHEMA_FIELDS


def test_from_dict_round_trips_all_fields():
    record = IuXrayCanonicalRecord.from_dict(_payload())
    assert record.study_id == "CXR1"
    assert record.image_ids == ["CXR1_IM-0001-3001", "CXR1_IM-0001-4001"]
    assert record.mesh_terms == ["Cardiomegaly"]


def test_multi_image_study_preserves_all_images_as_one_record():
    record = IuXrayCanonicalRecord.from_dict(_payload())
    assert record.image_count == 2
    assert len(record.image_paths) == 2
    assert record.is_multi_image is True


def test_single_image_study_is_not_multi_image():
    record = IuXrayCanonicalRecord.from_dict(_payload(
        image_ids=["CXR2_IM-0002-1001"],
        image_paths=["/data/iu_xray/extracted/CXR2_IM-0002-1001.png"],
        image_count=1,
    ))
    assert record.is_multi_image is False


def test_missing_optional_text_fields_preserved_as_none_not_invented():
    record = IuXrayCanonicalRecord.from_dict(_payload(
        indication=None, comparison=None, impression=None,
    ))
    assert record.indication is None
    assert record.comparison is None
    assert record.impression is None
    # findings is still present -- nothing was invented to fill the gaps.
    assert record.findings == "The heart size is normal."


def test_record_is_immutable():
    record = IuXrayCanonicalRecord.from_dict(_payload())
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.findings = "mutated"


def test_is_usable_reflects_validation_status():
    valid_record = IuXrayCanonicalRecord.from_dict(_payload(validation_status="valid"))
    excluded_record = IuXrayCanonicalRecord.from_dict(
        _payload(validation_status="excluded", exclusion_reason="missing_image")
    )
    assert valid_record.is_usable is True
    assert excluded_record.is_usable is False


def test_from_dict_raises_on_missing_required_field():
    payload = _payload()
    del payload["findings"]
    with pytest.raises(KeyError):
        IuXrayCanonicalRecord.from_dict(payload)
