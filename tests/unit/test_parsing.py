"""Unit tests for src/data/parsing.py's JsonReportParser.

Uses only synthetic fixture files under tests/fixtures/ — never real
patient data, per project rules.
"""

import json
import os

import pytest

from src.data.parsing import JsonReportParser

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def test_parse_mimic_fixture_extracts_records_and_ids():
    parser = JsonReportParser(dataset="mimic-cxr")
    records = parser.parse_file(fixture_path("sample_mimic.json"))

    assert len(records) == 2

    first = records[0]
    assert first.patient_id == "p10000032"
    assert first.study_id == "s50414267"
    assert first.dataset == "mimic-cxr"
    assert first.finding == "Synthetic finding text for record one."
    assert first.frontal_image_path == (
        "files/p10/p10000032/s50414267/view1_frontal.jpg"
    )

    second = records[1]
    assert second.patient_id == "p11000045"
    assert second.study_id == "s55123456"


def test_parse_chexpert_fixture_extracts_records_and_ids():
    parser = JsonReportParser(dataset="chexpert")
    records = parser.parse_file(fixture_path("sample_chexpert.json"))

    assert len(records) == 1
    record = records[0]
    assert record.patient_id == "patient64541"
    assert record.study_id == "study1"
    assert record.dataset == "chexpert"


def test_unknown_dataset_raises_on_construction():
    with pytest.raises(ValueError, match="Unknown dataset"):
        JsonReportParser(dataset="not-a-real-dataset")


def test_malformed_path_raises_clear_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps([
        {
            "image": ["no_patient_study_structure.jpg"],
            "finding": "f",
            "impression": "i",
        }
    ]))
    parser = JsonReportParser(dataset="mimic-cxr")
    with pytest.raises(ValueError, match="Could not extract patient/study IDs"):
        parser.parse_file(str(bad_file))


def test_wrong_dataset_convention_raises_error(tmp_path):
    # A CheXpert-style path parsed with the mimic-cxr parser should fail
    # clearly rather than silently extracting nonsense IDs.
    mismatched_file = tmp_path / "mismatched.json"
    mismatched_file.write_text(json.dumps([
        {
            "image": ["CheXpert-v1.0/valid/patient64541/study1/view1.jpg"],
            "finding": "f",
            "impression": "i",
        }
    ]))
    parser = JsonReportParser(dataset="mimic-cxr")
    with pytest.raises(ValueError, match="Could not extract patient/study IDs"):
        parser.parse_file(str(mismatched_file))
