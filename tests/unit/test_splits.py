"""Unit tests for src/data/splits.py's PatientSplitValidator."""

import pytest

from src.common.exceptions import PatientLeakageError
from src.data.splits import PatientSplitValidator
from src.data.schema import ReportRecord


def make_record(patient_id, study_id):
    return ReportRecord(
        image_paths=["view1_frontal.jpg"],
        finding="A finding.",
        impression="An impression.",
        patient_id=patient_id,
        study_id=study_id,
        dataset="mimic-cxr",
    )


def test_disjoint_patients_across_splits_has_no_issues():
    split_records = {
        "train": [make_record("p1", "s1"), make_record("p2", "s2")],
        "valid": [make_record("p3", "s3")],
        "test": [make_record("p4", "s4")],
    }
    validator = PatientSplitValidator()
    issues = validator.check(split_records)
    assert issues == []
    validator.check_and_raise(split_records)  # must not raise


def test_multiple_studies_same_patient_same_split_is_fine():
    # A patient with several studies, all in the same split, is normal
    # and must not be flagged as leakage.
    split_records = {
        "train": [make_record("p1", "s1"), make_record("p1", "s2")],
        "valid": [make_record("p2", "s3")],
    }
    validator = PatientSplitValidator()
    issues = validator.check(split_records)
    assert issues == []


def test_patient_in_two_splits_is_flagged():
    split_records = {
        "train": [make_record("p1", "s1")],
        "valid": [make_record("p1", "s2")],
    }
    validator = PatientSplitValidator()
    issues = validator.check(split_records)

    assert len(issues) == 1
    assert issues[0].patient_id == "p1"
    assert issues[0].splits == ("train", "valid")


def test_patient_in_three_splits_is_flagged_with_all_three():
    split_records = {
        "train": [make_record("p1", "s1")],
        "valid": [make_record("p1", "s2")],
        "test": [make_record("p1", "s3")],
    }
    validator = PatientSplitValidator()
    issues = validator.check(split_records)

    assert len(issues) == 1
    assert issues[0].splits == ("test", "train", "valid")


def test_check_and_raise_raises_with_patient_id_in_message():
    split_records = {
        "train": [make_record("p1", "s1")],
        "test": [make_record("p1", "s2")],
    }
    validator = PatientSplitValidator()
    with pytest.raises(PatientLeakageError, match="p1"):
        validator.check_and_raise(split_records)


def test_check_and_raise_truncates_preview_for_many_leaks():
    # 7 leaking patients -> preview should show 5 plus a "+2 more" tail.
    split_records = {
        "train": [make_record(f"p{i}", f"s{i}") for i in range(7)],
        "test": [make_record(f"p{i}", f"s{i}_dup") for i in range(7)],
    }
    validator = PatientSplitValidator()
    with pytest.raises(PatientLeakageError, match=r"\+2 more"):
        validator.check_and_raise(split_records)


def test_check_never_raises_only_reports():
    split_records = {
        "train": [make_record("p1", "s1")],
        "test": [make_record("p1", "s2")],
    }
    validator = PatientSplitValidator()
    issues = validator.check(split_records)  # must not raise
    assert len(issues) == 1
