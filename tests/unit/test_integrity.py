"""Unit tests for src/data/integrity.py's IntegrityChecker."""

import pytest

from src.common.exceptions import DataIntegrityError
from src.data.integrity import IntegrityChecker
from src.data.schema import ReportRecord


def make_record(**overrides):
    defaults = dict(
        image_paths=["view1_frontal.jpg"],
        finding="A valid finding.",
        impression="A valid impression.",
        patient_id="p1",
        study_id="s1",
        dataset="mimic-cxr",
    )
    defaults.update(overrides)
    return ReportRecord(**defaults)


def touch(path):
    """Create an empty file at path, including parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_clean_record_set_has_no_issues(tmp_path):
    touch(tmp_path / "view1_frontal.jpg")
    record = make_record()

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record])

    assert issues == []
    checker.check_and_raise([record])  # must not raise


def test_empty_finding_is_flagged(tmp_path):
    touch(tmp_path / "view1_frontal.jpg")
    record = make_record(finding="   ")

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record])

    assert len(issues) == 1
    assert issues[0].kind == "empty_finding"
    assert issues[0].study_id == "s1"


def test_empty_impression_is_flagged(tmp_path):
    touch(tmp_path / "view1_frontal.jpg")
    record = make_record(impression="")

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record])

    assert len(issues) == 1
    assert issues[0].kind == "empty_impression"


def test_no_image_paths_is_flagged(tmp_path):
    record = make_record(image_paths=[])

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record])

    kinds = {issue.kind for issue in issues}
    assert "no_image_paths" in kinds
    # Should not also try (and fail) to check a missing_image_file,
    # since there's no image path to check in the first place.
    assert "missing_image_file" not in kinds


def test_missing_image_file_is_flagged(tmp_path):
    # Note: file is never created at tmp_path.
    record = make_record(image_paths=["does_not_exist.jpg"])

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record])

    assert len(issues) == 1
    assert issues[0].kind == "missing_image_file"


def test_duplicate_study_is_flagged(tmp_path):
    touch(tmp_path / "view1_frontal.jpg")
    record_a = make_record(patient_id="p1", study_id="s1")
    record_b = make_record(patient_id="p1", study_id="s1")

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record_a, record_b])

    duplicate_issues = [i for i in issues if i.kind == "duplicate_study"]
    assert len(duplicate_issues) == 1


def test_same_study_id_different_patient_is_not_a_duplicate(tmp_path):
    # Guards against checking study_id alone — CheXpert reuses
    # study_id values like "study1" across different patients.
    touch(tmp_path / "view1_frontal.jpg")
    record_a = make_record(patient_id="patient1", study_id="study1")
    record_b = make_record(patient_id="patient2", study_id="study1")

    checker = IntegrityChecker(image_root=str(tmp_path))
    issues = checker.check([record_a, record_b])

    assert not any(i.kind == "duplicate_study" for i in issues)


def test_check_and_raise_raises_with_summary_counts(tmp_path):
    bad_record = make_record(finding="", impression="")

    checker = IntegrityChecker(image_root=str(tmp_path))
    with pytest.raises(DataIntegrityError) as exc_info:
        checker.check_and_raise([bad_record])

    message = str(exc_info.value)
    assert "empty_finding=1" in message
    assert "empty_impression=1" in message
    assert "missing_image_file=1" in message


def test_check_never_raises_only_reports(tmp_path):
    bad_record = make_record(finding="", image_paths=[])
    checker = IntegrityChecker(image_root=str(tmp_path))
    # Should return issues, not raise, even though the record is bad.
    issues = checker.check([bad_record])
    assert len(issues) >= 2
