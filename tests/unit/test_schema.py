"""Unit tests for src/data/schema.py's ReportRecord."""

import dataclasses

import pytest

from src.data.schema import ReportRecord


def make_record(**overrides):
    """Build a ReportRecord with sensible defaults, overridable per test."""
    defaults = dict(
        image_paths=["p10/p10000032/s50414267/frontal.jpg",
                      "p10/p10000032/s50414267/lateral.jpg"],
        finding="There is no focal consolidation.",
        impression="No acute cardiopulmonary process.",
        patient_id="p10000032",
        study_id="s50414267",
        dataset="mimic-cxr",
    )
    defaults.update(overrides)
    return ReportRecord(**defaults)


def test_combined_text_concatenates_finding_and_impression():
    record = make_record(
        finding="There is no focal consolidation.",
        impression="No acute cardiopulmonary process.",
    )
    assert record.combined_text == (
        "There is no focal consolidation. No acute cardiopulmonary process."
    )


def test_combined_text_strips_surrounding_whitespace():
    record = make_record(finding="  Findings text.  ", impression="  ")
    # finding + " " + impression, then stripped as a whole.
    assert record.combined_text == "Findings text."


def test_frontal_image_path_returns_first_path():
    record = make_record(
        image_paths=["frontal.jpg", "lateral.jpg", "extra.jpg"]
    )
    assert record.frontal_image_path == "frontal.jpg"


def test_frontal_image_path_single_image():
    record = make_record(image_paths=["only_view.jpg"])
    assert record.frontal_image_path == "only_view.jpg"


def test_frontal_image_path_raises_on_empty_image_list():
    record = make_record(image_paths=[])
    with pytest.raises(ValueError, match="no image_paths"):
        record.frontal_image_path


def test_record_is_immutable():
    record = make_record()
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.finding = "mutated"


def test_finding_and_impression_remain_separately_accessible():
    # Guards against collapsing finding/impression into only the
    # combined form — annotation and evaluation both require `finding`
    # alone (see docs/paper_analysis.md and src/data/schema.py's
    # ReportRecord docstring).
    record = make_record(finding="Finding only.", impression="Impression only.")
    assert record.finding == "Finding only."
    assert record.impression == "Impression only."
    assert record.finding != record.combined_text
