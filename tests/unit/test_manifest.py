"""Unit tests for src/data/manifest.py and src/common/manifest.py."""

import dataclasses
import os

from src.common.manifest import ManifestEntry, SplitManifest
from src.data.manifest import ManifestBuilder
from src.data.schema import ReportRecord

# Deliberately distinctive text so we can assert it never appears in a
# serialized manifest.
SECRET_FINDING = "UNIQUE_FINDING_TEXT_MUST_NOT_LEAK_INTO_MANIFEST"
SECRET_IMPRESSION = "UNIQUE_IMPRESSION_TEXT_MUST_NOT_LEAK_INTO_MANIFEST"


def make_records():
    return [
        ReportRecord(
            image_paths=["p1/s1/frontal.jpg"],
            finding=SECRET_FINDING,
            impression=SECRET_IMPRESSION,
            patient_id="p1",
            study_id="s1",
            dataset="mimic-cxr",
        ),
        ReportRecord(
            image_paths=["p1/s2/frontal.jpg"],
            finding="second finding",
            impression="second impression",
            patient_id="p1",
            study_id="s2",
            dataset="mimic-cxr",
        ),
        ReportRecord(
            image_paths=["p2/s3/frontal.jpg"],
            finding="third finding",
            impression="third impression",
            patient_id="p2",
            study_id="s3",
            dataset="mimic-cxr",
        ),
    ]


def test_manifest_entry_has_no_report_text_fields():
    # Structural guarantee: ManifestEntry can't carry finding/impression
    # even if someone tried to pass them, because the fields don't exist.
    field_names = {f.name for f in dataclasses.fields(ManifestEntry)}
    assert "finding" not in field_names
    assert "impression" not in field_names


def test_build_produces_correct_counts_and_entries():
    builder = ManifestBuilder()
    manifest = builder.build("train", make_records())

    assert manifest.split_name == "train"
    assert manifest.record_count == 3
    assert manifest.patient_count == 2  # p1 has 2 studies, p2 has 1

    assert manifest.entries[0].patient_id == "p1"
    assert manifest.entries[0].study_id == "s1"
    assert manifest.entries[0].image_paths == ["p1/s1/frontal.jpg"]
    assert manifest.entries[0].dataset == "mimic-cxr"


def test_save_and_load_round_trip(tmp_path):
    builder = ManifestBuilder()
    manifest = builder.build("valid", make_records())

    output_path = str(tmp_path / "valid_manifest.json")
    builder.save(manifest, output_path)
    loaded = builder.load(output_path)

    assert loaded == manifest


def test_save_creates_parent_directories(tmp_path):
    builder = ManifestBuilder()
    manifest = builder.build("test", make_records())

    nested_path = str(tmp_path / "does" / "not" / "exist" / "manifest.json")
    assert not os.path.isdir(os.path.dirname(nested_path))

    builder.save(manifest, nested_path)
    assert os.path.isfile(nested_path)


def test_saved_manifest_never_contains_report_text(tmp_path):
    builder = ManifestBuilder()
    manifest = builder.build("train", make_records())

    output_path = str(tmp_path / "train_manifest.json")
    builder.save(manifest, output_path)

    with open(output_path, "r") as f:
        raw_contents = f.read()

    assert SECRET_FINDING not in raw_contents
    assert SECRET_IMPRESSION not in raw_contents
    # Sanity: the identifiers we DO expect to be present, are present.
    assert "p1" in raw_contents
    assert "s1" in raw_contents
