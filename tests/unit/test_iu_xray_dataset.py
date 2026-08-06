"""Unit tests for src/data/iu_xray/dataset.py.

Covers: (4) excluded-record filtering, (5) deterministic ordering,
(6) exact split counts, (7) no split overlap, (15) repeated-run
determinism. Also exercises get_metadata()'s text-only access and
load_images_for()'s lazy loading against synthetic fixtures.
"""

import json

import pytest

from src.data.iu_xray.dataset import (
    EXPECTED_SPLIT_COUNTS,
    IuXrayDataset,
    load_split_manifest,
)
from src.data.iu_xray.exceptions import IuXraySplitError
from src.data.iu_xray.records import IuXrayCanonicalRecord


def _record(study_id, image_paths=None, **overrides):
    payload = dict(
        dataset_name="iu_xray", study_id=study_id, report_id=study_id,
        image_ids=[f"{study_id}_IM-0001-1001"],
        image_paths=image_paths or [f"/fake/{study_id}_IM-0001-1001.png"],
        image_count=len(image_paths) if image_paths else 1,
        findings="Findings.", impression="Impression.",
        indication=None, comparison=None, full_report="Findings. Impression.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def _split_manifest(train, validation, test):
    return {
        "train_study_ids": list(train),
        "validation_study_ids": list(validation),
        "test_study_ids": list(test),
        "train_count": len(train),
        "validation_count": len(validation),
        "test_count": len(test),
    }


def _dataset(records, split_manifest, storage_root=None):
    records_by_id = {r.study_id: r for r in records}
    return IuXrayDataset(storage_root or __import__("pathlib").Path("/fake"), records_by_id, split_manifest)


def test_load_split_manifest_raises_when_missing(tmp_path):
    with pytest.raises(IuXraySplitError, match="not found"):
        load_split_manifest(tmp_path)


def test_load_split_returns_records_in_manifest_order():
    records = [_record("CXR3"), _record("CXR1"), _record("CXR2")]
    split_manifest = _split_manifest(["CXR1", "CXR2", "CXR3"], [], [])
    dataset = _dataset(records, split_manifest)

    loaded = dataset.load_split("train")
    assert [r.study_id for r in loaded] == ["CXR1", "CXR2", "CXR3"]


def test_load_split_rejects_unknown_split_name():
    dataset = _dataset([], _split_manifest([], [], []))
    with pytest.raises(ValueError, match="Unknown split_name"):
        dataset.load_split("bogus")


def test_load_split_excludes_excluded_records_by_default():
    usable = _record("CXR1", validation_status="valid")
    excluded = _record("CXR2", validation_status="excluded", exclusion_reason="missing_image")
    split_manifest = _split_manifest(["CXR1", "CXR2"], [], [])
    dataset = _dataset([usable, excluded], split_manifest)

    with pytest.raises(IuXraySplitError, match="excluded record"):
        dataset.load_split("train")  # excluded record present -> real invariant violation


def test_load_split_include_excluded_true_allows_excluded_records():
    usable = _record("CXR1", validation_status="valid")
    excluded = _record("CXR2", validation_status="excluded", exclusion_reason="missing_image")
    split_manifest = _split_manifest(["CXR1", "CXR2"], [], [])
    dataset = _dataset([usable, excluded], split_manifest)

    loaded = dataset.load_split("train", include_excluded=True)
    assert {r.study_id for r in loaded} == {"CXR1", "CXR2"}


def test_load_split_raises_on_study_id_with_no_matching_record():
    split_manifest = _split_manifest(["CXR_MISSING"], [], [])
    dataset = _dataset([], split_manifest)
    with pytest.raises(IuXraySplitError, match="no matching canonical record"):
        dataset.load_split("train")


def test_dataset_construction_raises_on_split_overlap():
    records = [_record("CXR1")]
    split_manifest = _split_manifest(["CXR1"], ["CXR1"], [])
    with pytest.raises(IuXraySplitError, match="overlapping"):
        _dataset(records, split_manifest)


def test_validate_split_counts_matches_expected():
    train_ids = [f"CXR{i}" for i in range(EXPECTED_SPLIT_COUNTS["train"])]
    val_ids = [f"CXRV{i}" for i in range(EXPECTED_SPLIT_COUNTS["validation"])]
    test_ids = [f"CXRT{i}" for i in range(EXPECTED_SPLIT_COUNTS["test"])]
    all_records = (
        [_record(sid) for sid in train_ids]
        + [_record(sid) for sid in val_ids]
        + [_record(sid) for sid in test_ids]
    )
    split_manifest = _split_manifest(train_ids, val_ids, test_ids)
    dataset = _dataset(all_records, split_manifest)

    counts = dataset.validate_split_counts()
    assert counts == EXPECTED_SPLIT_COUNTS


def test_validate_split_counts_raises_on_mismatch():
    split_manifest = _split_manifest(["CXR1"], [], [])
    dataset = _dataset([_record("CXR1")], split_manifest)
    with pytest.raises(IuXraySplitError, match="mismatch"):
        dataset.validate_split_counts()


def test_repeated_load_split_calls_produce_identical_ids_and_order():
    records = [_record("CXR3"), _record("CXR1"), _record("CXR2")]
    split_manifest = _split_manifest(["CXR1", "CXR2", "CXR3"], [], [])
    dataset = _dataset(records, split_manifest)

    first = [r.study_id for r in dataset.load_split("train")]
    second = [r.study_id for r in dataset.load_split("train")]
    assert first == second


def test_get_metadata_never_touches_image_files(tmp_path):
    # image_paths point at files that do not exist on disk at all --
    # get_metadata must not raise, proving it never opens them.
    record = _record("CXR1", image_paths=[str(tmp_path / "does_not_exist.png")])
    dataset = _dataset([record], _split_manifest(["CXR1"], [], []))
    metadata = dataset.get_metadata("CXR1")
    assert metadata.study_id == "CXR1"


def test_get_metadata_raises_key_error_for_unknown_study_id():
    dataset = _dataset([], _split_manifest([], [], []))
    with pytest.raises(KeyError):
        dataset.get_metadata("CXR_NOPE")


def test_load_images_for_multi_image_study_preserves_grouping(tmp_path):
    from PIL import Image
    p1, p2 = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("L", (4, 4)).save(p1)
    Image.new("L", (4, 4)).save(p2)
    record = _record("CXR1", image_paths=[str(p1), str(p2)])
    dataset = _dataset([record], _split_manifest(["CXR1"], [], []))

    images = dataset.load_images_for("CXR1")
    assert len(images) == 2  # one dataset item, both images grouped -- not exploded
