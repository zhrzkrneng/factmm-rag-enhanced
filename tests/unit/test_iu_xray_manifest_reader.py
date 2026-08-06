"""Unit tests for src/data/iu_xray/manifest_reader.py.

Covers: (2) hash mismatch failure, (3) canonical manifest loading.
Uses synthetic fixtures only -- no real dataset required.
"""

import hashlib
import json

import pytest

from src.data.iu_xray.exceptions import IuXrayManifestError
from src.data.iu_xray.manifest_reader import (
    load_all_records,
    load_canonical_index,
    resolve_external_manifest_path,
    sha256_of_file,
    verify_external_manifest,
)
from src.data.iu_xray.paths import DATASET_ROOT_TOKEN
from src.data.iu_xray.records import CANONICAL_SCHEMA_FIELDS


def _full_manifest_payload(records):
    return {"schema_fields": list(CANONICAL_SCHEMA_FIELDS), "records": records}


def _record_dict(study_id, **overrides):
    payload = dict(
        dataset_name="iu_xray", study_id=study_id, report_id=study_id,
        image_ids=[f"{study_id}_IM-0001-1001"],
        image_paths=[f"/fake/extracted/{study_id}_IM-0001-1001.png"],
        image_count=1, findings="Findings.", impression="Impression.",
        indication=None, comparison=None, full_report="Findings. Impression.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return payload


def _write_external_manifest(storage_root, records):
    canonical_dir = storage_root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    full_path = canonical_dir / "iu_xray_canonical_full.json"
    full_path.write_text(json.dumps(_full_manifest_payload(records)), encoding="utf-8")
    return full_path, sha256_of_file(full_path)


def _write_index(output_dir, full_manifest_token_path, sha256, record_count):
    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": "1.0",
        "schema_fields": list(CANONICAL_SCHEMA_FIELDS),
        "record_count": record_count,
        "usable_record_count": record_count,
        "excluded_record_count": 0,
        "total_image_count": record_count,
        "full_manifest_external_path": full_manifest_token_path,
        "full_manifest_sha256": sha256,
        "full_manifest_bytes": 123,
        "split_manifest_reference": "iu_xray_split_manifest.json",
    }
    (output_dir / "iu_xray_canonical_manifest.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return index


def test_load_canonical_index_raises_when_missing(tmp_path):
    with pytest.raises(IuXrayManifestError, match="not found"):
        load_canonical_index(tmp_path / "results" / "reproduction" / "milestone_2_8")


def test_resolve_external_manifest_path_substitutes_token(tmp_path):
    storage_root = tmp_path / "storage"
    index = {"full_manifest_external_path": f"{DATASET_ROOT_TOKEN}/canonical/iu_xray_canonical_full.json"}
    resolved = resolve_external_manifest_path(index, storage_root)
    assert resolved == storage_root / "canonical" / "iu_xray_canonical_full.json"


def test_verify_external_manifest_succeeds_on_matching_hash(tmp_path):
    storage_root = tmp_path / "storage"
    output_dir = tmp_path / "results"
    records = [_record_dict("CXR1")]
    full_path, sha = _write_external_manifest(storage_root, records)
    index = _write_index(output_dir, f"{DATASET_ROOT_TOKEN}/canonical/iu_xray_canonical_full.json", sha, 1)

    verified_path = verify_external_manifest(index, storage_root)
    assert verified_path == full_path


def test_verify_external_manifest_raises_on_missing_file(tmp_path):
    storage_root = tmp_path / "storage"
    index = {"full_manifest_external_path": f"{DATASET_ROOT_TOKEN}/canonical/iu_xray_canonical_full.json",
             "full_manifest_sha256": "deadbeef"}
    with pytest.raises(IuXrayManifestError, match="not found"):
        verify_external_manifest(index, storage_root)


def test_verify_external_manifest_raises_on_hash_mismatch(tmp_path):
    storage_root = tmp_path / "storage"
    output_dir = tmp_path / "results"
    records = [_record_dict("CXR1")]
    full_path, real_sha = _write_external_manifest(storage_root, records)
    wrong_sha = hashlib.sha256(b"not the real content").hexdigest()
    index = _write_index(output_dir, f"{DATASET_ROOT_TOKEN}/canonical/iu_xray_canonical_full.json", wrong_sha, 1)

    with pytest.raises(IuXrayManifestError, match="SHA-256 mismatch"):
        verify_external_manifest(index, storage_root)


def test_load_all_records_builds_study_id_keyed_dict(tmp_path):
    storage_root = tmp_path / "storage"
    records = [_record_dict("CXR1"), _record_dict("CXR2")]
    full_path, _ = _write_external_manifest(storage_root, records)

    loaded = load_all_records(full_path)
    assert set(loaded) == {"CXR1", "CXR2"}
    assert loaded["CXR1"].study_id == "CXR1"


def test_load_all_records_raises_on_schema_fields_mismatch(tmp_path):
    storage_root = tmp_path / "storage"
    canonical_dir = storage_root / "canonical"
    canonical_dir.mkdir(parents=True)
    full_path = canonical_dir / "iu_xray_canonical_full.json"
    full_path.write_text(json.dumps({"schema_fields": ["only_one_field"], "records": []}), encoding="utf-8")

    with pytest.raises(IuXrayManifestError, match="schema_fields"):
        load_all_records(full_path)


def test_load_all_records_raises_on_duplicate_study_id(tmp_path):
    storage_root = tmp_path / "storage"
    records = [_record_dict("CXR1"), _record_dict("CXR1")]
    full_path, _ = _write_external_manifest(storage_root, records)

    with pytest.raises(IuXrayManifestError, match="duplicate"):
        load_all_records(full_path)
