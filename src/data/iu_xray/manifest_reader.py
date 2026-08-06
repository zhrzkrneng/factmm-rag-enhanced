"""IU X-Ray external full canonical manifest reader.

Responsibility: locate, SHA-256-verify, and parse Cell 43's external
full canonical manifest (iu_xray_canonical_full.json -- never committed
to Git; lives next to the raw dataset under IU_XRAY_STORAGE_ROOT), using
the committed lightweight index (results/reproduction/milestone_2_8/
iu_xray_canonical_manifest.json) as the source of truth for its expected
path and hash. Never trusts a manifest found on disk without verifying
its hash first -- a stale or partially-synced Drive copy must be
detected, not silently loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from src.data.iu_xray.exceptions import IuXrayManifestError
from src.data.iu_xray.paths import resolve_path
from src.data.iu_xray.records import CANONICAL_SCHEMA_FIELDS, IuXrayCanonicalRecord


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_canonical_index(output_dir: Path) -> dict:
    """Loads the committed, lightweight iu_xray_canonical_manifest.json.

    Args:
        output_dir: results/reproduction/milestone_2_8/ (or wherever
            Cell 43's artifacts live for this repo checkout).

    Raises:
        IuXrayManifestError: if the index file itself is missing or not
            valid JSON -- this is a committed, Git-tracked file, so its
            absence means Cell 43 has not been run/committed yet, not a
            real-data availability question.
    """
    index_path = output_dir / "iu_xray_canonical_manifest.json"
    if not index_path.exists():
        raise IuXrayManifestError(
            f"Committed canonical manifest index not found at {index_path}. "
            f"Run and commit Cell 43 before using this loader."
        )
    try:
        with index_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise IuXrayManifestError(
            f"{index_path} is not valid JSON: {exc}"
        ) from exc


def resolve_external_manifest_path(index: dict, storage_root: Path) -> Path:
    """Resolves the external full manifest's real path for this run."""
    token_path = index.get("full_manifest_external_path")
    if not token_path:
        raise IuXrayManifestError(
            "iu_xray_canonical_manifest.json has no full_manifest_external_path "
            "-- Cell 43's canonicalization produced zero usable records "
            "(record_count=0), so there is no external manifest to load."
        )
    return Path(resolve_path(token_path, storage_root))


def verify_external_manifest(
    index: dict, storage_root: Path
) -> Path:
    """Locates and SHA-256-verifies the external full manifest.

    Returns:
        The verified, existing external manifest path.

    Raises:
        IuXrayManifestError: if the file is missing, or its SHA-256
            does not match index["full_manifest_sha256"]. Never
            proceeds past a hash mismatch -- per Cell 44's own
            instructions, this is the definitive BLOCKED condition.
    """
    manifest_path = resolve_external_manifest_path(index, storage_root)
    if not manifest_path.exists():
        raise IuXrayManifestError(
            f"External full canonical manifest not found at {manifest_path} "
            f"(resolved from IU_XRAY_STORAGE_ROOT={storage_root}). It lives "
            f"outside Git next to the raw dataset -- if this environment "
            f"never ran real acquisition/canonicalization (Cell 43), it "
            f"will not be present here."
        )
    expected_sha256 = index.get("full_manifest_sha256")
    actual_sha256 = sha256_of_file(manifest_path)
    if expected_sha256 != actual_sha256:
        raise IuXrayManifestError(
            f"SHA-256 mismatch for {manifest_path}: expected "
            f"{expected_sha256!r} (from the committed canonical manifest "
            f"index), got {actual_sha256!r}. Refusing to load a manifest "
            f"that does not match what Cell 43 actually produced."
        )
    return manifest_path


def load_all_records(manifest_path: Path) -> Dict[str, IuXrayCanonicalRecord]:
    """Loads every record from an already hash-verified external manifest.

    Returns:
        study_id -> IuXrayCanonicalRecord, for every record in the
        manifest (usable and excluded alike -- filtering is
        dataset.py's job, not this module's).

    Raises:
        IuXrayManifestError: if the manifest's own schema_fields do not
            exactly match this package's IuXrayCanonicalRecord fields
            (a canonicalization-script/loader version mismatch), or if
            two records share the same study_id (this loader never
            silently keeps only one).
    """
    with manifest_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    on_disk_fields = tuple(payload.get("schema_fields", ()))
    if on_disk_fields != CANONICAL_SCHEMA_FIELDS:
        raise IuXrayManifestError(
            f"External manifest schema_fields {on_disk_fields!r} does not "
            f"match this loader's expected fields {CANONICAL_SCHEMA_FIELDS!r}. "
            f"The manifest was likely produced by a different version of "
            f"the Cell 43 canonicalization script than this loader targets."
        )

    records: Dict[str, IuXrayCanonicalRecord] = {}
    duplicate_study_ids: List[str] = []
    for raw in payload.get("records", []):
        record = IuXrayCanonicalRecord.from_dict(raw)
        if record.study_id in records:
            duplicate_study_ids.append(record.study_id)
            continue
        records[record.study_id] = record

    if duplicate_study_ids:
        raise IuXrayManifestError(
            f"External manifest contains {len(duplicate_study_ids)} "
            f"duplicate study_id(s) (first few: {duplicate_study_ids[:5]}). "
            f"Cell 43's own integrity report should have flagged these "
            f"under duplicate_study_ids -- refusing to silently keep one "
            f"copy."
        )
    return records
