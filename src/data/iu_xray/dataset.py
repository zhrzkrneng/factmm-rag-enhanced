"""IU X-Ray split-aware dataset loader.

Responsibility: combine Cell 43's committed split manifest
(iu_xray_split_manifest.json) with the SHA-256-verified external full
canonical manifest (manifest_reader.py) into a loader that returns one
item per study/report (never one per image), deterministically ordered,
excluded records rejected by default, with lazy image access and a
text-only metadata path that never touches image files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.iu_xray.exceptions import IuXraySplitError
from src.data.iu_xray.images import load_images
from src.data.iu_xray.manifest_reader import (
    load_all_records,
    load_canonical_index,
    verify_external_manifest,
)
from src.data.iu_xray.paths import resolve_path, resolve_storage_root
from src.data.iu_xray.records import IuXrayCanonicalRecord

SPLIT_NAMES = ("train", "validation", "test")

# The counts Cell 43 actually produced and committed -- used only to
# validate against, never to construct or pad a split.
EXPECTED_SPLIT_COUNTS = {"train": 3060, "validation": 382, "test": 384}


def load_split_manifest(output_dir: Path) -> dict:
    """Loads the committed iu_xray_split_manifest.json.

    Raises:
        IuXraySplitError: if the file is missing or not valid JSON.
    """
    path = output_dir / "iu_xray_split_manifest.json"
    if not path.exists():
        raise IuXraySplitError(
            f"Committed split manifest not found at {path}. Run and "
            f"commit Cell 43 before using this loader."
        )
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise IuXraySplitError(f"{path} is not valid JSON: {exc}") from exc


def _validate_no_split_overlap(split_manifest: dict) -> None:
    train = set(split_manifest["train_study_ids"])
    val = set(split_manifest["validation_study_ids"])
    test = set(split_manifest["test_study_ids"])
    overlaps = {
        "train_validation": sorted(train & val),
        "train_test": sorted(train & test),
        "validation_test": sorted(val & test),
    }
    non_empty = {k: v for k, v in overlaps.items() if v}
    if non_empty:
        raise IuXraySplitError(
            f"Split manifest has overlapping study_ids: {non_empty}"
        )


class IuXrayDataset:
    """A loaded IU X-Ray dataset: canonical records + split assignment.

    Construct via `IuXrayDataset.from_repo(repo_root)` -- the
    constructor itself takes already-loaded data so it stays trivially
    unit-testable against synthetic fixtures (see tests/unit/
    test_iu_xray_dataset.py), without requiring the real external
    manifest.
    """

    def __init__(
        self,
        storage_root: Path,
        records_by_study_id: Dict[str, IuXrayCanonicalRecord],
        split_manifest: dict,
    ):
        _validate_no_split_overlap(split_manifest)
        self.storage_root = storage_root
        self._records_by_study_id = records_by_study_id
        self._split_manifest = split_manifest

    @classmethod
    def from_repo(cls, repo_root: Path, output_dir: Path = None) -> "IuXrayDataset":
        """Builds a dataset against the real, committed Cell 43 artifacts
        and the SHA-256-verified external full manifest.

        Raises:
            IuXrayManifestError: propagated from manifest_reader.py if
                the external manifest is missing or hash-mismatched --
                this is Cell 44's BLOCKED condition.
            IuXraySplitError: if the split manifest is missing/invalid
                or has overlapping study_ids.
        """
        output_dir = output_dir or (repo_root / "results" / "reproduction" / "milestone_2_8")
        storage_root = resolve_storage_root(repo_root)
        index = load_canonical_index(output_dir)
        manifest_path = verify_external_manifest(index, storage_root)
        records = load_all_records(manifest_path)
        split_manifest = load_split_manifest(output_dir)
        return cls(storage_root, records, split_manifest)

    @property
    def record_count(self) -> int:
        return len(self._records_by_study_id)

    def get_metadata(self, study_id: str) -> IuXrayCanonicalRecord:
        """Text/metadata-only access -- never opens an image file.

        Raises:
            KeyError: if `study_id` is not in the loaded record set.
        """
        return self._records_by_study_id[study_id]

    def load_images_for(self, study_id: str):
        """Lazily opens every image for one study, in image_paths order.

        Raises:
            KeyError: if `study_id` is not in the loaded record set.
            IuXrayPathResolutionError: on the first unreadable image.
        """
        record = self.get_metadata(study_id)
        resolved = [resolve_path(p, self.storage_root) for p in record.image_paths]
        return load_images(resolved)

    def load_split(
        self, split_name: str, include_excluded: bool = False
    ) -> List[IuXrayCanonicalRecord]:
        """Returns every record assigned to `split_name`, deterministically
        ordered exactly as Cell 43's split manifest recorded it (sorted
        ascending study_id, per its own documented algorithm).

        Args:
            split_name: one of "train", "validation", "test".
            include_excluded: if True, includes records whose
                validation_status is "excluded" that nonetheless appear
                in the split assignment (should never happen for a
                genuine Cell 43 split, since Cell 43 only ever assigned
                usable study_ids to a split -- this flag exists to make
                that assumption explicit and testable, not because
                excluded records are expected here). Default False.

        Raises:
            ValueError: if `split_name` is not one of SPLIT_NAMES.
            IuXraySplitError: if a split's study_id has no matching
                canonical record, or (when include_excluded is False)
                an excluded record's study_id appears in the split.
        """
        if split_name not in SPLIT_NAMES:
            raise ValueError(
                f"Unknown split_name {split_name!r}; expected one of {SPLIT_NAMES}"
            )
        study_ids = self._split_manifest[f"{split_name}_study_ids"]

        records: List[IuXrayCanonicalRecord] = []
        missing: List[str] = []
        wrongly_excluded: List[str] = []
        for study_id in study_ids:
            record = self._records_by_study_id.get(study_id)
            if record is None:
                missing.append(study_id)
                continue
            if not record.is_usable and not include_excluded:
                wrongly_excluded.append(study_id)
                continue
            records.append(record)

        if missing:
            raise IuXraySplitError(
                f"{len(missing)} study_id(s) in split {split_name!r} have "
                f"no matching canonical record (first few: {missing[:5]})"
            )
        if wrongly_excluded:
            raise IuXraySplitError(
                f"{len(wrongly_excluded)} excluded record(s) found in "
                f"split {split_name!r} (first few: {wrongly_excluded[:5]}) "
                f"-- Cell 43's split manifest should only ever contain "
                f"usable study_ids"
            )
        return records

    def validate_split_counts(self) -> Dict[str, int]:
        """Loads all three splits and returns their counts.

        Raises:
            IuXraySplitError: if any split's count does not match
                EXPECTED_SPLIT_COUNTS.
        """
        counts = {name: len(self.load_split(name)) for name in SPLIT_NAMES}
        mismatches = {
            name: (counts[name], EXPECTED_SPLIT_COUNTS[name])
            for name in SPLIT_NAMES
            if counts[name] != EXPECTED_SPLIT_COUNTS[name]
        }
        if mismatches:
            raise IuXraySplitError(
                f"Split count mismatch(es) (actual, expected): {mismatches}"
            )
        return counts
