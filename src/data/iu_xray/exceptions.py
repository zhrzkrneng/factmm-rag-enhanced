"""IU X-Ray adaptation layer exception types.

Responsibility: exception classes specific to adapting IU X-Ray/Open-I
(Milestone 2.8, Cell 43's canonicalized output) into this project's
baseline data contracts. Kept local to this package rather than added to
src/common/exceptions.py, since none of baseline-v1.0's own code raises
or catches these -- this is new, additive, IU-X-Ray-specific surface
area, not a change to any existing interface.
"""

from __future__ import annotations


class IuXrayManifestError(Exception):
    """Raised when the external full canonical manifest (Cell 43's
    iu_xray_canonical_full.json, stored outside Git next to the raw
    dataset) is missing, unreadable, or fails SHA-256 verification
    against the value recorded in the committed
    iu_xray_canonical_manifest.json index.

    Also raised when the on-disk manifest's own schema_fields do not
    match this package's IuXrayCanonicalRecord field set, which would
    indicate a canonicalization-script/loader version mismatch rather
    than a recoverable edge case.
    """


class IuXrayPathResolutionError(Exception):
    """Raised when IU_XRAY_STORAGE_ROOT cannot be resolved (neither a
    mounted Google Drive path nor a local project data path convention
    exists), or when a record's image_paths entries do not exist on
    disk after resolution.
    """


class IuXraySchemaMappingError(Exception):
    """Raised when an IU X-Ray canonical record cannot be represented
    under the existing baseline ReportRecord contract without either
    fabricating clinical text or silently dropping a field the
    baseline contract requires to be non-null (e.g. `finding` or
    `impression` are None on an otherwise "valid" IU X-Ray record).

    Never raised for fields the baseline contract simply does not have
    (indication, comparison, mesh_terms, labels, ...) -- those are
    documented losses, not errors; see schema_mapping.MAPPING_TABLE.
    """


class IuXraySplitError(Exception):
    """Raised when a dataset split's real contents disagree with Cell
    43's iu_xray_split_manifest.json -- a study_id in a split with no
    matching canonical record, an excluded record's study_id appearing
    in any split, a split-count mismatch against the recorded
    train/validation/test counts, or cross-split overlap.
    """
