"""IU X-Ray storage-root resolution and ${IU_XRAY_STORAGE_ROOT} path
substitution.

Responsibility: reproduce, on the read/loader side, exactly the same
storage-location convention Cell 43's acquisition script used on the
write side (results/reproduction/milestone_2_8/iu_xray_canonical_manifest.json's
"dataset_root_convention" field) -- Google Drive first if mounted,
otherwise a local project-relative fallback -- and resolve the
${IU_XRAY_STORAGE_ROOT} symbolic token that appears in the Git-tracked
lightweight manifests (never in absolute, machine-specific form) back
into a real path for this run.
"""

from __future__ import annotations

from pathlib import Path

from src.data.iu_xray.exceptions import IuXrayPathResolutionError

DATASET_ROOT_TOKEN = "${IU_XRAY_STORAGE_ROOT}"

_DRIVE_MOUNT_ROOT = Path("/content/drive/MyDrive")
_DRIVE_SUBPATH = ("FactMM-RAG-Enhanced", "data", "iu_xray")
_LOCAL_SUBPATH = ("data", "iu_xray")


def resolve_storage_root(repo_root: Path, drive_mount_root: Path = None) -> Path:
    """Resolves IU_XRAY_STORAGE_ROOT the same way Cell 43 did.

    Args:
        repo_root: this repository's root, used for the local fallback.
        drive_mount_root: defaults to the real `/content/drive/MyDrive`.
            Overridable so tests can exercise the local-fallback branch
            deterministically regardless of whether the real environment
            this test happens to run in has Drive mounted or not (a
            hardcoded assumption that Drive is never mounted broke this
            same test the first time it ran in a real Colab session,
            where Drive genuinely is mounted).

    Returns:
        The mounted Drive path (`drive_mount_root/FactMM-RAG-Enhanced/
        data/iu_xray`) if `drive_mount_root` exists; otherwise
        `repo_root/data/iu_xray`. Existence of the *storage root itself*
        is not required here (a fresh environment may not have run
        acquisition yet) -- callers that need real data must check that
        explicitly (see manifest_reader.py).
    """
    drive_mount_root = drive_mount_root if drive_mount_root is not None else _DRIVE_MOUNT_ROOT
    if drive_mount_root.exists():
        return drive_mount_root.joinpath(*_DRIVE_SUBPATH)
    return repo_root.joinpath(*_LOCAL_SUBPATH)


def resolve_path(value: str, storage_root: Path) -> str:
    """Resolves one path string, substituting DATASET_ROOT_TOKEN if
    present at the start of `value`; otherwise returns `value` unchanged
    (it is already an absolute path, as the external full manifest's
    own per-record image_paths/source_metadata.source_file entries are
    -- only the small Git-tracked index files use the symbolic token).

    Args:
        value: a path string, either token-prefixed or already absolute.
        storage_root: this run's resolved IU_XRAY_STORAGE_ROOT.

    Raises:
        IuXrayPathResolutionError: if `value` is empty.
    """
    if not value:
        raise IuXrayPathResolutionError(
            "Cannot resolve an empty path string against IU_XRAY_STORAGE_ROOT."
        )
    if value.startswith(DATASET_ROOT_TOKEN):
        return str(storage_root) + value[len(DATASET_ROOT_TOKEN):]
    return value
