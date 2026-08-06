"""Unit tests for src/data/iu_xray/paths.py.

Covers: (1) symbolic path resolution.
"""

from pathlib import Path

import pytest

from src.data.iu_xray.exceptions import IuXrayPathResolutionError
from src.data.iu_xray.paths import DATASET_ROOT_TOKEN, resolve_path, resolve_storage_root


def test_resolve_path_substitutes_token_prefix():
    storage_root = Path("/some/drive/mount/data/iu_xray")
    value = f"{DATASET_ROOT_TOKEN}/canonical/iu_xray_canonical_full.json"
    resolved = resolve_path(value, storage_root)
    assert resolved == "/some/drive/mount/data/iu_xray/canonical/iu_xray_canonical_full.json"


def test_resolve_path_passes_through_already_absolute_path():
    storage_root = Path("/some/drive/mount/data/iu_xray")
    already_absolute = "/content/drive/MyDrive/FactMM-RAG-Enhanced/data/iu_xray/raw/x.png"
    assert resolve_path(already_absolute, storage_root) == already_absolute


def test_resolve_path_rejects_empty_string():
    with pytest.raises(IuXrayPathResolutionError, match="empty"):
        resolve_path("", Path("/anywhere"))


def test_resolve_storage_root_uses_local_fallback_when_drive_not_mounted(tmp_path):
    # Injects a guaranteed-nonexistent drive_mount_root rather than
    # relying on the real /content/drive/MyDrive's ambient state --
    # this must pass identically whether or not the environment this
    # test happens to run in actually has Drive mounted (it does in a
    # real Colab session, and must not in a sandbox with no Drive at
    # all; hardcoding either assumption breaks the other environment).
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    never_mounted = tmp_path / "not_a_real_drive_mount"
    result = resolve_storage_root(repo_root, drive_mount_root=never_mounted)
    assert result == repo_root / "data" / "iu_xray"


def test_resolve_storage_root_uses_drive_path_when_mounted(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    fake_drive_mount = tmp_path / "fake_drive_mount"
    fake_drive_mount.mkdir()
    result = resolve_storage_root(repo_root, drive_mount_root=fake_drive_mount)
    assert result == fake_drive_mount / "FactMM-RAG-Enhanced" / "data" / "iu_xray"
