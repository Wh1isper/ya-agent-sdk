"""Tests for managed temporary storage filesystem safety."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import ya_agent_sdk.environment.temporary_storage as temporary_storage_module
from ya_agent_sdk.environment.temporary_storage import (
    DirectoryIdentity,
    capture_directory_identity,
    create_owned_tmp_directory,
    remove_owned_tmp_directory,
    write_tmp_gitignore,
)


def test_write_tmp_gitignore_does_not_follow_existing_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exclusive marker creation cannot truncate a symlink target."""
    instance = tmp_path / "instance"
    instance.mkdir()
    target = tmp_path / "target.txt"
    target.write_text("keep")
    parent_identity = capture_directory_identity(tmp_path)
    directory_identity = capture_directory_identity(instance)
    original_verify = temporary_storage_module.verify_owned_tmp_directory

    def verify_then_place_symlink(
        path: Path,
        *,
        parent_identity: DirectoryIdentity,
        directory_identity: DirectoryIdentity,
    ) -> bool:
        verified = original_verify(
            path,
            parent_identity=parent_identity,
            directory_identity=directory_identity,
        )
        try:
            (instance / ".gitignore").symlink_to(target)
        except (NotImplementedError, OSError) as exc:
            pytest.skip(f"file symlinks are unavailable: {exc}")
        return verified

    monkeypatch.setattr(temporary_storage_module, "verify_owned_tmp_directory", verify_then_place_symlink)

    with pytest.raises(FileExistsError):
        write_tmp_gitignore(
            instance,
            parent_identity=parent_identity,
            directory_identity=directory_identity,
        )

    assert target.read_text() == "keep"


def test_write_tmp_gitignore_retries_short_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Marker creation writes the exact raw bytes even after a short write."""
    instance = tmp_path / "instance"
    instance.mkdir()
    parent_identity = capture_directory_identity(tmp_path)
    directory_identity = capture_directory_identity(instance)
    original_write = os.write
    write_count = 0

    def write_one_byte(descriptor: int, content: memoryview) -> int:
        nonlocal write_count
        write_count += 1
        return original_write(descriptor, content[:1])

    monkeypatch.setattr(os, "write", write_one_byte)

    marker = write_tmp_gitignore(
        instance,
        parent_identity=parent_identity,
        directory_identity=directory_identity,
    )

    assert marker.read_bytes() == b"*\n"
    assert write_count == 2


@pytest.mark.skipif(os.name != "posix", reason="directory-fd creation is POSIX-specific")
def test_create_owned_tmp_directory_rejects_replaced_parent(tmp_path: Path) -> None:
    """Instance creation does not enter a replacement parent directory."""
    parent = tmp_path / ".tmp"
    parent.mkdir()
    parent_identity = capture_directory_identity(parent)
    parent.rename(tmp_path / ".tmp-owned")
    parent.mkdir()

    with pytest.raises(RuntimeError, match="parent ownership changed"):
        create_owned_tmp_directory(
            parent,
            parent_identity=parent_identity,
            instance_name="ya-agent-instance",
        )

    assert not (parent / "ya-agent-instance").exists()


@pytest.mark.skipif(os.name != "posix", reason="directory-fd marker creation is POSIX-specific")
def test_write_tmp_gitignore_rejects_replaced_parent(tmp_path: Path) -> None:
    """Marker creation cannot enter a replacement parent directory."""
    parent = tmp_path / ".tmp"
    instance = parent / "ya-agent-instance"
    instance.mkdir(parents=True)
    parent_identity = capture_directory_identity(parent)
    directory_identity = capture_directory_identity(instance)
    parent.rename(tmp_path / ".tmp-owned")
    replacement_instance = parent / instance.name
    replacement_instance.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="parent ownership changed"):
        write_tmp_gitignore(
            instance,
            parent_identity=parent_identity,
            directory_identity=directory_identity,
        )

    assert not (replacement_instance / ".gitignore").exists()


@pytest.mark.skipif(os.name != "posix", reason="directory-fd cleanup is POSIX-specific")
def test_remove_owned_tmp_directory_stays_anchored_after_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup follows stable directory handles rather than a replaced parent path."""
    tmp_parent = tmp_path / ".tmp"
    instance = tmp_parent / "ya-agent-instance"
    nested = instance / "nested"
    nested.mkdir(parents=True)
    (nested / "owned.txt").write_text("owned")
    parent_identity = capture_directory_identity(tmp_parent)
    directory_identity = capture_directory_identity(instance)
    moved_parent = tmp_path / ".tmp-owned"
    victim = tmp_parent / instance.name
    original_remove_contents = temporary_storage_module._remove_directory_contents_fd
    replaced = False

    def replace_parent_then_remove(directory_fd: int) -> None:
        nonlocal replaced
        if not replaced:
            replaced = True
            tmp_parent.rename(moved_parent)
            victim.mkdir(parents=True)
            (victim / "keep.txt").write_text("not owned")
        original_remove_contents(directory_fd)

    monkeypatch.setattr(temporary_storage_module, "_remove_directory_contents_fd", replace_parent_then_remove)

    remove_owned_tmp_directory(
        instance,
        parent_identity=parent_identity,
        directory_identity=directory_identity,
    )

    assert not (moved_parent / instance.name).exists()
    assert (victim / "keep.txt").read_text() == "not owned"
