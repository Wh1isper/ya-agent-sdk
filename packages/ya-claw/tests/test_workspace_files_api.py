from __future__ import annotations

import asyncio
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pytest
from fastapi.testclient import TestClient
from ya_claw.api.workspace import _stream_download
from ya_claw.app import create_app
from ya_claw.config import ClawSettings, get_settings
from ya_claw.controller import windows_workspace_files as windows_workspace_files_module
from ya_claw.controller import workspace_files as workspace_files_module
from ya_claw.controller.workspace_files import WorkspaceDownload
from ya_claw.db.engine import create_engine
from ya_claw.orm.base import Base
from ya_claw.workspace import DockerWorkspaceProvider


@pytest.fixture(autouse=True)
def clear_claw_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for env_name in (
        "YA_CLAW_API_TOKEN",
        "YA_CLAW_DATABASE_URL",
        "YA_CLAW_DATA_DIR",
        "YA_CLAW_WEB_DIST_DIR",
        "YA_CLAW_WORKSPACE_DIR",
        "YA_CLAW_WORKSPACE_DOWNLOAD_MAX_BYTES",
        "YA_CLAW_WORKSPACE_PROVIDER_BACKEND",
        "YA_CLAW_PROFILE_SEED_FILE",
        "YA_CLAW_AUTO_SEED_PROFILES",
        "YA_CLAW_SCHEDULE_DISPATCH_ENABLED",
        "YA_CLAW_HEARTBEAT_ENABLED",
        "YA_CLAW_BRIDGE_DISPATCH_MODE",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("YA_CLAW_API_TOKEN", "test-token")
    monkeypatch.setenv("YA_CLAW_DATA_DIR", str(tmp_path / "runtime-data"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("YA_CLAW_WORKSPACE_PROVIDER_BACKEND", "local")
    monkeypatch.setenv("YA_CLAW_AUTO_SEED_PROFILES", "false")
    monkeypatch.setenv("YA_CLAW_SCHEDULE_DISPATCH_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_HEARTBEAT_ENABLED", "false")
    monkeypatch.setenv("YA_CLAW_BRIDGE_DISPATCH_MODE", "manual")

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    _create_schema()
    with TestClient(create_app()) as test_client:
        yield test_client


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def _create_schema() -> None:
    async def _run() -> None:
        settings = get_settings()
        engine = create_engine(settings.resolved_database_url)
        try:
            async with engine.begin() as connection:
                await connection.run_sync(Base.metadata.create_all)
        finally:
            await engine.dispose()

    asyncio.run(_run())


def _create_session(client: TestClient, *, workspace: dict[str, object] | None = None) -> str:
    payload = {"workspace": workspace} if workspace is not None else {}
    response = client.post("/api/v1/sessions", headers=_auth_headers(), json=payload)
    assert response.status_code == 201
    return response.json()["session"]["id"]


def test_workspace_files_list_read_and_download_use_virtual_paths(client: TestClient, tmp_path: Path) -> None:
    mounted_workspace = tmp_path / "mounted-project"
    session_id = _create_session(
        client,
        workspace={
            "mounts": [
                {
                    "id": "project",
                    "host_path": str(mounted_workspace),
                    "virtual_path": "/workspace/project",
                }
            ]
        },
    )
    mounted_workspace.mkdir(parents=True, exist_ok=True)
    (mounted_workspace / "notes.txt").write_text("hello, 世界\n", encoding="utf-8")
    (mounted_workspace / "artifact.bin").write_bytes(b"\x00\xffartifact")
    (mounted_workspace / ".secret").write_text("hidden", encoding="utf-8")
    (mounted_workspace / "src").mkdir()

    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace/project", "limit": 20},
    )
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert list_payload["session_id"] == session_id
    assert list_payload["path"] == "/workspace/project"
    assert list_payload["limit"] == 20
    assert list_payload["offset"] == 0
    assert list_payload["has_more"] is False
    assert list_payload["next_cursor"] is None
    assert list_payload["next_offset"] is None
    assert list_payload["truncated"] is False
    assert [item["name"] for item in list_payload["items"]] == ["artifact.bin", "notes.txt", "src"]
    assert list_payload["items"][1]["path"] == "/workspace/project/notes.txt"
    assert list_payload["items"][1]["kind"] == "file"
    assert list_payload["items"][2]["kind"] == "directory"
    assert str(mounted_workspace) not in list_response.text

    hidden_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace/project", "include_hidden": "true", "limit": 1},
    )
    assert hidden_response.status_code == 200
    hidden_payload = hidden_response.json()
    assert hidden_payload["items"][0]["name"] == ".secret"
    assert hidden_payload["items"][0]["hidden"] is True
    assert hidden_payload["offset"] == 0
    assert hidden_payload["has_more"] is True
    assert isinstance(hidden_payload["next_cursor"], str)
    assert hidden_payload["next_offset"] == 1
    assert hidden_payload["truncated"] is True

    second_page_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={
            "path": "/workspace/project",
            "include_hidden": "true",
            "limit": 2,
            "cursor": hidden_payload["next_cursor"],
        },
    )
    assert second_page_response.status_code == 200
    second_page = second_page_response.json()
    assert second_page["offset"] == 1
    assert [item["name"] for item in second_page["items"]] == ["artifact.bin", "notes.txt"]
    assert second_page["has_more"] is True
    assert isinstance(second_page["next_cursor"], str)
    assert second_page["next_offset"] == 3

    final_page_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={
            "path": "/workspace/project",
            "include_hidden": "true",
            "limit": 2,
            "cursor": second_page["next_cursor"],
        },
    )
    assert final_page_response.status_code == 200
    final_page = final_page_response.json()
    assert [item["name"] for item in final_page["items"]] == ["src"]
    assert final_page["has_more"] is False
    assert final_page["next_cursor"] is None
    assert final_page["next_offset"] is None

    legacy_offset_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={
            "path": "/workspace/project",
            "include_hidden": "true",
            "limit": 2,
            "offset": 1,
        },
    )
    assert legacy_offset_response.status_code == 200
    legacy_offset_page = legacy_offset_response.json()
    assert legacy_offset_page["offset"] == 1
    assert [item["name"] for item in legacy_offset_page["items"]] == ["artifact.bin", "notes.txt"]
    assert legacy_offset_page["next_offset"] == 3
    assert isinstance(legacy_offset_page["next_cursor"], str)

    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/project/notes.txt"},
    )
    assert read_response.status_code == 200
    assert read_response.json() == {
        "session_id": session_id,
        "path": "/workspace/project/notes.txt",
        "content": "hello, 世界\n",
        "encoding": "utf-8",
        "size_bytes": len("hello, 世界\n".encode()),
    }
    assert str(mounted_workspace) not in read_response.text

    download_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file:download",
        headers=_auth_headers(),
        params={"path": "/workspace/project/artifact.bin"},
    )
    assert download_response.status_code == 200
    assert download_response.content == b"\x00\xffartifact"
    assert download_response.headers["content-type"] == "application/octet-stream"
    assert download_response.headers["content-disposition"] == 'attachment; filename="artifact.bin"'
    assert "content-length" not in download_response.headers
    assert str(mounted_workspace) not in str(download_response.headers)


def test_workspace_cursor_is_stable_when_entries_before_it_are_inserted_or_deleted(client: TestClient) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    for name in ("alpha.txt", "bravo.txt", "charlie.txt", "delta.txt", "echo.txt"):
        (workspace / name).write_text(name, encoding="utf-8")

    first_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace", "limit": 2},
    )
    assert first_response.status_code == 200
    first_page = first_response.json()
    assert [item["name"] for item in first_page["items"]] == ["alpha.txt", "bravo.txt"]
    assert isinstance(first_page["next_cursor"], str)

    # Offset pagination would now either repeat bravo.txt or skip charlie.txt.
    # The cursor resumes strictly after bravo's stable case-insensitive name key.
    (workspace / "alpha.txt").unlink()
    (workspace / "aardvark.txt").write_text("new", encoding="utf-8")
    second_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={
            "path": "/workspace",
            "limit": 2,
            "cursor": first_page["next_cursor"],
        },
    )
    assert second_response.status_code == 200
    second_page = second_response.json()
    assert [item["name"] for item in second_page["items"]] == ["charlie.txt", "delta.txt"]

    (workspace / "aardvark.txt").unlink()
    (workspace / "able.txt").write_text("newer", encoding="utf-8")
    third_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={
            "path": "/workspace",
            "limit": 2,
            "cursor": second_page["next_cursor"],
        },
    )
    assert third_response.status_code == 200
    third_page = third_response.json()
    assert [item["name"] for item in third_page["items"]] == ["echo.txt"]
    assert third_page["has_more"] is False
    assert third_page["next_cursor"] is None

    collected = [
        *(item["name"] for item in first_page["items"]),
        *(item["name"] for item in second_page["items"]),
        *(item["name"] for item in third_page["items"]),
    ]
    assert collected == ["alpha.txt", "bravo.txt", "charlie.txt", "delta.txt", "echo.txt"]
    assert len(collected) == len(set(collected))


def test_workspace_files_reject_invalid_or_ambiguous_cursor(client: TestClient) -> None:
    session_id = _create_session(client)

    invalid_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace", "cursor": "not-a-cursor"},
    )
    ambiguous_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace", "cursor": "not-a-cursor", "offset": 1},
    )

    assert invalid_response.status_code == 400
    assert "cursor is invalid" in invalid_response.text
    assert ambiguous_response.status_code == 400


def test_workspace_download_rejects_files_above_configured_limit(client: TestClient) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    (workspace / "too-large.bin").write_bytes(b"12345")
    settings = client.app.state.settings
    assert isinstance(settings, ClawSettings)
    settings.workspace_download_max_bytes = 4

    response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file:download",
        headers=_auth_headers(),
        params={"path": "/workspace/too-large.bin"},
    )

    assert response.status_code == 413
    assert "4-byte download limit" in response.text


def test_workspace_download_stream_enforces_limit_if_open_file_grows() -> None:
    source = BytesIO(b"12345")
    download = WorkspaceDownload(
        file=source,
        filename="growing.bin",
        size_bytes=4,
        max_bytes=4,
    )

    with pytest.raises(RuntimeError, match="configured download limit"):
        list(_stream_download(download))

    assert source.closed is True


def test_workspace_files_return_404_for_unknown_session_and_path(client: TestClient) -> None:
    unknown_session_response = client.get(
        "/api/v1/sessions/missing/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace"},
    )
    assert unknown_session_response.status_code == 404

    session_id = _create_session(client)
    missing_path_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/missing.txt"},
    )
    assert missing_path_response.status_code == 404


def test_workspace_files_reject_traversal_and_host_or_relative_paths(client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(client)
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("must not leak", encoding="utf-8")

    traversal_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/../outside.txt"},
    )
    relative_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "outside.txt"},
    )
    outside_mount_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/outside.txt"},
    )
    host_path_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": str(outside_file)},
    )

    assert traversal_response.status_code == 400
    assert relative_response.status_code == 400
    assert outside_mount_response.status_code == 403
    assert host_path_response.status_code == 403
    combined_response = (
        traversal_response.text + relative_response.text + outside_mount_response.text + host_path_response.text
    )
    assert "must not leak" not in combined_response


def test_workspace_files_block_symlink_escape(client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    outside_file = tmp_path / "outside-secret.txt"
    outside_file.write_text("outside secret", encoding="utf-8")
    (workspace / "escape.txt").symlink_to(outside_file)

    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace"},
    )
    assert list_response.status_code == 200
    assert list_response.json()["items"][0]["kind"] == "symlink"
    assert str(outside_file) not in list_response.text

    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/escape.txt"},
    )
    download_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file:download",
        headers=_auth_headers(),
        params={"path": "/workspace/escape.txt"},
    )
    assert read_response.status_code == 403
    assert download_response.status_code == 403
    assert "outside secret" not in read_response.text + download_response.text
    assert str(outside_file) not in read_response.text + download_response.text


@pytest.mark.skipif(os.name != "nt", reason="requires Windows junctions")
def test_workspace_files_block_windows_junction_escape(client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    outside_directory = tmp_path / "junction-outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("junction outside secret", encoding="utf-8")
    junction = workspace / "junction"
    subprocess.run(  # noqa: S603 - fixed test arguments create a temporary junction
        [os.environ["COMSPEC"], "/c", "mklink", "/J", str(junction), str(outside_directory)],
        check=True,
        capture_output=True,
        text=True,
    )

    try:
        list_response = client.get(
            f"/api/v1/sessions/{session_id}/workspace/files",
            headers=_auth_headers(),
            params={"path": "/workspace"},
        )
        nested_list_response = client.get(
            f"/api/v1/sessions/{session_id}/workspace/files",
            headers=_auth_headers(),
            params={"path": "/workspace/junction"},
        )
        read_response = client.get(
            f"/api/v1/sessions/{session_id}/workspace/file",
            headers=_auth_headers(),
            params={"path": "/workspace/junction/secret.txt"},
        )
        download_response = client.get(
            f"/api/v1/sessions/{session_id}/workspace/file:download",
            headers=_auth_headers(),
            params={"path": "/workspace/junction/secret.txt"},
        )

        assert list_response.status_code == 200
        junction_entry = next(item for item in list_response.json()["items"] if item["name"] == "junction")
        assert junction_entry["kind"] == "symlink"
        assert nested_list_response.status_code == 403
        assert read_response.status_code == 403
        assert download_response.status_code == 403
        combined_response = nested_list_response.text + read_response.text + download_response.text
        assert "junction outside secret" not in combined_response
        assert str(outside_directory) not in combined_response
    finally:
        junction.rmdir()


@pytest.mark.skipif(os.name != "nt", reason="requires Windows handle sharing semantics")
def test_windows_directory_handles_block_component_replacement(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    nested_directory = workspace / "nested"
    nested_directory.mkdir()
    (nested_directory / "safe.txt").write_text("safe workspace data", encoding="utf-8")
    displaced_directory = workspace / "nested-before-race"
    outside_directory = tmp_path / "windows-race-outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("windows race secret", encoding="utf-8")

    real_open_handle = windows_workspace_files_module._open_handle
    blocked_replacements = 0

    def racing_open_handle(
        path: Path,
        *,
        expect_directory: bool,
    ) -> windows_workspace_files_module._WindowsHandle:
        nonlocal blocked_replacements
        opened_handle = real_open_handle(path, expect_directory=expect_directory)
        if expect_directory and path == nested_directory:
            try:
                nested_directory.rename(displaced_directory)
            except OSError:
                blocked_replacements += 1
            else:
                displaced_directory.rename(nested_directory)
                raise AssertionError("A pinned Windows directory was renameable.")
        return opened_handle

    monkeypatch.setattr(windows_workspace_files_module, "_open_handle", racing_open_handle)
    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace/nested"},
    )
    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/nested/safe.txt"},
    )

    assert blocked_replacements == 2
    assert list_response.status_code == 200
    assert [item["name"] for item in list_response.json()["items"]] == ["safe.txt"]
    assert read_response.status_code == 200
    assert read_response.json()["content"] == "safe workspace data"


def test_workspace_files_path_fallback_reads_files_and_blocks_symlinks(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    (workspace / "safe.txt").write_text("safe workspace data", encoding="utf-8")
    outside_file = tmp_path / "fallback-outside.txt"
    outside_file.write_text("outside fallback secret", encoding="utf-8")
    (workspace / "escape.txt").symlink_to(outside_file)
    monkeypatch.setattr(workspace_files_module, "_OPEN_SUPPORTS_DIR_FD", False)
    monkeypatch.setattr(workspace_files_module, "_PATH_FALLBACK_SUPPORTED", True)

    @contextmanager
    def pinned_directory(root: Path, relative_parts: tuple[str, ...]) -> Iterator[Path]:
        yield root.joinpath(*relative_parts)

    def open_regular_file(root: Path, relative_parts: tuple[str, ...]) -> tuple[BinaryIO, int]:
        target = root.joinpath(*relative_parts)
        if target.is_symlink():
            raise workspace_files_module.WindowsWorkspaceError("unsafe")
        return target.open("rb"), target.stat().st_size

    monkeypatch.setattr(workspace_files_module, "pinned_windows_directory", pinned_directory)
    monkeypatch.setattr(workspace_files_module, "open_windows_regular_file", open_regular_file)

    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace"},
    )
    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/safe.txt"},
    )
    download_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file:download",
        headers=_auth_headers(),
        params={"path": "/workspace/safe.txt"},
    )
    escape_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/escape.txt"},
    )

    assert list_response.status_code == 200
    assert [item["name"] for item in list_response.json()["items"]] == ["escape.txt", "safe.txt"]
    assert read_response.status_code == 200
    assert read_response.json()["content"] == "safe workspace data"
    assert download_response.status_code == 200
    assert download_response.content == b"safe workspace data"
    assert escape_response.status_code == 403
    assert "outside fallback secret" not in escape_response.text
    assert str(outside_file) not in escape_response.text


def test_workspace_files_block_intermediate_directory_symlink(client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    outside_directory = tmp_path / "outside-directory"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("outside directory secret", encoding="utf-8")
    (workspace / "linked-directory").symlink_to(outside_directory, target_is_directory=True)

    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace/linked-directory"},
    )
    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/linked-directory/secret.txt"},
    )
    download_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file:download",
        headers=_auth_headers(),
        params={"path": "/workspace/linked-directory/secret.txt"},
    )

    assert list_response.status_code == 403
    assert read_response.status_code == 403
    assert download_response.status_code == 403
    combined_response = list_response.text + read_response.text + download_response.text
    assert "outside directory secret" not in combined_response
    assert str(outside_directory) not in combined_response


def test_workspace_directory_scan_cap_preserves_global_sorting_contract(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    for name in ("a.txt", "b.txt", "c.txt"):
        (workspace / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(workspace_files_module, "MAX_WORKSPACE_DIRECTORY_SCAN_ENTRIES", 2)

    response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace", "limit": 1},
    )

    # Do not return a potentially incorrectly sorted partial page when the safe
    # scan cap prevents considering every directory entry.
    assert response.status_code == 413
    assert "2-entry safe scan limit" in response.text


def test_workspace_files_resist_intermediate_directory_replacement_race(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    nested_directory = workspace / "nested"
    nested_directory.mkdir()
    (nested_directory / "safe.txt").write_text("safe workspace data", encoding="utf-8")
    displaced_directory = workspace / "nested-before-race"
    outside_directory = tmp_path / "race-outside"
    outside_directory.mkdir()
    (outside_directory / "secret.txt").write_text("race outside secret", encoding="utf-8")

    real_open = os.open
    replacement_performed = False
    uses_secure_descriptors = workspace_files_module._secure_fd_operations_available()
    if not uses_secure_descriptors:
        pytest.skip("requires descriptor-relative no-follow traversal")

    def racing_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replacement_performed
        descriptor_race = path == "nested" and dir_fd is not None
        if descriptor_race and not replacement_performed:
            nested_directory.rename(displaced_directory)
            nested_directory.symlink_to(outside_directory, target_is_directory=True)
            replacement_performed = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(workspace_files_module.os, "open", racing_open)
    response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/nested/secret.txt"},
    )

    assert replacement_performed is True
    assert response.status_code == 403
    assert "race outside secret" not in response.text
    assert str(outside_directory) not in response.text


def test_workspace_text_read_rejects_binary_and_files_over_one_mib(client: TestClient) -> None:
    session_id = _create_session(client)
    workspace = get_settings().resolved_workspace_dir
    (workspace / "binary.dat").write_bytes(b"\x00\x01\x02")
    (workspace / "too-large.txt").write_bytes(b"a" * (1024 * 1024 + 1))

    binary_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/binary.dat"},
    )
    too_large_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/too-large.txt"},
    )

    assert binary_response.status_code == 415
    assert too_large_response.status_code == 413


@pytest.mark.parametrize(
    ("endpoint", "params"),
    [
        ("files", {"path": "/workspace"}),
        ("file", {"path": "/workspace/example.txt"}),
        ("file:download", {"path": "/workspace/example.txt"}),
    ],
)
def test_workspace_file_endpoints_require_auth(
    client: TestClient,
    endpoint: str,
    params: dict[str, str],
) -> None:
    response = client.get(f"/api/v1/sessions/session-1/workspace/{endpoint}", params=params)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_workspace_files_use_docker_service_visible_mount(client: TestClient, tmp_path: Path) -> None:
    session_id = _create_session(client)
    service_workspace = tmp_path / "service-visible-workspace"
    docker_host_workspace = tmp_path / "daemon-only-workspace"
    service_workspace.mkdir()
    docker_host_workspace.mkdir()
    (service_workspace / "service.txt").write_text("visible to the API", encoding="utf-8")
    (docker_host_workspace / "host-only.txt").write_text("not service data", encoding="utf-8")
    client.app.state.workspace_provider = DockerWorkspaceProvider(
        service_workspace,
        image="python:3.11",
        docker_host_workspace_dir=docker_host_workspace,
    )

    list_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/files",
        headers=_auth_headers(),
        params={"path": "/workspace"},
    )
    read_response = client.get(
        f"/api/v1/sessions/{session_id}/workspace/file",
        headers=_auth_headers(),
        params={"path": "/workspace/service.txt"},
    )

    assert list_response.status_code == 200
    assert [item["name"] for item in list_response.json()["items"]] == ["service.txt"]
    assert read_response.status_code == 200
    assert read_response.json()["content"] == "visible to the API"
    assert str(service_workspace) not in list_response.text + read_response.text
    assert str(docker_host_workspace) not in list_response.text + read_response.text
