"""Tests for FileOperator.walk_files contract."""

from pathlib import Path

from ya_agent_environment import LocalTmpFileOperator


async def test_local_tmp_file_operator_walk_files_lists_logical_entries(tmp_path: Path) -> None:
    """walk_files should yield logical paths with file metadata."""
    op = LocalTmpFileOperator(tmp_path)
    await op.mkdir("src", parents=True)
    await op.write_file("src/main.py", "print('hello')")
    await op.write_file("README.md", "hello")

    entries = [entry async for entry in op.walk_files(".")]
    paths = {entry["path"] for entry in entries}

    assert "src" in paths
    assert "src/main.py" in paths
    assert "README.md" in paths
    main = next(entry for entry in entries if entry["path"] == "src/main.py")
    assert main["is_file"] is True
    assert main["is_dir"] is False
    assert main["size"] == len("print('hello')")
    assert main["mtime"] is not None


async def test_local_tmp_file_operator_walk_files_hidden_filter(tmp_path: Path) -> None:
    """walk_files should hide dot paths unless include_hidden is enabled."""
    op = LocalTmpFileOperator(tmp_path)
    await op.mkdir(".cache", parents=True)
    await op.write_file(".cache/data.txt", "hidden")
    await op.write_file("visible.txt", "visible")

    visible_entries = [entry async for entry in op.walk_files(".")]
    visible_paths = {entry["path"] for entry in visible_entries}
    assert "visible.txt" in visible_paths
    assert ".cache" not in visible_paths
    assert ".cache/data.txt" not in visible_paths

    all_entries = [entry async for entry in op.walk_files(".", include_hidden=True)]
    all_paths = {entry["path"] for entry in all_entries}
    assert ".cache" in all_paths
    assert ".cache/data.txt" in all_paths
