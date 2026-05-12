"""Tests for generate_filetree and related utilities."""

from pathlib import Path

from ya_agent_environment import (
    LocalTmpFileOperator,
    generate_filetree,
)


async def test_generate_filetree_basic(tmp_path: Path) -> None:
    """generate_filetree should list files and directories."""
    op = LocalTmpFileOperator(tmp_path)

    # Create test structure
    await op.write_file("file1.txt", "content")
    await op.write_file("file2.py", "code")
    await op.mkdir("subdir")
    await op.write_file("subdir/nested.txt", "nested")

    tree = await generate_filetree(op, ".")
    assert "file1.txt" in tree
    assert "file2.py" in tree
    assert "subdir/nested.txt" in tree


async def test_generate_filetree_skip_dirs(tmp_path: Path) -> None:
    """generate_filetree should skip specified directories."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("node_modules")
    await op.write_file("node_modules/package.json", "{}")
    await op.write_file("app.js", "code")

    tree = await generate_filetree(op, ".")
    assert "app.js" in tree
    assert "node_modules/ (skipped)" in tree
    assert "package.json" not in tree


async def test_generate_filetree_nonexistent_dir(tmp_path: Path) -> None:
    """generate_filetree should handle nonexistent directory."""
    op = LocalTmpFileOperator(tmp_path)

    tree = await generate_filetree(op, "nonexistent")
    assert "not found" in tree.lower()


async def test_generate_filetree_hidden_dirs(tmp_path: Path) -> None:
    """generate_filetree should skip hidden directories (except .env)."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir(".hidden")
    await op.write_file(".hidden/secret.txt", "secret")
    await op.write_file(".env", "ENV=value")
    await op.write_file("visible.txt", "visible")

    tree = await generate_filetree(op, ".")
    assert "visible.txt" in tree
    assert ".env" in tree  # .env should always be visible
    assert ".hidden" not in tree  # Hidden dir should be skipped


async def test_generate_filetree_gitignore(tmp_path: Path) -> None:
    """generate_filetree should respect .gitignore patterns."""
    op = LocalTmpFileOperator(tmp_path)

    await op.write_file(".gitignore", "*.log\nbuild/")
    await op.write_file("app.js", "code")
    await op.write_file("error.log", "logs")
    await op.mkdir("build")
    await op.write_file("build/output.js", "built")

    tree = await generate_filetree(op, ".")
    assert "app.js" in tree
    assert "error.log" in tree and "(gitignored)" in tree
    assert "build/ (gitignored)" in tree


async def test_generate_filetree_max_depth(tmp_path: Path) -> None:
    """generate_filetree should respect max_depth parameter."""
    op = LocalTmpFileOperator(tmp_path)

    await op.mkdir("level1")
    await op.write_file("level1/file1.txt", "content")
    await op.mkdir("level1/level2")
    await op.write_file("level1/level2/file2.txt", "content")

    # With max_depth=1, should only see root level
    tree_d1 = await generate_filetree(op, ".", max_depth=1)
    assert "file1.txt" not in tree_d1  # level1 is at depth 1, but contents at depth 2

    # With max_depth=2, should see level1 contents
    tree_d2 = await generate_filetree(op, ".", max_depth=2)
    assert "level1/file1.txt" in tree_d2
