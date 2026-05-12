"""Utility functions for environment module.

This module provides helper functions for file tree generation and related utilities.
"""

import pathspec

from ya_agent_environment.file_operator import (
    DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    DEFAULT_INSTRUCTIONS_SKIP_DIRS,
    FileOperator,
)


def _should_skip_hidden_item(name: str, is_dir: bool, skip_dirs: frozenset[str]) -> tuple[bool, bool]:
    """Check if an item should be skipped.

    Returns:
        (should_skip, should_mark_skipped)
    """
    # Check explicit skip_dirs first (e.g., node_modules, __pycache__)
    if is_dir and name in skip_dirs:
        return True, True  # Skip but mark

    # Handle hidden items (starting with .)
    if not name.startswith("."):
        return False, False
    if name == ".env":
        return False, False  # Always show .env
    return True, False  # Skip hidden items completely


def _load_gitignore_spec(gitignore_content: str) -> pathspec.PathSpec | None:
    """Load .gitignore patterns from content."""
    try:
        patterns = gitignore_content.splitlines()
        return pathspec.PathSpec.from_lines("gitignore", patterns)
    except Exception:
        return None


async def generate_filetree(  # noqa: C901
    file_op: FileOperator,
    root_path: str = ".",
    *,
    max_depth: int = DEFAULT_INSTRUCTIONS_MAX_DEPTH,
    skip_dirs: frozenset[str] | None = None,
) -> str:
    """Generate a flat file listing using FileOperator interface.

    This function works with any FileOperator implementation.
    Output format is flat paths like: src/main.py, src/cli.py

    Args:
        file_op: FileOperator instance to use for file operations.
        root_path: Root path to generate file listing for.
        max_depth: Maximum depth to traverse.
        skip_dirs: Set of directory names to skip but mark.

    Returns:
        Newline-separated flat file paths.
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_INSTRUCTIONS_SKIP_DIRS

    if not await file_op.exists(root_path) or not await file_op.is_dir(root_path):
        return f"Directory not found: {root_path}"

    # Try to load gitignore
    gitignore_spec: pathspec.PathSpec | None = None
    gitignore_path = f"{root_path}/.gitignore" if root_path != "." else ".gitignore"
    try:
        if await file_op.exists(gitignore_path):
            content = await file_op.read_file(gitignore_path)
            gitignore_spec = _load_gitignore_spec(content)
    except Exception:  # noqa: S110
        pass

    def _is_gitignored(rel_path: str, is_dir: bool) -> bool:
        if gitignore_spec is None:
            return False
        path = rel_path + "/" if is_dir else rel_path
        return gitignore_spec.match_file(path)

    async def _collect_paths(current_path: str, current_depth: int, path_prefix: str = "") -> list[str]:
        """Collect all file paths recursively, returning flat paths."""
        result: list[str] = []
        try:
            # Use list_dir_with_types to avoid N+1 is_dir calls
            entries_with_types = await file_op.list_dir_with_types(current_path)

            # Separate directories and files
            dir_entries = [(name, True) for name, is_dir in entries_with_types if is_dir]
            file_entries = [(name, False) for name, is_dir in entries_with_types if not is_dir]

            # Process directories first
            for name, _ in dir_entries:
                entry_path = f"{current_path}/{name}" if current_path != "." else name
                flat_path = f"{path_prefix}{name}" if path_prefix else name

                should_skip, should_mark = _should_skip_hidden_item(name, True, skip_dirs)
                if should_skip:
                    if should_mark:
                        result.append(f"{flat_path}/ (skipped)")
                    continue

                # Check gitignore
                if _is_gitignored(flat_path, True):
                    result.append(f"{flat_path}/ (gitignored)")
                    continue

                if current_depth < max_depth:
                    result.extend(await _collect_paths(entry_path, current_depth + 1, f"{flat_path}/"))

            # Then files
            for name, _ in file_entries:
                flat_path = f"{path_prefix}{name}" if path_prefix else name

                should_skip, _ = _should_skip_hidden_item(name, False, skip_dirs)
                if should_skip:
                    continue

                suffix = " (gitignored)" if _is_gitignored(flat_path, False) else ""
                result.append(f"{flat_path}{suffix}")

        except Exception:  # noqa: S110
            pass
        return result

    all_paths = await _collect_paths(root_path, 1)
    return "\n".join(all_paths)
