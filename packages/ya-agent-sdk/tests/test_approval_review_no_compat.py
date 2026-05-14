from __future__ import annotations

import subprocess
from pathlib import Path

FORBIDDEN_REVIEW_TOKENS = (
    "shell_review",
    "ShellReview",
    "unattended_shell_review",
    "UNATTENDED_SHELL_REVIEW",
    "shell review",
    "Shell review",
)

SCAN_ROOTS = (
    "apps/ya-claw-web/src",
    "packages/ya-agent-sdk",
    "packages/ya-claw",
    "packages/yaacli",
    "skills/agent-builder",
    "skills/ya-claw-deploy",
)

SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "target",
}


def test_no_shell_specific_review_surface() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    this_file = Path(__file__).resolve().relative_to(repository_root).as_posix()
    matches: list[str] = []
    tracked_files = subprocess.check_output(
        ["git", "ls-files", *SCAN_ROOTS], cwd=repository_root, text=True
    ).splitlines()
    for relative_path in tracked_files:
        if relative_path == this_file:
            continue
        path = repository_root / relative_path
        if not path.exists() or any(part in SKIP_PARTS for part in path.parts):
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(token in line for token in FORBIDDEN_REVIEW_TOKENS):
                matches.append(f"{relative_path}:{line_number}:{line}")
    assert matches == []
