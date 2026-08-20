#!/usr/bin/env python3
"""Build release skill zip archives from canonical skill sources."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST_DIR = ROOT / "dist"

SKILL_PACKAGES = [
    (
        ROOT / "skills/agent-builder",
        "SKILL.zip",
        Path("ya-agent-sdk"),
        ((ROOT / "examples", Path("examples")),),
    ),
    (
        ROOT / "skills/ya-claw-deploy",
        "YA_CLAW_DEPLOY_SKILL.zip",
        Path("ya-claw-deploy"),
        (),
    ),
]

REQUIRED_ARCHIVE_MEMBERS = {
    "SKILL.zip": (Path("ya-agent-sdk/examples/capability_plugin/README.md"),),
}
IGNORED_TREE_NAMES = {".venv", "__pycache__"}


def _write_tree(zf: zipfile.ZipFile, source_dir: Path, archive_root: Path) -> None:
    for path in sorted(source_dir.rglob("*")):
        relative_path = path.relative_to(source_dir)
        if any(part in IGNORED_TREE_NAMES for part in relative_path.parts) or path.suffix == ".pyc":
            continue
        if path.is_file():
            zf.write(path, archive_root / relative_path)


def write_zip(
    source_dir: Path,
    output_path: Path,
    archive_root: Path,
    extra_trees: tuple[tuple[Path, Path], ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        _write_tree(zf, source_dir, archive_root)
        for extra_source_dir, extra_archive_root in extra_trees:
            _write_tree(zf, extra_source_dir, archive_root / extra_archive_root)


def validate_zip(output_path: Path, output_name: str) -> None:
    with zipfile.ZipFile(output_path) as zf:
        archive_members = set(zf.namelist())
    missing = [
        path.as_posix()
        for path in REQUIRED_ARCHIVE_MEMBERS.get(output_name, ())
        if path.as_posix() not in archive_members
    ]
    if missing:
        raise ValueError(f"{output_name} is missing required members: {missing!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Build archives in a temporary directory for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="ya-mono-skill-zips-") as tmp_dir:
            output_dir = Path(tmp_dir)
            for source_dir, output_name, archive_root, extra_trees in SKILL_PACKAGES:
                output_path = output_dir / output_name
                write_zip(source_dir, output_path, archive_root, extra_trees)
                validate_zip(output_path, output_name)
                print(f"Validated {output_name} from {source_dir.relative_to(ROOT)}")
        return

    for source_dir, output_name, archive_root, extra_trees in SKILL_PACKAGES:
        output_path = DIST_DIR / output_name
        write_zip(source_dir, output_path, archive_root, extra_trees)
        validate_zip(output_path, output_name)
        print(f"Built {output_path.relative_to(ROOT)} from {source_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
