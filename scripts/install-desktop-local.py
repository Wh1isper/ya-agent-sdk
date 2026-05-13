#!/usr/bin/env python3
"""Install the locally built YA Desktop app on the current machine."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    bundle_dir = repo_root / "apps" / "ya-desktop" / "src-tauri" / "target" / "release" / "bundle"
    system = platform.system()

    if system == "Darwin":
        app_dir = find_first(bundle_dir, "*.app", directories=True)
        if app_dir is None:
            msg = f"No .app bundle found under {bundle_dir}"
            raise SystemExit(msg)
        target = Path.home() / "Applications" / app_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(app_dir, target, symlinks=True)
        print(f"Installed {target}")
        return 0

    if system == "Linux":
        appimage = find_first(bundle_dir, "*.AppImage")
        if appimage is not None:
            target = Path.home() / ".local" / "bin" / "ya-desktop.AppImage"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(appimage, target)
            target.chmod(target.stat().st_mode | 0o755)
            print(f"Installed {target}")
            return 0

        deb = find_first(bundle_dir, "*.deb")
        if deb is not None:
            print(f"Built deb package: {deb}")
            print(f"Install with: sudo dpkg -i {deb}")
            return 0

        msg = f"No AppImage or deb bundle found under {bundle_dir}"
        raise SystemExit(msg)

    if system == "Windows":
        installer = find_first(bundle_dir, "*.msi") or find_first(bundle_dir, "*.exe")
        if installer is None:
            msg = f"No Windows installer found under {bundle_dir}"
            raise SystemExit(msg)
        os.startfile(installer)  # type: ignore[attr-defined]  # noqa: S606
        print(f"Started installer {installer}")
        return 0

    msg = f"Unsupported platform: {system}"
    raise SystemExit(msg)


def find_first(root: Path, pattern: str, *, directories: bool = False) -> Path | None:
    matches = sorted(root.rglob(pattern))
    for match in matches:
        if directories and match.is_dir():
            return match
        if not directories and match.is_file():
            return match
    return None


if __name__ == "__main__":
    raise SystemExit(main())
