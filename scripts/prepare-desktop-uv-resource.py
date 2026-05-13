#!/usr/bin/env python3
"""Copy the active uv binary into YA Desktop bundle resources."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    resource_dir = repo_root / "apps" / "ya-desktop" / "src-tauri" / "resources" / "uv"
    resource_dir.mkdir(parents=True, exist_ok=True)

    uv_path = shutil.which("uv")
    if uv_path is None:
        msg = "uv executable was not found on PATH"
        raise SystemExit(msg)

    target_name = "uv.exe" if os.name == "nt" else "uv"
    target = resource_dir / target_name
    shutil.copy2(uv_path, target)

    if os.name != "nt":
        mode = target.stat().st_mode
        target.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    print(f"Copied {uv_path} to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
