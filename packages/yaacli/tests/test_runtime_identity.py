from __future__ import annotations

import hashlib
from pathlib import Path

from yaacli.durable.application import build_runtime_descriptor
from yaacli.runtime_identity import (
    _CRITICAL_DISTRIBUTIONS,
    _FIRST_PARTY_MODULES,
    _hash_tree,
    _runtime_files,
    runtime_executable_version,
)


def test_runtime_asset_hash_includes_package_data_and_native_bytes(tmp_path: Path) -> None:
    (tmp_path / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "policy.json").write_text('{"enabled":true}\n', encoding="utf-8")
    (tmp_path / "native.so").write_bytes(b"native-v1")
    (tmp_path / "module.pyc").write_bytes(b"ignored")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "module.py").write_text("ignored\n", encoding="utf-8")

    files = {path.relative_to(tmp_path).as_posix() for path in _runtime_files(tmp_path)}
    assert files == {"module.py", "native.so", "policy.json"}

    first = hashlib.sha256()
    _hash_tree(first, label="runtime", root=tmp_path)
    (tmp_path / "policy.json").write_text('{"enabled":false}\n', encoding="utf-8")
    second = hashlib.sha256()
    _hash_tree(second, label="runtime", root=tmp_path)
    assert first.hexdigest() != second.hexdigest()


def test_runtime_identity_declares_complete_first_party_and_critical_dependency_surface() -> None:
    assert {distribution for distribution, _module in _FIRST_PARTY_MODULES} == {
        "yaacli",
        "ya-agent-sdk",
        "ya-agent-environment",
        "ya-agent-stream-protocol",
        "ya-oauth",
        "ya-oauth-provider",
        "ya-ripgrep-core",
    }
    assert {"pydantic-ai", "pydantic", "pydantic-graph"}.issubset(_CRITICAL_DISTRIBUTIONS)
    assert "sqlalchemy" not in _CRITICAL_DISTRIBUTIONS
    executable_version = runtime_executable_version()
    assert executable_version.startswith("yaacli-runtime-v2-sha256:")
    assert len(executable_version.rsplit(":", 1)[1]) == 64


def test_runtime_descriptor_uses_exact_executable_identity() -> None:
    descriptor = build_runtime_descriptor(
        agent_spec={"name": "main", "model": "test"},
        executable_version="test-build",
    )

    assert descriptor.executable_version == "test-build"
    assert descriptor.behavior_payload()["executable_version"] == "test-build"
