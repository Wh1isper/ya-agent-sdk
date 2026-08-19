"""Executable identity for exact YAACLI runtime reconstruction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Protocol

_RUNTIME_ABI = "yaacli-runtime-v2"
_FIRST_PARTY_MODULES = (
    ("yaacli", "yaacli"),
    ("ya-agent-sdk", "ya_agent_sdk"),
    ("ya-agent-environment", "ya_agent_environment"),
    ("ya-agent-stream-protocol", "ya_agent_stream_protocol"),
    ("ya-oauth", "ya_oauth"),
    ("ya-oauth-provider", "ya_oauth_provider"),
    ("ya-ripgrep-core", "ya_ripgrep_core"),
)
_CRITICAL_DISTRIBUTIONS = (
    "anyio",
    "click",
    "httpx",
    "prompt-toolkit",
    "pydantic",
    "pydantic-ai",
    "pydantic-ai-slim",
    "pydantic-core",
    "pydantic-graph",
    "rich",
)
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})


class _HashWriter(Protocol):
    def update(self, value: bytes, /) -> None: ...


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


@lru_cache(maxsize=1)
def runtime_executable_version() -> str:
    """Return a full build identity over runtime code, assets, and ABI dependencies."""
    digest = hashlib.sha256()
    metadata = {
        "abi": _RUNTIME_ABI,
        "first_party_versions": {
            distribution: _distribution_version(distribution) for distribution, _module in _FIRST_PARTY_MODULES
        },
        "dependency_versions": {
            distribution: _distribution_version(distribution) for distribution in _CRITICAL_DISTRIBUTIONS
        },
        "python": {
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }
    _update_json(digest, metadata)
    for distribution, module_name in _FIRST_PARTY_MODULES:
        sources = _module_sources(module_name)
        if not sources:
            _update_bytes(digest, f"module:{distribution}:missing", b"")
            continue
        for source in sources:
            if source.is_dir():
                _hash_tree(digest, label=distribution, root=source)
            else:
                _hash_file(digest, label=distribution, root=source.parent, path=source)
    return f"{_RUNTIME_ABI}-sha256:{digest.hexdigest()}"


def _module_sources(module_name: str) -> tuple[Path, ...]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return ()
    if spec.submodule_search_locations is not None:
        return tuple(sorted(Path(location).resolve() for location in spec.submodule_search_locations))
    if isinstance(spec.origin, str) and spec.origin not in {"built-in", "frozen"}:
        return (Path(spec.origin).resolve(),)
    return ()


def _hash_tree(digest: _HashWriter, *, label: str, root: Path) -> None:
    for path in _runtime_files(root):
        _hash_file(digest, label=label, root=root, path=path)


def _runtime_files(root: Path) -> Iterable[Path]:
    return (
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in _IGNORED_SUFFIXES
    )


def _hash_file(
    digest: _HashWriter,
    *,
    label: str,
    root: Path,
    path: Path,
) -> None:
    _update_bytes(
        digest,
        f"{label}/{path.relative_to(root).as_posix()}",
        path.read_bytes(),
    )


def _update_json(digest: _HashWriter, value: object) -> None:
    _update_bytes(
        digest,
        "metadata",
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
    )


def _update_bytes(digest: _HashWriter, label: str, value: bytes) -> None:
    label_bytes = label.encode()
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)
