"""Cache version + content-hash logic.

Cache keys MUST change when ANY of the following change:
    - Stockfish engine binary version (different eval semantics).
    - rules / models / cache logic (re-grade semantics).
    - the deployed build SHA.

Bundling these into the key prefix invalidates stale entries on restarts after
a semantic change, instead of silently serving stale results. Critical:
``_LOGIC_HASH`` must be derived from the ACTUAL FILE CONTENTS, not a hard-coded
literal — a constant string would silently keep serving stale evaluations
after a semantic fix, masking the bug forever.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from importlib import metadata
from pathlib import Path

from mcp_server.config import get_mcp_settings

# ---------------------------------------------------------------------------
# Files whose contents feed the logic-hash. Adding/removing here changes every
# cache key on the next restart, so be deliberate.
# ---------------------------------------------------------------------------
_LOGIC_FILES: tuple[str, ...] = (
    "mcp_server/cache/__init__.py",
    "mcp_server/cache/version.py",
    "mcp_server/cache/key.py",
    "mcp_server/cache/memory.py",
    "mcp_server/cache/disk.py",
    "mcp_server/cache/multi_tier.py",
    "mcp_server/cache/single_flight.py",
    "mcp_server/models.py",
    "mcp_server/actions.py",
    "mcp_server/server.py",
    "mcp_server/tcp_analyzer.py",
    "mcp_server/tcp_client.py",
    "core/engines/analyzer.py",
    "core/engines/analysis.py",
    "core/engines/grading.py",
    "core/winprob.py",
)


def _git_sha() -> str:
    """Return the deployed build SHA, falling back to the local git HEAD."""
    env_sha = os.environ.get("BUILD_SHA") or os.environ.get("CHESSY_BUILD_SHA")
    if env_sha and env_sha.strip():
        return env_sha.strip()
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            .decode()
            .strip()
        )
        return out or "unknown"
    except Exception:
        return "unknown"


def _package_version() -> str:
    """Resolve the chessy package version.

    Tries installed metadata first (production), then pyproject.toml (dev with
    a uv-managed venv), then env override, then a hard "0.0.0+unknown" so the
    field is never empty. Keeping this in sync with server._package_version()
    ensures the CACHE_VERSION segment matches what the rest of the system
    advertises as `service_version`.
    """
    try:
        return metadata.version("chessy")
    except Exception:
        pass

    env_override = os.environ.get("CHESSY_PACKAGE_VERSION")
    if env_override:
        return env_override

    try:
        pyproject = get_mcp_settings().project_root / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                return m.group(1)
    except Exception:
        pass

    return "0.0.0+unknown"


def _compute_logic_hash() -> str:
    """Hash the contents of all logic-bearing files.

    Any semantic change automatically invalidates cached entries.
    """
    h = hashlib.sha256()
    backend_root = get_mcp_settings().project_root
    for rel_path in _LOGIC_FILES:
        abs_path = backend_root / rel_path
        try:
            h.update(rel_path.encode())
            with open(abs_path, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:12]


_LOGIC_HASH = _compute_logic_hash()


def _resolve_cache_version() -> str:
    """Compose the cache version from build SHA, package version, and logic hash."""
    return f"v14+{_git_sha()}+{_package_version()}+{_LOGIC_HASH}"


CACHE_VERSION: str = _resolve_cache_version()
"""Composite version string embedded in every cache key prefix."""

ENGINE_VERSION_KEY: str = "stockfish"
"""Default engine fingerprint; callers override via ``_resolve_engine_version()``."""


def _resolve_engine_version(engine_version: str | None = None) -> str:
    """Fingerprint the engine for cache-key invalidation.

    A change in Stockfish binary version can change eval semantics — old cached
    entries must NOT be served by a new binary.
    """
    return (engine_version or ENGINE_VERSION_KEY).strip().lower().replace(" ", "_")
