"""Build identity, engine configuration, and Stockfish path resolution.

Extracted from ``mcp_server.server``. Provides a single
:class:`BuildIdentity` dataclass that snapshots service version, git SHA,
and the engine configuration that produced an evaluation — the three
fields that audit regressions need to correlate a cp number to the
binary that produced it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_STOCKFISH_PATH",
    "BuildIdentity",
    "build_identity",
    "engine_config",
    "git_sha",
    "package_version",
    "stockfish_path",
]

DEFAULT_STOCKFISH_PATH = "/usr/games/stockfish"


def _package_version() -> str:
    """Resolve the chessy package version.

    Tries, in order:
      1. Installed package metadata (production: chessy is installed).
      2. pyproject.toml in the parent of this file (dev: uv-managed venv).
      3. CHESSY_PACKAGE_VERSION env override.
      4. Hard-coded fallback "0.0.0+unknown" so the field never disappears.
    """
    try:
        return metadata.version("chessy")
    except metadata.PackageNotFoundError:
        pass

    env_override = os.environ.get("CHESSY_PACKAGE_VERSION")
    if env_override:  # pragma: no cover — env read kept for tests that bypass get_build_metadata
        return env_override

    try:
        import re

        pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                return m.group(1)
    except Exception:
        pass

    return "0.0.0+unknown"


package_version = _package_version


def _build_sha() -> str:
    """Best-effort git HEAD sha; falls back to env override or 'unknown'."""
    env_sha = os.environ.get("BUILD_SHA") or os.environ.get("CHESSY_BUILD_SHA")
    if env_sha:
        return env_sha
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
                timeout=2,
            )
            .decode()
            .strip()
        )
        return out or "unknown"
    except Exception:
        return "unknown"


git_sha = _build_sha


def _engine_config(pool: Any) -> dict[str, Any]:
    """Snapshot the engine configuration that produced the current response.

    Lets a debugger distinguish "cp=-3 was returned because Stockfish 18 at
    depth 14 Hash=256 said so" from "cp=-3 was returned because depth 1
    Stockfish 17 at Hash=16 said so" — without these, observability for
    grading regressions is impossible.
    """
    if pool is None:
        return {}
    config: dict[str, Any] = {}
    for attr in ("threads", "hash_mb", "depth"):
        val = getattr(pool, attr, None)
        if val is not None:
            config[attr] = val
    name = getattr(pool, "engine_version", None) or getattr(pool, "name", None)
    if name:
        config["engine_name"] = name
    return config


engine_config = _engine_config


def _stockfish_path() -> str:
    which_sf = shutil.which("stockfish")
    for candidate in (
        os.environ.get("STOCKFISH_PATH"),
        DEFAULT_STOCKFISH_PATH,
        which_sf,
        "/opt/homebrew/bin/stockfish",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError(
        "No Stockfish binary found. Set STOCKFISH_PATH to the correct binary location."
    )


stockfish_path = _stockfish_path


@dataclass(frozen=True)
class BuildIdentity:
    """The three audit-identity fields: service version, build SHA, engine config."""

    service_version: str = field(default_factory=_package_version)
    build_sha: str = field(default_factory=_build_sha)
    engine_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_pool(cls, pool: Any) -> BuildIdentity:
        """Build a :class:`BuildIdentity` snapshot for the given analyzer pool."""
        return cls(
            service_version=_package_version(),
            build_sha=_build_sha(),
            engine_config=_engine_config(pool),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "service_version": self.service_version,
            "build_sha": self.build_sha,
            "engine_config": dict(self.engine_config),
        }


def _engine_version_str(pool: Any) -> str:
    """Pool-agnostic engine version fingerprint for cache + observability."""
    return getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))


def _build_identity(pool: Any) -> dict[str, Any]:
    return BuildIdentity.for_pool(pool).as_dict()


build_identity = _build_identity


# Underscored aliases for backwards-compatible import paths.
_package_version = _package_version
_build_sha = _build_sha
_engine_config = _engine_config
_build_identity = _build_identity
_stockfish_path = _stockfish_path
