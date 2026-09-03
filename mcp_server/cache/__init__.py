"""Multi-tier cache for evaluation results.

Modules:
    - ``key`` — canonical cache-key derivation (eval / top / classify).
    - ``memory`` — ``AsyncLRUCache[T]`` generic L1 in-memory tier.
    - ``disk`` — ``SQLiteDiskCache`` L2 persistent WAL tier.
    - ``multi_tier`` — ``MultiTierCache`` composing L1+L2 with typed adapters.
    - ``single_flight`` — ``SingleFlight[T]`` request coalescer.
    - ``version`` — content-hash version fingerprint + engine version resolution.

The public API re-exports every name the rest of the codebase imports. New code
should construct ``MultiTierCache`` and reach for ``eval_cache_key(...)`` etc.
directly.
"""

from mcp_server.cache.disk import (
    LEGACY_CACHE_DB_PATH,
    _LEGACY_CACHE_DB_PATH,
    SQLiteDiskCache,
    _migrate_legacy_cache,
    migrate_legacy_cache,
)
from mcp_server.cache.key import (
    canonical_fen,
    classify_cache_key,
    eval_cache_key,
    history_fingerprint,
    top_moves_cache_key,
)
from mcp_server.cache.memory import AsyncLRUCache
from mcp_server.cache.multi_tier import MultiTierCache
from mcp_server.cache.single_flight import SingleFlight
from mcp_server.cache.version import (
    CACHE_VERSION,
    ENGINE_VERSION_KEY,
    _LOGIC_FILES,
    _compute_logic_hash,
    _git_sha,
    _package_version,
    _resolve_cache_version,
    _resolve_engine_version,
)

# Cached logic-hash constant — public so audit tests can patch/observe it
# directly via ``mcp_server.cache._LOGIC_HASH``.
_LOGIC_HASH = _compute_logic_hash()


__all__ = [
    "AsyncLRUCache",
    "CACHE_VERSION",
    "ENGINE_VERSION_KEY",
    "LEGACY_CACHE_DB_PATH",
    "MultiTierCache",
    "SQLiteDiskCache",
    "SingleFlight",
    "_LEGACY_CACHE_DB_PATH",
    "_LOGIC_FILES",
    "_LOGIC_HASH",
    "_compute_logic_hash",
    "_git_sha",
    "_migrate_legacy_cache",
    "_package_version",
    "_resolve_cache_version",
    "_resolve_engine_version",
    "canonical_fen",
    "classify_cache_key",
    "eval_cache_key",
    "history_fingerprint",
    "migrate_legacy_cache",
    "top_moves_cache_key",
]
