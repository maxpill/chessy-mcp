"""Lichess Opening Explorer HTTP client + cache for ``mcp_server.tools.explore_opening``.

Public surface:
    - ``LichessExplorerClient`` — async httpx wrapper, 4 s timeout, Bearer auth,
      NDJSON streaming for ``/lichess``, retry 1× on 5xx.
    - ``LichessExplorerCache`` — L1 in-memory LRU + L2 SQLite WAL via the existing
      ``mcp_server.cache.disk.SQLiteDiskCache``, 8 min TTL, single-flight coalescing.
    - ``explorer_cache_key`` — canonical key derivation so tests can predict it.
"""

from mcp_server.lichess_explorer.cache import (
    EXPLORER_CACHE_L1_SIZE,
    EXPLORER_CACHE_KEY_PREFIX,
    EXPLORER_CACHE_TTL_S,
    LichessExplorerCache,
    explorer_cache_key,
)
from mcp_server.lichess_explorer.client import (
    LICHESS_DBS,
    LICHESS_DB_LICHESS,
    LICHESS_DB_MASTERS,
    LICHESS_DB_PLAYER,
    LICHESS_EXPLORER_PLAYER_TIMEOUT_S,
    LICHESS_EXPLORER_TIMEOUT_S,
    LICHESS_VARIANTS,
    LichessExplorerAuthError,
    LichessExplorerClient,
    LichessExplorerError,
    LichessExplorerNotFound,
    LichessExplorerRateLimited,
    LichessExplorerResponseError,
    LichessExplorerTimeout,
    LichessExplorerUnreachable,
    LichessExplorerUnavailable,
    LichessVariant,
)

__all__ = [
    "EXPLORER_CACHE_KEY_PREFIX",
    "EXPLORER_CACHE_L1_SIZE",
    "EXPLORER_CACHE_TTL_S",
    "LICHESS_DBS",
    "LICHESS_DB_LICHESS",
    "LICHESS_DB_MASTERS",
    "LICHESS_DB_PLAYER",
    "LICHESS_EXPLORER_PLAYER_TIMEOUT_S",
    "LICHESS_EXPLORER_TIMEOUT_S",
    "LICHESS_VARIANTS",
    "LichessExplorerAuthError",
    "LichessExplorerCache",
    "LichessExplorerClient",
    "LichessExplorerError",
    "LichessExplorerNotFound",
    "LichessExplorerRateLimited",
    "LichessExplorerResponseError",
    "LichessExplorerTimeout",
    "LichessExplorerUnavailable",
    "LichessExplorerUnreachable",
    "LichessVariant",
    "explorer_cache_key",
]
