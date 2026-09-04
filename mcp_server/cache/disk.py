"""Persistent on-disk SQLite cache (L2).

Stores eval results in a single SQLite WAL DB. The DB is best-effort — every
operation is wrapped in try/except so a failing disk backend never crashes the
request path. WAL mode + synchronous=NORMAL trades durability for speed (eval
results are reproducible; losing them just means a cache miss on next boot).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sqlite3
import time

from mcp_server.config import get_mcp_settings

__all__ = [
    "LEGACY_CACHE_DB_PATH",
    "_LEGACY_CACHE_DB_PATH",
    "SQLiteDiskCache",
    "_migrate_legacy_cache",
    "migrate_legacy_cache",
]

# Path where the legacy cache used to live. Read late-bound by ``_migrate_legacy_cache``
# via the package-level symbol so audit tests can ``monkeypatch.setattr(cache_module, ...)``
# without touching this submodule.
LEGACY_CACHE_DB_PATH = "/tmp/chess_mcp_eval_cache.sqlite3"
_LEGACY_CACHE_DB_PATH = LEGACY_CACHE_DB_PATH  # back-compat alias for old import paths


def _read_legacy_path() -> str:
    """Read the legacy cache path late-bound via the package symbol.

    Tests patch ``mcp_server.cache._LEGACY_CACHE_DB_PATH``; reading through the
    package ensures the patched value is honored here.
    """
    import mcp_server.cache as _pkg

    return _pkg._LEGACY_CACHE_DB_PATH


def _migrate_legacy_cache(target_path: str) -> None:
    """One-shot legacy cache copy used when the L2 path moves.

    Idempotent — returns early if target exists, or legacy doesn't.
    """
    legacy = _read_legacy_path()
    if target_path == legacy:
        return
    if os.path.exists(target_path):
        return
    if not os.path.exists(legacy):
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        shutil.copy2(legacy, target_path)
        for suffix in ("-wal", "-shm"):
            legacy_side = legacy + suffix
            if os.path.exists(legacy_side):
                shutil.copy2(legacy_side, target_path + suffix)
    except Exception:
        pass  # best-effort; a missing migration just means a colder first run


# Public-name alias — pre-split callers (and tests) reach for the unprefixed form.
migrate_legacy_cache = _migrate_legacy_cache


class SQLiteDiskCache:
    """Persistent on-disk cache using SQLite WAL mode for fast O(1) reads/writes."""

    def __init__(self, db_path: str | None = None) -> None:
        default_path = get_mcp_settings().cache_db
        self.db_path = db_path or default_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        _migrate_legacy_cache(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                try:
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                except Exception:
                    pass
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache "
                    "(key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_eval_cache_created ON eval_cache(created_at);"
                )
                conn.commit()
        except Exception:
            pass

    def _get_sync(self, key: str) -> str | None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                # Per-connection PRAGMA: re-assert NORMAL on every connection so
                # older caches (synchronous=FULL) pick up the faster write path
                # without a manual VACUUM/rebuild.
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache "
                    "(key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                cur = conn.execute("SELECT val FROM eval_cache WHERE key = ?", (key,))
                row = cur.fetchone()
                return str(row[0]) if row else None
        except Exception:
            return None

    def _set_sync(self, key: str, val: str, max_entries: int = 100_000) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache "
                    "(key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO eval_cache (key, val, created_at) VALUES (?, ?, ?)",
                    (key, val, time.time()),
                )
                # Bounded pruning: keep top ``max_entries`` rows by recency.
                if (int(time.time()) % 100) == 0:
                    conn.execute(
                        "DELETE FROM eval_cache WHERE rowid IN "
                        "(SELECT rowid FROM eval_cache ORDER BY created_at ASC "
                        "LIMIT max(0, (SELECT count(*) FROM eval_cache) - ?))",
                        (max_entries,),
                    )
                conn.commit()
        except Exception:
            pass

    def _clear_sync(self) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                conn.execute("PRAGMA synchronous = NORMAL;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache "
                    "(key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                conn.execute("DELETE FROM eval_cache")
                conn.commit()
        except Exception:
            pass

    async def get(self, key: str) -> str | None:
        return await asyncio.to_thread(self._get_sync, key)

    async def set(self, key: str, val: str) -> None:
        await asyncio.to_thread(self._set_sync, key, val)

    async def clear(self) -> None:
        await asyncio.to_thread(self._clear_sync)
