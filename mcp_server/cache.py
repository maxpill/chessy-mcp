from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar, cast

import chess

from mcp_server.models import MCPEval, MCPMoveAnalysis

T = TypeVar("T")


def _git_sha() -> str:
    """Best-effort git HEAD sha for the working tree; 'unknown' if not a git repo."""
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
        from importlib import metadata

        return metadata.version("chessy")
    except Exception:
        pass

    import os
    import re

    env_override = os.environ.get("CHESSY_PACKAGE_VERSION")
    if env_override:  # pragma: no cover — env read kept for tests that bypass get_build_metadata
        return env_override

    try:
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                return m.group(1)
    except Exception:
        pass

    return "0.0.0+unknown"


# Cache keys must change whenever ANY of the following change:
#   - Stockfish engine binary version (different eval semantics)
#   - rules.py / models.py / cache.py logic (re-grade semantics)
#   - the deployed build SHA
# Bundling these into the key prefix invalidates stale entries on restarts
# after a semantic change, instead of silently serving stale results.
#
# CRITICAL: _LOGIC_HASH must be derived from the ACTUAL FILE CONTENTS, not a
# hard-coded literal. A constant string would silently keep serving stale
# evaluations after a semantic fix, masking the bug forever.
_LOGIC_FILES = (
    "mcp_server/cache.py",
    "mcp_server/rules.py",
    "mcp_server/models.py",
    "mcp_server/server.py",
)


def _compute_logic_hash() -> str:
    """Hash the contents of all logic-bearing files so any semantic change
    automatically invalidates cached entries."""
    h = hashlib.sha256()
    backend_root = Path(__file__).resolve().parent.parent
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
            # File missing or unreadable — fall back to a sentinel so we don't
            # accidentally hash an empty stream.
            h.update(b"<missing>")
    return h.hexdigest()[:12]


_LOGIC_HASH = _compute_logic_hash()


def _resolve_cache_version() -> str:
    """Compose the cache version from build SHA, package version, and logic hash."""
    return f"v13+{_git_sha()}+{_package_version()}+{_LOGIC_HASH}"


CACHE_VERSION = _resolve_cache_version()
ENGINE_VERSION_KEY = "stockfish"  # overridden by callers via _resolve_engine_version() when known


def _resolve_engine_version(engine_version: str | None = None) -> str:
    """Fingerprint the engine for cache-key invalidation.

    A change in Stockfish binary version can change eval semantics — old cached
    entries must NOT be served by a new binary.
    """
    return (engine_version or ENGINE_VERSION_KEY).strip().lower().replace(" ", "_")


def _board_transposition_key(b: chess.Board) -> tuple[Any, ...]:
    return (
        b.pawns,
        b.knights,
        b.bishops,
        b.rooks,
        b.queens,
        b.kings,
        b.occupied_co[chess.WHITE],
        b.occupied_co[chess.BLACK],
        b.turn,
        b.clean_castling_rights(),
        b.ep_square if b.has_legal_en_passant() else None,
    )


def history_fingerprint(board: chess.Board) -> str:
    """Fingerprint of reversible history affecting repetition / draw claims."""
    if not board.move_stack:
        return ""
    keys: list[str] = [str(_board_transposition_key(board))]
    switchyard: list[chess.Move] = []
    try:
        while board.move_stack:
            move = board.pop()
            switchyard.append(move)
            if board.is_irreversible(move):
                break
            keys.append(str(_board_transposition_key(board)))
    finally:
        while switchyard:
            board.push(switchyard.pop())
    import hashlib

    h = hashlib.sha256(";".join(keys).encode("utf-8")).hexdigest()[:12]
    return f":h={h}"


def canonical_fen(board: chess.Board) -> str:
    """Return full 6-field FEN position key."""
    return board.fen()


def eval_cache_key(board: chess.Board, depth: int, engine_version: str | None = None) -> str:
    """Generate canonical cache key for position evaluation."""
    fp = history_fingerprint(board)
    ev = _resolve_engine_version(engine_version)
    return f"mcp:{CACHE_VERSION}:eng={ev}:eval:{board.fen()}{fp}:{depth}"


def top_moves_cache_key(board: chess.Board, depth: int, n: int = 1, engine_version: str | None = None) -> str:
    """Generate canonical cache key for MultiPV top moves."""
    fp = history_fingerprint(board)
    n_part = f":n={n}" if n is not None else ""
    ev = _resolve_engine_version(engine_version)
    return f"mcp:{CACHE_VERSION}:eng={ev}:top:{board.fen()}{fp}:{depth}{n_part}"


def classify_cache_key(
    board: chess.Board,
    move_uci: str,
    depth: int,
    action_type: str = "play_move",
    engine_version: str | None = None,
) -> str:
    """Generate canonical cache key for move classification."""
    fp = history_fingerprint(board)
    act_part = f":{action_type}" if action_type and action_type != "play_move" else ""
    ev = _resolve_engine_version(engine_version)
    return f"mcp:{CACHE_VERSION}:eng={ev}:classify:{board.fen()}{fp}:{move_uci}:{depth}{act_part}"


class AsyncLRUCache[T]:
    """Thread-safe / async in-memory LRU cache."""

    def __init__(self, maxsize: int = 50_000) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict[str, T] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    async def set(self, key: str, value: T) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()

    async def size(self) -> int:
        async with self._lock:
            return len(self._cache)


class SQLiteDiskCache:
    """Persistent on-disk cache using SQLite WAL mode for fast O(1) reads/writes."""

    def __init__(self, db_path: str | None = None) -> None:
        from .config import get_mcp_settings

        default_path = get_mcp_settings().cache_db
        self.db_path = db_path or default_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
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
                    "CREATE TABLE IF NOT EXISTS eval_cache (key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_cache_created ON eval_cache(created_at);")
                conn.commit()
        except Exception:
            pass

    def _get_sync(self, key: str) -> str | None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache (key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
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
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache (key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
                )
                conn.execute(
                    "INSERT OR REPLACE INTO eval_cache (key, val, created_at) VALUES (?, ?, ?)",
                    (key, val, time.time()),
                )
                # Bounded pruning: keep top max_entries if database grows beyond limit
                if (int(time.time()) % 100) == 0:
                    conn.execute(
                        "DELETE FROM eval_cache WHERE rowid IN (SELECT rowid FROM eval_cache ORDER BY created_at ASC LIMIT max(0, (SELECT count(*) FROM eval_cache) - ?))",
                        (max_entries,),
                    )
                conn.commit()
        except Exception:
            pass

    def _clear_sync(self) -> None:
        try:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA busy_timeout = 15000;")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS eval_cache (key TEXT PRIMARY KEY, val TEXT NOT NULL, created_at REAL NOT NULL);"
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


class MultiTierCache:
    """Two-tier cache: Ultra-fast L1 In-Memory LRU + Persistent L2 SQLite WAL Disk Cache."""

    def __init__(self, l1_size: int = 50_000, db_path: str | None = None) -> None:
        self._l1 = AsyncLRUCache[Any](maxsize=l1_size)
        self._l2 = SQLiteDiskCache(db_path)

    async def get_eval(self, key: str) -> MCPEval | None:
        v = await self._l1.get(key)
        if isinstance(v, MCPEval):
            return v
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                val = MCPEval.model_validate_json(raw)
                await self._l1.set(key, val)
                return val
            except Exception:
                pass
        return None

    async def set_eval(self, key: str, val: MCPEval) -> None:
        await self._l1.set(key, val)
        await self._l2.set(key, val.model_dump_json())

    async def get_top_moves(self, key: str) -> list[MCPEval] | None:
        v = await self._l1.get(key)
        if isinstance(v, list):
            return cast(list[MCPEval], v)
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                data = json.loads(raw)
                vals = [MCPEval.model_validate(x) for x in data]
                await self._l1.set(key, vals)
                return vals
            except Exception:
                pass
        return None

    async def set_top_moves(self, key: str, vals: list[MCPEval]) -> None:
        await self._l1.set(key, vals)
        raw = json.dumps([x.model_dump() for x in vals])
        await self._l2.set(key, raw)

    async def get_classify(self, key: str) -> MCPMoveAnalysis | None:
        v = await self._l1.get(key)
        if isinstance(v, MCPMoveAnalysis):
            return v
        raw = await self._l2.get(key)
        if raw is not None:
            try:
                val = MCPMoveAnalysis.model_validate_json(raw)
                await self._l1.set(key, val)
                return val
            except Exception:
                pass
        return None

    async def set_classify(self, key: str, val: MCPMoveAnalysis) -> None:
        await self._l1.set(key, val)
        await self._l2.set(key, val.model_dump_json())

    async def clear(self) -> None:
        await self._l1.clear()
        await self._l2.clear()


class SingleFlight[T]:
    """Request coalescer (SingleFlight pattern).

    Ensures that for a given key, only one in-flight asynchronous operation is
    executed at a time. All concurrent callers for the same key await the same
    task result.

    Cancellation safety: every waiter awaits the shared future through
    `asyncio.shield(...)`. Without shielding, the cancellation of ANY one
    waiter (e.g. its HTTP request being aborted by a client) would propagate
    to the shared Future and cancel the work for every other waiter in flight
    — a single disconnected client could take down an expensive Stockfish
    search for everyone else. Shielding isolates per-waiter cancellation.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        async with self._lock:
            if key in self._in_flight:
                future = self._in_flight[key]
                # Release lock while waiting for future
                do_wait = True
            else:
                future = asyncio.get_running_loop().create_future()
                self._in_flight[key] = future
                do_wait = False

        if do_wait:
            # Shield: protect the shared future from THIS waiter's cancellation
            # while still letting this waiter raise CancelledError to its own caller.
            return await asyncio.shield(future)

        try:
            result = await fn()
            if not future.done():
                future.set_result(result)
            return result
        except asyncio.CancelledError:
            # Only cancel the shared future when the EXECUTOR (not a waiter)
            # was cancelled — waiters are shielded above.
            if not future.done():
                future.cancel()
            raise
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)
