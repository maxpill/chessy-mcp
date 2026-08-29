from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import re
import subprocess
import time
import zlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import chess

from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Message, Receive, Scope, Send

log = logging.getLogger("chessy_mcp.server")

from core.engines.analyzer import pv_to_san
from core.engines.openings import lookup_opening
from core.engines.pool import AnalyzerPool
from core.engines.types import MoveClass
from mcp_server.cache import (
    CACHE_VERSION as CACHE_VERSION,
)
from mcp_server.cache import (
    MultiTierCache,
    SingleFlight,
    classify_cache_key,
    eval_cache_key,
    top_moves_cache_key,
)
from mcp_server.metrics import metrics
from mcp_server.models import (
    GameAnalysisResult,
    MCPEval,
    MCPMoveAnalysis,
    PlyAnalysisItem,
    TopMovesResult,
    score_played_move,
)
from mcp_server.rules import (
    evaluate_rule_status,
    format_fen_status_errors,
    is_locked_dead_position,
    is_terminal_position,
    validate_mating_possibility,
)
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.urls import lichess_urls


def _format_exception(exc: BaseException) -> str:
    if isinstance(exc, (ExceptionGroup, BaseExceptionGroup)):
        sub_msgs = [_format_exception(e) for e in exc.exceptions]
        return "; ".join(sub_msgs) if sub_msgs else str(exc)
    return str(exc)


def _tool_error(code: str, message: str | BaseException, tool: str, **kwargs: Any) -> ToolError:
    """Create a clean human/agent-readable ToolError payload."""
    raw = _format_exception(message) if isinstance(message, BaseException) else str(message)
    clean_msg = raw.strip()
    clean_msg = re.sub(r"^(?:\[[A-Za-z0-9_]+\]|[A-Za-z0-9_]+:)\s*", "", clean_msg).strip()
    return ToolError(f"[{code.upper()}] {clean_msg}")


# Audit P1: whitelist the docker bridge / loopback network so chessy's own
# agents (coach / trainer / clara) can reach MCP when CHESS_MCP_LOCK_CHATGPT
# is set. The lock is intended to constrain public-facing clients
# (ChatGPT actions, ad-hoc curl, etc.), NOT to block the system's own
# service-to-service traffic which is already protected by docker network
# isolation.
def _is_private_or_loopback_ip(ip: str) -> bool:
    """Return True iff `ip` is loopback or an RFC1918 private address.

    Accepts dotted-quad IPv4 (with or without an IPv6 `[...]` wrapper that
    ASGI sometimes uses). Returns False for IPv6 link-local or anything
    we can't parse — better to err on the side of caution and require
    auth for an unknown address class.
    """
    s = ip.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    # Loopback
    if s == "127.0.0.1" or s == "::1" or s == "localhost":
        return True
    # IPv6 unique-local fc00::/7 — covers docker compose default networks
    if "::" in s and ":" in s.replace("::", "", 1):
        return False  # not a ULA
    if s.startswith("fc") or s.startswith("fd"):
        # Cheap ULA-prefix check (fc00::/7) — exact parsing isn't critical
        # because docker-compose IPv6 defaults are rare; we err on safety.
        return True
    # IPv4
    parts = s.split(".")
    if len(parts) == 4:
        try:
            a, b, c, d = (int(p) for p in parts)
        except ValueError:
            return False
        if a == 10:
            return True
        if a == 172 and 16 <= b <= 31:
            return True
        if a == 192 and b == 168:
            return True
    return False


# Verbosity levels (audit M-05). `compact` strips Lichess URLs/images and
# decision_value/engine_eval duplication from every candidate, dropping
# payload size ~70% for LLM-driven callers that don't need URLs.
VERBOSITY_FULL = "full"
VERBOSITY_COMPACT = "compact"

_VERBOSITY_ALIASES = {
    "compact": "compact",
    "minimal": "compact",
    "min": "compact",
    "full": "full",
    "standard": "full",
    "default": "full",
}


def _resolve_verbosity(value: Any) -> str:
    """Normalize a verbosity string to one of {compact, full}. Unknown
    values fall back to `full` (the legacy behavior)."""
    if value is None:
        return VERBOSITY_FULL
    s = str(value).strip().lower()
    return _VERBOSITY_ALIASES.get(s, VERBOSITY_FULL)


def _compact_mcpeval(mcp_eval: MCPEval) -> MCPEval:
    """Return a copy of an MCPEval with verbose fields stripped (audit M-05).

    Drops:
      - lichess_url, lichess_image
      - decision_value (engine-eval duplicates best_action)
      - engine_eval (raw Stockfish output; candidate is the canonical view)
      - history_completeness / repetition_status (only top-level responses
        need these; on candidates they're noise)
      - input_fen / canonical_fen / fen_was_canonicalized (only meaningful
        on the OUTER response, not per-candidate)
      - history_dependent_status / lichess_url_reproduces_history /
        requires_move_stack / fen_sufficient_for_status
      - post_can_claim_draw / post_can_claim_now / post_claim_reasons /
        post_claim_moves (post_position summary is the canonical view)
    Keeps:
      - status, winner, cp, mate, best_move, pv, depth
      - executable_move (typed contract)
      - best_action_obj, legal_actions (typed contract)
      - post_position (structured summary)
      - candidate_san (human-readable)
    """
    return mcp_eval.model_copy(
        update={
            "lichess_url": None,
            "lichess_image": None,
            "decision_value": None,
            "engine_eval": None,
            "history_dependent_status": False,
            "lichess_url_reproduces_history": True,
            "requires_move_stack": False,
            "fen_sufficient_for_status": True,
            "history_completeness": "complete",
            "repetition_status": "none",
            "input_fen": None,
            "canonical_fen": mcp_eval.canonical_fen,
            "fen_was_canonicalized": False,
            "post_can_claim_draw": False,
            "post_can_claim_now": False,
            "post_claim_reasons": [],
            "post_claim_moves": [],
        }
    )


_FIGURINE_MAP = str.maketrans(
    {
        "♔": "K",
        "♚": "K",
        "♕": "Q",
        "♛": "Q",
        "♖": "R",
        "♜": "R",
        "♗": "B",
        "♝": "B",
        "♘": "N",
        "♞": "N",
        "♙": "",
        "♟": "",
    }
)

TAG_PAIR_REGEX = re.compile(r'\[\s*([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\s*\]', re.DOTALL)


def _unescape_pgn_tag_value(val: str | None) -> str | None:
    if val is None:
        return None
    return val.replace('\\"', '"').replace("\\\\", "\\")


def _mask_comments_and_escapes(text: str) -> str:
    """Mask semicolon comments, % escape lines, and {braced comments} with spaces, preserving string length and linebreaks."""
    chars = list(text)
    n = len(chars)
    i = 0
    in_brace = False
    in_semi = False
    in_quote = False
    escape_next = False
    is_line_start = True

    while i < n:
        ch = chars[i]
        if ch in ("\r", "\n"):
            in_semi = False
            in_quote = False
            escape_next = False
            is_line_start = True
            i += 1
            continue

        if not in_brace and not in_semi:
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == '"':
                in_quote = not in_quote
            elif not in_quote:
                if is_line_start and ch == "%":
                    in_semi = True
                elif ch == ";":
                    in_semi = True
                    chars[i] = " "
                elif ch == "{":
                    in_brace = True
                    chars[i] = " "
        elif in_brace:
            if ch == "}":
                in_brace = False
            chars[i] = " "
        elif in_semi:
            chars[i] = " "

        if ch not in (" ", "\t"):
            is_line_start = False
        i += 1

    return "".join(chars)


def _sanitize_brackets_in_variations_and_comments(text: str) -> str:
    """Mask '[' and ']' characters occurring inside variations (...) or comments so PGN readers don't break."""
    chars = list(text)
    n = len(chars)
    i = 0
    in_brace = False
    in_semi = False
    var_depth = 0
    is_line_start = True

    while i < n:
        ch = chars[i]
        if ch in ("\r", "\n"):
            in_semi = False
            is_line_start = True
            i += 1
            continue

        if is_line_start and ch == "%":
            in_semi = True

        if not in_brace and not in_semi:
            if ch == ";":
                in_semi = True
            elif ch == "{":
                in_brace = True
            elif ch == "(":
                var_depth += 1
            elif ch == ")":
                var_depth = max(0, var_depth - 1)
            elif var_depth > 0 and ch in ("[", "]"):
                chars[i] = " "
        elif in_brace:
            if ch == "}":
                in_brace = False
            elif ch in ("[", "]"):
                chars[i] = " "
        elif in_semi:
            if ch in ("[", "]"):
                chars[i] = " "

        if ch not in (" ", "\t"):
            is_line_start = False
        i += 1

    return "".join(chars)


def _strip_pgn_escape_lines(text: str) -> str:
    """Strip lines starting with '%' in column 1 (or after whitespace) per PGN standard."""
    return re.sub(r"(?m)^[ \t]*%[^\r\n]*(?:\r?\n)?", "", text)


def _package_version() -> str:
    """Read the installed package version, with a graceful fallback for local runs.

    Order of preference:
    1. Installed package metadata (production: chessy is installed).
    2. pyproject.toml in the parent of this file (dev: uv-managed virtualenv,
       no installed dist-info). This makes service_version reflect the actual
       declared version (e.g. "0.1.0") instead of an opaque "unknown".
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
        from pathlib import Path

        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
            if m:
                return m.group(1)
    except Exception:
        pass

    return "0.0.0+unknown"


def _build_sha() -> str:
    """Best-effort git HEAD sha; falls back to env override or 'unknown'."""
    env_sha = os.environ.get("CHESSY_BUILD_SHA")
    if env_sha:
        return env_sha
    try:
        out = (
            subprocess.check_output(
                ["git", "rev-parse", "--short=12", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent.parent),
                timeout=2,
            )
            .decode()
            .strip()
        )
        return out or "unknown"
    except Exception:
        return "unknown"


def _engine_config(pool: AnalyzerPool | TCPAnalyzerPool | None) -> dict[str, Any]:
    """Snapshot the engine configuration that produced the current response.

    Lets a debugger distinguish 'cp=-3 was returned because Stockfish 18 at
    depth 14 Hash=256 said so' from 'cp=-3 was returned because depth 1
    Stockfish 17 at Hash=16 said so' — without these, observability for
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


def _build_identity(pool: AnalyzerPool | TCPAnalyzerPool | None) -> dict[str, Any]:
    """Single helper for service_version / build_sha / engine_config fields."""
    return {
        "service_version": _package_version(),
        "build_sha": _build_sha(),
        "engine_config": _engine_config(pool),
    }


_DEFAULT_STOCKFISH_PATH = "/usr/games/stockfish"


def _stockfish_path() -> str:
    import shutil

    which_sf = shutil.which("stockfish")
    for candidate in (
        os.environ.get("STOCKFISH_PATH"),
        _DEFAULT_STOCKFISH_PATH,
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


@asynccontextmanager
async def _mcp_lifespan(server: MCPServer) -> AsyncIterator[dict[str, Any]]:
    """Initialize the Stockfish pool at startup, tear it down at exit.

    Replaces the lazy-init path with eager startup so the first user
    request doesn't pay the TCP handshake / UCI isready round-trips. The
    pool is shared with every tool via the lifespan_context["pool"] indirection.

    Side jobs at startup:
      * Apply UCI options: ShowWDL (when enabled) + SyzygyPath (when set).
      * Warm-search: one depth=2 eval per worker so UCI isready completes
        and the engine is primed (saves ~120ms on the first real request).
      * Periodic structured pool-stats logging every
        CHESS_MCP_POOL_STATS_INTERVAL_S seconds (queue depth, alive count,
        cache hit rate). Set to 0 to disable.
    """
    from .config import get_mcp_settings

    cfg = get_mcp_settings()
    cpu = os.cpu_count() or 8
    pool_size = cfg.pool_size if cfg.pool_size is not None else min(cpu, 4)
    threads_per_worker = max(1, cfg.threads_per_worker)
    if sf_host := cfg.host:
        pool: AnalyzerPool | TCPAnalyzerPool = await TCPAnalyzerPool.create(
            sf_host,
            cfg.port,
            size=pool_size,
            name="stockfish",
            threads=threads_per_worker,
            hash_mb=cfg.hash_mb,
            show_wdl=cfg.show_wdl,
            syzygy_path=cfg.syzygy_path or None,
        )
        log.info(
            "TCP analyzer pool ready: %d engines @ %s:%d (threads=%d hash=%dMB wdl=%s syzygy=%s ponder=%s)",
            pool_size,
            sf_host,
            cfg.port,
            threads_per_worker,
            cfg.hash_mb,
            cfg.show_wdl,
            cfg.syzygy_path or "(none)",
            cfg.ponder_enabled,
        )
        pool._mcp_ponder_enabled = cfg.ponder_enabled  # type: ignore[attr-defined]
    else:
        pool = await AnalyzerPool.create(
            _stockfish_path(),
            size=pool_size,
            depth=14,
            threads=threads_per_worker,
            hash_mb=cfg.hash_mb,
        )
        log.info(
            "Subprocess analyzer pool ready: %d engines @ %s",
            pool_size,
            _stockfish_path(),
        )

    # Warm-search: one cheap eval per worker to prime UCI isready. Hidden
    # inside the healthcheck start_period (5s); saves ~120ms on the first
    # real user request.
    warmup_board = chess.Board()
    try:
        await asyncio.gather(*[pool.evaluate(warmup_board, depth=2) for _ in range(pool_size)])
        log.info("Pool warm-search complete (%d workers primed)", pool_size)
    except Exception as exc:
        log.warning("Pool warm-search failed (non-fatal): %s", exc)

    stats_task: asyncio.Task[None] | None = None
    stats_interval = float(cfg.pool_stats_interval_s)
    if stats_interval > 0:
        stats_task = asyncio.create_task(
            _pool_stats_logger(pool, stats_interval), name="pool-stats-logger"
        )

    try:
        yield {"pool": pool, "settings": cfg, "pool_size": pool_size}
    finally:
        if stats_task is not None:
            stats_task.cancel()
            try:
                await stats_task
            except (asyncio.CancelledError, Exception):
                pass
        log.info("Shutting down analyzer pool (%d engines)", pool_size)
        await pool.close()


async def _pool_stats_logger(pool: AnalyzerPool | TCPAnalyzerPool, interval_s: float) -> None:
    """Emit a structured pool-stats log line every `interval_s` seconds.

    One log line per interval; log aggregation (loki/journald/etc.) is the
    consumer. Includes queue depth, alive count, and the in-memory
    LocalMetricsTracker snapshot (uptime, total requests, cache hit rate,
    per-tool call counts and p50/p95 latencies).
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        try:
            qsize = pool._pool._q.qsize()  # type: ignore[attr-defined]
            alive = pool._pool._alive_count  # type: ignore[attr-defined]
            target = pool._pool._target_size  # type: ignore[attr-defined]
            from .metrics import metrics

            stats = await metrics.get_stats()
            log.info(
                "pool_stats queue_depth=%d alive=%d target=%d "
                "uptime_s=%s total=%s hit_rate_pct=%s tools=%s",
                qsize,
                alive,
                target,
                stats["uptime_seconds"],
                stats["total_requests"],
                stats["cache_hit_rate_percent"],
                {k: v["calls"] for k, v in stats["tools"].items()},
            )
        except Exception as exc:  # noqa: BLE001 — log and keep looping
            log.warning("pool_stats log iteration failed (continuing): %s", exc)




async def _ponder_warm_cache(
    pool: AnalyzerPool | TCPAnalyzerPool,
    predicted_fen: str,
    depth: int,
) -> None:
    """Background cache warmer — evaluate `predicted_fen` at `depth` and
    store the result in L1 so the next user request on the same FEN hits
    L1 instantly. Pure fire-and-forget; errors are logged and dropped.

    Note: this is NOT full UCI ponder (no go ponder / ponderhit). It's a
    background eval whose result is reusable by any caller.
    """
    try:
        board = chess.Board(predicted_fen)
        if board.is_game_over():
            return
        from .cache import eval_cache_key

        ckey = eval_cache_key(
            board,
            depth,
            engine_version=getattr(pool, "engine_version", None),
        )
        # Avoid duplicate work: skip if L1 already has it.
        if (await _cache.get_eval(ckey)) is not None:
            return
        ev, _hit = await _evaluate_game_position_cached(
            board, depth, pool, requested_depth=depth
        )
        await _cache.set_eval(ckey, ev)
    except Exception as exc:  # noqa: BLE001 — fire-and-forget
        log.debug("ponder pre-eval failed (silent): %s", exc)


def _maybe_ponder_warm(
    pool: AnalyzerPool | TCPAnalyzerPool,
    board: chess.Board,
    best_move_uci: str | None,
    depth: int,
    ponder_enabled: bool,
) -> None:
    """Schedule a background pre-eval if pondering is enabled and we have
    a legal best_move to extrapolate from. No-op otherwise.
    """
    if not ponder_enabled or not best_move_uci:
        return
    try:
        next_board = board.copy(stack=True)
        next_board.push_uci(best_move_uci)
        if next_board.is_game_over():
            return
        asyncio.create_task(
            _ponder_warm_cache(pool, next_board.fen(), depth),
            name="ponder-warm",
        )
    except Exception:
        pass  # best_move wasn't legal in this board — skip silently



mcp = MCPServer(
    "chess-analysis",
    description="Streamable Stockfish chess analysis and move grading MCP server",
    lifespan=_mcp_lifespan,
)


@mcp.custom_route("/health", methods=["GET"])
async def _health(request: Any) -> Any:
    """Liveness/readiness probe — no auth required, no MCP machinery touched.

    Returns 200 with a minimal payload so compose / orchestrator healthchecks
    can verify the service is up without engaging the JSON-RPC stack.
    """
    from starlette.responses import JSONResponse

    return JSONResponse(
        {
            "status": "ok",
            "service": "chessy-mcp",
            "version": _package_version(),
        }
    )


# Multi-Tier Cache (L1 Memory LRU 50k + L2 SQLite WAL Disk Cache) and SingleFlight Coalescer
_cache: MultiTierCache = MultiTierCache(l1_size=50_000)
_single_flight: SingleFlight[Any] = SingleFlight()


# P1 audit fix: bound concurrent evaluate calls so analyze_game at depth 30
# cannot self-inflict PoolBusy by spawning hundreds of simultaneous
# waiters. The semaphore is created lazily on first call so it always
# belongs to the live event loop (pytest-asyncio's per-function event loop
# means a module-level asyncio.Semaphore() would be bound to whichever
# loop runs first and explode on subsequent loops).
_evaluate_semaphore: asyncio.Semaphore | None = None
_evaluate_semaphore_lock = asyncio.Lock()


async def _get_evaluate_semaphore() -> asyncio.Semaphore:
    global _evaluate_semaphore
    async with _evaluate_semaphore_lock:
        if _evaluate_semaphore is None:
            from .config import get_mcp_settings

            cfg = get_mcp_settings()
            _evaluate_semaphore = asyncio.Semaphore(max(1, cfg.max_concurrent_evaluates))
        return _evaluate_semaphore


async def _gather_evaluate_positions_bounded(
    positions: list[chess.Board],
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    *,
    requested_depth: int,
) -> list[tuple[MCPEval, bool]]:
    """Evaluate N positions partitioned across the pool with TT reuse per slice.

    Two-stage fan-out:
      1. Split `positions` into K slices (K = pool size). Each slice is
         dispatched to its own worker via `pool._pool.run(...)`.
      2. Within a slice, evaluations run SEQUENTIALLY with `reuse_tt=True`
         between calls so Stockfish accumulates the TT across consecutive
         positions. Round-robin distribution keeps slices balanced; for
         long PGNs most adjacent plies still share a slice's TT context
         after the first iteration of the cycle.

    The semaphore (`CHESS_MCP_MAX_CONCURRENT_EVALUATES`) bounds the total
    in-flight evaluate work across all MCP tools. Each slice acquires the
    semaphore once at entry.

    Compared to the previous "gather N over the whole pool" approach:
    - Same parallelism across slices (one per worker).
    - TT reuse within a slice (the old code's TT was 100% cold because
      consecutive positions landed on different workers).
    - Fewer pool-acquire round-trips: K acquires instead of N.
    """
    if not positions:
        return []
    sem = await _get_evaluate_semaphore()

    # K slices, one per pool worker. Try to introspect the pool's target
    # size; fall back to 4 if the API differs. Tests pass MockPool objects
    # without the production `_pool` attribute, so be tolerant.
    pool_target: int
    try:
        pool_target = pool._pool._target_size  # type: ignore[attr-defined]
    except AttributeError:
        pool_target = 4
    k = max(1, min(pool_target, len(positions)))

    # Round-robin distribution: position i -> slice i % k. Slice 0 holds
    # positions 0, k, 2k, ...; within a slice the order is whatever
    # round-robin gave us. The TT reuse benefit is concentrated in the
    # second+ iteration of the round-robin cycle (positions k, 2k, ...
    # have k-1 prior positions in the same slice).
    slices: list[list[chess.Board]] = [[] for _ in range(k)]
    for idx, b in enumerate(positions):
        slices[idx % k].append(b)

    async def _run_slice(slice_positions: list[chess.Board]) -> list[tuple[MCPEval, bool]]:
        async with sem:
            # If the pool exposes the production _pool.run(analyzer-fn) API
            # (EnginePool), use it to hold one analyzer for the whole slice.
            # Otherwise (test MockPool objects), call pool.evaluate directly
            # per position — TT reuse is moot without a real analyzer.
            if hasattr(pool, "_pool"):
                async def _on_worker(analyzer: object) -> list[tuple[MCPEval, bool]]:
                    out: list[tuple[MCPEval, bool]] = []
                    for j, b in enumerate(slice_positions):
                        r, hit = await _evaluate_game_position_cached(
                            b,
                            depth,
                            pool,
                            requested_depth=requested_depth,
                            reuse_tt=(j > 0),
                            analyzer=analyzer,
                        )
                        out.append((r, hit))
                    return out

                return await pool._pool.run(_on_worker)  # type: ignore[attr-defined]

            # Fallback for tests / mock pools.
            out: list[tuple[MCPEval, bool]] = []
            for b in slice_positions:
                r, hit = await _evaluate_game_position_cached(
                    b,
                    depth,
                    pool,
                    requested_depth=requested_depth,
                    reuse_tt=False,
                )
                out.append((r, hit))
            return out

    slice_results = await asyncio.gather(*[_run_slice(s) for s in slices if s])
    # Reassemble in original order. slices[si] held positions at indices
    # si, si+k, si+2k, ... in that order.
    out: list[tuple[MCPEval, bool]] = [None] * len(positions)  # type: ignore[list-item]
    for slice_idx in range(k):
        if not slices[slice_idx]:
            continue
        for j, _ in enumerate(slices[slice_idx]):
            out[slice_idx + j * k] = slice_results[slice_idx][j]
    return out


async def _get_analyzer_pool(
    ctx: Context | None = None,
) -> AnalyzerPool | TCPAnalyzerPool:
    """Fetch the live analyzer pool from the FastMCP lifespan context.

    Falls back to the legacy lazy-init path when called outside a request
    (e.g. tests that don't go through the FastMCP runner), so existing test
    setups keep working without rewriting every fixture.
    """
    if ctx is not None:
        ls = ctx.request_context.lifespan_context
        pool = ls.get("pool")
        if pool is not None:
            return pool
    global _analyzer_pool
    async with _pool_lock:
        if _analyzer_pool is None:
            from .config import get_mcp_settings

            mcp_cfg = get_mcp_settings()
            default_pool_size = min(4, max(2, os.cpu_count() or 4))
            pool_size = mcp_cfg.pool_size if mcp_cfg.pool_size is not None else default_pool_size
            hash_mb = mcp_cfg.hash_mb
            sf_host = mcp_cfg.host
            sf_port = mcp_cfg.port
            if sf_host and sf_port:
                _analyzer_pool = await TCPAnalyzerPool.create(
                    sf_host,
                    sf_port,
                    size=pool_size,
                    name="stockfish",
                    threads=mcp_cfg.threads_per_worker,
                    hash_mb=hash_mb,
                )
            else:
                _analyzer_pool = await AnalyzerPool.create(
                    _stockfish_path(),
                    size=pool_size,
                    depth=14,
                    threads=mcp_cfg.threads_per_worker,
                    hash_mb=hash_mb,
                )
        return _analyzer_pool


_analyzer_pool: AnalyzerPool | TCPAnalyzerPool | None = None
_pool_lock = asyncio.Lock()


async def close_analyzer_pool() -> None:
    """Gracefully close all engine workers in the pool.

    Idempotent: safe to call from tests and from lifespan teardown. Also
    drops the cached evaluate semaphore so a fresh one is lazily created on
    the next request — keeps pytest-asyncio's per-function event loop happy.
    """
    global _analyzer_pool, _evaluate_semaphore
    async with _pool_lock:
        if _analyzer_pool is not None:
            await _analyzer_pool.close()
            _analyzer_pool = None
    _evaluate_semaphore = None


SUPPORTED_VARIANTS = {None, "", "standard", "from position"}


def _validate_variant(variant: str | None) -> None:
    if variant is not None and variant.strip().lower() not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"UNSUPPORTED_VARIANT: Variant '{variant.strip()}' is not supported. Chess MCP currently analyzes standard chess only."
        )


def normalize_termination(term: str | None) -> str | None:
    """Normalize PGN termination header string into standard taxonomy."""
    if not term:
        return None
    t = term.strip().lower()
    # 1. Normal — must be checked FIRST, before any general "time" regex that
    # would otherwise eat "Normal time control" as time_forfeit.
    if re.search(r"\bnormal\b", t):
        return "normal"
    # 2. Fifty moves rule
    if re.search(r"\b(?:50|fifty)[-\s]*moves?(?:\s*rule)?\b", t):
        return "fifty_moves"
    # 3. Seventy-five moves rule
    if re.search(r"\b(?:75|seventy[-\s]*five)[-\s]*moves?(?:\s*rule)?\b", t):
        return "seventyfive_moves"
    # 4. Fivefold repetition
    if re.search(r"\b(?:5[-\s]*fold|five[-\s]*fold)(?:\s*repetition)?\b", t):
        return "fivefold_repetition"
    # 5. Threefold repetition
    if re.search(
        r"\b(?:3[-\s]*fold|three[-\s]*fold)(?:\s*repetition)?(?:\s*claim)?\b|\bthreefold\b",
        t,
    ):
        return "threefold_repetition"
    if re.search(r"\brepetition\b", t):
        return "repetition"
    # 6. Checkmate / stalemate
    if re.search(r"\bcheckmate\b|\bmate\b", t):
        return "checkmate"
    if re.search(r"\bstalemate\b|\bstale\b", t):
        return "stalemate"
    # 7. Insufficient material
    if re.search(r"\binsufficient(?:\s*material)?\b", t):
        return "insufficient_material"
    # 8. Resignation
    if re.search(r"\bresign(?:ed|ation|s)?\b", t):
        return "resignation"
    # 9. Time forfeit — STRICT regex. The old /\btime\b/ matched "Normal time
    # control" and other innocuous phrases. Require an explicit forfeit
    # marker alongside time/flag: "forfeit", "out of", "expired", "exhausted",
    # "on time" (a losing condition). Plain "time control" or "increment" alone
    # is NOT a forfeit.
    if re.search(
        r"\btime\s*(?:forfeit|expired|exhausted|loss)\b"
        r"|\bout\s+of\s+time\b"
        r"|\bflag\s*(?:fell|fell|fall|dropped)\b"
        r"|\blost\s+on\s+time\b"
        r"|\bclock\s+(?:flagged|expired)\b",
        t,
    ):
        return "time_forfeit"
    # 10. Unterminated
    if re.search(r"\bunterminated\b|\bunfinished\b", t):
        return "unterminated"
    # 11. Abandoned
    if re.search(r"\babandon(?:ed)?\b", t):
        return "abandoned"
    # 12. Adjudication
    if re.search(r"\badjudicat(?:ed|ion)\b", t):
        return "adjudication"
    # 13. Death / emergency
    if re.search(r"\bdeath\b", t):
        return "death"
    if re.search(r"\bemergency\b", t):
        return "emergency"
    # 14. Rules infraction
    if re.search(
        r"\brules?\s+infraction\b|\b(?:second\s+)?illegal\s+move\b|\binfraction\b|\billegal\b",
        t,
    ):
        return "rules_infraction"
    # 15. Draw agreement — players mutually agreed to draw (vs an auto-terminal
    # draw from a 50-move claim, threefold, etc.). Without this branch, the
    # common PGN phrase "Draw by agreement" surfaces as termination=null and
    # callers have to invent their own classification.
    if re.search(r"\bdraw\s+by\s+agreement\b|\bagreement\b", t):
        return "draw_agreement"
    return None


def _find_movetext_result(text: str) -> str:
    """Extract the canonical result marker from the top level of movetext (outside comments and variations)."""
    # L-04 audit fix: normalize Unicode result markers (½-½, ½–½, etc.) before
    # scanning. PGN specifies ASCII "1/2-1/2" but chess programmes, lichess
    # exports, and tournament software often emit the Unicode fraction. Tolerate
    # the typographic variants so a well-formed-but-Unicode PGN is not rejected
    # as INVALID_PGN.
    text = _normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    movetext = masked[header_end:]
    var_depth = 0
    i = 0
    while i < len(movetext):
        ch = movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in ("1-0", "0-1", "1/2-1/2", "*"):
                if movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = right_idx == len(movetext) or movetext[right_idx] in " \t\r\n;"
                    if left_ok and right_ok:
                        return marker
        i += 1
    return None


# Unicode em-dash / en-dash / hyphen normalization used by both the
# movetext extractor and the canonical-game extractor (audit L-04).
_UNICODE_HYPHEN_MAP = str.maketrans(
    {
        "–": "-",  # en dash
        "—": "-",  # em dash
        "‐": "-",  # Unicode hyphen
        "−": "-",  # Unicode minus
    }
)


def _normalize_unicode_pgn_results(text: str) -> str:
    """Normalize Unicode PGN result markers and hyphens to ASCII (audit L-04).

    Tolerates the common Unicode variants emitted by chess software and
    online databases:
        ½-½, ½–½, ½—½  ->  1/2-1/2
        0–1, 0—1        ->  0-1
        1–0, 1—0        ->  1-0
    and en/em dashes in movetext castling/result lines.
    """
    # Map every ½ (U+00BD) run to "1/2"
    text = text.replace("½", "1/2")
    # Strip a wide zero-width joiner / non-breaking hyphen that some browsers
    # insert in PGN exports; harmless if absent.
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    text = text.translate(_UNICODE_HYPHEN_MAP)
    return text


def _normalize_movetext_figurines(text: str) -> str:
    """Translate Unicode chess figurines only in the movetext section (preserving headers and comments)."""
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    headers_part = text[:header_end]
    movetext = text[header_end:]

    result: list[str] = []
    in_brace = False
    in_semi = False
    i = 0
    while i < len(movetext):
        ch = movetext[i]
        if ch in ("\r", "\n"):
            in_semi = False
            result.append(ch)
        elif in_semi:
            result.append(ch)
        elif ch == ";":
            in_semi = True
            result.append(ch)
        elif ch == "{" and not in_brace:
            in_brace = True
            result.append(ch)
        elif ch == "}" and in_brace:
            in_brace = False
            result.append(ch)
        elif in_brace:
            result.append(ch)
        else:
            result.append(ch.translate(_FIGURINE_MAP))
        i += 1

    return headers_part + "".join(result)


def _validate_movetext_tokens(
    movetext: str, start_board: chess.Board | None = None, strict: bool = False
) -> list[str]:
    """Check that all tokens in the active movetext section are valid chess moves or PGN symbols."""
    # 1. Translate figurines in movetext and split attached NAGs (PGN-07) and attached asterisk
    t = _normalize_movetext_figurines(movetext)
    t = re.sub(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)", r"\1 \2", t)
    t = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", t)
    t = re.sub(r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*", r"\1 *", t)
    t = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", t)
    # 2. Normalize castling and en passant notation
    t = re.sub(r"\b0-0-0\b", "O-O-O", t)
    t = re.sub(r"\bo-o-o\b", "O-O-O", t, flags=re.IGNORECASE)
    t = re.sub(r"\b0-0\b", "O-O", t)
    t = re.sub(r"\bo-o\b", "O-O", t, flags=re.IGNORECASE)
    t = re.sub(
        r"\b([a-h][1-8]|[a-h]x[a-h][1-8])\s+\(?e\.?p\.?\)?(?=\s|$)",
        r"\1",
        t,
        flags=re.IGNORECASE,
    )
    # 3. Remove all PGN header tags
    t = TAG_PAIR_REGEX.sub(" ", t)
    # 4. Remove semicolon comments and % escape lines
    t = re.sub(r";[^\r\n]*", " ", t)
    t = re.sub(r"^[ \t]*%[^\r\n]*", " ", t, flags=re.MULTILINE)
    # 5. Remove all comments {...} (handling nested brackets in comments)
    while "{" in t and "}" in t:
        prev = t
        t = re.sub(r"\{[^{}]*\}", " ", t, flags=re.DOTALL)
        if t == prev:
            break
    # 6. Remove all variations (...)
    while "(" in t and ")" in t:
        prev = t
        t = re.sub(r"\([^()]*\)", " ", t, flags=re.DOTALL)
        if t == prev:
            break

    tokens = t.split()
    b = start_board.copy() if start_board else chess.Board()

    first_move_idx = None
    for i, tok in enumerate(tokens):
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )
        if re.match(r"^\d+[\.\:]*$", tok):
            first_move_idx = i
            break
        try:
            b.parse_san(clean_tok)
            first_move_idx = i
            break
        except Exception:
            pass

    if first_move_idx is None:
        return []

    invalid_tokens: list[str] = []
    b = start_board.copy() if start_board else chess.Board()
    for _idx, tok in enumerate(tokens[first_move_idx:], start=first_move_idx):
        if b.is_game_over(claim_draw=False):
            break
        clean_tok = tok.rstrip(".,:;!?").lstrip(".,:;!?")
        clean_tok = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_tok, flags=re.IGNORECASE).rstrip(
            ".,:;!?"
        )
        nag_m = re.match(r"^\$([0-9]+)$", clean_tok)
        if nag_m:
            nag_val = int(nag_m.group(1))
            if nag_val > 255 and strict:
                invalid_tokens.append(tok)
            continue
        clean_tok = re.sub(r"\$[0-9]+$", "", clean_tok)
        if not clean_tok or clean_tok.lower() in (
            "e.p.",
            "e.p",
            "ep",
            "(e.p.)",
            "(e.p)",
            "(ep)",
        ):
            continue
        if re.match(r"^\d+[\.\:]*$", tok) or clean_tok in (
            "1-0",
            "0-1",
            "1/2-1/2",
            "*",
        ):
            if clean_tok in ("1-0", "0-1", "1/2-1/2", "*"):
                break
            continue
        if re.match(r"^\$[0-9]+$", clean_tok) or clean_tok in (
            "!",
            "?",
            "!!",
            "??",
            "!?",
            "?!",
        ):
            continue
        try:
            m = b.parse_san(clean_tok)
            b.push(m)
        except Exception:
            try:
                m = chess.Move.from_uci(clean_tok)
                if m in b.legal_moves:
                    b.push(m)
                else:
                    invalid_tokens.append(tok)
            except Exception:
                invalid_tokens.append(tok)
    return invalid_tokens


def _truncate_movetext_at_result(text: str) -> str:
    masked = _mask_comments_and_escapes(text)
    header_end = 0
    for m in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break

    headers_part = text[:header_end]
    movetext = text[header_end:]
    masked_movetext = masked[header_end:]

    var_depth = 0
    i = 0
    while i < len(masked_movetext):
        ch = masked_movetext[i]
        if ch == "(":
            var_depth += 1
        elif ch == ")":
            var_depth = max(0, var_depth - 1)
        elif var_depth == 0:
            for marker in ("1-0", "0-1", "1/2-1/2", "*"):
                if masked_movetext[i : i + len(marker)] == marker:
                    left_ok = i == 0 or masked_movetext[i - 1] in " \t\r\n;"
                    right_idx = i + len(marker)
                    right_ok = (
                        right_idx == len(masked_movetext)
                        or masked_movetext[right_idx] in " \t\r\n;"
                    )
                    if left_ok and right_ok:
                        return headers_part + movetext[:right_idx]
        i += 1
    return text


def _parse_pgn_game_candidate(text: str, strict: bool = False) -> chess.pgn.Game | None:
    try:
        masked_for_tags = _mask_comments_and_escapes(text)
        for m in TAG_PAIR_REGEX.finditer(masked_for_tags):
            if m.group(1).lower() == "variant":
                _validate_variant(_unescape_pgn_tag_value(m.group(2)))

        has_real_tags = bool(TAG_PAIR_REGEX.search(masked_for_tags))
        text = _truncate_movetext_at_result(text)
        # Separate attached asterisks
        text = re.sub(
            r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*",
            r"\1 *",
            text,
        )
        text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", text)
        # Sanitize brackets inside variations/comments so read_game does not mistake them for PGN headers
        text_sanitized = _sanitize_brackets_in_variations_and_comments(text)
        text_for_reader = re.sub(
            r"(\[\s*[A-Za-z0-9_]+\s+\"(?:[^\"\\]|\\.)*\"\s*\])\s*(?=\b\d+\s*[\.\:]|[a-h][1-8]|[A-Z])",
            r"\1\n\n",
            text_sanitized,
        )
        game = chess.pgn.read_game(io.StringIO(text_for_reader))
        if game is not None:
            _validate_variant(game.headers.get("Variant"))
            root_b = game.board()
            if not root_b.is_valid() or root_b.status() != chess.STATUS_VALID:
                raise ValueError(
                    f"INVALID_FEN: Initial position '{root_b.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(root_b.status())})."
                )

            moves = list(game.mainline_moves())
            if not moves and not has_real_tags:
                return None

            invalid_tokens = _validate_movetext_tokens(
                text, start_board=game.board(), strict=strict
            )
            if invalid_tokens:
                raise ValueError(
                    f"INVALID_PGN: Invalid PGN syntax or unrecognized token in movetext: {invalid_tokens[0]!r}"
                )

            if game.errors:
                b = game.board()
                reached_game_over = False
                for node in game.mainline():
                    b.push(node.move)
                    if b.is_game_over(claim_draw=False):
                        reached_game_over = True
                        break
                if not reached_game_over:
                    raise ValueError(
                        f"Invalid PGN syntax or illegal move in game: {game.errors[0]}"
                    )

            return game
    except ValueError:
        raise
    except Exception:
        pass
    return None


def _check_multiple_games(cleaned: str) -> None:
    cleaned_escapes = _strip_pgn_escape_lines(cleaned)
    masked_cleaned = _mask_comments_and_escapes(cleaned_escapes)

    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        if m.group(1).lower() == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))

    # 1. Multiple markdown fenced code blocks
    fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```", cleaned_escapes))
    if len(fences) > 1:
        tagged_fences = [
            m for m in fences if (m.group(1) or "").strip().lower() in ("pgn", "chess")
        ]
        if tagged_fences:
            blocks_to_check = [m.group(2).strip() for m in tagged_fences]
        else:
            blocks_to_check = [m.group(2).strip() for m in fences]
        valid_games = 0
        for block in blocks_to_check:
            s = io.StringIO(_strip_pgn_escape_lines(block))
            g = chess.pgn.read_game(s)
            if g and (
                list(g.mainline_moves())
                or (
                    len(g.headers) >= 3
                    and any(k in g.headers for k in ("White", "Black", "FEN", "SetUp"))
                )
            ):
                valid_games += 1
        if valid_games > 1:
            raise ValueError(
                "MULTIPLE_GAMES: Multiple PGN games detected in input. This operation only supports analyzing a single game at a time."
            )

    # 2. Check multiple games via explicit header blocks
    tag_matches = list(TAG_PAIR_REGEX.finditer(masked_cleaned))
    if tag_matches:
        clusters: list[list[re.Match[str]]] = []
        curr = [tag_matches[0]]
        for m in tag_matches[1:]:
            if cleaned_escapes[curr[-1].end() : m.start()].strip() == "":
                curr.append(m)
            else:
                clusters.append(curr)
                curr = [m]
        clusters.append(curr)

        valid_header_games = 0
        for cl in clusters:
            after_cl = cleaned_escapes[cl[-1].end() :]
            first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_cl)
            has_ident = any(
                m.group(1) in ("White", "Black", "FEN", "SetUp", "Event")
                and _unescape_pgn_tag_value(m.group(2)) not in (None, "?")
                for m in cl
            )
            if has_ident and first_mv:
                mv_pos = first_mv.start()
                has_subsequent_cluster_before_mv = any(
                    other_cl is not cl
                    and cl[-1].end() <= other_cl[0].start() < cl[-1].end() + mv_pos
                    for other_cl in clusters
                )
                if not has_subsequent_cluster_before_mv:
                    valid_header_games += 1
        if valid_header_games > 1:
            raise ValueError(
                "MULTIPLE_GAMES: Multiple PGN games detected in input. This operation only supports analyzing a single game at a time."
            )


def _normalize_multiline_tags(text: str) -> str:
    """Normalize multiline tag pairs [Tag \\n "Value"] to [Tag "Value"] on a single line."""

    def _repl(m: re.Match[str]) -> str:
        tag_name = m.group(1).strip()
        tag_val = m.group(2)
        return f'[{tag_name} "{tag_val}"]'

    return TAG_PAIR_REGEX.sub(_repl, text)


def _has_completed_game_before(text: str, pos: int) -> bool:
    prefix = text[:pos]
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", prefix)
    if not first_mv:
        return False
    rest = prefix[first_mv.start() :]
    return bool(re.search(r"(?:^|\s)(?:1-0|0-1|1/2-1/2|\*)(?:\s|$)", rest))


def _is_prose_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if (
        stripped.startswith("[")
        or re.match(r"^\d+\s*[\.\:]", stripped)
        or stripped.startswith("{")
        or stripped.startswith("(")
        or stripped in ("1-0", "0-1", "1/2-1/2", "*")
    ):
        return False
    words = stripped.split()
    if not words:
        return False
    prose_words = {
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "think",
        "thought",
        "thinks",
        "thinking",
        "considered",
        "felt",
        "believe",
        "believed",
        "afterwards",
        "afterward",
        "after",
        "before",
        "during",
        "later",
        "then",
        "next",
        "also",
        "better",
        "best",
        "worse",
        "worst",
        "good",
        "bad",
        "nice",
        "great",
        "poor",
        "blunder",
        "mistake",
        "was",
        "were",
        "is",
        "are",
        "am",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "would",
        "could",
        "should",
        "can",
        "may",
        "might",
        "must",
        "will",
        "shall",
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "game",
        "play",
        "played",
        "moves",
        "move",
        "position",
        "opening",
        "variation",
        "because",
        "since",
        "so",
        "but",
        "and",
        "or",
        "if",
        "though",
        "although",
        "instead",
    }
    prose_count = sum(1 for w in words if w.strip(".,;:!?\"'()").lower() in prose_words)
    if len(words) >= 2 and (prose_count >= 2 or (prose_count / len(words)) >= 0.4):
        return True
    first_word = words[0].strip(".,;:!?\"'()").lower()
    if first_word in (
        "afterwards",
        "afterward",
        "what",
        "how",
        "why",
        "note",
        "comment",
        "analysis",
        "thoughts",
        "question",
        "here",
    ):
        return True
    return False


def _is_canonical_tag_line(line: str) -> bool:
    stripped = line.strip()
    if (
        not stripped
        or stripped.startswith(";")
        or stripped.startswith("%")
        or stripped.startswith("{")
    ):
        return False
    return bool(re.match(r'^(?:\[\s*[A-Za-z0-9_]+\s+"(?:[^"\\]|\\.)*"\s*\]\s*)+$', stripped))


def _clean_conversational_text(text: str) -> str:
    text = _strip_pgn_escape_lines(text)
    text = _normalize_multiline_tags(text)
    masked_text = _mask_comments_and_escapes(text)

    # 1. Find valid PGN tag pairs outside inline code and comments
    tag_matches: list[re.Match[str]] = []
    for m in TAG_PAIR_REGEX.finditer(masked_text):
        # Ignore tags enclosed in inline code backticks e.g. `[FEN "..."]`
        if (
            m.start() > 0
            and text[m.start() - 1] == "`"
            and m.end() < len(text)
            and text[m.end()] == "`"
        ):
            continue
        line_start = masked_text.rfind("\n", 0, m.start()) + 1
        prefix_on_line = masked_text[line_start : m.start()]
        line_end = masked_text.find("\n", m.end())
        line_end = len(masked_text) if line_end == -1 else line_end
        suffix_on_line = masked_text[m.end() : line_end]

        pref_clean = prefix_on_line.strip()
        suff_clean = suffix_on_line.strip()
        if pref_clean.endswith("`") or suff_clean.startswith("`"):
            continue
        # If suffix on the same line contains prose text (not additional tag pairs):
        if suff_clean and not suff_clean.startswith("["):
            continue
        tag_matches.append(m)

    best_header_str = ""
    best_movetext_str = ""

    first_mv_in_full = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", masked_text)

    if tag_matches:
        clusters: list[list[re.Match[str]]] = []
        curr_cluster: list[re.Match[str]] = [tag_matches[0]]
        for m in tag_matches[1:]:
            prev_m = curr_cluster[-1]
            gap = text[prev_m.end() : m.start()]
            if gap.strip() == "":
                curr_cluster.append(m)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [m]
        clusters.append(curr_cluster)

        def _cluster_eval(cl: list[re.Match[str]]) -> tuple[bool, int, int]:
            h_end = cl[-1].end()
            after_h = text[h_end:].strip()
            first_mv_after = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_h)
            has_direct_moves = False
            if first_mv_after:
                mv_pos = h_end + first_mv_after.start()
                # Ensure no other valid tag cluster appears between this cluster and the moves
                other_cluster_between = any(
                    other_cl is not cl and h_end <= other_cl[0].start() < mv_pos
                    for other_cl in clusters
                )
                if not other_cluster_between:
                    has_direct_moves = True
            std_count = sum(
                1
                for m in cl
                if m.group(1)
                in (
                    "Event",
                    "Site",
                    "Date",
                    "Round",
                    "White",
                    "Black",
                    "Result",
                    "FEN",
                    "SetUp",
                    "ECO",
                    "Opening",
                    "Termination",
                    "Annotator",
                    "WhiteElo",
                    "BlackElo",
                    "TimeControl",
                    "Variant",
                )
            )
            return (has_direct_moves, std_count, len(cl))

        clusters.sort(key=_cluster_eval, reverse=True)
        best_cluster = clusters[0]
        has_direct_moves, std_count, cl_len = _cluster_eval(best_cluster)

        if has_direct_moves or (first_mv_in_full is None and (std_count > 0 or cl_len >= 2)):
            h_start = best_cluster[0].start()
            h_end = best_cluster[-1].end()
            best_header_str = text[h_start:h_end].strip()

            after_header = text[h_end:].strip()
            first_move_after = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", after_header)
            if first_move_after:
                movetext_candidate = after_header[first_move_after.start() :]
            else:
                movetext_candidate = after_header
            best_movetext_str = movetext_candidate
        elif first_mv_in_full:
            best_movetext_str = text[first_mv_in_full.start() :]
        else:
            best_movetext_str = text
    else:
        if first_mv_in_full:
            best_movetext_str = text[first_mv_in_full.start() :]
        else:
            best_movetext_str = text

    # Trim trailing non-chess prose from movetext
    lines = best_movetext_str.splitlines()
    end_idx = len(lines)
    found_moves = False
    for i, line_item in enumerate(lines):
        stripped = line_item.strip()
        if stripped.startswith(";") or stripped.startswith("%"):
            continue
        if re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", stripped) or re.search(
            r"\b(?:[a-h][1-8]|O-O)", stripped
        ):
            found_moves = True
        elif found_moves:
            if any(
                stripped.startswith(res) or stripped.endswith(res)
                for res in ("1-0", "0-1", "1/2-1/2", "*")
            ):
                end_idx = i + 1
                break
            if (
                _is_prose_line(stripped)
                or stripped.startswith("Here ")
                or stripped.startswith("Note ")
            ):
                end_idx = i
                break
            has_chess = bool(re.search(r"\b\d+\s*[\.\:]|[a-h][1-8]|O-O|1-0|0-1|1/2", stripped))
            if not has_chess and stripped:
                end_idx = i
                break

    final_movetext = "\n".join(lines[:end_idx]).strip()

    if best_header_str and final_movetext:
        return f"{best_header_str}\n\n{final_movetext}"
    return best_header_str or final_movetext or text


def _extract_canonical_pgn_text(text: str) -> str:
    """Isolate the canonical PGN text from markdown fences, conversational preambles, and trailers."""
    cleaned = text.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "").strip()
    # L-04 audit fix: normalize Unicode PGN markers (½-½ etc.) before any
    # further processing so the rest of the parser sees ASCII PGN.
    cleaned = _normalize_unicode_pgn_results(cleaned)
    if not cleaned:
        raise ValueError("Empty chess game/PGN input provided")

    # 1. Enumerate markdown fenced code blocks and rank them (PGN-04: prefer pgn/chess tag)
    fences = list(re.finditer(r"```([a-zA-Z0-9_-]*)\s*([\s\S]*?)\s*```", cleaned))
    if fences:
        ranked: list[tuple[int, str]] = []
        for m in fences:
            lang = (m.group(1) or "").strip().lower()
            body = m.group(2).strip("`'\" \t\r\n")
            if not body:
                continue
            if lang in ("pgn", "chess"):
                score = 100
            elif re.search(r"\b1\s*[\.\:]\s*[A-Za-z]|\[\s*[A-Za-z0-9_]+\s+\"", body):
                score = 50
            else:
                score = 10
            ranked.append((score, body))

        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            best_body = ranked[0][1]
            return _strip_pgn_escape_lines(_normalize_multiline_tags(best_body))

    # 2. Conversational preamble and trailer cleaning
    cleaned = _normalize_multiline_tags(cleaned)
    cleaned_conv = _clean_conversational_text(cleaned)
    if cleaned_conv:
        return _strip_pgn_escape_lines(_normalize_multiline_tags(cleaned_conv))

    return _strip_pgn_escape_lines(cleaned)


def _extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a chess.pgn.Game from raw, dirty, annotated, or conversational text."""
    _check_multiple_games(text)
    canonical = _extract_canonical_pgn_text(text)
    _check_multiple_games(canonical)
    return _extract_game_inner(canonical, strict=strict)


def _extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
    masked_cleaned = _mask_comments_and_escapes(cleaned)
    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        if m.group(1).lower() == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))

    # Translate unicode figurines ONLY in movetext (headers and comments preserved)
    norm_text = _normalize_movetext_figurines(cleaned)
    norm_text = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
        r"\1 \2",
        norm_text,
    )
    norm_text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", norm_text)
    norm_text = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)\*",
        r"\1 *",
        norm_text,
    )
    norm_text = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)\*", r"\1\2 *", norm_text)
    norm_text = re.sub(r"\b0-0-0\b", "O-O-O", norm_text)
    norm_text = re.sub(r"\bo-o-o\b", "O-O-O", norm_text, flags=re.IGNORECASE)
    norm_text = re.sub(r"\b0-0\b", "O-O", norm_text)
    norm_text = re.sub(r"\bo-o\b", "O-O", norm_text, flags=re.IGNORECASE)
    norm_text = re.sub(
        r"\b([a-h][1-8]|[a-h]x[a-h][1-8])\s+\(?e\.?p\.?\)?(?=\s|$)",
        r"\1",
        norm_text,
        flags=re.IGNORECASE,
    )

    # 1. Contiguous leading tag block
    masked_norm = _mask_comments_and_escapes(norm_text)
    header_end = 0
    first_header = TAG_PAIR_REGEX.search(masked_norm)
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", masked_norm)
    if first_header and (not first_mv or first_header.start() < first_mv.start()):
        header_end = first_header.end()
        for m in TAG_PAIR_REGEX.finditer(masked_norm):
            if m.start() < header_end:
                continue
            if masked_norm[header_end : m.start()].strip() == "":
                header_end = m.end()
            else:
                break
    if header_end > 0:
        g = _parse_pgn_game_candidate(norm_text, strict=strict)
        if g is not None:
            if not list(g.mainline_moves()):
                has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
                if has_move_tokens:
                    raise ValueError(
                        f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                    )
            return g

    # 2. Direct parse attempt
    g = _parse_pgn_game_candidate(norm_text, strict=strict)
    if g is not None:
        if not list(g.mainline_moves()):
            has_move_tokens = bool(re.search(r"\b\d+\s*[\.\:]\s*[A-Za-z]", norm_text))
            if has_move_tokens:
                raise ValueError(
                    f"INVALID_PGN: Could not parse legal moves from movetext '{norm_text[:100]}'."
                )
        return g

    # 3. Fallback: movetext starting with 1. or 1...
    for move_match in re.finditer(r"\b1\s*[\.\:]\s*[A-Za-z]", norm_text):
        sub_movetext = norm_text[move_match.start() :]
        try:
            g = _parse_pgn_game_candidate(sub_movetext, strict=strict)
            if g is not None and list(g.mainline_moves()):
                return g
        except Exception:
            continue

    # 4. Fallback: bare SAN / UCI tokens
    tokens = norm_text.split()
    best_moves: list[chess.Move] = []
    best_result: str | None = None

    for start_idx in range(len(tokens)):
        b = chess.Board()
        cur_moves: list[chess.Move] = []
        cur_result: str | None = None
        for t in tokens[start_idx:]:
            clean_t = t.rstrip(".,;:!?").lstrip(".,;:!?")
            clean_t = re.sub(
                r"\s*\(?\s*e\.?p\.?\s*\)?$",
                "",
                clean_tok if "clean_tok" in locals() else clean_t,
                flags=re.IGNORECASE,
            ).rstrip(".,:;!?")
            if (
                not clean_t
                or clean_t.lower() in ("e.p.", "e.p", "ep", "(e.p.)", "(e.p)", "(ep)")
                or re.match(r"^\d+[\.\:]*$", clean_t)
            ):
                continue
            if clean_t in ("1-0", "0-1", "1/2-1/2", "*"):
                cur_result = clean_t
                break
            try:
                m = b.parse_san(clean_t)
                b.push(m)
                cur_moves.append(m)
            except Exception:
                try:
                    m = chess.Move.from_uci(clean_t)
                    if m in b.legal_moves:
                        b.push(m)
                        cur_moves.append(m)
                    else:
                        break
                except Exception:
                    break
        if len(cur_moves) > len(best_moves):
            best_moves = cur_moves
            best_result = cur_result

    if best_moves:
        game = chess.pgn.Game()
        if best_result:
            game.headers["Result"] = best_result
        curr: chess.pgn.GameNode = game
        for m in best_moves:
            curr = curr.add_variation(m)
        return game

    raise ValueError(
        f"INVALID_POSITION: Input '{cleaned[:100]}' could not be parsed as a valid FEN, PGN, or move sequence."
    )


def _parse_move_on_board_with_warning(
    board: chess.Board, move_str: str, strict: bool = False
) -> tuple[chess.Move, str | None]:
    """Parse a move string on a board, accepting either UCI or SAN notation.
    Also detects non-canonical SAN (e.g. false mate/check markers or redundant disambiguation)."""
    # Use is_terminal_position (single source of truth) instead of python-chess's
    # is_game_over(), which does NOT detect locked dead positions (FIDE 5.2.2).
    # Without this, classify_move and other tools would silently accept a move
    # that the rules layer has already declared terminal — disagreeing about
    # the same position across endpoints (audit P0).
    if is_terminal_position(board):
        # Try to name the actual terminal reason
        if board.is_checkmate():
            term = "checkmate"
        elif board.is_stalemate():
            term = "stalemate"
        elif board.is_insufficient_material():
            term = "insufficient_material"
        elif board.is_seventyfive_moves():
            term = "seventyfive_moves"
        elif board.is_fivefold_repetition():
            term = "fivefold_repetition"
        elif is_locked_dead_position(board):
            term = "dead_position"
        elif board.is_game_over():
            term = "game_over"
        else:
            term = "game_over"
        raise ValueError(
            f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over ({term}), no legal moves can be played."
        )

    clean_move = (
        move_str.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    clean_move = re.sub(r"^(\d+[\.\:]+|\.+)\s*", "", clean_move)
    clean_move = clean_move.translate(_FIGURINE_MAP)
    clean_move = re.sub(r"\s*\(?\s*e\.?p\.?\s*\)?$", "", clean_move, flags=re.IGNORECASE)

    # Normalize castling variants
    lower_cand = clean_move.lower()
    if lower_cand in ("o-o-o", "0-0-0", "o-o-o+", "0-0-0+", "o-o-o#", "0-0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O-O{suffix}"
    elif lower_cand in ("o-o", "0-0", "o-o+", "0-0+", "o-o#", "0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O{suffix}"

    san_cand = clean_move.rstrip("!?")

    # Try UCI first (accept lowercase or uppercase promo e.g. e7e8q, e7e8Q)
    for uci_cand in (clean_move, clean_move.lower()):
        try:
            m = chess.Move.from_uci(uci_cand)
            if m in board.legal_moves:
                return m, None
        except (ValueError, chess.InvalidMoveError):
            pass

    # Try SAN with candidates (e.g. clean, stripped !?, stripped +/#, with/without =, promo variants)
    cands = [clean_move, san_cand, san_cand.rstrip("+#!?")]
    # Handle promotion without equal sign e.g. e8Q -> e8=Q
    if re.search(r"[a-h][18][qrbnQRBN]", san_cand):
        cands.append(re.sub(r"([a-h][18])([qrbnQRBN])", r"\1=\2", san_cand))

    ambiguous_err: Exception | None = None
    for cand in cands:
        if not cand:
            continue
        try:
            m = board.parse_san(cand)
            if m in board.legal_moves:
                canonical = board.san(m)
                syntax_warning = None
                raw_s = move_str.strip(" \t\r\n`'\"")
                if raw_s != canonical and not re.fullmatch(
                    r"[a-h][1-8][a-h][1-8][qrbn]?", raw_s.lower()
                ):
                    syntax_warning = f"Input SAN '{raw_s}' normalized to '{canonical}'"
                if strict and syntax_warning:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input SAN '{raw_s}' requires syntax normalization: {syntax_warning}"
                    )
                return m, syntax_warning
        except (chess.AmbiguousMoveError, chess.IllegalMoveError) as exc:
            if "ambiguous" in str(exc).lower() or isinstance(exc, chess.AmbiguousMoveError):
                ambiguous_err = exc
        except (ValueError, chess.InvalidMoveError) as exc:
            if "STRICT" in str(exc):
                raise

    if ambiguous_err:
        raise ValueError(
            f"AMBIGUOUS_SAN: Move {move_str!r} is ambiguous in position {board.fen()!r}: {ambiguous_err}"
        )
    raise ValueError(
        f"ILLEGAL_MOVE: Move {move_str!r} is not a valid legal move in position {board.fen()!r}"
    )


def _parse_move_on_board(board: chess.Board, move_str: str) -> chess.Move:
    return _parse_move_on_board_with_warning(board, move_str)[0]


def _build_board(
    fen_or_pgn: str, moves: list[str] | None = None, strict: bool = False
) -> chess.Board:
    """Build a chess.Board from FEN, PGN, or movetext, optionally replaying additional UCI/SAN moves."""
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    if cleaned.lower() in ("startpos", "initial", "start"):
        board: chess.Board | None = chess.Board()
    else:
        board = None
        tokens = cleaned.split()
        if 1 <= len(tokens) <= 6 and not cleaned.startswith("[") and not tokens[0].endswith("."):
            if "/" in cleaned:
                if len(tokens) >= 5:
                    try:
                        halfmove_num = int(tokens[4])
                        if halfmove_num < 0:
                            raise ValueError(
                                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' cannot be negative (got {tokens[4]})."
                            )
                    except ValueError as exc:
                        if "INVALID_FEN" in str(exc):
                            raise
                        raise ValueError(
                            f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' must be a valid integer."
                        ) from exc
                if len(tokens) == 6:
                    try:
                        fullmove_num = int(tokens[5])
                        if fullmove_num < 1:
                            raise ValueError(
                                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a positive integer >= 1 (got {tokens[5]})."
                            )
                    except ValueError as exc:
                        if "INVALID_FEN" in str(exc):
                            raise
                        raise ValueError(
                            f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a valid integer."
                        ) from exc
            try:
                b = chess.Board(cleaned)
                if b.is_valid() or b.status() == chess.STATUS_VALID:
                    board = b
                elif "/" in cleaned:
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned}' is not a valid FEN ({format_fen_status_errors(b.status())})."
                    )
            except ValueError as exc:
                if "/" in cleaned or "INVALID_FEN" in str(exc):
                    if str(exc).startswith("INVALID_FEN:"):
                        raise
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned}' could not be parsed as a valid FEN: {exc}"
                    ) from exc
                board = None
            except IndexError as exc:
                if "/" in cleaned:
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned}' is not a valid FEN."
                    ) from exc
                board = None

    if board is None:
        game = _extract_game(cleaned)
        board = game.board()
        if not board.is_valid() or board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(board.status())})."
            )
        for move in game.mainline_moves():
            board.push(move)

    for move_str in moves or []:
        move, _ = _parse_move_on_board_with_warning(board, move_str, strict=strict)
        board.push(move)

    return board


def _build_board_with_metadata(
    fen_or_pgn: str, moves: list[str] | None = None, strict: bool = False
) -> tuple[chess.Board, str | None, str, bool]:
    """Build a chess.Board from FEN/PGN/movetext AND return canonicalization metadata (audit L-06).

    Returns:
        (board, input_fen, canonical_fen, fen_was_canonicalized)
        where input_fen is None when the input wasn't a single 6-field FEN
        (it was startpos, a PGN, or unparseable raw movetext).

    The flag tells callers whether python-chess silently rewrote the EP
    target — an honest observability hook for the common case where a
    Lichess export says "...KQkq e3 0 1" with no black pawn on d4,
    and python-chess's Board constructor drops the e3 to "-" because
    no piece can actually capture en passant.
    """
    input_fen: str | None = None
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    # Try to capture the raw input FEN so callers can see if we rewrote it.
    tokens = cleaned.split()
    if (
        1 <= len(tokens) <= 6
        and not cleaned.startswith("[")
        and not tokens[0].endswith(".")
        and "/" in cleaned
    ):
        input_fen = cleaned

    board = _build_board(fen_or_pgn, moves, strict)
    canonical = board.fen()
    was_canonicalized = bool(input_fen) and input_fen != canonical
    return board, input_fen, canonical, was_canonicalized


async def _eval_via_analyzer_or_pool(
    analyzer: object | None,
    pool: AnalyzerPool | TCPAnalyzerPool,
    b: chess.Board,
    *,
    depth: int,
    reuse_tt: bool,
    root_moves: list[chess.Move] | None = None,
) -> Eval:
    """Run a single eval call.

    When `analyzer` is given, use it directly (skips pool acquire round-trip
    and lets the caller control `reuse_tt` for TT accumulation across calls).
    When `analyzer` is None, route through `pool.evaluate` which acquires a
    fresh worker — `reuse_tt` is ignored on this path because the next call
    may land on a different worker.
    """
    if analyzer is not None:
        return await analyzer.evaluate(  # type: ignore[attr-defined]
            b, depth=depth, reuse_tt=reuse_tt, root_moves=root_moves
        )
    # pool.evaluate signature varies; pass root_moves only if analyzer pool
    # supports it (production TCPAnalyzerPool does; test mocks don't).
    import inspect
    try:
        sig = inspect.signature(pool.evaluate)
        if "root_moves" in sig.parameters:
            return await pool.evaluate(b, depth=depth, root_moves=root_moves)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        pass
    return await pool.evaluate(b, depth=depth)  # type: ignore[arg-type]


async def _evaluate_game_position_cached(
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    requested_depth: int | None = None,
    history_complete: bool = True,
    reuse_tt: bool = False,
    analyzer: object | None = None,
) -> tuple[MCPEval, bool]:
    """Evaluate a single board state with rule status, terminal short-circuits, and multi-tier cache.

    Args:
        history_complete: True when the caller had access to the full move stack
            (PGN, evaluate_position with moves param). False for naked FEN — drives
            `history_completeness` and `repetition_status` on the returned MCPEval
            (audit H-01).
        reuse_tt: pass True when consecutive calls on the same engine share
            position-tree history. Saves a `ucinewgame` round-trip and lets
            Stockfish accumulate the TT across calls. Caller is responsible
            for the semantic correctness — only use when the previous call's
            FEN is the predecessor of the current.
        analyzer: optional pre-acquired analyzer instance. When set, skips the
            pool.acquire() round-trip and calls `analyzer.evaluate(...)`
            directly. The `_gather_evaluate_positions_chunked` helper holds
            one analyzer per slice for sequential calls within the slice.
    """
    req_d = requested_depth if requested_depth is not None else depth
    canonical_fen_str = b.fen()
    url, img = lichess_urls(canonical_fen_str)

    rule_status = evaluate_rule_status(b, history_complete=history_complete)
    if rule_status.terminal is not None:
        # Terminal outcomes are reported from White's perspective (same convention as cp).
        if rule_status.terminal == "checkmate":
            term_outcome = "win" if rule_status.winner == "white" else "loss"
            term_cp: int | None = None
            term_mate: int | None = 0
        else:
            term_outcome = "draw"
            term_cp = 0
            term_mate = None
        return (
            MCPEval(
                status=rule_status.terminal,
                winner=rule_status.winner,
                cp=term_cp,
                mate=term_mate,
                best_move=None,
                pv=[],
                depth=0,
                requested_depth=req_d,
                searched_depth=0,
                can_claim_draw=False,
                claim_reasons=[],
                can_claim_now=False,
                claim_reasons_now=[],
                can_claim_with_intended_move=False,
                claim_moves=[],
                recommended_action="game_over",
                best_action="game_over",
                best_action_type="game_over",
                decision_value={
                    "outcome": term_outcome,
                    "cp_equivalent": term_cp,
                    "best_action": "game_over",
                    "perspective": "white",
                },
                engine_eval={
                    "cp": term_cp,
                    "mate": term_mate,
                    "best_move": None,
                    "pv": [],
                    "depth": 0,
                },
                history_dependent_status=rule_status.history_dependent_status,
                lichess_url_reproduces_history=rule_status.fen_sufficient_for_status,
                requires_move_stack=rule_status.requires_move_stack,
                fen_sufficient_for_status=rule_status.fen_sufficient_for_status,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                lichess_url=url,
                lichess_image=img,
                **_build_identity(pool),
            ),
            True,
        )

    ckey = eval_cache_key(
        b,
        depth,
        engine_version=getattr(pool, "engine_version", None),
    )
    cached = await _cache.get_eval(ckey)
    if cached is not None:
        return cached.model_copy(update={"requested_depth": req_d}), True

    async def _compute_pos() -> MCPEval:
        ev = await _eval_via_analyzer_or_pool(analyzer, pool, b, depth=depth, reuse_tt=reuse_tt)

        # Rule-aware root best-move check (P0 audit fix):
        # If at halfmove 149, or halfmove >= 100 with winning score:
        # Check if the raw best move walks into 75-move draw or concedes a
        # claim while another move preserves the win.
        #
        # CRITICAL: if we override `ev.best_move`, we MUST re-run Stockfish
        # with the new move as a root_moves constraint so cp/mate reflect the
        # actual move, NOT the original. The previous implementation kept the
        # original cp/mate, producing the contradictory `best_move=B, cp=eval(A)`
        # invariant violation flagged in the audit.
        if ev.best_move and (b.halfmove_clock == 149 or b.halfmove_clock >= 100):
            try:
                bm_obj = chess.Move.from_uci(ev.best_move)
                b_after = b.copy(stack=True)
                if bm_obj in b.legal_moves:
                    b_after.push(bm_obj)
                    is_bm_75 = b_after.is_seventyfive_moves()
                    is_bm_conceded = b.halfmove_clock >= 100 and b_after.is_fifty_moves()

                    if is_bm_75 or is_bm_conceded:
                        # Find a win-preserving reset move (capture / pawn move / mate).
                        override_move: chess.Move | None = None
                        override_is_mate = False
                        for cand in b.legal_moves:
                            b_sub = b.copy(stack=True)
                            b_sub.push(cand)
                            if b_sub.is_checkmate():
                                override_move = cand
                                override_is_mate = True
                                break
                            elif not b_sub.is_seventyfive_moves() and (
                                b.is_capture(cand)
                                or b.piece_type_at(cand.from_square) == chess.PAWN
                            ):
                                override_move = cand
                                break

                        if override_move is not None:
                            if override_is_mate:
                                # Stockfish's mate distance from the post-move board is
                                # either 0 (already checkmated — terminal) or 1 (mate in 1
                                # from the mover's POV). Use the post-move eval semantics.
                                ev.best_move = override_move.uci()
                                ev.mate = 1
                                ev.cp = None
                                ev.pv = [override_move.uci()]
                                ev.depth = depth
                            else:
                                # Re-run Stockfish with root_moves constrained to the
                                # override so cp/mate reflect THIS move's eval, not the
                                # original best move's. Without this we get the audit's
                                # contradictory `best_move=B, cp=eval(A)` invariant.
                                try:
                                    override_eval = await pool.evaluate(
                                        b, depth=depth, root_moves=[override_move]
                                    )
                                    if (
                                        override_eval.best_move
                                        and override_eval.best_move.lower()
                                        == override_move.uci().lower()
                                    ):
                                        ev.best_move = override_eval.best_move
                                        ev.cp = override_eval.cp
                                        ev.mate = override_eval.mate
                                        ev.pv = override_eval.pv
                                        ev.depth = override_eval.depth
                                    else:
                                        # Verification search did not confirm our override;
                                        # trust the original eval but mark it as unverified
                                        # by leaving Stockfish's choice intact.
                                        pass
                                except Exception:
                                    # Re-eval failed; trust original rather than ship
                                    # unverified cp/mate for a different best_move.
                                    pass
            except Exception:
                pass

        mcp_eval = MCPEval.from_eval(
            ev,
            canonical_fen_str,
            board=b,
            requested_depth=req_d,
            history_complete=history_complete,
        )
        # Stamp build identity so every cached eval records which build produced it.
        identity = _build_identity(pool)
        mcp_eval = mcp_eval.model_copy(
            update={
                "build_sha": identity["build_sha"],
                "engine_config": identity["engine_config"],
            }
        )
        await _cache.set_eval(ckey, mcp_eval)
        # Ponder: warm the L1 cache for the position AFTER the engine's best
        # move so the next user request on that FEN hits L1 (env-disabled
        # by default — costs CPU on small hosts).
        _maybe_ponder_warm(
            pool,
            b,
            mcp_eval.best_move,
            depth,
            ponder_enabled=getattr(pool, "_mcp_ponder_enabled", False),
        )
        return mcp_eval

    res = cast(MCPEval, await _single_flight.do(ckey, _compute_pos))
    return res.model_copy(update={"requested_depth": req_d}), False


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def evaluate_position(
    fen: str,
    moves: list[str] | None = None,
    depth: int = 14,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> MCPEval:
    """Evaluate a chess position with Stockfish.

    Args:
        fen: FEN or PGN string for the position (or position before `moves` are replayed).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).
        verbosity: "full" (default, every field) or "compact" (strips Lichess URLs,
            images, decision_value/engine_eval duplication). Use compact when the
            caller is an LLM and you want to minimize context spend (audit M-05).

    Returns:
        Eval with cp (from White's perspective), mate (from White's perspective),
        best_move (UCI), pv (principal variation), and Lichess board URLs.
    """
    t0 = time.time()
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    verbosity_mode = _resolve_verbosity(verbosity)
    try:
        board = _build_board(fen, moves or [], strict=strict)
        pool = await _get_analyzer_pool(ctx)
        # History completeness is derived from whether the caller had the move
        # stack. Naked FEN (no moves) cannot detect threefold repetition;
        # we MUST report `repetition_status="unknown"` for the audit H-01 fix.
        # When moves were supplied, the move stack is complete and we can
        # answer threefold claims definitively.
        history_complete = bool(moves)
        res, is_hit = await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete=history_complete,
        )
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, cache_hit=is_hit)
        # L-06: surface input vs canonical FEN so callers can detect when
        # python-chess silently rewrote the EP target or other fields.
        canonical_fen = board.fen()
        cleaned_input = (
            fen.replace("\u00a0", " ")
            .replace("\u200b", "")
            .replace("\ufeff", "")
            .strip("`'\" \t\r\n")
        )
        is_fen_input = (
            "/" in cleaned_input
            and 1 <= len(cleaned_input.split()) <= 6
            and not cleaned_input.startswith("[")
            and not cleaned_input.lower().startswith("startpos")
        )
        result = res.model_copy(
            update={
                "requested_depth": raw_requested_depth,
                "input_fen": cleaned_input if is_fen_input else None,
                "canonical_fen": canonical_fen,
                "fen_was_canonicalized": is_fen_input and cleaned_input != canonical_fen,
            }
        )
        if verbosity_mode == VERBOSITY_COMPACT:
            result = _compact_mcpeval(result)
        return result
    except ToolError:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = "invalid_input"
        if "STRICT" in msg:
            code = "strict_validation_error"
        elif "UNSUPPORTED_VARIANT" in msg:
            code = "unsupported_variant"
        elif "INVALID_FEN" in msg:
            code = "invalid_fen"
        elif "INVALID_POSITION" in msg:
            code = "invalid_position"
        elif "MULTIPLE_GAMES" in msg:
            code = "multiple_games_not_supported"
        elif "ILLEGAL_MOVE" in msg:
            code = "illegal_move"
        elif "AMBIGUOUS_SAN" in msg:
            code = "ambiguous_san"
        elif "GAME_ALREADY_OVER" in msg:
            code = "game_already_over"
        elif "Invalid PGN" in msg or "Could not parse PGN" in msg or "INVALID_PGN" in msg:
            code = "invalid_pgn"
        raise _tool_error(code=code, message=msg, tool="evaluate_position", input=fen) from exc
    except Exception as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="evaluate_position") from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def top_moves(
    fen: str,
    moves: list[str] | None = None,
    n: int = 3,
    depth: int = 14,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> TopMovesResult:
    """Get the top N candidate moves for a position, ranked best first.

    Args:
        fen: FEN or PGN string for the position.
        moves: Optional UCI or SAN moves to replay onto the position first.
        n: Number of candidates to return (default 3, clamped 1-20).
        depth: Stockfish search depth (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        TopMovesResult object with `status`, `winner`, `recommended_action`,
        `best_action_obj` (typed discriminated union per audit 10.2),
        `legal_actions` (typed list of legal actions), and a `result` array
        of candidate MCPEval objects ranked best first.

        IMPORTANT (audit C-02 / H-03):
          Each candidate in `result` represents a `play_move` action evaluated
          against its POST-CANDIDATE position. Draw-claim actions are reported
          separately at the outer level via `best_action_obj` and
          `legal_actions` — they are NOT mixed into candidate scores.
          Candidate `cp`/`mate` reflects the engine's evaluation of the
          position AFTER the move is played; the post-position status, winner,
          and Lichess URL refer to that post-state.

        For terminal positions (checkmate, stalemate, insufficient material,
        repetition, 75-move rule), returns TopMovesResult with status and
        empty `result: []`.
    """
    t0 = time.time()
    raw_requested_depth = depth
    raw_requested_n = n
    depth = max(1, min(depth, 30))
    clamped_n = max(1, min(n, 20))
    n = clamped_n
    verbosity_mode = _resolve_verbosity(verbosity)
    try:
        board = _build_board(fen, moves or [], strict=strict)
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
        history_complete = bool(moves)
        rule_status = evaluate_rule_status(board, history_complete=history_complete)
        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))
        legal_move_count = board.legal_moves.count()

        if rule_status.terminal is not None:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            # Build a typed game_over best_action
            from mcp_server.actions import build_best_action, build_legal_actions

            best_action_obj = build_best_action(
                recommended_action=rule_status.recommended_action,
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                sign=1 if board.turn == chess.WHITE else -1,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=None,
                board=board,
                legal_engine_moves=None,
            )
            return TopMovesResult(
                status=rule_status.terminal,
                winner=rule_status.winner,
                recommended_action="game_over",
                can_claim_draw=False,
                claim_reasons=[],
                can_claim_now=False,
                claim_reasons_now=[],
                can_claim_with_intended_move=False,
                claim_moves=[],
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=0,
                legal_move_count=legal_move_count,
                canonical_fen=board.fen(),
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=[],
            )

        canonical_fen_str = board.fen()
        cache_key = top_moves_cache_key(
            board,
            depth,
            n=n,
            engine_version=getattr(pool, "engine_version", None),
        )

        # sign = mover's perspective sign (White=+1, Black=-1). Used below for
        # both the cache-hit and the freshly-computed paths to decide whether a
        # candidate is winning FOR the side-to-move (cp is White-POV).
        sign = 1 if board.turn == chess.WHITE else -1

        from mcp_server.actions import build_best_action, build_legal_actions

        def _pick_root_recommended_action(items: list[MCPEval]) -> str:
            """Override root rule_status recommendation when a candidate shows a
            forced win for the side-to-move (positive mate or cp >= 100).
            Otherwise keep the rule-driven recommendation.
            """
            if not items:
                return rule_status.recommended_action
            for item in items:
                if item.mate is not None and sign * item.mate > 0:
                    return "play_move"
                if item.cp is not None and sign * item.cp >= 100:
                    return "play_move"
            return rule_status.recommended_action

        cached = await _cache.get_top_moves(cache_key)
        if cached is not None and len(cached) >= n:
            await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=True)
            items = [
                c.model_copy(update={"requested_depth": raw_requested_depth}) for c in cached[:n]
            ]
            # Apply compact verbosity to cached candidates too (audit M-05)
            if verbosity_mode == VERBOSITY_COMPACT:
                items = [_compact_mcpeval(c) for c in items]
            root_rec_action = _pick_root_recommended_action(items)
            best_action_obj = build_best_action(
                recommended_action=root_rec_action,
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                sign=sign,
            )
            legal_actions = build_legal_actions(
                rule_status=rule_status,
                engine_eval=items[0] if items else None,
                board=board,
                legal_engine_moves=list(items),
            )
            return TopMovesResult(
                status="active",
                winner=None,
                recommended_action=root_rec_action,
                can_claim_draw=rule_status.can_claim_draw,
                claim_reasons=rule_status.claim_reasons,
                claim_move=rule_status.claim_move,
                can_claim_now=rule_status.can_claim_now,
                claim_reasons_now=rule_status.claim_reasons_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                claim_moves=rule_status.claim_moves,
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=depth,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=len(items),
                legal_move_count=legal_move_count,
                canonical_fen=board.fen(),
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=items,
            )

        async def _compute() -> list[MCPEval]:
            # MultiPV search. Stockfish returns the top-N lines with multipv=N;
            # line 1 (multipv=1) is by definition the engine's canonical best,
            # same as a standalone `evaluate_position` would return. No need
            # for a redundant single-PV pre-search (was costing ~25% of
            # `top_moves` wall time at depth 14).
            res_list: list[MCPEval] = []
            results = await pool.top_moves(board, n=n, depth=depth)
            # AUDIT C-02 / H-03: each candidate is a play_move action evaluated
            # AGAINST ITS POST-POSITION. The post-candidate terminal state,
            # winner, and Lichess URL describe the position after the move —
            # NOT a hypothetical claim outcome.
            for r in results:
                b_cand = board.copy(stack=True)
                cand_san_val: str | None = None
                cand_post_terminal: str | None = None
                cand_can_claim_now = False
                cand_can_claim_draw = False
                cand_claim_reasons: list[str] = []
                cand_claim_reasons_now: list[str] = []
                cand_claim_moves: list[str] = []

                if r.best_move:
                    try:
                        bm_obj = chess.Move.from_uci(r.best_move.lower())
                        if bm_obj in board.legal_moves:
                            cand_san_val = board.san(bm_obj)
                            b_cand.push(bm_obj)
                            cand_sign = 1 if b_cand.turn == chess.WHITE else -1
                            cand_mover_score = cand_sign * (
                                r.cp if r.cp is not None else (r.mate * 1000 if r.mate is not None else 0)
                            )
                            cand_mate_for_mover = cand_sign * r.mate if r.mate is not None else None
                            cand_rule = evaluate_rule_status(
                                b_cand,
                                mover_score=cand_mover_score,
                                mate_for_mover=cand_mate_for_mover,
                                history_complete=history_complete,
                            )
                            cand_post_terminal = cand_rule.terminal
                            cand_can_claim_now = cand_rule.can_claim_now
                            cand_can_claim_draw = cand_rule.can_claim_draw
                            cand_claim_reasons = cand_rule.claim_reasons
                            cand_claim_reasons_now = cand_rule.claim_reasons_now
                            cand_claim_moves = cand_rule.claim_moves
                    except Exception:
                        pass

                mcp_eval = MCPEval.from_eval(
                    r,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": "game_over" if cand_post_terminal is not None else "play_move",
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": rule_status.winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                        },
                    }
                )
                res_list.append(mcp_eval)

            def _candidate_rank_key(eval_item: MCPEval) -> float:
                # eval_item.{cp,mate} are White-POV evaluations of the POST-candidate
                # position (per the per-candidate from_eval rebase above), so we rank
                # in MOVER-POV — sign*cp positive means the candidate is GOOD FOR THE
                # SIDE-TO-MOVE. Sorting descending puts the side-to-move's best
                # candidates first regardless of which color is on turn.
                if eval_item.post_terminal_status == "checkmate":
                    return 10000.0
                if eval_item.post_terminal_status in (
                    "stalemate",
                    "insufficient_material",
                    "seventyfive_moves",
                    "fivefold_repetition",
                    "dead_position",
                ):
                    return 0.0
                if eval_item.best_move:
                    try:
                        bm = chess.Move.from_uci(eval_item.best_move)
                        if (
                            board.is_capture(bm)
                            or board.piece_type_at(bm.from_square) == chess.PAWN
                        ):
                            if eval_item.cp is not None:
                                return float(sign * eval_item.cp)
                    except Exception:
                        pass
                if eval_item.mate is not None:
                    if sign * eval_item.mate > 0:
                        return 10000.0 - abs(eval_item.mate)
                    return -10000.0 + abs(eval_item.mate)
                if eval_item.cp is not None:
                    return float(sign * eval_item.cp)
                return 0.0

            has_terminal_cand = any(e.post_terminal_status is not None for e in res_list)
            if board.halfmove_clock >= 100 or has_terminal_cand:
                res_list.sort(key=_candidate_rank_key, reverse=True)

            await _cache.set_top_moves(cache_key, res_list)
            return res_list

        sf_key = f"{cache_key}:n={n}"
        res = cast(list[MCPEval], await _single_flight.do(sf_key, _compute))
        await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=False)
        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
        root_rec_action = _pick_root_recommended_action(items)
        best_action_obj = build_best_action(
            recommended_action=root_rec_action,
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            sign=sign,
        )
        legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=items[0] if items else None,
            board=board,
            legal_engine_moves=list(items),
        )
        return TopMovesResult(
            status="active",
            winner=None,
            recommended_action=root_rec_action,
            can_claim_draw=rule_status.can_claim_draw,
            claim_reasons=rule_status.claim_reasons,
            claim_move=rule_status.claim_move,
            can_claim_now=rule_status.can_claim_now,
            claim_reasons_now=rule_status.claim_reasons_now,
            can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
            claim_moves=rule_status.claim_moves,
            best_action_obj=best_action_obj,
            legal_actions=legal_actions,
            history_completeness=rule_status.history_completeness,
            repetition_status=rule_status.repetition_status,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            requested_n=raw_requested_n,
            clamped_n=clamped_n,
            returned_n=len(items),
            legal_move_count=legal_move_count,
            canonical_fen=board.fen(),
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            result=items,
        )
        # Audit M-05: apply compact verbosity if requested. Compact strips
        # Lichess URLs, images, and engine_eval/decision_value duplication
        # from every candidate to reduce LLM context spend (~70% smaller).
        if verbosity_mode == VERBOSITY_COMPACT:
            items = [_compact_mcpeval(c) for c in items]
            return TopMovesResult(
                status="active",
                winner=None,
                recommended_action=root_rec_action,
                can_claim_draw=rule_status.can_claim_draw,
                claim_reasons=rule_status.claim_reasons,
                claim_move=rule_status.claim_move,
                can_claim_now=rule_status.can_claim_now,
                claim_reasons_now=rule_status.claim_reasons_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                claim_moves=rule_status.claim_moves,
                best_action_obj=best_action_obj,
                legal_actions=legal_actions,
                history_completeness=rule_status.history_completeness,
                repetition_status=rule_status.repetition_status,
                requested_depth=raw_requested_depth,
                searched_depth=depth,
                requested_n=raw_requested_n,
                clamped_n=clamped_n,
                returned_n=len(items),
                legal_move_count=legal_move_count,
                canonical_fen=board.fen(),
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=items,
            )
        return TopMovesResult(
            status="active",
            winner=None,
            recommended_action=root_rec_action,
            can_claim_draw=rule_status.can_claim_draw,
            claim_reasons=rule_status.claim_reasons,
            claim_move=rule_status.claim_move,
            can_claim_now=rule_status.can_claim_now,
            claim_reasons_now=rule_status.claim_reasons_now,
            can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
            claim_moves=rule_status.claim_moves,
            best_action_obj=best_action_obj,
            legal_actions=legal_actions,
            history_completeness=rule_status.history_completeness,
            repetition_status=rule_status.repetition_status,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            requested_n=raw_requested_n,
            clamped_n=clamped_n,
            returned_n=len(items),
            legal_move_count=legal_move_count,
            canonical_fen=board.fen(),
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            result=items,
        )
    except ToolError:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = "invalid_input"
        if "STRICT" in msg:
            code = "strict_validation_error"
        elif "UNSUPPORTED_VARIANT" in msg:
            code = "unsupported_variant"
        elif "INVALID_FEN" in msg:
            code = "invalid_fen"
        elif "INVALID_POSITION" in msg:
            code = "invalid_position"
        elif "MULTIPLE_GAMES" in msg:
            code = "multiple_games_not_supported"
        elif "ILLEGAL_MOVE" in msg:
            code = "illegal_move"
        elif "AMBIGUOUS_SAN" in msg:
            code = "ambiguous_san"
        elif "GAME_ALREADY_OVER" in msg:
            code = "game_already_over"
        elif "Invalid PGN" in msg or "Could not parse PGN" in msg or "INVALID_PGN" in msg:
            code = "invalid_pgn"
        raise _tool_error(code=code, message=msg, tool="top_moves", input=fen) from exc
    except Exception as exc:
        await metrics.record("top_moves", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="top_moves") from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def classify_move(
    fen: str,
    move: str,
    moves: list[str] | None = None,
    depth: int = 14,
    action_type: str = "play_move",
    strict: bool = False,
    ctx: Context | None = None,
) -> MCPMoveAnalysis:
    """Grade a played move against Stockfish's best alternative.

    Grades: 'best', 'good', 'inaccuracy', 'mistake', 'blunder'. Note that `move_class`
    is derived from `effective_loss` (win probability impact & position context, e.g.
    decisive advantage saturation), NOT directly from raw `centipawn_loss`. Also returns
    centipawn loss, mate distance loss, and evals before/after the move.

    Args:
        fen: FEN or PGN string for the position BEFORE `move`.
        move: The move to grade in UCI (e.g. "e2e4") or SAN (e.g. "e4", "Bxf3", "O-O").
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 14, clamped 1-30).
        action_type: Intended chess action ('play_move', 'claim_draw', 'claim_draw_with_intended_move').
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        MoveAnalysis with move_class, centipawn_loss, effective_loss, eval_before, eval_after,
        best_move_san, best_line_san, and played_line_san.
    """
    t0 = time.time()
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        board = _build_board(fen, moves or [], strict=strict)
        chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)

        cache_key = classify_cache_key(
            board,
            chess_move.uci(),
            depth,
            action_type=action_type,
            engine_version=getattr(_analyzer_pool, "engine_version", None),
        )

        cached = await _cache.get_classify(cache_key)
        if cached is not None:
            await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=True)
            eval_bef = cached.eval_before.model_copy(
                update={"requested_depth": raw_requested_depth}
            )
            eval_aft = cached.eval_after.model_copy(update={"requested_depth": raw_requested_depth})
            return cached.model_copy(
                update={
                    "eval_before": eval_bef,
                    "eval_after": eval_aft,
                    "syntax_warning": syntax_warn,
                }
            )

        played_san = board.san(chess_move)
        board_after = board.copy(stack=True)
        board_after.push(chess_move)

        async def _compute() -> MCPMoveAnalysis:
            pool = await _get_analyzer_pool(ctx)

            if hasattr(pool, "classify_move") and type(pool) not in (
                AnalyzerPool,
                TCPAnalyzerPool,
            ):
                result = await pool.classify_move(board, chess_move, depth=depth)
                return MCPMoveAnalysis.from_analysis(
                    result,
                    fen_before=board.fen(),
                    fen_after=board_after.fen(),
                    played_san=played_san,
                    board_before=board,
                    board_after=board_after,
                    syntax_warning=None,
                    action_type=action_type,
                )

            eval_before, _ = await _evaluate_game_position_cached(
                board, depth, pool, requested_depth=raw_requested_depth
            )

            # Fast path: when the played move is the engine's canonical best,
            # skip the second engine call. eval_after is approximated by
            # applying eval_before's PV to the board (it's the line the engine
            # itself would play). For positions where best_move is None or
            # doesn't match the played move, fall through to the real eval.
            played_is_best = (
                eval_before.best_move
                and chess_move.uci().lower() == eval_before.best_move.lower()
            )
            if played_is_best:
                # Synthesize eval_after from the PV tail. Walk the PV starting
                # at move index 1 (the engine's chosen move is at index 0, the
                # played move). For short or missing PVs, fall back to a real
                # eval — it's cheap and rare.
                pv = eval_before.pv or []
                if len(pv) >= 2:
                    synth_board = board.copy(stack=True)
                    try:
                        for uci in pv[1:]:
                            synth_board.push_uci(uci)
                        synth_eval, synth_hit = await _evaluate_game_position_cached(
                            synth_board,
                            depth,
                            pool,
                            requested_depth=raw_requested_depth,
                            reuse_tt=True,
                            analyzer=None,
                        )
                        eval_after = synth_eval
                    except Exception:
                        # Fall back to real eval — TT reuse on this connection
                        # should still be a net win because best_move was already
                        # found by the engine.
                        eval_after, _ = await _evaluate_game_position_cached(
                            board_after, depth, pool, requested_depth=raw_requested_depth
                        )
                else:
                    eval_after, _ = await _evaluate_game_position_cached(
                        board_after, depth, pool, requested_depth=raw_requested_depth
                    )
            else:
                eval_after, _ = await _evaluate_game_position_cached(
                    board_after, depth, pool, requested_depth=raw_requested_depth
                )

            score = score_played_move(
                board,
                chess_move,
                eval_before,
                eval_after,
                board_after,
                action_type=action_type,
            )

            # Candidate Verification Search (Opera Morphy invariant enforcement):
            # If played move matched eval_before.best_move, but grading would produce mistake/blunder,
            # run a deeper verification search so eval_before is updated with the true best candidate.
            #
            # P1 audit fix: this branch used to be FAIL-OPEN — when the deeper
            # search threw any exception, the code silently flipped move_class to
            # BEST, effective_loss=0. That makes a buggy engine produce honest
            # answers and a buggy harness produce lies. The fixed behavior:
            #   - if verification succeeds and finds a better move, regrade.
            #   - if verification succeeds and confirms our move, lock to BEST.
            #   - if verification FAILS, do NOT silently overwrite grading;
            #     mark classification_verified=False so callers see the
            #     unverified result instead of a fabricated "best".
            verification_attempted = False
            if (
                chess_move.uci().lower() == (eval_before.best_move or "").lower()
                and score.move_class in (MoveClass.MISTAKE, MoveClass.BLUNDER)
                and not score.missed_draw_claim
                and not score.conceded_draw_claim
            ):
                try:
                    # Cache the depth+4 verification result via the same
                    # L1/L2 path as any other eval. Previously this went
                    # straight to pool.evaluate, bypassing the cache — every
                    # classify_move that hit this verification path paid the
                    # full uncached depth+4 cost. Now the depth+4 result is
                    # cached like any other eval.
                    verify_eval_result, _verify_hit = await _evaluate_game_position_cached(
                        board, depth + 4, pool, requested_depth=raw_requested_depth + 4
                    )
                    verify_ev: Eval = Eval(
                        cp=verify_eval_result.cp,
                        mate=verify_eval_result.mate,
                        best_move=verify_eval_result.best_move,
                        pv=verify_eval_result.pv,
                        depth=verify_eval_result.searched_depth or (depth + 4),
                    )
                    verification_attempted = True
                    if (
                        verify_ev.best_move
                        and verify_ev.best_move.lower() != chess_move.uci().lower()
                    ):
                        # Verification discovered a better move! Update eval_before
                        eval_before = MCPEval.from_eval(
                            verify_ev,
                            board.fen(),
                            board=board,
                            requested_depth=raw_requested_depth,
                        )
                        score = score_played_move(
                            board,
                            chess_move,
                            eval_before,
                            eval_after,
                            board_after,
                            action_type=action_type,
                        )
                    else:
                        # Played move is confirmed as the best legal attempt.
                        score.move_class = MoveClass.BEST
                        score.effective_loss = 0
                        score.is_best_engine_move = True
                except Exception:
                    # Verification FAILED — leave the original grading intact
                    # and mark the response unverified rather than fabricating
                    # a BEST verdict we cannot prove (audit P1 fix).
                    verification_attempted = True

            best_san: str | None = None
            if score.is_best_engine_move:
                best_san = played_san
            elif eval_before.best_move:
                try:
                    bm = chess.Move.from_uci(eval_before.best_move.lower())
                    if bm in board.legal_moves:
                        best_san = board.san(bm)
                except Exception:
                    pass

            best_line_san = pv_to_san(board, eval_before.pv) if eval_before.pv else best_san
            played_continuation: str | None = None
            if eval_after.pv and not board_after.is_game_over():
                played_continuation = pv_to_san(board_after, eval_after.pv)

            played_line_san = played_san
            if played_continuation:
                played_line_san = f"{played_san} {played_continuation}"

            verified = True
            if (
                action_type == "play_move"
                and score.best_action != "play_move"
                and score.is_best_action
                and not score.action_equivalent
            ):
                verified = False
            if (
                score.effective_loss
                and score.effective_loss > 0
                and (not score.loss_kind or score.loss_kind == "none")
            ):
                verified = False
            # P1 audit fix: verification failure must NOT silently downgrade
            # grading. If we tried to verify but couldn't reach a conclusion,
            # the response must be marked unverified.
            if verification_attempted and score.move_class in (
                MoveClass.MISTAKE,
                MoveClass.BLUNDER,
            ):
                verified = False

            mcp_analysis = MCPMoveAnalysis(
                played=chess_move.uci(),
                played_san=played_san,
                move_class=score.move_class,
                is_engine_best=score.is_best_engine_move,
                centipawn_loss=score.centipawn_loss,
                mate_distance_loss=score.mate_distance_loss,
                raw_centipawn_loss=score.raw_centipawn_loss,
                raw_centipawn_delta=score.raw_centipawn_delta,
                effective_loss=score.effective_loss,
                loss_kind=score.loss_kind,
                engine_cp_loss=score.engine_cp_loss,
                mate_distance_penalty=score.mate_distance_penalty,
                outcome_penalty=score.outcome_penalty,
                rule_action_penalty=score.rule_action_penalty,
                eval_before=eval_before,
                eval_after=eval_after,
                best_move_san=best_san,
                best_line_san=best_line_san,
                best_line_san_truncated=bool(eval_before.pv and len(eval_before.pv) > 6),
                played_line_san=played_line_san,
                played_continuation_san=played_continuation,
                syntax_warning=None,
                action_type=action_type,
                best_action=score.best_action,
                is_best_action=score.is_best_action,
                action_equivalent=score.action_equivalent,
                missed_draw_claim=score.missed_draw_claim,
                conceded_draw_claim=score.conceded_draw_claim,
                claim_reason=score.claim_reason,
                claim_move=score.claim_move,
                can_claim_now=score.can_claim_now,
                can_claim_with_intended_move=score.can_claim_with_intended_move,
                claim_moves=score.claim_moves,
                classification_verified=verified,
            )
            await _cache.set_classify(cache_key, mcp_analysis)
            return mcp_analysis

        res = cast(MCPMoveAnalysis, await _single_flight.do(cache_key, _compute))
        await metrics.record("classify_move", (time.time() - t0) * 1000, cache_hit=False)
        return res.model_copy(update={"syntax_warning": syntax_warn})
    except ToolError:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = "invalid_input"
        if "STRICT" in msg:
            code = "strict_validation_error"
        elif "UNSUPPORTED_VARIANT" in msg:
            code = "unsupported_variant"
        elif "INVALID_FEN" in msg:
            code = "invalid_fen"
        elif "INVALID_POSITION" in msg:
            code = "invalid_position"
        elif "MULTIPLE_GAMES" in msg:
            code = "multiple_games_not_supported"
        elif "ILLEGAL_MOVE" in msg:
            code = "illegal_move"
        elif "AMBIGUOUS_SAN" in msg:
            code = "ambiguous_san"
        elif "GAME_ALREADY_OVER" in msg:
            code = "game_already_over"
        elif "Invalid PGN" in msg or "Could not parse PGN" in msg or "INVALID_PGN" in msg:
            code = "invalid_pgn"
        raise _tool_error(code=code, message=msg, tool="classify_move", input=move) from exc
    except Exception as exc:
        await metrics.record("classify_move", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="classify_move") from exc


def _compute_game_metrics(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
) -> tuple[
    float | None,  # white_acc
    float | None,  # black_acc
    float | None,  # white_acpl
    float | None,  # black_acpl
    float | None,  # white_raw_acpl
    float | None,  # black_raw_acpl
    float | None,  # white_average_effective_loss
    float | None,  # black_average_effective_loss
    tuple[int, int, int],  # white blunders, mistakes, inaccuracies
    tuple[int, int, int],  # black blunders, mistakes, inaccuracies
    list[PlyAnalysisItem],  # turning_points
]:
    """Calculate ACPL, accuracy %, mistakes, and turning points from position evaluations."""
    white_cpls: list[int] = []
    black_cpls: list[int] = []
    white_raw_cpls: list[int] = []
    black_raw_cpls: list[int] = []
    white_eff_losses: list[int] = []
    black_eff_losses: list[int] = []
    white_accs: list[float] = []
    black_accs: list[float] = []
    white_blunders = white_mistakes = white_inaccuracies = 0
    black_blunders = black_mistakes = black_inaccuracies = 0
    turning_points: list[PlyAnalysisItem] = []

    is_game_draw = bool(
        evals
        and (
            positions[-1].is_game_over(claim_draw=False)
            or positions[-1].is_fifty_moves()
            or positions[-1].is_repetition(3)
        )
    )

    for ply_idx, move in enumerate(moves, start=1):
        board_before = positions[ply_idx - 1]
        board_after = positions[ply_idx]
        eval_before = evals[ply_idx - 1]
        eval_after = evals[ply_idx]

        is_white = board_before.turn == chess.WHITE
        move_san = board_before.san(move)

        # Only rewrite the final ply as a procedural draw-claim when:
        #  - the move is genuinely a 50-move or 3-fold claim (NOT an auto-terminal
        #    like 75-move / stalemate / checkmate / locked dead — those are real
        #    moves that LOST the game by blunder, not players taking a draw);
        #  - the move is one of the legal intended-claim moves (a non-resetting,
        #    non-capturing king move for the 50-move rule, or a repetition-completing
        #    move for the threefold rule).
        # Otherwise we score the move as a real play_move so a blunder into an
        # automatic terminal draw (e.g. Qf8+ at halfmove 149) is properly penalized.
        action_type_to_use = "play_move"
        if ply_idx == len(moves) and is_game_draw:
            intended_now = (
                board_after.is_fifty_moves() and not board_after.is_seventyfive_moves()
            ) or board_after.is_repetition(3)
            is_intended_claim = intended_now and (
                not board_after.is_game_over(claim_draw=False) or board_after.can_claim_draw
            )
            if is_intended_claim:
                played_uci = move.uci()
                rule_before = evaluate_rule_status(board_before)
                valid_for_intended = (
                    rule_before.can_claim_with_intended_move
                    and played_uci in rule_before.intended_claim_ucis
                )
                if valid_for_intended:
                    action_type_to_use = "claim_draw_with_intended_move"

        score = score_played_move(
            board_before,
            move,
            eval_before,
            eval_after,
            board_after,
            action_type=action_type_to_use,
        )

        mc = score.move_class.value
        cpl = score.centipawn_loss
        win_loss = score.win_loss
        move_acc = max(0.0, min(100.0, 103.1668 * math.exp(-0.04354 * win_loss) - 3.1669))
        effective_loss = score.effective_loss

        raw_cpl_val = (
            score.centipawn_loss
            if score.centipawn_loss is not None
            else (score.raw_centipawn_loss if score.raw_centipawn_loss is not None else 0)
        )
        if is_white:
            if raw_cpl_val is not None:
                white_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                white_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                white_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                white_eff_losses.append(effective_loss)
            white_accs.append(move_acc)
            if mc == "blunder":
                white_blunders += 1
            elif mc == "mistake":
                white_mistakes += 1
            elif mc == "inaccuracy":
                white_inaccuracies += 1
        else:
            if raw_cpl_val is not None:
                black_cpls.append(raw_cpl_val)
            if score.raw_centipawn_loss is not None:
                black_raw_cpls.append(score.raw_centipawn_loss)
            elif score.centipawn_loss is not None:
                black_raw_cpls.append(score.centipawn_loss)
            if effective_loss is not None:
                black_eff_losses.append(effective_loss)
            black_accs.append(move_acc)
            if mc == "blunder":
                black_blunders += 1
            elif mc == "mistake":
                black_mistakes += 1
            elif mc == "inaccuracy":
                black_inaccuracies += 1

        best_san: str | None = None
        if eval_before.best_move:
            try:
                move_obj = chess.Move.from_uci(eval_before.best_move.lower())
                if move_obj in board_before.legal_moves:
                    best_san = board_before.san(move_obj)
            except (
                ValueError,
                chess.IllegalMoveError,
                chess.InvalidMoveError,
                AssertionError,
            ):
                best_san = None

        if (
            (cpl is not None and cpl >= 150)
            or (effective_loss is not None and effective_loss >= 150)
            or mc in ("blunder", "mistake")
        ):
            turning_points.append(
                PlyAnalysisItem(
                    ply=ply_idx,
                    san=move_san,
                    uci=move.uci(),
                    move_class=mc,
                    centipawn_loss=cpl,
                    effective_loss=effective_loss,
                    loss_kind=score.loss_kind,
                    engine_cp_loss=score.engine_cp_loss,
                    mate_distance_penalty=score.mate_distance_penalty,
                    outcome_penalty=score.outcome_penalty,
                    rule_action_penalty=score.rule_action_penalty,
                    best_move_san=best_san,
                    best_action=score.best_action,
                    missed_draw_claim=score.missed_draw_claim,
                    conceded_draw_claim=score.conceded_draw_claim,
                    claim_reason=score.claim_reason,
                    claim_move=score.claim_move,
                )
            )

    white_acc = round(sum(white_accs) / len(white_accs), 1) if white_accs else None
    black_acc = round(sum(black_accs) / len(black_accs), 1) if black_accs else None
    white_raw_acpl = round(sum(white_raw_cpls) / len(white_raw_cpls), 1) if white_raw_cpls else None
    black_raw_acpl = round(sum(black_raw_cpls) / len(black_raw_cpls), 1) if black_raw_cpls else None
    white_avg_eff = (
        round(sum(white_eff_losses) / len(white_eff_losses), 1) if white_eff_losses else None
    )
    black_avg_eff = (
        round(sum(black_eff_losses) / len(black_eff_losses), 1) if black_eff_losses else None
    )
    white_acpl = white_avg_eff
    black_acpl = black_avg_eff

    top_turning_points = sorted(
        sorted(
            turning_points,
            key=lambda x: 1000 if x.effective_loss is None else x.effective_loss,
            reverse=True,
        )[:8],
        key=lambda x: x.ply,
    )

    return (
        white_acc,
        black_acc,
        white_acpl,
        black_acpl,
        white_raw_acpl,
        black_raw_acpl,
        white_avg_eff,
        black_avg_eff,
        (white_blunders, white_mistakes, white_inaccuracies),
        (black_blunders, black_mistakes, black_inaccuracies),
        top_turning_points,
    )


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def analyze_game(
    pgn: str,
    depth: int = 14,
    strict: bool = False,
    ctx: Context | None = None,
) -> GameAnalysisResult:
    """Analyze a full game in PGN format with Stockfish, providing accuracy scores, mistake counts, and metadata.

    Supports standard PGN, annotated PGNs (with comments, NAGs, variations), conversational
    preamble/trailer text, markdown-wrapped PGNs, and bare move lists. Side variations in parentheses
    and comments are ignored for the mainline analysis. `white_acpl` / `black_acpl` report the effective
    ACPL across all plies (including 1000cp mate transitions and draw claim forfeitures), while
    `white_raw_acpl` / `black_raw_acpl` report unweighted raw CPL on non-mate plies.

    Args:
        pgn: PGN string, annotated game, or move text.
        depth: Search depth per move (default 14, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax, move number mismatches, or metadata discrepancies (default False).

    Returns:
        GameAnalysisResult with player accuracy %, ACPL, blunder/mistake counts, turning points, and game metadata.
    """
    t0 = time.time()
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        _check_multiple_games(pgn)
        canonical_pgn = _extract_canonical_pgn_text(pgn)
        _check_multiple_games(canonical_pgn)
        game = _extract_game_inner(canonical_pgn)

        positions: list[chess.Board] = []
        moves: list[chess.Move] = []
        syntax_warnings: list[str] = []
        curr_board = game.board()
        if not curr_board.is_valid() or curr_board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{curr_board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(curr_board.status())})."
            )

        positions.append(curr_board.copy(stack=True))
        auto_termination: str | None = None
        reached_terminal = False
        ignored_trailing_plies = 0

        # Extract headers ONLY from contiguous header block at the start of canonical_pgn
        header_end = 0
        first_header = TAG_PAIR_REGEX.search(canonical_pgn)
        first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", canonical_pgn)
        if first_header and (not first_mv or first_header.start() < first_mv.start()):
            header_end = first_header.end()
            for m in TAG_PAIR_REGEX.finditer(canonical_pgn):
                if m.start() < header_end:
                    continue
                if canonical_pgn[header_end : m.start()].strip() == "":
                    header_end = m.end()
                else:
                    break

        header_section = canonical_pgn[:header_end]
        movetext_section = canonical_pgn[header_end:]

        # Clean movetext for token scanning (strip comments and variations, translate figurines, split attached NAGs)
        movetext_section = re.sub(
            r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*)(\$\d+)",
            r"\1 \2",
            movetext_section,
        )
        movetext_section = re.sub(r"\b(O-O-O|O-O)([\+#\?!]*)(\$\d+)", r"\1\2 \3", movetext_section)
        cleaned_movetext = _normalize_movetext_figurines(movetext_section)
        while "{" in cleaned_movetext and "}" in cleaned_movetext:
            prev = cleaned_movetext
            cleaned_movetext = re.sub(r"\{[^{}]*\}", " ", cleaned_movetext, flags=re.DOTALL)
            if cleaned_movetext == prev:
                break
        cleaned_movetext = re.sub(r";[^\r\n]*", " ", cleaned_movetext)
        while "(" in cleaned_movetext and ")" in cleaned_movetext:
            prev = cleaned_movetext
            cleaned_movetext = re.sub(r"\([^()]*\)", " ", cleaned_movetext, flags=re.DOTALL)
            if cleaned_movetext == prev:
                break

        movetext_tokens = cleaned_movetext.split()
        tok_idx = 0
        expected_fullmove = curr_board.fullmove_number

        # Skip leading non-chess tokens to align with the first actual move (PGN-01)
        for i, tok in enumerate(movetext_tokens):
            clean_tok = tok.strip(".,;:!?")
            num_m = re.match(r"^(\d+)[\.\:]*$", clean_tok)
            if num_m:
                tok_idx = i
                break
            try:
                curr_board.parse_san(clean_tok)
                tok_idx = i
                break
            except Exception:
                continue

        for node in game.mainline():
            if reached_terminal:
                ignored_trailing_plies += 1
                continue

            move = node.move
            if move not in curr_board.legal_moves:
                ignored_trailing_plies += 1
                reached_terminal = True
                continue

            canonical_san = curr_board.san(move)

            # Advance token index through move number tokens or result tokens
            while tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
                num_m = re.match(r"^(\d+)[\.\:]*$", raw_tok)
                if num_m:
                    move_num = int(num_m.group(1))
                    if move_num != expected_fullmove:
                        syntax_warnings.append(
                            f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                        )
                    tok_idx += 1
                    continue
                if raw_tok in ("1-0", "0-1", "1/2-1/2", "*") or re.match(r"^\$[0-9]+$", raw_tok):
                    tok_idx += 1
                    continue
                break

            if tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx].strip(".,;:!?")
                raw_tok_san = raw_tok.rstrip("!?")
                if raw_tok_san != canonical_san and not re.fullmatch(
                    r"[a-h][1-8][a-h][1-8][qrbn]?", raw_tok_san.lower()
                ):
                    syntax_warnings.append(
                        f"Input SAN '{movetext_tokens[tok_idx]}' normalized to '{canonical_san}'"
                    )
                tok_idx += 1

            moves.append(move)
            curr_board.push(move)
            positions.append(curr_board.copy(stack=True))
            if curr_board.turn == chess.WHITE:
                expected_fullmove += 1

            rule_after = evaluate_rule_status(curr_board)
            if rule_after.terminal is not None:
                reached_terminal = True
                auto_termination = rule_after.terminal

        # Extract headers with TAG_PAIR_REGEX from header_section to handle escaped quotes and robust tag parsing
        tags_dict: dict[str, str] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1)
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k not in tags_dict and tag_v is not None and tag_v != "?":
                tags_dict[tag_k] = tag_v

        h = game.headers
        white_name = tags_dict.get("White") or (
            _unescape_pgn_tag_value(h.get("White"))
            if h.get("White") and h.get("White") != "?"
            else None
        )
        black_name = tags_dict.get("Black") or (
            _unescape_pgn_tag_value(h.get("Black"))
            if h.get("Black") and h.get("Black") != "?"
            else None
        )
        event_name = tags_dict.get("Event") or (
            _unescape_pgn_tag_value(h.get("Event"))
            if h.get("Event") and h.get("Event") != "?"
            else None
        )
        site_name = tags_dict.get("Site") or (
            _unescape_pgn_tag_value(h.get("Site"))
            if h.get("Site") and h.get("Site") != "?"
            else None
        )
        round_name = tags_dict.get("Round") or (
            _unescape_pgn_tag_value(h.get("Round"))
            if h.get("Round") and h.get("Round") != "?"
            else None
        )
        white_elo_val = tags_dict.get("WhiteElo") or (
            h.get("WhiteElo") if h.get("WhiteElo") and h.get("WhiteElo") != "?" else None
        )
        black_elo_val = tags_dict.get("BlackElo") or (
            h.get("BlackElo") if h.get("BlackElo") and h.get("BlackElo") != "?" else None
        )
        time_control_val = tags_dict.get("TimeControl") or (
            h.get("TimeControl") if h.get("TimeControl") and h.get("TimeControl") != "?" else None
        )
        variant_val = tags_dict.get("Variant") or (
            h.get("Variant") if h.get("Variant") and h.get("Variant") != "?" else None
        )
        date_val = (
            tags_dict.get("Date") or tags_dict.get("UTCDate") or h.get("Date") or h.get("UTCDate")
        )
        if date_val in ("????.??.??", "?"):
            date_val = None

        result_movetext = _find_movetext_result(canonical_pgn)

        # Extract Result and Termination headers from header_section ONLY
        result_header_raw: str | None = None
        termination_header_val: str | None = None
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k == "result" and result_header_raw is None:
                result_header_raw = tag_v
            elif tag_k == "termination" and termination_header_val is None:
                termination_header_val = tag_v

        metadata_warnings: list[str] = []

        CANONICAL_RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}
        if result_header_raw is not None and result_header_raw != "?":
            if result_header_raw in CANONICAL_RESULTS:
                result_header = result_header_raw
            else:
                metadata_warnings.append(
                    f"Invalid Result header tag '{result_header_raw}'; expected 1-0, 0-1, 1/2-1/2, or *."
                )
                result_header = None
        else:
            result_header = None

        if white_elo_val is not None and white_elo_val != "-":
            if not (white_elo_val.isdigit() and 0 <= int(white_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid WhiteElo header tag '{white_elo_val}'; expected numeric integer rating."
                )
        if black_elo_val is not None and black_elo_val != "-":
            if not (black_elo_val.isdigit() and 0 <= int(black_elo_val) <= 4000):
                metadata_warnings.append(
                    f"Invalid BlackElo header tag '{black_elo_val}'; expected numeric integer rating."
                )
        if time_control_val is not None and time_control_val not in ("-", "?"):
            if not re.match(r"^\d+(?:\+\d+)?$|^\d+/\d+$", time_control_val):
                metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")

        eco_header = tags_dict.get("ECO") or h.get("ECO")
        opening_header = tags_dict.get("Opening") or h.get("Opening")

        # Detect duplicate headers in header block only
        tag_counts: dict[str, int] = {}
        for tag_name, _ in TAG_PAIR_REGEX.findall(header_section):
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
        for tag_name, count in tag_counts.items():
            if count > 1:
                metadata_warnings.append(
                    f"Duplicate PGN tag '[{tag_name}]' detected ({count} occurrences); using canonical tag value."
                )

        # Validate SetUp vs FEN tags
        setup_header = h.get("SetUp")
        fen_header = h.get("FEN")
        if setup_header == "1" and not fen_header:
            metadata_warnings.append(
                '[SetUp "1"] tag provided without FEN tag; defaulting to standard starting position.'
            )
        elif fen_header and setup_header != "1":
            metadata_warnings.append(
                'FEN tag provided without [SetUp "1"]; custom position loaded.'
            )

        if game.errors:
            ignored_trailing_plies += len(game.errors)

        raw_pgn_clean = _strip_pgn_escape_lines(canonical_pgn)
        raw_truncated = _truncate_movetext_at_result(raw_pgn_clean)
        if len(raw_truncated) < len(raw_pgn_clean):
            after_part = raw_pgn_clean[len(raw_truncated) :]
            after_clean = re.sub(r"\{[^{}]*\}", " ", after_part)
            after_clean = re.sub(r";[^\r\n]*", " ", after_clean)
            tokens_after = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                after_clean,
            )
            if tokens_after:
                ignored_trailing_plies += len(tokens_after)

        if ignored_trailing_plies > 0:
            ply_word = "ply" if ignored_trailing_plies == 1 else "plies"
            metadata_warnings.append(
                f"Movetext contained moves after game termination; ignored {ignored_trailing_plies} trailing {ply_word}."
            )

        # Validate game result consistency against final board state
        final_board = positions[-1]
        rule_final = evaluate_rule_status(final_board)
        result_board: str | None = None
        if rule_final.terminal is not None:
            if rule_final.terminal == "checkmate":
                result_board = "1-0" if final_board.turn == chess.BLACK else "0-1"
                auto_termination = "checkmate"
            else:
                result_board = "1/2-1/2"
                auto_termination = rule_final.terminal

        if result_board is not None:
            result_val = result_board
            if result_header and result_header not in ("*", "?") and result_header != result_board:
                metadata_warnings.append(
                    f"Result header '{result_header}' disagrees with board outcome '{result_board}'; using board outcome."
                )
            if (
                result_movetext
                and result_movetext not in ("*", "?")
                and result_movetext != result_board
            ):
                metadata_warnings.append(
                    f"Movetext result '{result_movetext}' disagrees with board outcome '{result_board}'; using board outcome."
                )
        else:
            if result_header_raw and result_movetext and result_header_raw != result_movetext:
                metadata_warnings.append(
                    f"Result header '{result_header_raw}' disagrees with movetext result '{result_movetext}'."
                )

            if result_header and result_header not in ("*", "?"):
                result_val = result_header
            elif result_movetext and result_movetext not in ("*", "?"):
                result_val = result_movetext
            else:
                result_val = result_header or result_movetext or "*"

        # Infer result from termination header if result is unstated ('*')
        if (result_val == "*" or result_val is None) and termination_header_val:
            term_low = termination_header_val.lower()
            if "white resign" in term_low or ("resign" in term_low and "white" in term_low):
                result_val = "0-1"
            elif "black resign" in term_low or ("resign" in term_low and "black" in term_low):
                result_val = "1-0"
            elif (
                "white time" in term_low
                or "white flag" in term_low
                or ("time" in term_low and "white" in term_low)
            ):
                result_val = "0-1"
            elif (
                "black time" in term_low
                or "black flag" in term_low
                or ("time" in term_low and "black" in term_low)
            ):
                result_val = "1-0"
            elif "white lost" in term_low or (
                "white" in term_low and ("illegal" in term_low or "infraction" in term_low)
            ):
                result_val = "0-1"
            elif "black lost" in term_low or (
                "black" in term_low and ("illegal" in term_low or "infraction" in term_low)
            ):
                result_val = "1-0"

        # Validate Resignation & Time Forfeit & Rules Infraction under FIDE mating possibility rules
        result_val, mate_warnings = validate_mating_possibility(
            final_board, result_val, termination_header_val
        )
        metadata_warnings.extend(mate_warnings)

        # Check for contradictory metadata
        if termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            if norm_term in (
                "stalemate",
                "insufficient_material",
                "fifty_moves",
                "seventyfive_moves",
                "threefold_repetition",
                "fivefold_repetition",
                "dead_position",
            ) and result_val in ("1-0", "0-1"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "checkmate" and result_val in ("1/2-1/2", "*"):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )
            elif norm_term == "unterminated" and result_val in (
                "1-0",
                "0-1",
                "1/2-1/2",
            ):
                metadata_warnings.append(
                    f"Contradictory PGN metadata: Termination '{termination_header_val}' contradicts Result '{result_val}'."
                )

        # Premature draw agreement warning
        if (
            termination_header_val
            and "agreement" in termination_header_val.lower()
            and len(moves) < 2
        ):
            metadata_warnings.append(
                "Draw agreement declared before both players completed at least one move."
            )

        if auto_termination is not None:
            termination_val = auto_termination
            if termination_header_val:
                norm_term_hdr = normalize_termination(termination_header_val)
                if norm_term_hdr == "normal":
                    pass
                else:
                    is_concurrent_draw = norm_term_hdr in (
                        "stalemate",
                        "seventyfive_moves",
                        "fivefold_repetition",
                        "insufficient_material",
                        "fifty_moves",
                        "threefold_repetition",
                        "dead_position",
                    ) and (
                        (norm_term_hdr == "threefold_repetition" and final_board.is_repetition(3))
                        or (
                            norm_term_hdr == "fifty_moves"
                            and (final_board.is_fifty_moves() or final_board.halfmove_clock >= 100)
                        )
                        or (
                            norm_term_hdr == "fivefold_repetition"
                            and final_board.is_fivefold_repetition()
                        )
                        or (
                            norm_term_hdr == "seventyfive_moves"
                            and final_board.is_seventyfive_moves()
                        )
                        or (
                            norm_term_hdr == "insufficient_material"
                            and final_board.is_insufficient_material()
                        )
                        or (
                            norm_term_hdr == "dead_position"
                            and is_locked_dead_position(final_board)
                        )
                        or (norm_term_hdr == "stalemate" and final_board.is_stalemate())
                    )
                    if norm_term_hdr != auto_termination and not is_concurrent_draw:
                        metadata_warnings.append(
                            f"Termination header '{termination_header_val}' disagrees with board outcome '{auto_termination}'; using board outcome."
                        )
        elif termination_header_val:
            norm_term_hdr = normalize_termination(termination_header_val)
            if norm_term_hdr == "normal":
                termination_val = "normal"
            elif norm_term_hdr in ("checkmate", "stalemate"):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            elif norm_term_hdr == "threefold_repetition":
                if not final_board.is_repetition(3):
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not threefold_repetition)."
                    )
                    termination_val = None
                else:
                    termination_val = "threefold_repetition"
            elif norm_term_hdr == "fifty_moves":
                if not final_board.is_fifty_moves() and final_board.halfmove_clock < 100:
                    metadata_warnings.append(
                        f"Termination header '{termination_header_val}' contradicts board state (position is not fifty_moves)."
                    )
                    termination_val = None
                else:
                    termination_val = "fifty_moves"
            elif norm_term_hdr in (
                "insufficient_material",
                "seventyfive_moves",
                "fivefold_repetition",
                "dead_position",
            ):
                metadata_warnings.append(
                    f"Termination header '{termination_header_val}' contradicts board state (position is not {norm_term_hdr})."
                )
                termination_val = None
            else:
                termination_val = norm_term_hdr
        else:
            termination_val = None

        is_standard_start = game.board().fen() == chess.STARTING_FEN

        pool = await _get_analyzer_pool(ctx)
        engine_name_str = getattr(pool, "engine_version", getattr(pool, "name", "Stockfish"))

        if not moves:
            detected_opening, detected_eco = (
                lookup_opening([])[:2] if is_standard_start else (None, None)
            )
            return GameAnalysisResult(
                total_plies=0,
                white_accuracy=None,
                black_accuracy=None,
                white_acpl=None,
                black_acpl=None,
                white_raw_acpl=None,
                black_raw_acpl=None,
                white_effective_acpl=None,
                black_effective_acpl=None,
                white_average_effective_loss=None,
                black_average_effective_loss=None,
                white_blunders=0,
                white_mistakes=0,
                white_inaccuracies=0,
                black_blunders=0,
                black_mistakes=0,
                black_inaccuracies=0,
                turning_points=[],
                white=white_name,
                black=black_name,
                event=event_name,
                site=site_name,
                date=date_val,
                round=round_name,
                result=result_val or result_header or "*",
                result_header=result_header,
                result_header_raw=result_header_raw,
                result_movetext=result_movetext,
                result_inferred=result_board,
                white_elo=white_elo_val,
                black_elo=black_elo_val,
                time_control=time_control_val,
                variant=variant_val,
                eco=detected_eco or eco_header,
                opening=detected_opening or opening_header,
                opening_header=opening_header,
                eco_header=eco_header,
                metadata_warnings=metadata_warnings,
                syntax_warnings=syntax_warnings,
                termination=termination_val,
                termination_header=termination_header_val,
                requested_depth=raw_requested_depth,
                searched_depth=0,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                accuracy_method="win_probability_logistic",
                mate_penalty_policy="1000_cp_mate_transition",
            )

        eval_pairs = await _gather_evaluate_positions_bounded(
            positions, depth, pool, requested_depth=raw_requested_depth
        )
        evals: list[MCPEval] = [ep[0] for ep in eval_pairs]
        all_cached = all(ep[1] for ep in eval_pairs)

        (
            white_acc,
            black_acc,
            white_acpl,
            black_acpl,
            white_raw_acpl,
            black_raw_acpl,
            white_avg_eff,
            black_avg_eff,
            (white_blunders, white_mistakes, white_inaccuracies),
            (black_blunders, black_mistakes, black_inaccuracies),
            top_turning_points,
        ) = _compute_game_metrics(positions, moves, evals)

        await metrics.record("analyze_game", (time.time() - t0) * 1000, cache_hit=all_cached)

        uci_moves = [m.uci() for m in moves]
        if is_standard_start:
            detected_opening, detected_eco, _ = lookup_opening(uci_moves)
        else:
            detected_opening, detected_eco = None, None

        final_opening = detected_opening or opening_header
        final_eco = detected_eco or eco_header

        if detected_opening and opening_header:
            det_clean = detected_opening.strip().lower()
            hdr_clean = opening_header.strip().lower()
            det_base = det_clean.split(":")[0].strip()
            hdr_base = hdr_clean.split(":")[0].strip()
            is_parent_child = (
                det_clean.startswith(hdr_clean)
                or hdr_clean.startswith(det_clean)
                or det_base == hdr_base
            )
            if not is_parent_child:
                metadata_warnings.append(
                    f"Opening header '{opening_header}' disagrees with detected opening '{detected_opening}'"
                )
        if (
            detected_eco
            and eco_header
            and detected_eco.strip().upper() != eco_header.strip().upper()
        ):
            metadata_warnings.append(
                f"ECO header '{eco_header}' disagrees with detected ECO '{detected_eco}'"
            )

        if strict:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

        return GameAnalysisResult(
            total_plies=len(moves),
            white_accuracy=white_acc,
            black_accuracy=black_acc,
            white_acpl=white_acpl,
            black_acpl=black_acpl,
            white_raw_acpl=white_raw_acpl,
            black_raw_acpl=black_raw_acpl,
            white_effective_acpl=white_avg_eff,
            black_effective_acpl=black_avg_eff,
            white_average_effective_loss=white_avg_eff,
            black_average_effective_loss=black_avg_eff,
            white_blunders=white_blunders,
            white_mistakes=white_mistakes,
            white_inaccuracies=white_inaccuracies,
            black_blunders=black_blunders,
            black_mistakes=black_mistakes,
            black_inaccuracies=black_inaccuracies,
            turning_points=top_turning_points,
            white=white_name,
            black=black_name,
            event=event_name,
            site=site_name,
            date=date_val,
            round=round_name,
            result=result_val,
            result_header=result_header,
            result_header_raw=result_header_raw,
            result_movetext=result_movetext,
            result_inferred=result_board
            or (
                result_val
                if (result_header_raw in ("*", None) and result_val in ("1-0", "0-1", "1/2-1/2"))
                else None
            ),
            white_elo=white_elo_val,
            black_elo=black_elo_val,
            time_control=time_control_val,
            variant=variant_val,
            eco=final_eco,
            opening=final_opening,
            opening_header=opening_header,
            eco_header=eco_header,
            metadata_warnings=metadata_warnings,
            syntax_warnings=syntax_warnings,
            termination=termination_val,
            termination_header=termination_header_val,
            requested_depth=raw_requested_depth,
            searched_depth=depth,
            engine="Stockfish",
            engine_version=engine_name_str,
            **_build_identity(pool),
            accuracy_method="win_probability_logistic",
            mate_penalty_policy="1000_cp_mate_transition",
        )
    except ToolError:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = "invalid_input"
        if "STRICT" in msg:
            code = "strict_validation_error"
        elif "UNSUPPORTED_VARIANT" in msg:
            code = "unsupported_variant"
        elif "INVALID_FEN" in msg:
            code = "invalid_fen"
        elif "INVALID_POSITION" in msg:
            code = "invalid_position"
        elif "MULTIPLE_GAMES" in msg:
            code = "multiple_games_not_supported"
        elif "ILLEGAL_MOVE" in msg:
            code = "illegal_move"
        elif "AMBIGUOUS_SAN" in msg:
            code = "ambiguous_san"
        elif "GAME_ALREADY_OVER" in msg:
            code = "game_already_over"
        elif "Invalid PGN" in msg or "Could not parse PGN" in msg or "INVALID_PGN" in msg:
            code = "invalid_pgn"
        raise _tool_error(code=code, message=msg, tool="analyze_game", input=pgn[:100]) from exc
    except Exception as exc:
        await metrics.record("analyze_game", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="analyze_game") from exc


class TokenBucketRateLimiter:
    """In-memory thread-safe token bucket rate limiter per client IP."""

    def __init__(self, rate: float = 5.0, capacity: float = 200.0) -> None:
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str, cost: float = 1.0) -> bool:
        now = time.time()
        async with self._lock:
            if len(self._buckets) > 10_000:
                cutoff = now - 3600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
                if len(self._buckets) > 10_000:
                    sorted_items = sorted(
                        self._buckets.items(), key=lambda item: item[1][1], reverse=True
                    )[:5000]
                    self._buckets = dict(sorted_items)

            tokens, last_time = self._buckets.get(client_id, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last_time) * self.rate)

            if tokens >= cost:
                self._buckets[client_id] = (tokens - cost, now)
                return True
            else:
                self._buckets[client_id] = (tokens, now)
                return False


class ASGIRequestLoggerMiddleware:
    """ASGI middleware to log all incoming MCP requests, rate limit per client, and filter clients if configured."""

    def __init__(
        self,
        app: ASGIApp,
        restrict_to_chatgpt: bool = False,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self.app = app
        self.restrict_to_chatgpt = restrict_to_chatgpt
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            raw_headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
            headers_dict: dict[bytes, bytes] = dict(raw_headers)
            ua = headers_dict.get(b"user-agent", b"").decode("utf-8", "ignore")
            origin = headers_dict.get(b"origin", b"").decode("utf-8", "ignore")
            host = headers_dict.get(b"host", b"").decode("utf-8", "ignore")
            path = str(scope.get("path", ""))
            method = str(scope.get("method", "")).upper()
            session_id = headers_dict.get(b"mcp-session-id", b"").decode("utf-8", "ignore")

            client_tuple = cast(tuple[str, int] | None, scope.get("client"))
            fallback_ip = client_tuple[0].strip("[]") if client_tuple else "127.0.0.1"
            ip = headers_dict.get(b"x-forwarded-for", b"").decode("utf-8", "ignore") or fallback_ip
            client_ip = ip.split(",")[0].strip().strip("[]")

            if method == "OPTIONS":
                await send(
                    cast(
                        Message,
                        {
                            "type": "http.response.start",
                            "status": 200,
                            "headers": [
                                (b"access-control-allow-origin", b"*"),
                                (
                                    b"access-control-allow-methods",
                                    b"GET, POST, OPTIONS",
                                ),
                                (b"access-control-allow-headers", b"*"),
                                (b"access-control-max-age", b"86400"),
                                (b"content-length", b"0"),
                            ],
                        },
                    )
                )
                await send(cast(Message, {"type": "http.response.body", "body": b""}))
                return

            # 1. Rate Limiting Check
            if not await self.rate_limiter.is_allowed(client_ip):
                log.warning("Rate limit exceeded for ip=%s path=%s", client_ip, path)
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Rate limit exceeded. Please slow down."}}\n'
                await send(
                    cast(
                        Message,
                        {
                            "type": "http.response.start",
                            "status": 429,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (
                                    b"content-length",
                                    str(len(response_body)).encode("ascii"),
                                ),
                                (b"retry-after", b"2"),
                            ],
                        },
                    )
                )
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

            log.info(
                "%s %s host=%s ip=%s session=%s origin=%s ua=%s",
                method,
                path,
                host,
                client_ip,
                session_id,
                origin,
                ua,
            )

            # /health is a public probe — exempt from the ChatGPT lock so
            # compose / orchestrator healthchecks can succeed without
            # carrying the auth token.
            if path == "/health":
                await self.app(scope, receive, send)
                return

            if self.restrict_to_chatgpt:
                # P1 audit fix: this used to whitelist every client whose
                # UA contained "curl", "python", "pydantic", or "chessy",
                # plus an empty UA — defeating the entire lock. Real auth
                # must come from a token (or from Cloudflare Access in
                # front of the deployment). When CHESS_MCP_LOCK_CHATGPT=true
                # but no auth token is configured, the lock degrades to a
                # *warning* rather than a passthrough: we still serve, but
                # we log every request and emit a clear admin-only warning
                # at startup so the operator knows auth is effectively off.
                ua_lower = ua.lower()
                origin_lower = origin.lower()
                is_known_chatgpt = (
                    "chatgpt" in ua_lower
                    or "openai" in ua_lower
                    or "chatgpt" in origin_lower
                    or "openai" in origin_lower
                )
                # Without an auth token configured, the lock is no-op
                # backwards-compatible; with one, only matching requests pass.
                # The lock is meant to be combined with an upstream auth
                # boundary (Cloudflare Access token, mTLS, etc.) — it is
                # not, and never was, a primary security control.
                from .config import get_mcp_settings

                mcp_cfg = get_mcp_settings()
                auth_token = mcp_cfg.auth_token
                provided_token = (
                    headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
                )
                has_valid_token = bool(auth_token) and provided_token == auth_token

                # The chessy app (the only consumer in production now) connects
                # via `X-Chessy-Auth`. Public callers must be the known
                # ChatGPT / OpenAI MCP clients, or carry the auth token.
                # Internal-network whitelisting is gone — chessy is no longer
                # in the same Docker network as MCP, so its requests arrive
                # on the public IP and MUST use the token.
                from .config import get_mcp_settings as _get_cfg

                _cfg = _get_cfg()
                if not _cfg.auth_token:
                    log.warning(
                        "CHESS_MCP_LOCK_CHATGPT=true but no CHESS_MCP_AUTH_TOKEN set — "
                        "lock degrades to a warning; configure a token before exposing publicly."
                    )

                is_allowed = is_known_chatgpt or has_valid_token

                if not is_allowed:
                    log.warning(
                        "Blocked client: ua=%r origin=%r chatgpt=%s valid_token=%s",
                        ua,
                        origin,
                        is_known_chatgpt,
                        has_valid_token,
                    )
                    response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: ChatGPT lock active; configure CHESS_MCP_AUTH_TOKEN or upstream auth"}}\n'
                    await send(
                        cast(
                            Message,
                            {
                                "type": "http.response.start",
                                "status": 403,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (
                                        b"content-length",
                                        str(len(response_body)).encode("ascii"),
                                    ),
                                ],
                            },
                        )
                    )
                    await send(
                        cast(
                            Message,
                            {"type": "http.response.body", "body": response_body},
                        )
                    )
                    return

        await self.app(scope, receive, send)


def _build_app(restrict_chatgpt: bool) -> ASGIApp:
    """Compose the ASGI middleware stack around the FastMCP streamable-HTTP app.

    Order matters: outermost runs first on the way in, last on the way out.
    """
    from starlette.middleware.gzip import GZipMiddleware

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    base = mcp.streamable_http_app(transport_security=security)
    # GZip is innermost so the JSON-RPC envelope sees the uncompressed payload;
    # analyze_game payloads (multi-MB for long PGNs) benefit ~6× on the wire.
    gzipped = GZipMiddleware(base, minimum_size=1024, compresslevel=5)
    return ASGIRequestLoggerMiddleware(gzipped, restrict_to_chatgpt=restrict_chatgpt)


def main() -> None:
    from .config import get_mcp_settings

    mcp_cfg = get_mcp_settings()
    transport = mcp_cfg.transport
    if transport == "streamable-http":
        host = mcp_cfg.http_host
        port = mcp_cfg.http_port
        restrict_chatgpt = mcp_cfg.lock_chatgpt
        wrapped_app = _build_app(restrict_chatgpt)
        import uvicorn

        # Single async worker is correct here: uvicorn is asyncio-native,
        # every tool is async, and the Stockfish pool is already sized to
        # the host's CPU count. Spinning more workers would just contend
        # on the engine queue without adding throughput.
        uvicorn.run(
            wrapped_app,
            host=host,
            port=port,
            log_level="info",
            loop="uvloop",
            http="httptools",
            access_log=False,  # ASGIRequestLoggerMiddleware logs requests itself
            timeout_keep_alive=75,
            h11_max_incomplete_event_size=32 * 1024 * 1024,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
