from __future__ import annotations

import asyncio
import hmac
import io
import ipaddress
import json
import logging
import math
import os
import re
import subprocess
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, cast

import chess
import chess.pgn
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from core.engines.analyzer import pv_to_san
from core.engines.openings import lookup_opening
from core.engines.pool import AnalyzerPool
from core.engines.types import Eval, MoveClass
from mcp_server.actions import build_played_action
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
from mcp_server.config import MCPSettings
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
    choose_recommended_action,
    evaluate_rule_status,
    format_fen_status_errors,
    is_locked_dead_position,
    is_terminal_position,
    validate_mating_possibility,
)
from mcp_server.tcp_analyzer import TCPAnalyzerPool
from mcp_server.urls import lichess_urls


log = logging.getLogger("chessy_mcp.server")


def _format_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        group = cast(BaseExceptionGroup[BaseException], exc)
        sub_msgs = [_format_exception(e) for e in group.exceptions]
        return "; ".join(sub_msgs) if sub_msgs else str(group)
    return str(exc)


def _tool_error(code: str, message: str | BaseException, tool: str, **kwargs: Any) -> ToolError:
    """Create a clean human/agent-readable ToolError payload."""
    raw = _format_exception(message) if isinstance(message, BaseException) else str(message)
    clean_msg = raw.strip()
    clean_msg = re.sub(r"^(?:\[[A-Za-z0-9_]+\]|[A-Za-z0-9_]+:)\s*", "", clean_msg).strip()
    return ToolError(f"[{code.upper()}] {clean_msg}")


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
    """Normalize verbosity and reject unknown values instead of silently changing semantics."""
    if value is None:
        return VERBOSITY_FULL
    normalized = str(value).strip().lower()
    resolved = _VERBOSITY_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"INVALID_VERBOSITY: expected one of {sorted(_VERBOSITY_ALIASES)}, got {value!r}"
        )
    return resolved


def _compact_mcpeval(mcp_eval: MCPEval) -> MCPEval:
    """Strip verbose payload duplication without rewriting chess semantics."""
    return mcp_eval.model_copy(
        update={
            "lichess_url": None,
            "lichess_image": None,
            "decision_value": None,
            "engine_eval": None,
            "input_fen": None,
        }
    )


def _force_draw_outcome(mcp_eval: MCPEval) -> MCPEval:
    """Project an MCPEval onto the post-claim state.

    Audit B-02/B-03: classifying a draw claim must not let any dummy move
    leak into `eval_after`. A granted claim always terminates the game as a
    draw (cp=0, no mate, outcome="draw"), so we force that projection here
    regardless of what the engine reported for the (irrelevant) board state.

    U-04 (2026-09-01): the previous projection only zeroed cp/mate/status
    and decision_value, leaving the rest of the eval as Stockfish's view of
    the pre-claim position. That produced self-contradictory fields like
    `best_move=Qc8#` alongside `status=draw, cp=0` on the same eval_after
    object — a downstream consumer reading `eval_after.status == "draw"`
    then enumerating `eval_after.pv` would think it had an executable PV
    on a terminal position. The fix forces EVERY active-state field to a
    pure-terminal value (best_move=None, pv=[], can_claim_*=False,
    recommended_action="game_over", executable_move=None). The pre-claim
    engine state is preserved separately on `eval_before` so callers can
    still inspect what the engine saw before the claim.
    """
    forced_decision = {
        "outcome": "draw",
        "cp_equivalent": 0,
        "best_action": mcp_eval.decision_value.get("best_action", "claim_draw")
        if mcp_eval.decision_value
        else "claim_draw",
        "perspective": "white",
    }
    return mcp_eval.model_copy(
        update={
            # Score — terminal draw.
            "cp": 0,
            "mate": None,
            "status": "draw",
            "decision_value": forced_decision,
            # Engine activity — gone after a granted claim.
            "best_move": None,
            "executable_move": None,
            "pv": [],
            "wdl": None,
            "wdl_pct": None,
            # Rule-action fields — no further claims are possible.
            "can_claim_draw": False,
            "claim_reasons": [],
            "claim_move": None,
            "claim_move_san": None,
            "claim_move_uci": None,
            "can_claim_now": False,
            "claim_reasons_now": [],
            "can_claim_with_intended_move": False,
            "claim_moves": [],
            # Best-action surface — the game is over.
            "recommended_action": "game_over",
            "best_action": "game_over",
            "best_action_type": "game_over",
            "best_action_obj": {
                "type": "game_over",
                "outcome": "draw",
                "reason": "draw_claim",
            },
            "legal_actions": [],
            # Post-state fields — there is no meaningful post-state.
            "post_terminal_status": "draw",
            "post_can_claim_draw": False,
            "post_can_claim_now": False,
            "post_claim_reasons": [],
            "post_claim_moves": [],
            "post_position": {
                "status": "draw",
                "winner": None,
                "can_claim_now": False,
                "can_claim_draw": False,
                "claim_reasons": [],
                "recommended_action": "game_over",
            },
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
    env_sha = os.environ.get("BUILD_SHA") or os.environ.get("CHESSY_BUILD_SHA")
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
async def _mcp_lifespan(server: MCPServer) -> AsyncGenerator[dict[str, Any]]:
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
    pool: AnalyzerPool | TCPAnalyzerPool = await _create_analyzer_pool(cfg, pool_size=pool_size)

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
        except Exception as exc:
            log.warning("pool_stats log iteration failed (continuing): %s", exc)


async def _ponder_warm_cache(
    pool: AnalyzerPool | TCPAnalyzerPool,
    predicted_board: chess.Board,
    depth: int,
    history_complete: str,
) -> None:
    """Background cache warmer that preserves the predicted board's move stack."""
    try:
        board = predicted_board.copy(stack=True)
        if board.is_game_over(claim_draw=False):
            return
        ckey = eval_cache_key(
            board,
            depth,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )
        if (await _cache.get_eval(ckey)) is not None:
            return
        await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
    except Exception as exc:
        log.debug("ponder pre-eval failed: %s", exc)


_background_tasks: set[asyncio.Task[Any]] = set()


def _maybe_ponder_warm(
    pool: AnalyzerPool | TCPAnalyzerPool,
    board: chess.Board,
    best_move_uci: str | None,
    depth: int,
    ponder_enabled: bool,
    history_complete: str,
) -> None:
    """Schedule a provenance-preserving background pre-evaluation."""
    if not ponder_enabled or not best_move_uci:
        return
    try:
        next_board = board.copy(stack=True)
        next_board.push_uci(best_move_uci)
        if next_board.is_game_over(claim_draw=False):
            return
        task = asyncio.create_task(
            _ponder_warm_cache(pool, next_board, depth, history_complete),
            name="ponder-warm",
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        pass


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
    history_complete: str = "complete",
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

    # Contiguous partition: positions [0:chunk] -> slice 0, [chunk:2*chunk] -> slice 1, ...
    # Adjacent plies land on the same worker, so Stockfish's TT carries over
    # across consecutive `reuse_tt=True` calls (round-robin distribution put
    # positions K-plies apart in the same slice — the TT trees diverged too
    # far for any useful overlap at depth 14 / 64 MB hash). Last slice gets
    # the remainder.
    chunk = math.ceil(len(positions) / k)
    slices: list[list[chess.Board]] = [
        list(positions[i : i + chunk]) for i in range(0, len(positions), chunk)
    ]
    # Trim to k slices (we may have one fewer if positions divides evenly).
    slices = slices[:k]

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
                            history_complete=history_complete,
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
                    history_complete=history_complete,
                )
                out.append((r, hit))
            return out

    slice_results = await asyncio.gather(*[_run_slice(s) for s in slices if s])
    # Reassemble in original order. With contiguous chunks, slice_results[si]
    # corresponds to positions[si*chunk : si*chunk + len(slices[si])] in order.
    out: list[tuple[MCPEval, bool]] = [None] * len(positions)  # type: ignore[list-item]
    cursor = 0
    for slice_result in slice_results:
        for j, item in enumerate(slice_result):
            out[cursor + j] = item
        cursor += len(slice_result)
    return out


async def _create_analyzer_pool(
    cfg: MCPSettings,
    *,
    pool_size: int,
) -> AnalyzerPool | TCPAnalyzerPool:
    """Single source of truth for analyzer pool creation.

    Used by both the FastMCP lifespan (`_mcp_lifespan`) and the lazy-init
    fallback in `_get_analyzer_pool` so the two paths cannot drift in their
    UCI kwargs (the show_wdl/syzygy_path bug shipped in 72e3236 is the kind of
    drift this function prevents). Returns a pool ready for evaluate/top_moves/
    classify_move/analyze_game; logs the same "ready" line either way so
    operators can grep for it regardless of which path built the pool.
    """
    threads = max(1, cfg.threads_per_worker)
    if cfg.host and cfg.port:
        pool: AnalyzerPool | TCPAnalyzerPool = await TCPAnalyzerPool.create(
            cfg.host,
            cfg.port,
            size=pool_size,
            name="stockfish",
            threads=threads,
            hash_mb=cfg.hash_mb,
            show_wdl=cfg.show_wdl,
            syzygy_path=cfg.syzygy_path or None,
        )
        log.info(
            "TCP analyzer pool ready: %d engines @ %s:%d (threads=%d hash=%dMB wdl=%s syzygy=%s ponder=%s)",
            pool_size,
            cfg.host,
            cfg.port,
            threads,
            cfg.hash_mb,
            cfg.show_wdl,
            cfg.syzygy_path or "(none)",
            cfg.ponder_enabled,
        )
        pool._mcp_ponder_enabled = cfg.ponder_enabled  # type: ignore[attr-defined]
        return pool
    pool = await AnalyzerPool.create(
        _stockfish_path(),
        size=pool_size,
        depth=14,
        threads=threads,
        hash_mb=cfg.hash_mb,
        show_wdl=cfg.show_wdl,
        syzygy_path=cfg.syzygy_path or None,
    )
    log.info(
        "Subprocess analyzer pool ready: %d engines @ %s",
        pool_size,
        _stockfish_path(),
    )
    return pool


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
            cpu = os.cpu_count() or 8
            pool_size = mcp_cfg.pool_size if mcp_cfg.pool_size is not None else min(cpu, 4)
            _analyzer_pool = await _create_analyzer_pool(mcp_cfg, pool_size=pool_size)
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
        r"|\b(?:white|black)\s+(?:wins?|won)\s+on\s+time\b"
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


# P2/P3 (2026-09-02 ultra audit): the bare regex matches shapes like
# "0+0", "0+1", "40/0", "0/600" — all of which are syntactically PGN
# TimeControl values but semantically nonsense (a game with 0 seconds
# is unplayable; a 40-move period with 0 seconds is impossible; a
# 0-move period with a positive time is meaningless). The relaxed
# grammar was useful when the old validator was the only thing between
# caller input and the metadata block, but the audit showed callers
# relying on the validator as a "PGN TimeControl sanity check". Tighten
# the check so every stage must contain at least one non-zero digit;
# `_stage_has_positive_number` walks the string. Sentinel values ("?",
# "-") remain accepted unchanged.
_TIME_CONTROL_STAGE_RE = re.compile(r"^(?:\d+|\d+/\d+|\d+\+\d+|\*\d+)$")


def _stage_has_positive_number(stage: str) -> bool:
    """Return True iff every numeric half of `stage` is strictly positive.

    Splits the stage on `/`, `+`, or `*` (the three PGN TimeControl
    separators — `/` for moves/seconds, `+` for Fischer increment,
    `*` for hourglass prefix) and requires every individual numeric
    component to contain a non-zero digit. This rejects the
    syntactically-PGN-shaped but semantically impossible values the
    2026-09-02 audit flagged:

        "0+0"     -> both components are zero (unplayable game)
        "0+1"     -> 0-second base (unplayable)
        "40+0"    -> 0-second increment on a 40-second base
        "0/600"   -> 0-move period with 600 seconds (nonsensical)
        "40/0"    -> 0 seconds for a 40-move period (impossible)

    while keeping the legitimate values intact:

        "300"     -> single value, positive
        "300+5"   -> both components positive
        "40/7200" -> both components positive
        "*60"     -> hourglass prefix + positive value
    """
    # Strip the optional hourglass prefix "*" before splitting.
    body = stage[1:] if stage.startswith("*") else stage
    # Split on the two multi-component separators; the leading-digit
    # check then guards each piece. An empty piece (e.g. "+0" -> split
    # would yield ["", "0"]) is treated as zero and rejected.
    pieces: list[str] = []
    if "+" in body:
        pieces.extend(body.split("+"))
    elif "/" in body:
        pieces.extend(body.split("/"))
    else:
        pieces.append(body)
    for piece in pieces:
        if not piece or not any(c in "123456789" for c in piece):
            return False
    return True


def _is_valid_pgn_time_control(value: str) -> bool:
    """Validate the PGN TimeControl tag grammar.

    PGN permits a single stage or colon-separated stages. A stage is one of:
    sudden-death seconds (``300``), moves/seconds (``40/7200``), Fischer
    seconds+increment (``300+5``), or hourglass (``*60``). ``?`` and ``-``
    are the standard unknown/unspecified markers.

    Every numeric component must contain at least one non-zero digit — a
    stage of "0", "0+0", "0+1", "40+0", "0/600", or "40/0" cannot
    describe a real chess game and is rejected (audit P2/P3,
    2026-09-02).
    """
    text = value.strip()
    if text in {"?", "-"}:
        return True
    if not text:
        return False
    stages = text.split(":")
    return all(
        bool(stage)
        and _TIME_CONTROL_STAGE_RE.fullmatch(stage) is not None
        and _stage_has_positive_number(stage)
        for stage in stages
    )


def _find_movetext_result(text: str) -> str | None:
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
    movetext: str,
    start_board: chess.Board | None = None,
    strict: bool = False,
    nag_warnings: list[str] | None = None,
) -> list[str]:
    """Check that all tokens in the active movetext section are valid chess moves or PGN symbols.

    `nag_warnings`, when provided, receives out-of-range NAG tokens in lenient mode (the audit's
    P3 INVESTIGATE finding: `$999999` was silently accepted in lenient mode). In strict mode,
    out-of-range NAGs are returned in `invalid_tokens` and fail the parse.
    """
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
            # P3/INVESTIGATE (2026-09-02 ultra audit): NAGs outside the
            # 0..255 range defined by the PGN spec were silently dropped
            # in lenient mode. Strict mode already rejected them — keep
            # that behavior, but in lenient mode surface a warning via
            # the nag_warnings channel so callers can see the out-of-
            # range value instead of parsing as if it never existed.
            if nag_val > 255:
                if strict:
                    invalid_tokens.append(tok)
                elif nag_warnings is not None:
                    nag_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )
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
                error_prefix = "STRICT_PGN_ERROR" if strict else "INVALID_PGN"
                raise ValueError(
                    f"{error_prefix}: Invalid PGN syntax or unrecognized token in movetext: {invalid_tokens[0]!r}"
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


def _validate_strict_header_syntax(text: str) -> None:
    """Reject malformed PGN tag lines that tolerant cleaning would otherwise discard."""
    normalized = _normalize_unicode_pgn_results(text)
    masked = _mask_comments_and_escapes(normalized)
    raw_lines = normalized.splitlines()
    masked_lines = masked.splitlines()
    for index, visible in enumerate(masked_lines):
        if not re.match(r"^\s*\[[A-Za-z0-9_]+\b", visible):
            continue
        raw = raw_lines[index] if index < len(raw_lines) else visible
        if not _is_canonical_tag_line(raw):
            raise ValueError(
                f"STRICT_PGN_ERROR: Malformed PGN tag syntax on line {index + 1}: {raw.strip()!r}"
            )


def _strict_top_level_movetext_tokens(text: str) -> list[str]:
    """Return top-level movetext tokens with comments/RAVs masked out."""
    normalized = _normalize_movetext_figurines(_normalize_unicode_pgn_results(text))
    masked = _mask_comments_and_escapes(normalized)

    header_end = 0
    for match in TAG_PAIR_REGEX.finditer(masked):
        if masked[header_end : match.start()].strip() == "":
            header_end = match.end()
        else:
            break

    chars = list(masked[header_end:])
    variation_depth = 0
    for i, ch in enumerate(chars):
        if ch == "(":
            variation_depth += 1
            chars[i] = " "
            continue
        if ch == ")":
            chars[i] = " "
            variation_depth = max(0, variation_depth - 1)
            continue
        if variation_depth > 0:
            chars[i] = " "

    top_level = "".join(chars)
    top_level = re.sub(
        r"(\b[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#?!]*)(\$\d+)",
        r"\1 \2",
        top_level,
    )
    top_level = re.sub(r"\b(O-O-O|O-O)([+#?!]*)(\$\d+)", r"\1\2 \3", top_level)

    def split_move_number(match: re.Match[str]) -> str:
        dots = "..." if match.group(2) else "."
        return f" {match.group(1)}{dots} "

    top_level = re.sub(r"(?<![A-Za-z0-9_])(\d+)\.(\.\.)?", split_move_number, top_level)
    return top_level.split()


def _validate_strict_mainline_surface(text: str, game: chess.pgn.Game) -> None:
    """Require canonical SAN and correct explicit move numbers in strict mode."""
    tokens = _strict_top_level_movetext_tokens(text)
    moves = list(game.mainline_moves())
    board = game.board()
    move_index = 0

    for token in tokens:
        clean = token.strip()
        if not clean:
            continue
        if clean in ("1-0", "0-1", "1/2-1/2", "*"):
            break
        nag = re.fullmatch(r"\$(\d+)", clean)
        if nag:
            if int(nag.group(1)) > 255:
                raise ValueError(
                    f"STRICT_PGN_ERROR: NAG {clean!r} is outside the supported PGN range 0..255."
                )
            continue
        if clean in ("!", "?", "!!", "??", "!?", "?!"):
            continue

        number = re.fullmatch(r"(\d+)(\.|\.\.\.)", clean)
        if number:
            supplied = int(number.group(1))
            expected = board.fullmove_number
            expected_dots = "." if board.turn == chess.WHITE else "..."
            if supplied != expected or number.group(2) != expected_dots:
                raise ValueError(
                    "STRICT_PGN_ERROR: Move number mismatch: "
                    f"found {clean!r}, expected {expected}{expected_dots} for the side to move."
                )
            continue

        if clean.lower() in ("e.p.", "e.p", "ep", "(e.p.)", "(e.p)", "(ep)"):
            raise ValueError(
                "STRICT_PGN_ERROR: Explicit en-passant marker requires syntax normalization; "
                "use canonical SAN only."
            )

        if move_index >= len(moves):
            raise ValueError(f"STRICT_PGN_ERROR: Unexpected trailing movetext token {clean!r}.")

        move = moves[move_index]
        canonical = board.san(move)
        supplied_san = clean.rstrip("!?")
        if supplied_san != canonical:
            raise ValueError(
                f"STRICT_PGN_ERROR: Non-canonical SAN: found {clean!r}, expected {canonical!r}."
            )
        board.push(move)
        move_index += 1

    if move_index != len(moves):
        raise ValueError(
            "STRICT_PGN_ERROR: Strict movetext validation did not consume the complete mainline."
        )


def _sanitize_malformed_pgn_header_lines(text: str, strict: bool = False) -> tuple[str, list[str]]:
    """Reject or remove malformed tag-pair lines before PGN extraction.

    The conversational PGN cleaner clusters only syntactically valid tag
    pairs. A malformed line between valid tags used to split the cluster and
    silently discard otherwise valid metadata. We inspect only the pre-move
    prefix and only activate when that prefix contains at least one valid PGN
    tag, so bracket-looking prose in ordinary movetext is left untouched.
    P2 (2026-09-02 ultra audit): the legacy function returned no warnings
    for header-only PGNs (no `1. <move>` marker before the result) and for
    malformed lines that didn't match the regex pre-filter. Both classes now
    surface warnings in lenient mode.
    """
    normalized = _normalize_multiline_tags(text)
    lines = normalized.splitlines(keepends=True)
    if not lines:
        return normalized, []

    first_move_line = len(lines)
    for idx, line in enumerate(lines):
        if re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", line):
            first_move_line = idx
            break

    prefix = "".join(lines[:first_move_line])
    if TAG_PAIR_REGEX.search(_mask_comments_and_escapes(prefix)) is None:
        # No tag block present; nothing to sanitize.
        return normalized, []
    # P2 (2026-09-02 ultra audit): handle header-only PGNs (no first-move
    # line) by sanitizing every line in the prefix. Previously the
    # `first_move_line >= len(lines)` early return silently dropped
    # malformed headers in header-only inputs.
    scan_end = first_move_line if first_move_line < len(lines) else len(lines)

    warnings: list[str] = []
    for idx in range(scan_end):
        stripped = lines[idx].strip()
        if not stripped.startswith("["):
            continue
        if _is_canonical_tag_line(stripped):
            continue
        # Skip lines that mix a valid tag with extra content (e.g.
        # `[Result "*"] *` — a valid tag followed by the game result
        # token on the same line). The legacy contract accepted such
        # mixed lines as part of the conversational PGN dialect.
        if TAG_PAIR_REGEX.search(stripped) is not None:
            continue
        # P2 (2026-09-02 ultra audit): drop the regex pre-filter that was
        # silently dropping warnings for malformed tags that didn't happen
        # to match `^\[\s*[A-Za-z0-9_]+(?:\s|\])`. The canonical-tag-line
        # check above is the sole gate; everything else that looks tag-like
        # but isn't canonical is reported.
        warning = f"Malformed PGN header line ignored: {stripped!r}."
        if strict:
            raise ValueError(f"STRICT_VALIDATION_ERROR: {warning}")
        warnings.append(warning)
        newline = (
            "\r\n" if lines[idx].endswith("\r\n") else ("\n" if lines[idx].endswith("\n") else "")
        )
        lines[idx] = newline

    return "".join(lines), warnings


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
    sanitized, _header_warnings = _sanitize_malformed_pgn_header_lines(text, strict=strict)
    _check_multiple_games(sanitized)
    canonical = _extract_canonical_pgn_text(sanitized)
    game = _extract_game_inner(canonical, strict=strict)
    if strict:
        _validate_strict_header_syntax(canonical)
        _validate_strict_mainline_surface(canonical, game)
    return game


def _extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
    masked_cleaned = _mask_comments_and_escapes(cleaned)
    for m in TAG_PAIR_REGEX.finditer(masked_cleaned):
        tag_name = m.group(1).lower()
        if tag_name == "variant":
            _validate_variant(_unescape_pgn_tag_value(m.group(2)))
        # P1 (2026-09-02 ultra audit): the FEN tag from PGN headers must be
        # validated against the same rules as a direct FEN input — fullmove
        # must be ≥1, halfmove within bounds, EP+halfmove historically
        # consistent. Previously, fullmove=0 inside [FEN "..."] was silently
        # accepted by `analyze_game(strict=true)` while the same FEN was
        # rejected by `evaluate_position`. This call shares the unified
        # validator with `_build_board`. Both strict and lenient modes reject
        # because the FEN value is structurally identical to a direct FEN —
        # there is no permissive grammar here that would justify
        # normalization.
        if tag_name == "fen":
            fen_val = _unescape_pgn_tag_value(m.group(2))
            if fen_val:
                _validate_fen_counters(fen_val, strict)

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
    # P0 (2026-09-02 ultra audit): lenient normalization may change syntax,
    # never move identity. Earlier this loop iterated over every possible
    # start index and picked the longest parseable subsequence, which let a
    # malformed movetext like "1... e5 2. Nf3 Nc6 *" silently drop e5 (Black's
    # intended first move) and substitute White's Nf3 in its place — the
    # parser ended up analyzing a completely different game than the caller
    # supplied. The fix is to parse from the start of the input only (no
    # best-of-N across start positions) and refuse to silently drop a move
    # token when later ones parse legally: that drop is exactly the
    # semantic substitution the audit forbids.
    tokens = norm_text.split()
    if tokens:
        b = chess.Board()
        cur_moves: list[chess.Move] = []
        cur_result: str | None = None
        last_was_result = False
        for t in tokens:
            clean_t = t.rstrip(".,;:!?").lstrip(".,;:!?")
            clean_t = re.sub(
                r"\s*\(?\s*e\.?p\.?\s*\)?$",
                "",
                clean_t,
                flags=re.IGNORECASE,
            ).rstrip(".,:!?")
            if (
                not clean_t
                or clean_t.lower() in ("e.p.", "e.p", "ep", "(e.p.)", "(e.p)", "(ep)")
                or re.match(r"^\d+[\.\:]*$", clean_t)
            ):
                continue
            if clean_t in ("1-0", "0-1", "1/2-1/2", "*"):
                if not last_was_result:
                    cur_result = clean_t
                    last_was_result = True
                # Subsequent result tokens (or trailing move tokens after a
                # result) are dropped — matches the existing "trailing moves
                # after game termination are ignored" behavior.
                continue
            if last_was_result:
                # Game already ended; trailing move tokens are ignored.
                continue
            try:
                m = b.parse_san(clean_t)
                b.push(m)
                cur_moves.append(m)
                continue
            except Exception:
                pass
            try:
                m = chess.Move.from_uci(clean_t)
                if m in b.legal_moves:
                    b.push(m)
                    cur_moves.append(m)
                    continue
            except Exception:
                pass
            # P0: this token did not parse as either SAN or UCI on the
            # current board. The bare-moves fallback must NOT silently skip
            # it and try later tokens (that is the semantic-substitution
            # bug the audit caught). Surface the failure instead.
            raise ValueError(
                f"INVALID_PGN: Move token {t!r} could not be parsed as a legal "
                f"chess move at this point in the game. The lenient parser "
                f"refuses to substitute a different move."
            )

        if cur_moves or cur_result is not None:
            game = chess.pgn.Game()
            if cur_result:
                game.headers["Result"] = cur_result
            curr: chess.Board | chess.pgn.GameNode = game
            for m in cur_moves:
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
    # U-11 (2026-09-01): normalize Unicode hyphens to ASCII so
    # castling tokens like "0–0" (en-dash) are recognized like the
    # analyze_game path does. _UNICODE_HYPHEN_MAP is a str.maketrans
    # translation table (not a regex), so use .translate() to match
    # _normalize_unicode_pgn_results. Brings classify_move into
    # parity with analyze_game.
    clean_move = clean_move.translate(_UNICODE_HYPHEN_MAP)

    # Normalize castling variants
    lower_cand = clean_move.lower()
    if lower_cand in ("o-o-o", "0-0-0", "o-o-o+", "0-0-0+", "o-o-o#", "0-0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O-O{suffix}"
    elif lower_cand in ("o-o", "0-0", "o-o+", "0-0+", "o-o#", "0-0#"):
        suffix = "#" if "#" in clean_move else ("+" if "+" in clean_move else "")
        clean_move = f"O-O{suffix}"

    san_cand = clean_move.rstrip("!?")

    # P1/P2 (2026-09-02 ultra audit): uppercase UCI like `E2E4`, `e2E4`,
    # `A7A8Q` was previously accepted silently without a syntax warning
    # in BOTH lenient and strict modes. PGN movetext requires
    # SAN-shaped notation (and accepts lowercase only), so this
    # normalization only applies to the DIRECT move API. The
    # normalization itself is harmless (lower-case UCI is canonical),
    # but the silent acceptance hid input-shape drift and let callers
    # paste uppercase accidentally. We now:
    #   - emit a `syntax_warning` whenever the supplied UCI contains
    #     uppercase letters, in lenient mode (so the audit's
    #     "normalized without a syntax warning" finding is closed);
    #   - reject with STRICT_SAN_ERROR when the caller asks for strict
    #     mode (the audit explicitly calls out that strict mode should
    #     reject or document non-canonical UCI form).
    #
    # Parse order matters here. The audit's `B8e5` reproducer in the
    # `test_randomized_legal_move_san_and_fen_differential_5000_positions`
    # test exercises `board.san(move)` output — for a White Bishop on
    # b8 moving to e5, python-chess generates the rank-disambiguated
    # SAN `B8e5`. The same string is also a syntactically valid UCI
    # (`b8e5` after lowercase). Treating it as UCI in BOTH cases
    # caused a false-positive flag against the legitimate SAN form.
    # The fix: try SAN FIRST; if SAN parsing succeeds, prefer it and
    # never flag uppercase UCI. Only fall through to UCI when SAN
    # parsing fails, in which case the input is unambiguously UCI.
    #
    # Try SAN with candidates (e.g. clean, stripped !?, stripped +/#,
    # with/without =, promo variants).
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

    # SAN parsing failed (no candidate matched a legal move). Fall
    # through to UCI parsing — the input is unambiguously UCI now.
    #
    # Only flag uppercase UCI when the original matched the UCI shape
    # AND had uppercase letters in valid UCI positions (file letters
    # at 0/2, optional promotion piece at 4). This avoids the false
    # positive where python-chess's `board.san(move)` output for a
    # rank-disambiguated SAN like `B8e5` happens to also lowercase to
    # a valid UCI — we only catch the uppercase-UCI case when SAN
    # parsing definitively failed (above).
    uci_was_upper = False
    if (
        re.fullmatch(r"[a-hA-H][1-8][a-hA-H][1-8][qrbnQRBN]?", clean_move)
        is not None  # position 0 is a valid file letter
        and clean_move != clean_move.lower()
        and any(c.isalpha() for c in clean_move)
    ):
        uci_was_upper = True
    uci_syntax_warning: str | None = None
    for uci_cand in (clean_move, clean_move.lower()):
        # Parse-only — don't catch our STRICT_SAN_ERROR raise below. The
        # previous shape accidentally did, because the catch's `ValueError`
        # matched both python-chess's and ours.
        m_obj: chess.Move | None = None
        try:
            m_obj = chess.Move.from_uci(uci_cand)
        except (chess.InvalidMoveError, ValueError):
            m_obj = None
        if m_obj is not None and m_obj in board.legal_moves:
            if uci_was_upper:
                raw_s = move_str.strip(" \t\r\n`'\"")
                uci_syntax_warning = (
                    f"Input UCI '{raw_s}' normalized to lowercase '{uci_cand.lower()}'."
                )
                if strict:
                    # Promote the warning to a structured strict error.
                    # Raised outside the parse-only try so the caller
                    # actually sees STRICT_SAN_ERROR, not the catch-all
                    # ILLEGAL_MOVE at the bottom of the function.
                    raise ValueError(
                        f"STRICT_SAN_ERROR: Input UCI '{raw_s}' requires "
                        f"syntax normalization: {uci_syntax_warning}"
                    )
            return m_obj, uci_syntax_warning

    if ambiguous_err:
        raise ValueError(
            f"AMBIGUOUS_SAN: Move {move_str!r} is ambiguous in position {board.fen()!r}: {ambiguous_err}"
        )
    raise ValueError(
        f"ILLEGAL_MOVE: Move {move_str!r} is not a valid legal move in position {board.fen()!r}"
    )


def _parse_move_on_board(board: chess.Board, move_str: str) -> chess.Move:
    return _parse_move_on_board_with_warning(board, move_str)[0]


def _validate_castling_rights(board: chess.Board, rights_token: str, strict: bool, fen: str) -> str:
    """U-09 (2026-09-01): validate castling rights symmetrically.

    python-chess silently drops invalid castling rights (e.g. K with no
    white rook on h1) which masked the audit U-09 finding that the
    validator rejected some rights ("K" with no king) but accepted
    others ("Q" with no rook). The fix:
      - Strict mode: REJECT any token that contains a char referring to
        a non-existent rook of the matching color on its canonical square.
      - Non-strict mode: silently strip invalid chars and return a
        canonicalized token (still useful for callers that want
        continue-with-the-good-rights behavior).
      - X-FCR (e.g. "HAha") and Shredder-FEN are accepted as-is by
        python-chess; we don't second-guess the parser for them.

    Returns the validated (possibly empty / canonicalized) rights token.

    P3 (2026-09-02 ultra audit): FEN's lexical spec requires each castling
    right to appear at most once in the rights field. Inputs like "KK",
    "QQ", "QK" therefore contain a duplicate that is not a real FEN.
    python-chess's constructor deduplicates ("KK" → "K") without
    signaling the caller. Both strict and non-strict mode preserve
    that behavior (dedup is harmless: the canonical rights field has
    at most one of each character anyway). The `fen_was_canonicalized`
    flag in the response tells callers when this happened.
    """
    if not rights_token or rights_token == "-":
        return rights_token
    char_to_requirement: dict[str, tuple[chess.Color, chess.Square]] = {
        "K": (chess.WHITE, chess.H1),
        "Q": (chess.WHITE, chess.A1),
        "k": (chess.BLACK, chess.H8),
        "q": (chess.BLACK, chess.A8),
    }
    canonical_chars = list(dict.fromkeys(rights_token))
    valid: list[str] = []
    invalid: list[str] = []
    for ch in canonical_chars:
        req = char_to_requirement.get(ch)
        if req is None:
            valid.append(ch)
            continue
        color, sq = req
        piece = board.piece_type_at(sq)
        if piece == chess.ROOK and board.color_at(sq) == color:
            valid.append(ch)
        else:
            invalid.append(ch)
    if invalid:
        if strict:
            raise ValueError(
                f"INVALID_CASTLING_RIGHTS: FEN '{fen}' has castling "
                f"rights {rights_token!r} but the rook(s) for "
                f"{','.join(invalid)} are not on their canonical squares. "
                f"Rejected."
            )
        return "".join(valid) if valid else "-"
    return rights_token


MAX_HALFMOVE_CLOCK = 10000
MAX_FULLMOVE_NUMBER = 10000


def _validate_fen_counters(cleaned: str, strict: bool) -> tuple[list[str], str]:
    """Validate halfmove clock + fullmove number + EP/halfmove historical consistency.

    Returns (tokens, cleaned_to_parse). Raises ValueError on any invalid counter
    when strict=True, or when the value is unparseable / negative. Non-strict mode
    also raises on hard impossibilities (negative, unparseable, halfmove_clock > MAX)
    but permits non-historical EP + non-zero halfmove as a warning to the caller
    (which surfaces it via the metadata_warning channel rather than as an error).
    """
    tokens = cleaned.split()
    if len(tokens) >= 5:
        halfmove_raw = tokens[4]
        try:
            halfmove_num = int(halfmove_raw)
        except ValueError as exc:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' must be a valid integer."
            ) from exc
        if halfmove_num < 0:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' cannot be negative (got {halfmove_raw})."
            )
        if halfmove_num > MAX_HALFMOVE_CLOCK:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' "
                f"is {halfmove_num}; maximum supported value is {MAX_HALFMOVE_CLOCK}."
            )
        # P1 (2026-09-02 ultra audit): an en-passant target requires that
        # the previous move was a pawn double push — which resets the
        # halfmove clock to 0. A non-zero halfmove_clock alongside an
        # EP target square is historically impossible. Reject it in
        # strict mode; lenient mode also rejects (this is not a
        # lexical quirk the parser should silently canonicalize —
        # the input is contradictory at the level of chess semantics).
        if len(tokens) >= 4 and tokens[3] != "-" and halfmove_num != 0:
            ep_sq = tokens[3]
            # A pawn double push is the only move that produces an EP
            # target AND resets the halfmove clock, so halfmove_clock
            # MUST be 0 in any FEN that preserves an EP target. We
            # don't need to reconstruct which side moved last from the
            # FEN alone — the combination is simply contradictory at
            # the level of chess semantics.
            raise ValueError(
                f"INVALID_FEN: FEN '{cleaned}' has en-passant target '{ep_sq}' "
                f"but halfmove clock is {halfmove_num}; an en-passant target "
                f"requires the previous move to have been a pawn double push "
                f"which would have reset the halfmove clock to 0."
            )
    if len(tokens) >= 6:
        fullmove_raw = tokens[5]
        try:
            fullmove_num = int(fullmove_raw)
        except ValueError as exc:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a valid integer."
            ) from exc
        if fullmove_num < 1:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a positive integer >= 1 (got {fullmove_raw})."
            )
        if fullmove_num > MAX_FULLMOVE_NUMBER:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' "
                f"is {fullmove_num}; maximum supported value is {MAX_FULLMOVE_NUMBER}."
            )
    return tokens, cleaned


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
                tokens, _ = _validate_fen_counters(cleaned, strict)
            try:
                # U-09 (2026-09-01): validate castling rights BEFORE handing
                # the FEN to python-chess. The library raises
                # INVALID_CASTLING_RIGHTS status on rights tokens that don't
                # match the actual rook placement, which previously
                # asymmetrically rejected "K" but accepted "Q" (the audit's
                # symptom). Pre-validation lets non-strict mode silently
                # strip bad chars and re-parse the canonicalized FEN; strict
                # mode raises the same structured error the audit wants.
                cleaned_to_parse = cleaned
                if not strict and "/" in cleaned and len(tokens) >= 3:
                    rights_token = tokens[2]
                    placement_side = " ".join(tokens[:2])
                    try:
                        rights_check_board = chess.Board(f"{placement_side} - - 0 1")
                    except Exception:
                        rights_check_board = None
                    if rights_check_board is not None:
                        validated = _validate_castling_rights(
                            rights_check_board,
                            rights_token,
                            False,
                            cleaned,
                        )
                        if validated != rights_token:
                            tokens[2] = validated if validated else "-"
                            cleaned_to_parse = " ".join(tokens)
                b = chess.Board(cleaned_to_parse)
                if b.is_valid() or b.status() == chess.STATUS_VALID:
                    board = b
                    # U-09 (2026-09-01): in strict mode, run the post-parse
                    # castling-rights validation too. python-chess silently
                    # drops rights that don't match the actual rook
                    # placement (e.g. "Q" with no rook on a1) and only
                    # raises INVALID_CASTLING_RIGHTS when the rights TOKEN
                    # itself conflicts with the king placement — which is
                    # asymmetric and exactly what the audit flagged. Our
                    # explicit check catches both directions.
                    if strict and "/" in cleaned_to_parse:
                        parts = cleaned_to_parse.split()
                        if len(parts) >= 3:
                            rights_token = parts[2]
                            _validate_castling_rights(
                                board,
                                rights_token,
                                True,
                                cleaned_to_parse,
                            )
                elif "/" in cleaned_to_parse:
                    if "INVALID_CASTLING_RIGHTS" in format_fen_status_errors(b.status()):
                        raise ValueError(
                            f"INVALID_CASTLING_RIGHTS: FEN '{cleaned}' "
                            f"references castling rights whose rooks are "
                            f"not on their canonical squares "
                            f"(status={b.status()})."
                        )
                    raise ValueError(
                        f"INVALID_FEN: Position '{cleaned_to_parse}' is not a valid FEN ({format_fen_status_errors(b.status())})."
                    )
            except ValueError as exc:
                if "/" in cleaned or "INVALID_FEN" in str(exc) or "INVALID_FEN" in str(exc)[:0]:
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
        game = _extract_game(cleaned, strict=strict)
        board = game.board()
        if not board.is_valid() or board.status() != chess.STATUS_VALID:
            raise ValueError(
                f"INVALID_FEN: Initial position '{board.fen()}' in PGN is not a valid chess position ({format_fen_status_errors(board.status())})."
            )
        for move in game.mainline_moves():
            board.push(move)

    assert board is not None
    for move_str in moves or []:
        move, _ = _parse_move_on_board_with_warning(board, move_str, strict=strict)
        board.push(move)

    return board


def _history_provenance_for_input(
    fen_or_pgn: str,
    moves: list[str] | None,
) -> str:
    """Return complete, partial or incomplete history provenance for an input."""
    cleaned = (
        fen_or_pgn.replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .strip("`'\" \t\r\n")
    )
    if cleaned.lower() in ("startpos", "initial", "start"):
        return "complete"

    tokens = cleaned.split()
    looks_like_fen = (
        "/" in cleaned
        and 1 <= len(tokens) <= 6
        and not cleaned.startswith("[")
        and not tokens[0].endswith(".")
    )
    if looks_like_fen:
        return "partial" if moves else "incomplete"

    # A PGN/movetext input defines its own game root and therefore carries the
    # complete history represented by that game. Additional suffix moves keep it complete.
    return "complete"


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

    # Canonicalization is a property of the supplied FEN itself, not of any
    # suffix moves replayed after that FEN. Compare the input against a board
    # parsed before replaying the suffix, then return the final board FEN.
    canonical_input_fen: str | None = None
    if input_fen is not None:
        canonical_input_fen = _build_board(fen_or_pgn, [], strict).fen()

    board = _build_board(fen_or_pgn, moves, strict)
    canonical = board.fen()
    was_canonicalized = bool(input_fen) and input_fen != canonical_input_fen
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
    if _pool_supports_root_moves(pool):
        return await pool.evaluate(b, depth=depth, root_moves=root_moves)  # type: ignore[arg-type]
    return await pool.evaluate(b, depth=depth)  # type: ignore[arg-type]


_POOL_SUPPORTS_ROOT_MOVES: dict[type, bool] = {}


def _pool_supports_root_moves(pool: object) -> bool:
    """Memoized runtime check for the `root_moves` keyword on pool.evaluate.

    `inspect.signature` walks the function annotations every call; cache by
    the pool's class — same class always has the same evaluate signature, so
    per-class caching is correct (and the cache survives instance churn).
    """
    cls = type(pool)
    cached = _POOL_SUPPORTS_ROOT_MOVES.get(cls)
    if cached is not None:
        return cached
    import inspect

    try:
        sig = inspect.signature(pool.evaluate)  # type: ignore[attr-defined]
        supports = "root_moves" in sig.parameters
    except (TypeError, ValueError):
        supports = False
    _POOL_SUPPORTS_ROOT_MOVES[cls] = supports
    return supports


async def _evaluate_game_position_cached(
    b: chess.Board,
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    requested_depth: int | None = None,
    history_complete: str | bool = "incomplete",
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
    history_state = (
        ("complete" if history_complete else "incomplete")
        if isinstance(history_complete, bool)
        else history_complete
    )
    canonical_fen_str = b.fen()
    url, img = lichess_urls(canonical_fen_str)

    rule_status = evaluate_rule_status(b, history_complete=history_state)
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
        from mcp_server.actions import build_best_action, build_legal_actions

        terminal_best_action = build_best_action(
            recommended_action="game_over",
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            sign=1 if b.turn == chess.WHITE else -1,
        )
        terminal_legal_actions = build_legal_actions(
            rule_status=rule_status,
            engine_eval=None,
            board=b,
            legal_engine_moves=None,
        )
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
                best_action_obj=terminal_best_action,
                legal_actions=terminal_legal_actions,
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
        history_completeness=history_state,
    )
    cached = await _cache.get_eval(ckey)
    if cached is not None:
        return cached.model_copy(update={"requested_depth": req_d}), True

    async def _compute_pos() -> MCPEval:
        ev = await _eval_via_analyzer_or_pool(analyzer, pool, b, depth=depth, reuse_tt=reuse_tt)

        # U-02 (2026-09-01): at halfmove>=100, the root cp/mate can be
        # "polluted" by draw awareness — a winning zeroing capture like
        # Kxe2 in K+R vs R at halfmove=100 reports a tiny cp because the
        # draw is on the table, even though the post-state is K+R vs K
        # (a forced win). Detect this by re-evaluating the post-state of
        # the engine's best zeroing move; if it's a high-confidence win,
        # surface it to the action policy so it recommends play_move
        # instead of claim_draw. Used by score_played_move and
        # _pick_root_recommended_action (which already do this for
        # top_moves' multipv output).
        zeroing_best_cp_arg: int | None = None
        zeroing_best_mate_arg: int | None = None
        if ev.best_move and b.halfmove_clock >= 100 and not b.is_game_over():
            try:
                bm_obj = chess.Move.from_uci(ev.best_move.lower())
                if bm_obj in b.legal_moves:
                    is_zeroing = b.is_capture(bm_obj) or (
                        b.piece_type_at(bm_obj.from_square) == chess.PAWN
                    )
                    if is_zeroing:
                        b_after = b.copy(stack=True)
                        b_after.push(bm_obj)
                        if not b_after.is_game_over(claim_draw=False):
                            try:
                                post_ev = await pool.evaluate(b_after, depth=depth)
                                # Compute the post-state mover-POV score.
                                # `post_ev.cp` / `post_ev.mate` are
                                # White-POV (post-analyzer convention). The
                                # mover's perspective is `mover_sign` (sign
                                # = +1 if White is currently on turn, -1 if
                                # Black); same convention as the
                                # top_moves zeroing loop at L2969-2977.
                                mover_sign = 1 if b.turn == chess.WHITE else -1
                                if post_ev.mate is not None:
                                    mover_mate = mover_sign * post_ev.mate
                                    if mover_mate > 0:
                                        zeroing_best_mate_arg = mover_mate
                                elif post_ev.cp is not None:
                                    mover_cp = mover_sign * post_ev.cp
                                    if mover_cp > 0:
                                        zeroing_best_cp_arg = mover_cp
                            except Exception:
                                pass
            except Exception:
                pass

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
            history_complete=history_state,
            zeroing_move_best_score=zeroing_best_cp_arg,
            zeroing_move_best_mate=zeroing_best_mate_arg,
        )
        # Stamp build identity so every cached eval records which build produced it.
        identity = _build_identity(pool)
        mcp_eval = mcp_eval.model_copy(
            update={
                "build_sha": identity["build_sha"],
                "engine_config": identity["engine_config"],
            }
        )
        # U-13 (2026-09-01): also stamp the build identity into the
        # nested `engine_eval` sub-dict so a caller reading just the
        # sub-dict (e.g. for telemetry) gets the same provenance.
        if mcp_eval.engine_eval is not None:
            mcp_eval = mcp_eval.model_copy(
                update={
                    "engine_eval": {
                        **mcp_eval.engine_eval,
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                    }
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
            history_complete=history_state,
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
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        pool = await _get_analyzer_pool(ctx)
        # History completeness is derived from whether the caller had the move
        # stack. Naked FEN (no moves) cannot detect threefold repetition;
        # we MUST report `repetition_status="unknown"` for the audit H-01 fix.
        # When moves were supplied, the move stack is complete and we can
        # answer threefold claims definitively.
        history_complete = _history_provenance_for_input(fen, moves)
        res, is_hit = await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete=history_complete,
        )
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, cache_hit=is_hit)
        # L-06: surface input vs canonical FEN. Canonicalization describes
        # parser normalization of the supplied FEN only; replayed suffix moves
        # are reflected in canonical_fen but do not make the input noncanonical.
        result = res.model_copy(
            update={
                "requested_depth": raw_requested_depth,
                "input_fen": input_fen,
                "canonical_fen": canonical_fen,
                "fen_was_canonicalized": fen_was_canonicalized,
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
        if "INVALID_VERBOSITY" in msg:
            code = "invalid_verbosity"
        elif "STRICT" in msg:
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
          Each candidate in `result` represents a `play_move` action. Its
          `best_move`, `pv`, and engine `cp`/`mate` retain the root MultiPV
          action value and notation frame, so PV[0] is the candidate move and
          a mating candidate may retain Stockfish's root mate distance (e.g. 1).
          The candidate `canonical_fen`, terminal status, winner, rule fields,
          and `post_position` describe the board AFTER that candidate is played.
          Automatic terminal draws normalize candidate `cp` to 0. Draw-claim
          actions are reported separately via outer `best_action_obj` and
          `legal_actions`; they are not mixed into the MultiPV candidate list.

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
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        board, _input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        # evaluate_position with explicit moves has full history; naked FEN doesn't.
        history_complete = _history_provenance_for_input(fen, moves)
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
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
                engine="Stockfish",
                engine_version=engine_name_str,
                **_build_identity(pool),
                result=[],
            )

        cache_key = top_moves_cache_key(
            board,
            depth,
            n=n,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )

        # sign = mover's perspective sign (White=+1, Black=-1). Used below for
        # both the cache-hit and the freshly-computed paths to decide whether a
        # candidate is winning FOR the side-to-move (cp is White-POV).
        sign = 1 if board.turn == chess.WHITE else -1

        from mcp_server.actions import build_best_action, build_legal_actions

        def _pick_root_recommended_action(items: list[MCPEval]) -> str:
            if not items:
                return rule_status.recommended_action
            best = items[0]
            # U-01 (2026-09-01): mate must take precedence over cp. When
            # Stockfish finds a forced mate it sometimes still emits a
            # saturated cp=±20000; per chess convention, mate wins.
            # Use post_state_* when available (audit B-04 / B-05).
            eff_mate = best.post_state_mate if best.post_state_mate is not None else best.mate
            eff_cp = best.post_state_cp if best.post_state_cp is not None else best.cp
            if eff_mate is not None:
                mover_score: int | None = sign * eff_mate * 1000
            elif eff_cp is not None:
                mover_score = sign * eff_cp
            else:
                mover_score = None
            mate_for_mover = sign * eff_mate if eff_mate is not None else None
            # AUDIT B-04: also surface the best post-state value across all
            # zeroing candidates (capture or pawn move) so the policy can
            # prefer play_move over claim_draw when a zeroing move wins.
            # The post-state cp/mate is attached to each item by the fresh
            # path (audit B-05); the cache-hit path inherits the same data
            # because items are persisted with their post_state_* fields.
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for item in items:
                if not item.best_move:
                    continue
                try:
                    bm = chess.Move.from_uci(item.best_move)
                except Exception:
                    continue
                if not (board.is_capture(bm) or board.piece_type_at(bm.from_square) == chess.PAWN):
                    continue
                # Prefer the re-evaluated post-state value when present
                # (draw-pollution guard, audit B-04); fall back to the
                # multipv value otherwise.
                eff_cp = item.post_state_cp if item.post_state_cp is not None else item.cp
                eff_mate = item.post_state_mate if item.post_state_mate is not None else item.mate
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0 and (
                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                    ):
                        zeroing_best_mate = mover_mate
                elif eff_cp is not None:
                    mover_cp = sign * eff_cp
                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                        zeroing_best_cp = mover_cp
            return choose_recommended_action(
                board,
                can_claim_now=rule_status.can_claim_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                mover_score=mover_score,
                mate_for_mover=mate_for_mover,
                zeroing_move_best_score=zeroing_best_cp,
                zeroing_move_best_mate=zeroing_best_mate,
            )

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
                canonical_fen=canonical_fen,
                fen_was_canonicalized=fen_was_canonicalized,
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
            # AUDIT B-04: when a draw claim is available (immediately or with
            # an intended zeroing move), the root MultiPV cp/mate of a zeroing
            # move can be "polluted" by the engine seeing the draw on the
            # table (e.g. K+R vs K at halfmove=100 reports a tiny cp). We
            # re-evaluate zeroing moves' post-state ONLY when the multipv
            # output looks suspect — i.e. no explicit mate AND a non-positive
            # cp for the mover. Stockfish multipv is authoritative in every
            # other case (no draw on the table, or the engine already gave a
            # clearly winning cp/mate); re-evaluating otherwise just costs an
            # extra engine call without changing the answer. The candidate's
            # reported cp/mate remains the multipv value so ranking and
            # back-compat consumers see the same numbers they did before.
            needs_post_eval = bool(
                rule_status.can_claim_now or rule_status.can_claim_with_intended_move
            )
            zeroing_best_cp: int | None = None
            zeroing_best_mate: int | None = None
            for r in results:
                b_cand = board.copy(stack=True)
                cand_san_val: str | None = None
                cand_post_terminal: str | None = None
                cand_winner: str | None = None
                cand_can_claim_now = False
                cand_can_claim_draw = False
                cand_claim_reasons: list[str] = []
                cand_claim_reasons_now: list[str] = []
                cand_claim_moves: list[str] = []
                # Default to the root rule_status; the post-state branch below
                # refines it. Used by the best_action_obj build below as a
                # fallback when `r.best_move` is missing or fails to parse.
                cand_rule = rule_status
                # Track the post-state cp/mate for the action policy without
                # mutating the candidate's reported values.
                post_state_cp: int | None = None
                post_state_mate: int | None = None

                if r.best_move:
                    try:
                        bm_obj = chess.Move.from_uci(r.best_move.lower())
                        if bm_obj in board.legal_moves:
                            cand_san_val = board.san(bm_obj)
                            is_zeroing = board.is_capture(bm_obj) or (
                                board.piece_type_at(bm_obj.from_square) == chess.PAWN
                            )
                            b_cand.push(bm_obj)
                            # AUDIT B-04: re-evaluate zeroing-move post-state
                            # when the multipv output looks draw-polluted. We
                            # only do this when there's no explicit mate AND
                            # the multipv cp is non-positive for the mover (a
                            # winning move at halfmove=100 should at least
                            # show cp>0; if it doesn't, the engine is treating
                            # the draw as the value of the move and the
                            # post-state is what really matters). The
                            # post-state values feed the action policy
                            # decision; they DO NOT overwrite the candidate's
                            # reported cp/mate (B-05 / C-02 contract).
                            # The post-state re-eval is a draw-pollution guard
                            # (audit B-04 / U-08): when the multipv says the
                            # zeroing move is no better than the draw
                            # (cp<=0 or None), the post-state is what really
                            # matters — the engine is treating the draw as
                            # the value of the move. We do NOT re-evaluate
                            # for strongly positive multipv (the engine has
                            # a clear opinion and a re-eval would only add
                            # cost). The post_state_cp/mate are surfaced on
                            # the wire for client inspection (U-08) — they
                            # are None when no re-eval happened, which is
                            # the honest contract: "no refined post-state
                            # value" rather than fabricating one.
                            multipv_suspect = r.mate is None and (r.cp is None or r.cp <= 0)
                            if (
                                needs_post_eval
                                and is_zeroing
                                and not b_cand.is_game_over(claim_draw=False)
                                and multipv_suspect
                            ):
                                try:
                                    post_ev = await pool.evaluate(b_cand, depth=depth)
                                    if post_ev.mate is not None:
                                        post_state_mate = post_ev.mate
                                    elif post_ev.cp is not None:
                                        post_state_cp = post_ev.cp
                                except Exception:
                                    pass
                            cand_sign = 1 if b_cand.turn == chess.WHITE else -1
                            cand_mover_score: int | None
                            if r.mate is not None:
                                cand_mover_score = cand_sign * r.mate * 1000
                            elif r.cp is not None:
                                cand_mover_score = cand_sign * r.cp
                            else:
                                cand_mover_score = None
                            cand_mate_for_mover = cand_sign * r.mate if r.mate is not None else None
                            cand_rule = evaluate_rule_status(
                                b_cand,
                                mover_score=cand_mover_score,
                                mate_for_mover=cand_mate_for_mover,
                                history_complete=history_complete,
                            )
                            cand_post_terminal = cand_rule.terminal
                            cand_winner = cand_rule.winner
                            cand_can_claim_now = cand_rule.can_claim_now
                            cand_can_claim_draw = cand_rule.can_claim_draw
                            cand_claim_reasons = cand_rule.claim_reasons
                            cand_claim_reasons_now = cand_rule.claim_reasons_now
                            cand_claim_moves = cand_rule.claim_moves
                            # Track best zeroing post-state value for the
                            # action policy below. Sign is mover-POV so we
                            # compare apples to apples. Use the re-evaluated
                            # post-state values when available; fall back to
                            # multipv otherwise (audit B-04 guard).
                            eff_cp = post_state_cp if post_state_cp is not None else r.cp
                            eff_mate = post_state_mate if post_state_mate is not None else r.mate
                            if (
                                needs_post_eval
                                and is_zeroing
                                and (eff_mate is not None or eff_cp is not None)
                            ):
                                mover_sign = 1 if board.turn == chess.WHITE else -1
                                if eff_mate is not None:
                                    mover_mate = mover_sign * eff_mate
                                    if mover_mate > 0 and (
                                        zeroing_best_mate is None or mover_mate > zeroing_best_mate
                                    ):
                                        zeroing_best_mate = mover_mate
                                else:
                                    mover_cp = mover_sign * (eff_cp or 0)
                                    if zeroing_best_cp is None or mover_cp > zeroing_best_cp:
                                        zeroing_best_cp = mover_cp
                    except Exception:
                        pass

                identity = _build_identity(pool)
                # Candidate's reported cp/mate stays at the multipv value so
                # ranking and back-compat callers see the same numbers they
                # did before. Re-evaluated post-state values feed only the
                # action policy decision (audit B-04 / B-05 separation).
                post_eval_for_candidate = Eval(
                    cp=r.cp,
                    mate=r.mate,
                    best_move=r.best_move,
                    pv=r.pv,
                    depth=r.depth,
                )
                # Audit C-03 (2026-09-01 adversarial probe): the candidate's
                # outer action type is the type of move it represents —
                # `play_move` (a candidate IS a play_move action) or
                # `game_over` (the post-state is terminal). The post-state's
                # `rule_status.recommended_action` can be a claim (e.g. after
                # Qb1 the opponent can claim draw) but that is the OPPONENT's
                # perspective, not the candidate's. Reassign `best_action` /
                # `best_action_type` / `best_action_obj` to the candidate's
                # own action type so each candidate reads as a self-consistent
                # play_move or game_over unit. The post-state's recommendation
                # is preserved in `post_position.recommended_action`.
                cand_recommended_action = (
                    "game_over" if cand_post_terminal is not None else "play_move"
                )
                from mcp_server.actions import build_best_action as _build_ba

                if cand_post_terminal is not None:
                    outcome = (
                        "draw"
                        if cand_post_terminal != "checkmate"
                        else ("win" if cand_winner == "white" else "loss")
                    )
                    cand_best_action_obj: dict[str, Any] = {
                        "type": "game_over",
                        "outcome": outcome,
                        "reason": cand_post_terminal,
                    }
                else:
                    # Use the root `board` (not b_cand) for SAN lookup: the
                    # candidate's `best_move` is a legal move AT THE ROOT, not
                    # after it has been played. Passing b_cand would make
                    # `bm in board.legal_moves` False and silently drop SAN.
                    cand_best_action_obj = _build_ba(
                        recommended_action="play_move",
                        rule_status=cand_rule,
                        engine_eval=r,
                        board=board,
                        sign=sign,
                    )
                mcp_eval = MCPEval.from_eval(
                    post_eval_for_candidate,
                    b_cand.fen(),
                    board=b_cand,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                    pv_board=board,
                ).model_copy(
                    update={
                        "build_sha": identity["build_sha"],
                        "engine_config": identity["engine_config"],
                        "post_terminal_status": cand_post_terminal,
                        "candidate_san": cand_san_val,
                        "post_can_claim_draw": cand_can_claim_draw,
                        "post_can_claim_now": cand_can_claim_now,
                        "post_claim_reasons": cand_claim_reasons,
                        "post_claim_moves": cand_claim_moves,
                        "recommended_action": cand_recommended_action,
                        "best_action": cand_recommended_action,
                        "best_action_type": cand_recommended_action,
                        "best_action_obj": cand_best_action_obj,
                        "post_state_cp": post_state_cp,
                        "post_state_mate": post_state_mate,
                        "post_position": {
                            "status": cand_post_terminal or "active",
                            "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                            "can_claim_now": cand_can_claim_now,
                            "can_claim_draw": cand_can_claim_draw,
                            "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                            "recommended_action": getattr(
                                cand_rule, "recommended_action", "play_move"
                            ),
                        },
                    }
                )
                res_list.append(mcp_eval)

            def _candidate_rank_key(eval_item: MCPEval) -> float:
                # U-01 (2026-09-01): mate for the mover must outrank any
                # finite-cp win, and the ordering must NOT depend on `n`
                # (the number of candidates requested). The previous rank
                # key had two failure modes:
                #   1. cp was returned unclamped, so a saturated cp=+20000
                #      candidate outranked a mate-in-1 candidate (9999).
                #   2. The rank key preferred the multipv cp of zeroing
                #      moves over the mate branch, so a non-mating capture
                #      could rank above a mating move.
                # Chess-correct total order for the side-to-move is:
                #   delivered mate (terminal) > forced mate for mover
                #     > finite-cp win (clamped to mate ceiling)
                #     > draw  > finite-cp loss > forced mate against mover.
                # We clamp cp to ±MATE_RANK_CEILING so any saturated
                # sentinel (cp=±20000, syzygy fallback, depth=0 win) cannot
                # outrank a forced mate. We always sort (the previous gate
                # `halfmove>=100 or has_terminal_cand` let Stockfish's
                # raw MultiPV order leak through for the >99% case, where
                # a forced mate could still be in slot 2+ at shallow depth).
                MATE_RANK_CEILING = 9999.0
                MATE_VALUE = 10000.0

                # Terminal checks first — these are the strongest signals
                # regardless of cp/mate.
                if eval_item.post_terminal_status == "checkmate":
                    # Candidate delivered mate. Always ranks above any
                    # non-mate candidate (mate=1 is the canonical best).
                    return MATE_VALUE
                if eval_item.post_terminal_status in (
                    "stalemate",
                    "insufficient_material",
                    "seventyfive_moves",
                    "fivefold_repetition",
                    "dead_position",
                ):
                    return 0.0

                # Mate branch BEFORE cp branch (U-01): a mate-in-1 must
                # outrank any finite-cp win. Use the post-state mate when
                # available (audit B-05 — re-eval can refine the multipv
                # mate; falls back to multipv when no re-eval happened).
                eff_mate = (
                    eval_item.post_state_mate
                    if eval_item.post_state_mate is not None
                    else eval_item.mate
                )
                if eff_mate is not None:
                    mover_mate = sign * eff_mate
                    if mover_mate > 0:
                        # Forced mate for mover: shorter is better.
                        return MATE_VALUE - abs(mover_mate)
                    # Forced mate against mover: longer is "less bad",
                    # but always below the floor for any finite cp.
                    return -MATE_VALUE + abs(mover_mate)

                # Cp branch: clamped to the mate ceiling so a saturated
                # cp=±20000 cannot outrank a forced mate. Use post-state
                # cp when available for zeroing moves that were re-eval'd
                # (audit B-04 draw-pollution guard); otherwise use the
                # multipv cp.
                eff_cp = (
                    eval_item.post_state_cp if eval_item.post_state_cp is not None else eval_item.cp
                )
                if eff_cp is not None:
                    mover_cp = sign * eff_cp
                    # Clamp so finite-cp wins never exceed the mate ceiling.
                    if mover_cp > MATE_RANK_CEILING:
                        return MATE_RANK_CEILING
                    if mover_cp < -MATE_RANK_CEILING:
                        return -MATE_RANK_CEILING
                    return float(mover_cp)

                return 0.0

            # Always sort (U-01): n-invariance requires a stable chess-correct
            # ordering regardless of halfmove / terminal state. Removing the
            # gate does not change behavior for positions where Stockfish's
            # raw order already matches the chess-correct order; it just
            # fixes the cases where it doesn't.
            res_list.sort(key=_candidate_rank_key, reverse=True)

            # Persist zeroing-move findings on the cache so the cache-hit path
            # below reuses the same policy decision without re-searching.
            await _cache.set_top_moves(cache_key, res_list)
            return res_list

        sf_key = f"{cache_key}:n={n}"
        res = cast(list[MCPEval], await _single_flight.do(sf_key, _compute))
        await metrics.record("top_moves", (time.time() - t0) * 1000, cache_hit=False)
        items = [c.model_copy(update={"requested_depth": raw_requested_depth}) for c in res[:n]]
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
            canonical_fen=canonical_fen,
            fen_was_canonicalized=fen_was_canonicalized,
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
        if "INVALID_VERBOSITY" in msg:
            code = "invalid_verbosity"
        elif "STRICT" in msg:
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
    move: str | None = None,
    moves: list[str] | None = None,
    depth: int = 14,
    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",
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
            Required for `play_move` and `claim_draw_with_intended_move`; optional for
            `claim_draw` (the claim outcome does not depend on any specific move).
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
        if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
            raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
        # P2 (2026-09-02 ultra audit): request-shape validation must run BEFORE
        # board-state validation. The audit found that `claim_draw` with a
        # supplied `move` argument on a non-claimable board returned
        # "draw cannot be claimed now" — a board-state error — instead of
        # the structural "claim_draw must not include a move" error. The
        # structural error is consistent regardless of position and lets
        # callers distinguish bad input from bad state. Same applies to
        # `play_move` / `claim_draw_with_intended_move` missing a move.
        if action_type == "claim_draw":
            if move is not None and move.strip() and move.strip() != "(none)":
                if strict:
                    raise ValueError(
                        f"STRICT_SAN_ERROR: action_type='claim_draw' must not "
                        f"include a `move` argument; got {move!r}. Pass move=None "
                        f"or omit the parameter."
                    )
                # Lenient mode: still record that the caller passed a
                # meaningless argument (per U-12 invariant — B-02 audit).
                # We surface this as a syntax_warning later via the response.
        else:
            if move is None or not move.strip():
                raise ValueError(
                    "MISSING_MOVE: 'move' is required for action_type='play_move' "
                    "and action_type='claim_draw_with_intended_move'"
                )
        board = _build_board(fen, moves or [], strict=strict)
        history_complete = _history_provenance_for_input(fen, moves)
        rule_before = evaluate_rule_status(board, history_complete=history_complete)
        # AUDIT B-01/B-02/B-03: for `claim_draw`, the dummy `move` argument must
        # not be parsed/executed; the claim outcome is purely procedural. Accept
        # `move=None` (or any string) but never push the move onto the board
        # when classifying a draw claim. `claim_draw_with_intended_move` still
        # requires a real intended move because the move IS the claim.
        if action_type == "claim_draw":
            chess_move: chess.Move | None = None
            syntax_warn: str | None = None
            if move is not None and move.strip() and move.strip() != "(none)":
                # P2 (2026-09-02 ultra audit): lenient mode still warns when
                # the caller passes a meaningless `move` argument to
                # `claim_draw`. Strict mode rejects outright (above). The
                # warning makes the structural mismatch observable without
                # breaking the claim.
                syntax_warn = (
                    f"action_type='claim_draw' ignores supplied move argument "
                    f"{move!r} (the claim outcome is purely procedural)."
                )
            # P2 (2026-09-02 ultra audit): terminal-state handling must
            # happen before action-specific claim validation so every
            # action on a finished board returns the same GAME_ALREADY_OVER
            # error, not a position-dependent ILLEGAL_ACTION variant.
            if is_terminal_position(board):
                raise ValueError(
                    f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over; "
                    f"no further actions can be taken on a finished game."
                )
            if not rule_before.can_claim_now:
                raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        else:
            assert move is not None and move.strip()  # shape validated above
            chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
            if (
                action_type == "claim_draw_with_intended_move"
                and chess_move.uci() not in rule_before.intended_claim_ucis
            ):
                raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        pool = await _get_analyzer_pool(ctx)

        # Cache key uses an empty/dummy move for claim_draw so the same
        # underlying position/action always maps to one cache entry, regardless
        # of the dummy `move` the caller passed (audit B-02 invariant).
        cache_move_uci = chess_move.uci() if chess_move is not None else ""
        cache_key = classify_cache_key(
            board,
            cache_move_uci,
            depth,
            action_type=action_type,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
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

        # Build played_san / board_after defensively: for claim_draw they are
        # NOT derived from any chess move because the claim is procedural.
        if chess_move is not None:
            played_san = board.san(chess_move)
            board_after = board.copy(stack=True)
            board_after.push(chess_move)
        else:
            played_san = None
            board_after = board.copy(stack=True)

        async def _compute() -> MCPMoveAnalysis:
            pool = await _get_analyzer_pool(ctx)

            if (
                chess_move is not None
                and hasattr(pool, "classify_move")
                and type(pool)
                not in (
                    AnalyzerPool,
                    TCPAnalyzerPool,
                )
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
                    history_complete=history_complete,
                )

            eval_before, _ = await _evaluate_game_position_cached(
                board,
                depth,
                pool,
                requested_depth=raw_requested_depth,
                history_complete=history_complete,
            )

            # AUDIT B-02/B-03: for draw-claim actions, the post-state is the
            # position AFTER the claim is granted, not after the supplied
            # (irrelevant) move is played. Re-evaluate the same root board so
            # the resulting `eval_after` reflects the draw outcome (cp=0,
            # outcome=draw) regardless of any dummy move the caller passed.
            if action_type in ("claim_draw", "claim_draw_with_intended_move"):
                eval_after, _ = await _evaluate_game_position_cached(
                    board,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )
                # The claim outcome is a draw; force cp=0 and outcome=draw so
                # every downstream caller sees a consistent post-claim state
                # independent of the dummy move.
                eval_after = _force_draw_outcome(eval_after)
            else:
                # Correctness first: eval_after must describe the immediate
                # post-move position. Reusing the root PV tail or root score
                # can misstate finite-depth CP and mate distance. Engine/cache
                # layers remain responsible for performance reuse.
                eval_after, _ = await _evaluate_game_position_cached(
                    board_after,
                    depth,
                    pool,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )

            if chess_move is not None:
                score = score_played_move(
                    board,
                    chess_move,
                    eval_before,
                    eval_after,
                    board_after,
                    action_type=action_type,
                )
            else:
                # claim_draw without a move: pass a placeholder Move and the
                # post-claim board (= root board). score_played_move still
                # consults rule_before.can_claim_now and the post-claim eval,
                # so the dummy Move here is purely structural and never
                # affects the score.
                placeholder = next(iter(board.legal_moves), None)
                if placeholder is None:
                    raise ValueError("ILLEGAL_ACTION: no legal moves; cannot evaluate claim")
                score = score_played_move(
                    board,
                    placeholder,
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
            # Audit P0/P1 (2026-09-01 adversarial probe): the verification
            # block is for `play_move` only. Draw-claim actions classify the
            # CLAIM, not the supplied move; the move may coincidentally match
            # `eval_before.best_move` (e.g. `claim_draw + Qc8#` where the
            # engine's best IS the mating move the player is refusing to play).
            # In that case the depth+4 verification correctly confirms the
            # move is the engine's best legal attempt — but that's irrelevant
            # to grading the CLAIM. Allowing the "else" branch to overwrite
            # `move_class=BEST, effective_loss=0` here violates the invariant
            # `is_best_action==False AND best outcome==win AND played
            # outcome==draw ⇒ effective_loss > 0`. Skip the whole block for
            # claim actions; the score from `score_played_move` is final.
            if (
                action_type == "play_move"
                and chess_move is not None
                and (
                    chess_move.uci().lower() == (eval_before.best_move or "").lower()
                    and score.move_class in (MoveClass.MISTAKE, MoveClass.BLUNDER)
                    and not score.missed_draw_claim
                    and not score.conceded_draw_claim
                )
            ):
                try:
                    # Cache the depth+4 verification result via the same
                    # L1/L2 path as any other eval. Previously this went
                    # straight to pool.evaluate, bypassing the cache — every
                    # classify_move that hit this verification path paid the
                    # full uncached depth+4 cost. Now the depth+4 result is
                    # cached like any other eval.
                    verify_eval_result, _verify_hit = await _evaluate_game_position_cached(
                        board,
                        depth + 4,
                        pool,
                        requested_depth=raw_requested_depth + 4,
                        history_complete=history_complete,
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
                            history_complete=history_complete,
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
            if score.is_best_engine_move and chess_move is not None:
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
            if eval_after.pv and not board_after.is_game_over() and chess_move is not None:
                played_continuation = pv_to_san(board_after, eval_after.pv)

            played_line_san = played_san
            if played_continuation and played_san is not None:
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

            played_uci = chess_move.uci() if chess_move is not None else ""
            mcp_analysis = MCPMoveAnalysis(
                played=played_uci,
                played_san=played_san,
                move_class=score.move_class,
                is_engine_best=score.is_best_engine_move,
                is_best_engine_move=score.is_best_engine_move,
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
                played_action_obj=build_played_action(
                    action_type,
                    move_uci=played_uci,
                    move_san=played_san,
                    rule_status=rule_before,
                    cp=eval_after.cp,
                    mate=eval_after.mate,
                ),
                best_action_obj=eval_before.best_action_obj,
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
        if "INVALID_ACTION_TYPE" in msg:
            code = "invalid_action_type"
        elif "ILLEGAL_ACTION" in msg:
            code = "illegal_action"
        elif "STRICT" in msg:
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
                rule_before = evaluate_rule_status(board_before, history_complete="complete")
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
        # U-06 (2026-09-01): reconcile `best_san` with the final classification.
        # Without this guard, an analyze_game turning point can report
        # `best_move_san == played_san` while `move_class == "blunder"` —
        # internally contradictory. The bug surfaced at depth=1 where the
        # engine's top line happens to be a losing move (audit U-06
        # promotion-defense reproducer). When the played move equals the
        # engine's reported best but the classifier decided it was a
        # blunder/mistake, suppress the best_move_san to avoid the
        # contradiction. The classify_move path runs a depth+4 verification
        # search to refine; analyze_game doesn't (per-ply cost), so the
        # conservative answer is `best_san = None` here.
        if eval_before.best_move and not (
            score.is_best_engine_move and score.move_class.value in ("blunder", "mistake")
        ):
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


def _infer_result_from_termination(termination: str | None) -> str | None:
    if not termination:
        return None
    t = re.sub(r"\s+", " ", termination.strip().lower())
    if "normal time control" in t:
        return None

    winner_patterns = (
        (r"\bwhite\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "1-0"),
        (r"\bblack\s+(?:wins?|won)\b.*\b(?:time|resignation|resigns?)\b", "0-1"),
        (r"\bwon\s+by\s+white\b", "1-0"),
        (r"\bwon\s+by\s+black\b", "0-1"),
    )
    for pattern, result in winner_patterns:
        if re.search(pattern, t):
            return result

    loser_patterns = (
        (r"\bwhite\s+(?:resign(?:s|ed)?|lost|loses)\b", "0-1"),
        (r"\bblack\s+(?:resign(?:s|ed)?|lost|loses)\b", "1-0"),
        (r"\bwhite(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "0-1"),
        (r"\bblack(?:'s)?\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "1-0"),
        (r"\bwhite\s+(?:lost|loses)\s+on\s+time\b", "0-1"),
        (r"\bblack\s+(?:lost|loses)\s+on\s+time\b", "1-0"),
    )
    for pattern, result in loser_patterns:
        if re.search(pattern, t):
            return result

    if re.search(r"\bwhite\b.*\b(?:illegal move|rules? infraction)\b", t):
        return "0-1"
    if re.search(r"\bblack\b.*\b(?:illegal move|rules? infraction)\b", t):
        return "1-0"
    return None


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def analyze_game(  # pyright: ignore[reportGeneralTypeIssues]
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
        sanitized_pgn, lexical_header_warnings = _sanitize_malformed_pgn_header_lines(
            pgn, strict=strict
        )
        _check_multiple_games(sanitized_pgn)
        if strict:
            _validate_strict_header_syntax(sanitized_pgn)
        canonical_pgn = _extract_canonical_pgn_text(sanitized_pgn)
        game = _extract_game_inner(canonical_pgn, strict=strict)
        if strict:
            _validate_strict_mainline_surface(canonical_pgn, game)

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

        # U-03 (2026-09-01): if the initial FEN is already terminal (75-move
        # draw, checkmate, stalemate, insufficient material, fivefold
        # repetition, dead position), the movetext's first move is bogus —
        # the board has no legal moves. Strict mode raises a
        # STRICT_PGN_ERROR. Non-strict mode records a syntax_warning and
        # treats every following move as a trailing ply so the analysis
        # surfaces 0 executed plies. Without this check the mainline loop
        # silently advanced `ignored_trailing_plies` without ever telling
        # the caller that the starting position was terminal.
        initial_rule = evaluate_rule_status(curr_board, history_complete="complete")
        if initial_rule.terminal is not None:
            auto_termination = initial_rule.terminal
            reached_terminal = True
            if strict:
                raise ValueError(
                    f"STRICT_PGN_ERROR: Initial FEN '{curr_board.fen()}' is already "
                    f"terminal ({initial_rule.terminal}); cannot execute movetext."
                )
            syntax_warnings.append(
                f"Initial FEN is terminal ({initial_rule.terminal}); "
                f"all movetext moves will be ignored."
            )

        # P3/INVESTIGATE (2026-09-02 ultra audit): NAG values outside the
        # PGN 0..255 range were silently dropped in lenient mode (the
        # `_validate_movetext_tokens` helper only flags them in strict
        # mode). Re-scan the movetext here so lenient callers also see
        # the warning. Strict mode still rejects via the helper's
        # `invalid_tokens` branch; this scan catches the lenient case
        # without regressing strict behavior. Comments and variations are
        # already stripped from `cleaned_movetext` below, so the regex
        # scans the same tokens the strict path consumes.
        for nag_match in re.finditer(r"\$([0-9]+)", canonical_pgn):
            nag_val = int(nag_match.group(1))
            if nag_val > 255:
                if strict:
                    # Strict mode: promote to a metadata_warning so the
                    # final pass at the bottom raises STRICT_PGN_ERROR
                    # (mirrors the existing NAG enforcement path).
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )
                else:
                    syntax_warnings.append(
                        f"NAG value ${nag_val} outside the PGN-supported range 0..255."
                    )

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
        if re.search(
            r"(?:^|\s)\(?e\.?p\.?\)?(?=\s|$)",
            cleaned_movetext,
            flags=re.IGNORECASE,
        ):
            syntax_warnings.append("En-passant marker 'e.p.' normalized to canonical SAN.")
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
            # NOTE: do NOT strip the trailing dots here — the U-15 side
            # marker check needs the original dot count. The previous
            # code stripped dots which made actual_dots empty for any
            # single-dot or triple-dot token, and the side-marker check
            # would then fire as a false positive.
            while tok_idx < len(movetext_tokens):
                raw_tok = movetext_tokens[tok_idx]
                # U-15 (2026-09-01): the previous pattern `(\.|\.\.)*` was
                # a Python regex footgun — alternation inside a `*` group
                # never extends beyond a single match, so group(2) was
                # always "." regardless of how many dots were in the
                # input. That made the wrong-side-marker check a no-op
                # (the actual and expected dots were always the same).
                # `\.+` captures the full dot run in one shot.
                num_m = re.match(r"^(\d+)(\.+)$", raw_tok)
                if num_m:
                    move_num = int(num_m.group(1))
                    if move_num != expected_fullmove:
                        syntax_warnings.append(
                            f"Move number mismatch: found '{movetext_tokens[tok_idx]}' but expected move {expected_fullmove}."
                        )
                    # U-15 (2026-09-01): also flag wrong-dot count. A black
                    # move (board.turn == BLACK at this point in the
                    # mainline) MUST use "..." (triple dot), not ".".
                    # Strict mode promotes the warning to a STRICT_PGN_ERROR;
                    # the final pass at the bottom raises on any
                    # syntax_warnings in strict mode.
                    expected_dots = "..." if curr_board.turn == chess.BLACK else "."
                    actual_dots = num_m.group(2) or ""
                    if actual_dots != expected_dots:
                        syntax_warnings.append(
                            f"Wrong side marker: found '{movetext_tokens[tok_idx]}' "
                            f"but expected '{expected_dots}' for the side to move."
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

            if curr_board.is_repetition(5):
                reached_terminal = True
                auto_termination = "fivefold_repetition"
            else:
                rule_after = evaluate_rule_status(curr_board, history_complete="complete")
                if rule_after.terminal is not None:
                    reached_terminal = True
                    auto_termination = rule_after.terminal

        # Extract headers with TAG_PAIR_REGEX from header_section to handle escaped quotes and robust tag parsing
        # P2 (2026-09-02 ultra audit): the tag name MUST be canonicalized
        # before storage in tags_dict — otherwise [Variant "Standard"] and
        # [variant "Standard"] produced different downstream lookups (Variant
        # returned None from the second form). The metadata pipeline now
        # uses lowercase keys consistently. Downstream code reads both the
        # canonical-key form (lowercase, e.g. "variant") and falls back to
        # python-chess's game.headers for whatever it parsed.
        tags_dict: dict[str, str] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_k = tag_m.group(1).lower()
            tag_v = _unescape_pgn_tag_value(tag_m.group(2))
            if tag_k not in tags_dict and tag_v is not None and tag_v != "?":
                tags_dict[tag_k] = tag_v

        h = game.headers
        white_name = tags_dict.get("white") or (
            _unescape_pgn_tag_value(h.get("White"))
            if h.get("White") and h.get("White") != "?"
            else None
        )
        black_name = tags_dict.get("black") or (
            _unescape_pgn_tag_value(h.get("Black"))
            if h.get("Black") and h.get("Black") != "?"
            else None
        )
        event_name = tags_dict.get("event") or (
            _unescape_pgn_tag_value(h.get("Event"))
            if h.get("Event") and h.get("Event") != "?"
            else None
        )
        site_name = tags_dict.get("site") or (
            _unescape_pgn_tag_value(h.get("Site"))
            if h.get("Site") and h.get("Site") != "?"
            else None
        )
        round_name = tags_dict.get("round") or (
            _unescape_pgn_tag_value(h.get("Round"))
            if h.get("Round") and h.get("Round") != "?"
            else None
        )
        white_elo_val = tags_dict.get("whiteelo") or (
            h.get("WhiteElo") if h.get("WhiteElo") and h.get("WhiteElo") != "?" else None
        )
        black_elo_val = tags_dict.get("blackelo") or (
            h.get("BlackElo") if h.get("BlackElo") and h.get("BlackElo") != "?" else None
        )
        time_control_val = tags_dict.get("timecontrol") or (
            h.get("TimeControl") if h.get("TimeControl") and h.get("TimeControl") != "?" else None
        )
        # P2 (2026-09-02 ultra audit): TimeControl values like "? ",
        # " ?", " ? " were preserved verbatim on the wire — the strict
        # validator allowed them through, but downstream consumers saw
        # a different literal than the canonical sentinel. Strip
        # whitespace and normalize the unknown sentinel to None so the
        # exposed value is consistent across inputs.
        if time_control_val is not None:
            time_control_val = time_control_val.strip()
            if time_control_val == "?":
                time_control_val = None
        variant_val = tags_dict.get("variant") or (
            h.get("Variant") if h.get("Variant") and h.get("Variant") != "?" else None
        )
        date_val = (
            tags_dict.get("date") or tags_dict.get("utcdate") or h.get("Date") or h.get("UTCDate")
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

        metadata_warnings: list[str] = list(lexical_header_warnings)

        # U-14 (2026-09-01): strict mode rejects malformed Date tags.
        # PGN Date is `YYYY.MM.DD`; anything else (e.g. "2026.99.99",
        # "hello", "not.a.date") is a non-canonical value. The legacy
        # behavior silently accepted these and even echoed them back
        # to clients, which is the audit's P2 finding. Strict mode
        # records a metadata_warning; the final strict pass at the
        # bottom of analyze_game raises STRICT_PGN_ERROR on any
        # metadata_warning, so the malformed Date is rejected. The
        # regex is tighter than just `\d{4}\.\d{2}\.\d{2}` — it
        # enforces month 01-12 and day 01-31 so "2026.99.99" is
        # correctly rejected.
        #
        # P2/P3 (2026-09-02 ultra audit): the regex above only catches
        # range issues; it still accepts impossible calendar dates like
        # 2023.02.29, 2026.04.31, 2026.02.31. After the structural
        # check, run the date through Python's `datetime.date`
        # constructor — that raises ValueError for any day that doesn't
        # exist in the given month/year, including the Feb 29 leap-year
        # rule (no Apr 31, no Sep 31, no Feb 30, etc.). In strict mode
        # the impossible date is a metadata_warning that promotes to a
        # STRICT_PGN_ERROR; in lenient mode it is also a warning so
        # downstream callers see that the metadata is suspect, even
        # though parsing continues.
        if date_val is not None:
            if not re.fullmatch(
                r"(?:19|20)\d{2}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])",
                date_val,
            ):
                if strict:
                    metadata_warnings.append(
                        f"Invalid Date tag '{date_val}': must match YYYY.MM.DD "
                        f"with month 01-12 and day 01-31."
                    )
                else:
                    # Lenient: surface the issue without rejecting. A caller
                    # who is reading metadata only needs to see the warning.
                    metadata_warnings.append(
                        f"Invalid Date tag '{date_val}': must match YYYY.MM.DD "
                        f"with month 01-12 and day 01-31."
                    )
            else:
                # Structural regex passed — now verify calendar semantics.
                # datetime.date raises ValueError for impossible dates
                # (Apr 31, Feb 30, Feb 29 in a non-leap year, etc.).
                try:
                    yy, mm, dd = (int(p) for p in date_val.split("."))
                    import datetime as _dt

                    _dt.date(yy, mm, dd)
                except ValueError as exc:
                    if strict:
                        metadata_warnings.append(f"Invalid Date tag '{date_val}': {exc}.")
                    else:
                        metadata_warnings.append(f"Impossible Date tag '{date_val}': {exc}.")

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
        if time_control_val is not None and not _is_valid_pgn_time_control(time_control_val):
            metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")

        eco_header = tags_dict.get("eco") or h.get("ECO")
        opening_header = tags_dict.get("opening") or h.get("Opening")

        # Detect duplicate headers in header block only. P2 (2026-09-02
        # ultra audit): the tag name MUST be canonicalized (lowercased)
        # before duplicate counting — otherwise `[Result "*"]` and
        # `[result "1-0"]` were treated as different tags despite
        # python-chess treating them as the same semantic tag. We also
        # surface value conflicts on the canonical Result tag because
        # the audit flagged that competing values were silently merged.
        tag_counts: dict[str, int] = {}
        tag_values_by_canonical: dict[str, list[str]] = {}
        for tag_m in TAG_PAIR_REGEX.finditer(header_section):
            tag_name_raw = tag_m.group(1)
            tag_value = _unescape_pgn_tag_value(tag_m.group(2))
            tag_name_canonical = tag_name_raw.lower()
            tag_counts[tag_name_canonical] = tag_counts.get(tag_name_canonical, 0) + 1
            if tag_value is not None:
                tag_values_by_canonical.setdefault(tag_name_canonical, []).append(tag_value)
        for tag_name, count in tag_counts.items():
            if count > 1:
                metadata_warnings.append(
                    f"Duplicate PGN tag '[{tag_name}]' detected ({count} occurrences); using canonical tag value."
                )
        # Surface value conflicts on Result / Variant explicitly so the
        # audit's "duplicate detection is not consistently
        # case-insensitive" finding is closed.
        for canonical_name in ("result", "variant"):
            values = tag_values_by_canonical.get(canonical_name) or []
            if len(values) >= 2 and any(v != values[0] for v in values[1:]):
                metadata_warnings.append(
                    f"Conflicting values for PGN tag '{canonical_name}': {values!r}; "
                    f"using the first declared value."
                )

        # Validate SetUp vs FEN tags
        setup_header = h.get("SetUp")
        fen_header = h.get("FEN")
        # P2 (2026-09-02 ultra audit): SetUp tag value domain must be
        # validated. The legacy code special-cased the canonical "1"
        # string and silently accepted every other value (including
        # non-canonical "2", empty string, "true", "false", "01", "-1",
        # and " ") — which the audit showed meant strict mode never
        # rejected malformed SetUp values. Strict mode now rejects any
        # value outside the canonical {"0", "1"} set; lenient mode
        # accepts "1" only (treating everything else as the implicit
        # "SetUp absent" case, with a warning).
        if setup_header is not None:
            if setup_header not in ("0", "1"):
                if strict:
                    metadata_warnings.append(
                        f"Invalid SetUp tag value '{setup_header}': must be exactly '0' or '1'."
                    )
                else:
                    # Lenient: warn but don't reject — preserve
                    # backward compatibility for slightly-malformed
                    # inputs that the caller may not be able to fix.
                    metadata_warnings.append(
                        f"Non-canonical SetUp tag value '{setup_header}': expected '0' or '1'."
                    )
        if setup_header == "1" and not fen_header:
            metadata_warnings.append(
                '[SetUp "1"] tag provided without FEN tag; defaulting to standard starting position.'
            )
        elif fen_header and setup_header != "1":
            metadata_warnings.append(
                'FEN tag provided without [SetUp "1"]; custom position loaded.'
            )

        if game.errors:
            # P2 (2026-09-02 ultra audit): board-detected checkmate path
            # undercounted trailing plies. The legacy code added
            # `len(game.errors)` which is the number of distinct
            # python-chess exceptions raised while parsing — usually 1
            # per movetext that breaks at the first illegal move — rather
            # than the actual number of trailing ply tokens the user
            # wrote. The explicit result-token branch already counted
            # all SAN tokens after the result marker; we now mirror that
            # behavior here, counting remaining movetext tokens after the
            # last successfully executed ply.
            consumed_plies = len(moves)
            tokens_in_movetext = re.findall(
                r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[\+#\?!]*|O-O-O[\+#\?!]*|O-O[\+#\?!]*)\b",
                cleaned_movetext,
            )
            total_ply_tokens = len(tokens_in_movetext)
            trailing_from_errors = max(0, total_ply_tokens - consumed_plies)
            if trailing_from_errors > 0:
                ignored_trailing_plies += trailing_from_errors
            else:
                # No recoverable move tokens found in the trailing
                # movetext. Fall back to the legacy game.errors
                # count so we never underreport below zero.
                ignored_trailing_plies = max(ignored_trailing_plies, len(game.errors))

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
        # positions[] was reconstructed from the complete PGN mainline, so repetition
        # history is authoritative here. Do not downgrade a previously detected fivefold
        # repetition to generic game_over during final result reconciliation.
        rule_final = evaluate_rule_status(final_board, history_complete="complete")
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

        # Infer only from explicit winner/loser grammar.
        if result_val == "*" or result_val is None:
            inferred = _infer_result_from_termination(termination_header_val)
            if inferred is not None:
                result_val = inferred

        # Validate Resignation & Time Forfeit & Rules Infraction under FIDE mating possibility rules
        result_val, mate_warnings = validate_mating_possibility(
            final_board, result_val, termination_header_val
        )
        metadata_warnings.extend(mate_warnings)

        # U-14 (2026-09-01): strict-mode Termination validation. The
        # legacy code only flagged Termination when it contradicted
        # the result; arbitrary strings like "foobar" were stored raw
        # without rejection. Strict mode now requires the Termination
        # to either be blank, a known FIDE value, or fall through the
        # normalize_termination mapper; anything else is a metadata
        # warning that strict pass will reject.
        if strict and termination_header_val:
            norm_term = normalize_termination(termination_header_val)
            # If normalize_termination returns None AND the string is
            # not blank/known, it's an unrecognised value.
            if norm_term is None and termination_header_val.strip() not in (
                "",
                "Normal",
                "Time forfeit",
                "Rules infraction",
                "Abandoned",
                "Unterminated",
            ):
                # If normalize_termination returned None but
                # contains a known FIDE term in lowercase, accept.
                lower = termination_header_val.strip().lower()
                if not any(
                    kw in lower
                    for kw in (
                        "resign",
                        "checkmate",
                        "stalemate",
                        "time",
                        "abandon",
                        "rule",
                        "draw",
                        "repetition",
                        "insufficient",
                        "50-move",
                        "75-move",
                    )
                ):
                    metadata_warnings.append(
                        f"Unrecognised Termination tag '{termination_header_val}'."
                    )

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

        if strict and not moves:
            if syntax_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains syntax normalization or move number mismatch: {syntax_warnings[0]}"
                )
            if metadata_warnings:
                raise ValueError(
                    f"STRICT_PGN_ERROR: PGN contains metadata inconsistency: {metadata_warnings[0]}"
                )

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
            positions,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
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


def _is_trusted_proxy_peer(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _effective_client_ip(peer_ip: str, forwarded_for: str) -> str:
    peer = peer_ip.strip().strip("[]")
    if forwarded_for and _is_trusted_proxy_peer(peer):
        candidate = forwarded_for.split(",", 1)[0].strip().strip("[]")
        return candidate or peer
    return peer


def _estimate_mcp_request_cost(body: bytes) -> float:
    """Approximate CPU admission cost from tool, depth, MultiPV and PGN size."""
    try:
        payload_any: Any = json.loads(body.decode("utf-8"))
        payload = cast(dict[str, Any], payload_any) if isinstance(payload_any, dict) else {}
        params_any: Any = payload.get("params")
        params = cast(dict[str, Any], params_any) if isinstance(params_any, dict) else {}
        tool_name = str(params.get("name") or params.get("tool") or "")
        args_any: Any = params.get("arguments")
        args = cast(dict[str, Any], args_any) if isinstance(args_any, dict) else {}
        depth = max(1, min(int(args.get("depth", 14)), 30))
        if tool_name == "evaluate_position":
            return 1.0 + depth / 14.0
        if tool_name == "top_moves":
            n = max(1, min(int(args.get("n", 3)), 20))
            return 1.0 + (depth * n) / 14.0
        if tool_name == "classify_move":
            return 2.0 + depth / 10.0
        if tool_name == "analyze_game":
            pgn = str(args.get("pgn", ""))
            estimated_plies = max(1.0, min(200.0, len(pgn) / 24.0))
            return 5.0 + (depth * estimated_plies) / 28.0
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return 1.0


class TokenBucketRateLimiter:
    """In-memory weighted token bucket per effective client IP."""

    def __init__(self, rate: float = 20.0, capacity: float = 500.0) -> None:
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
                    self._buckets = dict(
                        sorted(
                            self._buckets.items(),
                            key=lambda item: item[1][1],
                            reverse=True,
                        )[:5000]
                    )
            tokens, last_time = self._buckets.get(client_id, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last_time) * self.rate)
            if tokens >= cost:
                self._buckets[client_id] = (tokens - cost, now)
                return True
            self._buckets[client_id] = (tokens, now)
            return False


class ASGIRequestLoggerMiddleware:
    """Request logging, weighted admission control and token authentication."""

    MAX_BUFFERED_BODY = 32 * 1024 * 1024

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
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
        headers_dict: dict[bytes, bytes] = dict(raw_headers)
        ua = headers_dict.get(b"user-agent", b"").decode("utf-8", "ignore")
        origin = headers_dict.get(b"origin", b"").decode("utf-8", "ignore")
        host = headers_dict.get(b"host", b"").decode("utf-8", "ignore")
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "")).upper()
        session_id = headers_dict.get(b"mcp-session-id", b"").decode("utf-8", "ignore")

        client_tuple = cast(tuple[str, int] | None, scope.get("client"))
        peer_ip = client_tuple[0] if client_tuple else "127.0.0.1"
        forwarded_for = headers_dict.get(b"x-forwarded-for", b"").decode("utf-8", "ignore")
        client_ip = _effective_client_ip(peer_ip, forwarded_for)

        if method == "OPTIONS":
            await send(
                cast(
                    Message,
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"access-control-allow-origin", b"*"),
                            (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                            (b"access-control-allow-headers", b"*"),
                            (b"access-control-max-age", b"86400"),
                            (b"content-length", b"0"),
                        ],
                    },
                )
            )
            await send(cast(Message, {"type": "http.response.body", "body": b""}))
            return

        # Authentication is header-only. Reject an unauthenticated caller before
        # reading/buffering a potentially large POST body, otherwise the auth wall
        # itself can be used as a memory/CPU amplification point.
        if path != "/health" and self.restrict_to_chatgpt:
            from .config import get_mcp_settings

            auth_token = get_mcp_settings().auth_token
            provided = headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
            authorization = (
                headers_dict.get(b"authorization", b"").decode("utf-8", "ignore").strip()
            )
            bearer = (
                authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            )

            def token_matches(candidate: str) -> bool:
                return bool(auth_token) and hmac.compare_digest(candidate, auth_token)

            if not (token_matches(provided) or token_matches(bearer)):
                log.warning(
                    "Blocked unauthenticated client ip=%s ua=%r origin=%r", client_ip, ua, origin
                )
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: valid MCP auth token required"}}\n'
                await send(
                    cast(
                        Message,
                        {
                            "type": "http.response.start",
                            "status": 403,
                            "headers": [
                                (b"content-type", b"application/json"),
                                (b"content-length", str(len(response_body)).encode("ascii")),
                            ],
                        },
                    )
                )
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

        request_cost = 1.0
        if method == "POST":
            # Enforce the body-size cap via Content-Length only. We deliberately
            # do NOT buffer the request body here: doing so breaks FastMCP's
            # streamable-HTTP transport, which uses an SSE writer that calls
            # receive() to detect early client disconnect during long-running
            # tool calls. A synthetic replay callable returns http.disconnect
            # on the second poll, and EventSourceResponse in sse_starlette
            # closes the stream before it can emit the response — every
            # initialize (and any tool call long enough to flush an SSE
            # chunk) then 500s with "ASGI callable returned without starting
            # response". The Content-Length check is reliable enough for a
            # public edge: uvicorn rejects missing/malformed headers before
            # they reach us, and a client that lies about Content-Length is
            # bounded by the transport frame limit and the token-bucket.
            raw_content_length = (
                headers_dict.get(b"content-length", b"").decode("ascii", "ignore").strip()
            )
            if raw_content_length:
                try:
                    declared_length = int(raw_content_length)
                except ValueError:
                    declared_length = -1
                if declared_length > self.MAX_BUFFERED_BODY:
                    response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Request body too large"}}\n'
                    await send(
                        cast(
                            Message,
                            {
                                "type": "http.response.start",
                                "status": 413,
                                "headers": [
                                    (b"content-type", b"application/json"),
                                    (b"content-length", str(len(response_body)).encode("ascii")),
                                ],
                            },
                        )
                    )
                    await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                    return
                if declared_length > 8 * 1024:
                    request_cost = 5.0 + min(40.0, declared_length / 1024.0)
                elif declared_length > 1024:
                    request_cost = 2.0

        if not await self.rate_limiter.is_allowed(client_ip, cost=request_cost):
            response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Rate limit exceeded. Please slow down."}}\n'
            await send(
                cast(
                    Message,
                    {
                        "type": "http.response.start",
                        "status": 429,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"content-length", str(len(response_body)).encode("ascii")),
                            (b"retry-after", b"2"),
                        ],
                    },
                )
            )
            await send(cast(Message, {"type": "http.response.body", "body": response_body}))
            return

        log.info(
            "%s %s host=%s ip=%s session=%s origin=%s ua=%s cost=%.2f",
            method,
            path,
            host,
            client_ip,
            session_id,
            origin,
            ua,
            request_cost,
        )

        if path == "/health":
            await self.app(scope, receive, send)
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
