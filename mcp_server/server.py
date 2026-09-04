from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from core.engines.pool import AnalyzerPool
from mcp_server.cache import (
    CACHE_VERSION as CACHE_VERSION,
)
from mcp_server.cache import (  # noqa: F401
    classify_cache_key,
    eval_cache_key,
    MultiTierCache,
    SingleFlight,
    top_moves_cache_key,
)
from mcp_server.metrics import metrics  # noqa: F401


# Tools _common helpers (verbosity / error formatting / validation) — used
# by the tool modules. Re-exported here so the test suite can keep importing
# them via server_module._tool_error etc.
from mcp_server.tools._common import (  # noqa: F401
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    _compact_mcpeval,
    _format_exception,
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
    normalize_termination,
)
from mcp_server.models import (
    GameAnalysisResult,  # noqa: F401
    MCPEval,  # noqa: F401
    MCPMoveAnalysis,  # noqa: F401
    PlyAnalysisItem,  # noqa: F401
    TopMovesResult,  # noqa: F401
)
from mcp_server.move_grading import score_played_move  # noqa: F401
from mcp_server.tcp_analyzer import TCPAnalyzerPool


log = logging.getLogger("chessy_mcp.server")


# Draw-claim projection — implementation lives in mcp_server.claims.draw_projection.
# Bound as a module attribute so existing call sites keep working unchanged.
from mcp_server.claims.draw_projection import _force_draw_outcome  # noqa: E402,F401


# PGN / FEN / SAN parsing — implementation lives in mcp_server.parsers.
# All parser helpers used by the tools are bound here so existing call
# sites (``_build_board(...)``, ``_extract_game(...)``, ...) keep working
# unchanged. Underscored + unprefixed names are both exposed.
from mcp_server.parsers import (  # noqa: E402,F401
    SUPPORTED_VARIANTS,
    TAG_PAIR_REGEX,
    _build_board,
    _build_board_with_metadata,
    _check_multiple_games,
    _clean_conversational_text,
    _extract_canonical_pgn_text,
    _extract_game,
    _extract_game_inner,
    _find_movetext_result,
    _has_completed_game_before,
    _history_provenance_for_input,
    _infer_result_from_termination,
    _is_canonical_tag_line,
    _is_prose_line,
    _mask_comments_and_escapes,
    _normalize_multiline_tags,
    _normalize_movetext_figurines,
    _normalize_unicode_pgn_results,
    _parse_move_on_board,
    _parse_move_on_board_with_warning,
    _parse_pgn_game_candidate,
    _sanitize_brackets_in_variations_and_comments,
    _sanitize_malformed_pgn_header_lines,
    _stage_has_positive_number,
    _strict_top_level_movetext_tokens,
    _strip_pgn_escape_lines,
    _strip_promotion_eq,
    _truncate_movetext_at_result,
    _unescape_pgn_tag_value,
    _validate_castling_rights,
    _validate_fen_counters,
    _validate_movetext_tokens,
    _validate_pgn_date,
    _validate_strict_header_syntax,
    _validate_strict_mainline_surface,
    _validate_variant,
    _is_valid_pgn_time_control,
)


# Re-export facade for the engine layer. The implementation now lives in
# ``mcp_server.engine.pool_factory`` and ``mcp_server.engine.identity``;
# every public/private symbol is re-bound here so existing call sites
# (and ``monkeypatch.setattr(server_module, ...)`` in the test suite)
# keep working unchanged.
from mcp_server.engine import (  # noqa: E402,F401
    _create_analyzer_pool,
    _eval_via_analyzer_or_pool,
    _evaluate_game_position_cached,
    _gather_evaluate_positions_bounded,
    _get_analyzer_pool,
    _get_evaluate_semaphore,
    _maybe_ponder_warm,
    _mcp_lifespan,
    _pool_stats_logger,
    _pool_supports_root_moves,
    _ponder_warm_cache,
    cache as _cache,
    close_analyzer_pool,
    single_flight as _single_flight,
    build_identity as _build_identity,
    engine_config as _engine_config,
    package_version as _package_version,
    git_sha as _build_sha,
    stockfish_path as _stockfish_path,
)

# Pool / semaphore globals live HERE (not in pool_factory) because the test
# suite mutates ``server_module._analyzer_pool`` directly to install mock
# analyzers. ``pool_factory`` reads and writes these exact bindings at call
# time so the mock reaches the live lookup path.
_analyzer_pool: AnalyzerPool | TCPAnalyzerPool | None = None
_pool_lock = asyncio.Lock()
_evaluate_semaphore: asyncio.Semaphore | None = None
_evaluate_semaphore_lock = asyncio.Lock()


# MCP server instance. Defined BEFORE the tool imports so that
# ``from mcp_server.server import mcp`` (used by tool modules) resolves
# without a circular-import error.
mcp = MCPServer(
    "chess-analysis",
    description="Streamable Stockfish chess analysis and move grading MCP server",
    lifespan=_mcp_lifespan,
)


# MCP tools — implementations live in mcp_server.tools. Importing
# each one here triggers the ``@mcp.tool(...)`` decorator so FastMCP
# registers them on the server instance. Existing call sites that use
# ``server_module.evaluate_position(...)`` etc. keep working unchanged.
from mcp_server.tools.evaluate_position import evaluate_position  # noqa: E402,F401
from mcp_server.tools.top_moves import top_moves  # noqa: E402,F401
from mcp_server.tools.classify_move import classify_move  # noqa: E402,F401
from mcp_server.tools.analyze_game import analyze_game  # noqa: E402,F401
from mcp_server.tools.game_metrics import _compute_game_metrics  # noqa: E402,F401


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


# Middleware re-exports — implementation lives in mcp_server.middleware.
# Each name is bound here as a module attribute so existing call sites
# (and ``monkeypatch.setattr(server_module, \"ASGIRequestLoggerMiddleware\", ...)``)
# keep working unchanged.
from mcp_server.middleware import (  # noqa: E402,F401
    ASGIRequestLoggerMiddleware,
    TokenBucketRateLimiter,
    _build_app,
    _effective_client_ip,
    _estimate_mcp_request_cost,
    _is_trusted_proxy_peer,
    main,
)


if __name__ == "__main__":
    main()
