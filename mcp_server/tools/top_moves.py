"""``top_moves`` MCP tool.

Thin entry point. The end-to-end orchestration lives in
:class:`mcp_server.analysis.top_moves_finder.TopMovesFinder`. This module
just unwraps the FastMCP context, normalizes inputs, forwards the call,
and translates ``ToolError`` failures consistently.
"""

from __future__ import annotations

import logging
import time

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.analysis.top_moves_finder import TopMovesFinder
from mcp_server.metrics import metrics
from mcp_server.models import TopMovesResult
from mcp_server.server import mcp
from mcp_server.tools._common import (
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

log = logging.getLogger("chessy_mcp.top_moves")

_FINDER = TopMovesFinder.with_defaults()


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def top_moves(
    fen: str,
    moves: list[str] | None = None,
    n: int = 3,
    depth: int = 20,
    strict: bool = False,
    verbosity: str | None = None,
    ctx: Context | None = None,
) -> TopMovesResult:
    """Get the top N candidate moves for a position, ranked best first.

    Args:
        fen: FEN or PGN string for the position.
        moves: Optional UCI or SAN moves to replay onto the position first.
        n: Number of candidates to return (default 3, clamped 1-20).
        depth: Stockfish search depth (default 20, clamped 1-30).
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).

    Returns:
        TopMovesResult with ranked candidates and rule-aware action surface.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="top_moves")
    raw_requested_depth = max(1, min(depth, 30))
    raw_requested_n = n
    clamped_n = max(1, min(n, 20))
    try:
        verbosity_mode = _resolve_verbosity(verbosity)
        out = await _FINDER.run(
            fen=fen,
            moves=moves,
            n=clamped_n,
            depth=raw_requested_depth,
            raw_requested_depth=raw_requested_depth,
            raw_requested_n=raw_requested_n,
            clamped_n=clamped_n,
            strict=strict,
            verbosity_mode=verbosity_mode,
            ctx=ctx,
        )
        await metrics.record(
            "top_moves",
            (time.time() - t0) * 1000,
            cache_hit=out.cache_hit,
        )
        return out.result
    except ToolError:
        await metrics.record("top_moves", 0.0, is_error=True)
        raise
    except ValueError as exc:
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="top_moves", input=fen) from exc
    except Exception as exc:
        await metrics.record("top_moves", 0.0, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="top_moves") from exc
