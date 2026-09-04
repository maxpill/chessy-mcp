"""``analyze_game`` MCP tool.

Thin entry point. The end-to-end orchestration lives in
:class:`mcp_server.analysis.game_analyzer.GameAnalyzer` — this module
just unwraps the FastMCP ``Context``, forwards the call, and translates
:class:`ToolError` failures consistently with the other three analysis
tools.
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.analysis.game_analyzer import GameAnalyzer
from mcp_server.metrics import metrics
from mcp_server.models import GameAnalysisResult
from mcp_server._mcp import mcp
from mcp_server.tools._common import (
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

log = logging.getLogger("chessy_mcp.analyze_game")

_ANALYZER = GameAnalyzer.with_defaults()


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def analyze_game(  # pyright: ignore[reportGeneralTypeIssues]
    pgn: str,
    depth: int = 18,
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
        depth: Search depth per move (default 18, clamped 1-30). ``analyze_game`` fans
            one Stockfish search per mainline ply — default 18 trims compute vs the
            previous d14 while staying accurate enough to separate inaccuracy/mistake
            classes. For "find the turning points" mode, d18 is enough. For precise
            post-mortems where borderline decisions matter, push to 20 or selectively
            re-classify borderline plies at d22-24. Avoid going above d24 except for
            3-7 critical positions: nodes scale roughly 5x from d20→d24 and ~10x to d30
            (Stockfish 18 figures), with sharply diminishing Elo per added depth.
        strict: When True, reject non-canonical SAN syntax, move number mismatches, or metadata discrepancies (default False).

    Returns:
         GameAnalysisResult with player accuracy %, ACPL, blunder/mistake counts, turning points, and game metadata.
    """
    depth = _validate_requested_depth(depth, tool="analyze_game")
    try:
        return await _ANALYZER.analyze(
            pgn=pgn,
            depth=depth,
            strict=strict,
            ctx=ctx,
            metrics=metrics,
        )
    except ToolError:
        await metrics.record("analyze_game", 0.0, is_error=True)
        raise
    except ValueError as exc:
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="analyze_game", input=pgn[:100]) from exc
    except Exception as exc:
        await metrics.record("analyze_game", 0.0, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="analyze_game") from exc
