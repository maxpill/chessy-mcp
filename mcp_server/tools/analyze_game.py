"""``analyze_game`` MCP tool.

Thin entry point. The end-to-end orchestration lives in
:class:`mcp_server.analysis.game_analyzer.GameAnalyzer` - this module
just unwraps the FastMCP ``Context``, forwards the call, and translates
:class:`ToolError` failures consistently with the other three analysis
tools.
"""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server._mcp import mcp
from mcp_server.analysis.game_analyzer import GameAnalyzer
from mcp_server.analysis.game_termination import build_game_termination_assessment
from mcp_server.metrics import metrics
from mcp_server.models.game_coaching import ForensicGameAnalysisResult
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
    detail: Literal["standard", "coach", "forensic"] = "standard",
    perspective: Literal["white", "black"] = "white",
    max_critical_moments: int = 6,
    ctx: Context | None = None,
) -> ForensicGameAnalysisResult:
    """Analyze a full PGN, optionally as a structured coaching post-mortem.

    The default ``detail="standard"`` path preserves the existing accuracy,
    ACPL, mistake counts and legacy ``turning_points`` behavior.

    ``detail="coach"`` adds an evidence-first story of the game without extra
    engine searches: perspective-relative game segments, advantage/recovery
    events, 1-7 pedagogically selected critical moments, positive resources,
    root-cause-to-materialization links, player comments from the PGN mainline
    and a final-position defensibility snapshot.

    ``detail="forensic"`` automatically deep-verifies only the selected
    critical positions. A normal d18 scan is re-checked around d22, d20 around
    d24, and unstable classifications can escalate selectively to at most d26.
    It also measures top-2 candidate gaps, tags strongest forcing replies, marks
    unique defensive resources and counts reasonable final-position resources.

    Rich modes also attach an evidence-bounded termination assessment. An
    explicit resignation-style PGN ``Termination`` header is treated as confirmed
    resignation; a decisive result on a non-terminal board without that header is
    only a resignation candidate because timeout/adjudication remain possible.
    ``objectively_forced`` is true only for rules termination/forced mate, never
    merely because the engine evaluation is strongly negative.

    ``perspective`` controls whose practical story is selected. The legacy
    engine metrics still cover both sides. ``max_critical_moments`` is clamped
    to 1-7 so full-game output stays coach-like instead of becoming an engine
    dump.

    Annotated PGN comments are retained as ``user_comment_raw`` on the matching
    critical ply. Chess MCP reports the board/engine evidence but deliberately
    does not infer a human process label such as "incomplete CCT" on its own.
    """
    depth = _validate_requested_depth(depth, tool="analyze_game")
    if detail not in {"standard", "coach", "forensic"}:
        raise _tool_error(
            code="invalid_argument",
            message=f"INVALID_DETAIL: {detail}",
            tool="analyze_game",
        )
    if perspective not in {"white", "black"}:
        raise _tool_error(
            code="invalid_argument",
            message=f"INVALID_PERSPECTIVE: {perspective}",
            tool="analyze_game",
        )
    if max_critical_moments < 1 or max_critical_moments > 7:
        raise _tool_error(
            code="invalid_argument",
            message="INVALID_MAX_CRITICAL_MOMENTS: expected 1..7",
            tool="analyze_game",
        )

    try:
        result = await _ANALYZER.analyze(
            pgn=pgn,
            depth=depth,
            strict=strict,
            detail=detail,
            perspective=perspective,
            max_critical_moments=max_critical_moments,
            ctx=ctx,
            metrics=metrics,
        )
        if result.coaching is not None:
            termination = build_game_termination_assessment(
                pgn,
                final_position=result.coaching.final_position,
            )
            result = result.model_copy(
                update={
                    "coaching": result.coaching.model_copy(
                        update={"termination": termination}
                    )
                }
            )
        return result
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
