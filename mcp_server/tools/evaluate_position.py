"""``evaluate_position`` MCP tool.

Extracted from ``mcp_server.server``. Single-position Stockfish evaluation
with rule status, multi-tier cache, draw-claim projection, verbosity trimming,
and opt-in deterministic position-integrity evidence.
"""

from __future__ import annotations

import time
from typing import Literal

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server._mcp import mcp
from mcp_server.analysis.position_integrity import enrich_position_eval
from mcp_server.analysis.tactical_snapshot_extensions import extend_position_eval
from mcp_server.engine import _evaluate_game_position_cached, _get_analyzer_pool
from mcp_server.metrics import metrics
from mcp_server.models.forensics import ForensicEval
from mcp_server.parsers import _build_board_with_metadata, _history_provenance_for_input
from mcp_server.tools._common import (
    VERBOSITY_COMPACT,
    _compact_mcpeval,
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def evaluate_position(
    fen: str,
    moves: list[str] | None = None,
    depth: int = 22,
    strict: bool = False,
    verbosity: str | None = None,
    detail: Literal["standard", "coach", "forensic"] = "standard",
    ctx: Context | None = None,
) -> ForensicEval:
    """Evaluate a chess position with Stockfish.

    Args:
        fen: FEN or PGN string for the position (or position before `moves` are replayed).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 22, clamped 1-30). ``evaluate_position``
            is the on-demand critical-position evaluator. Bump to 24-26 only when
            d22 still shifts the best move, and to 28-30 only when d24 is unstable.
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).
        verbosity: "full" (default) or "compact".
        detail: ``standard`` preserves the previous engine payload and adds only
            ``forensics=null``. ``coach`` adds a deterministic position fingerprint
            plus CCT/defender geometry. ``forensic`` additionally echoes the engine
            best move's resulting position and a static position delta. Rich modes do
            not add another Stockfish search.

    Returns:
         Eval with cp/mate from White's perspective, best move/PV and optional
         evidence-first position integrity metadata.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="evaluate_position")
    raw_requested_depth = depth
    depth = max(1, min(depth, 30))
    try:
        if detail not in {"standard", "coach", "forensic"}:
            raise ValueError(f"INVALID_DETAIL: {detail}")
        verbosity_mode = _resolve_verbosity(verbosity)
        board, input_fen, canonical_fen, fen_was_canonicalized = _build_board_with_metadata(
            fen, moves or [], strict=strict
        )
        pool = await _get_analyzer_pool(ctx)
        # History completeness is derived from whether the caller had the move
        # stack. Naked FEN (no moves) cannot detect threefold repetition.
        history_complete = _history_provenance_for_input(fen, moves)
        res, is_hit = await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete=history_complete,
        )
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, cache_hit=is_hit)
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
        if detail == "standard":
            return ForensicEval(**result.model_dump())
        evidence_detail: Literal["coach", "forensic"] = (
            "forensic" if detail == "forensic" else "coach"
        )
        enriched = enrich_position_eval(result, board, detail=evidence_detail)
        return extend_position_eval(enriched, board)
    except ToolError:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise
    except ValueError as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        msg = str(exc)
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="evaluate_position", input=fen) from exc
    except Exception as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="evaluate_position") from exc
