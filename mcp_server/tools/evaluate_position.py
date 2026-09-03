"""``evaluate_position`` MCP tool.

Extracted from ``mcp_server.server``. Single-position Stockfish evaluation
with rule status, multi-tier cache, draw-claim projection, and verbosity
trimming.
"""

from __future__ import annotations

import time


from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server.engine import _evaluate_game_position_cached, _get_analyzer_pool
from mcp_server.metrics import metrics
from mcp_server.models import MCPEval
from mcp_server.parsers import _build_board_with_metadata, _history_provenance_for_input
from mcp_server.server import mcp
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
    ctx: Context | None = None,
) -> MCPEval:
    """Evaluate a chess position with Stockfish.

    Args:
        fen: FEN or PGN string for the position (or position before `moves` are replayed).
        moves: Optional UCI or SAN moves to replay onto the position first.
        depth: Stockfish search depth (default 22, clamped 1-30). ``evaluate_position``
            is the on-demand critical-position evaluator — the higher default reflects
            per-call usage (caller paid the hit). Bump to 24-26 only when d22 still
            shifts the best move, and to 28-30 only when d24 is unstable.
            Sweet-spot reference (Stockfish 18): d20 ≈ 782k nodes, d22 ≈ 1.29M,
            d24 ≈ 2.06M, d30 ≈ 8M. Depth does NOT map linearly to Elo — fixed depth
            is a budget signal, not a quality signal; ``nodes``/``movetime`` would
            be more uniform but are out of MCP's parameter surface today.
        strict: When True, reject non-canonical SAN syntax or move numbers (default False).
        verbosity: "full" (default, every field) or "compact" (strips Lichess URLs,
            images, decision_value/engine_eval duplication). Use compact when the
            caller is an LLM and you want to minimize context spend (audit M-05).

    Returns:
         Eval with cp (from White's perspective), mate (from White's perspective),
         best_move (UCI), pv (principal variation), and Lichess board URLs.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="evaluate_position")
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
        code = error_code_for(msg)
        raise _tool_error(code=code, message=msg, tool="evaluate_position", input=fen) from exc
    except Exception as exc:
        await metrics.record("evaluate_position", (time.time() - t0) * 1000, is_error=True)
        raise _tool_error(code="engine_error", message=str(exc), tool="evaluate_position") from exc
