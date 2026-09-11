"""``top_moves`` MCP tool.

Thin entry point. The end-to-end ranking orchestration lives in
:class:`mcp_server.analysis.top_moves_finder.TopMovesFinder`. Optional coaching
forensics are attached after the cached ranking path so the default API keeps
its existing cost and cache semantics.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Literal
from pydantic import Field

from mcp.server.mcpserver import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from mcp_server._mcp import mcp
from mcp_server.analysis.forensic_integration import upgrade_top_moves_forensics
from mcp_server.analysis.top_moves_finder import TopMovesFinder
from mcp_server.analysis.top_moves_forensics import enrich_top_moves_result
from mcp_server.engine import _get_analyzer_pool
from mcp_server.metrics import metrics
from mcp_server.models.forensics import ForensicTopMovesResult
from mcp_server.parsers import _build_board_with_metadata
from mcp_server.rules.constants import (
    DEPTH_MAX,
    DEPTH_MIN,
    MAX_INCLUDE_MOVES,
    TOP_MOVES_MAX_N,
    TOP_MOVES_MIN_N,
)
from mcp_server.tools._common import (
    VerbosityInput,
    _compact_mcpeval,
    _minimal_mcpeval,
    _resolve_verbosity,
    _tool_error,
    _validate_requested_depth,
    error_code_for,
)

log = logging.getLogger("chessy_mcp.top_moves")

_FINDER = TopMovesFinder.with_defaults()


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))
async def top_moves(
    fen: Annotated[
        str,
        Field(description="FEN or PGN string representing the position to evaluate."),
    ],
    moves: Annotated[
        list[str] | None,
        Field(description="Optional list of UCI or SAN moves to replay onto the position first."),
    ] = None,
    n: Annotated[
        int,
        Field(description="Number of top candidate moves to return (default 3, clamped 1-20)."),
    ] = 3,
    depth: Annotated[
        int,
        Field(description="Stockfish search depth (default 20, clamped 1-30)."),
    ] = 20,
    strict: Annotated[
        bool,
        Field(description="When True, reject non-canonical SAN syntax or move numbers."),
    ] = False,
    verbosity: Annotated[
        VerbosityInput | None,
        Field(
            description="Response verbosity: 'full' (default), 'compact', or 'minimal' (aliases 'min', 'standard', 'default' accepted)."
        ),
    ] = None,
    detail: Annotated[
        Literal["standard", "coach", "forensic"],
        Field(
            description="Detail level: 'standard' (fast ranking), 'coach' (snapshots), 'forensic' (endpoint deltas & proofs)."
        ),
    ] = "standard",
    include_moves: Annotated[
        list[str] | None,
        Field(
            description="Explicit legal candidate moves for the side to move that MUST be evaluated and included (up to 8 moves). Must be legal moves for the side to move, NOT opponent replies."
        ),
    ] = None,
    proof_mode: Annotated[
        Literal["none", "tactical"],
        Field(
            description="'none' (default) or 'tactical' to evaluate the reply tree of the best move."
        ),
    ] = "none",
    proof_defenses: Annotated[
        int,
        Field(
            description="Number of defensive replies to analyze in tactical proof mode (default 3)."
        ),
    ] = 3,
    ctx: Context | None = None,
) -> ForensicTopMovesResult:
    """Get top candidates, with optional explicit comparisons and tactical proof.

    The default ``detail="standard"`` path keeps the previous ranking/caching
    behavior and adds only ``forensics=null`` to the response schema. Rich modes:

    - ``detail="coach"`` adds a deterministic board fingerprint and CCT-style
      tactical snapshot.
    - ``detail="forensic"`` additionally evaluates the returned root candidates'
      resulting positions and walks each already returned candidate PV to an
      evidence-bounded continuation endpoint. Each endpoint exposes its FEN,
      fingerprint, tactical snapshot, root-to-endpoint delta, irreversible events,
      consumed plies and termination reason. ``pv_exhausted`` explicitly means
      only that the available PV ended, not that the position is quiet.
    - ``candidate_differences`` compares both the immediate resulting positions
      and, when available, ``continuation_endpoint_difference``. Endpoint
      comparisons preserve unequal PV lengths/termination reasons instead of
      pretending that two engine lines are a controlled causal experiment.
    - ``include_moves`` evaluates up to eight explicit SAN/UCI alternatives even
      when they are outside the engine's top-N. Supplying explicit alternatives
      automatically uses forensic comparison semantics and reserves a separate
      slot for the engine-best reference move, so a long top-N list cannot silently
      displace the moves the caller explicitly asked to compare.
    - ``proof_mode="tactical"`` evaluates the engine-best move's reply tree. If
      the opponent has at most eight legal replies every immediate reply is checked
      and the proof is labelled ``exhaustive``. Otherwise only engine-ranked defenses
      are sampled and the response explicitly says ``sampled_top_defenses``.

    Continuation-endpoint reconstruction adds no new Stockfish search; it uses
    only candidate PVs that were already returned by the existing engine work.
    ``proof_defenses`` controls the sampled defense count and is clamped to 1-8.

    Action semantics:
    - Root candidate action: ``recommended_action`` / ``root_candidate_action`` on each candidate
      represents the action for the player to move at the root position (e.g. ``play_move``).
    - Post-position action: ``post_position["recommended_action"]`` /
      ``post_position["post_position_recommended_action"]`` represents the recommended action
      for the replying player in the position resulting from this candidate move (e.g. ``claim_draw``).
    - Terminal boards: when the game is over by chess rules (checkmate, stalemate, insufficient
      material, 75-move rule, 5-fold repetition), ``returned_n = 0`` moves are returned because no
      legal game actions can be played, even when ``board_legal_move_count > 0`` (such as kings having
      geometric moves in insufficient material endings).
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="top_moves")
    raw_requested_depth = depth
    depth = max(DEPTH_MIN, min(depth, DEPTH_MAX))
    raw_requested_n = n
    clamped_n = max(TOP_MOVES_MIN_N, min(n, TOP_MOVES_MAX_N))
    try:
        if detail not in {"standard", "coach", "forensic"}:
            raise ValueError(f"INVALID_DETAIL: {detail}")
        if proof_mode not in {"none", "tactical"}:
            raise ValueError(f"INVALID_PROOF_MODE: {proof_mode}")
        if len(include_moves or []) > MAX_INCLUDE_MOVES:
            raise ValueError("INVALID_PARAMETER_COUNT: include_moves supports at most 8 moves")

        verbosity_mode = _resolve_verbosity(verbosity)
        out = await _FINDER.run(
            fen=fen,
            moves=moves,
            n=clamped_n,
            depth=depth,
            raw_requested_depth=raw_requested_depth,
            raw_requested_n=raw_requested_n,
            clamped_n=clamped_n,
            strict=strict,
            verbosity_mode=verbosity_mode,
            ctx=ctx,
            include_moves=include_moves,
        )
        result = ForensicTopMovesResult(**out.result.model_dump())

        rich_requested = detail != "standard" or bool(include_moves) or proof_mode != "none"
        if rich_requested and result.status == "active":
            board, _input_fen, _canonical_fen, _canonicalized = _build_board_with_metadata(
                fen,
                moves or [],
                strict=strict,
            )
            pool = await _get_analyzer_pool(ctx)
            effective_detail: Literal["coach", "forensic"] = (
                "forensic"
                if detail == "forensic" or bool(include_moves) or proof_mode == "tactical"
                else "coach"
            )
            result = await enrich_top_moves_result(
                result,
                board,
                pool=pool,
                depth=depth,
                detail=effective_detail,
                include_moves=include_moves,
                proof_mode=proof_mode,
                proof_defenses=max(1, min(int(proof_defenses), 8)),
                strict=strict,
            )
            result = upgrade_top_moves_forensics(result, board)

        if verbosity_mode == "compact":
            result = result.model_copy(
                update={
                    "result": [
                        _compact_mcpeval(c) if not getattr(c, "is_compact", False) else c
                        for c in result.result
                    ]
                }
            )
        elif verbosity_mode == "minimal":
            result = result.model_copy(
                update={
                    "result": [
                        _minimal_mcpeval(c) if not getattr(c, "is_minimal", False) else c
                        for c in result.result
                    ]
                }
            )

        await metrics.record(
            "top_moves",
            (time.time() - t0) * 1000,
            cache_hit=out.cache_hit,
        )
        return result
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
