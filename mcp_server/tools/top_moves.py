"""``top_moves`` MCP tool.

Thin entry point. The end-to-end ranking orchestration lives in
:class:`mcp_server.analysis.top_moves_finder.TopMovesFinder`. Optional coaching
forensics are attached after the cached ranking path so the default API keeps
its existing cost and cache semantics.
"""

from __future__ import annotations

import logging
import time
from typing import Literal

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
    detail: Literal["standard", "coach", "forensic"] = "standard",
    include_moves: list[str] | None = None,
    proof_mode: Literal["none", "tactical"] = "none",
    proof_defenses: int = 3,
    ctx: Context | None = None,
) -> ForensicTopMovesResult:
    """Get top candidates, with optional explicit comparisons and tactical proof.

    The default ``detail="standard"`` path keeps the previous ranking/caching
    behavior and adds only ``forensics=null`` to the response schema. Rich modes:

    - ``detail="coach"`` adds a deterministic board fingerprint and CCT-style
      tactical snapshot.
    - ``detail="forensic"`` additionally evaluates the returned root candidates'
      resulting positions.
    - ``include_moves`` evaluates up to eight explicit SAN/UCI alternatives even
      when they are outside the engine's top-N. Supplying explicit alternatives
      automatically uses forensic comparison semantics and reserves a separate
      slot for the engine-best reference move, so a long top-N list cannot silently
      displace the moves the caller explicitly asked to compare.
    - ``proof_mode="tactical"`` evaluates the engine-best move's reply tree. If
      the opponent has at most eight legal replies every immediate reply is checked
      and the proof is labelled ``exhaustive``. Otherwise only engine-ranked defenses
      are sampled and the response explicitly says ``sampled_top_defenses``.

    ``proof_defenses`` controls the sampled defense count and is clamped to 1-8.
    """
    t0 = time.time()
    depth = _validate_requested_depth(depth, tool="top_moves")
    raw_requested_depth = max(1, min(depth, 30))
    raw_requested_n = n
    clamped_n = max(1, min(n, 20))
    try:
        if detail not in {"standard", "coach", "forensic"}:
            raise ValueError(f"INVALID_DETAIL: {detail}")
        if proof_mode not in {"none", "tactical"}:
            raise ValueError(f"INVALID_PROOF_MODE: {proof_mode}")
        if len(include_moves or []) > 8:
            raise ValueError("INVALID_COMPARE_MOVE: include_moves supports at most 8 moves")

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
                depth=raw_requested_depth,
                detail=effective_detail,
                include_moves=include_moves,
                proof_mode=proof_mode,
                proof_defenses=max(1, min(int(proof_defenses), 8)),
            )
            result = upgrade_top_moves_forensics(result, board)

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
