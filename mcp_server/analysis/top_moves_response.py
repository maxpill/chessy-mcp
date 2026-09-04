"""Top-moves response builder.

Extracted from :mod:`mcp_server.analysis.top_moves_finder`. Owns the
:class:`TopMovesResult` construction logic — the boilerplate at the
end of :meth:`TopMovesFinder.run` that copies cache-hit metadata,
action objects, and engine identity into a typed response.
"""

from __future__ import annotations

from typing import Any

import chess


from mcp_server.engine import _build_identity
from mcp_server.models import MCPEval, TopMovesResult


def build_top_moves_response(
    *,
    items: list[MCPEval],
    pool: Any,
    board: chess.Board,
    rule_status: Any,
    sign: int,
    cached_items: list[MCPEval],
    root_rec_action: str,
    best_action_obj: dict[str, Any],
    legal_actions: list[dict[str, Any]],
    engine_name_str: str,
    canonical_fen: str,
    fen_was_canonicalized: bool,
    raw_requested_depth: int,
    searched_depth: int,
    raw_requested_n: int,
    clamped_n: int,
    legal_move_count: int,
) -> TopMovesResult:
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
        searched_depth=searched_depth,
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


# Back-compat shim.
_build_top_moves_response = build_top_moves_response
