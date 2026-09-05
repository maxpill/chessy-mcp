"""Single-position eval pipeline helpers."""

from __future__ import annotations

import chess

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.engine.identity import build_identity
from mcp_server.engine.protocol import EnginePoolLike
from mcp_server.models import MCPEval
from mcp_server.urls import lichess_urls

__all__ = ["build_terminal_mcpeval", "evaluate_zeroing_post_state"]


def build_terminal_mcpeval(
    *,
    rule_status: RuleStatus,
    board: chess.Board,
    requested_depth: int,
    pool: EnginePoolLike,
) -> MCPEval:
    """Build MCPEval for an already-terminal position without an engine call."""
    terminal_status = rule_status.terminal
    if terminal_status is None:
        raise ValueError("build_terminal_mcpeval requires a terminal rule status")

    if terminal_status == "checkmate":
        term_outcome = "win" if rule_status.winner == "white" else "loss"
        term_cp: int | None = None
        term_mate: int | None = 0
    else:
        term_outcome = "draw"
        term_cp = 0
        term_mate = None

    canonical_fen_str = board.fen()
    url, img = lichess_urls(canonical_fen_str)
    terminal_best_action = build_best_action(
        recommended_action="game_over",
        rule_status=rule_status,
        engine_eval=None,
        board=board,
        sign=1 if board.turn == chess.WHITE else -1,
    )
    terminal_legal_actions = build_legal_actions(
        rule_status=rule_status,
        engine_eval=None,
        board=board,
        legal_engine_moves=None,
    )
    terminal_rule_actions = [
        a
        for a in terminal_legal_actions
        if a.get("type") in ("claim_draw", "claim_draw_with_intended_move")
    ]
    return MCPEval(
        status=terminal_status,
        winner=rule_status.winner,
        cp=term_cp,
        mate=term_mate,
        best_move=None,
        pv=[],
        depth=0,
        requested_depth=requested_depth,
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
        legal_rule_actions=terminal_rule_actions,
        canonical_fen=canonical_fen_str,
        fen_was_canonicalized=False,
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
        **build_identity(pool),
    )


async def evaluate_zeroing_post_state(
    *,
    board: chess.Board,
    best_move_uci: str | None,
    depth: int,
    pool: EnginePoolLike,
) -> tuple[int | None, int | None]:
    """Re-evaluate a zeroing move's post-state when 50-move draw pollution is possible."""
    if not best_move_uci or board.halfmove_clock < 100 or board.is_game_over():
        return None, None
    try:
        bm_obj = chess.Move.from_uci(best_move_uci.lower())
    except Exception:
        return None, None
    if bm_obj not in board.legal_moves:
        return None, None

    is_zeroing = board.is_capture(bm_obj) or (board.piece_type_at(bm_obj.from_square) == chess.PAWN)
    if not is_zeroing:
        return None, None

    board_after = board.copy(stack=True)
    board_after.push(bm_obj)
    if board_after.is_game_over(claim_draw=False):
        return None, None

    try:
        post_ev = await pool.evaluate(board_after, depth=depth)
    except Exception:
        return None, None

    mover_sign = 1 if board.turn == chess.WHITE else -1
    zeroing_cp: int | None = None
    zeroing_mate: int | None = None
    if post_ev.mate is not None:
        mover_mate = mover_sign * post_ev.mate
        if mover_mate > 0:
            zeroing_mate = mover_mate
    elif post_ev.cp is not None:
        mover_cp = mover_sign * post_ev.cp
        if mover_cp > 0:
            zeroing_cp = mover_cp
    return zeroing_cp, zeroing_mate
