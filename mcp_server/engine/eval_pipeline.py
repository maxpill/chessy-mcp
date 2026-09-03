"""Single-position eval pipeline helpers.

Extracted from :mod:`mcp_server.engine.pool_factory` so the eval logic is
broken into named, individually testable pieces. Two helpers live here now:

    - :func:`build_terminal_mcpeval` — short-circuit when the position is
      already terminal (checkmate / stalemate / 75-move / fivefold / dead).
      Returns the ``MCPEval`` with a populated ``game_over`` typed action.
    - :func:`evaluate_zeroing_post_state` — audit B-04/B-05 re-evaluation:
      at halfmove>=100, the engine's root cp can be polluted by the draw on
      the table; re-evaluate the post-state of the engine's best zeroing
      capture so the action policy can distinguish "draw is available" from
      "post-state is a forced win".

The rule-aware best-move override (audit P0) is intentionally NOT extracted —
it's tightly coupled to the inline ``pool.evaluate(board, root_moves=...)``
loop in the orchestrator and has many small branches; pulling it out buys
less readability than it costs. See ``pool_factory._evaluate_game_position_cached``
for the live implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import chess

from mcp_server.actions import build_best_action, build_legal_actions
from mcp_server.domain.rule_status import RuleStatus
from mcp_server.engine.identity import build_identity
from mcp_server.engine.protocol import EnginePoolLike
from mcp_server.models import MCPEval
from mcp_server.urls import lichess_urls

if TYPE_CHECKING:
    pass  # EnginePoolLike re-exported below — TYPE_CHECKING guards not needed


__all__ = ["build_terminal_mcpeval", "evaluate_zeroing_post_state"]


def build_terminal_mcpeval(
    *,
    rule_status: RuleStatus,
    board: chess.Board,
    requested_depth: int,
    pool: EnginePoolLike,
) -> MCPEval:
    """Build the ``MCPEval`` for an already-terminal position.

    No engine call — terminal positions are deterministic, so the response
    carries the game-over typed action, an empty PV, and the rule-status
    metadata that callers surface in ``best_action_obj`` /
    ``legal_actions``.
    """
    if rule_status.terminal == "checkmate":
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
    return MCPEval(
        status=rule_status.terminal,
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
    """Audit B-04/B-05: re-evaluate the post-state of a winning zeroing move.

    At halfmove>=100, Stockfish's root multipv can be polluted by the draw
    being on the table — a winning Kxe2 in K+R vs R can report a tiny cp.
    Re-running on the post-state of the zeroing capture surfaces the true
    mover-POV advantage.

    Returns ``(zeroing_cp, zeroing_mate)`` — one or the other is non-None
    only when the post-state is materially winning for the mover.
    Returns ``(None, None)`` when no re-eval was performed (not eligible,
    not a zeroing move, post-state is terminal, or engine error).
    """
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
