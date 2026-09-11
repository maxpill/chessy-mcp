"""MultiPV candidate evaluator (one candidate = one MCPEval entry).

Extracted from :mod:`mcp_server.analysis.top_moves_finder`. Owns the
per-candidate post-state evaluation: zeroing-move re-eval (audit
B-04/B-05), candidate claim policy, candidate action object, and the
post-terminal outcome dict.

Audit invariants preserved: B-04, B-05, C-03.
"""

from __future__ import annotations

from typing import Any

import chess

from core.engines.types import Eval

from mcp_server.engine import _build_identity
from mcp_server.models import MCPEval
from mcp_server.rules import evaluate_rule_status


async def evaluate_candidate(
    *,
    board: chess.Board,
    candidate: Eval,
    pool: Any,
    rule_status: Any,
    sign: int,
    history_complete: str,
    raw_requested_depth: int,
    depth: int,
    needs_post_eval: bool,
    multipv: int | None = None,
) -> MCPEval:
    """Build the :class:`MCPEval` entry for one MultiPV candidate.

    Audit B-04/B-05/C-03 invariants: zeroing-move post-state is re-eval'd
    when multipv looks draw-polluted, but the candidate's reported cp/mate
    stays at multipv values so ranking and back-compat consumers see the
    same numbers they did before. Best action is recomputed as a
    play_move / game_over discriminator independent of the OPPONENT's
    post-state claim options (audit C-03).
    """
    b_cand = board.copy(stack=True)
    cand_san_val: str | None = None
    cand_post_terminal: str | None = None
    cand_winner: str | None = None
    cand_can_claim_now = False
    cand_can_claim_draw = False
    cand_claim_reasons: list[str] = []
    cand_claim_reasons_now: list[str] = []
    cand_claim_moves: list[str] = []
    cand_rule = rule_status
    post_state_cp: int | None = None
    post_state_mate: int | None = None

    if candidate.best_move:
        try:
            post = await _walk_candidate_post_state(
                board=board,
                candidate=candidate,
                b_cand=b_cand,
                depth=depth,
                pool=pool,
                needs_post_eval=needs_post_eval,
                history_complete=history_complete,
            )
            (
                b_cand,
                cand_san_val,
                cand_post_terminal,
                cand_winner,
                cand_can_claim_now,
                cand_can_claim_draw,
                cand_claim_reasons,
                cand_claim_reasons_now,
                cand_claim_moves,
                cand_rule,
                post_state_cp,
                post_state_mate,
            ) = post
        except Exception:
            pass

    post_eval_for_candidate = Eval(
        cp=candidate.cp,
        mate=candidate.mate,
        best_move=candidate.best_move,
        pv=candidate.pv,
        depth=candidate.depth,
    )
    cand_recommended_action = "play_move"
    cand_best_action_obj = _candidate_best_action_obj(
        candidate=candidate,
        board=board,
        sign=sign,
        cand_san_val=cand_san_val,
        cand_post_terminal=cand_post_terminal,
        cand_winner=cand_winner,
        cand_rule=cand_rule,
    )

    identity = _build_identity(pool)
    post_rec_action = (
        "game_over"
        if cand_post_terminal is not None
        else getattr(cand_rule, "recommended_action", "play_move")
    )
    return MCPEval.from_eval(
        post_eval_for_candidate,
        b_cand.fen(),
        board=b_cand,
        requested_depth=raw_requested_depth,
        history_complete=history_complete,
        pv_board=board,
    ).model_copy(
        update={
            "build_sha": identity["build_sha"],
            "engine_config": identity["engine_config"],
            "best_move": candidate.best_move,
            "executable_move": candidate.best_move,
            "post_terminal_status": cand_post_terminal,
            "candidate_san": cand_san_val,
            "post_can_claim_draw": cand_can_claim_draw,
            "post_can_claim_now": cand_can_claim_now,
            "post_claim_reasons": cand_claim_reasons,
            "post_claim_moves": cand_claim_moves,
            "recommended_action": cand_recommended_action,
            "root_candidate_action": cand_recommended_action,
            "best_action": cand_recommended_action,
            "best_action_type": cand_recommended_action,
            "best_action_obj": cand_best_action_obj,
            "multipv": multipv if multipv is not None else getattr(candidate, "multipv", 1),
            "search_provenance": {
                "kind": "multipv_root",
                "depth": getattr(candidate, "depth", depth),
                "multipv": multipv if multipv is not None else getattr(candidate, "multipv", 1),
                "score_comparability": "same_root_multipv",
            },
            "post_state_cp": post_state_cp,
            "post_state_mate": post_state_mate,
            "post_position": {
                "status": cand_post_terminal or "active",
                "winner": cand_winner if cand_post_terminal == "checkmate" else None,
                "can_claim_now": cand_can_claim_now,
                "can_claim_draw": cand_can_claim_draw,
                "claim_reasons": cand_claim_reasons_now or cand_claim_reasons,
                "recommended_action": post_rec_action,
                "post_position_recommended_action": post_rec_action,
            },
        }
    )



async def _walk_candidate_post_state(
    *,
    board: chess.Board,
    candidate: Eval,
    b_cand: chess.Board,
    depth: int,
    pool: Any,
    needs_post_eval: bool,
    history_complete: str,
) -> tuple[
    chess.Board,
    str | None,
    str | None,
    str | None,
    bool,
    bool,
    list[str],
    list[str],
    list[str],
    Any,
    int | None,
    int | None,
]:
    """Push the candidate move, re-eval the zeroing post-state if needed,
    and collect audit-relevant fields. Returns a 12-tuple matching the
    receiver signature in :func:`evaluate_candidate`.

    Raises on any decode / parse error so the caller can swallow it and
    fall back to a no-op candidate row.
    """
    best_move = candidate.best_move
    if best_move is None:
        raise ValueError("candidate has no best_move")
    bm_obj = chess.Move.from_uci(best_move.lower())
    if bm_obj not in board.legal_moves:
        raise ValueError(f"best_move {best_move} not legal on board")
    cand_san_val = board.san(bm_obj)
    is_zeroing = board.is_capture(bm_obj) or (board.piece_type_at(bm_obj.from_square) == chess.PAWN)
    b_cand.push(bm_obj)
    multipv_suspect = candidate.mate is None and (candidate.cp is None or candidate.cp <= 0)

    post_state_cp: int | None = None
    post_state_mate: int | None = None
    if (
        needs_post_eval
        and is_zeroing
        and not b_cand.is_game_over(claim_draw=False)
        and multipv_suspect
    ):
        try:
            post_ev = await pool.evaluate(b_cand, depth=depth)
            if post_ev.mate is not None:
                post_state_mate = post_ev.mate
            elif post_ev.cp is not None:
                post_state_cp = post_ev.cp
        except Exception:
            pass

    cand_sign = 1 if b_cand.turn == chess.WHITE else -1
    if candidate.mate is not None:
        cand_mover_score = cand_sign * candidate.mate * 1000
    elif candidate.cp is not None:
        cand_mover_score = cand_sign * candidate.cp
    else:
        cand_mover_score = None
    cand_mate_for_mover = cand_sign * candidate.mate if candidate.mate is not None else None
    cand_rule = evaluate_rule_status(
        b_cand,
        mover_score=cand_mover_score,
        mate_for_mover=cand_mate_for_mover,
        history_complete=history_complete,
    )
    return (
        b_cand,
        cand_san_val,
        cand_rule.terminal,
        cand_rule.winner,
        cand_rule.can_claim_now,
        cand_rule.can_claim_draw,
        list(cand_rule.claim_reasons),
        list(cand_rule.claim_reasons_now),
        list(cand_rule.claim_moves),
        cand_rule,
        post_state_cp,
        post_state_mate,
    )


def _candidate_best_action_obj(
    *,
    candidate: Eval,
    board: chess.Board,
    sign: int,
    cand_san_val: str | None = None,
    cand_post_terminal: str | None = None,
    cand_winner: str | None = None,
    cand_rule: Any = None,
) -> dict[str, Any]:
    bm_uci = candidate.best_move or ""
    bm_san = cand_san_val
    if bm_san is None and bm_uci and board is not None:
        try:
            m = chess.Move.from_uci(bm_uci.lower())
            if m in board.legal_moves:
                bm_san = board.san(m)
        except Exception:
            pass
    payload: dict[str, Any] = {
        "type": "play_move",
        "move": {"uci": bm_uci, "san": bm_san},
    }
    if candidate.cp is not None or candidate.mate is not None:
        payload["value"] = {"cp": candidate.cp, "mate": candidate.mate}
    return payload



# Back-compat shim.
_eval_one_candidate = evaluate_candidate
