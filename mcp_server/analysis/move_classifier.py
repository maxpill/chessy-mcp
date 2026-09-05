"""``MoveClassifier`` service: typed orchestration for per-move classification."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import chess
from core.engines.types import Eval, MoveClass

from mcp_server.claims.draw_projection import force_draw_outcome
from mcp_server.models import MCPMoveAnalysis, MCPEval, PlayedMoveScore
from mcp_server.parsers import (
    _build_board,
    _history_provenance_for_input,
    _parse_move_on_board_with_warning,
)
from mcp_server.rules import evaluate_rule_status, is_terminal_position

log = logging.getLogger("chessy_mcp.analysis.move_classifier")


class _ValidationOutcome:
    def __init__(
        self,
        board: chess.Board,
        history_complete: str,
        rule_before,
        chess_move: chess.Move | None,
        syntax_warning: str | None,
    ):
        self.board = board
        self.history_complete = history_complete
        self.rule_before = rule_before
        self.chess_move = chess_move
        self.syntax_warning = syntax_warning


def validate_classify_input(
    *,
    fen: str,
    moves: list[str] | None,
    move: str | None,
    action_type: str,
    strict: bool,
) -> _ValidationOutcome:
    if move is not None and not isinstance(move, str):
        raise ValueError(f"INVALID_INPUT: 'move' must be a string, got {type(move).__name__}.")
    syntax_warning: str | None = None
    chess_move: chess.Move | None = None

    if action_type == "claim_draw":
        if move is not None and move.strip() and move.strip() != "(none)":
            if strict:
                raise ValueError(
                    f"STRICT_SAN_ERROR: action_type='claim_draw' must not include a "
                    f"`move` argument; got {move!r}."
                )
            syntax_warning = (
                f"action_type='claim_draw' ignores supplied move argument "
                f"{move!r} (the claim outcome is purely procedural)."
            )
    else:
        if move is None or not move.strip():
            raise ValueError(
                "MISSING_MOVE: 'move' is required for action_type='play_move' "
                "and action_type='claim_draw_with_intended_move'"
            )

    board = _build_board(fen, moves or [], strict=strict)
    history_complete = _history_provenance_for_input(fen, moves)
    rule_before = evaluate_rule_status(board, history_complete=history_complete)

    if action_type == "claim_draw":
        if is_terminal_position(board):
            raise ValueError(
                f"GAME_ALREADY_OVER: Position '{board.fen()}' is already game over; "
                f"no further actions can be taken on a finished game."
            )
        if not rule_before.can_claim_now:
            raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
    else:
        assert move is not None and move.strip()
        chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
        if syntax_warn and not syntax_warning:
            syntax_warning = syntax_warn
        if (
            action_type == "claim_draw_with_intended_move"
            and chess_move.uci() not in rule_before.intended_claim_ucis
        ):
            raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
    return _ValidationOutcome(
        board=board,
        history_complete=history_complete,
        rule_before=rule_before,
        chess_move=chess_move,
        syntax_warning=syntax_warning,
    )


class MoveClassifier:
    def __init__(
        self,
        evaluate_position: Callable[..., Awaitable[tuple[MCPEval, bool]]],
        compute_score: Callable[..., PlayedMoveScore],
        build_classification: Callable[..., MCPMoveAnalysis],
    ) -> None:
        self._evaluate_position = evaluate_position
        self._compute_score = compute_score
        self._build_classification = build_classification

    @classmethod
    def with_defaults(cls) -> MoveClassifier:
        from mcp_server.engine import _evaluate_game_position_cached
        from mcp_server.move_grading import score_played_move

        return cls(
            evaluate_position=_evaluate_game_position_cached,
            compute_score=score_played_move,
            build_classification=MCPMoveAnalysis.from_analysis,
        )

    async def compute(
        self,
        *,
        outcome: _ValidationOutcome,
        action_type: str,
        depth: int,
        raw_requested_depth: int,
        pool,
    ) -> tuple[MCPEval, MCPEval, PlayedMoveScore, bool]:
        board = outcome.board
        history_complete = outcome.history_complete
        chess_move = outcome.chess_move
        eval_before, _ = await self._evaluate_position(
            board,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete=history_complete,
        )
        if action_type in ("claim_draw", "claim_draw_with_intended_move"):
            eval_after, _ = await self._evaluate_position(
                board,
                depth,
                pool,
                requested_depth=raw_requested_depth,
                history_complete=history_complete,
            )
            eval_after = force_draw_outcome(eval_after)
            board_after = board.copy(stack=True)
        else:
            if chess_move is None:
                raise ValueError("MISSING_MOVE: play_move requires a parsed chess move")
            board_after = board.copy(stack=True)
            board_after.push(chess_move)
            eval_after, _ = await self._evaluate_position(
                board_after,
                depth,
                pool,
                requested_depth=raw_requested_depth,
                history_complete=history_complete,
            )

        score_move = chess_move if chess_move is not None else next(iter(board.legal_moves))
        score = self._compute_score(
            board,
            score_move,
            eval_before,
            eval_after,
            board_after=board_after,
            eval_played=None,
            action_type=action_type,
        )
        return eval_before, eval_after, score, False

    async def verify_best_if_needed(
        self,
        *,
        eval_before: MCPEval,
        eval_after: MCPEval,
        score: PlayedMoveScore,
        board: chess.Board,
        chess_move: chess.Move | None,
        depth: int,
        raw_requested_depth: int,
        history_complete: str,
        pool,
        action_type: str,
    ) -> tuple[MCPEval, MCPEval, PlayedMoveScore, bool]:
        verification_attempted = False
        if (
            action_type != "play_move"
            or chess_move is None
            or chess_move.uci().lower() != (eval_before.best_move or "").lower()
            or score.move_class not in (MoveClass.MISTAKE, MoveClass.BLUNDER)
            or score.missed_draw_claim
            or score.conceded_draw_claim
        ):
            return eval_before, eval_after, score, False
        try:
            verify_eval, _verify_hit = await self._evaluate_position(
                board,
                depth + 4,
                pool,
                requested_depth=raw_requested_depth + 4,
                history_complete=history_complete,
            )
            verify_ev = Eval(
                cp=verify_eval.cp,
                mate=verify_eval.mate,
                best_move=verify_eval.best_move,
                pv=verify_eval.pv,
                depth=verify_eval.searched_depth or (depth + 4),
            )
            verification_attempted = True
            if verify_ev.best_move and verify_ev.best_move.lower() != chess_move.uci().lower():
                eval_before = MCPEval.from_eval(
                    verify_ev,
                    board.fen(),
                    board=board,
                    requested_depth=raw_requested_depth,
                    history_complete=history_complete,
                )
                score = self._compute_score(
                    board,
                    chess_move,
                    eval_before,
                    eval_after,
                    board_after=board.copy(stack=True),
                    eval_played=None,
                    action_type=action_type,
                )
            else:
                score.move_class = MoveClass.BEST
                score.effective_loss = 0
                score.is_best_engine_move = True
        except Exception:
            verification_attempted = True
        return eval_before, eval_after, score, verification_attempted
