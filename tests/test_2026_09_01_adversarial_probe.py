"""Regression tests for the 2026-09-01 adversarial probe findings.

Three P0/P1 issues found by a deep adversarial probe of the live MCP:

P0/P1 (classify_move): when action_type=claim_draw and the supplied `move`
happens to be the engine's best legal move, the verification block used to
confirm the move is best (it IS) and overwrite `move_class=BEST,
effective_loss=0` — silently turning a blundered win (claim draw when mate
is available) into a free pass. The invariant
    is_best_action==False AND best outcome==win AND played outcome==draw
        => effective_loss > 0 AND move_class != best
was violated.

P1 (classify_move, claim_draw_with_intended_move): same bug surfaced when the
intended claim move coincidentally matched the engine's best — the
verification overwrote the blundered-win loss to zero.

P1/P2 (top_moves): each candidate's nested MCPEval inherited the post-state's
`rule_status.recommended_action` for `best_action` / `best_action_obj.type`,
producing play_move candidates whose `best_action_obj.type` was
"claim_draw" (e.g. Qb1 in a halfmove=100 position with mate available). The
candidate's outer action type (play_move) and inner recommendation
(claim_draw) were self-contradictory.

These tests pin the fixed behavior:
  * claim_draw + dummy-move-equals-engine-best -> BLUNDER, effective_loss=1000
  * claim_draw_with_intended_move + intended-move-equals-engine-best AND
    forced-mate-available -> BLUNDER, effective_loss=1000
  * score_played_move: claim actions never set is_best_engine_move=True
  * top_moves: each candidate's best_action matches the candidate's action
    type (play_move or game_over), and the post-state's recommendation is
    preserved in post_position.recommended_action.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval, MoveClass
from mcp_server import server as server_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MateAtQc8Pool:
    """Stockfish reports a forced mate in 1 via Qc8# (UCI f5c8) for any
    board. The post-move board is also evaluated (post-state cp=0, no mate
    for the draw-outcome projection).
    """

    name = "MateAtQc8Pool"
    engine_version = "MateAtQc8Pool"

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        # If the position is a checkmate (e.g. after Qc8#), cp=0, mate=0.
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        # Otherwise the engine's best move is f5c8 (Qc8#) with mate in 1.
        return Eval(
            cp=None,
            mate=1,
            best_move="f5c8",
            pv=["f5c8"],
            depth=depth,
        )

    async def top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 14,
    ) -> list[Eval]:
        if board.is_checkmate():
            return []
        return [
            Eval(cp=None, mate=1, best_move="f5c8", pv=["f5c8"], depth=depth),
        ]

    async def close(self) -> None:
        pass


class _KingEscapeBestPool:
    """Stockfish reports best move = a specific king move (h5h4) with cp=80.
    Used to set up claim_draw_with_intended_move scenarios where the
    intended claim move happens to also be the engine's best."""

    def __init__(self, best_move: str = "h5h4", cp: int = 80, mate: int | None = None) -> None:
        self.best_move = best_move
        self.cp = cp
        self.mate = mate
        self.name = "KingEscapeBestPool"
        self.engine_version = "KingEscapeBestPool"

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        return Eval(
            cp=self.cp,
            mate=self.mate,
            best_move=self.best_move,
            pv=[self.best_move],
            depth=depth,
        )

    async def top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 14,
    ) -> list[Eval]:
        return [
            Eval(
                cp=self.cp,
                mate=self.mate,
                best_move=self.best_move,
                pv=[self.best_move],
                depth=depth,
            ),
        ]

    async def close(self) -> None:
        pass


class _MultiPV3Pool:
    """MultiPV pool returning three candidates: a forced mate, a normal move
    that doesn't reset the halfmove clock (so the post-state still has a
    claim available), and a second mating move."""

    name = "MultiPV3Pool"
    engine_version = "MultiPV3Pool"

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(
            cp=None,
            mate=1,
            best_move="f5c8",
            pv=["f5c8"],
            depth=depth,
        )

    async def top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 14,
    ) -> list[Eval]:
        if board.is_checkmate():
            return []
        return [
            Eval(cp=None, mate=1, best_move="f5c8", pv=["f5c8"], depth=depth),
            Eval(cp=10, mate=None, best_move="f5b1", pv=["f5b1", "h8g7"], depth=depth),
            Eval(cp=None, mate=1, best_move="f5f8", pv=["f5f8"], depth=depth),
        ]

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# P0/P1: claim_draw + dummy-move-equals-engine-best must be BLUNDER
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_draw_with_engine_best_dummy_move_is_blunder():
    """White has Qc8# (mate in 1) at halfmove=100 and can claim draw.

    action_type=claim_draw with move=Qc8# (the engine's best) MUST be
    classified as BLUNDER with effective_loss=1000 — the dummy move is
    purely informational, the action is the claim, and the claim is
    throwing away a forced win.
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtQc8Pool()  # type: ignore[assignment]

    fen = "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51"
    cl = await server_module.classify_move(
        fen,
        "Qc8#",
        depth=10,
        action_type="claim_draw",
    )

    # Invariant: is_best_action==False AND best outcome==win AND
    # played outcome==draw => effective_loss > 0 AND move_class != best
    assert cl.is_best_action is False
    assert cl.best_action == "play_move"
    assert cl.action_type == "claim_draw"
    assert cl.move_class != MoveClass.BEST
    assert (cl.effective_loss or 0) > 0
    assert cl.loss_kind == "outcome_penalty"
    assert cl.outcome_penalty == 1000
    # The player did NOT play the engine's best — they claimed draw.
    assert cl.is_engine_best is False
    assert cl.is_best_engine_move is False


@pytest.mark.asyncio
async def test_claim_draw_with_non_engine_best_dummy_move_is_blunder():
    """Sanity: same scenario but dummy move=Qf8# (not the engine's best).

    Pre-fix this was the only path that graded correctly. Post-fix both
    paths return BLUNDER; this test guards against the fix accidentally
    narrowing the bug to "only when the dummy matches the engine best"."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtQc8Pool()  # type: ignore[assignment]

    fen = "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51"
    cl = await server_module.classify_move(
        fen,
        "Qf8#",
        depth=10,
        action_type="claim_draw",
    )

    assert cl.is_best_action is False
    assert cl.move_class != MoveClass.BEST
    assert (cl.effective_loss or 0) > 0
    assert cl.loss_kind == "outcome_penalty"
    assert cl.outcome_penalty == 1000


# ---------------------------------------------------------------------------
# P1: claim_draw_with_intended_move + intended-move-equals-engine-best at
# forced mate must be BLUNDER
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claim_draw_with_intended_move_engine_best_forced_mate_is_blunder():
    """At halfmove=99 with a forced mate available, claim_draw_with_intended_move
    must be BLUNDER even when the intended move (a non-resetting king move)
    happens to be what the engine also reports as best at finite depth.

    Engine best = Kh4 (h5h4) with mate=1 (a contrived mock — real engines
    would report the mating move here, but the test forces the scenario the
    user reported).
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _KingEscapeBestPool(  # type: ignore[assignment]
        best_move="h5h4", cp=None, mate=1
    )

    fen = "7k/8/8/7K/8/8/P7/8 w - - 99 51"
    cl = await server_module.classify_move(
        fen,
        "Kh4",
        depth=10,
        action_type="claim_draw_with_intended_move",
    )

    assert cl.is_best_action is False
    assert cl.best_action == "play_move"
    assert cl.action_type == "claim_draw_with_intended_move"
    assert cl.move_class == MoveClass.BLUNDER
    assert cl.effective_loss == 1000
    assert cl.loss_kind == "outcome_penalty"
    assert cl.outcome_penalty == 1000
    # The action IS the claim, not the move. The player chose the claim even
    # though the engine's best was Kh4.
    assert cl.is_engine_best is False
    assert cl.is_best_engine_move is False


@pytest.mark.asyncio
async def test_score_played_move_claim_action_never_sets_is_best_engine_move():
    """score_played_move must force is_best_engine_move=False for any
    action_type != "play_move", regardless of whether the supplied move
    happens to match eval_before.best_move. This is the semantic fix that
    keeps the result self-consistent for callers reading is_engine_best."""
    from mcp_server.models import MCPEval, score_played_move

    fen = "7k/8/8/7K/8/8/P7/8 w - - 99 51"
    board = chess.Board(fen)

    # Intended claim move that is also the engine's best
    intended_move = chess.Move.from_uci("h5h4")
    raw_eval = Eval(cp=None, mate=1, best_move="h5h4", pv=["h5h4"], depth=10)
    eval_before = MCPEval.from_eval(raw_eval, board.fen(), board=board, history_complete="complete")
    eval_after = eval_before

    score_claim = score_played_move(
        board,
        intended_move,
        eval_before,
        eval_after,
        board_after=None,
        action_type="claim_draw_with_intended_move",
    )
    assert score_claim.is_best_engine_move is False, (
        "claim action must never claim is_best_engine_move=True — the "
        "action is the claim, not the move"
    )

    # For claim_draw (no intended move), use a position with halfmove>=100
    # so can_claim_now is True. The same force-False rule applies.
    fen_now = "7k/8/8/7K/8/8/P7/8 w - - 100 51"
    board_now = chess.Board(fen_now)
    raw_eval_now = Eval(cp=None, mate=1, best_move="h5h4", pv=["h5h4"], depth=10)
    eval_before_now = MCPEval.from_eval(
        raw_eval_now, board_now.fen(), board=board_now, history_complete="complete"
    )
    score_claim_now = score_played_move(
        board_now,
        intended_move,
        eval_before_now,
        eval_before_now,
        board_after=None,
        action_type="claim_draw",
    )
    assert score_claim_now.is_best_engine_move is False

    # For comparison, a play_move with the same move in a claim-free,
    # non-winning position IS the engine's best. Use an early-game FEN
    # where neither side can claim and the score is moderate.
    fen_play = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    board_play = chess.Board(fen_play)
    # Engine best is e7e5 (response to e4) with cp=20.
    raw_eval_play = Eval(cp=20, mate=None, best_move="e7e5", pv=["e7e5"], depth=10)
    eval_before_play = MCPEval.from_eval(
        raw_eval_play,
        board_play.fen(),
        board=board_play,
        history_complete="complete",
    )
    play_move = chess.Move.from_uci("e7e5")
    board_play_after = board_play.copy(stack=True)
    board_play_after.push(play_move)
    score_play = score_played_move(
        board_play,
        play_move,
        eval_before_play,
        eval_before_play,
        board_after=board_play_after,
        action_type="play_move",
    )
    assert score_play.is_best_engine_move is True, (
        "play_move with a matching engine best must still set is_best_engine_move=True"
    )


# ---------------------------------------------------------------------------
# P1/P2: top_moves candidate outer action type must match the candidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_moves_candidate_best_action_matches_outer_action_type():
    """Each candidate in top_moves is a play_move candidate. Its
    best_action / best_action_type / best_action_obj.type must all read as
    `play_move` (or `game_over` for terminal post-states), NEVER
    `claim_draw` — the post-state's recommendation belongs in
    `post_position.recommended_action`, not in the candidate's outer
    best_action.
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _MultiPV3Pool()  # type: ignore[assignment]

    fen = "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51"
    res = await server_module.top_moves(fen, n=3, depth=10)

    # The non-mating Qb1 candidate is the case the user reported as
    # inconsistent: post-state still has a 50-move claim available, so
    # rule_status.recommended_action was "claim_draw" and leaked into
    # best_action_obj.
    by_uci = {c.best_move: c for c in res.result if c.best_move}
    qb1 = by_uci.get("f5b1")
    assert qb1 is not None, "Qb1 candidate missing"

    assert qb1.recommended_action == "play_move"
    assert qb1.best_action == "play_move"
    assert qb1.best_action_type == "play_move"
    assert qb1.best_action_obj is not None
    assert qb1.best_action_obj.get("type") == "play_move", (
        f"play_move candidate leaked {qb1.best_action_obj.get('type')} from post-state rule_status"
    )
    # Post-state info preserved separately.
    assert qb1.post_position is not None
    assert qb1.post_position.get("can_claim_now") is True
    assert qb1.post_position.get("can_claim_draw") is True
    assert qb1.post_position.get("recommended_action") in (
        "claim_draw",
        "claim_draw_with_intended_move",
    ), (
        f"post-state's recommendation must be preserved on post_position; "
        f"got {qb1.post_position.get('recommended_action')!r}"
    )


@pytest.mark.asyncio
async def test_top_moves_mating_candidate_best_action_is_game_over():
    """A candidate whose post-state is checkmate must read as `game_over`
    on the outer action surface (recommended_action, best_action,
    best_action_obj.type) — never `play_move`."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MultiPV3Pool()  # type: ignore[assignment]

    fen = "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51"
    res = await server_module.top_moves(fen, n=3, depth=10)

    by_uci = {c.best_move: c for c in res.result if c.best_move}
    qc8 = by_uci.get("f5c8")
    assert qc8 is not None
    assert qc8.recommended_action == "game_over"
    assert qc8.best_action == "game_over"
    assert qc8.best_action_type == "game_over"
    assert qc8.best_action_obj is not None
    assert qc8.best_action_obj.get("type") == "game_over"
    assert qc8.best_action_obj.get("outcome") == "win"
    assert qc8.best_action_obj.get("reason") == "checkmate"
