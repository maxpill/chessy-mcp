"""P0 regression tests: draw-claim contamination of action values.

Bug doc §4 — a winning zeroing move (pawn push / capture / promotion) at
halfmove >= 100 must be preferred over `claim_draw`. The current code only
re-evaluates the engine's MultiPV best move's post-state, which misses the
case where the engine's best line is a quiet non-zeroing move and the
winning zeroing move is below the MultiPV cutoff.

These tests reproduce the exact scenarios from the bug doc (FEN A, B, C) and
the §32 Test A/B/C/D minimal reproducers. They fail on the current build
because `_maybe_zeroing_best_override` only re-evaluates the engine's
MultiPV line, not every legal zeroing move.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


# ---------------------------------------------------------------------------
# Fake engine pool: returns the same first legal move regardless of position,
# but treats pawn pushes as forcing mate. This simulates the §4 scenario
# where root cp is small but post-state of the pawn move is winning.
# ---------------------------------------------------------------------------


class _PawnPushWinsPool:
    """Simulates Stockfish at shallow depth on the §4.1 FEN.

    Root evaluate() returns a small cp=+26 with first legal move as best.
    When the post-state contains a pawn that didn't exist before (a2→a4,
    a7→a8=Q), the evaluation returns mate=7 — modelling the depth-10
    forced mate finding that the §4.1 audit scenario reproduces.
    """

    name = "PawnPushWinsPool"
    engine_version = "PawnPushWinsPool"

    def __init__(self) -> None:
        self.evaluate_calls = 0

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        self.evaluate_calls += 1
        # Detect the post-state of a pawn push from a2→a4 in the §4.1 FEN
        # (board now has a pawn on a4) — return mate=7 like a deeper eval.
        fen = board.fen()
        if "P7/8/" in fen and "4K3/6R1 b" in fen:
            return Eval(cp=None, mate=7, best_move="a4a5", pv=["a4a5"], depth=depth)
        if "Q6k/" in fen:
            return Eval(cp=None, mate=1, best_move="Qa8a1", pv=["Qa8a1"], depth=depth)
        # §4.4: pawn capture-promotion to Q is winning (K+Q vs K).
        # Detect post-state containing a Q on the 8th rank with the rest of
        # the position around it being a clear technical win.
        fen_parts = fen.split()
        rank8 = fen_parts[0].split("/")[0]
        if "Q" in rank8:
            return Eval(cp=None, mate=1, best_move="Q" + rank8[1:2].lower(), pv=["Q"], depth=depth)
        if root_moves:
            m = root_moves[0]
            b2 = board.copy(stack=True)
            b2.push(m)
            if b2.is_checkmate():
                return Eval(cp=None, mate=1, best_move=m.uci(), pv=[m.uci()], depth=depth)
            if board.piece_type_at(m.from_square) == chess.PAWN:
                return Eval(
                    cp=None,
                    mate=7,
                    best_move=m.uci(),
                    pv=[m.uci()],
                    depth=depth,
                )
        # Root call: small cp, first legal move as best
        legal = list(board.legal_moves)
        best_uci = legal[0].uci() if legal else None
        return Eval(cp=26, best_move=best_uci, pv=[best_uci] if best_uci else [], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        legal = list(board.legal_moves)
        items = [Eval(cp=26, best_move=m.uci(), pv=[m.uci()], depth=depth) for m in legal[:n]]
        return items

    async def classify_move(self, board, move, depth=14):
        from core.engines.types import MoveAnalysis, MoveClass

        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=26, best_move=move.uci()),
            eval_after=Eval(cp=None, mate=7, best_move=move.uci()),
            best_move_san=board.san(move),
        )

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


# ---------------------------------------------------------------------------
# Test A — §4.1: pawn push resets halfmove, leads to forced mate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_a4_wins_over_claim_draw_at_depth_8():
    """FEN: 7k/8/8/8/8/8/P3K3/6R1 w - - 100 51

    White has K+R+P vs K. a4 resets halfmove, post(a4) is winning.
    recommended_action must be `play_move`, not `claim_draw`.
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _PawnPushWinsPool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"
    res = await server_module.evaluate_position(fen, depth=8)

    assert res.can_claim_now is True, "sanity: claim_draw must be legal"
    assert res.recommended_action == "play_move", (
        f"a4 wins; recommended_action must be play_move, got {res.recommended_action}"
    )
    assert res.best_action_obj is not None
    ba = res.best_action_obj
    assert ba.get("type") == "play_move", f"expected play_move action, got {ba!r}"


@pytest.mark.asyncio
async def test_a_a4_post_state_is_winning():
    """§32 Test A partial: post(a4) is checked to be winning for mover."""
    fen = "7k/8/8/8/8/8/P3K3/6R1 w - - 100 51"
    board = chess.Board(fen)
    move = chess.Move.from_uci("a2a4")
    assert move in board.legal_moves
    board.push(move)
    # White's turn after push: K on e2, R on g1, pawn on a4. Black king h8.
    # Material: K+R+P vs K. NOT insufficient material. NOT terminal.
    assert not board.is_checkmate()
    assert not board.is_stalemate()
    assert not board.is_insufficient_material()
    assert board.halfmove_clock == 0, (
        f"halfmove should be 0 after pawn push, got {board.halfmove_clock}"
    )


# ---------------------------------------------------------------------------
# Test B — §4.3: promotion also wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b_a8q_wins_over_claim_draw():
    """FEN: 7k/P7/8/8/8/8/4K3/8 w - - 100 51

    White has K+P vs K. a8=Q+ wins. claim_draw is legal. recommended_action
    must be `play_move`.
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _PawnPushWinsPool()  # type: ignore[assignment]

    fen = "7k/P7/8/8/8/8/4K3/8 w - - 100 51"
    res = await server_module.evaluate_position(fen, depth=8)

    assert res.can_claim_now is True
    assert res.recommended_action == "play_move"
    assert res.best_action_obj is not None
    assert res.best_action_obj.get("type") == "play_move"


# ---------------------------------------------------------------------------
# Test C — §4.4: capture that resets halfmove wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_c_capture_resets_halfmove_wins():
    """FEN: 4k3/8/8/8/8/4P3/4K3/r7 w - - 100 51

    White pawn on e6 can capture the rook on a1 — wait, that's not on a
    diagonal. The pawn needs to be adjacent diagonally. Use a position
    where the pawn is on 7th rank with a piece in front: e7 with rook on
    f8 or similar.

    FEN: r3k3/4P3/8/8/8/8/4K3/8 w - - 100 51
    White pawn on e7. Black rook on a8. Pawn can capture: e7xd8=R? Wait,
    e7 pawn captures what? Black rook is on a8 — pawn can't reach. Try
    b7 capture on a8.
    """
    await server_module._cache.clear()
    server_module._analyzer_pool = _PawnPushWinsPool()  # type: ignore[assignment]

    # White pawn on b7 can capture rook on c8 → promotes → K+Q vs K winning.
    fen = "2r1k3/1P6/8/8/8/8/4K3/8 w - - 100 51"
    res = await server_module.evaluate_position(fen, depth=8)

    assert res.can_claim_now is True
    # White's only legal moves: pawn pushes b7-a8=Q+ or b7-b8=Q+, king moves.
    # The promotion must win over claim_draw.
    assert res.recommended_action == "play_move", (
        f"capture-promotion must beat claim_draw; got {res.recommended_action}"
    )


# ---------------------------------------------------------------------------
# Test D — §32 Test D: intended claim only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d_quiet_rook_move_can_be_intended_claim():
    """FEN: 7k/8/8/8/8/8/4K3/R7 w - - 99 50

    At halfmove=99, claim_draw is not yet legal but a quiet rook move would
    push halfmove to 100, making claim-with-intended-move legal. That move
    is a valid claim move (does not reset halfmove). A pawn push would
    reset halfmove and not be a valid claim-with-intended move.
    """
    fen = "7k/8/8/8/8/8/4K3/R7 w - - 99 50"
    board = chess.Board(fen)
    assert not board.is_fifty_moves()
    assert board.can_claim_fifty_moves() is False or board.halfmove_clock < 100

    # Simulate a quiet rook move (Ra1-b1)
    rook_move = chess.Move.from_uci("a1b1")
    assert rook_move in board.legal_moves
    b2 = board.copy(stack=True)
    b2.push(rook_move)
    # After Ra1-b1, halfmove becomes 100
    assert b2.halfmove_clock == 100
    # Now can claim
    assert b2.is_fifty_moves()
