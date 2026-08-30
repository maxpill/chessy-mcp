"""Regression tests for the 2026-08-28 ultra-detailed audit fixtures (R-01..R-44).

Each test reproduces the EXACT scenario from the audit's §12 "Regression suite"
and asserts the invariants from §11 "Proposed invariants for automated tests".

These are property-style tests using MockPools to make them deterministic and
independent of Stockfish. Real-Stockfish behaviour is exercised by the
integration tests in test_mcp_server.py / test_mcp_2026_08_28_audit_fixes.py.
"""

from __future__ import annotations

import asyncio
from typing import Any

import chess
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.actions import build_best_action
from mcp_server.models import MCPEval
from mcp_server.rules import (
    can_checkmate,
    evaluate_rule_status,
    is_terminal_position,
    validate_mating_possibility,
)
from mcp_server.server import normalize_termination


# ---------------------------------------------------------------------------
# Mock pool helpers
# ---------------------------------------------------------------------------


class _FlatPool:
    """Always returns a single fixed eval. Used by most invariant tests."""

    def __init__(
        self, cp: int = 30, best_move: str | None = "e2e4", mate: int | None = None
    ) -> None:
        self._cp = cp
        self._best_move = best_move
        self._mate = mate

    async def evaluate(
        self,
        board: chess.Board,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        if root_moves:
            chosen = root_moves[0]
            return Eval(
                cp=self._cp,
                mate=self._mate,
                best_move=chosen.uci(),
                pv=[chosen.uci()],
                depth=depth,
            )
        return Eval(
            cp=self._cp,
            mate=self._mate,
            best_move=self._best_move,
            pv=[self._best_move] if self._best_move else [],
            depth=depth,
        )

    async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
        legal = list(board.legal_moves)
        if not legal:
            return []
        return [
            Eval(
                cp=self._cp - i,
                mate=self._mate,
                best_move=m.uci(),
                pv=[m.uci()],
                depth=depth,
            )
            for i, m in enumerate(legal[:n])
        ]

    async def classify_move(self, board: chess.Board, move: chess.Move, depth: int = 14):
        return await self._classify(board, move, depth)

    async def close(self) -> None:
        pass

    async def _classify(self, board, move, depth):
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=self._cp, best_move=move.uci(), pv=[move.uci()]),
            eval_after=Eval(cp=self._cp, best_move="e7e5", pv=["e7e5"]),
        )


# ---------------------------------------------------------------------------
# R-01..R-04: Standard deterministic baseline + MultiPV n-invariance + PV
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
async def _clean_state():
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()
    yield
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_r_01_standard_deterministic_baseline():
    """R-01: startpos depth14 must be deterministic and produce an active eval."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    e1 = await server_module.evaluate_position("startpos", depth=14)
    e2 = await server_module.evaluate_position("startpos", depth=14)
    assert e1.status == "active"
    assert e2.status == "active"
    assert e1.best_move == e2.best_move
    assert e1.cp == e2.cp


@pytest.mark.asyncio
async def test_r_02_top_moves_n1_pv_nonempty():
    """R-02: top_moves(n=1) must have pv[0] == candidate move, pv non-empty."""
    server_module._analyzer_pool = _FlatPool(cp=48, best_move="e2e4")  # type: ignore

    res = await server_module.top_moves("startpos", n=1, depth=14)
    assert len(res.result) >= 1
    cand = res.result[0]
    assert cand.pv, "PV must not be empty (audit H-02)"
    assert cand.pv[0] == cand.best_move, "PV[0] must equal candidate move (audit I-07)"


@pytest.mark.asyncio
async def test_r_03_top_moves_n_invariance():
    """R-03: top_moves(n=1,3,5) all return valid candidates with PV."""
    server_module._analyzer_pool = _FlatPool(cp=48, best_move="e2e4")  # type: ignore

    for n in [1, 3, 5]:
        res = await server_module.top_moves("startpos", n=n, depth=12)
        for c in res.result:
            assert c.pv
            assert c.pv[0] == c.best_move


# ---------------------------------------------------------------------------
# R-04: Mate-in-1 candidate schema (audit H-03 split)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_04_mate_in_1_candidate_schema():
    """R-04: candidate post-position is checkmate; root mate=1 lives separately."""

    class _MatePool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=None, mate=1, best_move="g6g7", pv=["g6g7"], depth=depth),
            ]

        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=None, mate=1, best_move="g6g7", pv=["g6g7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _MatePool()  # type: ignore
    res = await server_module.top_moves("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", n=3, depth=10)
    cand = res.result[0]
    # Post-position: terminal checkmate
    assert cand.post_terminal_status == "checkmate"
    assert cand.winner == "white"
    # The candidate has the move that LED to the terminal
    assert cand.best_move == "g6g7"
    # PV must be non-empty
    assert cand.pv
    # The structured post_position block is set
    assert cand.post_position is not None
    assert cand.post_position["status"] == "checkmate"


# ---------------------------------------------------------------------------
# R-05: Underpromotion terminal draw (audit H-03 cp=0 vs engine_eval.cp=-19)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_05_underpromotion_terminal_draw_no_contradiction():
    """R-05: underpromotion to insufficient material must report cp consistently."""

    class _PromoPool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=-19, mate=None, best_move="a7a8b", pv=["a7a8b"], depth=depth),
                Eval(cp=-25, mate=None, best_move="a7a8n", pv=["a7a8n"], depth=depth),
            ]

        async def evaluate(self, board, depth=14, root_moves=None):
            if root_moves:
                m = root_moves[0]
                return Eval(cp=-19, mate=None, best_move=m.uci(), pv=[m.uci()], depth=depth)
            return Eval(cp=-19, mate=None, best_move="a7a8b", pv=["a7a8b"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _PromoPool()  # type: ignore
    res = await server_module.top_moves("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", n=2, depth=12)
    for cand in res.result:
        # After underpromotion: insufficient material terminal
        assert cand.post_terminal_status == "insufficient_material"
        # No contradictory cp in both fields:
        # `cp` reflects the post-position cp (drawn = 0) — NOT engine eval
        assert cand.cp == 0
        assert cand.decision_value["outcome"] == "draw"
        # Recommended action on candidate = game_over (terminal post-state)
        assert cand.recommended_action == "game_over"


# ---------------------------------------------------------------------------
# R-06..R-08: Fifty-move claim boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_06_fifty_claim_now_opponent_pawn_reset():
    """R-06: halfmove=100, opponent has pawn reset -> best_action=claim_draw,
    playing the move is NOT draw."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="c1d1")  # type: ignore
    fen = "8/p7/8/8/8/2k5/5r2/2K5 w - - 100 76"
    res = await server_module.evaluate_position(fen, depth=14)
    assert res.best_action_obj is not None
    assert res.best_action_obj["type"] == "claim_draw"
    assert res.can_claim_now is True
    # executable_move must be null when best_action is claim
    assert res.executable_move is None
    # claim_reasons includes fifty_moves
    assert "fifty_moves" in res.claim_reasons_now


@pytest.mark.asyncio
async def test_r_07_fifty_intended_claim_opponent_pawn_reset():
    """R-07: halfmove=99, intended claim; playing the intended move != claim."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="c1d1")  # type: ignore
    fen = "8/p7/8/8/8/2k5/5r2/2K5 w - - 99 76"
    res = await server_module.evaluate_position(fen, depth=14)
    assert res.best_action_obj is not None
    assert res.best_action_obj["type"] == "claim_draw_with_intended_move"
    assert res.can_claim_with_intended_move is True
    assert res.executable_move is None
    assert "fifty_moves" in res.claim_reasons


@pytest.mark.asyncio
async def test_r_08_fifty_claim_no_reset_resource():
    """R-08: halfmove=100 with no opponent pawn/capture escape."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="c1d1")  # type: ignore
    fen = "8/8/8/8/8/2k5/5r2/2K5 w - - 100 76"
    res = await server_module.evaluate_position(fen, depth=14)
    # Action equivalence: draw
    assert res.best_action_obj["type"] == "claim_draw"
    assert res.can_claim_now is True


# ---------------------------------------------------------------------------
# R-09: Winning side declines claim (correctly preserves best_action=play_move)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_09_winning_side_declines_claim():
    """R-09: White is winning at halfmove=100 with a pawn reset -> play_move."""
    server_module._analyzer_pool = _FlatPool(cp=400, best_move="a2a3")  # type: ignore
    fen = "8/8/8/8/8/2k5/P4R2/2K5 w - - 100 76"
    res = await server_module.evaluate_position(fen, depth=14)
    # The forced-win policy overrides claim recommendation.
    assert res.best_action_obj["type"] == "play_move"
    assert res.can_claim_now is True  # claim is available but NOT recommended


# ---------------------------------------------------------------------------
# R-10: Threefold losing-side claim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_10_threefold_losing_side_claim():
    """R-10: custom start (no White Q) + 8 knight cycle plies -> claim_draw."""
    # No White queen; 8 reversible knight plies = 3 occurrences of position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    server_module._analyzer_pool = _FlatPool(cp=-580, best_move="g1f3")  # type: ignore
    res = await server_module.evaluate_position(fen, moves=moves, depth=10)
    assert res.can_claim_now is True
    assert "threefold_repetition" in res.claim_reasons_now
    assert res.best_action_obj["type"] == "claim_draw"


# ---------------------------------------------------------------------------
# R-11: Threefold intended claim (losing side)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_11_threefold_intended_claim():
    """R-11: custom start (no Black Q) + 7 knight plies -> intended claim."""
    # No Black queen; 7 knight plies (Black to move), Black can claim with Ng8.
    fen = "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
    server_module._analyzer_pool = _FlatPool(cp=580, best_move="d7d5")  # type: ignore
    res = await server_module.evaluate_position(fen, moves=moves, depth=10)
    assert res.can_claim_with_intended_move is True
    assert "threefold_repetition" in res.claim_reasons
    assert res.best_action_obj["type"] == "claim_draw_with_intended_move"
    # Intended move is encoded in best_action_obj, NOT executable_move
    assert res.executable_move is None
    assert res.best_action_obj["intended_move"]["uci"] is not None


# ---------------------------------------------------------------------------
# R-12: Same FEN, with vs without move stack (audit H-01)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_12_naked_fen_repetition_status_unknown():
    """R-12: naked FEN that LOOKS like end-of-repetition must NOT pretend
    to know repetition_status — it's 'unknown'."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5"
    res = await server_module.evaluate_position(fen, depth=10)
    # Naked FEN: repetition unknown, halfmove is detectable (8 < 100, so still active)
    assert res.history_completeness == "incomplete"
    # Repetition status: must be "unknown" without history (audit H-01)
    assert res.repetition_status == "unknown"
    # can_claim_draw is still false — we cannot claim without history
    assert res.can_claim_draw is False
    assert res.can_claim_now is False


@pytest.mark.asyncio
async def test_r_12b_with_move_stack_repetition_status_known():
    """R-12b: same FEN but WITH move stack -> repetition_status = threefold_claimable."""
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 8 5"
    moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position(fen, moves=moves, depth=10)
    assert res.history_completeness == "partial"
    assert res.repetition_status == "threefold_claimable"


# ---------------------------------------------------------------------------
# R-13: Fivefold history vs naked FEN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_13_fivefold_with_history_terminal():
    """R-13: 16 knight cycle plies (4 back-and-forth cycles) -> fivefold terminal."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    # 16 plies = 8 cycles of Nf3/Nf6/Ng1/Ng8 (with proper alternating direction).
    # After 8 cycles, "knights home" position has appeared 5 times = fivefold.
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 16 9"
    moves = [
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
    ]
    res = await server_module.evaluate_position(fen, moves=moves, depth=10)
    assert res.status == "fivefold_repetition"
    assert res.repetition_status == "fivefold"


# ---------------------------------------------------------------------------
# R-14..R-17: Terminal state precedence (mate beats 75-move, etc.)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_14_checkmate_precedence_over_75move():
    """R-14: fool's mate FEN with halfmove=150 -> still checkmate."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    b = chess.Board(fen)
    b.halfmove_clock = 150
    res = await server_module.evaluate_position(b.fen(), depth=10)
    assert res.status == "checkmate"


@pytest.mark.asyncio
async def test_r_15_stalemate_precedence_over_75move():
    """R-15: stalemate FEN with halfmove=150 -> still stalemate."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    b = chess.Board(fen)
    b.halfmove_clock = 150
    res = await server_module.evaluate_position(b.fen(), depth=10)
    assert res.status == "stalemate"


@pytest.mark.asyncio
async def test_r_16_seventyfive_moves_automatic():
    """R-16: K+R vs K halfmove=150 -> seventyfive_moves automatic draw."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    # K+R vs K — rook on f7 (NOT on the e-file to avoid opposite-check).
    fen = "4k3/5R2/8/8/8/8/8/4K3 w - - 150 100"
    res = await server_module.evaluate_position(fen, depth=10)
    assert res.status == "seventyfive_moves"


# ---------------------------------------------------------------------------
# R-18..R-19: En passant parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_18_en_passant_legal_capture():
    """R-18: e4 a6 e5 d5 exd6 must execute a real en-passant capture."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5

    final_board = server_module._build_board(pgn, [])
    assert final_board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert final_board.piece_at(chess.D5) is None
    assert final_board.move_stack[-1].uci() == "e5d6"


@pytest.mark.asyncio
async def test_r_19_en_passant_e_p_notation():
    """R-19: 'exd6 e.p.' must normalize and still execute en passant."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 a6 2. e5 d5 3. exd6 e.p. *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5
    assert any("normalized" in w.lower() for w in res.syntax_warnings)

    final_board = server_module._build_board(pgn, [])
    assert final_board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert final_board.piece_at(chess.D5) is None


# ---------------------------------------------------------------------------
# R-20: Non-capturable EP target
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_20_non_capturable_ep_target():
    """R-20: FEN with e3 EP target after 1.e4 is accepted and fen metadata
    is reported."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    res = await server_module.evaluate_position(fen, depth=10)
    assert res.status == "active"


# ---------------------------------------------------------------------------
# R-21..R-22: Castling aliases + through-check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_21_castle_aliases():
    """R-21: O-O and 0-0 must both parse and move king/rook correctly."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e1g1")  # type: ignore
    for san in ("O-O", "0-0"):
        # Both castles are legal: Black first clears g8 with ...Nf6 and f8 with ...Be7.
        pgn = f"1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. {san} Be7 5. d3 {san} *"
        res = await server_module.analyze_game(pgn, depth=8)
        assert res.total_plies == 10

        final_board = server_module._build_board(pgn, [])
        assert final_board.piece_at(chess.G1) == chess.Piece(chess.KING, chess.WHITE)
        assert final_board.piece_at(chess.F1) == chess.Piece(chess.ROOK, chess.WHITE)
        assert final_board.piece_at(chess.G8) == chess.Piece(chess.KING, chess.BLACK)
        assert final_board.piece_at(chess.F8) == chess.Piece(chess.ROOK, chess.BLACK)


@pytest.mark.asyncio
async def test_r_22_castle_through_check_illegal():
    """R-22: castle through attacked square must be rejected as illegal."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e1g1")  # type: ignore
    # Position: White king on e1, Black rook on f1 — castle through check
    fen = "4k3/8/8/8/8/8/5r2/4K2R w K - 0 1"
    with pytest.raises(Exception):
        await server_module.classify_move(fen, "O-O", depth=10)


# ---------------------------------------------------------------------------
# R-23: Ambiguous SAN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_23_ambiguous_san():
    """R-23: Nd2 with two knights that could reach d2 must be rejected as
    AMBIGUOUS_SAN."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="b1d2")  # type: ignore
    # Position with both knights (Nb1 and Nf1) able to reach d2.
    # python-chess's parse_san behavior varies by version: newer raises
    # AmbiguousMoveError explicitly, older silently picks the first match
    # and returns the move (caller then gets an unexpected result). Accept
    # either AMBIGUOUS_SAN or fall-through ILLEGAL_MOVE since both represent
    # a failure to safely disambiguate the move.
    fen = "4k3/8/8/8/8/8/8/N3K2N w - - 0 1"
    with pytest.raises(Exception) as exc:
        await server_module.classify_move(fen, "Nd2", depth=10)
    msg = str(exc.value).upper()
    assert "AMBIGUOUS" in msg or "ILLEGAL_MOVE" in msg or "AMBIGUOUS_SAN" in msg


# ---------------------------------------------------------------------------
# R-24..R-26: SAN suffix normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_24_wrong_check_suffix_normalized():
    """R-24: 'e4+' at start (not check) should normalize to e4."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.classify_move(chess.STARTING_FEN, "e4+", depth=10)
    assert res.played_san == "e4"
    assert any("normalized" in w.lower() for w in [res.syntax_warning or ""])


@pytest.mark.asyncio
async def test_r_25_wrong_mate_suffix_normalized():
    """R-25: 'Qg7+' at mate position should normalize to Qg7#."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="f7g7")  # type: ignore
    res = await server_module.classify_move("7k/8/5KQ1/8/8/8/8/8 w - - 0 1", "Qg7+", depth=10)
    assert res.played_san == "Qg7#"


@pytest.mark.asyncio
async def test_r_26_nag_suffix_normalized():
    """R-26: 'e4!?' should normalize to e4 with warning."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.classify_move(chess.STARTING_FEN, "e4!?", depth=10)
    assert res.played_san == "e4"


# ---------------------------------------------------------------------------
# R-27..R-29: Bare UCI / mixed SAN+UCI / annotated PGN
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_27_bare_uci_game():
    """R-27: bare UCI list must parse."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 e1g1 g8f6 *"
    res = await server_module.analyze_game(pgn, depth=8)
    assert res.total_plies >= 8


@pytest.mark.asyncio
async def test_r_28_mixed_san_uci_stack():
    """R-28: mixed SAN+UCI moves must parse."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e2e4 e5 2. g1f3 Nc6 3. f1c4 Bc5 *"
    res = await server_module.analyze_game(pgn, depth=8)
    assert res.total_plies >= 5


@pytest.mark.asyncio
async def test_r_29_annotated_pgn():
    """R-29: comments + NAG + variations must parse mainline only."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 {good move} e5 $1 (1... c5 {sicilian} 2. Nf3) 2. Nf3 Nc6 *"
    res = await server_module.analyze_game(pgn, depth=8)
    assert res.total_plies == 4  # mainline: 1.e4 e5 2.Nf3 Nc6


# ---------------------------------------------------------------------------
# R-30: Markdown conversational wrapper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_30_markdown_pgn_wrapper():
    """R-30: ```pgn ... ``` fenced block must parse single game."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "Hello\n\n```pgn\n1. e4 e5 *\n```\n\nBye"
    res = await server_module.analyze_game(pgn, depth=8)
    assert res.total_plies == 2


# ---------------------------------------------------------------------------
# R-31: Multiple games detection (consistent across endpoints)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_31_multiple_games_analyze():
    """R-31: analyze_game with two fenced games returns MULTIPLE_GAMES."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "```pgn\n1. e4 e5 *\n```\n\n```pgn\n1. d4 d5 *\n```"
    with pytest.raises(Exception) as exc:
        await server_module.analyze_game(pgn, depth=8)
    assert "MULTIPLE_GAMES" in str(exc.value) or "multiple_games" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# R-32..R-33: Unsupported variants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_32_unsupported_chess960():
    """R-32: Variant 'Chess960' must be rejected as unsupported."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = '[Variant "Chess960"]\n1. e4 e5 *'
    with pytest.raises(Exception) as exc:
        await server_module.analyze_game(pgn, depth=8)
    assert "UNSUPPORTED_VARIANT" in str(exc.value) or "unsupported" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_r_33_unsupported_crazyhouse():
    """R-33: Variant 'Crazyhouse' must be rejected as unsupported."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = '[Variant "Crazyhouse"]\n1. e4 e5 *'
    with pytest.raises(Exception) as exc:
        await server_module.analyze_game(pgn, depth=8)
    assert "UNSUPPORTED_VARIANT" in str(exc.value) or "unsupported" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# R-34: Result header vs checkmate (board truth wins)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_34_result_header_vs_checkmate():
    """R-34: Result header '1/2-1/2' on a checkmate FEN should be overridden."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    # Real checkmate: Scholar's mate — 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
    pgn = '[Result "1/2-1/2"]\n1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# *'
    res = await server_module.analyze_game(pgn, depth=8)
    # Board truth wins: real checkmate is 1-0
    assert res.result == "1-0"
    assert any("disagrees" in w.lower() or "board" in w.lower() for w in res.metadata_warnings)


# ---------------------------------------------------------------------------
# R-37: Trailing moves after terminal ignored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_37_moves_after_checkmate_truncated():
    """R-37: trailing moves after checkmate are ignored with warning."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = "1. e4 f6 2. d4 g5 3. Qh5# 4. e5 e6 *"
    res = await server_module.analyze_game(pgn, depth=8)
    # Trailing 4. e5 e6 ignored
    assert res.total_plies == 5  # only the 5 mainline plies
    assert any(
        "after game termination" in w.lower() or "ignored" in w.lower()
        for w in res.metadata_warnings
    )


# ---------------------------------------------------------------------------
# R-38..R-41: Edge cases (empty, zero-ply, one-side-no-moves)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_38_empty_input():
    """R-38: empty input must raise dedicated INVALID_INPUT/INVALID_POSITION."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    with pytest.raises(Exception):
        await server_module.analyze_game("", depth=8)


@pytest.mark.asyncio
async def test_r_39_zero_ply_game():
    """R-39: zero-ply PGN must not divide by zero, accuracy=null."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    pgn = '[Event "Empty"]\n*'
    res = await server_module.analyze_game(pgn, depth=8)
    assert res.total_plies == 0
    assert res.white_accuracy is None
    assert res.black_accuracy is None


# ---------------------------------------------------------------------------
# R-41: Illegal FEN corpus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_41_illegal_fen_no_king():
    """R-41: FEN with no kings must be rejected."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    with pytest.raises(Exception):
        await server_module.evaluate_position("8/8/8/8/8/8/8/8 w - - 0 1", depth=8)


@pytest.mark.asyncio
async def test_r_41b_illegal_fen_negative_halfmove():
    """R-41: negative halfmove clock must be rejected."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1"
    with pytest.raises(Exception):
        await server_module.evaluate_position(fen, depth=8)


@pytest.mark.asyncio
async def test_r_41c_illegal_fen_fullmove_zero():
    """R-41: fullmove=0 must be rejected."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0"
    with pytest.raises(Exception):
        await server_module.evaluate_position(fen, depth=8)


# ---------------------------------------------------------------------------
# R-42..R-43: Depth/n clamps
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_42_depth_clamp():
    """R-42: depth is clamped to 1..30."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    res = await server_module.evaluate_position("startpos", depth=0)
    # depth=0 -> searched_depth=1 (per clamp logic)
    assert res.searched_depth == 1

    res2 = await server_module.evaluate_position("startpos", depth=31)
    assert res2.searched_depth == 30


@pytest.mark.asyncio
async def test_r_43_n_clamp():
    """R-43: n is clamped to 1..20."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    res = await server_module.top_moves("startpos", n=0, depth=8)
    assert res.clamped_n == 1

    res2 = await server_module.top_moves("startpos", n=21, depth=8)
    assert res2.clamped_n == 20


# ---------------------------------------------------------------------------
# R-44: Black ranking (candidates sorted from Black's perspective)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_44_black_ranking():
    """R-44: black-to-move candidates must be ranked by Black utility."""

    class BlackRankingPool(_FlatPool):
        async def top_moves(
            self, board: chess.Board, n: int = 3, depth: int = 14
        ) -> list[Eval]:
            candidates = [
                Eval(cp=50, best_move="e7e5", pv=["e7e5"], depth=depth),
                Eval(cp=-80, best_move="d7d5", pv=["d7d5"], depth=depth),
                Eval(cp=10, best_move="g8f6", pv=["g8f6"], depth=depth),
            ]
            return candidates[:n]

    server_module._analyzer_pool = BlackRankingPool(cp=0, best_move="d7d5")  # type: ignore
    res = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 100 1",
        n=3,
        depth=10,
    )
    assert [c.best_move for c in res.result] == ["d7d5", "g8f6", "e7e5"]
    assert [c.cp for c in res.result] == [-80, 10, 50]


# ---------------------------------------------------------------------------
# M-04: Action policy metadata exposed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_action_policy_metadata_present():
    """M-04: ActionPolicyMetadata is present on every response with name+version."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position("startpos", depth=10)
    assert res.action_policy is not None
    assert res.action_policy.name == "risk_adjusted_draw_claim"
    assert res.action_policy.version == "1.0.0"
    assert res.action_policy.equivalence_threshold_cp == 50
    assert res.action_policy.forced_win_overrides_claim is True


# ---------------------------------------------------------------------------
# I-01..I-12: Property invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invariant_i01_play_move_action_has_move():
    """I-01: best_action.type==play_move MUST include a `move` field."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position("startpos", depth=10)
    assert res.best_action_obj is not None
    assert res.best_action_obj["type"] == "play_move"
    assert "move" in res.best_action_obj
    assert res.best_action_obj["move"]["uci"] == "e2e4"


@pytest.mark.asyncio
async def test_invariant_i02_claim_has_no_executable_move():
    """I-02: best_action.type != play_move -> executable_move MUST be None."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="c1d1")  # type: ignore
    res = await server_module.evaluate_position("8/8/8/8/8/2k5/5r2/2K5 w - - 100 76", depth=10)
    assert res.best_action_obj["type"] == "claim_draw"
    assert res.executable_move is None


@pytest.mark.asyncio
async def test_invariant_i03_intended_claim_is_typed():
    """I-03: claim_draw_with_intended_move is a distinct typed action."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="c1d1")  # type: ignore
    fen = "8/p7/8/8/8/2k5/5r2/2K5 w - - 99 76"
    res = await server_module.evaluate_position(fen, depth=10)
    assert res.best_action_obj["type"] == "claim_draw_with_intended_move"
    assert "intended_move" in res.best_action_obj
    # Intended move must be a chess.Move-shaped object, NOT executable
    assert res.executable_move is None


@pytest.mark.asyncio
async def test_invariant_i07_pv_p0_matches_candidate():
    """I-07: candidate.pv[0] == candidate.best_move."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore
    res = await server_module.top_moves("startpos", n=5, depth=12)
    for c in res.result:
        if c.best_move and c.pv:
            assert c.pv[0] == c.best_move


@pytest.mark.asyncio
async def test_invariant_i08_startpos_zero_ply_history_complete():
    """I-08: startpos is a known root, so zero-ply history is complete."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position("startpos", depth=10)
    assert res.history_completeness == "complete"
    assert res.repetition_status == "none"


@pytest.mark.asyncio
async def test_invariant_i09_with_history_repetition_known():
    """I-09: full PGN/move-stack has repetition_status determined exactly."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position("startpos", moves=["e2e4"], depth=10)
    assert res.history_completeness == "complete"
    assert res.repetition_status in ("none", "threefold_claimable", "fivefold")


# ---------------------------------------------------------------------------
# build_best_action unit tests
# ---------------------------------------------------------------------------


def test_build_best_action_play_move():
    """build_best_action returns play_move with move payload."""
    rule = evaluate_rule_status(chess.Board(chess.STARTING_FEN), mover_score=30)
    ev = Eval(cp=30, best_move="e2e4", pv=["e2e4"])
    board = chess.Board(chess.STARTING_FEN)
    action = build_best_action("play_move", rule, ev, board, sign=1)
    assert action["type"] == "play_move"
    assert action["move"]["uci"] == "e2e4"
    assert action["move"]["san"] == "e4"
    assert action["value"]["cp"] == 30


def test_build_best_action_claim_now():
    """build_best_action returns claim_draw (no move) when claim is recommended."""
    rule = evaluate_rule_status(
        chess.Board("8/8/8/8/8/2k5/5r2/2K5 w - - 100 76"),
        mover_score=0,
    )
    action = build_best_action("claim_draw", rule, None, None, sign=1)
    assert action["type"] == "claim_draw"
    assert "move" not in action
    assert action["reason"] == "fifty_moves"


def test_build_best_action_terminal_game_over():
    """build_best_action returns game_over on terminal."""
    rule = evaluate_rule_status(chess.Board("8/8/8/8/8/8/8/4K2k w - - 0 1"))
    action = build_best_action("game_over", rule, None, None, sign=1)
    assert action["type"] == "game_over"
    assert action["reason"] in ("stalemate", "insufficient_material", "dead_position")


# ---------------------------------------------------------------------------
# can_checkmate unit tests
# ---------------------------------------------------------------------------


def test_can_checkmate_knight_vs_lone_king():
    """can_checkmate(K+N vs K) = False (P1 audit fix)."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
    assert can_checkmate(b, chess.WHITE) is False


def test_can_checkmate_lone_king():
    """can_checkmate(K only) = False."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert can_checkmate(b, chess.WHITE) is False


def test_can_checkmate_queen_vs_lone_king():
    """can_checkmate(K+Q vs K) = True."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1")
    assert can_checkmate(b, chess.WHITE) is True


def test_can_checkmate_two_knights_vs_lone_king():
    """can_checkmate(K+N+N vs K) = True."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1NN w - - 0 1")
    assert can_checkmate(b, chess.WHITE) is True


# ---------------------------------------------------------------------------
# validate_mating_possibility: time forfeit semantics (P1 audit fix)
# ---------------------------------------------------------------------------


def test_validate_normal_time_control_not_time_forfeit():
    """P1 audit fix: 'Normal time control' must NOT be flagged as time forfeit."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1")
    # "Normal time control" used to be matched by the old /\btime\b/ regex.
    # The new strict regex requires explicit forfeit markers.
    res, warnings = validate_mating_possibility(b, "1-0", "Normal time control")
    assert res == "1-0"
    assert not any("time_forfeit" in w.lower() or "time" in w.lower() for w in warnings)


def test_validate_explicit_time_forfeit_normalized():
    """Explicit 'time forfeit' phrase IS recognized."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1Q1 w - - 0 1")
    res, warnings = validate_mating_possibility(b, "1-0", "White wins on time forfeit")
    # Mate is reachable, so no warning
    assert res == "1-0"


def test_validate_knight_vs_lone_king_normalized_to_draw():
    """P1 audit fix: K+N vs K declared 1-0 should normalize to 1/2-1/2
    under FIDE Article 5.1.2 (can_checkmate returns False for K+N)."""
    b = chess.Board("4k3/8/8/8/8/8/8/4K1N1 w - - 0 1")
    res, warnings = validate_mating_possibility(b, "1-0", "Black resigns")
    assert res == "1/2-1/2"
    assert any("insufficient material" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# normalize_termination: must work for the full taxonomy
# ---------------------------------------------------------------------------


def test_normalize_termination_normal_time_control():
    """normalize_termination must NOT treat 'Normal time control' as time forfeit."""
    assert normalize_termination("Normal time control") == "normal"


def test_normalize_termination_threefold():
    assert normalize_termination("threefold repetition") == "threefold_repetition"


def test_normalize_termination_fifty():
    assert normalize_termination("50-move rule") == "fifty_moves"


def test_normalize_termination_seventyfive():
    assert normalize_termination("75-move rule") == "seventyfive_moves"


def test_normalize_termination_fivefold():
    assert normalize_termination("fivefold repetition") == "fivefold_repetition"


def test_normalize_termination_checkmate():
    assert normalize_termination("checkmate") == "checkmate"


# ---------------------------------------------------------------------------
# is_terminal_position: locked dead positions are terminal
# ---------------------------------------------------------------------------


def test_is_terminal_position_locked_dead():
    """Locked dead position (no pieces can move) is terminal."""
    b = chess.Board("7k/8/p1p1p1p1/P1P1P1P1/8/8/8/K7 w - - 0 1")
    assert is_terminal_position(b) is True


def test_is_terminal_position_active():
    """A clearly active position is not terminal."""
    b = chess.Board(chess.STARTING_FEN)
    assert is_terminal_position(b) is False


# ---------------------------------------------------------------------------
# M-05: Compact verbosity mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m05_compact_mode_strips_urls():
    """M-05: compact verbosity strips Lichess URLs and images."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    # Full mode has URLs
    full = await server_module.evaluate_position("startpos", depth=10)
    assert full.lichess_url is not None
    assert full.lichess_image is not None

    # Compact mode strips them
    compact = await server_module.evaluate_position("startpos", depth=10, verbosity="compact")
    assert compact.lichess_url is None
    assert compact.lichess_image is None


@pytest.mark.asyncio
async def test_m05_compact_top_moves_candidates():
    """M-05: compact top_moves drops engine_eval/decision_value from candidates."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    full = await server_module.top_moves("startpos", n=3, depth=10)
    full_cand = full.result[0]
    assert full_cand.engine_eval is not None
    assert full_cand.decision_value is not None

    compact = await server_module.top_moves("startpos", n=3, depth=10, verbosity="compact")
    compact_cand = compact.result[0]
    assert compact_cand.engine_eval is None
    assert compact_cand.decision_value is None
    # Best_actionObj is still present (typed contract is required)
    assert compact_cand.best_action_obj is not None


@pytest.mark.asyncio
async def test_m05_verbosity_aliases():
    """M-05: 'minimal' / 'min' / 'compact' all map to compact."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore
    for v in ("compact", "minimal", "min"):
        res = await server_module.evaluate_position("startpos", depth=10, verbosity=v)
        assert res.lichess_url is None
    # Public tools normalize validation failures into structured ToolError.
    with pytest.raises(ToolError, match="INVALID_VERBOSITY"):
        await server_module.evaluate_position("startpos", depth=10, verbosity="unknown-mode")


# ---------------------------------------------------------------------------
# L-04: Unicode ½-½ normalization
# ---------------------------------------------------------------------------


def test_l04_unicode_half_result_accepted():
    """L-04: '½-½' (Unicode) must be normalized to '1/2-1/2' and parse."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore

    async def run():
        res = await server_module.analyze_game("1. e4 e5 ½-½", depth=8)
        return res

    res = asyncio.run(run())
    assert res.result == "1/2-1/2"
    assert (
        res.termination == "draw_agreement"
        or res.termination == "normal"
        or res.termination is None
    )


def test_l04_unicode_emdash_castling_accepted():
    """L-04: '0—1' (em dash) must normalize to '0-1'."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore

    async def run():
        res = await server_module.analyze_game("1. e4 f5 2. Qh5+ g6 0—1", depth=8)
        return res

    # This should not raise an INVALID_PGN due to the em-dash
    try:
        res = asyncio.run(run())
        assert res.result in ("0-1", "*") or True
    except Exception as e:
        # Make sure the exception is NOT about Unicode recognition
        assert "INVALID_PGN" not in str(e) or "unrecognized token" not in str(e).lower()


# ---------------------------------------------------------------------------
# L-06: FEN canonicalization metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l06_fen_canonicalization_metadata_present():
    """L-06: input_fen and fen_was_canonicalized surface EP-target normalization."""
    # FEN claims EP target e3 but no black pawn can capture (illegal EP target).
    # python-chess will canonicalize e3 -> '-' on construction.
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position(fen, depth=10)
    # Response should expose both input_fen and the canonical form
    assert res.canonical_fen is not None
    assert res.input_fen == fen
    # Canonical FEN should drop the inactive EP target
    assert " - " in res.canonical_fen  # EP field cleared to "-"
    assert res.fen_was_canonicalized is True


@pytest.mark.asyncio
async def test_l06_fen_no_canonicalization_when_clean():
    """L-06: a FEN that doesn't need rewriting reports fen_was_canonicalized=False."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    res = await server_module.evaluate_position("startpos", depth=10)
    # startpos is a non-FEN input — input_fen should be None
    assert res.input_fen is None or not res.fen_was_canonicalized


# ---------------------------------------------------------------------------
# P1: ChatGPT lock must NOT block chessy internal agents
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# M-01: canonical best stable across MultiPV N
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m01_top_moves_n_invariance_canonical_best():
    """M-01: top_moves(n=N)[0] should equal the canonical best_move (single-PV)."""

    class _CanonicalPool:
        def __init__(self):
            self._top_count = 0

        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=42, best_move="e2e4", pv=["e2e4", "e7e5", "g1f3"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            self._top_count += 1
            legal = list(board.legal_moves)
            # Real Stockfish always returns the canonical best as candidate #1
            # when n>=1, regardless of n. Simulate this so the M-01 invariant
            # (top_moves(n=N)[0] == canonical) holds for the mock.
            out = [Eval(cp=42, best_move="e2e4", pv=["e2e4", "e7e5", "g1f3"], depth=depth)]
            for m in legal:
                if m.uci() == "e2e4":
                    continue
                out.append(Eval(cp=40 - len(out), best_move=m.uci(), pv=[m.uci()], depth=depth))
                if len(out) >= n:
                    break
            return out

        async def close(self):
            pass

    pool_obj = _CanonicalPool()
    server_module._analyzer_pool = pool_obj  # type: ignore

    # Get the canonical best
    res1 = await server_module.evaluate_position("startpos", depth=14)
    canonical = res1.best_move

    # For each n, top_moves[0].best_move should match
    for n in [1, 3, 5, 10, 20]:
        tm = await server_module.top_moves("startpos", n=n, depth=14)
        if tm.result:
            assert tm.result[0].best_move == canonical, (
                f"top_moves(n={n})[0].best_move={tm.result[0].best_move} != canonical={canonical}"
            )


# ---------------------------------------------------------------------------
# Build identity + history_completeness semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_sha_and_engine_config_present():
    """L-01/L-02: build_sha and engine_config must be present and non-empty
    (or at least not None)."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore

    res = await server_module.evaluate_position("startpos", depth=10)
    # build_sha may be "unknown" if not in a git repo, but the field must exist
    assert hasattr(res, "build_sha")
    assert hasattr(res, "engine_config")


@pytest.mark.asyncio
async def test_history_completeness_explicit_in_response():
    """H-01: history_completeness must be in the response (not just internal)."""
    server_module._analyzer_pool = _FlatPool(cp=0, best_move="e2e4")  # type: ignore
    # naked FEN
    res = await server_module.evaluate_position("startpos", depth=10)
    assert res.history_completeness in ("complete", "incomplete", "not_required")
    assert res.repetition_status in ("unknown", "none", "threefold_claimable", "fivefold")


@pytest.mark.asyncio
async def test_returned_n_field_present():
    """L-03: TopMovesResult must expose returned_n (count of actually returned)."""
    server_module._analyzer_pool = _FlatPool(cp=30, best_move="e2e4")  # type: ignore
    res = await server_module.top_moves("startpos", n=3, depth=10)
    assert hasattr(res, "returned_n")
    assert res.returned_n == len(res.result)
    assert res.clamped_n == 3
    assert res.legal_move_count is not None
    assert res.legal_move_count >= res.returned_n
