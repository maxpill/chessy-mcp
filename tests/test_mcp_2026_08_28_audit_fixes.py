"""Tests verifying fixes for all issues identified in Chess_MCP_Ultra_Deep_Post_Fix_Audit_2026-08-28.md."""

import chess
import pytest

from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.server import normalize_termination


class _MockFixedPool:
    async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None):
        return Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=depth)

    async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14):
        legal = list(board.legal_moves)
        res = []
        for m in legal[:n]:
            res.append(Eval(cp=0, best_move=m.uci(), pv=[m.uci()], depth=depth))
        return res

    async def classify_move(self, board: chess.Board, move: chess.Move, depth: int = 14):
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=Eval(cp=0, best_move=move.uci(), pv=[move.uci()]),
            eval_after=Eval(cp=0, best_move="e7e5", pv=["e7e5"]),
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_claim50_persist_001_no_false_blunder_when_draw_preserved():
    """CLAIM50-PERSIST-001: Move preserves draw at halfmove 100 without blunder/effective loss."""
    await server_module._cache.clear()

    class _FiftyPool:
        async def evaluate(
            self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None
        ):
            return Eval(cp=0, best_move="b1c1", pv=["b1c1"], depth=depth)

        async def classify_move(self, board: chess.Board, move: chess.Move, depth: int = 14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=0, best_move="b1c1", pv=["b1c1"]),
                eval_after=Eval(cp=0, best_move="h8g8", pv=["h8g8"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = _FiftyPool()  # type: ignore
    fen = "6kq/8/8/8/8/8/8/1K6 w - - 100 51"
    res = await server_module.classify_move(fen, move="Kc1", depth=14)
    assert res.played == "b1c1"
    assert res.move_class == MoveClass.BEST
    assert res.is_engine_best is True
    assert res.centipawn_loss == 0
    assert res.raw_centipawn_loss == 0
    assert res.effective_loss == 0
    assert res.action_equivalent is True
    assert res.classification_verified is True


@pytest.mark.asyncio
async def test_game_claim_001_final_ply_intended_claim_not_blunder():
    """GAME-CLAIM-001: Full game with final ply reaching 50-move draw is graded BEST."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MockFixedPool()  # type: ignore

    pgn = """[Event "50 Move Claim Game"]
[SetUp "1"]
[FEN "8/8/8/8/8/8/P4k2/R6K w - - 99 50"]
[Result "1/2-1/2"]
[Termination "50-move rule"]

50. Rf1+ 1/2-1/2"""
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 1
    assert res.white_blunders == 0
    assert res.result == "1/2-1/2"
    assert res.termination == "fifty_moves"


@pytest.mark.asyncio
async def test_term_normalization_and_validation():
    """TERM-001 through TERM-009: Termination taxonomy and validation checks."""
    assert normalize_termination("Threefold repetition") == "threefold_repetition"
    assert normalize_termination("50-move rule") == "fifty_moves"
    assert normalize_termination("Rules infraction: 50-move rule exceeded") == "fifty_moves"
    assert normalize_termination("Normal") == "normal"
    assert normalize_termination("White wins by adjudication") == "adjudication"
    assert normalize_termination("Time forfeit; mating material exists") == "time_forfeit"
    assert normalize_termination("Unknown random text") is None

    # Test analyze_game with standard "normal" termination
    await server_module._cache.clear()
    server_module._analyzer_pool = _MockFixedPool()  # type: ignore
    pgn_normal = """[Event "Casual"]
[Termination "Normal"]
1. e4 e5 *"""
    res = await server_module.analyze_game(pgn_normal, depth=10)
    assert res.termination == "normal"
    assert not any("disagrees" in w for w in res.metadata_warnings)


@pytest.mark.asyncio
async def test_pgn_semicolons_comments_and_escapes():
    """PGN-SEMICOLON, PGN-CONVERSATION, PGN-DIAG-SCOPE tests."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MockFixedPool()  # type: ignore

    # 1. Semicolon inside tag value
    pgn_tag_semi = """[Event "Match; Round 1"]\n1. e4 e5 *"""
    res1 = await server_module.analyze_game(pgn_tag_semi, depth=10)
    assert res1.event == "Match; Round 1"
    assert res1.total_plies == 2

    # 2. Semicolon with result token inside comment
    pgn_semi_res = "1. e4 e5 ; fake 1-0 comment\n2. Nf3 Nc6 *"
    res2 = await server_module.analyze_game(pgn_semi_res, depth=10)
    assert res2.total_plies == 4

    # 3. Conversational preamble with example tag
    pgn_conv = """I mentioned [FEN "7k/8/8/8/8/8/8/7K w - - 0 1"] as an example.\n\n[Event "Real Game"]\n1. e4 e5 *"""
    res3 = await server_module.analyze_game(pgn_conv, depth=10)
    assert res3.event == "Real Game"
    assert res3.total_plies == 2

    # 4. Escape % line with moves
    pgn_pct = "1. e4 e5\n% 1-0 ignored line with Nf3 Nc6\n2. Nf3 Nc6 *"
    res4 = await server_module.analyze_game(pgn_pct, depth=10)
    assert res4.total_plies == 4
    assert not any("ignored" in w for w in res4.metadata_warnings)


@pytest.mark.asyncio
async def test_top_moves_terminal_state_schema():
    """TOP-STATE-001 & TOP-TERMINAL-SCHEMA-001: Terminal candidate fields in top_moves."""
    await server_module._cache.clear()

    class _MatePool:
        async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14):
            return [
                Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = _MatePool()  # type: ignore
    fen_mate = "7k/5Q2/5K2/8/8/8/8/8 w - - 0 1"
    tm = await server_module.top_moves(fen_mate, n=3, depth=10)
    assert tm.status == "active"
    best_cand = tm.result[0]
    assert best_cand.post_terminal_status == "checkmate"
    assert best_cand.status == "checkmate"
    # mate field is reported as the terminal-state marker (0 or None for already-checkmated);
    # the Stockfish-style distance is preserved in decision_value via winner="white".
    assert best_cand.winner == "white"
    assert best_cand.mate in (0, 1)  # 0=terminal marker, 1=stockfish-style; both valid
    # CRITICAL P1#3 fix: post-candidate state is consistent — winner is set, decision_value
    # is from White's perspective (matching the cp convention).
    assert best_cand.decision_value.get("perspective") == "white"
    assert best_cand.decision_value.get("outcome") == "win"


@pytest.mark.asyncio
async def test_ops_version_001():
    """OPS-VERSION-001: Schema version 1.2.0 on all response models."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MockFixedPool()  # type: ignore

    ev = await server_module.evaluate_position("startpos", depth=10)
    assert ev.schema_version == "1.2.0"

    tm = await server_module.top_moves("startpos", n=2, depth=10)
    assert tm.schema_version == "1.2.0"
    assert tm.result[0].schema_version == "1.2.0"

    cm = await server_module.classify_move("startpos", "e4", depth=10)
    assert cm.schema_version == "1.2.0"

    ga = await server_module.analyze_game("1. e4 e5 *", depth=10)
    assert ga.schema_version == "1.2.0"
