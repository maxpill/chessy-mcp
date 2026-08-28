# pyright: reportPrivateUsage=false
"""Comprehensive regression tests for all 26 issues in Chess_MCP_Ultra_Detailed_Issue_Report.md."""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module
from mcp_server.models import MCPEval, MoveClass, score_played_move
from mcp_server.rules import format_fen_status_errors


class DummyPool:
    def __init__(self, eval_map: dict[str, Eval] | None = None) -> None:
        self.eval_map = eval_map or {}

    async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
        fen = board.fen()
        if fen in self.eval_map:
            return self.eval_map[fen]
        # Default mock
        legal = list(board.legal_moves)
        bm = legal[0].uci() if legal else None
        return Eval(cp=0, best_move=bm, pv=[bm] if bm else [], depth=depth)

    async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
        legal = list(board.legal_moves)
        res: list[Eval] = []
        for m in legal[:n]:
            res.append(Eval(cp=0, best_move=m.uci(), pv=[m.uci()], depth=depth))
        return res

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_act_001_unified_best_action():
    """ACT-001: Ensure top-level best_action matches eval_before.best_action and decision policy."""
    await server_module._cache.clear()
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]

    class RepetitionPool:
        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=60, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self) -> None:
            pass

    server_module._analyzer_pool = RepetitionPool()  # type: ignore

    cl = await server_module.classify_move(fen, "e4", moves=moves, depth=11)
    assert cl.eval_before.best_action == "play_move"
    assert cl.best_action == "play_move"
    assert cl.is_best_action is True
    assert cl.missed_draw_claim is False


@pytest.mark.asyncio
async def test_act_002_winning_position_claim_not_missed():
    """ACT-002: Winning positions (+5 Queen ahead) do not flag missed_draw_claim when playing on."""
    await server_module._cache.clear()
    fen = "7k/8/8/8/8/8/2Q5/K7 w - - 0 1"
    moves = ["Qc1", "Kg8", "Qc2", "Kh8", "Qc1", "Kg8", "Qc2", "Kh8"]

    class WinClaimPool:
        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=515, best_move="a1a2", pv=["a1a2"], depth=depth)

        async def close(self) -> None:
            pass

    server_module._analyzer_pool = WinClaimPool()  # type: ignore
    cl = await server_module.classify_move(fen, "Ka2", moves=moves, depth=11)
    assert cl.best_action == "play_move"
    assert cl.missed_draw_claim is False
    assert cl.move_class in (MoveClass.BEST, MoveClass.GOOD)


@pytest.mark.asyncio
async def test_act_003_distinguish_play_move_and_intended_claim():
    """ACT-003: Distinguish PlayMove from explicit claim actions."""
    await server_module._cache.clear()
    fen = "4k3/3q4/8/8/8/8/8/R3K3 w Q - 99 51"

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=-580, best_move="a1b1", pv=["a1b1"], depth=depth)

        async def close(self) -> None:
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # When action_type="claim_draw_with_intended_move"
    cl_claim = await server_module.classify_move(
        fen, "Rb1", depth=12, action_type="claim_draw_with_intended_move"
    )
    assert cl_claim.move_class == MoveClass.BEST
    assert cl_claim.effective_loss == 0
    assert cl_claim.is_best_action is True


@pytest.mark.asyncio
async def test_act_004_no_discontinuity_at_zero():
    """ACT-004: Ensure minimal cp fluctuations around zero (-5 vs +1) don't jump to 300cp blunder."""
    await server_module._cache.clear()
    fen = "6nk/8/8/8/8/8/8/1N2K3 w - - 0 1"
    moves = ["Nc3", "Nh6", "Nb1", "Ng8", "Nc3", "Nh6", "Nb1", "Ng8"]

    class ZeroPool:
        def __init__(self, cp_val: int):
            self.cp_val = cp_val

        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=self.cp_val, best_move="e1e2", pv=["e1e2"], depth=depth)

        async def close(self) -> None:
            pass

    # Test depth with -5 cp
    server_module._analyzer_pool = ZeroPool(-5)  # type: ignore
    cl_neg = await server_module.classify_move(fen, "Ke2", moves=moves, depth=1)
    assert cl_neg.move_class in (MoveClass.BEST, MoveClass.GOOD)
    assert cl_neg.effective_loss is not None and cl_neg.effective_loss <= 20
    assert cl_neg.missed_draw_claim is False

    await server_module._cache.clear()
    # Test depth with +1 cp
    server_module._analyzer_pool = ZeroPool(1)  # type: ignore
    cl_pos = await server_module.classify_move(fen, "Ke2", moves=moves, depth=14)
    assert cl_pos.move_class in (MoveClass.BEST, MoveClass.GOOD)
    assert cl_pos.effective_loss is not None and cl_pos.effective_loss <= 20
    assert cl_pos.missed_draw_claim is False



@pytest.mark.asyncio
async def test_cache_001_syntax_warning_isolation():
    """CACHE-001: Ensure syntax_warning is not contaminated or stuck in cache across queries."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore
    fen = chess.STARTING_FEN

    # 1. Non-canonical SAN generates warning
    cl1 = await server_module.classify_move(fen, "e4!!")
    assert cl1.syntax_warning is not None
    assert "normalized" in cl1.syntax_warning

    # 2. Canonical query hits cache and returns syntax_warning=None
    cl2 = await server_module.classify_move(fen, "e4")
    assert cl2.syntax_warning is None


@pytest.mark.asyncio
async def test_top_001_and_002_underpromotion_terminal_draw():
    """TOP-001 / TOP-002: Underpromotion leading to immediate terminal draw is normalized with post_terminal_status."""
    await server_module._cache.clear()
    fen = "7k/P7/8/8/8/8/8/7K w - - 0 1"

    class PromoPool:
        async def top_moves(self, board: chess.Board, n: int = 4, depth: int = 14):
            return [
                Eval(cp=None, mate=1, best_move="a7a8q", pv=["a7a8q"], depth=depth),
                Eval(cp=900, best_move="a7a8r", pv=["a7a8r"], depth=depth),
                Eval(cp=300, best_move="a7a8b", pv=["a7a8b"], depth=depth),
                Eval(cp=300, best_move="a7a8n", pv=["a7a8n"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = PromoPool()  # type: ignore
    res = await server_module.top_moves(fen, n=4, depth=14)
    assert len(res.result) == 4
    # Queen promo is mate
    assert res.result[0].best_move == "a7a8q"
    assert res.result[0].mate == 1
    # Rook promo is winning cp
    assert res.result[1].best_move == "a7a8r"
    assert res.result[1].cp is not None and res.result[1].cp > 0
    # Bishop & Knight promos are insufficient material (draw)
    assert res.result[2].best_move in ("a7a8b", "a7a8n")
    assert res.result[2].cp == 0
    assert res.result[2].mate is None
    assert res.result[2].post_terminal_status == "insufficient_material"
    assert res.result[3].cp == 0
    assert res.result[3].post_terminal_status == "insufficient_material"


@pytest.mark.asyncio
async def test_pgn_001_002_003_conversational_and_fence_stripping():
    """PGN-001, 002, 003: Conversational preambles, trailers without results, and markdown code blocks."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    # PGN with preamble and trailing natural language discussion without a result marker
    input_text = (
        "Here is a game I played yesterday against my friend:\n\n"
        "```pgn\n"
        '[Event "Casual Game"]\n'
        '[Site "Internet"]\n'
        '[Date "2026.08.27"]\n'
        '[White "Alice"]\n'
        '[Black "Bob"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6\n"
        "```\n\n"
        "I was thinking about playing d4 instead of Ba4 on move 4. What do you think?"
    )

    res = await server_module.analyze_game(input_text, depth=10)
    assert res.total_plies == 8
    assert res.white == "Alice"
    assert res.black == "Bob"


@pytest.mark.asyncio
async def test_pgn_004_scoped_figurine_normalization():
    """PGN-004: Figurine normalization applies only to movetext, preserving tags and comments."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        '[Event "Game ♘ vs ♔"]\n'
        '[White "Player {with ♞ in name}"]\n'
        '[Black "Opponent"]\n'
        '[Result "*"]\n\n'
        "1. e4 e5 2. ♘f3 ♞c6 {Comment mentioning ♗c4} *"
    )

    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 4
    assert "♘" in (res.event or "")
    assert "♞" in (res.white or "")


@pytest.mark.asyncio
async def test_pgn_005_escaped_quote_in_tag_value():
    """PGN-005: Escaped quotes in PGN headers are unescaped and preserved."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        '[Event "He said \\"Hello\\" Match"]\n'
        '[Site "New \\"York\\""]\n'
        '[White "Magnus \\"The Boss\\" Carlsen"]\n'
        '[Black "Hikaru Nakamura"]\n'
        '[Result "*"]\n\n'
        "1. e4 e5 *"
    )

    res = await server_module.analyze_game(pgn, depth=10)
    assert res.event == 'He said "Hello" Match'
    assert res.site == 'New "York"'
    assert res.white == 'Magnus "The Boss" Carlsen'


@pytest.mark.asyncio
async def test_pgn_006_duplicate_header_isolated_to_header_block():
    """PGN-006: Duplicate header warnings scan strictly within header section."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        '[Event "World Championship"]\n'
        '[Site "London"]\n'
        '[Result "*"]\n\n'
        '1. e4 e5 {In this game [Event "Fake"] occurred in notes} *'
    )

    res = await server_module.analyze_game(pgn, depth=10)
    # Ensure no false duplicate event warning
    dup_warns = [w for w in res.metadata_warnings if "Duplicate PGN tag" in w]
    assert len(dup_warns) == 0


@pytest.mark.asyncio
async def test_pgn_007_escape_percent_lines():
    """PGN-007: PGN % escape lines are stripped."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        "% This is an escape line\n"
        '[Event "Test"]\n'
        "% Another escape line\n"
        '[Result "*"]\n\n'
        "% Movetext escape line\n"
        "1. e4 e5 *"
    )

    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_pgn_008_multiline_tag_pair():
    """PGN-008: Multiline tag pairs are normalized."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        '[Event\n "Spanning Event"]\n'
        '[Site\n "Spanning Site"]\n'
        '[Result "*"]\n\n'
        "1. e4 e5 *"
    )

    res = await server_module.analyze_game(pgn, depth=10)
    assert res.event == "Spanning Event"
    assert res.site == "Spanning Site"
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_pgn_009_contradictory_termination():
    """PGN-009: Contradictory termination header vs board state emits metadata warning."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = (
        '[Event "Test"]\n'
        '[Termination "Time forfeit"]\n'
        '[Result "1-0"]\n\n'
        "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    )

    res = await server_module.analyze_game(pgn, depth=10)
    assert res.termination == "checkmate"
    assert any("disagrees with board outcome" in w for w in res.metadata_warnings)


@pytest.mark.asyncio
async def test_pgn_010_move_number_mismatch():
    """PGN-010: Move number mismatches in movetext emit syntax warnings."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = "1. e4 e5 5. Nf3 Nc6 *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert any("Move number mismatch" in w for w in res.syntax_warnings)


def test_fen_001_readable_fen_errors():
    """FEN-001: FEN validation errors format status bitmask into human-readable text."""
    err_str = format_fen_status_errors(chess.STATUS_TOO_MANY_KINGS | chess.STATUS_OPPOSITE_CHECK)
    assert "TOO_MANY_KINGS" in err_str
    assert "OPPOSITE_CHECK" in err_str


@pytest.mark.asyncio
async def test_metric_001_and_002_loss_decomposition():
    """METRIC-001 & METRIC-002: Loss breakdown fields are explicitly populated."""
    b_before = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    move = chess.Move.from_uci("e2e4")
    b_after = b_before.copy()
    b_after.push(move)

    ev_bef = MCPEval(status="active", cp=20, best_move="e2e4", pv=["e2e4"], depth=14)
    ev_aft = MCPEval(status="active", cp=20, best_move="e7e5", pv=["e7e5"], depth=14)

    score = score_played_move(b_before, move, ev_bef, ev_aft, b_after)
    assert score.loss_kind == "none"
    assert score.engine_cp_loss is None

    # Test blundering into mate
    ev_mate_after = MCPEval(status="active", cp=None, mate=-1, depth=14)
    score_mate = score_played_move(b_before, move, ev_bef, ev_mate_after, b_after)
    assert score_mate.loss_kind == "mate_transition"
    assert score_mate.outcome_penalty == 1000


@pytest.mark.asyncio
async def test_parser_001_strict_mode():
    """PARSER-001: strict=True rejects non-canonical SAN and metadata anomalies."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    # 1. Classify with strict=True on non-canonical SAN
    with pytest.raises(Exception) as exc_info:
        await server_module.classify_move(chess.STARTING_FEN, "e4!!", strict=True)
    assert "STRICT_VALIDATION_ERROR" in str(exc_info.value) or "strict_validation_error" in str(exc_info.value).lower()

    # 2. Analyze game with strict=True on move number mismatch
    with pytest.raises(Exception) as exc_info2:
        await server_module.analyze_game("1. e4 e5 9. Nf3 Nc6 *", strict=True)
    assert "STRICT_VALIDATION_ERROR" in str(exc_info2.value) or "strict_validation_error" in str(exc_info2.value).lower()


@pytest.mark.asyncio
async def test_act_005_classification_verification_invariants():
    """ACT-005: classification_verified is true and semantic invariants hold."""
    await server_module._cache.clear()
    class BestMovePool:
        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)
        async def close(self) -> None:
            pass
    server_module._analyzer_pool = BestMovePool()  # type: ignore
    cl = await server_module.classify_move(chess.STARTING_FEN, "e4")
    assert cl.classification_verified is True
    assert cl.best_action == cl.eval_before.best_action
    assert cl.is_best_action is True
    assert cl.is_engine_best is True


@pytest.mark.asyncio
async def test_api_001_is_engine_best_vs_is_best_action():
    """API-001: Distinguish is_engine_best from is_best_action."""
    await server_module._cache.clear()
    fen = "4k3/3q4/8/8/8/8/8/R3K3 w Q - 99 51"

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14, root_moves: list[chess.Move] | None = None) -> Eval:
            return Eval(cp=-580, best_move="a1b1", pv=["a1b1"], depth=depth)

        async def close(self) -> None:
            pass


    server_module._analyzer_pool = MockPool()  # type: ignore
    cl = await server_module.classify_move(
        fen, "Ra2", depth=12, action_type="claim_draw_with_intended_move"
    )
    # Ra2 is not the engine's best move (engine best is a1b1), but it is a valid claim action
    assert cl.is_engine_best is False
    assert cl.is_best_action is True
    assert cl.move_class == MoveClass.BEST


@pytest.mark.asyncio
async def test_api_002_structured_decision_and_engine_eval():
    """API-002: Verify structured decision_value and engine_eval contracts."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore
    ev = await server_module.evaluate_position(chess.STARTING_FEN, depth=14)
    assert ev.decision_value is not None
    assert "outcome" in ev.decision_value
    assert "best_action" in ev.decision_value
    assert ev.engine_eval is not None
    assert "best_move" in ev.engine_eval
    assert "depth" in ev.engine_eval


@pytest.mark.asyncio
async def test_api_003_pgn_action_continuation_vs_terminal():
    """API-003: Distinction between game continuation and terminal outcome in PGN."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore

    # Game with terminal 1/2-1/2 after repetition
    pgn_draw = (
        '[Event "Repetition"]\n'
        '[Result "1/2-1/2"]\n\n'
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 1/2-1/2"
    )
    res_draw = await server_module.analyze_game(pgn_draw, depth=10)
    assert res_draw.result == "1/2-1/2"
    assert res_draw.total_plies == 8

    # Game continued without claiming draw
    pgn_cont = (
        '[Event "Continued"]\n'
        '[Result "*"]\n\n'
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. e4 e5 *"
    )
    res_cont = await server_module.analyze_game(pgn_cont, depth=10)
    assert res_cont.total_plies == 10


@pytest.mark.asyncio
async def test_cache_001_alias_permutations():
    """CACHE-001: Permutation test of SAN aliases against multi-tier cache."""
    await server_module._cache.clear()
    server_module._analyzer_pool = DummyPool()  # type: ignore
    fen = chess.STARTING_FEN

    # 1. Canonical e4 -> no warning
    r1 = await server_module.classify_move(fen, "e4")
    assert r1.syntax_warning is None

    # 2. Alias e4! -> warning
    r2 = await server_module.classify_move(fen, "e4!")
    assert r2.syntax_warning == "Input SAN 'e4!' normalized to 'e4'"

    # 3. UCI e2e4 -> no warning
    r3 = await server_module.classify_move(fen, "e2e4")
    assert r3.syntax_warning is None

    # 4. Alias e2-e4 -> warning
    r4 = await server_module.classify_move(fen, "e2-e4")
    assert r4.syntax_warning == "Input SAN 'e2-e4' normalized to 'e4'"

    # 5. Canonical e4 again -> cache hit -> no warning
    r5 = await server_module.classify_move(fen, "e4")
    assert r5.syntax_warning is None


@pytest.mark.asyncio
async def test_top_001_stalemate_and_dead_position_terminals():
    """TOP-001: Candidate moves reaching stalemate or dead position are scored 0.0 with post_terminal_status."""
    await server_module._cache.clear()

    # Position where b2e5 gives stalemate and b2h8 gives checkmate
    fen = "k7/8/1K6/8/8/8/1Q6/8 w - - 0 1"

    class StalePool:
        async def top_moves(self, board: chess.Board, n: int = 2, depth: int = 14):
            return [
                Eval(cp=500, best_move="b2e5", pv=["b2e5"], depth=depth),
                Eval(cp=300, best_move="b2h8", pv=["b2h8"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = StalePool()  # type: ignore
    res = await server_module.top_moves(fen, n=2, depth=14)
    assert len(res.result) == 2
    # b2h8 is checkmate (ranked 1st with mate=1), b2e5 is stalemate draw (ranked 2nd with cp=0)
    assert res.result[0].best_move == "b2h8"
    assert res.result[0].mate == 1
    assert res.result[0].post_terminal_status == "checkmate"
    assert res.result[1].best_move == "b2e5"
    assert res.result[1].post_terminal_status == "stalemate"
    assert res.result[1].cp == 0




