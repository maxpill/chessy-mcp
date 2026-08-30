from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import chess
import chess.pgn
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from core.engines.pool import AnalyzerPool
from core.engines.types import Eval
from mcp_server import cache as cache_module
from mcp_server import server as server_module
from mcp_server.actions import build_played_action
from mcp_server.models import MCPEval, MCPMoveAnalysis
from mcp_server.rules import RuleStatus, can_checkmate, evaluate_rule_status


@dataclass(frozen=True)
class GeneratedGame:
    idx: int
    sans: tuple[str, ...]
    ucis: tuple[str, ...]
    fen: str
    pgn: str


def _generated_games(count: int = 80) -> list[GeneratedGame]:
    rng = random.Random(0xC0FFEE)
    out: list[GeneratedGame] = []
    for idx in range(count):
        board = chess.Board()
        sans: list[str] = []
        ucis: list[str] = []
        game = chess.pgn.Game()
        node: chess.pgn.GameNode = game
        target_plies = 1 + rng.randrange(28)
        for _ in range(target_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = rng.choice(legal)
            sans.append(board.san(move))
            ucis.append(move.uci())
            node = node.add_variation(move)
            board.push(move)
            if board.is_game_over(claim_draw=False):
                break
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        pgn = game.accept(exporter).strip()
        out.append(GeneratedGame(idx, tuple(sans), tuple(ucis), board.fen(), pgn))
    return out


GENERATED_GAMES = _generated_games()


class DeterministicPool:
    name = "stress-v5"
    engine_version = "stress-v5"

    def __init__(self, cp: int = 0) -> None:
        self.cp = cp
        self.eval_calls = 0

    async def evaluate(
        self,
        board: chess.Board,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        self.eval_calls += 1
        legal = list(root_moves) if root_moves else list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(
            cp=self.cp,
            best_move=best,
            pv=[best] if best else [],
            depth=depth,
            wdl=(250, 500, 250),
        )

    async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
        legal = list(board.legal_moves)[:n]
        return [
            Eval(cp=self.cp + i, best_move=m.uci(), pv=[m.uci()], depth=depth)
            for i, m in enumerate(legal)
        ]

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _isolate_server_state():
    old_pool = server_module._analyzer_pool
    await server_module._cache.clear()
    server_module._analyzer_pool = DeterministicPool()
    yield
    await server_module._cache.clear()
    server_module._analyzer_pool = old_pool


# 80 deterministic legal-game replay cases. These exercise SAN parsing, UCI parsing,
# captures, checks, disambiguation, promotions when encountered, clocks and board identity.
@pytest.mark.parametrize("case", GENERATED_GAMES, ids=lambda c: f"legal-seq-{c.idx:02d}")
def test_generated_san_and_uci_replay_are_identical(case: GeneratedGame):
    from_san = server_module._build_board("startpos", list(case.sans))
    from_uci = server_module._build_board("startpos", list(case.ucis))
    assert from_san.fen() == case.fen
    assert from_uci.fen() == case.fen
    assert from_san.fen() == from_uci.fen()


# 40 independent PGN round trips from deterministic generated games.
@pytest.mark.parametrize("case", GENERATED_GAMES[:40], ids=lambda c: f"pgn-roundtrip-{c.idx:02d}")
def test_generated_pgn_roundtrip(case: GeneratedGame):
    rebuilt = server_module._build_board(case.pgn, [])
    assert rebuilt.fen() == case.fen
    assert [m.uci() for m in rebuilt.move_stack] == list(case.ucis)


INVALID_FENS = [
    "8/8/8/8/8/8/8/8 w - - 0 1",
    "8/8/8/8/8/8/8/K7 w - - 0 1",
    "k7/8/8/8/8/8/8/8 w - - 0 1",
    "k7/8/8/8/8/8/8/KK6 w - - 0 1",
    "kk6/8/8/8/8/8/8/K7 w - - 0 1",
    "4k3/8/8/8/8/8/PPPPPPPP/P3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w q - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - e3 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
    "4k3/8/8/8/8/8/8/4K3 w - - x 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 -1",
    "4k3/8/8/8/8/8/8/4K3 white - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w z - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - z9 0 1",
]


@pytest.mark.parametrize("fen", INVALID_FENS, ids=lambda s: f"bad-fen-{INVALID_FENS.index(s):02d}")
def test_invalid_fen_matrix_is_rejected(fen: str):
    with pytest.raises(ValueError):
        server_module._build_board(fen, [])


def test_five_field_epd_like_position_is_tolerated_and_canonicalized():
    board = server_module._build_board("4k3/8/8/8/8/8/8/4K3 w - - 0", [])
    assert board.fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


TERMINATION_CASES = [
    ("Normal", "normal"),
    ("Normal time control", "normal"),
    ("Normal time control - White", "normal"),
    ("50 moves rule", "fifty_moves"),
    ("fifty-move rule", "fifty_moves"),
    ("75 moves rule", "seventyfive_moves"),
    ("seventy-five moves rule", "seventyfive_moves"),
    ("fivefold repetition", "fivefold_repetition"),
    ("5-fold repetition", "fivefold_repetition"),
    ("threefold repetition", "threefold_repetition"),
    ("3-fold repetition claim", "threefold_repetition"),
    ("repetition", "repetition"),
    ("checkmate", "checkmate"),
    ("mate", "checkmate"),
    ("stalemate", "stalemate"),
    ("insufficient material", "insufficient_material"),
    ("White resigned", "resignation"),
    ("Black resignation", "resignation"),
    ("White lost on time", "time_forfeit"),
    ("Black lost on time", "time_forfeit"),
    ("White wins on time", "time_forfeit"),
    ("Black won on time", "time_forfeit"),
    ("White flag fell", "time_forfeit"),
    ("Black clock expired", "time_forfeit"),
    ("unfinished", "unterminated"),
    ("abandoned", "abandoned"),
    ("adjudication", "adjudication"),
    ("death", "death"),
    ("emergency", "emergency"),
    ("second illegal move", "rules_infraction"),
    ("rules infraction", "rules_infraction"),
    ("draw by agreement", "draw_agreement"),
]


@pytest.mark.parametrize("text,expected", TERMINATION_CASES, ids=[f"term-{i:02d}" for i in range(len(TERMINATION_CASES))])
def test_termination_normalization_matrix(text: str, expected: str):
    assert server_module.normalize_termination(text) == expected


RESULT_INFERENCE_CASES = [
    ("White wins on time", "1-0"),
    ("White won on time", "1-0"),
    ("Black wins on time", "0-1"),
    ("Black won on time", "0-1"),
    ("White lost on time", "0-1"),
    ("Black lost on time", "1-0"),
    ("White resigned", "0-1"),
    ("Black resigned", "1-0"),
    ("White wins by resignation", "1-0"),
    ("Black wins by resignation", "0-1"),
    ("White won by resignation", "1-0"),
    ("Black won by resignation", "0-1"),
    ("White flag fell", "0-1"),
    ("Black flag fell", "1-0"),
    ("Normal time control - White", None),
    ("Normal time control - Black", None),
    ("time forfeit", None),
    ("resignation", None),
]


@pytest.mark.parametrize("text,expected", RESULT_INFERENCE_CASES, ids=[f"result-{i:02d}" for i in range(len(RESULT_INFERENCE_CASES))])
def test_result_inference_matrix(text: str, expected: str | None):
    assert server_module._infer_result_from_termination(text) == expected


HISTORY_CASES = [
    ("startpos", None, "complete"),
    ("initial", None, "complete"),
    ("start", None, "complete"),
    ("startpos", [], "complete"),
    ("startpos", ["Nf3"], "complete"),
    ("1. Nf3 Nf6", None, "complete"),
    ("1. Nf3 Nf6", ["Ng1"], "complete"),
    (chess.STARTING_FEN, None, "incomplete"),
    (chess.STARTING_FEN, [], "incomplete"),
    (chess.STARTING_FEN, ["Nf3"], "partial"),
    ("7k/8/8/8/8/8/R7/K7 w - - 0 1", None, "incomplete"),
    ("7k/8/8/8/8/8/R7/K7 w - - 0 1", ["Ra3"], "partial"),
]


@pytest.mark.parametrize("base,moves,expected", HISTORY_CASES, ids=[f"history-{i:02d}" for i in range(len(HISTORY_CASES))])
def test_history_provenance_matrix(base: str, moves: list[str] | None, expected: str):
    assert server_module._history_provenance_for_input(base, moves) == expected


@dataclass(frozen=True)
class MoveParseCase:
    base: str
    prefix: tuple[str, ...]
    token: str
    uci: str


MOVE_PARSE_CASES = [
    MoveParseCase("startpos", (), "e4", "e2e4"),
    MoveParseCase("startpos", (), "e2e4", "e2e4"),
    MoveParseCase("startpos", (), "Nf3", "g1f3"),
    MoveParseCase("startpos", (), "g1f3", "g1f3"),
    MoveParseCase("startpos", (), "d4", "d2d4"),
    MoveParseCase("startpos", (), "c4", "c2c4"),
    MoveParseCase("startpos", (), "f4", "f2f4"),
    MoveParseCase("startpos", (), "1. e4", "e2e4"),
    MoveParseCase("startpos", ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"), "O-O", "e1g1"),
    MoveParseCase("startpos", ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"), "0-0", "e1g1"),
    MoveParseCase("startpos", ("e4", "a6", "e5", "d5"), "exd6 e.p.", "e5d6"),
    MoveParseCase("7k/P7/8/8/8/8/8/K7 w - - 0 1", (), "a8=Q+", "a7a8q"),
    MoveParseCase("7k/P7/8/8/8/8/8/K7 w - - 0 1", (), "a8=R+", "a7a8r"),
    MoveParseCase("7k/P7/8/8/8/8/8/K7 w - - 0 1", (), "a8=B", "a7a8b"),
    MoveParseCase("7k/P7/8/8/8/8/8/K7 w - - 0 1", (), "a8=N", "a7a8n"),
    MoveParseCase("startpos", (), "♘f3", "g1f3"),
]


@pytest.mark.parametrize("case", MOVE_PARSE_CASES, ids=[f"move-{i:02d}" for i in range(len(MOVE_PARSE_CASES))])
def test_move_parser_matrix(case: MoveParseCase):
    board = server_module._build_board(case.base, list(case.prefix))
    move, _ = server_module._parse_move_on_board_with_warning(board, case.token, strict=False)
    assert move.uci() == case.uci


STRICT_REJECT_CASES = [
    ("startpos", (), "e4!"),
    ("startpos", (), "e4+"),
    ("startpos", (), "1. e4"),
    ("startpos", ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"), "0-0"),
]


@pytest.mark.parametrize("base,prefix,token", STRICT_REJECT_CASES, ids=[f"strict-reject-{i:02d}" for i in range(len(STRICT_REJECT_CASES))])
def test_strict_parser_rejects_normalized_san(base: str, prefix: tuple[str, ...], token: str):
    board = server_module._build_board(base, list(prefix))
    with pytest.raises(ValueError, match="STRICT_SAN_ERROR"):
        server_module._parse_move_on_board_with_warning(board, token, strict=True)


@pytest.mark.parametrize("clock", [0, 1, 50, 98, 99, 100, 101, 149, 150])
def test_fifty_seventyfive_boundaries(clock: int):
    board = chess.Board(f"7k/8/8/8/8/8/R7/K7 w - - {clock} 51")
    status = evaluate_rule_status(board, history_complete="incomplete")
    if clock >= 150:
        assert status.terminal == "seventyfive_moves"
    elif clock >= 100:
        assert status.can_claim_now is True
        assert "fifty_moves" in status.claim_reasons_now
    elif clock == 99:
        assert status.can_claim_with_intended_move is True
        assert "fifty_moves" in status.claim_reasons
    else:
        assert "fifty_moves" not in status.claim_reasons_now


def test_mate_precedes_seventyfive_move_rule():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 150 76")
    assert board.is_checkmate()
    status = evaluate_rule_status(board, history_complete="complete")
    assert status.terminal == "checkmate"
    assert status.winner == "white"


MATERIAL_CASES = [
    ("7k/8/8/8/8/8/8/K7 w - - 0 1", chess.WHITE, False),
    ("7k/8/8/8/8/8/2B5/K7 w - - 0 1", chess.WHITE, False),
    ("7k/8/8/8/8/8/2N5/K7 w - - 0 1", chess.WHITE, False),
    ("7k/7p/8/8/8/8/2N5/K7 w - - 0 1", chess.WHITE, True),
    ("7k/7p/8/8/8/8/2B5/K7 w - - 0 1", chess.WHITE, True),
    ("7k/8/8/8/8/8/2NN4/K7 w - - 0 1", chess.WHITE, True),
    ("7k/8/8/8/8/8/2BN4/K7 w - - 0 1", chess.WHITE, True),
    ("7k/8/8/8/8/8/2R5/K7 w - - 0 1", chess.WHITE, True),
    ("7k/8/8/8/8/8/2Q5/K7 w - - 0 1", chess.WHITE, True),
    ("7k/8/8/8/8/8/2P5/K7 w - - 0 1", chess.WHITE, True),
]


@pytest.mark.parametrize("fen,color,expected", MATERIAL_CASES, ids=[f"mate-material-{i:02d}" for i in range(len(MATERIAL_CASES))])
def test_mating_possibility_matrix(fen: str, color: chess.Color, expected: bool):
    assert can_checkmate(chess.Board(fen), color) is expected


def test_repetition_status_complete_partial_incomplete_matrix():
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    board = server_module._build_board("startpos", moves)
    for provenance in ("complete", "partial"):
        status = evaluate_rule_status(board, history_complete=provenance)
        assert status.repetition_status == "threefold_claimable"
        assert "threefold_repetition" in status.claim_reasons_now
    naked = chess.Board(board.fen())
    status = evaluate_rule_status(naked, history_complete="incomplete")
    assert status.repetition_status == "unknown"
    assert "threefold_repetition" not in status.claim_reasons_now


@pytest.mark.parametrize("depth", [1, 2, 3, 7, 14, 30])
def test_cache_key_depth_isolation(depth: int):
    board = chess.Board()
    key = cache_module.eval_cache_key(board, depth, history_completeness="complete")
    assert f":{depth}" in key
    assert key != cache_module.eval_cache_key(board, depth + 1, history_completeness="complete")


@pytest.mark.parametrize("provenance", ["complete", "partial", "incomplete", "not_required"])
def test_cache_key_provenance_isolation(provenance: str):
    board = chess.Board()
    key = cache_module.eval_cache_key(board, 4, history_completeness=provenance)
    assert f"hist={provenance}" in key
    for other in {"complete", "partial", "incomplete", "not_required"} - {provenance}:
        assert key != cache_module.eval_cache_key(board, 4, history_completeness=other)


def test_cache_fingerprint_distinguishes_same_fen_different_history():
    a = chess.Board()
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"]:
        a.push_san(san)
    b = chess.Board(a.fen())
    assert a.fen() == b.fen()
    assert cache_module.history_fingerprint(a) != cache_module.history_fingerprint(b)
    assert cache_module.eval_cache_key(a, 4, history_completeness="complete") != cache_module.eval_cache_key(
        b, 4, history_completeness="incomplete"
    )


def test_typed_action_builder_rejects_unknown_type():
    status = RuleStatus()
    with pytest.raises(ValueError, match="INVALID_ACTION_TYPE"):
        build_played_action("banana", move_uci="e2e4", move_san="e4", rule_status=status)


def test_typed_action_builder_rejects_illegal_immediate_claim():
    status = RuleStatus(can_claim_now=False)
    with pytest.raises(ValueError, match="ILLEGAL_ACTION"):
        build_played_action("claim_draw", move_uci="e2e4", move_san="e4", rule_status=status)


def test_typed_action_builder_rejects_wrong_intended_move():
    status = RuleStatus(can_claim_with_intended_move=True, intended_claim_ucis=["a2a3"])
    with pytest.raises(ValueError, match="ILLEGAL_ACTION"):
        build_played_action(
            "claim_draw_with_intended_move",
            move_uci="a2a4",
            move_san="Ra4",
            rule_status=status,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("verbosity", ["full", "standard", "default", None])
async def test_full_verbosity_aliases_keep_semantics(verbosity: str | None):
    result = await server_module.evaluate_position("startpos", depth=1, verbosity=verbosity)
    assert result.history_completeness == "complete"
    assert result.repetition_status == "none"


@pytest.mark.asyncio
@pytest.mark.parametrize("verbosity", ["compact", "minimal", "min"])
async def test_compact_verbosity_aliases_keep_semantics(verbosity: str):
    full = await server_module.evaluate_position(chess.STARTING_FEN, depth=1, verbosity="full")
    compact = await server_module.evaluate_position(chess.STARTING_FEN, depth=1, verbosity=verbosity)
    assert compact.history_completeness == full.history_completeness == "incomplete"
    assert compact.repetition_status == full.repetition_status == "unknown"
    assert compact.status == full.status
    assert compact.cp == full.cp


@pytest.mark.asyncio
@pytest.mark.parametrize("verbosity", ["banana", "verbose", "tiny", "FULLER"])
async def test_unknown_verbosity_is_rejected(verbosity: str):
    with pytest.raises(ToolError, match="INVALID_VERBOSITY"):
        await server_module.evaluate_position("startpos", depth=1, verbosity=verbosity)


@pytest.mark.asyncio
@pytest.mark.parametrize("depth,expected", [(-100, 1), (0, 1), (1, 1), (14, 14), (30, 30), (31, 30), (999, 30)])
async def test_evaluate_depth_clamping(depth: int, expected: int):
    result = await server_module.evaluate_position("startpos", depth=depth)
    assert result.requested_depth == depth
    assert result.searched_depth == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("n,expected", [(-10, 1), (0, 1), (1, 1), (2, 2), (5, 5), (20, 20), (21, 20), (999, 20)])
async def test_top_moves_n_clamping(n: int, expected: int):
    result = await server_module.top_moves("startpos", n=n, depth=1)
    assert len(result.result) == expected


@pytest.mark.asyncio
async def test_terminal_top_moves_is_empty_and_game_over_action_typed():
    result = await server_module.top_moves("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", n=20, depth=30)
    assert result.status == "checkmate"
    assert result.winner == "white"
    assert result.result == []
    assert result.best_action_obj is not None
    assert result.best_action_obj["type"] == "game_over"


@pytest.mark.asyncio
async def test_unknown_action_type_rejected_at_public_boundary():
    with pytest.raises(ToolError, match="INVALID_ACTION_TYPE"):
        await server_module.classify_move("startpos", "e4", action_type="definitely-not-an-action")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_immediate_claim_and_intended_claim_remain_distinct():
    now = await server_module.classify_move(
        "7k/8/8/8/8/8/R7/K7 w - - 100 51",
        "Ra3",
        depth=1,
        action_type="claim_draw",
    )
    intended = await server_module.classify_move(
        "7k/8/8/8/8/8/R7/K7 w - - 99 51",
        "Ra3",
        depth=1,
        action_type="claim_draw_with_intended_move",
    )
    assert now.played_action_obj is not None
    assert intended.played_action_obj is not None
    assert now.played_action_obj["type"] == "claim_draw"
    assert intended.played_action_obj["type"] == "claim_draw_with_intended_move"


@pytest.mark.asyncio
async def test_real_stockfish_startpos_and_best_move_classification():
    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    if not os.path.isfile(path):
        pytest.skip("Stockfish not installed")
    old_pool = server_module._analyzer_pool
    pool = await AnalyzerPool.create(path, 1, depth=5, threads=1, hash_mb=16)
    server_module._analyzer_pool = pool
    await server_module._cache.clear()
    try:
        root = await server_module.evaluate_position("startpos", depth=5)
        assert root.best_move is not None
        board = chess.Board()
        best = chess.Move.from_uci(root.best_move)
        assert best in board.legal_moves
        graded = await server_module.classify_move("startpos", root.best_move, depth=5)
        assert graded.is_engine_best is True
        assert graded.is_best_engine_move is True
        assert graded.effective_loss == 0
        after = board.copy(stack=True)
        after.push(best)
        assert graded.eval_after.canonical_fen == after.fen()
    finally:
        await server_module._cache.clear()
        await pool.close()
        server_module._analyzer_pool = old_pool


@pytest.mark.asyncio
async def test_real_stockfish_mate_in_one_candidate_is_terminal_post_state():
    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    if not os.path.isfile(path):
        pytest.skip("Stockfish not installed")
    old_pool = server_module._analyzer_pool
    pool = await AnalyzerPool.create(path, 1, depth=5, threads=1, hash_mb=16)
    server_module._analyzer_pool = pool
    await server_module._cache.clear()
    try:
        result = await server_module.top_moves("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", n=3, depth=5)
        assert result.result
        assert any(c.post_terminal_status == "checkmate" and c.post_position and c.post_position["winner"] == "white" for c in result.result)
    finally:
        await server_module._cache.clear()
        await pool.close()
        server_module._analyzer_pool = old_pool


@pytest.mark.asyncio
async def test_real_stockfish_analyze_fools_mate_metadata():
    path = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
    if not os.path.isfile(path):
        pytest.skip("Stockfish not installed")
    old_pool = server_module._analyzer_pool
    pool = await AnalyzerPool.create(path, 1, depth=4, threads=1, hash_mb=16)
    server_module._analyzer_pool = pool
    await server_module._cache.clear()
    try:
        result = await server_module.analyze_game(
            '[Event "stress-v5"]\n[White "A"]\n[Black "B"]\n[Result "0-1"]\n\n1. f3 e5 2. g4 Qh4# 0-1',
            depth=4,
        )
        assert result.total_plies == 4
        assert result.result == "0-1"
        assert result.termination == "checkmate"
        assert result.white == "A"
        assert result.black == "B"
    finally:
        await server_module._cache.clear()
        await pool.close()
        server_module._analyzer_pool = old_pool
