"""Adversarial regression/property tests from the 2026-08-31 extreme MCP audit.

These tests deliberately target bugs reproduced against the deployed MCP before the
corresponding fixes were applied.  They also add deterministic randomized coverage for
move parsing, FEN round-trips, PGN round-trips, concurrency, state isolation, and
failure recovery.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import chess
import chess.pgn
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


class DeterministicPool:
    """Small deterministic engine double with observable call counts."""

    def __init__(self, *, cp: int = 23, delay: float = 0.0, fail_first: bool = False) -> None:
        self.cp = cp
        self.delay = delay
        self.fail_first = fail_first
        self.eval_calls = 0
        self.top_calls = 0
        self.engine_version = "ExtremeFake 20260831"
        self.name = self.engine_version

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        self.eval_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail_first and self.eval_calls == 1:
            raise OSError("synthetic engine crash")
        legal = list(board.legal_moves)
        if root_moves:
            legal = [m for m in root_moves if m in board.legal_moves]
        best = legal[0] if legal else None
        return Eval(
            cp=self.cp,
            mate=None,
            best_move=best.uci() if best else None,
            pv=[best.uci()] if best else [],
            depth=depth,
        )

    async def top_moves(
        self, board: chess.Board, *, n: int = 3, depth: int = 14
    ) -> list[Eval]:
        self.top_calls += 1
        out: list[Eval] = []
        for i, move in enumerate(list(board.legal_moves)[:n]):
            out.append(
                Eval(
                    cp=self.cp - i,
                    mate=None,
                    best_move=move.uci(),
                    pv=[move.uci()],
                    depth=depth,
                )
            )
        return out

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _clean_global_state() -> Any:
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()
    yield
    await server_module._cache.clear()
    await server_module.close_analyzer_pool()


@pytest.mark.asyncio
async def test_top_moves_compact_is_identical_on_fresh_and_cached_paths() -> None:
    """Regression: fresh compact leaked verbose fields while cached compact stripped them."""
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]

    first = await server_module.top_moves("startpos", n=3, depth=4, verbosity="compact")
    second = await server_module.top_moves("startpos", n=3, depth=4, verbosity="compact")

    assert pool.top_calls == 1, "second call must exercise the cached branch"
    assert first.model_dump() == second.model_dump()
    for candidate in first.result:
        assert candidate.lichess_url is None
        assert candidate.lichess_image is None
        assert candidate.decision_value is None
        assert candidate.engine_eval is None
        assert candidate.input_fen is None


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["evaluate_position", "top_moves"])
async def test_invalid_verbosity_is_a_clean_deterministic_tool_error(tool_name: str) -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    tool = getattr(server_module, tool_name)
    kwargs: dict[str, Any] = {"fen": "startpos", "depth": 2, "verbosity": "definitely-not-valid"}
    if tool_name == "top_moves":
        kwargs["n"] = 1
    with pytest.raises(Exception, match=r"\[INVALID_VERBOSITY\]"):
        await tool(**kwargs)


@pytest.mark.asyncio
async def test_evaluate_fen_canonicalization_is_about_input_not_replayed_suffix() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    start = chess.STARTING_FEN

    result = await server_module.evaluate_position(start, moves=["e4"], depth=2)
    expected = chess.Board()
    expected.push_san("e4")

    assert result.input_fen == start
    assert result.canonical_fen == expected.fen()
    assert result.fen_was_canonicalized is False


@pytest.mark.asyncio
async def test_evaluate_detects_real_noncanonical_en_passant_rewrite() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    noncanonical = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    result = await server_module.evaluate_position(noncanonical, depth=2)
    assert result.input_fen == noncanonical
    assert result.canonical_fen is not None
    assert result.canonical_fen.split()[3] == "-"
    assert result.fen_was_canonicalized is True


@pytest.mark.asyncio
async def test_top_moves_surfaces_true_fen_canonicalization_state() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    noncanonical = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    rewritten = await server_module.top_moves(noncanonical, n=1, depth=2)
    assert rewritten.canonical_fen is not None
    assert rewritten.canonical_fen.split()[3] == "-"
    assert rewritten.fen_was_canonicalized is True

    await server_module._cache.clear()
    canonical = await server_module.top_moves(
        chess.STARTING_FEN, moves=["e4"], n=1, depth=2
    )
    assert canonical.fen_was_canonicalized is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["evaluate_position", "top_moves", "classify_move"])
async def test_strict_primary_pgn_rejects_wrong_move_number_across_tools(tool_name: str) -> None:
    """Regression: strict was dropped when the primary `fen` argument was PGN/movetext."""
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    malformed = "1. e4 e5 3. Nf3 Nc6"

    kwargs: dict[str, Any] = {"fen": malformed, "depth": 2, "strict": True}
    if tool_name == "top_moves":
        kwargs["n"] = 1
    elif tool_name == "classify_move":
        kwargs["move"] = "Bb5"
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await getattr(server_module, tool_name)(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["evaluate_position", "analyze_game"])
async def test_strict_rejects_malformed_header_that_would_otherwise_be_discarded(
    tool_name: str,
) -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    malformed = '[Event "Broken]\n[White "A"]\n[Black "B"]\n\n1. e4 e5 *'
    kwargs: dict[str, Any]
    if tool_name == "evaluate_position":
        kwargs = {"fen": malformed, "depth": 2, "strict": True}
    else:
        kwargs = {"pgn": malformed, "depth": 2, "strict": True}
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await getattr(server_module, tool_name)(**kwargs)


@pytest.mark.asyncio
async def test_analyze_game_strict_rejects_out_of_range_nag() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game("1. e4 $999 e5 *", depth=2, strict=True)


@pytest.mark.asyncio
async def test_analyze_game_strict_zero_ply_cannot_bypass_metadata_validation() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    pgn = '[Event "Zero"]\n[Result "bogus"]\n\n*'
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game(pgn, depth=2, strict=True)


@pytest.mark.asyncio
async def test_terminal_evaluate_has_typed_game_over_action_like_top_moves() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    fen = "7k/8/8/8/8/8/8/K7 w - - 0 1"
    ev = await server_module.evaluate_position(fen, depth=2)
    top = await server_module.top_moves(fen, n=3, depth=2)

    assert ev.status == "insufficient_material"
    assert ev.best_action_obj is not None
    assert ev.best_action_obj["type"] == "game_over"
    assert ev.legal_actions == [ev.best_action_obj]
    assert top.best_action_obj == ev.best_action_obj
    assert top.legal_actions == ev.legal_actions


@pytest.mark.asyncio
async def test_top_moves_candidates_carry_build_identity() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    result = await server_module.top_moves("startpos", n=2, depth=2)
    assert result.build_sha is not None
    assert result.engine_config
    assert result.result
    for candidate in result.result:
        assert candidate.build_sha == result.build_sha
        assert candidate.engine_config == result.engine_config


@pytest.mark.asyncio
async def test_strict_accepts_canonical_annotated_pgn() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    pgn = (
        '[Event "Strict valid"]\n'
        '[White "Łukasz"]\n'
        '[Black "Zoë"]\n'
        '[Result "*"]\n\n'
        '1. e4! {comment [inside]} e5 $1 2. Nf3 Nc6 '
        '(2... Nf6 3. Nxe5 (3. d4 exd4)) 3. Bb5 a6!? *'
    )
    result = await server_module.analyze_game(pgn, depth=2, strict=True)
    assert result.total_plies == 6
    assert result.white == "Łukasz"
    assert result.black == "Zoë"


@pytest.mark.asyncio
async def test_strict_primary_pgn_accepts_correct_move_numbers_and_san() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    pgn = "1. e4 e5 2. Nf3 Nc6"
    result = await server_module.evaluate_position(pgn, depth=2, strict=True)
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        board.push_san(san)
    assert result.canonical_fen == board.fen()


@pytest.mark.asyncio
async def test_singleflight_coalesces_many_identical_concurrent_requests() -> None:
    pool = DeterministicPool(delay=0.02)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    results = await asyncio.gather(
        *[server_module.evaluate_position("startpos", depth=3, verbosity="compact") for _ in range(64)]
    )
    assert pool.eval_calls == 1
    assert len({(r.cp, r.best_move, r.canonical_fen) for r in results}) == 1


@pytest.mark.asyncio
async def test_concurrent_different_positions_do_not_leak_state() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    first_moves = [
        "a3", "a4", "b3", "b4", "c3", "c4", "d3", "d4", "e3", "e4",
        "f3", "f4", "g3", "g4", "h3", "h4", "Na3", "Nc3", "Nf3", "Nh3",
    ]
    results = await asyncio.gather(
        *[
            server_module.evaluate_position("startpos", moves=[san], depth=2, verbosity="compact")
            for san in first_moves
        ]
    )
    assert len({r.canonical_fen for r in results}) == len(first_moves)
    for san, result in zip(first_moves, results, strict=True):
        board = chess.Board()
        board.push_san(san)
        assert result.canonical_fen == board.fen()


@pytest.mark.asyncio
async def test_a_b_a_state_isolation() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    a1 = await server_module.evaluate_position("startpos", moves=["e4"], depth=2)
    _b = await server_module.evaluate_position("startpos", moves=["d4"], depth=2)
    a2 = await server_module.evaluate_position("startpos", moves=["e4"], depth=2)
    assert a1.model_dump() == a2.model_dump()


@pytest.mark.asyncio
async def test_engine_failure_does_not_poison_subsequent_request_state() -> None:
    pool = DeterministicPool(fail_first=True)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    with pytest.raises(Exception, match=r"\[ENGINE_ERROR\]"):
        await server_module.evaluate_position("startpos", depth=2)

    recovered = await server_module.evaluate_position("startpos", moves=["e4"], depth=2)
    assert recovered.status == "active"
    assert recovered.best_move is not None


def test_randomized_legal_move_san_and_fen_differential_5000_positions() -> None:
    """Deterministic fuzzing against python-chess legality/SAN/FEN semantics."""
    rng = random.Random(20260831)
    board = chess.Board()
    games = 1
    comparisons = 0

    while comparisons < 5000:
        if board.is_game_over(claim_draw=False) or not any(board.legal_moves):
            board = chess.Board()
            games += 1

        legal = list(board.legal_moves)
        move = legal[rng.randrange(len(legal))]
        san = board.san(move)

        parsed, warning = server_module._parse_move_on_board_with_warning(
            board.copy(stack=True), san, strict=True
        )
        assert warning is None
        assert parsed == move

        parsed_fen = server_module._build_board(board.fen())
        assert parsed_fen.fen() == board.fen()
        assert set(parsed_fen.legal_moves) == set(board.legal_moves)
        assert parsed_fen.is_check() == board.is_check()
        assert parsed_fen.is_checkmate() == board.is_checkmate()
        assert parsed_fen.is_stalemate() == board.is_stalemate()

        board.push(move)
        comparisons += 1

    print(f"EXTREME_RANDOM_POSITION_COMPARISONS={comparisons}")
    print(f"EXTREME_RANDOM_GAMES_FOR_5000_POSITIONS={games}")
    assert comparisons == 5000


def test_randomized_strict_pgn_roundtrip_200_games() -> None:
    rng = random.Random(8312026)
    total_plies = 0

    for game_idx in range(200):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Event"] = f"Fuzz {game_idx}"
        game.headers["Result"] = "*"
        node: chess.pgn.GameNode = game
        target_plies = 20 + rng.randrange(41)

        for _ in range(target_plies):
            if board.is_game_over(claim_draw=False):
                break
            legal = list(board.legal_moves)
            move = legal[rng.randrange(len(legal))]
            node = node.add_variation(move)
            board.push(move)
            total_plies += 1

        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        pgn = game.accept(exporter)
        parsed = server_module._extract_game(pgn, strict=True)
        parsed_board = parsed.end().board()
        assert parsed_board.fen() == board.fen()

    print("EXTREME_RANDOM_PGN_GAMES=200")
    print(f"EXTREME_RANDOM_PGN_PLIES={total_plies}")
    assert total_plies > 3000


@pytest.mark.parametrize(
    "fen",
    [
        "8/8/8/8/8/8/8/7 w - - 0 1",
        "8/8/8/8/8/8/8/8 w - - 0 1",
        "4k3/8/8/8/8/8/4K3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/P3K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w Z - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - e4 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - -1 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 0",
    ],
)
def test_invalid_fen_corpus_is_rejected_deterministically(fen: str) -> None:
    with pytest.raises(ValueError, match="INVALID_FEN"):
        server_module._build_board(fen)


@pytest.mark.asyncio
async def test_strict_rejects_uci_mainline_across_primary_pgn_and_analyze_game() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    movetext = "1. e2e4 e7e5 *"
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.evaluate_position(movetext, depth=2, strict=True)
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game(movetext, depth=2, strict=True)


@pytest.mark.asyncio
async def test_strict_rejects_zero_character_castling_notation() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. 0-0 Be7 *"
    with pytest.raises(Exception, match=r"\[STRICT_VALIDATION_ERROR\]"):
        await server_module.analyze_game(pgn, depth=2, strict=True)


@pytest.mark.asyncio
async def test_nonstrict_still_accepts_uci_and_zero_castling_compatibility_inputs() -> None:
    pool = DeterministicPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    uci = await server_module.evaluate_position("1. e2e4 e7e5 *", depth=2, strict=False)
    assert uci.status == "active"
    castling = await server_module.analyze_game(
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. 0-0 Be7 *", depth=2, strict=False
    )
    assert castling.total_plies == 8
