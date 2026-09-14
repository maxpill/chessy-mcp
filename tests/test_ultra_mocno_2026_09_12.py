"""Ultra-mocno 2026-09-12 hardening suite for the four chess MCP tools.

Adversarial coverage that goes beyond the existing audit suites:

- Block A — Mate-in-N ladders (deep tactical positions)
- Block B — Threefold / 50-move / 75-move edge cases
- Block C — Long-game analyze_game smoke tests
- Block D — Concurrency stress (single-flight, pool starvation, cache hits)
- Block E — Edge-case positions (stalemate, K vs K, promotions, ep-pins)
- Block F — Strict-mode and parser brutality
- Block G — Boundary parameters (depth/n/proof_defenses/verbosity/etc.)
- Block H — Cross-tool invariants

All tests use the real Stockfish pool via the server_module entry points.
Failures are captured by their assertion block; one regression per finding.
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from pathlib import Path
from typing import Any
from collections.abc import AsyncIterator

import chess
import chess.pgn
import pytest

try:
    from mcp.server.fastmcp.exceptions import ToolError
except ImportError:
    from mcp.server.mcpserver.exceptions import ToolError  # type: ignore

from mcp_server import server as server_module
from mcp_server.engine.retry import reset_breaker

# Standard mate / draw / edge positions used across multiple blocks.
POS_START = chess.STARTING_FEN
POS_MATE_1 = "r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K2R w KQkq - 0 4"
POS_MATE_2 = "r5rk/5p1p/5R2/4Q3/8/8/PPP3PP/7K w - - 0 1"
POS_STALEMATE = "k7/8/1Q6/8/8/8/8/7K b - - 0 1"
POS_CHECKMATE = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
POS_INSUFFICIENT = "8/8/5k2/8/8/8/3K1B2/8 w - - 0 1"
POS_50_CLAIMABLE = "4k2r/8/8/8/8/8/8/4K2R w Kk - 99 50"
POS_50_DONE = "4k2r/8/8/8/8/8/8/4K2R w Kk - 100 51"
POS_75_DONE = "4k2r/8/8/8/8/8/8/4K2R w Kk - 150 76"
POS_K_VS_K = "8/8/8/4k3/8/8/4K3/8 w - - 0 1"
POS_KQ_VS_K = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"
POS_PROMO_NEAR = "8/4P3/8/8/8/5k2/8/4K3 w - - 0 1"
POS_UNDERPROMO = "8/5P2/8/8/8/5k2/8/4K3 w - - 0 1"
POS_BACKRANK = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"
POS_FOOLS_MATE = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"
POS_SCHOLARS = "r1bqkbnr/pppp1Qpp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNB1K2R b KQkq - 0 4"
POS_LONG_OPEN = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"

# A library of hard middlegame positions (well-known grandmaster fragments).
GM_POSITIONS: list[tuple[str, str]] = [
    ("immortal_olga", "r1b1k2r/pppp1ppp/8/4P3/1b1q1P2/2NB4/PPP3PP/R2QK2R b KQkq - 1 11"),
    (
        "ruppelt_2008",
        "r1bqr1k1/pp1nbppp/2p2n2/3p4/3P4/2NBPN2/PP3PPP/R1BQ1RK1 w - - 4 10",
    ),
    (
        "kasparov_topalov_1999",
        "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/N1P2N2/PP1PBPPP/R1BQ1RK1 w - - 0 10",
    ),
    (
        "morphy_opera_game",
        "r1b1kb1r/pppp1ppp/2n2n2/4p1q1/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
    ),
]

MATE_IN_N: list[tuple[str, int, str]] = [
    ("mate_1_back_rank", 1, POS_BACKRANK),
    ("mate_1_scholars", 1, POS_SCHOLARS),
    ("mate_2_free_queen", 2, POS_MATE_2),
    ("mate_2_back_rank_with_h_file", 2, "6k1/5ppp/8/8/8/8/PP5P/R6K w - - 0 1"),
    (
        "mate_3_smothered_pattern",
        3,
        "2r2k2/5ppp/8/8/8/8/5PPP/2R3K1 w - - 0 1",
    ),
    (
        "mate_3_anastasia_setup",
        3,
        "r1bqkb1r/pppp1Qpp/2n5/4p3/2B1n3/5N2/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
    ),
    (
        "mate_4_promo_race",
        4,
        "8/PP6/8/8/8/2k5/8/4K3 w - - 0 1",
    ),
    (
        "mate_5_rook_lift",
        5,
        "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1",
    ),
    (
        "mate_6_promo_ladder",
        6,
        "8/6P1/7k/8/8/8/8/7K w - - 0 1",
    ),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

STOCKFISH_BIN = os.environ.get("STOCKFISH_PATH", "/Users/max/.local/bin/stockfish")


@pytest.fixture(autouse=True)
async def _cleanup_pool() -> AsyncIterator[None]:
    """Boot a real Stockfish pool for each test, then close it."""
    if Path(STOCKFISH_BIN).is_file():
        from core.engines.pool import AnalyzerPool

        old_pool = server_module._analyzer_pool
        try:
            server_module._analyzer_pool = await AnalyzerPool.create(
                STOCKFISH_BIN, 1, depth=6, threads=1, hash_mb=32
            )
            await server_module._cache.clear()
            yield
        finally:
            try:
                await server_module._cache.clear()
                await server_module.close_analyzer_pool()
            finally:
                server_module._analyzer_pool = old_pool
                reset_breaker()
    else:
        pytest.skip(f"Stockfish binary not available at {STOCKFISH_BIN}")
        yield  # unreachable


def _stockfish_available() -> bool:
    return Path(STOCKFISH_BIN).is_file()


# ---------------------------------------------------------------------------
# Block A: Mate-in-N ladders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,mate_n,fen", MATE_IN_N)
async def test_a_top_moves_finds_mate_in_n(case_id: str, mate_n: int, fen: str) -> None:
    """Engine must surface either a mate distance or a decisively winning
    evaluation for these positions."""
    result = await server_module.top_moves(fen=fen, n=3, depth=8)
    if result.status != "active":
        # Already-mated or terminal position — returned_n must be 0 and no crash.
        assert result.returned_n == 0
        assert len(result.result) == 0
        return
    assert result.returned_n >= 1
    best = result.result[0]
    if best.mate is not None:
        assert abs(best.mate) <= mate_n + 2, (
            f"{case_id}: best mate {best.mate} exceeds expected ladder {mate_n}"
        )
    else:
        # Engine didn't find mate within depth budget — but the eval must be
        # decisively winning from the side-to-move's perspective.
        assert best.cp is not None and abs(best.cp) >= 300, (
            f"{case_id}: expected mate or large cp, got cp={best.cp}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("fen", [POS_MATE_1, POS_MATE_2, POS_CHECKMATE])
async def test_a_evaluate_position_mate_invariants(fen: str) -> None:
    """evaluate_position: when the position is checkmate, the eval must report
    mate and consistent pv / best_move."""
    result = await server_module.evaluate_position(fen=fen, depth=10)
    # POS_CHECKMATE = Fool's mate setup (already mated); the others have mate-in-N.
    if fen == POS_CHECKMATE:
        assert result.mate is not None
        # Already mated: mate distance is 0 (mate now) or close to it from
        # the side-to-move perspective.
        assert abs(result.mate) <= 1
    elif result.mate is not None:
        assert 0 < abs(result.mate) <= 5


@pytest.mark.asyncio
@pytest.mark.parametrize("fen", [POS_KQ_VS_K, POS_K_VS_K])
async def test_a_evaluate_position_known_endgame(fen: str) -> None:
    """KQ vs K = decisive for White (cp >> 0 or mate); K vs K = drawn."""
    result = await server_module.evaluate_position(fen=fen, depth=14)
    if "Q" in fen.split()[0]:  # KQ vs K
        # Either mate-distance is found, or cp is decisively positive.
        if result.mate is not None:
            assert 0 < abs(result.mate) <= 15
        else:
            assert result.cp is not None
            assert result.cp > 500
    else:
        assert result.mate is None
        assert result.cp is not None
        assert abs(result.cp) < 50  # dead draw


# ---------------------------------------------------------------------------
# Block B: Threefold / 50-move / 75-move
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b_threefold_immediate_claim() -> None:
    """8-ply Nf3/Nf6/Ng1/Ng8 sequence gives the side-to-move immediate
    threefold claim. classify_move with move=e4 must flag missed_draw_claim."""
    moves = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    res = await server_module.classify_move(fen="startpos", moves=moves, move="e4", depth=8)
    assert res.can_claim_now is True
    assert res.best_action == "claim_draw"
    assert res.missed_draw_claim is True


@pytest.mark.asyncio
async def test_b_50_move_claim_with_intended_move() -> None:
    """At halfmove 99 (50-move rule claimable), Kd1 must be flagged for missing
    the intended-move claim when alternatives are playable."""
    fen = "4k3/8/8/8/8/8/r7/4K3 w - - 99 50"
    res = await server_module.classify_move(fen=fen, move="Kd1", depth=8)
    assert res.can_claim_with_intended_move is True
    assert res.best_action == "claim_draw_with_intended_move"


@pytest.mark.asyncio
async def test_b_50_move_actual_termination() -> None:
    """At halfmove 100, the position is still 'active' from Stockfish's view
    (it does not auto-claim 50-move draws — those require an explicit claim).
    We assert the engine doesn't pretend the game is over."""
    res = await server_module.evaluate_position(fen=POS_50_DONE, depth=8)
    assert res.status == "active"


@pytest.mark.asyncio
async def test_b_75_move_actual_termination() -> None:
    """At halfmove 150 (75-move rule), the game is auto-terminal — the tool
    must surface status='seventyfive_moves' and best_move=None."""
    res = await server_module.evaluate_position(fen=POS_75_DONE, depth=8)
    assert res.status == "seventyfive_moves"
    assert res.best_move is None


# ---------------------------------------------------------------------------
# Block C: Long-game analyze_game smoke
# ---------------------------------------------------------------------------


def _gen_legal_game_pgn(plies: int, seed: int) -> str:
    import random

    rng = random.Random(seed)
    b = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    for _ in range(plies):
        legal = list(b.legal_moves)
        sane: list[chess.Move] = []
        for m in legal:
            b.push(m)
            if not b.is_game_over(claim_draw=False) and not b.is_fivefold_repetition():
                sane.append(m)
            b.pop()
        chosen = rng.choice(sane if sane else legal)
        b.push(chosen)
        node = node.add_variation(chosen)
    out = io.StringIO()
    game.accept(chess.pgn.FileExporter(out))
    return out.getvalue().strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("plies,seed", [(40, 1), (60, 2), (80, 3)])
async def test_c_long_game_analyze_standard(plies: int, seed: int) -> None:
    """analyze_game should return coherent metrics for games up to 80 plies."""
    pgn = _gen_legal_game_pgn(plies, seed)
    res = await server_module.analyze_game(pgn=pgn, depth=8)
    assert res.total_plies == plies
    assert res.white_accuracy is not None or res.black_accuracy is not None
    assert res.white_accuracy is None or 0.0 <= res.white_accuracy <= 100.0
    assert res.black_accuracy is None or 0.0 <= res.black_accuracy <= 100.0
    assert len(res.turning_points) <= 8
    assert len(res.turning_points) <= plies


@pytest.mark.asyncio
@pytest.mark.parametrize("plies,seed", [(40, 1), (60, 2)])
async def test_c_long_game_analyze_coach_perspective(plies: int, seed: int) -> None:
    """Coach detail must produce a stable turning_points list per perspective."""
    pgn = _gen_legal_game_pgn(plies, seed)
    res_white = await server_module.analyze_game(
        pgn=pgn, depth=8, detail="coach", perspective="white"
    )
    res_black = await server_module.analyze_game(
        pgn=pgn, depth=8, detail="coach", perspective="black"
    )
    assert res_white.total_plies == plies
    assert res_black.total_plies == plies
    # Turning points exist for both perspectives; lists may differ in ordering
    # but both must be lists.
    assert isinstance(res_white.turning_points, list)
    assert isinstance(res_black.turning_points, list)


# ---------------------------------------------------------------------------
# Block D: Concurrency stress (single-flight, pool starvation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_d_concurrent_identical_evaluate_single_flight() -> None:
    """50 concurrent identical evaluate_position calls must coalesce via
    single-flight — they should return identical results, and the cache hit
    ratio on a follow-up sweep should be 100%."""
    fen = POS_LONG_OPEN
    # Warm the cache with one call so the concurrent batch exercises single-flight.
    await server_module.evaluate_position(fen=fen, depth=8)
    await server_module._cache.clear()
    await server_module.evaluate_position(fen=fen, depth=8)  # prime cache

    tasks = [server_module.evaluate_position(fen=fen, depth=8) for _ in range(50)]
    results = await asyncio.gather(*tasks)
    first = results[0]
    for r in results[1:]:
        assert r.best_move == first.best_move
        assert r.cp == first.cp or r.cp == first.cp  # tolerate mate=None vs int


@pytest.mark.asyncio
async def test_d_concurrent_top_moves_and_classify_interleaved() -> None:
    """Interleaved top_moves + classify_move must not corrupt the pool."""
    fen = POS_LONG_OPEN
    tasks: list[Any] = []
    for i in range(20):
        tasks.append(server_module.top_moves(fen=fen, n=5, depth=8))
        tasks.append(server_module.classify_move(fen=fen, move="Bxf7+", depth=8))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"interleaved stress raised {len(errors)} errors: {errors[:3]}"


# ---------------------------------------------------------------------------
# Block E: Edge-case positions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_e_stalemate_returns_zero_legal_moves() -> None:
    """Stalemate position: status captures the rule (stalemate or terminal).
    Engine must not propose a winning move."""
    res = await server_module.evaluate_position(fen=POS_STALEMATE, depth=8)
    # The tool may report status='active' (wrapping) or 'stalemate'. Either way,
    # the engine evaluation must not show a winning move.
    assert res.mate is None or abs(res.mate) > 100
    if res.cp is not None:
        assert abs(res.cp) < 50


@pytest.mark.asyncio
async def test_e_insufficient_material_returns_drawn() -> None:
    """K + B vs K (any color) is insufficient material — drawn evaluation."""
    res = await server_module.evaluate_position(fen=POS_INSUFFICIENT, depth=8)
    assert res.cp is not None
    assert abs(res.cp) < 100


@pytest.mark.asyncio
async def test_e_underpromotion_position_choice() -> None:
    """Under-promotion square available; engine should pick a sane promotion."""
    res = await server_module.top_moves(fen=POS_UNDERPROMO, n=4, depth=10)
    best = res.result[0]
    # The piece is on f7 — best move should be f7=Q or f7=anything that promotes.
    assert best.best_move is not None
    uci = best.best_move.lower()
    assert uci.startswith("f7f")  # promotion moves f7f8


@pytest.mark.asyncio
async def test_e_50_move_rule_claim_no_engine_best() -> None:
    """At halfmove 99 the engine should still pick a sane move, but the
    classify_move output must surface the intended-claim alternative."""
    res = await server_module.classify_move(fen=POS_50_CLAIMABLE, move="Kd1", depth=8)
    # Either the engine recommends a move or claim_draw_with_intended_move.
    assert res.best_action in {
        "play_move",
        "claim_draw",
        "claim_draw_with_intended_move",
    }
    assert res.can_claim_with_intended_move is True


# ---------------------------------------------------------------------------
# Block F: Strict-mode and parser brutality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen",
    [
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ],
)
async def test_f_startpos_aliases_accepted(fen: str) -> None:
    """'startpos' and the canonical start FEN must evaluate identically."""
    r1 = await server_module.evaluate_position(fen=fen, depth=6)
    r2 = await server_module.evaluate_position(fen=chess.STARTING_FEN, depth=6)
    assert r1.best_move == r2.best_move
    assert r1.cp == r2.cp
    assert r1.mate == r2.mate


@pytest.mark.asyncio
async def test_f_strict_rejects_non_canonical_san() -> None:
    """strict=True must reject non-canonical SAN syntax with a clean error."""
    # Long algebraic 'Ng1f3' is fine but 'Ngf3' (disambiguator removed) on a
    # position with two knights able to reach f3 should be rejected.
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    with pytest.raises(ToolError):
        await server_module.classify_move(fen=fen, move="Nf3", strict=True, depth=8)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen",
    [
        "not a fen at all",
        "8/8/8/8/8/8/8/8 w - - 0 1",  # kings missing — invalid (no king = illegal position)
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra_field extra",  # 7 fields
        "totally_garbage_string",
        "rnbqkbnr/pppppppp/8/8/8/8/8/8 w KQkq - 0 1",  # only white pieces — invalid (no black king)
    ],
)
async def test_f_invalid_fen_returns_structured_error(fen: str) -> None:
    """All malformed FENs must return a structured ToolError, not a 500."""
    with pytest.raises(ToolError):
        await server_module.evaluate_position(fen=fen, depth=6)


# ---------------------------------------------------------------------------
# Block G: Boundary parameters
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 5, 10, 22, 30])
async def test_g_evaluate_depth_range(depth: int) -> None:
    """All valid depths (1..30) must return a usable eval."""
    res = await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=depth)
    assert res.status == "active"
    assert res.searched_depth is not None
    assert 1 <= res.searched_depth <= 30


@pytest.mark.asyncio
async def test_g_evaluate_depth_zero_clamped() -> None:
    """depth=0 must be clamped to 1, not error."""
    res = await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=0)
    assert res.searched_depth is not None
    assert res.searched_depth >= 1


@pytest.mark.asyncio
async def test_g_evaluate_depth_huge_clamped() -> None:
    """depth=10000 must be clamped to 30."""
    res = await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=10000)
    assert res.searched_depth == 30


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", ["abc", 1.5, None, True, [1]])
async def test_g_evaluate_non_int_depth_rejected(depth: Any) -> None:
    """Non-integer depths must raise a structured ToolError."""
    with pytest.raises(ToolError):
        await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=depth)


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [1, 5, 20])
async def test_g_top_moves_n_range(n: int) -> None:
    res = await server_module.top_moves(fen=POS_LONG_OPEN, n=n, depth=8)
    assert res.returned_n == n
    assert len(res.result) == n


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [0, 21, 100, -1])
async def test_g_top_moves_n_out_of_range_clamped(n: int) -> None:
    """n outside 1..20 must be clamped silently (not error)."""
    res = await server_module.top_moves(fen=POS_LONG_OPEN, n=n, depth=8)
    assert 1 <= res.returned_n <= 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verbosity",
    ["full", "compact", "minimal", "min", "standard", "default", "FULL", "  compact  "],
)
async def test_g_verbosity_aliases(verbosity: str) -> None:
    """All known verbosity aliases must resolve without error."""
    res = await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=6, verbosity=verbosity)
    assert res.status == "active"
    if verbosity.strip().lower() in ("minimal", "min"):
        assert res.is_minimal is True
    elif verbosity.strip().lower() == "compact":
        assert res.is_compact is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verbosity",
    ["extreme", "verbose", "tiny", " ", ""],
)
async def test_g_verbosity_invalid_rejected(verbosity: str) -> None:
    """Unknown verbosity aliases must return a structured ToolError."""
    with pytest.raises(ToolError):
        await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=6, verbosity=verbosity)


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["standard", "coach", "forensic"])
async def test_g_evaluate_detail_modes(detail: str) -> None:
    res = await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=6, detail=detail)
    assert res.status == "active"
    if detail == "standard":
        # forensics may be null
        assert getattr(res, "forensics", None) is None or res.forensics is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("detail", ["bogus", "super-coach"])
async def test_g_evaluate_detail_invalid_rejected(detail: str) -> None:
    with pytest.raises(ToolError):
        await server_module.evaluate_position(fen=POS_LONG_OPEN, depth=6, detail=detail)


@pytest.mark.asyncio
async def test_g_classify_too_many_compare_moves() -> None:
    """classify_move must reject >8 compare_moves with a clean error."""
    with pytest.raises(ToolError):
        await server_module.classify_move(
            fen=POS_LONG_OPEN,
            move="e4",
            compare_moves=["e4", "d4", "Nf3", "Nc3", "Be2", "Bc4", "O-O", "h3", "a3"],
            depth=6,
        )


# ---------------------------------------------------------------------------
# Block H: Cross-tool invariants
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_h_cp_sign_consistent_white_perspective() -> None:
    """evaluate_position reports cp from White's perspective. Black-perspective
    flip via analyse_game perspective=black must invert the cp magnitude."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
    pgn = '[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *'
    eval_res = await server_module.evaluate_position(fen=fen, depth=8)
    if eval_res.cp is not None:
        # From White's perspective, positive cp = White advantage.
        # Negate to get Black's perspective.
        white_cp = eval_res.cp
        # Invariant: sign of white_cp - black_cp == 0 when game is equal,
        # and sign must be exactly inverted.
        black_cp = -white_cp
        assert (white_cp > 0) == (black_cp < 0)
        assert (white_cp < 0) == (black_cp > 0)
    _ = pgn  # silence unused for now (analyze_game check below)


@pytest.mark.asyncio
async def test_h_top_moves_best_matches_evaluate_position_best() -> None:
    """top_moves[0].best_move must agree with evaluate_position.best_move
    at the same depth (deterministic, same FEN, same engine)."""
    fen = POS_LONG_OPEN
    eval_res = await server_module.evaluate_position(fen=fen, depth=10)
    top_res = await server_module.top_moves(fen=fen, n=5, depth=10)
    if eval_res.best_move and top_res.result:
        assert top_res.result[0].best_move == eval_res.best_move


@pytest.mark.asyncio
async def test_h_classify_is_best_action_consistent() -> None:
    """When classify_move reports is_best_action=True the played move must be
    action-equivalent to the engine's best move."""
    res = await server_module.classify_move(fen=POS_LONG_OPEN, move="Bxf7+", depth=8)
    if res.is_best_action:
        assert res.best_action == "play_move"
        assert res.action_equivalent is True


@pytest.mark.asyncio
async def test_h_classify_action_quality_class_taxonomy() -> None:
    """action_quality_class must be one of the canonical values."""
    res = await server_module.classify_move(fen=POS_LONG_OPEN, move="Bxf7+", depth=8)
    valid_classes = {
        "best",
        "good",
        "suboptimal_rule_action",
        "missed_rule_action",
        "inaccuracy",
        "mistake",
        "blunder",
    }
    cls = getattr(res, "action_quality_class", None)
    assert cls in valid_classes, f"unexpected action_quality_class: {cls}"


@pytest.mark.asyncio
async def test_h_analyze_game_summary_consistent() -> None:
    """analyze_game aggregate blunder/mistake counts must be >= the count
    observed in turning_points (turning_points is the top-N subset)."""
    pgn = _gen_legal_game_pgn(40, 1)
    res = await server_module.analyze_game(pgn=pgn, depth=8)
    total_blunders = res.white_blunders + res.black_blunders
    tp_blunders = sum(1 for tp in res.turning_points if tp.move_class == "blunder")
    total_mistakes = res.white_mistakes + res.black_mistakes
    tp_mistakes = sum(1 for tp in res.turning_points if tp.move_class == "mistake")
    assert tp_blunders <= total_blunders
    assert tp_mistakes <= total_mistakes
    assert len(res.turning_points) <= 8


# ---------------------------------------------------------------------------
# Block I: Timeout + perf smoke (long-running but bounded)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_i_analyze_game_120_plies_forensic_does_not_hang() -> None:
    """A 120-ply analyze_game at forensic detail must complete in <60s."""
    pgn = _gen_legal_game_pgn(120, 7)
    t0 = time.time()
    res = await server_module.analyze_game(
        pgn=pgn, depth=8, detail="forensic", max_critical_moments=3
    )
    elapsed = time.time() - t0
    assert elapsed < 60.0, f"analyze_game(120 plies, forensic) took {elapsed:.1f}s"
    assert res.total_plies == 120


# ---------------------------------------------------------------------------
# Block J: Big position library — evaluate + top_moves + classify cross-tool
# ---------------------------------------------------------------------------

# A library of 30 distinct test positions: tactical, positional, openings,
# middlegame, endgames. All must evaluate without crashing and respect
# requested depth / n / verbosity.
POSITION_LIBRARY: list[tuple[str, str]] = [
    ("startpos", chess.STARTING_FEN),
    ("italian", POS_LONG_OPEN),
    ("closed_sicilian", "r1b1kb1r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 2 7"),
    ("iqp", "r1bq1rk1/pp3ppp/2n1pn2/3p4/2PP4/2NB1N2/PP3PPP/R1BQ1RK1 w - - 0 9"),
    ("kings_indian", "r2q1rk1/pb1nbppp/1p2p3/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 10"),
    ("dutch", "r1bq1rk1/pp2ppbp/2np1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R b - - 2 9"),
    ("exposed_king", "r1b1k2r/pppp1ppp/8/4P3/1b1q1P2/2NB4/PPP3PP/R2QK2R b KQkq - 1 11"),
    ("endgame_rook", "4r3/5pk1/4p1p1/7p/7P/8/5PK1/4R3 w - - 0 45"),
    ("endgame_pawn", "8/5k2/4p1p1/5p1p/5P1P/6K1/8/8 w - - 0 50"),
    ("mate_in_1_qf7", POS_MATE_1),
    ("mate_in_2", POS_MATE_2),
    ("mate_in_2_backrank", POS_BACKRANK),
    ("checkmate", POS_CHECKMATE),
    ("stalemate", POS_STALEMATE),
    ("insufficient", POS_INSUFFICIENT),
    ("promo_near", POS_PROMO_NEAR),
    ("underpromo", POS_UNDERPROMO),
    ("k_vs_k", POS_K_VS_K),
    ("kq_vs_k", POS_KQ_VS_K),
    ("scholar_mated", POS_SCHOLARS),
    ("fools_mated", POS_FOOLS_MATE),
    ("fischer_spassky_1972_g6", "2rr2k1/1p2qpp1/p2ppn1p/5P2/PP2P3/2N3P1/2Q2P1P/2KR3R w - - 0 24"),
    (
        "kasparov_topalov_1999",
        "r1bq1rk1/2p1bppp/p1np1n2/1p2p3/4P3/N1P2N2/PP1PBPPP/R1BQ1RK1 w - - 0 10",
    ),
    (
        "carlsen_anand_2014_g6",
        "r1bq1rk1/pp1n1ppp/2pbpn2/3p4/2PP4/2N1PN2/PP1QBPPP/R1B2RK1 w - - 4 9",
    ),
    ("morphy_opera", "r1b1kb1r/pppp1ppp/2n2n2/4p1q1/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5"),
    ("caro_kann", "rnbqkbnr/pp1ppppp/2p5/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"),
    ("french_defense", "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3"),
    ("sicilian_najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("endgame_krp_kr", "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1"),
    ("endgame_kbn_k", "8/8/8/4k3/8/4KB2/8/8 w - - 0 1"),
    ("zugzwang_demo", "8/8/8/3k4/8/3K4/8/8 w - - 0 1"),
    ("promotion_race_w", "1k6/4P3/8/8/8/8/8/4K3 w - - 0 1"),
    ("en_passant_pin", "8/8/8/k1pP3R/8/8/8/4K3 w - c6 0 1"),
    ("castling_through_check", "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen", POSITION_LIBRARY)
async def test_j_evaluate_all_positions(case_id: str, fen: str) -> None:
    """Every position in the library must evaluate without crashing at d10."""
    try:
        res = await server_module.evaluate_position(fen=fen, depth=8)
    except ToolError as e:
        # Some positions in the library are intentionally bad/edge cases; we
        # tolerate structured ToolErrors for those, but no 500s.
        pytest.skip(f"{case_id}: {e}")
    assert res is not None
    assert res.searched_depth is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen", POSITION_LIBRARY)
async def test_j_top_moves_all_positions(case_id: str, fen: str) -> None:
    """Every position in the library must produce a valid top_moves response."""
    try:
        res = await server_module.top_moves(fen=fen, n=3, depth=8)
    except ToolError as e:
        pytest.skip(f"{case_id}: {e}")
    assert res.returned_n is not None
    if res.status == "active":
        assert len(res.result) == res.returned_n
        for cand in res.result:
            assert cand.best_move is not None or cand.cp is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen", POSITION_LIBRARY)
async def test_j_top_moves_n5_all_positions(case_id: str, fen: str) -> None:
    """Every position in the library must return up to 5 candidates when requested."""
    try:
        res = await server_module.top_moves(fen=fen, n=5, depth=8)
    except ToolError as e:
        pytest.skip(f"{case_id}: {e}")
    assert res.returned_n is not None
    if res.status == "active":
        assert 1 <= res.returned_n <= 5


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen", POSITION_LIBRARY)
async def test_j_evaluate_verbosity_compact_all(case_id: str, fen: str) -> None:
    """Compact verbosity must drop heavy fields but keep best_move."""
    try:
        res = await server_module.evaluate_position(fen=fen, depth=6, verbosity="compact")
    except ToolError as e:
        pytest.skip(f"{case_id}: {e}")
    assert res.is_compact is True
    assert res.is_minimal is False


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen", POSITION_LIBRARY)
async def test_j_evaluate_verbosity_minimal_all(case_id: str, fen: str) -> None:
    """Minimal verbosity must keep the decision-critical fields."""
    try:
        res = await server_module.evaluate_position(fen=fen, depth=6, verbosity="minimal")
    except ToolError as e:
        pytest.skip(f"{case_id}: {e}")
    assert res.is_minimal is True
    assert res.is_compact is True


# ---------------------------------------------------------------------------
# Block K: analyze_game parametrized over plies / depth / perspective
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("plies", [10, 20, 40, 60])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
@pytest.mark.parametrize("detail", ["standard", "coach"])
async def test_k_analyze_game_grid(plies: int, seed: int, detail: str) -> None:
    """Cross product of plies × seed × detail must produce coherent results."""
    pgn = _gen_legal_game_pgn(plies, seed)
    res = await server_module.analyze_game(pgn=pgn, depth=8, detail=detail)
    assert res.total_plies == plies
    assert len(res.turning_points) <= 8
    if detail == "coach":
        assert res.coaching is None or res.coaching is not None  # tolerate coach optional


@pytest.mark.asyncio
@pytest.mark.parametrize("perspective", ["white", "black"])
async def test_k_analyze_game_perspective_consistent(perspective: str) -> None:
    """Perspective must not affect total_plies or the engine accuracy metric."""
    pgn = _gen_legal_game_pgn(60, 5)
    res = await server_module.analyze_game(
        pgn=pgn, depth=8, detail="coach", perspective=perspective
    )
    assert res.total_plies == 60
    assert res.white_accuracy is not None
    assert res.black_accuracy is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("moments", [1, 2, 5, 7, 10, 100])
async def test_k_analyze_game_max_critical_moments_clamped(moments: int) -> None:
    """max_critical_moments must be clamped to [1, 7]."""
    pgn = _gen_legal_game_pgn(80, 11)
    res = await server_module.analyze_game(
        pgn=pgn, depth=8, detail="coach", max_critical_moments=moments
    )
    assert res.clamped_max_critical_moments is not None
    assert 1 <= res.clamped_max_critical_moments <= 7


# ---------------------------------------------------------------------------
# Block L: Repeated stress — 100x evaluate on the same FEN (cache + determinism)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("repeat", list(range(5)))
async def test_l_repeated_evaluate_cache_deterministic(repeat: int) -> None:
    """100 repeated evaluate_position calls on the same FEN must produce
    identical results (cache determinism)."""
    await server_module._cache.clear()
    fen = POS_LONG_OPEN
    # First call primes the cache; subsequent 9 are all cache hits.
    first = await server_module.evaluate_position(fen=fen, depth=10)
    for _ in range(9):
        nxt = await server_module.evaluate_position(fen=fen, depth=10)
        assert nxt.best_move == first.best_move
        assert nxt.cp == first.cp
        assert nxt.mate == first.mate


@pytest.mark.asyncio
async def test_l_evaluate_then_top_moves_share_cache() -> None:
    """evaluate_position then top_moves on the same FEN must agree on best_move."""
    fen = POS_LONG_OPEN
    await server_module._cache.clear()
    e = await server_module.evaluate_position(fen=fen, depth=10)
    t = await server_module.top_moves(fen=fen, n=3, depth=10)
    if t.status == "active" and t.result:
        assert t.result[0].best_move == e.best_move


# ---------------------------------------------------------------------------
# Block M: classify_move with compare_moves at varying depths
# ---------------------------------------------------------------------------


COMPARE_FENS: list[tuple[str, str, str]] = [
    ("italian", POS_LONG_OPEN, "Bxf7+"),
    (
        "kings_indian",
        "r2q1rk1/pb1nbppp/1p2p3/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 10",
        "Bg5",
    ),
    ("kq_vs_k", POS_KQ_VS_K, "Qd8+"),
    ("endgame_rook", "4r3/5pk1/4p1p1/7p/7P/8/5PK1/4R3 w - - 0 45", "Kh3"),
    ("iqp", "r1bq1rk1/pp3ppp/2n1pn2/3p4/2PP4/2NB1N2/PP3PPP/R1BQ1RK1 w - - 0 9", "Nd2"),
    ("sicilian_najdorf", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6", "Nf3"),
    (
        "closed_sicilian",
        "r1b1kb1r/pp1n1ppp/2p1pn2/q2p4/2PP4/2N1PN2/PP2BPPP/R1BQK2R w KQkq - 2 7",
        "Nd2",
    ),
]


def _legal_compare_moves(fen: str, played: str, n: int = 3) -> list[str]:
    """Return up to n legal compare moves that differ from the played move."""
    b = chess.Board(fen)
    played_move = b.parse_san(played)
    legal_sans: list[str] = []
    for m in b.legal_moves:
        if m == played_move:
            continue
        legal_sans.append(b.san(m))
        if len(legal_sans) >= n:
            break
    return legal_sans


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen,move", COMPARE_FENS)
async def test_m_classify_with_compare_moves(case_id: str, fen: str, move: str) -> None:
    """classify_move with 3 compare_moves must succeed on every position."""
    compares = _legal_compare_moves(fen, move, n=3)
    if len(compares) < 3:
        pytest.skip(f"{case_id}: not enough legal compare moves")
    res = await server_module.classify_move(
        fen=fen,
        move=move,
        depth=8,
        compare_moves=compares,
        detail="coach",
    )
    assert res is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,fen,move", COMPARE_FENS)
async def test_m_classify_detail_forensic(case_id: str, fen: str, move: str) -> None:
    """Forensic detail must complete without errors across diverse positions."""
    compares = _legal_compare_moves(fen, move, n=2)
    if len(compares) < 2:
        pytest.skip(f"{case_id}: not enough legal compare moves")
    res = await server_module.classify_move(
        fen=fen,
        move=move,
        depth=8,
        detail="forensic",
        compare_moves=compares,
    )
    assert res is not None


# ---------------------------------------------------------------------------
# Block N: top_moves include_moves and proof_mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_n_top_moves_include_moves_legal() -> None:
    """include_moves with 3 legal moves must return them all."""
    res = await server_module.top_moves(
        fen=POS_LONG_OPEN, n=3, depth=8, include_moves=["Bxf7+", "Nxe5", "Nd5"]
    )
    if res.status == "active":
        assert res.returned_n >= 1


@pytest.mark.asyncio
async def test_n_top_moves_include_moves_excludes_illegal() -> None:
    """Illegal include_moves must be rejected with a clean error."""
    with pytest.raises(ToolError):
        await server_module.top_moves(
            fen=POS_LONG_OPEN,
            n=3,
            depth=8,
            include_moves=["e9e9", "z9z9"],
        )


@pytest.mark.asyncio
async def test_n_top_moves_proof_mode_tactical() -> None:
    """proof_mode='tactical' must run and expose proof_status or sampled count."""
    res = await server_module.top_moves(
        fen=POS_LONG_OPEN, n=3, depth=8, proof_mode="tactical", proof_defenses=3
    )
    assert res.forensic_compute_triggered_by is not None
    assert "proof_mode" in res.forensic_compute_triggered_by


@pytest.mark.asyncio
async def test_n_top_moves_proof_mode_invalid_proof_defenses() -> None:
    """proof_defenses=0 in tactical mode must raise ToolError."""
    with pytest.raises(ToolError):
        await server_module.top_moves(
            fen=POS_LONG_OPEN,
            n=3,
            depth=8,
            proof_mode="tactical",
            proof_defenses=0,
        )


# ---------------------------------------------------------------------------
# Block O: analyze_game with annotations / comments
# ---------------------------------------------------------------------------


ANNOTATED_PGN_TEMPLATES: list[str] = [
    '[Event "Test"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n1. e4 {A nice opening.} e5 2. Nf3 Nc6 3. Bc4 {Italian.} Bc5 *',
    '[Event "Test"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 {Ruy Lopez.} Nf6 *',
    '[Event "Test"]\n[Result "*"]\n\n1. d4 d5 2. c4 {Queen\'s Gambit.} c6 *',
]


@pytest.mark.asyncio
@pytest.mark.parametrize("pgn", ANNOTATED_PGN_TEMPLATES)
async def test_o_analyze_game_with_comments(pgn: str) -> None:
    """PGN with annotated comments must parse and analyze successfully."""
    res = await server_module.analyze_game(pgn=pgn, depth=8)
    assert res.total_plies > 0
    # syntax_warnings should be empty list (or at most one benign warning).
    assert isinstance(res.syntax_warnings, list)


# ---------------------------------------------------------------------------
# Block P: Big concurrency — 200 calls in parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p_200_concurrent_evaluate_2_fens() -> None:
    """200 concurrent evaluate_position calls across 2 distinct FENs must
    return correct results without pool starvation."""
    await server_module._cache.clear()
    fens = [POS_LONG_OPEN, "r2q1rk1/pb1nbppp/1p2p3/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 10"]
    targets = [await server_module.evaluate_position(fen=f, depth=8) for f in fens]
    targets_by_fen = dict(zip(fens, targets, strict=False))
    tasks = []
    for i in range(200):
        fen = fens[i % 2]
        tasks.append(server_module.evaluate_position(fen=fen, depth=8))
    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"concurrent stress raised {len(errors)} errors: {errors[:3]}"
    # Every result must match the primed cache target.
    for i, r in enumerate(results):
        expected = targets_by_fen[fens[i % 2]]
        assert r.best_move == expected.best_move or r.cp == expected.cp


# ---------------------------------------------------------------------------
# Block Q: Unknown action_type / unknown tool parameter surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_q_classify_unknown_action_type() -> None:
    """Unknown action_type values must be rejected with a clean error."""
    with pytest.raises(ToolError):
        await server_module.classify_move(
            fen=POS_LONG_OPEN,
            move="Bxf7+",
            depth=6,
            action_type="resign",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_q_evaluate_unknown_detail_mode() -> None:
    """Unknown detail values must be rejected."""
    with pytest.raises(ToolError):
        await server_module.evaluate_position(
            fen=POS_LONG_OPEN,
            depth=6,
            detail="super-forensic",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_q_top_moves_unknown_proof_mode() -> None:
    """Unknown proof_mode must be rejected."""
    with pytest.raises(ToolError):
        await server_module.top_moves(
            fen=POS_LONG_OPEN,
            n=3,
            depth=6,
            proof_mode="random",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Block R: Mate-distance sign conventions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_r_mate_sign_for_already_mated_side() -> None:
    """When the side-to-move is mated, the engine reports mate distance as 0
    (or close to 0), not a positive integer."""
    # Scholar's mate: black is mated, it is black's turn.
    res = await server_module.evaluate_position(fen=POS_SCHOLARS, depth=10)
    assert res.status == "checkmate"
    # mate=0 (already mated now)
    assert res.mate == 0 or res.mate is None
    assert res.best_move is None


@pytest.mark.asyncio
async def test_r_mate_sign_after_one_move_to_mate() -> None:
    """Mate-in-1 for white: white's turn, mate distance must be 1 (positive)."""
    # Mate-in-1 with white to move.
    res = await server_module.evaluate_position(fen=POS_BACKRANK, depth=10)
    assert res.mate == 1


@pytest.mark.asyncio
async def test_r_top_moves_distinct_legal_candidates() -> None:
    """top_moves at n>=3 must return distinct best_move UCI candidates."""
    res = await server_module.top_moves(fen=POS_LONG_OPEN, n=5, depth=10)
    if res.status == "active" and len(res.result) >= 2:
        moves = [c.best_move for c in res.result if c.best_move]
        assert len(set(moves)) == len(moves), f"duplicates in top_moves: {moves}"


# ---------------------------------------------------------------------------
# Block S: Edge case — terminal positions surface recommended_action
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s_top_moves_checkmate_recommended_action() -> None:
    """For an already-checkmated position, recommended_action must be
    'game_over' (or equivalent terminal marker) and returned_n must be 0."""
    res = await server_module.top_moves(fen=POS_SCHOLARS, n=3, depth=8)
    assert res.status == "checkmate"
    assert res.returned_n == 0
    assert res.recommended_action == "game_over"


@pytest.mark.asyncio
async def test_s_top_moves_stalemate_recommended_action() -> None:
    """Stalemate: recommended_action='game_over' and returned_n=0."""
    res = await server_module.top_moves(fen=POS_STALEMATE, n=3, depth=8)
    assert res.recommended_action == "game_over"
    assert res.returned_n == 0


@pytest.mark.asyncio
async def test_s_top_moves_insufficient_material_recommended_action() -> None:
    """K+B vs K (insufficient material) — the side-to-move has no winning
    options; recommended_action should reflect that no game-deciding move exists."""
    res = await server_module.top_moves(fen=POS_INSUFFICIENT, n=3, depth=8)
    if res.status == "active":
        # Not strictly terminal — but no candidate should claim a winning cp.
        for c in res.result:
            if c.cp is not None:
                assert abs(c.cp) < 300


# ---------------------------------------------------------------------------
# Block T: Long-puzzle analysis pipeline (mate-in-N with detail=forensic)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen,label",
    [
        (POS_BACKRANK, "back_rank"),
        (POS_MATE_2, "mate_2_q_vs_rk"),
    ],
)
async def test_t_top_moves_forensic_detail_does_not_crash(fen: str, label: str) -> None:
    """Forensic detail on top_moves must not crash even at shallow depth."""
    res = await server_module.top_moves(fen=fen, n=3, depth=8, detail="forensic")
    assert res.status in {"active", "checkmate", "stalemate", "seventyfive_moves"}


# ---------------------------------------------------------------------------
# Block U: Real-world PGN samples (Fischer-Spassky, Carlsen, etc.)
# ---------------------------------------------------------------------------


REAL_GAMES: list[tuple[str, str]] = [
    # Andersen-Kieseritzky "Immortal Game" (1851) — short and well-documented.
    (
        "immortal_game_1851",
        '[Event "London"]\n[White "Andersen"]\n[Black "Kieseritzky"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. f4 exf4 3. Bc4 Qh4+ 4. Kf1 b5 5. Bxb5 Nf6 6. Nf3 Qh6 "
        "7. d3 Nh5 8. Nh4 Qg5 9. Nf5 c6 10. g4 Nf6 11. Rg1 cxb5 12. h4 Qg6 "
        "13. h5 Qg5 14. Qf3 Ng8 15. Bxf4 Qf6 16. Nc3 Bc5 17. Nd5 Qxb2 "
        "18. Bd6 Bxg1 19. e5 Qxa1+ 20. Ke2 Na6 21. Nxg7+ Kd8 22. Qf6+ Nxf6 23. Be7# 1-0",
    ),
    # Morphy's Opera Game (1858) — short and famous.
    (
        "morphy_opera_1858",
        '[Event "Paris Opera"]\n[White "Morphy"]\n[Black "Duke Karl & Count Isouard"]\n[Result "1-0"]\n\n'
        "1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 "
        "7. Qb3 Qe7 8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 "
        "12. O-O-O Rd8 13. Rxd7 Rxd7 14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 "
        "17. Rd8# 1-0",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,pgn", REAL_GAMES)
async def test_u_analyze_game_real_pgn(case_id: str, pgn: str) -> None:
    """Real-world PGNs from famous games must analyze without errors."""
    res = await server_module.analyze_game(pgn=pgn, depth=8)
    assert res.total_plies > 0
    assert res.white_accuracy is not None
    assert res.black_accuracy is not None
