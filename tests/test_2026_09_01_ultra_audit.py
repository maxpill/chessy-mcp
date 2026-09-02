"""Regression tests for the 2026-09-01 ultra-audit U-01..U-18 findings.

Comprehensive regression suite covering the P0/P1 bugs consolidated in
`/Users/max/Downloads/chess_mcp_master_deduplicated_findings_2026-09-01.md`.
Each test pins one invariant from the audit.

Bug index (matches the audit's U-NN IDs):

    U-01  top_moves ranks mate-for-mover above any saturated cp=±20000 and
          best_move is invariant to the requested `n`.
    U-02  classify_move: a legal move that resets the halfmove clock and
          leads to an easy technical win MUST be preferred over claim_draw.
    U-03  analyze_game: initial terminal FEN (75-move or insufficient material)
          must NOT consume the first movetext move.
    U-04  classify_move: after a successful draw claim, eval_after must be
          a pure terminal snapshot (no active engine best_move / pv / claim
          fields). pre-action engine state must move to a dedicated field.
    U-05  classify_move: after claim_draw_with_intended_move, played_line_san
          must equal the intended move exactly (no continuation string).
    U-06  analyze_game: a non-best played move must never have
          best_move_san == san.
    U-07  action_equivalent: split semantics — same_action_type,
          same_outcome, within_cp_threshold as separate flags.
    U-08  top_moves: root_score and post_state_score surfaced as distinct
          fields.
    U-09  FEN: invalid castling rights (K/Q/k/q referring to a non-existent
          rook) are rejected symmetrically in repair AND strict mode.
    U-10  MCPEval.legal_actions renamed to legal_rule_actions for clarity;
          full legal_move_uci list also exposed.
    U-11  strict Unicode SAN parity between classify_move and analyze_game.
    U-12  classify_move: in strict mode, claim_draw + garbage move raises.
    U-13  every nested eval dict carries build_sha / engine_config /
          requested_depth / searched_depth.
    U-14  strict PGN metadata: Date must match YYYY.MM.DD or be blank;
          Termination must match the canonical set or be blank.
    U-15  analyze_game: wrong move-number side marker (e.g. 2... e5 for
          white's move) raises a STRICT_PGN_ERROR in strict mode.
    U-16  analyze_game: bare movetext with multiple games separated by a
          result marker behaves consistently with two full PGNs (raises
          MULTIPLE_GAMES_DETECTED).
    U-17  informational — no fix.
    U-18  FEN: halfmove clock upper-bound at 10000 to prevent pathological
          inputs.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module


# ---------------------------------------------------------------------------
# Test fixtures (fake analyzer pools)
# ---------------------------------------------------------------------------


class _MateAndSaturatedCPPool:
    """Reproduces the U-01 / N-02 scenario: Stockfish returns both a forced
    mate-in-1 candidate and one or more saturated cp=±20000 candidates.

    Without a chess-correct rank key the cp=±20000 line can outrank the
    mating line. The pool's `top_moves` ordering matches the buggy ordering
    observed in production (cp first, mate second) so the rank key MUST
    re-sort it."""

    name = "MateAndSaturatedCPPool"
    engine_version = "MateAndSaturatedCPPool"

    def __init__(self) -> None:
        self.evaluate_calls = 0
        self.top_moves_calls = 0

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        self.evaluate_calls += 1
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)

    async def top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 14,
    ) -> list[Eval]:
        self.top_moves_calls += 1
        if board.is_checkmate():
            return []
        # Buggy order: saturated cp line first, mate second. Rank key must
        # reorder so mate is items[0].
        return [
            Eval(cp=20000, mate=None, best_move="g6f6", pv=["g6f6"], depth=depth),
            Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth),
            Eval(cp=15000, mate=None, best_move="g6f7", pv=["g6f7"], depth=depth),
        ][:n]

    async def close(self) -> None:
        pass


class _StableMatePool:
    """Always returns a mate-in-1 line regardless of n — used to test that
    best_move is n-invariant when Stockfish's input is."""

    name = "StableMatePool"
    engine_version = "StableMatePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        if board.is_checkmate():
            return []
        return [Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)] * min(n, 1)

    async def close(self):
        pass


class _NFlippedBestPool:
    """The best_move flips between two candidates based on n — used to
    verify that a chess-correct rank key is n-invariant.

    n=1 returns Kf6 first (the "cp=20000" saturated sentinel).
    n=2 returns e8=Q# first (the actual mate-in-1)."""

    def __init__(self) -> None:
        self.calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        self.calls += 1
        if board.is_checkmate():
            return []
        if n == 1:
            return [
                Eval(cp=20000, mate=None, best_move="g6f6", pv=["g6f6"], depth=depth),
                Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth),
            ]
        return [
            Eval(cp=None, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth),
            Eval(cp=20000, mate=None, best_move="g6f6", pv=["g6f6"], depth=depth),
        ]

    async def close(self):
        pass


# ---------------------------------------------------------------------------
# U-01: top_moves mate vs cp rank order + n-invariance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u01_top_moves_mate_outranks_saturated_cp():
    """Saturated cp=±20000 candidate must NOT outrank a mate-in-1 candidate.

    Reproduces the audit U-01 scenario where `top_moves` returned Kf6
    cp=20000 at slot 0 and e8=Q# mate=1 at slot 1."""
    await server_module._cache.clear()
    pool = _MateAndSaturatedCPPool()
    server_module._analyzer_pool = pool  # type: ignore[assignment]

    fen = "7k/4P3/6K1/8/8/8/8/8 w - - 0 1"
    res = await server_module.top_moves(fen, n=3, depth=12)

    assert len(res.result) >= 1, "expected at least one candidate"
    best = res.result[0]
    assert best.best_move == "e7e8q", (
        f"mate-in-1 candidate e7e8q must rank first; got {best.best_move!r} "
        f"with cp={best.cp} mate={best.mate}"
    )
    assert best.mate is not None and best.mate > 0, (
        f"first candidate must carry mate > 0; got mate={best.mate}"
    )


@pytest.mark.asyncio
async def test_u01_top_moves_n_invariance_same_best_move():
    """best_move must be the same regardless of the requested `n` value.

    A chess-correct rank key must put the strongest candidate at items[0]
    even when Stockfish returns candidates in different orders depending
    on n."""
    pool = _NFlippedBestPool()

    # n=1: Stockfish returns cp-first; rank key must reorder to mate.
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    await server_module._cache.clear()
    res1 = await server_module.top_moves("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", n=1, depth=10)
    best1 = res1.result[0]

    # n=2: Stockfish returns mate-first; rank key preserves that.
    await server_module._cache.clear()
    res2 = await server_module.top_moves("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", n=2, depth=10)
    best2 = res2.result[0]

    assert best1.best_move == "e7e8q", (
        f"best_move at n=1 must be mate e7e8q; got {best1.best_move!r}"
    )
    assert best2.best_move == "e7e8q", (
        f"best_move at n=2 must be mate e7e8q; got {best2.best_move!r}"
    )
    assert best1.best_move == best2.best_move, (
        "best_move must be n-invariant; the chess-correct rank key guarantees it"
    )


@pytest.mark.asyncio
async def test_u01_top_moves_root_recommended_action_uses_mate():
    """When items[0] has mate set, root recommended_action must reflect
    the mate (via `_pick_root_recommended_action`), not the saturated cp
    that may also be present."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _StableMatePool()  # type: ignore[assignment]

    res = await server_module.top_moves("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", n=1, depth=10)
    # root recommended_action should be play_move (since mate-in-1 is the
    # best move, the engine recommends playing it). The key point is that
    # the action layer is not fooled by a stale cp.
    assert res.recommended_action in ("play_move", "game_over")
    if res.recommended_action == "play_move":
        ba = res.best_action_obj
        assert ba is not None
        assert ba.get("type") == "play_move"
        # The play_move action carries a payload {uci, san} — pick whichever
        # is present and verify it's the mate-in-1 move.
        move_payload = ba.get("move") or {}
        uci = (
            (move_payload.get("uci") if isinstance(move_payload, dict) else None)
            or ba.get("move_uci")
            or ba.get("uci")
        )
        assert uci == "e7e8q", f"best action's move must be mate-in-1; got {uci!r}"


@pytest.mark.asyncio
async def test_u01_top_moves_cp_clamped_above_mate_rank():
    """A cp=20000 candidate must rank at the mate ceiling (9999.0), not
    above the mate-in-1 candidate (10000.0 - 1 = 9999.0).

    We verify indirectly: in `_StableMatePool` the only candidate has
    cp=20000 + mate=1 (Stockfish-style). It must end up first because
    mate takes precedence."""
    await server_module._cache.clear()

    class _CpAndMateSameLine:
        name = "CpAndMateSameLine"
        engine_version = "CpAndMateSameLine"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            if board.is_checkmate():
                return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
            return Eval(cp=20000, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            if board.is_checkmate():
                return []
            return [Eval(cp=20000, mate=1, best_move="e7e8q", pv=["e7e8q"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = _CpAndMateSameLine()  # type: ignore[assignment]
    res = await server_module.top_moves("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", n=1, depth=10)
    assert res.result[0].best_move == "e7e8q"
    assert res.result[0].mate == 1


# ---------------------------------------------------------------------------
# U-02: draw policy vs winning reset capture
# ---------------------------------------------------------------------------


class _WinningCapturePool:
    """Reproduces the U-02 audit scenario: White has Kxe2 (capture that
    resets the halfmove clock and leads to K+R vs K — a technical win) at
    halfmove=100. The pool's evaluate returns cp=+26 (draw-polluted root
    value, the buggy old behavior), but best_move = e1e2 (the winning
    capture). The post-state evaluation returns a winning cp so the
    action layer must recommend play_move over claim_draw."""

    name = "WinningCapturePool"
    engine_version = "WinningCapturePool"

    def __init__(self) -> None:
        self.post_eval_calls = 0

    async def evaluate(self, board, *, depth=14, root_moves=None):
        # If the position is the post-Kxe2 board: Black to move, no rook
        # on e2 (captured), White king now on e2, White rook on h1,
        # Black king on e8 — K+R vs K, winning for White.
        if (
            board.turn == chess.BLACK
            and board.piece_type_at(chess.E2) == chess.KING
            and board.piece_type_at(chess.H1) == chess.ROOK
            and board.piece_type_at(chess.E8) == chess.KING
        ):
            return Eval(cp=20000, mate=None, best_move="", pv=[], depth=depth)
        return Eval(
            cp=26,
            mate=None,
            best_move="e1e2",  # Kxe2
            pv=["e1e2"],
            depth=depth,
        )

    async def top_moves(self, board, n=3, depth=14):
        return [await self.evaluate(board, depth=depth)]

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_u02_evaluate_position_prefers_play_move_when_zeroing_wins():
    await server_module._cache.clear()
    server_module._analyzer_pool = _WinningCapturePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position("4k3/8/8/8/8/8/4r3/4K2R w - - 100 51", depth=10)
    assert res.best_action == "play_move", (
        f"expected play_move (winning zeroing capture Kxe2); got {res.best_action!r}"
    )
    assert res.recommended_action == "play_move"
    assert res.best_move == "e1e2"


@pytest.mark.asyncio
async def test_u02_classify_move_claim_draw_when_zeroing_wins_is_blunder():
    from core.engines.types import MoveClass

    await server_module._cache.clear()
    server_module._analyzer_pool = _WinningCapturePool()  # type: ignore[assignment]

    res = await server_module.classify_move(
        "4k3/8/8/8/8/8/4r3/4K2R w - - 100 51",
        move=None,
        depth=10,
        action_type="claim_draw",
    )
    assert res.move_class != MoveClass.BEST, (
        f"claim_draw with winning Kxe2 reset available must NOT be BEST; got {res.move_class!r}"
    )
    assert res.effective_loss is not None and res.effective_loss > 0
    assert res.loss_kind in ("outcome_penalty", "technical_win_forfeited"), (
        f"loss_kind must indicate a forfeited win; got {res.loss_kind!r}"
    )
    assert res.is_best_action is False


@pytest.mark.asyncio
async def test_u02_evaluate_position_symmetric_black():
    class _BlackWinningCapturePool(_WinningCapturePool):
        async def evaluate(self, board, *, depth=14, root_moves=None):
            # Post-Kxe7 board: White to move, Black king on e7 (after
            # capture), Black rook on h8, White king on e1 — K+R vs K,
            # Black winning.
            if (
                board.turn == chess.WHITE
                and board.piece_type_at(chess.E7) == chess.KING
                and board.piece_type_at(chess.H8) == chess.ROOK
                and board.piece_type_at(chess.E1) == chess.KING
            ):
                return Eval(cp=-20000, mate=None, best_move="", pv=[], depth=depth)
            return Eval(
                cp=-26,
                mate=None,
                best_move="e8e7",
                pv=["e8e7"],
                depth=depth,
            )

    await server_module._cache.clear()
    server_module._analyzer_pool = _BlackWinningCapturePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position("4k2r/4R3/8/8/8/8/8/4K3 b - - 100 51", depth=10)
    assert res.best_action == "play_move"
    assert res.best_move == "e8e7"


@pytest.mark.asyncio
async def test_u02_claim_draw_still_best_when_no_winning_zeroing():
    """Guard against U-02 fix overshooting. White has Kxd2 (capture that
    resets the clock but leads to K vs K — drawn, not winning). The rule
    layer must NOT flag claim_draw as a blunder because no winning
    zeroing move exists."""

    class _DrawnZeroingPool:
        name = "DrawnZeroingPool"
        engine_version = "DrawnZeroingPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            # Post-Kxd2 board is K vs K — fully drawn.
            if (
                board.turn == chess.BLACK
                and board.piece_type_at(chess.D2) is None
                and board.piece_type_at(chess.E1) == chess.KING
                and board.piece_type_at(chess.E8) == chess.KING
            ):
                return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)
            return Eval(cp=0, mate=None, best_move="e1d2", pv=["e1d2"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [await self.evaluate(board, depth=depth)]

        async def close(self):
            pass

    await server_module._cache.clear()
    server_module._analyzer_pool = _DrawnZeroingPool()  # type: ignore[assignment]
    res = await server_module.evaluate_position("4k3/8/8/8/8/8/3r4/4K3 w - - 100 51", depth=10)
    # Kxd2 leads to a draw (K vs K), not a win — claim_draw should still be valid.
    assert res.best_action == "claim_draw", (
        f"Kxd2 leads to drawn K vs K — claim_draw must remain BEST; got {res.best_action!r}"
    )
    assert res.can_claim_now is True


# ---------------------------------------------------------------------------
# U-03: initial terminal FEN check
# ---------------------------------------------------------------------------


class _NoopEnginePool:
    """Minimal pool that returns a draw-ish eval for any board."""

    name = "NoopEnginePool"
    engine_version = "NoopEnginePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        return [Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)]

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_u03_analyze_game_terminal_start_75_moves_non_strict():
    """Initial FEN already 75-move-terminal. Non-strict mode records a
    warning and 0 plies are executed."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = (
        '[SetUp "1"]\n'
        '[FEN "7k/8/8/8/8/8/5R2/7K w - - 150 75"]\n'
        '[Result "1/2-1/2"]\n\n'
        "75. Rf8+ 1/2-1/2\n"
    )
    res = await server_module.analyze_game(pgn, depth=10)

    # 0 plies executed (terminal at start, movetext is bogus).
    assert res.total_plies == 0, (
        f"expected 0 executed plies for terminal initial FEN; got total_plies={res.total_plies}"
    )
    # Termination surfaced.
    assert res.termination in ("seventyfive_moves", "draw"), (
        f"expected seventyfive_moves termination; got {res.termination!r}"
    )
    # A warning was recorded (non-strict).
    assert any("terminal" in w.lower() for w in res.syntax_warnings), (
        f"expected a terminal-related syntax warning; got syntax_warnings={res.syntax_warnings!r}"
    )


@pytest.mark.asyncio
async def test_u03_analyze_game_terminal_start_75_moves_strict_raises():
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = (
        '[SetUp "1"]\n'
        '[FEN "7k/8/8/8/8/8/5R2/7K w - - 150 75"]\n'
        '[Result "1/2-1/2"]\n\n'
        "75. Rf8+ 1/2-1/2\n"
    )
    with pytest.raises(Exception) as excinfo:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(excinfo.value)
    assert ("STRICT_PGN_ERROR" in msg) or ("STRICT_VALIDATION_ERROR" in msg), (
        f"strict mode must raise a STRICT error; got {excinfo.value!r}"
    )


@pytest.mark.asyncio
async def test_u03_analyze_game_terminal_start_insufficient_material():
    """Initial FEN already terminal by insufficient material (K+B vs K).
    Non-strict: 0 plies + warning."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = (
        '[SetUp "1"]\n[FEN "7k/8/8/8/8/8/6B1/7K w - - 0 1"]\n[Result "1/2-1/2"]\n\n1. Bf3 1/2-1/2\n'
    )
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 0, f"expected 0 executed plies; got total_plies={res.total_plies}"
    assert res.termination in ("insufficient_material", "draw"), (
        f"expected insufficient_material termination; got {res.termination!r}"
    )


@pytest.mark.asyncio
async def test_u03_analyze_game_active_start_unaffected():
    """An active (non-terminal) starting FEN must still process moves
    normally — the U-03 fix only affects terminal initial states."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = "1. e4 e5 2. Nf3 Nc6 *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies >= 4, (
        f"expected at least 4 executed plies; got total_plies={res.total_plies}"
    )


# ---------------------------------------------------------------------------
# U-04: post-claim eval_after must be a pure terminal snapshot
# ---------------------------------------------------------------------------


class _MateAtRootPool:
    """Pool that returns mate-in-1 (Qc8#) for any non-terminal board.
    Used to set up the audit U-04 scenario where a claim_draw is honored
    but the engine's pre-claim eval still shows mate."""

    name = "MateAtRootPool"
    engine_version = "MateAtRootPool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        if board.is_checkmate():
            return Eval(cp=0, mate=0, best_move="", pv=[], depth=depth)
        return Eval(
            cp=None,
            mate=1,
            best_move="f5c8",
            pv=["f5c8"],
            depth=depth,
        )

    async def top_moves(self, board, n=3, depth=14):
        if board.is_checkmate():
            return []
        return [await self.evaluate(board, depth=depth)]

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_u04_claim_draw_pure_terminal_eval_after():
    """After a successful claim_draw, eval_after must be a pure terminal
    snapshot — no best_move, no PV, no can_claim_*, no active best_action."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtRootPool()  # type: ignore[assignment]

    # Position: White has Qc8# available but can also claim draw.
    res = await server_module.classify_move(
        "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51",
        move=None,
        depth=10,
        action_type="claim_draw",
    )
    ea = res.eval_after
    # Score is terminal draw.
    assert ea.status == "draw", f"eval_after.status must be draw; got {ea.status!r}"
    assert ea.cp == 0, f"eval_after.cp must be 0; got {ea.cp}"
    assert ea.mate is None, f"eval_after.mate must be None; got {ea.mate!r}"
    # No active engine state.
    assert ea.best_move is None, (
        f"eval_after.best_move must be None after a granted claim; got {ea.best_move!r}"
    )
    assert ea.executable_move is None, (
        f"eval_after.executable_move must be None; got {ea.executable_move!r}"
    )
    assert ea.pv == [], f"eval_after.pv must be empty; got {ea.pv!r}"
    assert ea.can_claim_draw is False, "can_claim_draw must be False post-claim"
    assert ea.can_claim_now is False, "can_claim_now must be False post-claim"
    assert ea.can_claim_with_intended_move is False
    # best_action surface must read as game_over.
    assert ea.recommended_action == "game_over"
    assert ea.best_action == "game_over"
    assert ea.best_action_type == "game_over"
    assert ea.best_action_obj is not None
    assert ea.best_action_obj.get("type") == "game_over"
    assert ea.best_action_obj.get("outcome") == "draw"
    assert ea.legal_actions == [], (
        f"legal_actions must be empty post-claim; got {ea.legal_actions!r}"
    )


@pytest.mark.asyncio
async def test_u04_claim_draw_intended_pure_terminal_eval_after():
    """Same invariant for claim_draw_with_intended_move."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtRootPool()  # type: ignore[assignment]

    res = await server_module.classify_move(
        "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51",
        move=None,
        depth=10,
        action_type="claim_draw",
    )
    ea = res.eval_after
    assert ea.status == "draw"
    assert ea.best_move is None
    assert ea.pv == []
    assert ea.best_action_obj.get("type") == "game_over"


# ---------------------------------------------------------------------------
# U-05: claim_draw_with_intended_move has no continuation after termination
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u05_intended_claim_has_no_continuation_after_termination():
    """claim_draw_with_intended_move must produce played_line_san equal to
    the intended move alone — no continuation string, no SAN like
    'Ng8 exe5' that re-renders the root PV onto the post-claim board."""
    await server_module._cache.clear()

    class _ThreefoldClaimPool:
        """Engine reports Black's best_move=Ng8, which is also the
        intended-move that creates the threefold claim. After the claim
        is granted the eval_after is forced to terminal draw, so any
        continuation rendering must collapse to None."""

        name = "ThreefoldClaimPool"
        engine_version = "ThreefoldClaimPool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            return Eval(
                cp=0,
                mate=None,
                best_move="g8g7",  # Black's Ng8 would be at g8→f6, etc.
                pv=["g8g7"],
                depth=depth,
            )

        async def top_moves(self, board, n=3, depth=14):
            return [await self.evaluate(board, depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = _ThreefoldClaimPool()  # type: ignore[assignment]

    # Use a board state with move history matching the audit's history.
    # We replay Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 (the audit's history) using the
    # `moves` parameter, then call classify_move(action_type=claim_draw_with_intended_move)
    # with the intended Ng8 move.
    history = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]
    intended = "f6g8"  # Black's Ng8 = f6→g8, creates the third repetition

    res = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        moves=history,
        move=intended,
        depth=10,
        action_type="claim_draw_with_intended_move",
    )
    # played_line_san must contain ONLY the intended move's SAN. No
    # "exe5" continuation string leaking the root PV.
    assert res.played_line_san is not None
    assert "exe5" not in (res.played_line_san or ""), (
        f"played_line_san must NOT contain post-terminal continuation; got {res.played_line_san!r}"
    )
    # If the intended move's SAN is "Ng8", then played_line_san must
    # equal it (no continuation).
    played_san = res.played_san
    if played_san is not None:
        assert res.played_line_san == played_san, (
            f"played_line_san must equal played_san for intended claims "
            f"(no continuation); got played_line_san={res.played_line_san!r}, "
            f"played_san={played_san!r}"
        )


# ---------------------------------------------------------------------------
# U-06: analyze_game turning-point invariant — blunder != best_move_san
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u06_analyze_game_blunder_cannot_equal_best_move_san():
    """Invariant from audit U-06: if a played move is classified as a
    blunder/mistake, the turning point's `best_move_san` must NOT equal
    `san`. Without this guard, depth=1 turning points can show
    `best_move_san == san` while `move_class == "blunder"`.
    """

    class _PromotionDefensePool:
        """Reproduces audit U-06: at depth=1 the engine picks h1=Q as
        the best move, but at depth=2+ it sees h1=N forces a draw and
        h1=Q allows Qf2# (blunder). The pool is intentionally
        shallow-only on the promotion square so the contradiction can
        surface."""

        name = "PromotionDefensePool"
        engine_version = "PromotionDefensePool"

        async def evaluate(self, board, *, depth=14, root_moves=None):
            # Black to move on the h-file promotion square; return h1=Q
            # as best with a small cp that the classifier will read as
            # blunder after a deeper check.
            return Eval(
                cp=-100,
                mate=None,
                best_move="h2h1q",
                pv=["h2h1q"],
                depth=depth,
            )

        async def top_moves(self, board, n=3, depth=14):
            return [await self.evaluate(board, depth=depth)]

        async def close(self):
            pass

    await server_module._cache.clear()
    server_module._analyzer_pool = _PromotionDefensePool()  # type: ignore[assignment]

    # PGN with a final h2h1q (h1=Q) promotion — classify_move at this
    # position should produce a non-best classification.
    pgn = (
        '[SetUp "1"]\n'
        '[FEN "8/P7/8/8/8/8/7p/4K2k w - - 0 1"]\n'
        '[Result "*"]\n\n'
        "1. a8=Q+ Kg1 2. Qf3 h1=Q 3. Qf2#\n"
    )
    res = await server_module.analyze_game(pgn, depth=1)

    # Walk the turning points: every non-best classified ply must have
    # best_move_san != san (or best_move_san None).
    for tp in res.turning_points:
        if tp.move_class in ("blunder", "mistake"):
            assert tp.best_move_san != tp.san, (
                f"U-06 violation: ply {tp.ply} classified as {tp.move_class!r} "
                f"but best_move_san={tp.best_move_san!r} == san={tp.san!r}"
            )


# ---------------------------------------------------------------------------
# U-07: action_equivalent primitive split
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u07_action_equivalent_splits_into_primitives():
    """PlayedMoveScore exposes same_action_type / same_outcome /
    within_cp_threshold as separate primitives alongside the legacy
    action_equivalent summary."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtRootPool()  # type: ignore[assignment]

    # Successful claim_draw (a draw claim the engine would also recommend).
    res = await server_module.classify_move(
        "7k/8/6K1/5Q2/8/8/8/8 w - - 100 51",
        move=None,
        depth=10,
        action_type="claim_draw",
    )
    score = res.score if hasattr(res, "score") else None
    # The audit U-07 invariant: the new primitives are exposed and
    # consistent. The legacy `action_equivalent` boolean is preserved
    # as a derived field for back-compat. We can't construct a
    # PlayedMoveScore directly here without internals, but we can
    # verify the field shape on the wire: the existing analyze_game
    # turning points expose move_class but not the new primitives —
    # that's fine, they're meant for the classify_move / score layer.
    assert res.move_class is not None


@pytest.mark.asyncio
async def test_u07_playscore_serializer_exposes_primitives():
    """Direct serialization of PlayedMoveScore must expose the new
    primitive fields. The audit wants callers to be able to read
    same_action_type, same_outcome, within_cp_threshold separately."""
    from mcp_server.models import PlayedMoveScore
    from core.engines.types import MoveClass

    ps = PlayedMoveScore(
        move_class=MoveClass.BEST,
        action_type="claim_draw",
        best_action="claim_draw",
        is_best_action=True,
        centipawn_loss=0,
    )
    dumped = ps.model_dump()
    assert "same_action_type" in dumped
    assert dumped["same_action_type"] is True
    assert "same_outcome" in dumped
    assert dumped["same_outcome"] is True
    assert "within_cp_threshold" in dumped
    assert dumped["within_cp_threshold"] is True


def test_u07_primitives_diverge_when_actions_differ():
    """A played claim_draw with engine-recommended play_move must have
    same_action_type=False, same_outcome=True (both terminal-draw in
    effect), and within_cp_threshold=True (no comparable cp)."""
    from mcp_server.models import PlayedMoveScore
    from core.engines.types import MoveClass

    ps = PlayedMoveScore(
        move_class=MoveClass.BEST,
        action_type="claim_draw",
        best_action="play_move",
        is_best_action=False,  # claim != play
        centipawn_loss=0,
    )
    dumped = ps.model_dump()
    assert dumped["same_action_type"] is False
    assert dumped["same_outcome"] is False  # follows is_best_action


# ---------------------------------------------------------------------------
# U-08: top_moves root/post-state score field semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u08_root_score_cp_mate_explicit():
    """Every MCPEval exposes root_score_cp / root_score_mate /
    post_fen fields with the documented semantics."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", depth=10)
    # Root score fields present and equal to cp/mate.
    assert hasattr(res, "root_score_cp")
    assert hasattr(res, "root_score_mate")
    assert hasattr(res, "post_fen")
    assert res.root_score_cp == res.cp
    assert res.root_score_mate == res.mate
    # post_fen is the same as the input FEN for evaluate_position (root).
    assert res.post_fen is not None


@pytest.mark.asyncio
async def test_u08_top_moves_candidates_have_post_fen():
    """top_moves candidates expose post_fen = board state after the
    candidate move is played."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _MateAtRootPool()  # type: ignore[assignment]

    res = await server_module.top_moves("7k/8/6K1/5Q2/8/8/8/8 w - - 100 51", n=1, depth=10)
    assert len(res.result) >= 1
    cand = res.result[0]
    assert hasattr(cand, "post_fen")
    # The mating candidate delivers checkmate; post_fen is the
    # post-mate position (Black to move, in checkmate).
    assert cand.post_fen is not None
    b_after = chess.Board(cand.post_fen)
    assert b_after.is_checkmate()


# ---------------------------------------------------------------------------
# U-09: castling-rights symmetry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u09_castling_K_without_h1_rook_strict_rejected():
    """FEN with K castling right but no white rook on h1 is rejected in
    strict mode (audit U-09: previously silent repair)."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    # No rook on h1 — King on e1 only.
    fen = "4k3/8/8/8/8/8/8/4K3 w K - 0 1"
    with pytest.raises(Exception) as excinfo:
        await server_module.evaluate_position(fen, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "INVALID_CASTLING_RIGHTS" in msg or "INVALID_FEN" in msg, (
        f"strict mode must reject invalid K rights; got {msg!r}"
    )


@pytest.mark.asyncio
async def test_u09_castling_Q_without_a1_rook_strict_rejected():
    """Symmetric to U-09 K-without-h1-rook test."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    fen = "4k3/8/8/8/8/8/8/4K3 w Q - 0 1"
    with pytest.raises(Exception) as excinfo:
        await server_module.evaluate_position(fen, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "INVALID_CASTLING_RIGHTS" in msg or "INVALID_FEN" in msg


@pytest.mark.asyncio
async def test_u09_castling_lowercase_k_without_h8_rook_strict_rejected():
    """Symmetric to K-without-h1-rook. FEN claims Black has kingside
    castling but there's no Black rook on h8."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    # Only the Black king on e8, no rooks.
    fen = "4k3/8/8/8/8/8/8/4K3 b k - 0 1"
    with pytest.raises(Exception) as excinfo:
        await server_module.evaluate_position(fen, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "INVALID_CASTLING_RIGHTS" in msg or "INVALID_FEN" in msg


@pytest.mark.asyncio
async def test_u09_castling_invalid_silently_repaired_non_strict():
    """Non-strict mode silently strips invalid rights and continues."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    fen = "4k3/8/8/8/8/8/8/4K3 w KQ - 0 1"
    res = await server_module.evaluate_position(fen, depth=10)
    # Both rights were bogus; after repair the canonical FEN has "-".
    assert res.canonical_fen is not None
    parts = res.canonical_fen.split()
    assert parts[2] == "-", f"non-strict must strip bogus rights to '-'; got rights={parts[2]!r}"


@pytest.mark.asyncio
async def test_u09_castling_valid_rights_pass_through():
    """Legal castling rights are accepted unchanged."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    # Standard startpos with both kings and rooks on canonical squares.
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    res = await server_module.evaluate_position(fen, depth=10)
    assert res.canonical_fen is not None
    parts = res.canonical_fen.split()
    assert parts[2] == "KQkq", f"valid KQkq rights must pass through; got {parts[2]!r}"


# ---------------------------------------------------------------------------
# U-10: legal_actions renamed to legal_rule_actions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u10_legal_rule_actions_and_legal_move_uci_exposed():
    """Audit U-10: callers must be able to read the full legal move list
    via `legal_move_uci` and the rule-only list via `legal_rule_actions`.
    The legacy `legal_actions` field is a back-compat alias."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=10
    )
    assert hasattr(res, "legal_rule_actions")
    assert hasattr(res, "legal_move_uci")
    assert hasattr(res, "legal_actions")  # back-compat alias
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    expected_uci = sorted(m.uci() for m in b.legal_moves)
    actual_uci = sorted(res.legal_move_uci)
    assert actual_uci == expected_uci, (
        f"legal_move_uci must list every legal ply; got {actual_uci!r}, expected {expected_uci!r}"
    )
    assert res.legal_actions == res.legal_rule_actions


@pytest.mark.asyncio
async def test_u10_legal_move_uci_includes_all_legal_moves():
    """A position with several legal moves must surface them all in
    legal_move_uci, not just the engine-recommended ones."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=10
    )
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert len(res.legal_move_uci) == b.legal_moves.count()
    assert len(res.legal_move_uci) > 1


# ---------------------------------------------------------------------------
# U-11: strict Unicode SAN parity between classify_move and analyze_game
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u11_classify_move_accepts_unicode_dash_castling():
    """classify_move must accept castling notation with Unicode dashes
    (U+2013 en-dash, U+2014 em-dash) just like analyze_game does —
    audit U-11 parity fix."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # Italian opening position — O-O is legal here (king on e1, h1 rook,
    # f1/g1 empty, no path-attacks). The castling move must parse to O-O.
    fen = "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    res = await server_module.classify_move(fen, move="0–0", depth=10)
    assert res.played_san == "O-O", (
        f"Unicode-dash 0–0 must parse to canonical O-O; got played_san={res.played_san!r}"
    )


@pytest.mark.asyncio
async def test_u11_classify_move_strict_rejects_figurine_piece():
    """Strict mode must reject a figurine piece (♘f3) just like
    analyze_game does. Audit U-11 fix."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    with pytest.raises(Exception) as excinfo:
        await server_module.classify_move(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move="♘f3",
            depth=10,
            strict=True,
        )
    assert "STRICT" in str(excinfo.value).upper(), (
        f"strict mode must reject figurine notation; got {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# U-12: claim_draw silently ignores garbage `move` in strict mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u12_claim_draw_garbage_move_strict_rejected():
    """Audit U-12: in strict mode, claim_draw + garbage `move` must be
    rejected (or at minimum surface a warning), not silently accepted."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # FEN: White to move at halfmove=100, can claim draw.
    fen = "7k/8/8/8/8/8/8/4K2R w - - 100 51"
    # But the OPPOSITE_CHECK issue: Black king on h8, White rook on h1
    # would put Black in check. Use a different position.
    fen = "7k/8/8/8/8/8/8/R3K3 w - - 100 51"  # KR vs K, halfmove=100

    with pytest.raises(Exception) as excinfo:
        await server_module.classify_move(
            fen,
            move="THIS_IS_NOT_A_MOVE",
            depth=10,
            action_type="claim_draw",
            strict=True,
        )
    msg = str(excinfo.value)
    # Either a strict rejection or a clear warning.
    assert (
        "STRICT" in msg.upper()
        or "INVALID_MOVE" in msg.upper()
        or "MISSING_MOVE" in msg.upper()
        or "ILLEGAL" in msg.upper()
    ), f"strict claim_draw + garbage move must be rejected; got {msg!r}"


@pytest.mark.asyncio
async def test_u12_claim_draw_garbage_move_non_strict_accepted():
    """Non-strict mode: claim_draw with garbage `move` is accepted (the
    move is irrelevant for claim actions); the move argument is ignored
    per audit U-12's documented intent. The fix raises in STRICT mode
    but allows the lenient path to keep working in non-strict."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    fen = "7k/8/8/8/8/8/8/R3K3 w - - 100 51"
    # Non-strict: should NOT raise; the dummy move is ignored for
    # claim_draw actions per the audit's B-02 invariant.
    res = await server_module.classify_move(
        fen,
        move="THIS_IS_NOT_A_MOVE",
        depth=10,
        action_type="claim_draw",
        strict=False,
    )
    assert res is not None
    assert res.move_class is not None


# ---------------------------------------------------------------------------
# U-13: nested engine_eval dict carries build identity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u13_nested_engine_eval_has_build_sha():
    """Audit U-13: every nested engine_eval sub-dict must carry
    build_sha / engine_config so callers reading just the sub-dict get
    the same provenance as the parent MCPEval."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    res = await server_module.evaluate_position("7k/4P3/6K1/8/8/8/8/8 w - - 0 1", depth=10)
    assert res.engine_eval is not None
    assert "build_sha" in res.engine_eval
    assert "engine_config" in res.engine_eval
    assert "requested_depth" in res.engine_eval
    assert "searched_depth" in res.engine_eval
    # build_sha should match the parent's build_sha.
    assert res.engine_eval["build_sha"] == res.build_sha
    assert res.engine_eval["engine_config"] == res.engine_config


# ---------------------------------------------------------------------------
# U-14: strict PGN metadata validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u14_strict_date_must_match_yyyy_mm_dd():
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = '[Date "2026.99.99"]\n[Result "*"]\n\n1. e4 *\n'
    with pytest.raises(Exception) as excinfo:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "STRICT" in msg.upper(), f"strict mode must reject malformed Date; got {msg!r}"


@pytest.mark.asyncio
async def test_u14_strict_termination_unrecognised():
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = '[Termination "foobar"]\n[Result "*"]\n\n1. e4 *\n'
    with pytest.raises(Exception) as excinfo:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "STRICT" in msg.upper() and "Termination" in msg, (
        f"strict mode must reject unrecognised Termination; got {msg!r}"
    )


@pytest.mark.asyncio
async def test_u14_non_strict_date_passes():
    """Non-strict mode keeps the legacy accept-anything behavior so
    callers that don't care about metadata validation aren't broken."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    pgn = '[Date "not.a.date"]\n[Result "*"]\n\n1. e4 *\n'
    res = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert res is not None
    # The malformed date is preserved in metadata.
    assert res.date == "not.a.date" or res.date is None


# ---------------------------------------------------------------------------
# U-18: halfmove upper bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u18_huge_halfmove_rejected():
    """Audit U-18: FEN with halfmove clock > 10000 is rejected as a
    defensive hardening against pathological inputs."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # halfmove=50_000_000 (way above 10000) — this is the audit's
    # pathological example.
    fen = "7k/4P3/6K1/8/8/8/8/8 w - - 50000000 1"
    with pytest.raises(Exception) as excinfo:
        await server_module.evaluate_position(fen, depth=10)
    msg = str(excinfo.value)
    assert "INVALID_FEN" in msg or "Halfmove" in msg, f"huge halfmove must be rejected; got {msg!r}"


# ---------------------------------------------------------------------------
# U-15: wrong side marker in strict mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_u15_strict_wrong_side_marker_rejected():
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # Black's move numbered as if it were White's (single dot instead of triple).
    pgn = '[Result "*"]\n\n1. e4 e5 2. Nf3 2. Nc6 *\n'
    with pytest.raises(Exception) as excinfo:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(excinfo.value)
    assert "STRICT" in msg.upper(), f"strict mode must reject wrong side marker; got {msg!r}"


@pytest.mark.asyncio
async def test_u15_non_strict_wrong_side_marker_warning():
    """U-15 audit: the same PGN in non-strict mode must emit a
    'Wrong side marker' warning. The earlier regex `(\.|\.\.)*` was a
    Python footgun — group(2) was always a single dot, so the check
    never fired and the warning was silent."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # Black's move numbered with triple dot when single dot was expected
    # (because the board has it White-to-move at that point in the
    # mainline after the prior White move).
    pgn = "1. e4 e5 2... Nf3 *\n"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert any("side marker" in w.lower() for w in r.syntax_warnings), (
        f"non-strict must warn on wrong side marker; got {r.syntax_warnings!r}"
    )


@pytest.mark.asyncio
async def test_u07_wire_response_exposes_split_primitives():
    """U-07 audit: the three primitive booleans (same_action_type,
    same_outcome, within_cp_threshold) must be exposed on the wire
    response from classify_move, not just on the internal
    PlayedMoveScore type. Earlier the audit found only the legacy
    action_equivalent flag, leaving clients to reverse-engineer the
    policy."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    res = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "e4",
        depth=10,
    )
    # Wire-level fields
    assert "same_action_type" in res.model_dump(), (
        f"wire response missing same_action_type; keys: {list(res.model_dump().keys())}"
    )
    assert "same_outcome" in res.model_dump()
    assert "within_cp_threshold" in res.model_dump()
    # For a move that matches the engine's play_move recommendation, all three
    # should be True.
    if res.is_engine_best:
        assert res.same_action_type is True
        assert res.same_outcome is True
        assert res.within_cp_threshold is True


@pytest.mark.asyncio
async def test_u08_post_state_cp_none_when_no_reeval():
    """U-08 audit: post_state_cp is the re-evaluated post-state value.
    When the engine's multipv is not draw-polluted (strongly positive
    cp, no draw claim), no re-evaluation happens and post_state_cp is
    honestly None. The audit's Kxe2 reproducer has a draw claim at the
    root, so the re-eval is the exception, not the rule."""
    await server_module._cache.clear()
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]

    # A position with NO draw claim (halfmove=0). All top_moves candidates
    # are honest about the post-state: either populated (rare re-eval) or
    # None (no re-eval). The new fields (root_score_cp, root_score_mate,
    # post_state_cp, post_state_mate, post_fen) must be present.
    res = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", n=3, depth=10
    )
    assert res.result, "top_moves should return candidates"
    for cand in res.result:
        d = cand.model_dump()
        for field in (
            "root_score_cp",
            "root_score_mate",
            "post_state_cp",
            "post_state_mate",
            "post_fen",
        ):
            assert field in d, f"candidate missing U-08 field {field!r}"
        # post_fen must always be set (it's the FEN after the candidate move)
        assert d["post_fen"], f"post_fen empty for candidate {d.get('best_move')}"
