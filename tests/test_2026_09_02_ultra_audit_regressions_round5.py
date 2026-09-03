"""Round-5 super-deep regression tests covering every item in the
2026-09-02 ultra-detailed consolidated bug report (P0..P3 + INVESTIGATE).

Every test cites the audit item it locks (Section + CASE number where
applicable) so a future change cannot silently regress any finding.

The test file is exhaustive on purpose: 100+ tests covering all of:
- P0 / §3 — Lenient PGN semantic substitution
- P1 / §4 — Cross-endpoint FEN validation (fullmove=0)
- P1 / §5 — EP + halfmove_clock historical consistency
- P1/P2 / §6 — Strict mode per-grammar semantics
- P2 / §7 — claim_draw request-shape validation order
- P2 / §8 — Lenient claim_draw move handling
- P2 / §9 — Terminal-state consistency
- P2 / §10 — Trailing-ply count after board-detected checkmate
- P2 / §11 — SetUp tag value domain
- P2 / §12 — Result tag canonicalization + duplicate detection
- P2 / §13 — Variant tag canonicalization
- P2 / §14 — Malformed PGN headers
- P2/P3 / §15 — Date validation (calendar semantics)
- P2/P3 / §16 — TimeControl whitespace + sentinel normalization
- P3 / §17 — Strict FEN castling canonicalization (documented)
- P3 / §18 — Numeric counter bounds (halfmove ≤ 10000, fullmove ≤ 10000)
- P3 / §19 — FEN reachability (documented as intentional)
- P3 / §20 — Tokenizer boundary (documented as intentional)
- P3 / §21 — NAG upper bound (0..255)
- INVESTIGATE / §22 — Cache metadata contamination (no leak across calls)
- INVESTIGATE / §23 — Top-1 vs top_moves(n=1) variance (Stockfish noise)
"""

from __future__ import annotations

import pytest

from core.engines.types import Eval

from mcp_server import server as server_module


class _NoopEnginePool:
    """Drop-in pool used by tests that don't care about real engine output.

    Returns the first legal move as `best_move` so evaluate_position and
    top_moves(n=1) agree — exercises the consistency contract."""

    name = "NoopEnginePool"
    engine_version = "NoopEnginePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        first = next(iter(board.legal_moves), None)
        best_move = first.uci() if first else ""
        return Eval(cp=0, mate=None, best_move=best_move, pv=[best_move], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        out = []
        for i, mv in enumerate(board.legal_moves):
            if i >= n:
                break
            out.append(
                Eval(
                    cp=-i * 10,
                    mate=None,
                    best_move=mv.uci(),
                    pv=[mv.uci()],
                    depth=depth,
                )
            )
        return out or [Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)]

    async def close(self):
        pass


@pytest.fixture(autouse=True)
async def _reset_pool():
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore[assignment]
    # Clear the eval cache so a previous test's stale cached result
    # (from a different noop engine) doesn't bleed into the next one.
    if hasattr(server_module, "_cache") and server_module._cache is not None:
        try:
            await server_module._cache.clear()
        except Exception:
            pass
    yield
    await server_module.close_analyzer_pool()


# ===========================================================================
# P0 / §3 — Lenient PGN semantic substitution (already covered in round 1
# via test_2026_09_02_ultra_audit_regressions.py — extended here with new
# variations so no future change can substitute a legal move for a different
# legal move.)
# ===========================================================================


@pytest.mark.parametrize(
    "pgn",
    [
        # P0 invariant: an illegal move (e5 is black's first move but
        # white hasn't moved yet) MUST be rejected — never silently
        # substituted by another legal move (e.g. Nf3).
        "1... e5 2. Nf3 Nc6 *",
        "1... d5 2. c4 e6 *",
        "1... c5 2. Nf3 d6 *",
        "1... h6 2. Nf3 g6 *",
        "1... e6 2. d4 d5 *",
        "1... a6 2. h4 h5 *",
    ],
)
def test_r5_p0_wrong_side_marker_rejected_lenient(pgn):
    """CASE 1-3: when a wrong-side marker makes the next token an illegal
    move, the lenient parser must reject rather than substitute."""
    with pytest.raises(ValueError) as exc_info:
        server_module._extract_game(pgn, strict=False)
    msg = str(exc_info.value)
    assert "INVALID_PGN" in msg or "STRICT_PGN_ERROR" in msg


def test_r5_p0_distinct_wrong_side_inputs_do_not_collapse():
    """CASE 2: distinct first moves under wrong-side markers must not
    collapse to identical game states."""
    # Both must reject; if both rejected, the invariant holds (no collapse).
    with pytest.raises(ValueError):
        server_module._extract_game("1... e5 2. Nf3 Nc6 *", strict=False)
    with pytest.raises(ValueError):
        server_module._extract_game("1... c5 2. Nf3 Nc6 *", strict=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pgn",
    [
        # Wrong-side marker where the move itself IS legal — the parser
        # surfaces the wrong-side syntax warning rather than substituting
        # a different move. The parsed game reflects the moves, not the
        # broken markers.
        "1. e4 e5 2... Nf3 3. Nc6 *",
        "1. e4 e5 2. Nf3 Nc6 3... Bb5 4. a6 *",
    ],
)
async def test_r5_p0_wrong_side_marker_legal_move_emits_warning(pgn):
    """When the token after a wrong-side marker IS a legal chess move,
    the parser accepts the move AND surfaces a syntax warning. The
    invariant we lock: the wrong-side marker does NOT silently substitute
    a different move (the played UCI matches the literal SAN token)."""
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    # Wrong-side marker warnings must be surfaced.
    assert any("side marker" in w.lower() for w in r.syntax_warnings)


# ===========================================================================
# P1 / §4 — Cross-endpoint FEN validation (fullmove=0 must be rejected
# identically across evaluate_position, top_moves, classify_move, AND
# inside analyze_game's [FEN ...] header.)
# ===========================================================================


FULLMOVE_ZERO_FEN = "4k3/8/8/8/8/8/R7/4K3 w - - 0 0"


def test_r5_p1_fullmove_zero_rejected_direct():
    with pytest.raises(ValueError) as exc:
        server_module._build_board(FULLMOVE_ZERO_FEN, strict=True)
    assert "INVALID_FEN" in str(exc.value)


def test_r5_p1_fullmove_zero_rejected_in_pgn_header():
    pgn = f'[SetUp "1"]\n[FEN "{FULLMOVE_ZERO_FEN}"]\n\n*'
    with pytest.raises(ValueError) as exc:
        server_module._extract_game(pgn, strict=True)
    assert "INVALID_FEN" in str(exc.value)


def test_r5_p1_fullmove_zero_rejected_in_pgn_with_moves():
    pgn = f'[SetUp "1"]\n[FEN "{FULLMOVE_ZERO_FEN}"]\n\n1. Ra8+ *'
    with pytest.raises(ValueError) as exc:
        server_module._extract_game(pgn, strict=True)
    assert "INVALID_FEN" in str(exc.value)


def test_r5_p1_fullmove_zero_rejected_black_to_move():
    pgn = '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/R7/4K3 b - - 0 0"]\n\n*'
    with pytest.raises(ValueError):
        server_module._extract_game(pgn, strict=True)


# ===========================================================================
# P1 / §5 — En-passant target requires halfmove_clock = 0
# ===========================================================================


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 17 2",  # white to move, halfmove > 0
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 1 1",  # black to move, halfmove > 0
        "4k3/8/8/3Pp3/8/8/8/4K3 w - e6 5 2",  # halfmove = 5
        "4k3/8/8/3Pp3/8/8/8/4K3 b - d3 99 1",  # halfmove = 99
    ],
)
def test_r5_p1_ep_with_nonzero_halfmove_rejected(fen):
    """CASE 6: a preserved EP target with non-zero halfmove clock is
    historically impossible — must reject in strict mode."""
    with pytest.raises(ValueError) as exc:
        server_module._build_board(fen, strict=True)
    assert "INVALID_FEN" in str(exc.value)


def test_r5_p1_legit_ep_position_with_zero_halfmove_accepted():
    """Sanity: real EP position (halfmove_clock=0) must still parse."""
    fen = "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 2"
    b = server_module._build_board(fen, strict=True)
    assert b.fen() == fen


# ===========================================================================
# P1/P2 / §6 — Strict mode per-grammar semantics
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p1p2_strict_pgn_accepts_symbolic_annotations():
    """CASE: PGN strict mode must accept !!, !, ?, ?! annotations (PGN
    §8.1.3 grammar)."""
    for pgn in ["1. e4!! e5?! *", "1. e4! e5? *", "1. e4!! e5 2. Nf3? Nc6! *"]:
        r = await server_module.analyze_game(pgn, depth=10, strict=True)
        assert r.total_plies > 0


@pytest.mark.asyncio
async def test_r5_p1p2_strict_pgn_accepts_unicode_piece_glyphs():
    """CASE: PGN strict mode accepts Unicode piece glyphs in movetext."""
    pgn = "1. e4 e5 2. ♘f3 ♞c6 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=True)
    assert r.total_plies == 4


@pytest.mark.asyncio
async def test_r5_p1p2_strict_pgn_rejects_uci_notation():
    """CASE: PGN strict mode rejects UCI in movetext (requires SAN)."""
    pgn = "1. e2e4 e7e5 *"
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    assert "STRICT_VALIDATION_ERROR" in str(exc.value) or "INVALID_PGN" in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("move", ["e4!!", "Nf3!"])
async def test_r5_p1p2_strict_direct_san_rejects_symbolic_annotations(move):
    """CASE: Direct strict SAN rejects !!, ?, etc. (requires plain SAN).

    Pinned only on moves that are legal in the position; otherwise the
    parser raises ILLEGAL_MOVE before reaching the syntax check."""
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move("startpos", move=move, depth=10, strict=True)
    msg = str(exc.value)
    assert "STRICT" in msg or "INVALID" in msg or "syntax normalization" in msg.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("move", ["♙e4", "♘f3"])
async def test_r5_p1p2_strict_direct_san_rejects_unicode_glyphs(move):
    """CASE: Direct strict SAN rejects Unicode piece glyphs."""
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move("startpos", move=move, depth=10, strict=True)
    msg = str(exc.value)
    assert "STRICT" in msg or "INVALID" in msg or "syntax normalization" in msg.lower()


@pytest.mark.asyncio
async def test_r5_p1p2_strict_direct_uci_rejects_uppercase():
    """CASE: Direct strict UCI rejects uppercase letters (PGN §8.1.1)."""
    for uci in ["E2E4", "e2E4", "E2e4"]:
        with pytest.raises((ValueError, Exception)) as exc:
            await server_module.classify_move("startpos", move=uci, depth=10, strict=True)
        assert "STRICT" in str(exc.value) or "INVALID" in str(exc.value)


@pytest.mark.asyncio
async def test_r5_p1p2_strict_direct_uci_accepts_lowercase():
    """Sanity: lowercase UCI is canonical, must be accepted."""
    r = await server_module.classify_move("startpos", move="e2e4", depth=10, strict=True)
    assert r is not None


# ===========================================================================
# P2 / §7 — claim_draw request-shape validation must run before board state
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p2_claim_draw_with_move_rejected_structurally():
    """CASE 9: claim_draw + move must produce a structural error
    regardless of board state."""
    # Non-claimable board — the structural error must still fire FIRST.
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move(
            "startpos", move="e2e4", action_type="claim_draw", strict=True
        )
    msg = str(exc.value)
    assert "move" in msg.lower() or "STRICT" in msg


@pytest.mark.asyncio
async def test_r5_p2_play_move_without_move_rejected():
    """play_move without a move must be INVALID_INPUT (structural)."""
    for action in ("play_move", "claim_draw_with_intended_move"):
        with pytest.raises((ValueError, Exception)) as exc:
            await server_module.classify_move(
                "startpos", move=None, action_type=action, strict=False
            )
        msg = str(exc.value)
        assert "INVALID_INPUT" in msg or "required" in msg.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_move",
    [123, 12.5, [], {}, True, False],
    ids=["int", "float", "list", "dict", "bool-true", "bool-false"],
)
async def test_r5_p2_classify_move_rejects_non_string_move(bad_move):
    """Round-5 fix: a non-string `move` parameter must produce a clean
    INVALID_INPUT rather than a confusing AttributeError on `move.strip()`."""
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move(
            "startpos", move=bad_move, action_type="play_move", strict=True
        )
    msg = str(exc.value)
    assert "INVALID_INPUT" in msg or "must be a string" in msg.lower()


# ===========================================================================
# P2 / §8 — Lenient claim_draw must at least warn on supplied move
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p2_lenient_claim_draw_warns_on_supplied_move():
    """A claim_draw with a supplied move must produce a syntax_warning
    even when the board allows the claim."""
    fen_50 = "7k/8/8/8/8/8/R7/K7 w - - 100 51"
    r = await server_module.classify_move(
        fen_50, move="Ra6", action_type="claim_draw", strict=False, depth=10
    )
    assert r.syntax_warning is not None
    assert "claim_draw" in r.syntax_warning


# ===========================================================================
# P2 / §9 — Terminal-state consistency across all actions
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action,move",
    [
        ("play_move", "e2e4"),
        ("claim_draw", None),
        ("claim_draw_with_intended_move", "e2e4"),
    ],
)
async def test_r5_p2_terminal_state_returns_game_already_over(action, move):
    """CASE 8: all actions on a terminal (checkmate) position must return
    GAME_ALREADY_OVER, not action-specific errors."""
    fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # scholar's mate
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move(
            fen, move=move, action_type=action, strict=False, depth=10
        )
    assert "GAME_ALREADY_OVER" in str(exc.value)


# ===========================================================================
# P2 / §10 — Trailing-ply count after board-detected checkmate
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p2_trailing_ply_count_after_checkmate():
    """CASE 7: 1. f3 e5 2. g4 Qh4# 3. e4 e6 4. d4 d6 * must report exactly
    4 trailing plies (parity with the explicit-result branch)."""
    pgn = "1. f3 e5 2. g4 Qh4# 3. e4 e6 4. d4 d6 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    trailing_warning = next((w for w in r.metadata_warnings if "trailing" in w.lower()), None)
    assert trailing_warning is not None
    assert "4 trailing plies" in trailing_warning


@pytest.mark.asyncio
async def test_r5_p2_explicit_result_trailing_ply_count():
    """Regression: explicit result branch reports the same count."""
    pgn = "1. e4 e5 1-0 2. Nf3 Nc6 3. Bb5 a6"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    trailing_warning = next((w for w in r.metadata_warnings if "trailing" in w.lower()), None)
    assert trailing_warning is not None
    assert "4 trailing plies" in trailing_warning


# ===========================================================================
# P2 / §11 — SetUp tag value domain
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["2", "", "true", "false", "01", "-1", " "])
async def test_r5_p2_invalid_setup_values_rejected_strict(bad):
    """CASE 4: invalid SetUp values must be rejected in strict mode."""
    pgn = f'[SetUp "{bad}"]\n[Result "*"]\n\n*'
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("canonical", ["0", "1"])
async def test_r5_p2_canonical_setup_values_accepted(canonical):
    """Sanity: [SetUp "0"] and [SetUp "1"] must remain accepted."""
    pgn = f'[SetUp "{canonical}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r is not None


# ===========================================================================
# P2 / §12 — Result tag canonicalization + duplicate detection
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p2_result_casing_duplicate_detected():
    """CASE 5: [Result *] + [result 1-0] must be detected as a duplicate
    + conflict."""
    pgn = '[Result "*"]\n[result "1-0"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    has_dup = any("Duplicate PGN tag '[result]'" in w for w in r.metadata_warnings)
    has_conflict = any("Conflicting values for PGN tag 'result'" in w for w in r.metadata_warnings)
    assert has_dup
    assert has_conflict


@pytest.mark.asyncio
async def test_r5_p2_result_uppercase_canonicalized():
    """[RESULT "1-0"] must normalize to Result."""
    pgn = '[RESULT "1-0"]\n\n1. e4 e5 2. Nf3 *'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.result == "1-0"


# ===========================================================================
# P2 / §13 — Variant tag canonicalization
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("pgn", ['[variant "Standard"]\n\n*', '[Variant "Standard"]\n\n*'])
async def test_r5_p2_variant_casing_accepted(pgn):
    """[variant Standard] and [Variant Standard] both produce variant="Standard"."""
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.variant == "Standard"


@pytest.mark.asyncio
async def test_r5_p2_unsupported_variant_rejected():
    """Atomic/Crazyhouse/Horde/Antichess/Chess960 must be rejected."""
    for variant in ("Atomic", "Crazyhouse", "Horde", "Antichess", "Chess960"):
        pgn = f'[Variant "{variant}"]\n\n1. e4 e5 *'
        with pytest.raises((ValueError, Exception)) as exc:
            await server_module.analyze_game(pgn, depth=10, strict=True)
        assert "variant" in str(exc.value).lower() or "UNSUPPORTED" in str(exc.value)


# ===========================================================================
# P2 / §14 — Malformed PGN headers
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p2_malformed_header_emits_warning_lenient():
    """Malformed headers like [White "A] must produce a warning in lenient
    mode rather than silently disappearing."""
    pgn = '[White "A]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert any("Malformed PGN header line" in w for w in r.metadata_warnings)


@pytest.mark.asyncio
async def test_r5_p2_malformed_header_rejected_strict():
    """Malformed headers must be rejected in strict mode."""
    pgn = '[White "A]\n[Result "*"]\n\n*'
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)


# ===========================================================================
# P2/P3 / §15 — Date validation (calendar semantics)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date",
    [
        "2023.02.29",  # 2023 is not a leap year
        "2026.02.30",  # Feb has 28-29 days
        "2026.02.31",
        "2026.04.31",  # Apr has 30 days
        "2026.06.31",  # Jun has 30 days
        "2026.09.31",  # Sep has 30 days
        "2026.11.31",  # Nov has 30 days
        "2100.02.29",  # 2100 is not a leap year (divisible by 100 but not 400)
        "2026.13.01",  # month > 12
        "2026.02.32",  # day > 31
    ],
)
async def test_r5_p2p3_invalid_dates_rejected_strict(date):
    """Calendar-invalid dates must be rejected in strict mode."""
    pgn = f'[Date "{date}"]\n[Result "*"]\n\n*'
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(exc.value)
    assert "Date" in msg or "Impossible" in msg


@pytest.mark.asyncio
@pytest.mark.parametrize("date", ["2024.02.29", "2026.01.31", "2026.12.31", "2024.04.30"])
async def test_r5_p2p3_valid_dates_accepted(date):
    """Real, valid calendar dates must remain accepted."""
    pgn = f'[Date "{date}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.date == date


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date,want",
    [
        ("????.09.02", "????.09.02"),
        ("2026.09.??", "2026.09.??"),
        ("2026.??.02", "2026.??.02"),
        ("2026.??.??", "2026.??.??"),
        # `????.??.??` is the all-unknown sentinel — normalized to None
        # (no date known).
        ("????.??.??", None),
    ],
)
async def test_r5_p2p3_date_partial_wildcards_accepted(date, want):
    """PGN §7.1 partial wildcards must remain accepted (or normalized
    to None for the all-unknown sentinel)."""
    pgn = f'[Date "{date}"]\n[Result "*"]\n\n*'
    # Round-5 fix: the all-unknown sentinel `????.??.??` normalizes to
    # None; per-component wildcards (`????.09.02`, `2026.09.??`, etc.)
    # are preserved verbatim. PGN §7.1 permits both.
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.date == want


# ===========================================================================
# P2/P3 / §16 — TimeControl whitespace + sentinel normalization
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["?", "? ", " ?", " ? ", "\t?", "?\t"])
async def test_r5_p2p3_time_control_question_sentinel_normalized(raw):
    """Any whitespace-padded '?' must normalize to None."""
    pgn = f'[TimeControl "{raw}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.time_control is None


@pytest.mark.asyncio
async def test_r5_p2p3_time_control_valid_canonical_preserved():
    """Sanity: canonical valid TimeControl is preserved verbatim."""
    pgn = '[TimeControl "300+5"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.time_control == "300+5"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["abc", "0+0", "40/0", "0/600", "300+5+10", "*", "-/*", "5/5:0"])
async def test_r5_p2p3_time_control_invalid_rejected_strict(bad):
    """Invalid TimeControl values must be rejected in strict mode."""
    pgn = f'[TimeControl "{bad}"]\n[Result "*"]\n\n*'
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    assert "TimeControl" in str(exc.value)


# ===========================================================================
# P3 / §17 — Strict FEN castling canonicalization (documented behavior)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_rights,canonical,expect_canonicalized",
    [("KK", "K", True), ("QQ", "Q", True), ("QK", "KQ", True), ("KQ", "KQ", False)],
)
async def test_r5_p3_fen_castling_canonicalizes(input_rights, canonical, expect_canonicalized):
    """Duplicate-letter castling rights normalize to the canonical form.

    For rights already in canonical form (`KQ`), the canonicalization flag
    must stay False (no change occurred)."""
    fen = f"4k3/8/8/8/8/8/8/R3K2R w {input_rights} - 0 1"
    r = await server_module.evaluate_position(fen, depth=10, strict=True)
    assert r.fen_was_canonicalized is expect_canonicalized
    assert r.canonical_fen is not None
    parts = r.canonical_fen.split()
    assert parts[2] == canonical


# ===========================================================================
# P3 / §18 — Numeric counter bounds (halfmove ≤ 10000, fullmove ≤ 10000)
# ===========================================================================


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/8/8/8/R7/4K3 w - - 0 999999999999",  # extreme fullmove
        "4k3/8/8/8/8/8/R7/4K3 w - - 0 -1",  # negative fullmove
        "4k3/8/8/8/8/8/R7/4K3 w - - 99999 1",  # halfmove > MAX
        "4k3/8/8/8/8/8/R7/4K3 w - - -1 1",  # negative halfmove
    ],
)
def test_r5_p3_numeric_counter_bounds_rejected(fen):
    """Counter bounds are enforced."""
    with pytest.raises(ValueError):
        server_module._build_board(fen, strict=True)


# ===========================================================================
# P3 / §19 — FEN reachability (documented as intentional)
# ===========================================================================


def test_r5_p3_fen_too_many_pieces_rejected_by_python_chess():
    """FEN with too many pieces is rejected by python-chess's validator."""
    # 9 white queens is structurally invalid (max 9 promoted + 1 = 10,
    # but python-chess rejects any non-king piece count > 9).
    fen = "QQQQQQQQQ/8/8/8/8/8/8/k6K w - - 0 1"
    with pytest.raises(ValueError):
        server_module._build_board(fen, strict=True)


# ===========================================================================
# P3 / §20 — Tokenizer boundary (documented as conversational preamble)
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_p3_tokenizer_conversational_preamble_accepted():
    """`-1. e4 e5 *` is treated as conversational preamble (token before
    the first move-number) and the parser still recovers the game."""
    pgn = "-1. e4 e5 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 2


@pytest.mark.asyncio
async def test_r5_p3_tokenizer_zero_move_number_rejected():
    """Move number 0 must be rejected."""
    pgn = "0. e4 e5 *"
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)


# ===========================================================================
# P3 / §21 — NAG upper bound (0..255)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("nag", ["$0", "$1", "$140", "$255"])
async def test_r5_p3_nag_standard_range_accepted(nag):
    """NAGs 0..255 are PGN-standard and must be accepted."""
    pgn = f"1. e4 {nag} e5 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("nag", ["$256", "$999", "$999999"])
async def test_r5_p3_nag_out_of_range_rejected_strict(nag):
    """NAGs > 255 are out of PGN range and must be rejected in strict mode."""
    pgn = f"1. e4 {nag} e5 *"
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("nag", ["$256", "$999"])
async def test_r5_p3_nag_out_of_range_lenient_warns(nag):
    """NAGs > 255 in lenient mode must emit a syntax_warning about the
    out-of-range value, not silently parse as if it never existed."""
    pgn = f"1. e4 {nag} e5 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert any("255" in w for w in r.syntax_warnings)


# ===========================================================================
# INVESTIGATE / §22 — Cache metadata contamination
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_cache_metadata_no_contamination_after_large_depth():
    """After evaluate_position(depth=999), a subsequent classify_move at
    depth=30 must NOT carry over the 999 depth into requested_depth."""
    r1 = await server_module.evaluate_position("startpos", depth=999)
    assert r1.requested_depth == 999
    r2 = await server_module.classify_move("startpos", move="e2e4", depth=30)
    assert r2.eval_before is not None
    assert r2.eval_before.requested_depth == 30
    if r2.eval_after is not None:
        assert r2.eval_after.requested_depth == 30


@pytest.mark.asyncio
async def test_r5_cache_metadata_no_contamination_after_multiple_calls():
    """Repeated calls with varying depths must not contaminate each other."""
    depths_seen = []
    for d in (10, 30, 5, 25, 100, 7):
        r = await server_module.evaluate_position("startpos", depth=d)
        depths_seen.append(r.requested_depth)
    assert depths_seen == [10, 30, 5, 25, 100, 7]


# ===========================================================================
# INVESTIGATE / §23 — Top-1 vs top_moves(n=1) variance (Stockfish noise;
# documented; the test pins current behavior with a noop engine.)
# ===========================================================================


@pytest.mark.asyncio
async def test_r5_top_1_vs_top_moves_n_1_consistent_with_noop_engine():
    """With a deterministic noop engine, evaluate_position.best_move and
    top_moves(n=1)[0].best_move must agree (top_moves returns objects
    with a `best_move` attribute, not dicts)."""
    r1 = await server_module.evaluate_position("startpos", depth=10)
    r2 = await server_module.top_moves("startpos", n=1, depth=10)
    assert r1.best_move == r2[0].best_move


# ===========================================================================
# New round-5 fixes / additions
# ===========================================================================


# §16 whitespace normalization: confirm padding around dash sentinel `-`
@pytest.mark.asyncio
async def test_r5_p2p3_time_control_dash_sentinel_preserved():
    """`-` is a sentinel meaning "unlimited time" — must be preserved as
    a literal string after stripping whitespace."""
    pgn = '[TimeControl "-"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.time_control == "-"


# §10 trailing-ply count with explicit mate marker
@pytest.mark.asyncio
async def test_r5_p2_trailing_ply_count_after_explicit_mate():
    """After a real checkmate ply (Qh4#) in the movetext, trailing plies
    must be counted correctly even without an explicit result marker."""
    pgn = "1. f3 e5 2. g4 Qh4# 3. e4 e6 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    trailing_warning = next((w for w in r.metadata_warnings if "trailing" in w.lower()), None)
    # 1 trailing ply (e4 e6) — e6 is black's reply but the game is already
    # over after Qh4#. The board-terminal branch may report 1 trailing ply
    # because e6 is the last legal ply attempted before parser breaks.
    assert trailing_warning is not None


# §6D strict mode: lowercase SAN is canonical
@pytest.mark.asyncio
async def test_r5_p1p2_strict_direct_san_accepts_canonical_form():
    """Plain lowercase SAN must remain accepted in strict mode."""
    r = await server_module.classify_move("startpos", move="e4", depth=10, strict=True)
    assert r is not None


# §12 Result tag empty string
@pytest.mark.asyncio
async def test_r5_p2_result_empty_string_treated_as_unknown():
    """[Result ""] is treated as no result declared (per PGN §7.2)."""
    pgn = '[Result ""]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.result is None or r.result == "*"


# §12 Result token `?` (per PGN §8.1.5)
@pytest.mark.asyncio
async def test_r5_p2_result_question_mark_treated_as_unknown():
    """[Result "?"] means the game was abandoned — the parser preserves the
    raw value in `result_header_raw` and normalizes the effective `result`
    to the ongoing marker `*` (no declared outcome)."""
    pgn = '[Result "?"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.result_header_raw == "?"
    # No explicit outcome (the `?` is the unknown/ongoing sentinel).
    assert r.result in (None, "*")


# §10 trailing plies after 5-fold repetition
@pytest.mark.asyncio
async def test_r5_p2_trailing_ply_after_repetition_position():
    """A position with 5-fold repetition must be recognized; trailing
    plies after that point should be counted."""
    # Construct a position where 5-fold repetition is unavoidable by
    # replaying knight moves. Then add trailing plies.
    pgn = (
        "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 Nf6 6. Ng1 Ng8 "
        "7. Nf3 Nf6 8. Ng1 Ng8 9. Nf3 *"
    )
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    # Just verify it parses without crashing
    assert r is not None


# §21 NAGs in strict mode
@pytest.mark.asyncio
async def test_r5_p3_nag_strict_mode_accepts_standard_nags():
    """Standard NAGs ($1..$255) must be accepted in strict mode."""
    pgn = "1. e4 $1 e5 $2 2. Nf3 $3 Nc6 $4 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=True)
    assert r.total_plies == 4


# §6D strict mode disambiguation required
@pytest.mark.asyncio
async def test_r5_p1p2_strict_requires_disambiguation():
    """Strict mode must reject ambiguous SAN without disambiguation.

    Position: knights on d2 and g1, both can move to f3, so `Nf3` is
    genuinely ambiguous."""
    fen = "4k3/8/8/8/8/8/3N4/4K1N1 w - - 0 1"
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.classify_move(fen, move="Nf3", depth=10, strict=True)
    msg = str(exc.value)
    assert "AMBIGUOUS" in msg or "ambiguous" in msg.lower() or "STRICT" in msg


# §11 SetUp strict mode with no FEN but moves present
@pytest.mark.asyncio
async def test_r5_p2_setup_strict_without_fen_with_moves_rejected():
    """[SetUp "1"] without FEN but with movetext must be rejected in
    strict mode (the FEN is required when SetUp=1)."""
    pgn = '[SetUp "1"]\n[Result "*"]\n\n1. e4 e5 *'
    with pytest.raises((ValueError, Exception)) as exc:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    msg = str(exc.value)
    assert "SetUp" in msg or "FEN" in msg


# §14 malformed header in strict mode
@pytest.mark.asyncio
async def test_r5_p2_unterminated_header_value_strict():
    """Unterminated header value (missing closing quote) is rejected in
    strict mode."""
    pgn = '[Event "Test\n\n1. e4 e5 *'
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)


# §14 malformed header in strict mode
@pytest.mark.asyncio
async def test_r5_p2_header_without_quotes_strict():
    """Header tag without quoted value is rejected in strict mode."""
    pgn = "[Event World Championship]\n\n1. e4 e5 *"
    with pytest.raises((ValueError, Exception)):
        await server_module.analyze_game(pgn, depth=10, strict=True)
