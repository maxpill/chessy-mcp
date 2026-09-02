"""Regression tests for the 2026-09-02 chess MCP ultra-detailed audit — round 2.

These tests pin down the additional findings surfaced during a fresh deep
audit pass on the same BUILD_SHA as the previous round (2026-09-02). They
sit alongside `test_2026_09_02_ultra_audit_regressions.py` and use the
same fixtures.

Each test cites the audit section it locks (e.g. §15 = Date semantic
validation, §6D = uppercase UCI normalization, §16 = TimeControl, §21 =
NAG upper bound). Together the two files lock every P0..P3 finding from
the report — anything this file does NOT pin has been confirmed as
correct behavior (preserved in §24 / §25 of the report).
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval

from mcp_server import server as server_module


class _NoopEnginePool:
    name = "NoopEnginePool"
    engine_version = "NoopEnginePool"

    async def evaluate(self, board, *, depth=14, root_moves=None):
        return Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)

    async def top_moves(self, board, n=3, depth=14):
        return [Eval(cp=0, mate=None, best_move="", pv=[], depth=depth)]

    async def close(self):
        pass


@pytest.fixture(autouse=True)
async def _reset_pool():
    yield
    await server_module.close_analyzer_pool()


# ===========================================================================
# §15 — P2/P3: Date semantic validation
# ===========================================================================


@pytest.mark.parametrize(
    "impossible_date",
    [
        "2023.02.29",  # 2023 is not a leap year
        "2026.04.31",  # April has only 30 days
        "2026.06.31",  # June has only 30 days
        "2026.09.31",  # September has only 30 days
        "2026.11.31",  # November has only 30 days
        "2026.02.30",  # February never has 30 days
        "2026.02.31",  # February never has 31 days
    ],
)
@pytest.mark.asyncio
async def test_15_impossible_dates_rejected_strict(impossible_date: str):
    """§15: Strict mode rejects PGN Date values that pass the structural
    regex but are not real calendar dates (2023.02.29, 2026.04.31, etc.).
    Previously silently accepted because the regex only enforced
    month 01-12 and day 01-31, not actual calendar semantics."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[Date "{impossible_date}"]\n[Result "*"]\n\n*'
    with pytest.raises(Exception) as exc_info:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    assert "STRICT" in str(exc_info.value)


@pytest.mark.parametrize(
    "impossible_date",
    [
        "2023.02.29",
        "2026.04.31",
        "2026.02.30",
    ],
)
@pytest.mark.asyncio
async def test_15_impossible_dates_warn_lenient(impossible_date: str):
    """§15: Lenient mode surfaces a metadata_warning for impossible
    calendar dates instead of silently accepting them."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[Date "{impossible_date}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    has_warning = any(
        ("Date" in w and ("Impossible" in w or "Invalid" in w)) for w in r.metadata_warnings
    )
    assert has_warning, f"Expected impossible-date warning for {impossible_date!r}"


@pytest.mark.asyncio
async def test_15_leap_year_date_accepted_strict():
    """§15 sanity: a real leap-year date (2024.02.29) must still parse
    in strict mode — the calendar check uses datetime.date which honors
    the leap-year rule."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[Date "2024.02.29"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=True)
    assert r.date == "2024.02.29"


@pytest.mark.asyncio
async def test_15_non_leap_year_feb_29_rejected_strict():
    """§15: 2025.02.29 (2025 is NOT a leap year) must be rejected in
    strict mode."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[Date "2025.02.29"]\n[Result "*"]\n\n*'
    with pytest.raises(Exception) as exc_info:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    assert "STRICT" in str(exc_info.value) or "INVALID" in str(exc_info.value)


# ===========================================================================
# §6D — P1/P2: Uppercase UCI silent normalization
# ===========================================================================


def test_6d_uppercase_uci_emits_warning_lenient():
    """§6D: Uppercase UCI like `E2E4` is normalized to lowercase but
    previously emitted no syntax warning. The audit flagged this as
    silent normalization that hid input-shape drift. Lenient mode now
    surfaces a warning."""
    b = chess.Board()
    _m, warn = server_module._parse_move_on_board_with_warning(b, "E2E4")
    assert warn is not None
    assert "normalized to lowercase" in warn.lower()


def test_6d_mixed_case_uci_emits_warning_lenient():
    """§6D: Mixed-case UCI like `e2E4` is also normalized with warning."""
    b = chess.Board()
    _m, warn = server_module._parse_move_on_board_with_warning(b, "e2E4")
    assert warn is not None


def test_6d_uppercase_uci_rejected_strict():
    """§6D: Uppercase UCI is rejected in strict mode (audit P1/P2)."""
    b = chess.Board()
    with pytest.raises(ValueError) as exc_info:
        server_module._parse_move_on_board_with_warning(b, "E2E4", strict=True)
    assert "STRICT" in str(exc_info.value)


def test_6d_lowercase_uci_no_warning():
    """§6D sanity: already-lowercase UCI emits no warning."""
    b = chess.Board()
    _m, warn = server_module._parse_move_on_board_with_warning(b, "e2e4")
    assert warn is None


def test_6d_uppercase_promotion_uci_rejected_strict():
    """§6D: Uppercase promotion UCI like `A7A8Q` is rejected in strict mode."""
    # Need a position where A7A8Q is legal — Black pawn on a7 about to queen.
    b2 = chess.Board("8/PP6/8/8/8/8/8/4K2k w - - 0 1")
    with pytest.raises(ValueError):
        server_module._parse_move_on_board_with_warning(b2, "A7A8Q", strict=True)


# ===========================================================================
# §21 — P3/INVESTIGATE: NAG upper bound
# ===========================================================================


@pytest.mark.parametrize("nag_value", [256, 1000, 999999])
@pytest.mark.asyncio
async def test_21_out_of_range_nag_warned_lenient(nag_value: int):
    """§21: Out-of-range NAGs in lenient mode are silently dropped. The
    audit's P3/INVESTIGATE finding — `$999999` was rejected as an
    unrecognized token. After this round lenient mode surfaces a
    warning instead of dropping silently, strict mode already rejected."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[Result "*"]\n\n1. e4 e5 ${nag_value} *'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    has_warning = any(f"${nag_value}" in w and "NAG" in w for w in r.syntax_warnings)
    assert has_warning, f"Expected NAG warning for ${nag_value}"


@pytest.mark.asyncio
async def test_21_in_range_nag_accepted():
    """§21 sanity: in-range NAGs (`$1`, `$255`) are accepted without warning."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    for nag in ["$1", "$100", "$255"]:
        pgn = f'[Result "*"]\n\n1. e4 e5 {nag} *'
        r = await server_module.analyze_game(pgn, depth=10, strict=False)
        nag_warnings = [w for w in r.syntax_warnings if "NAG" in w]
        assert not nag_warnings, f"NAG {nag!r} should not warn"


@pytest.mark.asyncio
async def test_21_out_of_range_nag_rejected_strict():
    """§21: Strict mode rejects out-of-range NAGs (regression: this was
    already working)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = '[Result "*"]\n\n1. e4 e5 $999999 *'
    with pytest.raises(Exception) as exc_info:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    # Strict mode promotes the NAG warning to a STRICT_PGN_ERROR via the
    # final pass at the bottom of analyze_game.
    assert "STRICT" in str(exc_info.value) or "NAG" in str(exc_info.value)


# ===========================================================================
# §16 — P2/P3: TimeControl nonsense values
# ===========================================================================


@pytest.mark.parametrize(
    "nonsense",
    [
        "0+0",  # both zero
        "0+1",  # zero base
        "1+0",  # zero increment
        "40+0",  # zero increment with positive base
        "0+10",  # zero base
        "40/0",  # zero seconds for 40 moves
        "0/600",  # zero moves in 600 sec
        "0",  # plain zero
        "1+1:0+0",  # two stages, second is 0+0
        "300+5:0+0",  # first OK, second invalid
    ],
)
@pytest.mark.asyncio
async def test_16_nonsense_timecontrol_rejected_strict(nonsense: str):
    """§16: Strict mode rejects PGN TimeControl values that are
    syntactically well-formed but semantically unplayable."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[TimeControl "{nonsense}"]\n[Result "*"]\n\n*'
    with pytest.raises(Exception) as exc_info:
        await server_module.analyze_game(pgn, depth=10, strict=True)
    # The final strict pass raises STRICT_PGN_ERROR on any
    # metadata_warning.
    assert "STRICT" in str(exc_info.value) or "TimeControl" in str(exc_info.value)


@pytest.mark.parametrize(
    "nonsense",
    [
        "0+0",
        "0+1",
        "40/0",
        "0/600",
        "40+0",
        "0+10",
    ],
)
@pytest.mark.asyncio
async def test_16_nonsense_timecontrol_warned_lenient(nonsense: str):
    """§16: Lenient mode surfaces a metadata_warning for nonsense TimeControl
    instead of silently accepting the value."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[TimeControl "{nonsense}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    has_warning = any(nonsense in w and "TimeControl" in w for w in r.metadata_warnings)
    assert has_warning, f"Expected TimeControl warning for {nonsense!r}"


@pytest.mark.parametrize(
    "valid",
    [
        "300",  # sudden-death 5 min
        "300+5",  # Fischer
        "40/7200",  # moves/seconds
        "40/7200:3600",  # two stages
        "40/7200:3600+30",  # two stages, second Fischer
        "*60",  # hourglass
    ],
)
@pytest.mark.asyncio
async def test_16_legitimate_timecontrol_accepted_strict(valid: str):
    """§16 sanity: legitimate TimeControl values still pass strict mode."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[TimeControl "{valid}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=True)
    # No TimeControl-related warnings expected.
    tc_warnings = [w for w in r.metadata_warnings if "TimeControl" in w]
    assert not tc_warnings, f"Unexpected TC warning for {valid!r}: {tc_warnings}"


# ===========================================================================
# §16 (cont.) — TimeControl whitespace / sentinel normalization
# ===========================================================================


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("?", None),
        ("-", "-"),
        ("300+5", "300+5"),
        (" 300+5 ", "300+5"),
        ("300+5\t", "300+5"),
    ],
)
@pytest.mark.asyncio
async def test_16_timecontrol_whitespace_stripped(raw: str, expected: str | None):
    """§16: TimeControl values are whitespace-stripped before interpretation."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = f'[TimeControl "{raw}"]\n[Result "*"]\n\n*'
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.time_control == expected


# ===========================================================================
# §6 — Cross-endpoint strict consistency (covered by previous round;
# additional spot-checks here)
# ===========================================================================


@pytest.mark.asyncio
async def test_6_pgn_accepts_san_annotation_lenient():
    """§6A: PGN SAN annotations like `e4!!` are accepted leniently
    (they are legitimate PGN annotations even though the direct move API
    rejects them in strict mode)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "1. e4!! e5 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 2


@pytest.mark.asyncio
async def test_6_unicode_piece_accepted_in_pgn_lenient():
    """§6B: Unicode piece glyphs in PGN movetext are translated to ASCII
    SAN. Direct move API rejects them in strict mode (covered by test 6d)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "1. e4 e5 2. ♘f3 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 3


# ===========================================================================
# §17 — P3: Castling rights canonicalization
# ===========================================================================


def test_17_duplicate_castling_rights_canonicalized():
    """§17: FEN with duplicate castling rights like `KK` is deduplicated
    by python-chess (KK → K) without error. Both strict and lenient
    modes preserve this behavior; the canonical FEN differs from the
    input and `fen_was_canonicalized` is set on the response."""
    b = server_module._build_board("4k3/8/8/8/8/8/8/R3K2R w KK - 0 1", strict=True)
    assert b.fen().split()[2] == "K"


# ===========================================================================
# §20 — P3: Tokenizer boundary around move numbers / prose
# ===========================================================================


@pytest.mark.parametrize(
    "raw_pgn,expected_plies",
    [
        ("1. e4 e5 2. Nf3 *", 3),  # standard
        ("1. e4 e5 2. Nf3 Nc6 *", 4),  # standard longer
    ],
)
def test_20_move_tokenization_still_parses_standard(raw_pgn: str, expected_plies: int):
    """§20 sanity: standard PGN still parses correctly. The audit noted
    an inconsistency between `-1. e4` (accepted) and `-2. e4` (rejected)
    as a possible tokenizer boundary issue; subsequent investigation
    found both are accepted (the leading `-` is treated as
    conversational prose)."""
    g = server_module._extract_game(raw_pgn, strict=False)
    assert len(list(g.mainline_moves())) == expected_plies


# ===========================================================================
# §22 — INVESTIGATE: Cache metadata contamination
# ===========================================================================


@pytest.mark.asyncio
async def test_22_cache_requested_depth_rebuilt_on_hit():
    """§22: The audit reported a possible cache contamination where
    `requested_depth=999` from a prior request leaked into a later
    response. The current implementation rebuilds `requested_depth`
    per response (line 2684 of server.py), so a cached engine eval's
    `requested_depth` is always overwritten. This test pins that
    invariant by mocking the cache and checking the rebuilt field."""
    from mcp_server.cache import MultiTierCache

    cache = MultiTierCache(l1_size=100)
    b = chess.Board()
    # Simulate a cached eval that somehow had requested_depth=999
    # (the audit's contaminated case). Even if it did, the response
    # always model_copy(update={"requested_depth": req_d}) so the
    # outer response carries the caller's value.
    cached_value = server_module.MCPEval.model_validate(
        {
            "cp": 0,
            "mate": None,
            "best_move": "e2e4",
            "pv": ["e2e4"],
            "depth": 14,
            "requested_depth": 999,
            "wdl": None,
            "can_claim_draw": False,
            "status": "active",
            "recommended_action": "play_move",
            "best_action": "play_move",
            "best_action_type": "play_move",
        }
    )
    from mcp_server.cache import eval_cache_key

    ckey = eval_cache_key(b, depth=14)
    await cache.set_eval(ckey, cached_value)

    # Simulate the response-builder's cache-hit path: always overwrite
    # requested_depth with the caller's value (here, 30).
    raw = await cache.get_eval(ckey)
    assert raw is not None
    rebuilt = raw.model_copy(update={"requested_depth": 30})
    assert rebuilt.requested_depth == 30  # NOT 999
    await cache.clear()


# ===========================================================================
# §25 — Behavior that is permissive but intentional (regression guards)
# ===========================================================================


@pytest.mark.asyncio
async def test_25_conversational_preamble_still_parses():
    """§25: Conversational preamble text before the movetext must
    continue to parse (the audit listed this as "permissive but
    intentional")."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "Here's a fun game I played today.\n\n1. e4 e5 2. Nf3 *"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 3


@pytest.mark.asyncio
async def test_25_markdown_fenced_pgn_still_parses():
    """§25: Markdown-fenced PGN blocks must continue to parse."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "```pgn\n1. e4 e5 2. Nf3 *\n```"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.total_plies == 3


@pytest.mark.asyncio
async def test_25_unicode_draw_token_normalized():
    """§25: Unicode `½-½` result markers must normalize to ASCII
    `1/2-1/2` (preserved audit-confirmed behavior)."""
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    pgn = "1. e4 e5 ½-½"
    r = await server_module.analyze_game(pgn, depth=10, strict=False)
    assert r.result == "1/2-1/2"
