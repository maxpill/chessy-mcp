"""Round-4 regressions: trailing-junk FEN, comment-only PGN, strict
whitespace in move numbers, depth type/negative validation.

These tests pin down the additional findings surfaced during a fresh
super-deep audit pass beyond rounds 1/2/3. They sit alongside
test_2026_09_02_ultra_audit_regressions_round3.py and use the same fixtures.
"""

from __future__ import annotations

import asyncio

import pytest
from mcp.server.mcpserver.exceptions import ToolError

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
def _install_pool():
    server_module._analyzer_pool = _NoopEnginePool()  # type: ignore
    yield


def _run(coro):
    """Run an async coroutine in a fresh loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ===========================================================================
# R4-§A — Trailing-junk FEN: explicit INVALID_FEN (not INVALID_PGN)
# ===========================================================================


@pytest.mark.parametrize(
    "fen",
    [
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk",  # 7 fields (1 trailing)
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra more",  # 8 fields (2 trailing)
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1\n4k3/8/8/8/8/8/8/4K3",  # embedded newline
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_r4_trailing_junk_fen_rejected_with_invalid_fen(fen: str, strict: bool):
    """R4-§A: A FEN with > 6 whitespace-separated fields must be rejected
    with INVALID_FEN, not fall through to PGN parsing (which produces a
    misleading INVALID_PGN error saying 'Move token <fen> could not be
    parsed')."""
    with pytest.raises(ValueError) as exc_info:
        server_module._build_board(fen, strict=strict)
    msg = str(exc_info.value)
    assert "INVALID_FEN" in msg, f"Expected INVALID_FEN, got: {msg}"
    assert "INVALID_PGN" not in msg, f"Should not fall through to PGN: {msg}"
    assert "junk" in msg or "extra" in msg or "field" in msg.lower() or "tokens" in msg.lower()


def test_r4_trailing_junk_fen_in_pgn_header_rejected_with_invalid_fen():
    """R4-§A: The trailing-junk-FEN check applies inside a [FEN ...] header too."""
    pgn = '[SetUp "1"]\n[FEN "4k3/8/8/8/8/8/8/4K3 w - - 0 1 junk"]\n\n*'
    with pytest.raises(Exception) as exc_info:
        server_module._extract_game(pgn, strict=True)
    assert "INVALID_FEN" in str(exc_info.value)


def test_r4_exactly_six_field_fen_still_accepted():
    """R4-§A sanity: A real 6-field FEN must still pass."""
    fen = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
    b = server_module._build_board(fen, strict=True)
    assert b.fen() == fen


def test_r4_five_field_fen_still_completes():
    """R4-§A sanity: A 5-field FEN (no fullmove) still completes per §25
    audit behavior."""
    fen = "4k3/8/8/8/8/8/8/4K3 w - -"
    b = server_module._build_board(fen, strict=True)
    assert b.fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"


# ===========================================================================
# R4-§B — Comment-only PGN: accept with warning, do not reject
# ===========================================================================


@pytest.mark.parametrize(
    "pgn",
    [
        "{just a comment}",
        "{c1} {c2} {c3}",
        "{comment}\n\n*",
        "{only comment}",
    ],
)
def test_r4_comment_only_pgn_accepted_lenient(pgn: str):
    """R4-§B: A PGN consisting entirely of comments (with or without a
    result token) represents a zero-move game and must be accepted in
    lenient mode. Previously rejected with INVALID_PGN ('Move token
    {just could not be parsed') which is wrong because the input is a
    valid empty game."""

    async def _go():
        return await server_module.analyze_game(pgn, depth=10, strict=False)

    r = _run(_go())
    assert r.total_plies == 0


@pytest.mark.asyncio
async def test_r4_comment_only_pgn_emits_warning():
    """R4-§B: A comment-only PGN emits a clear metadata warning so callers
    know the input was non-empty but contained no moves."""
    r = await server_module.analyze_game("{just a comment}", depth=10, strict=False)
    assert r.total_plies == 0
    has_warning = any(
        "comment" in w.lower() and "no moves" in w.lower() for w in r.metadata_warnings
    )
    assert has_warning, f"Expected comment-only warning, got {r.metadata_warnings}"


def test_r4_comment_only_with_result_token_accepted():
    """R4-§B: A comment-only PGN that includes the result token '*' is
    valid and represents an empty game with an unfinished result."""
    pgn = "{c}\n*"
    r = _run(server_module.analyze_game(pgn, depth=10, strict=False))
    assert r.total_plies == 0
    assert r.result == "*"


# ===========================================================================
# R4-§C — Strict mode: whitespace around move numbers
# ===========================================================================


@pytest.mark.parametrize(
    "pgn,expected_plies",
    [
        ("1 .  e4  e5  *", 2),  # spaces around dot
        ("1 . e4 e5 *", 2),  # space before dot only
        ("1. e4 e5 2 .  Nf3 Nc6 *", 4),  # spaces around mid-game dot
    ],
)
def test_r4_strict_whitespace_around_move_numbers(pgn: str, expected_plies: int):
    """R4-§C: PGN §8.1 allows whitespace around the move number dot.
    The strict tokenizer splits on whitespace and treated '1' as a bare
    SAN token, failing to consume '1.' as a single move-number token.
    Round-4 fix: collapse 'N' followed by '.' (with optional whitespace)
    into 'N.' before tokenization."""

    async def _go():
        return await server_module.analyze_game(pgn, depth=10, strict=True)

    r = _run(_go())
    assert r.total_plies == expected_plies, (
        f"PGN {pgn!r}: expected {expected_plies} plies, got {r.total_plies}"
    )


def test_r4_strict_whitespace_does_not_collide_with_san():
    """R4-§C sanity: collapsing 'N' + '.' should only apply when 'N' is
    followed by exactly one or three dots, NOT when 'N' is followed by
    some other SAN-relevant token."""
    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"
    r = _run(server_module.analyze_game(pgn, depth=10, strict=True))
    assert r.total_plies == 6


# ===========================================================================
# R4-§D — Depth validation: reject non-int and negative depth cleanly
# ===========================================================================


@pytest.mark.parametrize("depth_value", [None, "abc", [], {}, 1.5])
def test_r4_depth_non_int_returns_invalid_input_for_analyze_game(depth_value):
    """R4-§D: analyze_game must return INVALID_INPUT (not a raw TypeError)
    when depth is not an integer."""

    async def _go():
        return await server_module.analyze_game("1. e4 e5 *", depth=depth_value)

    with pytest.raises(ToolError) as exc_info:
        _run(_go())
    assert "INVALID_INPUT" in str(exc_info.value) or "INVALID_DEPTH" in str(exc_info.value)


@pytest.mark.parametrize("depth_value", [None, "abc", [], {}, 1.5])
def test_r4_depth_non_int_returns_invalid_input_for_evaluate_position(depth_value):
    """R4-§D: evaluate_position must return INVALID_INPUT (not a raw
    TypeError) when depth is not an integer."""

    async def _go():
        return await server_module.evaluate_position("startpos", depth=depth_value)

    with pytest.raises(ToolError) as exc_info:
        _run(_go())
    assert "INVALID_INPUT" in str(exc_info.value) or "INVALID_DEPTH" in str(exc_info.value)


@pytest.mark.parametrize("depth_value", [None, "abc", [], {}, 1.5])
def test_r4_depth_non_int_returns_invalid_input_for_classify_move(depth_value):
    """R4-§D: classify_move must return INVALID_INPUT (not a raw TypeError)
    when depth is not an integer."""

    async def _go():
        return await server_module.classify_move("startpos", move="e4", depth=depth_value)

    with pytest.raises(ToolError) as exc_info:
        _run(_go())
    assert "INVALID_INPUT" in str(exc_info.value) or "INVALID_DEPTH" in str(exc_info.value)


@pytest.mark.parametrize("depth_value", [None, "abc", [], {}, 1.5])
def test_r4_depth_non_int_returns_invalid_input_for_top_moves(depth_value):
    """R4-§D: top_moves must return INVALID_INPUT (not a raw TypeError)
    when depth is not an integer."""

    async def _go():
        return await server_module.top_moves("startpos", depth=depth_value)

    with pytest.raises(ToolError) as exc_info:
        _run(_go())
    assert "INVALID_INPUT" in str(exc_info.value) or "INVALID_DEPTH" in str(exc_info.value)


@pytest.mark.parametrize("depth_value", [-1, -100])
def test_r4_depth_negative_preserves_raw_requested_for_analyze_game(depth_value):
    """R4-§D: analyze_game preserves the legacy `requested_depth` contract
    for negative depth — the raw caller value is preserved on the
    response while the engine-side depth is clamped to >= 1. This is the
    same behavior documented in test_mcp_new_11_requested_depth_semantics_
    clamping. Only the type check is new in R4-§D."""

    async def _go():
        return await server_module.analyze_game("1. e4 e5 *", depth=depth_value)

    r = _run(_go())
    assert r.requested_depth == depth_value


# ===========================================================================
# R4-§E — NUL character handling: strip silently
# ===========================================================================


def test_r4_nul_char_in_movetext_accepted_stripped():
    """R4-§E: NUL bytes embedded in the movetext are stripped before
    parsing. Previously rejected with INVALID_PGN 'Move token e5\\x00
    could not be parsed'."""
    pgn = '[Result "*"]\n\n1. e4 e5\x00 *'
    r = _run(server_module.analyze_game(pgn, depth=10, strict=False))
    assert r.total_plies == 2


def test_r4_nul_char_in_header_accepted_stripped():
    """R4-§E: NUL bytes in header values are stripped before parsing."""
    pgn = '[White "A\x00lice"]\n[Result "*"]\n\n1. e4 e5 *'
    r = _run(server_module.analyze_game(pgn, depth=10, strict=False))
    assert r.total_plies == 2
    # The NUL byte should be stripped from the white field
    assert r.white == "Alice"


# ===========================================================================
# R4-§F — Depth=0: clamp and report consistently
# ===========================================================================


@pytest.mark.parametrize("depth_value", [0])
def test_r4_depth_zero_preserves_raw_requested(depth_value):
    """R4-§F: analyze_game preserves the legacy `requested_depth` contract
    for depth=0 — the raw caller value is preserved on the response while
    the engine-side depth is clamped to 1."""

    async def _go():
        return await server_module.analyze_game("1. e4 e5 *", depth=depth_value)

    r = _run(_go())
    assert r.requested_depth == 0


# ===========================================================================
# R4-§G — Verbosity validation parity
# ===========================================================================


def test_r4_verbosity_invalid_value_returns_invalid_input():
    """R4-§G: Invalid verbosity values are rejected with INVALID_INPUT,
    not silently passed through."""

    async def _go():
        return await server_module.evaluate_position("startpos", verbosity="totally-unknown-mode")

    with pytest.raises(ToolError) as exc_info:
        _run(_go())
    assert "INVALID_VERBOSITY" in str(exc_info.value)
