"""Phase 13 — 2026-09-14 ultra-audit repair: PGN tag resolver contract.

Root cause: the three call sites that walked ``[Tag "Value"]`` pairs each
had their own loop with implicit duplicate-tag semantics that disagreed:

    - ``Result`` — first occurrence won (``chess.pgn.read_game`` writes in
      source order so the last assignment survives).
    - ``FEN`` — last occurrence won (python-chess re-builds the root board
      from the last ``FEN`` it saw).
    - ``Variant`` — every occurrence was validated, so a single unsupported
      value anywhere poisoned the whole game.
    - ``SetUp`` — same as ``FEN`` because it gates the FEN setup path.

Fix: every call site now goes through :func:`resolve_pgn_tags` (a single
module living under ``mcp_server.contracts.pgn_resolver``). The resolver
returns deterministic :class:`ResolvedTag` records and:

    - lenient mode → ``"first"`` policy for **all** tags (so ``FEN`` no
      longer masks its own duplicates).
    - strict mode → ``StrictValidationError`` for conflicting duplicates of
      ``fen``/``setup``/``variant``/``result``.

This file pins the new contract end-to-end, including via ``analyze_game``.
"""

from __future__ import annotations

import chess
import pytest

from core.engines.types import Eval
from mcp_server import server as server_module
from mcp_server.contracts.errors import StrictValidationError
from mcp_server.contracts.pgn_resolver import (
    ResolvedTag,
    STRICT_VALIDATED_KEYS,
    resolve_pgn_tags,
)


# ---------------------------------------------------------------------------
# Helpers — Dummy engine pool for end-to-end analyze_game tests
# ---------------------------------------------------------------------------


class _DummyPool:
    """Minimal engine pool returning zero eval for any board."""

    name = "DummyPool"
    engine_version = "DummyPool"

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        return Eval(cp=0, best_move=None, pv=[], depth=depth)

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int = 14) -> list[Eval]:
        return []

    async def classify_move(self, board: chess.Board, move: chess.Move, depth: int = 14) -> Eval:
        return Eval(cp=0, best_move=None, pv=[], depth=depth)

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
async def _swap_pool():
    saved = server_module._analyzer_pool
    server_module._analyzer_pool = _DummyPool()  # type: ignore
    try:
        yield
    finally:
        server_module._analyzer_pool = saved  # type: ignore
        await server_module._cache.clear()


# ---------------------------------------------------------------------------
# Resolver unit tests — Result
# ---------------------------------------------------------------------------


def test_result_duplicate_same_value_lenient_first_wins_with_warning() -> None:
    text = '[Result "1-0"]\n[Result "1-0"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)

    assert "result" in resolved
    tag = resolved["result"]
    assert tag.occurrences == ("1-0", "1-0")
    assert tag.selected == "1-0"
    assert tag.policy == "first"
    assert tag.conflict is False
    assert tag.warning is not None
    assert "Duplicate" in tag.warning
    assert "selected first" in tag.warning


def test_result_duplicate_conflicting_values_lenient_first_wins() -> None:
    text = '[Result "1-0"]\n[Result "0-1"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)

    tag = resolved["result"]
    assert tag.selected == "1-0"
    assert tag.occurrences == ("1-0", "0-1")
    assert tag.conflict is True
    assert tag.warning is not None
    assert "1-0" in tag.warning and "0-1" in tag.warning
    assert "selected first" in tag.warning


def test_result_duplicate_conflicting_values_strict_raises() -> None:
    text = '[Result "1-0"]\n[Result "0-1"]\n\n1. e4 e5 *'
    with pytest.raises(StrictValidationError) as exc:
        resolve_pgn_tags(text, strict=True)
    assert "result" in str(exc.value).lower()
    assert "1-0" in str(exc.value) and "0-1" in str(exc.value)


# ---------------------------------------------------------------------------
# Resolver unit tests — FEN (the audit's headline fix)
# ---------------------------------------------------------------------------


def test_fen_duplicate_same_value_lenient_first_wins() -> None:
    """Pre-fix this silently took the last FEN via python-chess internals.

    Post-fix the resolver is deterministic: first occurrence wins, a warning
    is attached so the duplicate is observable.
    """
    text = (
        '[SetUp "1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        "\n1. Kd3 Kd1 *"
    )
    resolved = resolve_pgn_tags(text, strict=False)

    tag = resolved["fen"]
    assert tag.selected == "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"
    assert tag.occurrences == (
        "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
        "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
    )
    assert tag.policy == "first"
    assert tag.conflict is False
    assert tag.warning is not None
    assert "Duplicate" in tag.warning
    assert "selected first" in tag.warning


def test_fen_duplicate_conflicting_values_lenient_first_wins() -> None:
    text = (
        '[SetUp "1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n'
        "\n1. Kd3 Kd1 *"
    )
    resolved = resolve_pgn_tags(text, strict=False)

    tag = resolved["fen"]
    # The audit fix: first-wins even when the values conflict.
    assert tag.selected == "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"
    assert tag.occurrences == (
        "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    )
    assert tag.conflict is True
    assert tag.policy == "first"
    assert tag.warning is not None


def test_fen_duplicate_conflicting_values_strict_raises() -> None:
    text = (
        '[SetUp "1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n'
        "\n1. Kd3 Kd1 *"
    )
    with pytest.raises(StrictValidationError) as exc:
        resolve_pgn_tags(text, strict=True)
    assert "fen" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Resolver unit tests — Variant
# ---------------------------------------------------------------------------


def test_variant_lenient_with_conflict_first_wins() -> None:
    """Pre-fix a single unsupported Variant value poisoned the whole game.

    Post-fix lenient mode picks first and warns.
    """
    text = '[Variant "Crazyhouse"]\n[Variant "Standard"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)

    tag = resolved["variant"]
    assert tag.selected == "Crazyhouse"
    assert tag.occurrences == ("Crazyhouse", "Standard")
    assert tag.conflict is True
    assert tag.policy == "first"
    assert tag.warning is not None


def test_variant_strict_with_conflict_raises() -> None:
    text = '[Variant "Crazyhouse"]\n[Variant "Standard"]\n\n1. e4 e5 *'
    with pytest.raises(StrictValidationError) as exc:
        resolve_pgn_tags(text, strict=True)
    assert "variant" in str(exc.value).lower()
    assert "Crazyhouse" in str(exc.value)
    assert "Standard" in str(exc.value)


# ---------------------------------------------------------------------------
# Three-occurrence case — explicit "first-occurrence wins" pin
# ---------------------------------------------------------------------------


def test_three_occurrences_reversed_order_first_occurrence_wins() -> None:
    text = '[Result "1-0"]\n[Result "0-1"]\n[Result "1/2-1/2"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)

    tag = resolved["result"]
    assert tag.selected == "1-0"
    assert tag.occurrences == ("1-0", "0-1", "1/2-1/2")
    assert tag.policy == "first"
    assert tag.conflict is True
    assert tag.warning is not None
    # The warning must list all three occurrences.
    assert "1-0" in tag.warning
    assert "0-1" in tag.warning
    assert "1/2-1/2" in tag.warning


# ---------------------------------------------------------------------------
# Resolver edge cases — single tag, no duplicates, empty text
# ---------------------------------------------------------------------------


def test_single_tag_no_duplicates_no_warning() -> None:
    text = '[Result "1-0"]\n[White "Alice"]\n[Black "Bob"]\n\n1. e4 e5 1-0'
    resolved = resolve_pgn_tags(text, strict=False)

    assert resolved["result"].selected == "1-0"
    assert resolved["result"].warning is None
    assert resolved["result"].conflict is False
    assert resolved["white"].selected == "Alice"
    assert resolved["black"].selected == "Bob"


def test_empty_header_text_returns_empty_dict() -> None:
    assert resolve_pgn_tags("", strict=False) == {}
    assert resolve_pgn_tags("1. e4 e5 *", strict=False) == {}


def test_resolved_tag_is_frozen_dataclass() -> None:
    """Frozen dataclass so consumers cannot accidentally mutate the resolved view."""
    text = '[Result "1-0"]\n[Result "0-1"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)
    tag = resolved["result"]
    with pytest.raises((AttributeError, Exception)):
        tag.selected = "bogus"  # type: ignore[misc]


def test_strict_validated_keys_exposes_audit_contract() -> None:
    """The strict-mode sensitive keys must match the audit's contract."""
    assert STRICT_VALIDATED_KEYS == frozenset({"fen", "setup", "variant", "result"})


def test_resolved_tag_is_instance_of_dataclass() -> None:
    text = '[Result "1-0"]\n\n1. e4 e5 *'
    resolved = resolve_pgn_tags(text, strict=False)
    assert isinstance(resolved["result"], ResolvedTag)


# ---------------------------------------------------------------------------
# Strict mode same-value duplicates are tolerated (only conflicts raise)
# ---------------------------------------------------------------------------


def test_result_same_value_duplicate_strict_does_not_raise() -> None:
    text = '[Result "1-0"]\n[Result "1-0"]\n\n1. e4 e5 *'
    # Same value → no conflict → strict mode accepts (but still warns).
    resolved = resolve_pgn_tags(text, strict=True)
    tag = resolved["result"]
    assert tag.selected == "1-0"
    assert tag.conflict is False
    assert tag.warning is not None


# ---------------------------------------------------------------------------
# End-to-end — analyze_game must use the resolver path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_analyze_game_duplicate_fen_lenient_first_wins() -> None:
    """End-to-end: duplicate FEN tags in lenient mode → first FEN is selected.

    Pre-fix python-chess took the last FEN, so the resulting board was
    the start position and the movetext was bogus. Post-fix the resolver
    pins first-occurrence wins.
    """
    pgn = (
        '[SetUp "1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n'
        "\n1. Kd3 Kd1 *"
    )
    res = await server_module.analyze_game(pgn=pgn, depth=1)
    # First FEN was the Ke3-vs-ke3 endgame with Kd3 Kd1 as legal moves.
    # If python-chess had taken the second FEN (start position), Kd3 would
    # not have been a legal first move.
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_analyze_game_duplicate_fen_lenient_same_value_first_wins() -> None:
    """Duplicate-but-identical FEN must also be accepted deterministically."""
    fen = "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"
    pgn = f'[SetUp "1"]\n[FEN "{fen}"]\n[FEN "{fen}"]\n\n1. Kd3 Kd1 *'
    res = await server_module.analyze_game(pgn=pgn, depth=1)
    assert res.total_plies == 2


@pytest.mark.asyncio
async def test_analyze_game_duplicate_result_lenient_first_wins() -> None:
    """End-to-end: duplicate Result tags in lenient mode → first wins.

    The result header drives the analysis outcome. With conflicting Result
    tags and a movetext that does NOT end in a result token, the resolver
    must pick first so the caller's intent is honored deterministically.
    """
    pgn = '[White "A"]\n[Black "B"]\n[Result "1-0"]\n[Result "0-1"]\n\n1. e4 e5 *'
    res = await server_module.analyze_game(pgn=pgn, depth=1)
    assert res.white == "A"
    assert res.black == "B"
    # Resolved Result was "1-0" (first occurrence wins). Movetext has no
    # result token, so the analysis result must reflect the header.
    assert res.result_header == "1-0"


@pytest.mark.asyncio
async def test_analyze_game_duplicate_variant_lenient_first_wins() -> None:
    """End-to-end: first Variant is selected; second is recorded as duplicate."""
    pgn = '[Variant "Crazyhouse"]\n[Variant "Standard"]\n\n1. e4 e5 *'
    # In lenient mode the resolver picks "Crazyhouse" (first) — but the
    # downstream code still validates the resolved value and raises
    # UNSUPPORTED_VARIANT. The point of this test is that the FIRST value
    # is what flows through (not whichever the legacy loop happened to
    # inspect last). Either of these two messages is acceptable as long
    # as we know the FIRST value was the one rejected.
    with pytest.raises(Exception) as exc:
        await server_module.analyze_game(pgn=pgn, depth=1)
    msg = str(exc.value)
    assert "UNSUPPORTED_VARIANT" in msg or "unsupported" in msg.lower()
    assert "Crazyhouse" in msg


@pytest.mark.asyncio
async def test_analyze_game_duplicate_fen_strict_conflict_raises() -> None:
    """End-to-end strict mode: conflicting duplicate FENs raise."""
    pgn = (
        '[SetUp "1"]\n'
        '[FEN "8/8/8/8/8/4K3/4P3/4k3 w - - 0 1"]\n'
        '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]\n'
        "\n1. Kd3 Kd1 *"
    )
    with pytest.raises(Exception) as exc:
        await server_module.analyze_game(pgn=pgn, depth=1, strict=True)
    msg = str(exc.value)
    assert (
        "STRICT_VALIDATION_ERROR" in msg
        or "strict_validation_error" in msg.lower()
        or "Duplicate" in msg
    )
    assert "fen" in msg.lower()
