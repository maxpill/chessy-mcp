"""2026-09-08 ultra-hard test notes §4: candidate comparison provenance preservation.

For ``top_moves(... include_moves=['e2-e4'])`` the candidate comparison's
``requested`` field must round-trip the user's exact spelling. Before the
fix, ``_comparison_requests`` returned canonical SANs, so the response
silently dropped "e2-e4" → "e4" — losing the user's input spelling and
making strict-mode debugging harder.
"""

from __future__ import annotations

import chess

from mcp_server.analysis.top_moves_forensics import _comparison_requests
from mcp_server.models import MCPEval
from mcp_server.models.legacy import TopMovesResult


def test_explicit_alternative_requested_field_preserves_user_spelling() -> None:
    """§4: include_moves=['e2-e4'] must round-trip the original spelling."""
    board = chess.Board()
    result = TopMovesResult(result=[MCPEval(best_move="e2e4")])

    requested = _comparison_requests(
        result,
        board,
        detail="coach",
        include_moves=["e2-e4"],
    )

    assert len(requested) == 1
    canonical_san, original_text = requested[0]
    assert canonical_san == "e4"
    # The original user spelling must round-trip, not silently canonicalize.
    assert original_text == "e2-e4", (
        f"original_text must preserve the user's exact spelling; got {original_text!r}"
    )


def test_noncanonical_san_input_preserved() -> None:
    """§4: non-canonical SAN like 'Nf6?!' must also round-trip."""
    board = chess.Board()
    board.push(board.parse_san("e4"))
    board.push(board.parse_san("e5"))
    result = TopMovesResult(result=[MCPEval(best_move="g1f3")])

    requested = _comparison_requests(
        result,
        board,
        detail="coach",
        include_moves=["Nf3"],
    )

    assert len(requested) == 1
    canonical_san, original_text = requested[0]
    assert canonical_san == "Nf3"
    assert original_text == "Nf3"


def test_explicit_dedup_keeps_original_of_first_seen() -> None:
    """§4: dedupe is by canonical SAN. When duplicates are submitted, the first
    original text wins."""
    board = chess.Board()
    result = TopMovesResult(result=[MCPEval(best_move="e2e4")])

    requested = _comparison_requests(
        result,
        board,
        detail="coach",
        include_moves=["e2-e4", "e4"],  # both canonicalize to "e4"
    )

    assert len(requested) == 1
    canonical_san, original_text = requested[0]
    assert canonical_san == "e4"
    assert original_text == "e2-e4", f"first spelling seen wins dedupe; got {original_text!r}"


def test_engine_best_candidate_uses_canonical_san_as_original() -> None:
    """§4: when the engine-best is the only candidate, the original text is the canonical SAN itself."""
    board = chess.Board()
    result = TopMovesResult(result=[MCPEval(best_move="e2e4")])

    requested = _comparison_requests(
        result,
        board,
        detail="forensic",
        include_moves=None,
    )

    assert len(requested) >= 1
    canonical_san, original_text = requested[0]
    # For engine-derived candidates the canonical SAN IS the original.
    assert original_text == canonical_san
