"""2026-09-08 ultra-hard test notes §2: continuation resolution invariant.

For the same continuation, ``candidate_comparisons[].continuation_
tactical_sequence_resolved`` and the nested
``continuation_endpoint.tactical_sequence_resolved`` MUST agree.

Before the fix, ``_principal_line`` (in ``analysis/forensics.py``) used
"final position is quiet" alone (resolved=True on quiet endpoint even
when no forcing was seen during the walk), while ``_walk_endpoint`` (in
``analysis/candidate_continuation.py``) required ``forcing_seen`` first.
The audit note pinned the exact reproduction: ``top_moves(startpos,
include_moves=['e2-e4'], depth=2)`` reports ``tactical_sequence_resolved
= true`` on the candidate comparison but ``= false`` on the nested
continuation endpoint for the same continuation.

The shared helper ``tactical_sequence_resolved(forcing_seen, board)``
in ``analysis/tactical_continuation_resolved.py`` pins the single
semantic: forcing seen during the walk AND final position quiet.
"""

from __future__ import annotations

import chess

from mcp_server.analysis.tactical_continuation_resolved import (
    tactical_sequence_resolved,
)


def test_resolved_false_when_no_forcing_seen() -> None:
    """Quiet walk with no forcing must NOT report resolved — engine continuation only."""
    # K vs K endgame far apart: no forcing moves available, no forcing seen.
    final = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert tactical_sequence_resolved(forcing_seen=False, final_board=final) is False


def test_resolved_true_when_forcing_seen_and_quiet_end() -> None:
    """Forcing seen + quiet end → resolved."""
    # K vs K — kings far apart. No forcing moves available.
    final = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert tactical_sequence_resolved(forcing_seen=True, final_board=final) is True


def test_resolved_false_when_forcing_seen_but_check_persists() -> None:
    """Forcing seen + final position still in check → not resolved yet."""
    # Position where the side to move is in check from a rook on the
    # adjacent rank. Construct programmatically so we don't rely on SAN
    # parsing of complex sequences.
    final = chess.Board.empty()
    # Place White king on a1, Black rook on a2 (giving check).
    final.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.WHITE))
    final.set_piece_at(chess.A2, chess.Piece(chess.ROOK, chess.BLACK))
    final.turn = chess.WHITE
    final.castling_xfen = lambda: "-"  # type: ignore[assignment]
    assert final.is_check(), "White king must be in check from rook on a2"
    assert tactical_sequence_resolved(forcing_seen=True, final_board=final) is False


def test_resolved_false_when_forcing_seen_but_forcing_available() -> None:
    """Forcing seen + final position has another forcing move → not resolved."""
    # King's Indian opening: a capture (Nxe5) is still available.
    final = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    assert tactical_sequence_resolved(forcing_seen=True, final_board=final) is False


def test_shared_helper_used_by_principal_line() -> None:
    """Regression guard: ``_principal_line`` must import the shared helper."""
    from mcp_server.analysis import forensics as forensics_module

    src = open(forensics_module.__file__).read()
    assert "tactical_continuation_resolved" in src, (
        "_principal_line must import tactical_sequence_resolved from "
        "the shared module to keep the cross-helper invariant."
    )


def test_shared_helper_used_by_walk_endpoint() -> None:
    """Regression guard: ``_walk_endpoint`` must use the shared helper."""
    from mcp_server.analysis import candidate_continuation as cc_module

    src = open(cc_module.__file__).read()
    assert "tactical_sequence_resolved(forcing_seen, work)" in src, (
        "_walk_endpoint must call the shared tactical_sequence_resolved helper."
    )
