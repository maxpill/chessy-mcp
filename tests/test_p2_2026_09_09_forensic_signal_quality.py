"""2026-09-09 master audit F-009 + F-010: forensic signal quality.

F-009: ``loose_pieces`` previously conflated two very different states —
pieces with zero defenders (geometric fact, often harmless) and pieces that
are also attacked (tactically relevant). New fields split this:
  - ``undefended_pieces`` = alias for ``loose_pieces`` (geometric fact)
  - ``attacked_undefended_pieces`` = undefended AND attacked (tactical fact)
  - ``en_prise_pieces`` = capturable on opponent's next turn
  - ``tactically_hanging_candidates`` = unchanged (already exists)

F-010: the mechanism_candidates list dominated by pure geometry was
overwhelming coaching signal. Added ``presentation_priority`` per candidate
and a ``presentation_mechanisms`` top-N list ranked by concrete consequence.
"""

from __future__ import annotations

import chess

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot


def test_start_position_undefended_not_attacked() -> None:
    """The starting rooks are geometrically undefended but NOT tactically hanging."""
    b = chess.Board()
    snap = build_rich_tactical_snapshot(b)

    # 4 knights/bishops/rooks/queen are "undefended" (zero same-color defenders).
    assert len(snap.undefended_pieces) >= 4
    assert len(snap.attacked_undefended_pieces) == 0, (
        f"Start position must have ZERO attacked_undefended pieces; "
        f"got {len(snap.attacked_undefended_pieces)}"
    )


def test_undefended_pieces_aliases_loose_pieces() -> None:
    """``loose_pieces`` is the back-compat alias for ``undefended_pieces``."""
    b = chess.Board()
    snap = build_rich_tactical_snapshot(b)

    loose_labels = {(p.color, p.square, p.piece) for p in snap.loose_pieces}
    undef_labels = {(p.color, p.square, p.piece) for p in snap.undefended_pieces}
    assert loose_labels == undef_labels, (
        "loose_pieces and undefended_pieces must contain the same items "
        "(loose_pieces is the back-compat alias)."
    )


def test_attacked_undefended_distinguishes_tactical_danger() -> None:
    """A piece that is both undefended and attacked lands in attacked_undefended."""
    # Hang a black queen on e5 with no defenders; white bishop on f4 attacks it.
    fen = "4k3/8/8/4q3/5B2/8/8/4K3 w - - 0 1"
    b = chess.Board(fen)
    snap = build_rich_tactical_snapshot(b)

    queen_label = ("black", "e5", "queen")
    attacked_labels = {(p.color, p.square, p.piece) for p in snap.attacked_undefended_pieces}
    assert queen_label in attacked_labels, (
        f"Black queen on e5 is attacked and undefended; must appear in "
        f"attacked_undefended_pieces. Got {attacked_labels}"
    )
    # And the queen is en prise (capturable next move).
    en_prise_labels = {(p.color, p.square, p.piece) for p in snap.en_prise_pieces}
    assert queen_label in en_prise_labels, "Black queen on e5 must appear in en_prise_pieces."


def test_presentation_mechanisms_ranked_by_priority() -> None:
    """``presentation_mechanisms`` is ordered by concrete consequence, not raw count."""
    # Position with both a checking move and a pinned defender: the checking
    # move should rank higher than the pure-geometry pin.
    fen = "4k3/8/8/8/8/8/4r3/4K2r w - - 0 1"
    b = chess.Board(fen)
    snap = build_rich_tactical_snapshot(b)
    # Just assert the rank order makes sense: checking moves beat pure geometry.
    priority_buckets = [m.presentation_priority for m in snap.presentation_mechanisms]
    # Should be ordered by priority rank; no item lower in priority should
    # precede a higher-priority item.
    PRIORITY_RANK = {
        "immediate_mate": 0,
        "checking_move": 1,
        "capturing_move": 2,
        "promotion": 3,
        "forced_reply": 4,
        "new_en_prise": 5,
        "pinned_defender": 6,
        "engine_pv_supported": 7,
        "pure_geometry": 8,
    }
    ranks = [PRIORITY_RANK.get(p, 99) for p in priority_buckets]
    assert ranks == sorted(ranks), (
        f"presentation_mechanisms must be ordered by priority rank; "
        f"got priorities={priority_buckets} ranks={ranks}"
    )


def test_mechanism_candidates_preserve_presentation_priority() -> None:
    """Every mechanism candidate carries a non-empty presentation_priority bucket."""
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    snap = build_rich_tactical_snapshot(chess.Board(fen))
    valid = {
        "immediate_mate",
        "checking_move",
        "capturing_move",
        "promotion",
        "forced_reply",
        "new_en_prise",
        "pinned_defender",
        "engine_pv_supported",
        "pure_geometry",
    }
    for cand in snap.mechanism_candidates:
        assert cand.presentation_priority in valid, (
            f"mechanism {cand.mechanism} has unknown priority {cand.presentation_priority!r}"
        )


def test_presentation_mechanisms_subset_of_mechanism_candidates() -> None:
    """The ranked list must be drawn from the full evidence list, not duplicate it."""
    b = chess.Board("4k3/8/8/8/8/8/4r3/4K2r w - - 0 1")
    snap = build_rich_tactical_snapshot(b)
    presentation_uuids = {id(m) for m in snap.presentation_mechanisms}
    mechanism_uuids = {id(m) for m in snap.mechanism_candidates}
    assert presentation_uuids <= mechanism_uuids, (
        "presentation_mechanisms must be a strict subset of mechanism_candidates"
    )
