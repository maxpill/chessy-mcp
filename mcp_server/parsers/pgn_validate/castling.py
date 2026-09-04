"""FEN castling-rights validation (audit U-09 + P3).

:func:`validate_castling_rights` validates the FEN castling rights
field symmetrically. Strict mode rejects tokens that reference
non-existent rooks. Non-strict mode silently strips invalid chars.
"""

from __future__ import annotations

import chess


def validate_castling_rights(board: chess.Board, rights_token: str, strict: bool, fen: str) -> str:
    """U-09 (2026-09-01): validate castling rights symmetrically.

    python-chess silently drops invalid castling rights (e.g. K with no
    white rook on h1) which masked the audit U-09 finding that the
    validator rejected some rights ("K" with no king) but accepted
    others ("Q" with no rook). The fix:

      * Strict mode: REJECT any token that contains a char referring to
        a non-existent rook of the matching color on its canonical square.
      * Non-strict mode: silently strip invalid chars and return a
        canonicalized token (still useful for callers that want
        continue-with-the-good-rights behavior).
      * X-FCR (e.g. "HAha") and Shredder-FEN are accepted as-is by
        python-chess; we don't second-guess the parser for them.

    P3 (2026-09-02 ultra audit): FEN's lexical spec requires each castling
    right to appear at most once in the rights field. Inputs like "KK",
    "QQ", "QK" therefore contain a duplicate that is not a real FEN.
    Both strict and non-strict mode preserve python-chess's deduplication
    behavior (the canonical rights field has at most one of each character
    anyway). The ``fen_was_canonicalized` flag in the response tells
    callers when this happened.
    """
    if not rights_token or rights_token == "-":
        return rights_token
    char_to_requirement: dict[str, tuple[chess.Color, chess.Square]] = {
        "K": (chess.WHITE, chess.H1),
        "Q": (chess.WHITE, chess.A1),
        "k": (chess.BLACK, chess.H8),
        "q": (chess.BLACK, chess.A8),
    }
    canonical_chars = list(dict.fromkeys(rights_token))
    valid: list[str] = []
    invalid: list[str] = []
    for ch in canonical_chars:
        req = char_to_requirement.get(ch)
        if req is None:
            valid.append(ch)
            continue
        color, sq = req
        piece = board.piece_type_at(sq)
        if piece == chess.ROOK and board.color_at(sq) == color:
            valid.append(ch)
        else:
            invalid.append(ch)
    if invalid:
        if strict:
            raise ValueError(
                f"INVALID_CASTLING_RIGHTS: FEN '{fen}' has castling "
                f"rights {rights_token!r} but the rook(s) for "
                f"{','.join(invalid)} are not on their canonical squares. "
                f"Rejected."
            )
        return "".join(valid) if valid else "-"
    # Bug fix (chessy-mcp-deep-audit §17): strict mode requires canonical
    # KQkq ordering and rejects duplicates. Non-strict mode canonicalizes
    # (dedup + reorder) silently. The four canonical chars in canonical
    # order are "KQkq" — anything not in that order, or any non-canonical
    # char (X-FCR / Shredder-FEN), is rejected in strict mode.
    canonical_order = "KQkq"
    if strict:
        non_canonical = [c for c in canonical_chars if c not in canonical_order]
        if non_canonical:
            raise ValueError(
                f"INVALID_CASTLING_RIGHTS: FEN '{fen}' has non-canonical "
                f"castling char(s) {','.join(non_canonical)!r}; "
                f"accepted chars are exactly {list(canonical_order)!r}."
            )
        expected = "".join(c for c in canonical_order if c in rights_token)
        if "".join(c for c in rights_token if c in canonical_order) != expected:
            raise ValueError(
                f"INVALID_CASTLING_RIGHTS: FEN '{fen}' has non-canonical "
                f"ordering {rights_token!r}; strict mode requires the order "
                f"{list(canonical_order)!r}."
            )
        return rights_token
    # Non-strict: canonicalize to KQkq order, dedup.
    canonicalized = "".join(c for c in canonical_order if c in rights_token)
    return canonicalized or rights_token


# Back-compat shim.
_validate_castling_rights = validate_castling_rights
