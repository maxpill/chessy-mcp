"""FEN counter validation (halfmove clock + fullmove number + EP / halfmove consistency).

:func:`validate_fen_counters` validates the halfmove / fullmove
counters and the EP / halfmove historical consistency check (audit
P1). Constants :data:`MAX_HALFMOVE_CLOCK` + :data:`MAX_FULLMOVE_NUMBER`
bound the accepted range.
"""

from __future__ import annotations

from typing import Final


MAX_HALFMOVE_CLOCK: Final[int] = 10_000
MAX_FULLMOVE_NUMBER: Final[int] = 10_000


def validate_fen_counters(cleaned: str, strict: bool) -> tuple[list[str], str]:
    """Validate halfmove clock + fullmove number + EP/halfmove historical consistency.

    Returns ``(tokens, cleaned_to_parse)``. Raises ``ValueError` on any
    invalid counter when strict=True, or when the value is unparseable /
    negative. Non-strict mode also raises on hard impossibilities
    (negative, unparseable, halfmove_clock > MAX) but permits non-historical
    EP + non-zero halfmove as a warning to the caller (which surfaces it
    via the metadata_warning channel rather than as an error).
    """
    tokens = cleaned.split()
    if len(tokens) >= 5:
        halfmove_raw = tokens[4]
        try:
            halfmove_num = int(halfmove_raw)
        except ValueError as exc:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' must be a valid integer."
            ) from exc
        if halfmove_num < 0:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' cannot be negative (got {halfmove_raw})."
            )
        if halfmove_num > MAX_HALFMOVE_CLOCK:
            raise ValueError(
                f"INVALID_FEN: Halfmove clock in FEN '{cleaned}' "
                f"is {halfmove_num}; maximum supported value is {MAX_HALFMOVE_CLOCK}."
            )
        if len(tokens) >= 4 and tokens[3] != "-" and halfmove_num != 0:
            ep_sq = tokens[3]
            raise ValueError(
                f"INVALID_FEN: FEN '{cleaned}' has en-passant target '{ep_sq}' "
                f"but halfmove clock is {halfmove_num}; an en-passant target "
                f"requires the previous move to have been a pawn double push "
                f"which would have reset the halfmove clock to 0."
            )
    if len(tokens) >= 6:
        fullmove_raw = tokens[5]
        try:
            fullmove_num = int(fullmove_raw)
        except ValueError as exc:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a valid integer."
            ) from exc
        if fullmove_num < 1:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' must be a positive integer >= 1 (got {fullmove_raw})."
            )
        if fullmove_num > MAX_FULLMOVE_NUMBER:
            raise ValueError(
                f"INVALID_FEN: Fullmove number in FEN '{cleaned}' "
                f"is {fullmove_num}; maximum supported value is {MAX_FULLMOVE_NUMBER}."
            )
    return tokens, cleaned


# Back-compat shim.
_validate_fen_counters = validate_fen_counters
