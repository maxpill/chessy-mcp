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


def validate_fen_counters(cleaned: str, strict: bool) -> tuple[list[str], str, list[str]]:
    """Validate halfmove clock + fullmove number + EP/halfmove historical consistency.

    Returns ``(tokens, cleaned_to_parse, warnings)``.

    Strict mode raises on:
      * any unparseable counter,
      * negative halfmove clock / fullmove number,
      * counter exceeding :data:`MAX_HALFMOVE_CLOCK` / :data:`MAX_FULLMOVE_NUMBER`,
      * a non-historical EP target (EP square set with halfmove clock > 0).

    Lenient mode also raises on hard impossibilities (negative,
    unparseable, exceeding max) but **permits** a non-historical EP +
    non-zero halfmove by stripping the EP target to ``-`` and recording
    a structured warning (audit Bug 6, 2026-09-08). The halfmove counter
    is left untouched — callers can detect the canonicalization via
    the returned ``tokens`` or the canonical-FEN diff.
    """
    tokens = cleaned.split()
    warnings: list[str] = []
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
            if strict:
                raise ValueError(
                    f"INVALID_FEN: FEN '{cleaned}' has en-passant target '{ep_sq}' "
                    f"but halfmove clock is {halfmove_num}; an en-passant target "
                    f"requires the previous move to have been a pawn double push "
                    f"which would have reset the halfmove clock to 0."
                )
            tokens[3] = "-"
            cleaned = " ".join(tokens)
            warnings.append(
                f"EP target '{ep_sq}' stripped because halfmove clock is "
                f"{halfmove_num}; EP requires a prior pawn double push which "
                f"would have reset the clock to 0."
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
    return tokens, cleaned, warnings


# Back-compat shim.
_validate_fen_counters = validate_fen_counters
