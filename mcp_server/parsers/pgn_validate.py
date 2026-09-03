"""PGN validation: variant, TimeControl, dates, castling, FEN counters.

Smaller cousin of :mod:`mcp_server.parsers.pgn_canonical`. Owns the
validators that reject semantically impossible PGN inputs (variants
beyond standard chess, zero-second TimeControl stages, malformed PGN
dates, bad FEN counters).

Underscored aliases preserved for backwards compatibility.
"""

from __future__ import annotations

import re
from typing import Final

import chess

# Bound the halfmove / fullmove counters we will accept in a FEN string.
# Anything beyond these is rejected at validation time so a malformed caller
# input can't make the engine spin forever or OOM the FEN parser.
MAX_HALFMOVE_CLOCK: Final[int] = 10_000
MAX_FULLMOVE_NUMBER: Final[int] = 10_000

__all__ = [
    "SUPPORTED_VARIANTS",
    "TIME_CONTROL_STAGE_RE",
    "validate_variant",
    "stage_has_positive_number",
    "is_valid_pgn_time_control",
    "validate_pgn_date",
    "validate_fen_counters",
    "validate_castling_rights",
]


SUPPORTED_VARIANTS: Final[frozenset[str | None]] = frozenset(
    {None, "", "standard", "from position"}
)

TIME_CONTROL_STAGE_RE: Final[re.Pattern[str]] = re.compile(r"^(?:\d+|\d+/\d+|\d+\+\d+|\*\d+)$")
_TIME_CONTROL_STAGE_RE = TIME_CONTROL_STAGE_RE


def validate_variant(variant: str | None) -> None:
    if variant is not None and variant.strip().lower() not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"UNSUPPORTED_VARIANT: Variant '{variant.strip()}' is not supported. Chess MCP currently analyzes standard chess only."
        )


def stage_has_positive_number(stage: str) -> bool:
    """Return True iff every numeric half of `stage` is strictly positive.

    Splits the stage on `/`, `+`, or `*` (the three PGN TimeControl
    separators — `/` for moves/seconds, `+` for Fischer increment,
    `*` for hourglass prefix) and requires every individual numeric
    component to contain a non-zero digit. This rejects the
    syntactically-PGN-shaped but semantically impossible values the
    2026-09-02 audit flagged:

        "0+0"     -> both components are zero (unplayable game)
        "0+1"     -> 0-second base (unplayable)
        "40+0"    -> 0-second increment on a 40-second base
        "0/600"   -> 0-move period with 600 seconds (nonsensical)
        "40/0"    -> 0 seconds for a 40-move period (impossible)

    while keeping the legitimate values intact:

        "300"     -> single value, positive
        "300+5"   -> both components positive
        "40/7200" -> both components positive
        "*60"     -> hourglass prefix + positive value
    """
    # Strip the optional hourglass prefix "*" before splitting.
    body = stage[1:] if stage.startswith("*") else stage
    # Split on the two multi-component separators; the leading-digit
    # check then guards each piece. An empty piece (e.g. "+0" -> split
    # would yield ["", "0"]) is treated as zero and rejected.
    pieces: list[str] = []
    if "+" in body:
        pieces.extend(body.split("+"))
    elif "/" in body:
        pieces.extend(body.split("/"))
    else:
        pieces.append(body)
    for piece in pieces:
        if not piece or not any(c in "123456789" for c in piece):
            return False
    return True


def is_valid_pgn_time_control(value: str) -> bool:
    """Validate the PGN TimeControl tag grammar.

    PGN permits a single stage or colon-separated stages. A stage is one of:
    sudden-death seconds (``300``), moves/seconds (``40/7200``), Fischer
    seconds+increment (``300+5``), or hourglass (``*60``). ``?`` and ``-``
    are the standard unknown/unspecified markers.

    Every numeric component must contain at least one non-zero digit — a
    stage of "0", "0+0", "0+1", "40+0", "0/600", or "40/0" cannot
    describe a real chess game and is rejected (audit P2/P3,
    2026-09-02).
    """
    text = value.strip()
    if text in {"?", "-"}:
        return True
    if not text:
        return False
    stages = text.split(":")
    return all(
        bool(stage)
        and _TIME_CONTROL_STAGE_RE.fullmatch(stage) is not None
        and _stage_has_positive_number(stage)
        for stage in stages
    )


def validate_pgn_date(date_val: str) -> str | None:
    """Validate a PGN Date tag value. Returns an error message or None.

    PGN §7.1 allows `????` for unknown year and `??` for unknown month /
    day. Each component is validated independently; calendar semantics
    (Apr 31, Sep 31, Feb 29 in non-leap year) only run when all three
    components are concrete. Caller is responsible for normalizing
    sentinel values (empty, '?', '????.??.??') to None before calling.
    """
    parts = date_val.split(".")
    if len(parts) != 3:
        return (
            f"Invalid Date tag '{date_val}': must match YYYY.MM.DD "
            f"(with ? wildcards allowed per component)."
        )

    year_str, month_str, day_str = parts

    if year_str == "????":
        year: int | None = None
    elif re.fullmatch(r"(?:19|20)\d{2}", year_str):
        year = int(year_str)
    else:
        return f"Invalid Date tag '{date_val}': year must be 19xx/20xx or '????' for unknown."

    if month_str == "??":
        month: int | None = None
    elif re.fullmatch(r"(?:0[1-9]|1[0-2])", month_str):
        month = int(month_str)
    else:
        return f"Invalid Date tag '{date_val}': month must be 01-12 or '??' for unknown."

    if day_str == "??":
        day: int | None = None
    elif re.fullmatch(r"(?:0[1-9]|[12]\d|3[01])", day_str):
        day = int(day_str)
    else:
        return f"Invalid Date tag '{date_val}': day must be 01-31 or '??' for unknown."

    if year is not None and month is not None and day is not None:
        try:
            import datetime as _dt

            _dt.date(year, month, day)
        except ValueError as exc:
            return f"Impossible Date tag '{date_val}': {exc}."

    return None


def validate_fen_counters(cleaned: str, strict: bool) -> tuple[list[str], str]:
    """Validate halfmove clock + fullmove number + EP/halfmove historical consistency.

    Returns (tokens, cleaned_to_parse). Raises ValueError on any invalid counter
    when strict=True, or when the value is unparseable / negative. Non-strict mode
    also raises on hard impossibilities (negative, unparseable, halfmove_clock > MAX)
    but permits non-historical EP + non-zero halfmove as a warning to the caller
    (which surfaces it via the metadata_warning channel rather than as an error).
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
        # P1 (2026-09-02 ultra audit): an en-passant target requires that
        # the previous move was a pawn double push — which resets the
        # halfmove clock to 0. A non-zero halfmove_clock alongside an
        # EP target square is historically impossible. Reject it in
        # strict mode; lenient mode also rejects (this is not a
        # lexical quirk the parser should silently canonicalize —
        # the input is contradictory at the level of chess semantics).
        if len(tokens) >= 4 and tokens[3] != "-" and halfmove_num != 0:
            ep_sq = tokens[3]
            # A pawn double push is the only move that produces an EP
            # target AND resets the halfmove clock, so halfmove_clock
            # MUST be 0 in any FEN that preserves an EP target. We
            # don't need to reconstruct which side moved last from the
            # FEN alone — the combination is simply contradictory at
            # the level of chess semantics.
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


def validate_castling_rights(board: chess.Board, rights_token: str, strict: bool, fen: str) -> str:
    """U-09 (2026-09-01): validate castling rights symmetrically.

    python-chess silently drops invalid castling rights (e.g. K with no
    white rook on h1) which masked the audit U-09 finding that the
    validator rejected some rights ("K" with no king) but accepted
    others ("Q" with no rook). The fix:
      - Strict mode: REJECT any token that contains a char referring to
        a non-existent rook of the matching color on its canonical square.
      - Non-strict mode: silently strip invalid chars and return a
        canonicalized token (still useful for callers that want
        continue-with-the-good-rights behavior).
      - X-FCR (e.g. "HAha") and Shredder-FEN are accepted as-is by
        python-chess; we don't second-guess the parser for them.

    Returns the validated (possibly empty / canonicalized) rights token.

    P3 (2026-09-02 ultra audit): FEN's lexical spec requires each castling
    right to appear at most once in the rights field. Inputs like "KK",
    "QQ", "QK" therefore contain a duplicate that is not a real FEN.
    python-chess's constructor deduplicates ("KK" → "K") without
    signaling the caller. Both strict and non-strict mode preserve
    that behavior (dedup is harmless: the canonical rights field has
    at most one of each character anyway). The `fen_was_canonicalized`
    flag in the response tells callers when this happened.
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
    return rights_token


# Underscored aliases for backwards-compatible import paths.
_validate_castling_rights = validate_castling_rights
_validate_fen_counters = validate_fen_counters
_validate_pgn_date = validate_pgn_date
_is_valid_pgn_time_control = is_valid_pgn_time_control
_stage_has_positive_number = stage_has_positive_number
_validate_variant = validate_variant
