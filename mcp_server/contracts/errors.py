"""Stable error taxonomy for the Chess MCP.

Each public error family is a distinct subclass of :class:`ChessMCPError`. The
class name is the source of truth; the ``code`` class attribute is the public
wire code that the four tools report.

This replaces the prior ``raise ValueError("INVALID_FEN: ..."`` + substring
matching table in :mod:`mcp_server.tools._common`. Old raise sites keep working
because the new helpers still accept strings (which the legacy ValueError
``error_code_for`` substring matcher handles), and every new raise site uses a
typed class so cross-tool consistency is enforced by Python's type system.

Phase 2 of the 2026-09-14 ultra-audit repair.
"""

from __future__ import annotations

from typing import Any, Final


class ChessMCPError(Exception):
    """Base class for all stable chess MCP errors.

    Subclasses must define ``code`` (the public wire code) and may carry
    structured context via keyword args. The ``message`` is the human-readable
    body; the wire format is ``[code] message (key=value ...)``.
    """

    code: str = "invalid_input"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = dict(context)


# --- input / argument validation -------------------------------------------


class InvalidInput(ChessMCPError):
    code = "invalid_input"


class InvalidArgument(ChessMCPError):
    code = "invalid_argument"


class SchemaValidationError(ChessMCPError):
    code = "schema_validation_error"


class InvalidDetail(ChessMCPError):
    code = "invalid_detail"


class InvalidVerbosity(ChessMCPError):
    code = "invalid_verbosity"


class InvalidActionType(ChessMCPError):
    code = "invalid_action_type"


# --- chess content errors ---------------------------------------------------


class InvalidFEN(ChessMCPError):
    code = "invalid_fen"


class InvalidPGN(ChessMCPError):
    code = "invalid_pgn"


class InvalidPosition(ChessMCPError):
    code = "invalid_position"


class InvalidPositionInput(ChessMCPError):
    code = "invalid_position_input"


class IllegalMove(ChessMCPError):
    code = "illegal_move"


class IllegalAction(ChessMCPError):
    code = "illegal_action"


class GameAlreadyOver(ChessMCPError):
    code = "game_already_over"


class PositionHistoryMismatch(ChessMCPError):
    code = "position_history_mismatch"


class AmbiguousSAN(ChessMCPError):
    code = "ambiguous_san"


# --- strictness / mode -----------------------------------------------------


class StrictValidationError(ChessMCPError):
    code = "strict_validation_error"


class UnsupportedVariant(ChessMCPError):
    code = "unsupported_variant"


class MultipleGamesNotSupported(ChessMCPError):
    code = "multiple_games_not_supported"


# --- engine / runtime ------------------------------------------------------


class EngineError(ChessMCPError):
    code = "engine_error"


# Map Python class -> wire code. Stable; legacy substring matching is kept as
# a fallback for any ValueError that has not yet been migrated.
_ERROR_CLASS_TO_CODE: Final[dict[type[ChessMCPError], str]] = {
    cls: cls.code for cls in ChessMCPError.__subclasses__()
}
for _sub in list(ChessMCPError.__subclasses__()):
    for _grand in _sub.__subclasses__():
        _ERROR_CLASS_TO_CODE[_grand] = _grand.code


# Stable marker prefixes recognized by the legacy substring matcher. Kept so
# un-migrated raise sites still map to the correct wire code.
_LEGACY_MARKERS: Final[tuple[tuple[str, type[ChessMCPError]], ...]] = (
    ("INVALID_VERBOSITY", InvalidVerbosity),
    ("INVALID_DETAIL", InvalidDetail),
    ("INVALID_ARGUMENT", InvalidArgument),
    ("SCHEMA_VALIDATION_ERROR", SchemaValidationError),
    ("INVALID_ACTION_TYPE", InvalidActionType),
    ("ILLEGAL_ACTION", IllegalAction),
    ("STRICT", StrictValidationError),
    ("UNSUPPORTED_VARIANT", UnsupportedVariant),
    ("INVALID_FEN", InvalidFEN),
    ("INVALID_POSITION_INPUT", InvalidPositionInput),
    ("INVALID_POSITION", InvalidPosition),
    ("MULTIPLE_GAMES", MultipleGamesNotSupported),
    ("INVALID_MOVE_SYNTAX", IllegalMove),
    ("INVALID_PARAMETER_COUNT", InvalidArgument),
    ("INVALID_COMPARE_MOVE", IllegalMove),
    ("ILLEGAL_MOVE", IllegalMove),
    ("AMBIGUOUS_SAN", AmbiguousSAN),
    ("GAME_ALREADY_OVER", GameAlreadyOver),
    ("POSITION_HISTORY_MISMATCH", PositionHistoryMismatch),
    ("INVALID_PGN", InvalidPGN),
    ("Could not parse PGN", InvalidPGN),
    ("Invalid PGN", InvalidPGN),
    ("MISSING_MOVE", InvalidInput),
)


def classify_error(exc: BaseException | str) -> tuple[str, str]:
    """Return ``(wire_code, clean_message)`` for any exception or message string.

    Class-first dispatch: if the exception is a :class:`ChessMCPError`, the
    class's ``code`` is returned directly. If it's a string (legacy ``ValueError``
    message), the legacy substring table is consulted. The fallback for
    unmatched strings is ``invalid_input`` (preserved from the previous
    implementation).
    """
    if isinstance(exc, ChessMCPError):
        return exc.code, exc.message
    if isinstance(exc, BaseException):
        return _classify_message(str(exc))
    return _classify_message(str(exc))


def _classify_message(message: str) -> tuple[str, str]:
    for marker, cls in _LEGACY_MARKERS:
        if marker in message:
            return cls.code, _strip_prefix(message)
    return InvalidInput.code, _strip_prefix(message)


def _strip_prefix(message: str) -> str:
    """Strip leading ``[code]`` or ``CODE:`` decoration from a message."""
    import re

    return re.sub(r"^(?:\[[A-Za-z0-9_]+\]|[A-Za-z0-9_]+:)\s*", "", message).strip()


def error_code_for(message: str) -> str:
    """Backwards-compatible helper — returns only the wire code."""
    code, _ = _classify_message(message)
    return code


__all__ = [
    "AmbiguousSAN",
    "ChessMCPError",
    "EngineError",
    "GameAlreadyOver",
    "IllegalAction",
    "IllegalMove",
    "InvalidActionType",
    "InvalidArgument",
    "InvalidDetail",
    "InvalidFEN",
    "InvalidInput",
    "InvalidPGN",
    "InvalidPosition",
    "InvalidPositionInput",
    "InvalidVerbosity",
    "MultipleGamesNotSupported",
    "PositionHistoryMismatch",
    "SchemaValidationError",
    "StrictValidationError",
    "UnsupportedVariant",
    "classify_error",
    "error_code_for",
]
