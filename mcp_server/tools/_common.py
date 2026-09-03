"""Shared helpers for the MCP tools.

These were extracted from ``mcp_server.server`` to keep each tool file focused.
Re-exported through :mod:`mcp_server.server` so existing call sites and the
test suite keep working with their current import paths.
"""

from __future__ import annotations

import re
from typing import Any, Final

from mcp.server.mcpserver.exceptions import ToolError

__all__ = [
    "VERBOSITY_COMPACT",
    "VERBOSITY_FULL",
    "compact_mcpeval",
    "error_code_for",
    "format_exception",
    "normalize_termination",
    "resolve_verbosity",
    "tool_error",
    "validate_requested_depth",
]


# Audit M-05: `compact` strips Lichess URLs/images and decision_value/engine_eval
# duplication from every candidate, dropping payload size ~70% for LLM-driven
# callers that don't need URLs.
VERBOSITY_FULL: Final[str] = "full"
VERBOSITY_COMPACT: Final[str] = "compact"

_VERBOSITY_ALIASES: Final[dict[str, str]] = {
    "compact": "compact",
    "minimal": "compact",
    "min": "compact",
    "full": "full",
    "standard": "full",
    "default": "full",
}


def _format_exception(exc: BaseException) -> str:
    if isinstance(exc, BaseExceptionGroup):
        sub_msgs = [_format_exception(e) for e in exc.exceptions]
        return "; ".join(sub_msgs) if sub_msgs else str(exc)
    return str(exc)


def _tool_error(code: str, message: str | BaseException, tool: str, **kwargs: Any) -> ToolError:
    """Create a clean human/agent-readable ToolError payload."""
    raw = _format_exception(message) if isinstance(message, BaseException) else str(message)
    clean_msg = raw.strip()
    clean_msg = re.sub(r"^(?:\[[A-Za-z0-9_]+\]|[A-Za-z0-9_]+:)\s*", "", clean_msg).strip()
    return ToolError(f"[{code.upper()}] {clean_msg}")


# Public, unprefixed aliases used internally. The underscore-prefixed originals
# remain re-exported from ``mcp_server.server`` for backwards compatibility.
format_exception = _format_exception
tool_error = _tool_error


def _resolve_verbosity(value: Any) -> str:
    """Normalize verbosity and reject unknown values instead of silently changing semantics."""
    if value is None:
        return VERBOSITY_FULL
    normalized = str(value).strip().lower()
    resolved = _VERBOSITY_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"INVALID_VERBOSITY: expected one of {sorted(_VERBOSITY_ALIASES)}, got {value!r}"
        )
    return resolved


resolve_verbosity = _resolve_verbosity


def _compact_mcpeval(mcp_eval: Any) -> Any:
    """Strip verbose payload duplication without rewriting chess semantics."""
    return mcp_eval.model_copy(
        update={
            "lichess_url": None,
            "lichess_image": None,
            "decision_value": None,
            "engine_eval": None,
            "input_fen": None,
        }
    )


compact_mcpeval = _compact_mcpeval


def _validate_requested_depth(depth: Any, tool: str) -> int:
    """Validate the ``depth`` argument type for any analysis endpoint.

    R4-§D (2026-09-02 ultra audit round 4): previously non-integer ``depth``
    values raised a raw Python TypeError. The endpoint now rejects those with
    ``ToolError(INVALID_INPUT)`` so every endpoint behaves identically and
    callers get a structured error.
    """
    if not isinstance(depth, int) or isinstance(depth, bool):
        raise _tool_error(
            "INVALID_INPUT",
            f"depth must be a positive integer (got {type(depth).__name__}: {depth!r}).",
            tool,
        )
    return depth


validate_requested_depth = _validate_requested_depth


def normalize_termination(term: str | None) -> str | None:
    """Normalize PGN termination header string into standard taxonomy."""
    if not term:
        return None
    t = term.strip().lower()
    if re.search(r"\bnormal\b", t):
        return "normal"
    if re.search(r"\b(?:50|fifty)[-\s]*moves?(?:\s*rule)?\b", t):
        return "fifty_moves"
    if re.search(r"\b(?:75|seventy[-\s]*five)[-\s]*moves?(?:\s*rule)?\b", t):
        return "seventyfive_moves"
    if re.search(r"\b(?:5[-\s]*fold|five[-\s]*fold)(?:\s*repetition)?\b", t):
        return "fivefold_repetition"
    if re.search(
        r"\b(?:3[-\s]*fold|three[-\s]*fold)(?:\s*repetition)?(?:\s*claim)?\b|\bthreefold\b",
        t,
    ):
        return "threefold_repetition"
    if re.search(r"\brepetition\b", t):
        return "repetition"
    if re.search(r"\bcheckmate\b|\bmate\b", t):
        return "checkmate"
    if re.search(r"\binsufficient(?:\s*material)?\b", t):
        return "insufficient_material"
    if re.search(r"\bresign(?:ed|ation|s)?\b", t):
        return "resignation"
    if re.search(
        r"\btime\s*(?:forfeit|expired|exhausted|loss)\b"
        r"|\bout\s+of\s+time\b"
        r"|\bflag\s*(?:fell|fell|fall|dropped)\b"
        r"|\blost\s+on\s+time\b"
        r"|\b(?:white|black)\s+(?:wins?|won)\s+on\s+time\b"
        r"|\bclock\s+(?:flagged|expired)\b",
        t,
    ):
        return "time_forfeit"
    if re.search(r"\bunterminated\b|\bunfinished\b", t):
        return "unterminated"
    if re.search(r"\babandon(?:ed)?\b", t):
        return "abandoned"
    if re.search(r"\badjudicat(?:ed|ion)\b", t):
        return "adjudication"
    if re.search(r"\bdeath\b", t):
        return "death"
    if re.search(r"\bemergency\b", t):
        return "emergency"
    if re.search(
        r"\brules?\s+infraction\b|\b(?:second\s+)?illegal\s+move\b|\binfraction\b|\billegal\b",
        t,
    ):
        return "rules_infraction"
    if re.search(r"\bdraw\s+by\s+agreement\b|\bagreement\b", t):
        return "draw_agreement"
    return None


# Map ValueError-message markers to the structured error codes the four tool
# entry points raise. Replaces the four near-identical inline maps that were
# duplicated across evaluate_position, top_moves, classify_move, analyze_game.
_ERROR_CODE_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("INVALID_VERBOSITY", "invalid_verbosity"),
    ("INVALID_ACTION_TYPE", "invalid_action_type"),
    ("ILLEGAL_ACTION", "illegal_action"),
    ("STRICT", "strict_validation_error"),
    ("UNSUPPORTED_VARIANT", "unsupported_variant"),
    ("INVALID_FEN", "invalid_fen"),
    ("INVALID_POSITION", "invalid_position"),
    ("MULTIPLE_GAMES", "multiple_games_not_supported"),
    ("ILLEGAL_MOVE", "illegal_move"),
    ("AMBIGUOUS_SAN", "ambiguous_san"),
    ("GAME_ALREADY_OVER", "game_already_over"),
    ("INVALID_PGN", "invalid_pgn"),
    ("Could not parse PGN", "invalid_pgn"),
    ("Invalid PGN", "invalid_pgn"),
    ("MISSING_MOVE", "invalid_input"),
)


def error_code_for(message: str) -> str:
    """Infer the structured error code for a raised ``ValueError``.

    The four MCP tools wrap their inner ``ValueError`` into a ``ToolError``
    with a structured code. Centralizing this mapping keeps the four tools
    in sync — adding a new code is a one-line change here, not four.
    """
    for marker, code in _ERROR_CODE_PREFIXES:
        if marker in message:
            return code
    return "invalid_input"


error_code_for.__doc__ = error_code_for.__doc__
