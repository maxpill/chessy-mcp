"""Public contract constants for the Chess MCP server.

Single source of truth for every numeric bound, enum literal, and verbosity
vocabulary exposed by the four MCP tools. Schemas, runtime clamps, request-cost
estimators, and regression tests all import from here so the public contract
cannot drift again.

Phase 1 of the 2026-09-14 ultra-audit repair.
"""

from __future__ import annotations

from typing import Final, Literal

# --- top_moves.n -----------------------------------------------------------
TOP_MOVES_MIN_N: Final[int] = 1
TOP_MOVES_MAX_N: Final[int] = 20
TOP_MOVES_DEFAULT_N: Final[int] = 3

# --- Stockfish depth ------------------------------------------------------
DEPTH_MIN: Final[int] = 1
DEPTH_MAX: Final[int] = 30
DEPTH_DEFAULT_EVALUATE: Final[int] = 20
DEPTH_DEFAULT_CLASSIFY: Final[int] = 16
DEPTH_DEFAULT_ANALYZE_GAME: Final[int] = 18

# --- candidate / compare / include limits ---------------------------------
MAX_INCLUDE_MOVES: Final[int] = 8
MAX_COMPARE_MOVES: Final[int] = 8

# --- proof defenses (strict bounded — outside 1..8 is rejected) -----------
PROOF_DEFENSES_MIN: Final[int] = 1
PROOF_DEFENSES_MAX: Final[int] = 8
PROOF_DEFENSES_DEFAULT: Final[int] = 3

# --- analyze_game critical moments (clamped to 1..7) ----------------------
MAX_CRITICAL_MOMENTS: Final[int] = 7

# --- FEN counter ceilings (deliberate service safety limit) ---------------
MAX_HALFMOVE_CLOCK: Final[int] = 10_000
MAX_FULLMOVE_NUMBER: Final[int] = 10_000
DEFAULT_FULLMOVE_NUMBER: Final[int] = 1

# --- canonical / accepted vocabularies ------------------------------------
ALLOWED_DETAIL_LEVELS: Final[tuple[str, ...]] = ("standard", "coach", "forensic")
DetailLevel = Literal["standard", "coach", "forensic"]

# Verbosity: canonical values + accepted aliases. Same model as the prior
# local map, but now centralized so all four tools share one set.
ALLOWED_VERBOSITY_CANONICAL: Final[tuple[str, ...]] = ("full", "compact", "minimal")
ALLOWED_VERBOSITY_ALIASES: Final[tuple[str, ...]] = ("min", "standard", "default")
ALLOWED_VERBOSITY_INPUTS: Final[tuple[str, ...]] = (
    ALLOWED_VERBOSITY_CANONICAL + ALLOWED_VERBOSITY_ALIASES
)

VERBOSITY_FULL: Final[str] = "full"
VERBOSITY_COMPACT: Final[str] = "compact"
VERBOSITY_MINIMAL: Final[str] = "minimal"

# Canonical verbosity values; aliases normalize to one of these.
_CANONICAL_VERBOSITY: Final[frozenset[str]] = frozenset(ALLOWED_VERBOSITY_CANONICAL)

# Alias normalization table. Anything outside raises InvalidVerbosity.
VERBOSITY_ALIAS_MAP: Final[dict[str, str]] = {
    "compact": "compact",
    "minimal": "minimal",
    "min": "minimal",
    "full": "full",
    "standard": "full",
    "default": "full",
}


# Verbosity aliases trigger a normalization_change entry to keep the wire
# metadata honest.
def _alias_reason(canonical: str) -> str:
    return f"verbosity_alias_to_{canonical}"


def normalize_verbosity(value: str | None) -> tuple[str, str | None]:
    """Map any accepted verbosity spelling to its canonical form.

    Returns ``(canonical, reason_or_None)``. ``reason`` is non-None when the
    caller used an alias and the response should record a
    ``normalization_changes`` entry for transparency.
    """
    if value is None:
        return VERBOSITY_FULL, None
    normalized = str(value).strip().lower()
    if normalized not in VERBOSITY_ALIAS_MAP:
        raise ValueError(
            f"INVALID_VERBOSITY: expected one of {sorted(VERBOSITY_ALIAS_MAP)}, got {value!r}"
        )
    canonical = VERBOSITY_ALIAS_MAP[normalized]
    reason = None if normalized == canonical else _alias_reason(canonical)
    return canonical, reason


__all__ = [
    "ALLOWED_DETAIL_LEVELS",
    "ALLOWED_VERBOSITY_ALIASES",
    "ALLOWED_VERBOSITY_CANONICAL",
    "ALLOWED_VERBOSITY_INPUTS",
    "DEPTH_DEFAULT_ANALYZE_GAME",
    "DEPTH_DEFAULT_CLASSIFY",
    "DEPTH_DEFAULT_EVALUATE",
    "DEPTH_MAX",
    "DEPTH_MIN",
    "MAX_COMPARE_MOVES",
    "MAX_CRITICAL_MOMENTS",
    "MAX_FULLMOVE_NUMBER",
    "MAX_HALFMOVE_CLOCK",
    "MAX_INCLUDE_MOVES",
    "PROOF_DEFENSES_DEFAULT",
    "PROOF_DEFENSES_MAX",
    "PROOF_DEFENSES_MIN",
    "TOP_MOVES_DEFAULT_N",
    "TOP_MOVES_MAX_N",
    "TOP_MOVES_MIN_N",
    "VERBOSITY_ALIAS_MAP",
    "VERBOSITY_COMPACT",
    "VERBOSITY_FULL",
    "VERBOSITY_MINIMAL",
    "DetailLevel",
    "normalize_verbosity",
]
