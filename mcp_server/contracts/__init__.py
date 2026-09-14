"""Public contracts module — single source of truth for shared validation.

Importable from anywhere; no third-party dependencies beyond the standard
library + pydantic (which is already a project dependency).
"""

from __future__ import annotations

from mcp_server.contracts.constants import (
    ALLOWED_DETAIL_LEVELS,
    ALLOWED_VERBOSITY_ALIASES,
    ALLOWED_VERBOSITY_CANONICAL,
    ALLOWED_VERBOSITY_INPUTS,
    DEPTH_DEFAULT_ANALYZE_GAME,
    DEPTH_DEFAULT_CLASSIFY,
    DEPTH_DEFAULT_EVALUATE,
    DEPTH_MAX,
    DEPTH_MIN,
    MAX_COMPARE_MOVES,
    MAX_CRITICAL_MOMENTS,
    MAX_FULLMOVE_NUMBER,
    MAX_HALFMOVE_CLOCK,
    MAX_INCLUDE_MOVES,
    PROOF_DEFENSES_DEFAULT,
    PROOF_DEFENSES_MAX,
    PROOF_DEFENSES_MIN,
    TOP_MOVES_DEFAULT_N,
    TOP_MOVES_MAX_N,
    TOP_MOVES_MIN_N,
    VERBOSITY_ALIAS_MAP,
    VERBOSITY_COMPACT,
    VERBOSITY_FULL,
    VERBOSITY_MINIMAL,
    DetailLevel,
    normalize_verbosity,
)
from mcp_server.contracts.detail_validator import validate_detail_verbosity
from mcp_server.contracts.normalized_input import (
    InputKind,
    NormalizedPositionInput,
)

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
    "InputKind",
    "NormalizedPositionInput",
    "normalize_verbosity",
    "validate_detail_verbosity",
]
