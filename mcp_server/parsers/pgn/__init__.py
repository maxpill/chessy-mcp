"""PGN canonical extraction package — focused modules replacing the old monolith.

Modules:

    - ``tags`` — ``TAG_PAIR_REGEX`` (single source of truth for ``[Tag "Value"]``).
    - ``unicode`` — Unicode normalization (figurines, hyphens, results).
    - ``multiline_tags`` — multiline-tag collapsing + canonical-tag-line detection.
    - ``header_syntax`` — strict / lenient PGN tag-line validation + sanitization.
    - ``tokens`` — movetext token validation (canonical SAN, NAG range, ``+``/``#`` markers, promotion form).
    - ``movetext`` — result token detection, prose discrimination, completion-detection.
    - ``multiple_games`` — multi-game detection.
    - ``conversational`` — freeform prose cleanup (best PGN block extraction).
    - ``extractor`` — top-level orchestration (``extract_game``, ``parse_pgn_game_candidate``).

The flat public surface (every name available at the package root) is what
``mcp_server.parsers.pgn_canonical`` re-exports. Tests reach individual helpers
directly when they need to.
"""

from mcp_server.parsers.pgn.conversational import clean_conversational_text
from mcp_server.parsers.pgn.extractor import (
    extract_canonical_pgn_text,
    extract_game,
    extract_game_inner,
    parse_pgn_game_candidate,
)
from mcp_server.parsers.pgn.header_syntax import (
    sanitize_malformed_pgn_header_lines,
    validate_strict_header_syntax,
)
from mcp_server.parsers.pgn.movetext import (
    find_movetext_result,
    has_completed_game_before,
    infer_result_from_termination,
    is_prose_line,
    truncate_movetext_at_result,
)
from mcp_server.parsers.pgn.multiline_tags import (
    is_canonical_tag_line,
    normalize_multiline_tags,
)
from mcp_server.parsers.pgn.multiple_games import check_multiple_games
from mcp_server.parsers.pgn.tags import TAG_PAIR_REGEX
from mcp_server.parsers.pgn.tokens import (
    strict_top_level_movetext_tokens,
    strip_promotion_eq,
    validate_movetext_tokens,
    validate_strict_mainline_surface,
)
from mcp_server.parsers.pgn.unicode import (
    FIGURINE_MAP,
    UNICODE_HYPHEN_MAP,
    normalize_movetext_figurines,
    normalize_unicode_pgn_results,
)

# Underscored aliases — public surface preserved for back-compat with
# legacy import paths that pre-date the package split.
_UNICODE_HYPHEN_MAP = UNICODE_HYPHEN_MAP
_FIGURINE_MAP = FIGURINE_MAP
_unicode_hyphen_map = UNICODE_HYPHEN_MAP
_figurine_map = FIGURINE_MAP
_normalize_movetext_figurines = normalize_movetext_figurines
_normalize_unicode_pgn_results = normalize_unicode_pgn_results
_validate_movetext_tokens = validate_movetext_tokens
_truncate_movetext_at_result = truncate_movetext_at_result
_parse_pgn_game_candidate = parse_pgn_game_candidate
_check_multiple_games = check_multiple_games
_normalize_multiline_tags = normalize_multiline_tags
_is_canonical_tag_line = is_canonical_tag_line
_is_prose_line = is_prose_line
_has_completed_game_before = has_completed_game_before
_find_movetext_result = find_movetext_result
_sanitize_malformed_pgn_header_lines = sanitize_malformed_pgn_header_lines
_infer_result_from_termination = infer_result_from_termination
_extract_game = extract_game
_extract_game_inner = extract_game_inner
_clean_conversational_text = clean_conversational_text
_extract_canonical_pgn_text = extract_canonical_pgn_text
_validate_strict_mainline_surface = validate_strict_mainline_surface
_strict_top_level_movetext_tokens = strict_top_level_movetext_tokens
_validate_strict_header_syntax = validate_strict_header_syntax
_strip_promotion_eq = strip_promotion_eq


__all__ = [
    "FIGURINE_MAP",
    "TAG_PAIR_REGEX",
    "UNICODE_HYPHEN_MAP",
    "_check_multiple_games",
    "_clean_conversational_text",
    "_extract_canonical_pgn_text",
    "_extract_game",
    "_extract_game_inner",
    "_FIGURINE_MAP",
    "_figurine_map",
    "_find_movetext_result",
    "_has_completed_game_before",
    "_infer_result_from_termination",
    "_is_canonical_tag_line",
    "_is_prose_line",
    "_normalize_movetext_figurines",
    "_normalize_multiline_tags",
    "_normalize_unicode_pgn_results",
    "_parse_pgn_game_candidate",
    "_sanitize_malformed_pgn_header_lines",
    "_strip_promotion_eq",
    "_strict_top_level_movetext_tokens",
    "_truncate_movetext_at_result",
    "_UNICODE_HYPHEN_MAP",
    "_unicode_hyphen_map",
    "_validate_movetext_tokens",
    "_validate_strict_header_syntax",
    "_validate_strict_mainline_surface",
    "check_multiple_games",
    "clean_conversational_text",
    "extract_canonical_pgn_text",
    "extract_game",
    "extract_game_inner",
    "find_movetext_result",
    "has_completed_game_before",
    "infer_result_from_termination",
    "is_canonical_tag_line",
    "is_prose_line",
    "normalize_movetext_figurines",
    "normalize_multiline_tags",
    "normalize_unicode_pgn_results",
    "parse_pgn_game_candidate",
    "sanitize_malformed_pgn_header_lines",
    "strip_promotion_eq",
    "strict_top_level_movetext_tokens",
    "truncate_movetext_at_result",
    "validate_movetext_tokens",
    "validate_strict_header_syntax",
    "validate_strict_mainline_surface",
]
