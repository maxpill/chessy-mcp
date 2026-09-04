"""PGN validation: variant + TimeControl + dates + castling + FEN counters.

Phase 30: package-ized from :mod:\`mcp_server.parsers.pgn_validate\`
(303 lines) into four focused modules:

  * :mod:\`mcp_server.parsers.pgn_validate.variant\` — :func:\`validate_variant\`
    + :data:\`SUPPORTED_VARIANTS\`.
  * :mod:\`mcp_server.parsers.pgn_validate.time_control\` — TimeControl
    tag grammar validation (:func:\`is_valid_pgn_time_control\`,
    :func:\`stage_has_positive_number\`).
  * :mod:\`mcp_server.parsers.pgn_validate.date\` — :func:\`validate_pgn_date\`.
  * :mod:\`mcp_server.parsers.pgn_validate.castling\` — :func:\`validate_castling_rights\`
    (U-09 + P3 audit).
  * :mod:\`mcp_server.parsers.pgn_validate.fen_counters\` — :func:\`validate_fen_counters\`
    (P1 audit).

Module-level :data:\`MAX_HALFMOVE_CLOCK\` + :data:\`MAX_FULLMOVE_NUMBER\`
consts live in :mod:\`mcp_server.parsers.pgn_validate.fen_counters\`.

Public symbols are re-exported here for backward-compatible import paths
(\`from mcp_server.parsers.pgn_validate import validate_variant\`,
\`from mcp_server.parsers.pgn_validate import _validate_variant\`, etc.).
"""

from __future__ import annotations

from mcp_server.parsers.pgn_validate.castling import (
    _validate_castling_rights,
    validate_castling_rights,
)
from mcp_server.parsers.pgn_validate.date import _validate_pgn_date, validate_pgn_date
from mcp_server.parsers.pgn_validate.fen_counters import (
    MAX_FULLMOVE_NUMBER,
    MAX_HALFMOVE_CLOCK,
    _validate_fen_counters,
    validate_fen_counters,
)
from mcp_server.parsers.pgn_validate.time_control import (
    TIME_CONTROL_STAGE_RE,
    _is_valid_pgn_time_control,
    _stage_has_positive_number,
    is_valid_pgn_time_control,
    stage_has_positive_number,
)
from mcp_server.parsers.pgn_validate.variant import (
    SUPPORTED_VARIANTS,
    _validate_variant,
    validate_variant,
)


__all__ = [
    "MAX_FULLMOVE_NUMBER",
    "MAX_HALFMOVE_CLOCK",
    "SUPPORTED_VARIANTS",
    "TIME_CONTROL_STAGE_RE",
    "_is_valid_pgn_time_control",
    "_stage_has_positive_number",
    "_validate_castling_rights",
    "_validate_fen_counters",
    "_validate_pgn_date",
    "_validate_variant",
    "is_valid_pgn_time_control",
    "stage_has_positive_number",
    "validate_castling_rights",
    "validate_fen_counters",
    "validate_pgn_date",
    "validate_variant",
]
