"""Pure PGN header extraction and validation for ``analyze_game``.

Extracted from ``mcp_server.tools.analyze_game``. These functions take a
canonical PGN string and an optional python-chess ``Game`` header object,
and return the validated metadata + warnings. They are side-effect free
(no engine calls, no file I/O), which makes them cheap to unit-test in
isolation and reuse from any tool entry point that needs PGN metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mcp_server.parsers import (
    TAG_PAIR_REGEX,
    _is_valid_pgn_time_control,
    _unescape_pgn_tag_value,
    _validate_pgn_date,
)

if TYPE_CHECKING:
    import chess.pgn


@dataclass
class GameMetadata:
    """Validated PGN header metadata for ``analyze_game``.

    Fields are populated from the canonical PGN header block with strict
    validation applied when ``strict`` is True. Values mirror the
    ``GameAnalysisResult`` header shape so the orchestrator can pass them
    through unchanged.
    """

    white: str | None = None
    black: str | None = None
    event: str | None = None
    site: str | None = None
    round: str | None = None
    date: str | None = None
    result_header: str | None = None
    result_header_raw: str | None = None
    result_movetext: str | None = None
    white_elo: str | None = None
    black_elo: str | None = None
    time_control: str | None = None
    variant: str | None = None
    eco_header: str | None = None
    opening_header: str | None = None
    termination_header: str | None = None
    metadata_warnings: list[str] = field(default_factory=list[str])
    syntax_warnings: list[str] = field(default_factory=list[str])
    duplicate_tag_counts: dict[str, int] = field(default_factory=dict[str, int])


_CANONICAL_RESULTS = frozenset({"1-0", "0-1", "1/2-1/2", "*"})


def _scan_header_block(canonical_pgn: str) -> str:
    """Return the contiguous header block at the start of ``canonical_pgn``.

    Walks ``TAG_PAIR_REGEX`` matches and stops at the first non-tag token,
    matching PGN §7's convention that the header section is a contiguous
    run of bracket-tag lines at the top of the file. Used to scope strict
    validation to header content only (not conversational preamble).
    """
    first_header = TAG_PAIR_REGEX.search(canonical_pgn)
    first_mv = re.search(r"\b1\s*[\.\:]\s*[A-Za-z]", canonical_pgn)
    if first_header is None or (first_mv and first_mv.start() < first_header.start()):
        return ""
    header_end = first_header.end()
    for m in TAG_PAIR_REGEX.finditer(canonical_pgn):
        if m.start() < header_end:
            continue
        if canonical_pgn[header_end : m.start()].strip() == "":
            header_end = m.end()
        else:
            break
    return canonical_pgn[:header_end]


def _collect_tags(header_section: str) -> dict[str, str]:
    """Parse the first occurrence of every canonical (lowercase) tag from
    the header section. Returns ``{tag_name: value}`` for canonical-only
    lookups (audit P2 / ultra-audit finding)."""
    tags: dict[str, str] = {}
    for tag_m in TAG_PAIR_REGEX.finditer(header_section):
        tag_k = tag_m.group(1).lower()
        tag_v = _unescape_pgn_tag_value(tag_m.group(2))
        if tag_k not in tags and tag_v is not None and tag_v != "?":
            tags[tag_k] = tag_v
    return tags


def _header_lookup(
    tags_dict: dict[str, str],
    game_headers: chess.pgn.Headers,
    tag_name: str,
    header_attr: str,
) -> str | None:
    """Prefer the lowercased tags-dict value; fall back to python-chess's
    native header object for any tag python-chess may have normalized."""
    v = tags_dict.get(tag_name)
    if v:
        return v
    raw = game_headers.get(header_attr)
    if raw and raw != "?":
        return _unescape_pgn_tag_value(raw)
    return None


def _validate_elo_value(value: str | None) -> str | None:
    """Return ``value`` unchanged if it's a valid PGN Elo, else None.

    Strict-validation surfaces invalid Elo values as a metadata_warning
    upstream — this helper is the predicate the strict path calls. The
    lenient path uses ``_header_lookup`` directly with no validation.
    """
    if value is None or value == "-":
        return value
    if value.isdigit() and 0 <= int(value) <= 4000:
        return value
    return value


def extract_game_metadata(
    canonical_pgn: str,
    game: chess.pgn.Game,
    *,
    strict: bool,
    lexical_warnings: list[str] | None = None,
    syntax_warnings: list[str] | None = None,
    is_comment_only_input: bool = False,
    result_movetext: str | None = None,
) -> GameMetadata:
    """Build a :class:`GameMetadata` from the canonical PGN.

    The orchestrator supplies the already-parsed ``game``, the
    accumulated ``syntax_warnings`` / ``lexical_warnings`` lists, and
    the movetext result token it parsed with
    :func:`mcp_server.parsers.find_movetext_result`. The helper adds
    strict-mode-specific validation warnings and bundles everything
    together so the response model can be constructed without further
    string-munging.
    """
    md = GameMetadata()
    md.syntax_warnings.extend(syntax_warnings or [])
    md.metadata_warnings.extend(lexical_warnings or [])
    md.result_movetext = result_movetext

    header_section = _scan_header_block(canonical_pgn)
    tags = _collect_tags(header_section)
    h = game.headers

    md.white = _header_lookup(tags, h, "white", "White")
    md.black = _header_lookup(tags, h, "black", "Black")
    md.event = _header_lookup(tags, h, "event", "Event")
    md.site = _header_lookup(tags, h, "site", "Site")
    md.round = _header_lookup(tags, h, "round", "Round")
    md.white_elo = tags.get("whiteelo") or (
        h.get("WhiteElo") if h.get("WhiteElo") and h.get("WhiteElo") != "?" else None
    )
    md.black_elo = tags.get("blackelo") or (
        h.get("BlackElo") if h.get("BlackElo") and h.get("BlackElo") != "?" else None
    )
    md.time_control = tags.get("timecontrol") or (
        h.get("TimeControl") if h.get("TimeControl") and h.get("TimeControl") != "?" else None
    )
    if md.time_control is not None:
        md.time_control = md.time_control.strip()
        if md.time_control == "?":
            md.time_control = None
    md.variant = tags.get("variant") or (
        h.get("Variant") if h.get("Variant") and h.get("Variant") != "?" else None
    )
    md.date = tags.get("date") or tags.get("utcdate") or h.get("Date") or h.get("UTCDate")
    if md.date is not None and md.date.strip() in ("", "?", "????.??.??"):
        md.date = None

    if md.date is not None:
        date_err = _validate_pgn_date(md.date)
        if date_err is not None:
            md.metadata_warnings.append(date_err)

    md.eco_header = tags.get("eco") or h.get("ECO")
    md.opening_header = tags.get("opening") or h.get("Opening")

    for tag_m in TAG_PAIR_REGEX.finditer(header_section):
        tag_k = tag_m.group(1).lower()
        tag_v = _unescape_pgn_tag_value(tag_m.group(2))
        if tag_k == "result" and md.result_header_raw is None:
            md.result_header_raw = tag_v
        elif tag_k == "termination" and md.termination_header is None:
            md.termination_header = tag_v

    if md.result_header_raw is not None and md.result_header_raw != "?":
        if md.result_header_raw in _CANONICAL_RESULTS:
            md.result_header = md.result_header_raw
        else:
            md.metadata_warnings.append(
                f"Invalid Result header tag '{md.result_header_raw}'; "
                f"expected 1-0, 0-1, 1/2-1/2, or *."
            )

    if md.white_elo is not None and md.white_elo != "-":
        if not (md.white_elo.isdigit() and 0 <= int(md.white_elo) <= 4000):
            md.metadata_warnings.append(
                f"Invalid WhiteElo header tag '{md.white_elo}'; expected numeric integer rating."
            )
    if md.black_elo is not None and md.black_elo != "-":
        if not (md.black_elo.isdigit() and 0 <= int(md.black_elo) <= 4000):
            md.metadata_warnings.append(
                f"Invalid BlackElo header tag '{md.black_elo}'; expected numeric integer rating."
            )
    if md.time_control is not None and not _is_valid_pgn_time_control(md.time_control):
        md.metadata_warnings.append(f"Invalid TimeControl header tag '{md.time_control}'.")

    if is_comment_only_input and not strict:
        md.metadata_warnings.append(
            "Input PGN contained only comments (and optionally a result "
            "token) with no moves; returning an empty game."
        )

    setup_header = h.get("SetUp")
    fen_header = h.get("FEN")
    if setup_header is not None and setup_header not in ("0", "1"):
        md.metadata_warnings.append(
            f"Invalid SetUp tag value '{setup_header}': must be exactly '0' or '1'."
            if strict
            else f"Non-canonical SetUp tag value '{setup_header}': expected '0' or '1'."
        )
    if setup_header == "1" and not fen_header:
        md.metadata_warnings.append(
            '[SetUp "1"] tag provided without FEN tag; defaulting to standard starting position.'
        )
    elif fen_header and setup_header != "1":
        md.metadata_warnings.append('FEN tag provided without [SetUp "1"]; custom position loaded.')

    return _finalize_header_warnings(md, header_section)


def _finalize_header_warnings(md: GameMetadata, header_section: str) -> GameMetadata:
    """Surface duplicate-tag and conflicting-value warnings (audit P2)."""
    tag_counts: dict[str, int] = {}
    tag_values_by_canonical: dict[str, list[str]] = {}
    for tag_m in TAG_PAIR_REGEX.finditer(header_section):
        tag_name_raw = tag_m.group(1)
        tag_value = _unescape_pgn_tag_value(tag_m.group(2))
        tag_name_canonical = tag_name_raw.lower()
        tag_counts[tag_name_canonical] = tag_counts.get(tag_name_canonical, 0) + 1
        if tag_value is not None:
            tag_values_by_canonical.setdefault(tag_name_canonical, []).append(tag_value)

    md.duplicate_tag_counts = tag_counts
    for tag_name, count in tag_counts.items():
        if count > 1:
            md.metadata_warnings.append(
                f"Duplicate PGN tag '[{tag_name}]' detected ({count} occurrences); "
                f"using canonical tag value."
            )
    for canonical_name in ("result", "variant"):
        values = tag_values_by_canonical.get(canonical_name) or []
        if len(values) >= 2 and any(v != values[0] for v in values[1:]):
            md.metadata_warnings.append(
                f"Conflicting values for PGN tag '{canonical_name}': {values!r}; "
                f"using the first declared value."
            )
    return md
