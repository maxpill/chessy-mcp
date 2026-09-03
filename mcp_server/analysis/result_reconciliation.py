"""Pure result + termination reconciliation for ``analyze_game``.

Takes the final board, headers, movetext result, and orchestration
metadata, and returns the canonical ``(result, termination, warnings)``
triple plus :class:`ResultInferred`. Side-effect free; no engine calls.

The reconciliation order matters and is documented inline:
  1. Board truth: if the final board is terminal, derive ``result`` and
     ``auto_termination`` from the rule status (checkmate vs draw).
  2. Header / movetext disagreement: warn if the three signals disagree.
  3. Header truth: if no board truth, prefer ``Result`` header or
     movetext token (skipping ``*`` / ``?``).
  4. Termination inference: lift the result from a winner/loser
     Termination header when otherwise unknown.
  5. Mating possibility: drop a "mate impossible" finding if the
     declared Result says decisive but no checkmate sequence exists.
  6. Termination reconciliation: cross-check header against board state,
     normalize, surface contradictions.
  7. Strict-mode extras: rejection of unknown Termination tokens,
     premature-draw-agreement detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from mcp_server.rules import (
    evaluate_rule_status,
    is_locked_dead_position,
    validate_mating_possibility,
)
from mcp_server.tools._common import normalize_termination

if TYPE_CHECKING:
    from mcp_server.analysis.game_validation import GameMetadata


@dataclass
class ReconciledResult:
    """Final result/termination triple produced by :func:`reconcile_result`."""

    result: str
    termination: str | None
    warnings: list[str]
    auto_termination: str | None
    result_inferred: str | None


def reconcile_result(
    final_board: chess.Board,
    metadata: "GameMetadata",
    *,
    result_movetext: str | None,
    moves_count: int,
    strict: bool,
) -> ReconciledResult:
    """Reconcile PGN ``Result`` / ``Termination`` headers, movetext token,
    and board truth into a single canonical answer.

    The function copies ``metadata.metadata_warnings`` into a local list
    and appends to it — callers receive the merged warnings plus the
    canonical ``result`` / ``termination`` values.
    """
    warnings: list[str] = list(metadata.metadata_warnings)
    rule_final = evaluate_rule_status(final_board, history_complete="complete")
    result_board: str | None = None
    auto_termination: str | None = None
    if rule_final.terminal is not None:
        if rule_final.terminal == "checkmate":
            result_board = "1-0" if final_board.turn == chess.BLACK else "0-1"
            auto_termination = "checkmate"
        else:
            result_board = "1/2-1/2"
            auto_termination = rule_final.terminal

    if result_board is not None:
        result_val = result_board
        if metadata.result_header and metadata.result_header not in ("*", "?"):
            if metadata.result_header != result_board:
                warnings.append(
                    f"Result header '{metadata.result_header}' disagrees with "
                    f"board outcome '{result_board}'; using board outcome."
                )
        if (
            result_movetext
            and result_movetext not in ("*", "?")
            and result_movetext != result_board
        ):
            warnings.append(
                f"Movetext result '{result_movetext}' disagrees with "
                f"board outcome '{result_board}'; using board outcome."
            )
    else:
        if (
            metadata.result_header_raw
            and result_movetext
            and metadata.result_header_raw != result_movetext
        ):
            warnings.append(
                f"Result header '{metadata.result_header_raw}' disagrees with "
                f"movetext result '{result_movetext}'."
            )
        if metadata.result_header and metadata.result_header not in ("*", "?"):
            result_val = metadata.result_header
        elif result_movetext and result_movetext not in ("*", "?"):
            result_val = result_movetext
        else:
            result_val = metadata.result_header or result_movetext or "*"

    if result_val == "*" or result_val is None:
        inferred = _infer_from_termination(metadata.termination_header)
        if inferred is not None:
            result_val = inferred

    result_val, mate_warnings = validate_mating_possibility(
        final_board, result_val, metadata.termination_header
    )
    warnings.extend(mate_warnings)

    termination_val, term_warnings = _resolve_termination(
        final_board,
        metadata.termination_header,
        auto_termination,
        result_val,
        result_movetext,
        warnings,
    )
    warnings = term_warnings

    _emit_strict_termination_warning(
        strict=strict,
        termination_header=metadata.termination_header,
        warnings=warnings,
    )
    _emit_contradiction_warnings(
        termination_header=metadata.termination_header,
        result_val=result_val,
        warnings=warnings,
    )
    _emit_premature_draw_warning(
        termination_header=metadata.termination_header,
        moves_count=moves_count,
        warnings=warnings,
    )

    return ReconciledResult(
        result=result_val or "*",
        termination=termination_val,
        warnings=warnings,
        auto_termination=auto_termination,
        result_inferred=_compute_result_inferred(
            final_board=final_board,
            result_board=result_board,
            result_val=result_val,
            result_header_raw=metadata.result_header_raw,
            result_movetext=result_movetext,
        ),
    )


def _infer_from_termination(termination_header: str | None) -> str | None:
    """Infer the PGN Result token from a Termination header's wording.

    Only winner/loser grammar triggers an inference; ambiguous terminations
    (``Normal``, ``Unterminated``) leave the result alone so the caller can
    fall back to the board truth.
    """
    if not termination_header:
        return None
    t = termination_header.lower()
    if "white" in t and ("won" in t or "wins" in t):
        return "1-0"
    if "black" in t and ("won" in t or "wins" in t):
        return "0-1"
    return None


def _resolve_termination(
    final_board: chess.Board,
    termination_header: str | None,
    auto_termination: str | None,
    result_val: str,
    result_movetext: str | None,
    warnings: list[str],
) -> tuple[str | None, list[str]]:
    """Cross-check the Termination header against the auto-derived termination.

    Mutates ``warnings`` with disagreement findings (preserving the
    audit-equivalent ordering vs. the inline version) and returns the
    surface termination value.
    """
    if auto_termination is not None:
        termination_val = auto_termination
        if termination_header:
            norm_term_hdr = normalize_termination(termination_header)
            if norm_term_hdr != "normal":
                is_concurrent = _is_concurrent_board_match(norm_term_hdr, final_board)
                if norm_term_hdr != auto_termination and not is_concurrent:
                    warnings.append(
                        f"Termination header '{termination_header}' disagrees "
                        f"with board outcome '{auto_termination}'; using board outcome."
                    )
        return termination_val, warnings

    if not termination_header:
        return None, warnings

    norm_term_hdr = normalize_termination(termination_header)
    if norm_term_hdr == "normal":
        return "normal", warnings
    if norm_term_hdr in ("checkmate", "stalemate"):
        warnings.append(
            f"Termination header '{termination_header}' contradicts board "
            f"state (position is not {norm_term_hdr})."
        )
        return None, warnings
    if norm_term_hdr == "threefold_repetition":
        if not final_board.is_repetition(3):
            warnings.append(
                f"Termination header '{termination_header}' contradicts "
                f"board state (position is not threefold_repetition)."
            )
            return None, warnings
        return "threefold_repetition", warnings
    if norm_term_hdr == "fifty_moves":
        if not final_board.is_fifty_moves() and final_board.halfmove_clock < 100:
            warnings.append(
                f"Termination header '{termination_header}' contradicts "
                f"board state (position is not fifty_moves)."
            )
            return None, warnings
        return "fifty_moves", warnings
    if norm_term_hdr in (
        "insufficient_material",
        "seventyfive_moves",
        "fivefold_repetition",
        "dead_position",
    ):
        warnings.append(
            f"Termination header '{termination_header}' contradicts "
            f"board state (position is not {norm_term_hdr})."
        )
        return None, warnings
    return norm_term_hdr, warnings


def _is_concurrent_board_match(norm_term: str, board: chess.Board) -> bool:
    """True iff ``board`` independently exhibits ``norm_term`` (so header
    and board-trace are concurrent, not contradictory)."""
    if norm_term == "stalemate":
        return board.is_stalemate()
    if norm_term == "seventyfive_moves":
        return board.is_seventyfive_moves()
    if norm_term == "fivefold_repetition":
        return board.is_fivefold_repetition()
    if norm_term == "insufficient_material":
        return board.is_insufficient_material()
    if norm_term == "fifty_moves":
        return board.is_fifty_moves() or board.halfmove_clock >= 100
    if norm_term == "threefold_repetition":
        return board.is_repetition(3)
    if norm_term == "dead_position":
        return is_locked_dead_position(board)
    return False


def _emit_strict_termination_warning(
    *,
    strict: bool,
    termination_header: str | None,
    warnings: list[str],
) -> None:
    """Strict mode rejects Termination tokens that don't normalize to a
    known FIDE value AND don't contain a recognised lowercase keyword.
    The strict-pass at the end of ``analyze_game`` promotes this warning
    to a ``STRICT_PGN_ERROR``."""
    if not (strict and termination_header):
        return
    norm_term = normalize_termination(termination_header)
    if norm_term is None and termination_header.strip() not in (
        "",
        "Normal",
        "Time forfeit",
        "Rules infraction",
        "Abandoned",
        "Unterminated",
    ):
        lower = termination_header.strip().lower()
        if not any(
            kw in lower
            for kw in (
                "resign",
                "checkmate",
                "stalemate",
                "time",
                "abandon",
                "rule",
                "draw",
                "repetition",
                "insufficient",
                "50-move",
                "75-move",
            )
        ):
            warnings.append(f"Unrecognised Termination tag '{termination_header}'.")


def _emit_contradiction_warnings(
    *,
    termination_header: str | None,
    result_val: str,
    warnings: list[str],
) -> None:
    """Surface contradictory PGN metadata pairs (Termination says draw,
    Result says decisive — and the symmetrical mismatches)."""
    if not termination_header:
        return
    norm_term = normalize_termination(termination_header)
    if norm_term in (
        "stalemate",
        "insufficient_material",
        "fifty_moves",
        "seventyfive_moves",
        "threefold_repetition",
        "fivefold_repetition",
        "dead_position",
    ) and result_val in ("1-0", "0-1"):
        warnings.append(
            f"Contradictory PGN metadata: Termination '{termination_header}' "
            f"contradicts Result '{result_val}'."
        )
    elif norm_term == "checkmate" and result_val in ("1/2-1/2", "*"):
        warnings.append(
            f"Contradictory PGN metadata: Termination '{termination_header}' "
            f"contradicts Result '{result_val}'."
        )
    elif norm_term == "unterminated" and result_val in ("1-0", "0-1", "1/2-1/2"):
        warnings.append(
            f"Contradictory PGN metadata: Termination '{termination_header}' "
            f"contradicts Result '{result_val}'."
        )


def _emit_premature_draw_warning(
    *,
    termination_header: str | None,
    moves_count: int,
    warnings: list[str],
) -> None:
    """A draw agreement declared before either side completed a move is a
    metadata inconsistency."""
    if termination_header and "agreement" in termination_header.lower() and moves_count < 2:
        warnings.append("Draw agreement declared before both players completed at least one move.")


def _compute_result_inferred(
    *,
    final_board: chess.Board,
    result_board: str | None,
    result_val: str,
    result_header_raw: str | None,
    result_movetext: str | None,
) -> str | None:
    """``result_inferred`` on the response: the board-derived result if
    available, else the canonical Result token when both header and
    movetext agreed it was unterminated (``*``).
    """
    if result_board is not None:
        return result_board
    if (
        result_header_raw in ("*", None)
        and result_movetext in ("*", None)
        and result_val in ("1-0", "0-1", "1/2-1/2")
    ):
        return result_val
    return None
