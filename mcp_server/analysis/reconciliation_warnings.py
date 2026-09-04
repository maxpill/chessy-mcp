"""Reconciliation warnings — strict-mode rejection + contradiction findings.

Extracted from :mod:`mcp_server.analysis.result_reconciliation`. Owns
the four ``emit_*` warning helpers + :func:`compute_result_inferred`:

  * :func:`emit_strict_termination_warning` — strict-mode rejection of
    unknown Termination tokens that don't normalize to a FIDE value.
  * :func:`emit_contradiction_warnings` — Termination says draw, Result
    says decisive — and the symmetrical mismatches.
  * :func:`emit_premature_draw_warning` — draw agreement declared
    before either side completed a move.
  * :func:`infer_from_termination` — winner/loser grammar in the
    Termination header that triggers a Result inference.
  * :func:`compute_result_inferred` — the ``result_inferred` field
    on the response: board-derived result if available, else the
    canonical Result token when header and movetext agreed on
    unterminated.
"""

from __future__ import annotations

import chess


def infer_from_termination(termination_header: str | None) -> str | None:
    """Infer the PGN Result token from a Termination header's wording.

    Only winner/loser grammar triggers an inference; ambiguous terminations
    (``Normal`, ``Unterminated`) leave the result alone so the caller can
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


def emit_strict_termination_warning(
    *,
    strict: bool,
    termination_header: str | None,
    warnings: list[str],
) -> None:
    """Strict mode rejects Termination tokens that don't normalize to a
    known FIDE value AND don't contain a recognised lowercase keyword.

    The strict-pass at the end of ``analyze_game` promotes this warning
    to a ``STRICT_PGN_ERROR`.
    """
    if not (strict and termination_header):
        return
    from mcp_server.tools._common import normalize_termination

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


def emit_contradiction_warnings(
    *,
    termination_header: str | None,
    result_val: str,
    warnings: list[str],
) -> None:
    """Surface contradictory PGN metadata pairs (Termination says draw,
    Result says decisive — and the symmetrical mismatches)."""
    if not termination_header:
        return
    from mcp_server.tools._common import normalize_termination

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


def emit_premature_draw_warning(
    *,
    termination_header: str | None,
    moves_count: int,
    warnings: list[str],
) -> None:
    """A draw agreement declared before either side completed a move is a
    metadata inconsistency."""
    if termination_header and "agreement" in termination_header.lower() and moves_count < 2:
        warnings.append("Draw agreement declared before both players completed at least one move.")


def compute_result_inferred(
    *,
    final_board: chess.Board,
    result_board: str | None,
    result_val: str,
    result_header_raw: str | None,
    result_movetext: str | None,
) -> str | None:
    """``result_inferred` on the response: the board-derived result if
    available, else the canonical Result token when both header and
    movetext agreed it was unterminated (``*`).
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


# Back-compat shims.
_infer_from_termination = infer_from_termination
_emit_strict_termination_warning = emit_strict_termination_warning
_emit_contradiction_warnings = emit_contradiction_warnings
_emit_premature_draw_warning = emit_premature_draw_warning
_compute_result_inferred = compute_result_inferred
