"""Pure result + termination reconciliation for ``analyze_game``.

Takes the final board, headers, movetext result, and orchestration
metadata, and returns the canonical ``(result, termination, warnings)``
triple plus :class:\`ResultInferred\`. Side-effect free; no engine calls.

The reconciliation order matters and is documented inline:

  1. Board truth: if the final board is terminal, derive ``result\`
     and ``auto_termination\` from the rule status (checkmate vs draw).
  2. Header / movetext disagreement: warn if the three signals disagree.
  3. Header truth: if no board truth, prefer ``Result\` header or
     movetext token (skipping ``*\` / ``?\`).
  4. Termination inference: lift the result from a winner/loser
     Termination header when otherwise unknown.
  5. Mating possibility: drop a "mate impossible" finding if the
     declared Result says decisive but no checkmate sequence exists.
  6. Termination reconciliation: cross-check header against board state,
     normalize, surface contradictions.
  7. Strict-mode extras: rejection of unknown Termination tokens,
     premature-draw-agreement detection.

Termination-resolution logic lives in
:mod:\`mcp_server.analysis.termination_resolution\`. Warning emission
lives in :mod:\`mcp_server.analysis.reconciliation_warnings\`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import chess

from mcp_server.analysis.reconciliation_warnings import (
    compute_result_inferred,
    emit_contradiction_warnings,
    emit_premature_draw_warning,
    emit_strict_termination_warning,
    infer_from_termination,
)
from mcp_server.analysis.termination_resolution import resolve_termination
from mcp_server.rules import evaluate_rule_status, validate_mating_possibility

if TYPE_CHECKING:
    from mcp_server.analysis.game_validation import GameMetadata


@dataclass
class ReconciledResult:
    """Final result/termination triple produced by :func:\`reconcile_result\`.

    Public surface is the dataclass; module-level functions are private.
    """

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
    """Reconcile PGN ``Result\` / ``Termination\` headers, movetext token,
    and board truth into a single canonical answer.

    The function copies ``metadata.metadata_warnings\` into a local list
    and appends to it — callers receive the merged warnings plus the
    canonical ``result\` / ``termination\` values.
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

    result_val = _pick_result_value(
        result_board=result_board,
        metadata=metadata,
        result_movetext=result_movetext,
        warnings=warnings,
    )

    if result_val == "*" or result_val is None:
        inferred = infer_from_termination(metadata.termination_header)
        if inferred is not None:
            result_val = inferred

    result_val, mate_warnings = validate_mating_possibility(
        final_board, result_val, metadata.termination_header
    )
    warnings.extend(mate_warnings)

    termination_val, term_warnings = resolve_termination(
        final_board,
        metadata.termination_header,
        auto_termination,
        result_val,
        result_movetext,
        warnings,
    )
    warnings = term_warnings

    emit_strict_termination_warning(
        strict=strict,
        termination_header=metadata.termination_header,
        warnings=warnings,
    )
    emit_contradiction_warnings(
        termination_header=metadata.termination_header,
        result_val=result_val,
        warnings=warnings,
    )
    emit_premature_draw_warning(
        termination_header=metadata.termination_header,
        moves_count=moves_count,
        warnings=warnings,
    )

    return ReconciledResult(
        result=result_val or "*",
        termination=termination_val,
        warnings=warnings,
        auto_termination=auto_termination,
        result_inferred=compute_result_inferred(
            final_board=final_board,
            result_board=result_board,
            result_val=result_val,
            result_header_raw=metadata.result_header_raw,
            result_movetext=result_movetext,
        ),
    )


def _pick_result_value(
    *,
    result_board: str | None,
    metadata: "GameMetadata",
    result_movetext: str | None,
    warnings: list[str],
) -> str | None:
    """Pick the canonical ``result\` string given the board truth, header,
    and movetext token. Mutates ``warnings\` with disagreement findings."""
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
        return result_val

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
        return metadata.result_header
    if result_movetext and result_movetext not in ("*", "?"):
        return result_movetext
    return metadata.result_header or result_movetext or "*"
