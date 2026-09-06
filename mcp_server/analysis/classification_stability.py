"""Selective multi-depth verification for rich ``classify_move`` evidence.

The ordinary move classifier already has a narrow consistency check for the
special case where the played move is reported as engine-best but still grades
as a large error. Coaching needs a different question: does the pedagogically
important classification itself survive a deeper search?

This module performs that verification only for ``forensic`` move analysis and
only for inaccuracy/mistake/blunder results. It re-searches the position before
and after the played move at a higher depth, recomputes the normal move grade,
and escalates once more when the first verification disagrees. Standard and
coach modes keep their existing cost.

Production callers can inject the repository's cached/rule-aware evaluator, so
these extra searches retain the same history, terminal and SingleFlight/cache
semantics as the rest of the MCP. Tests and isolated callers may fall back to a
minimal raw-pool adapter.

The output is evidence, not a new grading policy. The original wire-level
classification is left untouched; the richer ``forensics.stability`` block says
whether deeper searches agree and what changed.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal

import chess

from mcp_server.models import MCPEval
from mcp_server.models.forensics import ForensicMoveAnalysis
from mcp_server.move_grading import score_played_move

_REVERIFY_CLASSES = {"inaccuracy", "mistake", "blunder"}
CachedEvaluator = Callable[..., Awaitable[tuple[MCPEval, bool]]]


def _class_name(value: Any) -> str:
    return str(getattr(value, "value", value))


def _mate_signature(ev: MCPEval) -> Literal["white_mates", "black_mates", "no_mate"]:
    if ev.mate is None:
        return "no_mate"
    return "white_mates" if ev.mate > 0 else "black_mates"


def _wdl_expectation(
    wdl: tuple[int, int, int] | None,
    color: chess.Color,
) -> float | None:
    if wdl is None:
        return None
    wins, draws, losses = wdl
    total = wins + draws + losses
    if total <= 0:
        return None
    white = (wins + 0.5 * draws) / total
    return white if color == chess.WHITE else 1.0 - white


def _wdl_loss_percentage_points(before: MCPEval, after: MCPEval, mover: chess.Color) -> float | None:
    before_expectation = _wdl_expectation(before.wdl, mover)
    after_expectation = _wdl_expectation(after.wdl, mover)
    if before_expectation is None or after_expectation is None:
        return None
    return max(0.0, (before_expectation - after_expectation) * 100.0)


def _as_mcpeval(
    raw: Any,
    board: chess.Board,
    *,
    depth: int,
    history_complete: str,
) -> MCPEval:
    if isinstance(raw, MCPEval):
        return raw.model_copy(update={"requested_depth": depth})
    return MCPEval.from_eval(
        raw,
        board.fen(),
        board=board,
        requested_depth=depth,
        history_complete=history_complete,
    )


async def _evaluate_one(
    pool: Any,
    board: chess.Board,
    *,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None,
) -> MCPEval:
    if evaluate_position is not None:
        evaluated, _cache_hit = await evaluate_position(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
        return evaluated
    raw = await pool.evaluate(board, depth=depth)
    return _as_mcpeval(raw, board, depth=depth, history_complete=history_complete)


async def _evaluate_pair(
    pool: Any,
    board_before: chess.Board,
    board_after: chess.Board,
    *,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None,
) -> tuple[MCPEval, MCPEval]:
    return (
        await _evaluate_one(
            pool,
            board_before,
            depth=depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        ),
        await _evaluate_one(
            pool,
            board_after,
            depth=depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        ),
    )


def _score_at_depth(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    board_after: chess.Board,
    played_move: chess.Move,
    eval_before: MCPEval,
    eval_after: MCPEval,
) -> Any:
    return score_played_move(
        board_before,
        played_move,
        eval_before,
        eval_after,
        board_after,
        action_type=result.action_type,
    )


def _comparison(
    result: ForensicMoveAnalysis,
    verified_before: MCPEval,
    verified_after: MCPEval,
    verified_score: Any,
    *,
    mover: chess.Color,
) -> dict[str, Any]:
    initial_class = _class_name(result.move_class)
    verified_class = _class_name(verified_score.move_class)
    initial_best = (result.eval_before.best_move or "").lower() or None
    verified_best = (verified_before.best_move or "").lower() or None
    initial_mate_before = _mate_signature(result.eval_before)
    initial_mate_after = _mate_signature(result.eval_after)
    verified_mate_before = _mate_signature(verified_before)
    verified_mate_after = _mate_signature(verified_after)
    return {
        "verified_class": verified_class,
        "verified_effective_loss": verified_score.effective_loss,
        "verified_centipawn_loss": verified_score.centipawn_loss,
        "verified_best_move_uci": verified_best,
        "verified_wdl_loss_percentage_points": _wdl_loss_percentage_points(
            verified_before,
            verified_after,
            mover,
        ),
        "verified_mate_before": verified_mate_before,
        "verified_mate_after": verified_mate_after,
        "class_stable": verified_class == initial_class,
        "best_move_stable": verified_best == initial_best,
        "mate_status_stable": (
            verified_mate_before == initial_mate_before
            and verified_mate_after == initial_mate_after
        ),
    }


async def verify_forensic_classification_stability(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
    pool: Any,
    depth: int,
    history_complete: str,
    evaluate_position: CachedEvaluator | None = None,
) -> ForensicMoveAnalysis:
    """Attach real higher-depth classification stability to forensic evidence.

    The helper is deliberately best-effort. If the extra verification search
    fails, the already-computed move analysis is returned with a status marker
    rather than converting an optional coaching check into a tool failure.
    """
    evidence = result.forensics
    if evidence is None or evidence.detail != "forensic":
        return result

    stability = dict(evidence.stability)
    initial_class = _class_name(result.move_class)
    stability.update(
        {
            "initial_depth": depth,
            "initial_class": initial_class,
            "initial_effective_loss": result.effective_loss,
            "initial_centipawn_loss": result.centipawn_loss,
            "initial_best_move_uci": (result.eval_before.best_move or "").lower() or None,
            "initial_wdl_loss_percentage_points": _wdl_loss_percentage_points(
                result.eval_before,
                result.eval_after,
                board_before.turn,
            ),
            "initial_mate_before": _mate_signature(result.eval_before),
            "initial_mate_after": _mate_signature(result.eval_after),
            "verification_performed": False,
            "verification_status": "not_needed",
            "verification_depth": None,
            "escalation_depth": None,
            "verification_converged": None,
            "stable": None,
            "verification_uses_cached_rule_aware_evaluator": evaluate_position is not None,
            "proof_scope": (
                "Selective multi-depth engine re-search of the same before/after positions. "
                "It tests classification stability, not whether the engine has solved the "
                "position or whether a human could find the verified move OTB."
            ),
        }
    )

    if result.action_type != "play_move" or played_move is None:
        stability["verification_status"] = "non_move_action"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    if initial_class not in _REVERIFY_CLASSES:
        stability["verification_status"] = "low_coaching_priority_class"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    if depth >= 30:
        stability["verification_status"] = "depth_cap_reached"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )

    board_after = board_before.copy(stack=True)
    if played_move not in board_after.legal_moves:
        stability["verification_status"] = "played_move_not_legal_on_reconstructed_board"
        return result.model_copy(
            update={"forensics": evidence.model_copy(update={"stability": stability})}
        )
    board_after.push(played_move)

    verification_depth = min(max(depth + 4, 24), 30)
    try:
        verified_before, verified_after = await _evaluate_pair(
            pool,
            board_before,
            board_after,
            depth=verification_depth,
            history_complete=history_complete,
            evaluate_position=evaluate_position,
        )
        verified_score = _score_at_depth(
            result,
            board_before,
            board_after,
            played_move,
            verified_before,
            verified_after,
        )
        first = _comparison(
            result,
            verified_before,
            verified_after,
            verified_score,
            mover=board_before.turn,
        )
        stability.update(first)
        stability["verification_performed"] = True
        stability["verification_status"] = "verified"
        stability["verification_depth"] = verification_depth

        disagreement = not (
            bool(first["class_stable"])
            and bool(first["best_move_stable"])
            and bool(first["mate_status_stable"])
        )
        final_score = verified_score
        final_before = verified_before
        final_after = verified_after
        verification_converged: bool | None = None

        if disagreement and verification_depth < 30:
            escalation_depth = min(verification_depth + 2, 30)
            escalated_before, escalated_after = await _evaluate_pair(
                pool,
                board_before,
                board_after,
                depth=escalation_depth,
                history_complete=history_complete,
                evaluate_position=evaluate_position,
            )
            escalated_score = _score_at_depth(
                result,
                board_before,
                board_after,
                played_move,
                escalated_before,
                escalated_after,
            )
            verification_converged = (
                _class_name(escalated_score.move_class) == _class_name(verified_score.move_class)
                and (escalated_before.best_move or "").lower()
                == (verified_before.best_move or "").lower()
                and _mate_signature(escalated_before) == _mate_signature(verified_before)
                and _mate_signature(escalated_after) == _mate_signature(verified_after)
            )
            stability["escalation_depth"] = escalation_depth
            stability["verification_converged"] = verification_converged
            final_score = escalated_score
            final_before = escalated_before
            final_after = escalated_after
            final = _comparison(
                result,
                final_before,
                final_after,
                final_score,
                mover=board_before.turn,
            )
            stability.update(final)
            stability["verification_depth"] = escalation_depth
            stability["verification_status"] = "escalated"

        class_stable = _class_name(final_score.move_class) == initial_class
        best_stable = (final_before.best_move or "").lower() == (
            result.eval_before.best_move or ""
        ).lower()
        mate_stable = (
            _mate_signature(final_before) == _mate_signature(result.eval_before)
            and _mate_signature(final_after) == _mate_signature(result.eval_after)
        )
        stability["class_stable"] = class_stable
        stability["best_move_stable"] = best_stable
        stability["mate_status_stable"] = mate_stable
        stability["stable"] = bool(
            class_stable
            and best_stable
            and mate_stable
            and (verification_converged is not False)
        )
    except Exception:
        stability["verification_performed"] = False
        stability["verification_status"] = "engine_verification_failed"

    return result.model_copy(
        update={"forensics": evidence.model_copy(update={"stability": stability})}
    )
