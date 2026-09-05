"""Evidence-first full-game coaching analysis.

This module turns the eval trajectory already produced by ``analyze_game`` into
structured coaching evidence: phases, pedagogically useful critical moments,
recovery/conversion events, positive resources, root-cause/materialization
links and a final-position defensibility snapshot. ``forensic`` mode selectively
re-searches only the chosen moments at higher depth.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

import chess
import chess.pgn

from mcp_server.analysis.position_integrity import build_rich_tactical_snapshot
from mcp_server.models import MCPEval
from mcp_server.models.game_coaching import (
    AdvantageEvent,
    CriticalMoment,
    FinalPositionAssessment,
    GameCoachingEvidence,
    GameSegment,
    PositiveMoment,
    RootCauseLink,
)
from mcp_server.move_grading import score_played_move

Perspective = Literal["white", "black"]
PositionState = Literal[
    "decisively_better",
    "better",
    "slightly_better",
    "approximately_equal",
    "slightly_worse",
    "worse",
    "decisively_worse",
]

MATE_VALUE = 100_000
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
STATE_ORDER: dict[PositionState, int] = {
    "decisively_worse": 0,
    "worse": 1,
    "slightly_worse": 2,
    "approximately_equal": 3,
    "slightly_better": 4,
    "better": 5,
    "decisively_better": 6,
}


@dataclass
class _PlyRecord:
    ply: int
    board_before: chess.Board
    board_after: chess.Board
    move: chess.Move
    san: str
    side: Perspective
    score: Any
    before_cp: int
    after_cp: int
    best_move_san: str | None
    user_comment_raw: str | None


def _side_name(color: chess.Color) -> Perspective:
    return "white" if color == chess.WHITE else "black"


def _white_effective_cp(ev: MCPEval) -> int:
    if ev.mate is not None:
        if ev.mate == 0:
            return ev.cp if ev.cp is not None else MATE_VALUE
        return (MATE_VALUE - min(abs(ev.mate), MATE_VALUE - 1)) * (1 if ev.mate > 0 else -1)
    return ev.cp if ev.cp is not None else 0


def _perspective_cp(ev: MCPEval, perspective: Perspective) -> int:
    value = _white_effective_cp(ev)
    return value if perspective == "white" else -value


def _perspective_raw_value(cp: int | None, mate: int | None, perspective: Perspective) -> int:
    if mate is not None:
        value = (MATE_VALUE - min(abs(mate), MATE_VALUE - 1)) * (1 if mate > 0 else -1)
    else:
        value = cp or 0
    return value if perspective == "white" else -value


def _position_state(value: int) -> PositionState:
    if value >= 350:
        return "decisively_better"
    if value >= 150:
        return "better"
    if value >= 60:
        return "slightly_better"
    if value > -60:
        return "approximately_equal"
    if value > -150:
        return "slightly_worse"
    if value > -350:
        return "worse"
    return "decisively_worse"


def _confirm_segment_states(
    raw_states: list[PositionState],
) -> tuple[list[PositionState], dict[int, int]]:
    """Suppress one-ply threshold flicker without hiding large state jumps.

    A one-band transition needs two consecutive plies before it becomes a new
    game phase. A jump of at least two bands, or any jump directly into a
    decisive state, is accepted immediately. Confirmed transitions are applied
    retroactively to the first ply on which the new state appeared.
    """
    if not raw_states:
        return [], {}

    current = raw_states[0]
    confirmed: list[PositionState] = [current]
    transition_confirmed_ply: dict[int, int] = {}
    pending_state: PositionState | None = None
    pending_start_ply: int | None = None
    pending_count = 0

    for idx, observed in enumerate(raw_states[1:], start=1):
        ply = idx + 1
        if observed == current:
            pending_state = None
            pending_start_ply = None
            pending_count = 0
            confirmed.append(current)
            continue

        jump = abs(STATE_ORDER[observed] - STATE_ORDER[current])
        immediate = jump >= 2 or observed in {"decisively_better", "decisively_worse"}
        if immediate:
            current = observed
            transition_confirmed_ply[ply] = ply
            pending_state = None
            pending_start_ply = None
            pending_count = 0
            confirmed.append(current)
            continue

        if pending_state == observed:
            pending_count += 1
        else:
            pending_state = observed
            pending_start_ply = ply
            pending_count = 1

        confirmed.append(current)
        if pending_count < 2 or pending_start_ply is None:
            continue

        current = observed
        start_index = pending_start_ply - 1
        for rewrite in range(start_index, len(confirmed)):
            confirmed[rewrite] = current
        transition_confirmed_ply[pending_start_ply] = ply
        pending_state = None
        pending_start_ply = None
        pending_count = 0

    return confirmed, transition_confirmed_ply


def _segment_stability(
    raw_states: list[PositionState],
    confirmed_state: PositionState,
) -> tuple[Literal["high", "medium", "low"], int]:
    if not raw_states:
        return "high", 0
    raw_changes = sum(left != right for left, right in pairwise(raw_states))
    matching = sum(state == confirmed_state for state in raw_states) / len(raw_states)
    if matching >= 0.8 and raw_changes <= 1:
        return "high", raw_changes
    if matching >= 0.6:
        return "medium", raw_changes
    return "low", raw_changes


def _mainline_comments(game: chess.pgn.Game) -> dict[int, str]:
    comments: dict[int, str] = {}
    node: chess.pgn.GameNode = game
    ply = 0
    while node.variations:
        node = node.variations[0]
        ply += 1
        comment = str(getattr(node, "comment", "") or "").strip()
        starting = str(getattr(node, "starting_comment", "") or "").strip()
        combined = "\n".join(part for part in (starting, comment) if part)
        if combined:
            comments[ply] = combined
    return comments


def _best_san(board: chess.Board, ev: MCPEval) -> str | None:
    if not ev.best_move:
        return None
    try:
        move = chess.Move.from_uci(ev.best_move.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if move not in board.legal_moves:
        return None
    return board.san(move)


def _build_records(
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
    *,
    perspective: Perspective,
    comments: dict[int, str],
) -> list[_PlyRecord]:
    records: list[_PlyRecord] = []
    for ply, move in enumerate(moves, start=1):
        before = positions[ply - 1]
        after = positions[ply]
        score = score_played_move(before, move, evals[ply - 1], evals[ply], after)
        records.append(
            _PlyRecord(
                ply=ply,
                board_before=before,
                board_after=after,
                move=move,
                san=before.san(move),
                side=_side_name(before.turn),
                score=score,
                before_cp=_perspective_cp(evals[ply - 1], perspective),
                after_cp=_perspective_cp(evals[ply], perspective),
                best_move_san=_best_san(before, evals[ply - 1]),
                user_comment_raw=comments.get(ply),
            )
        )
    return records


def _build_segments(evals: list[MCPEval], *, perspective: Perspective) -> list[GameSegment]:
    if len(evals) <= 1:
        return []

    values = [_perspective_cp(ev, perspective) for ev in evals]
    raw_states: list[PositionState] = [
        cast(PositionState, _position_state(value)) for value in values[1:]
    ]
    states, transition_confirmed = _confirm_segment_states(raw_states)
    if not states:
        return []

    segments: list[GameSegment] = []
    start = 1
    current = states[0]
    for ply in range(2, len(evals)):
        state = states[ply - 1]
        if state == current:
            continue
        end = ply - 1
        raw_slice = raw_states[start - 1 : end]
        stability, raw_changes = _segment_stability(raw_slice, current)
        segment_values = values[start : end + 1]
        segments.append(
            GameSegment(
                start_ply=start,
                end_ply=end,
                perspective=perspective,
                state=current,
                eval_start_effective_cp=values[start],
                eval_end_effective_cp=values[end],
                eval_peak_effective_cp=max(segment_values),
                eval_trough_effective_cp=min(segment_values),
                transition_cause_ply=start if start > 1 else None,
                transition_confirmed_ply=(
                    transition_confirmed.get(start) if start > 1 else None
                ),
                stability=stability,
                raw_state_change_count=raw_changes,
            )
        )
        start = ply
        current = state

    end = len(evals) - 1
    raw_slice = raw_states[start - 1 : end]
    stability, raw_changes = _segment_stability(raw_slice, current)
    segment_values = values[start : end + 1]
    segments.append(
        GameSegment(
            start_ply=start,
            end_ply=end,
            perspective=perspective,
            state=current,
            eval_start_effective_cp=values[start],
            eval_end_effective_cp=values[end],
            eval_peak_effective_cp=max(segment_values),
            eval_trough_effective_cp=min(segment_values),
            transition_cause_ply=start if start > 1 else None,
            transition_confirmed_ply=transition_confirmed.get(start) if start > 1 else None,
            stability=stability,
            raw_state_change_count=raw_changes,
        )
    )
    return segments


def _build_advantage_events(
    records: list[_PlyRecord],
    evals: list[MCPEval],
    *,
    perspective: Perspective,
) -> list[AdvantageEvent]:
    events: list[AdvantageEvent] = []
    for idx, record in enumerate(records):
        before = record.before_cp
        after = record.after_cp
        kinds: list[str] = []
        if before < 150 <= after:
            kinds.append("gained_advantage")
        if before >= 150 and after < 75:
            kinds.append("lost_advantage")
            if record.side == perspective:
                kinds.append("missed_conversion")
        if before > -150 and after <= -150:
            kinds.append("fell_behind")
        if before <= -150 and after > -75:
            kinds.append("recovered")

        if idx >= 1 and record.side == perspective:
            two_plies_before = _perspective_cp(evals[record.ply - 2], perspective)
            if two_plies_before <= -150 and before >= -75 and after <= -150:
                kinds.append("missed_recovery")

        for kind in dict.fromkeys(kinds):
            events.append(
                AdvantageEvent(
                    ply=record.ply,
                    san=record.san,
                    side=record.side,
                    perspective=perspective,
                    kind=kind,  # type: ignore[arg-type]
                    before_effective_cp=before,
                    after_effective_cp=after,
                    evidence={"delta_effective_cp": after - before},
                )
            )
    return events


def _importance(record: _PlyRecord) -> float:
    loss = record.score.effective_loss
    if loss is None:
        loss = record.score.centipawn_loss or 0
    bonus = {"blunder": 500, "mistake": 280, "inaccuracy": 80}.get(
        record.score.move_class.value, 0
    )
    comment_bonus = 120 if record.user_comment_raw else 0
    return float(max(0, loss) + bonus + comment_bonus)


def _select_critical_moments(
    records: list[_PlyRecord],
    events: list[AdvantageEvent],
    *,
    perspective: Perspective,
    max_moments: int,
) -> list[CriticalMoment]:
    own = [record for record in records if record.side == perspective]
    if not own:
        return []

    reasons_by_ply: dict[int, list[str]] = {}

    significant = [
        record
        for record in own
        if (record.score.effective_loss or 0) >= 100
        or record.score.move_class.value in {"mistake", "blunder"}
    ]
    if significant:
        reasons_by_ply.setdefault(significant[0].ply, []).append("first_significant_error")
        largest = max(significant, key=_importance)
        reasons_by_ply.setdefault(largest.ply, []).append("largest_error")
        reasons_by_ply.setdefault(significant[-1].ply, []).append("last_significant_error")

    for event in events:
        if event.side != perspective:
            continue
        if event.kind in {"missed_recovery", "missed_conversion", "fell_behind", "lost_advantage"}:
            reasons_by_ply.setdefault(event.ply, []).append(event.kind)

    for record in own:
        if record.user_comment_raw:
            reasons_by_ply.setdefault(record.ply, []).append("player_self_report")

    ranked = sorted(own, key=_importance, reverse=True)
    for record in ranked:
        if len(reasons_by_ply) >= max_moments:
            break
        if _importance(record) >= 100:
            reasons_by_ply.setdefault(record.ply, []).append("high_engine_loss")

    selected_records = [record for record in own if record.ply in reasons_by_ply]
    selected_records = sorted(selected_records, key=_importance, reverse=True)[:max_moments]
    selected_records.sort(key=lambda record: record.ply)

    moments: list[CriticalMoment] = []
    for record in selected_records:
        moments.append(
            CriticalMoment(
                ply=record.ply,
                san=record.san,
                uci=record.move.uci(),
                side=record.side,
                move_class=record.score.move_class.value,
                centipawn_loss=record.score.centipawn_loss,
                effective_loss=record.score.effective_loss,
                eval_before_effective_cp=record.before_cp,
                eval_after_effective_cp=record.after_cp,
                best_move_san=record.best_move_san,
                reasons=sorted(set(reasons_by_ply[record.ply])),
                importance_score=round(_importance(record), 1),
                user_comment_raw=record.user_comment_raw,
            )
        )
    return moments


def _select_positive_moments(
    records: list[_PlyRecord],
    *,
    perspective: Perspective,
) -> list[PositiveMoment]:
    candidates: list[tuple[float, PositiveMoment]] = []
    for record in records:
        if record.side != perspective or not record.score.is_best_engine_move:
            continue
        reason: str | None = None
        weight = 0.0
        if record.before_cp <= -150 and record.after_cp >= record.before_cp - 50:
            reason = "held_difficult_position"
            weight = abs(record.before_cp) + 150
        elif record.before_cp >= 150 and record.after_cp >= record.before_cp - 75:
            reason = "converted_advantage_without_slippage"
            weight = record.before_cp
        elif abs(record.before_cp) >= 100:
            reason = "best_engine_move_under_pressure"
            weight = abs(record.before_cp)
        if reason is None:
            continue
        candidates.append(
            (
                weight,
                PositiveMoment(
                    ply=record.ply,
                    san=record.san,
                    uci=record.move.uci(),
                    side=record.side,
                    eval_before_effective_cp=record.before_cp,
                    eval_after_effective_cp=record.after_cp,
                    reason=reason,  # type: ignore[arg-type]
                ),
            )
        )
    return [item for _weight, item in sorted(candidates, key=lambda pair: pair[0], reverse=True)[:3]]


def _material_balance(board: chess.Board, perspective: Perspective) -> int:
    white = 0
    black = 0
    for piece in board.piece_map().values():
        if piece.color == chess.WHITE:
            white += PIECE_VALUES[piece.piece_type]
        else:
            black += PIECE_VALUES[piece.piece_type]
    value = white - black
    return value if perspective == "white" else -value


def _build_root_cause_links(
    critical: list[CriticalMoment],
    records: list[_PlyRecord],
    positions: list[chess.Board],
    *,
    perspective: Perspective,
) -> list[RootCauseLink]:
    by_ply = {record.ply: record for record in records}
    links: list[RootCauseLink] = []
    for moment in critical:
        if (moment.effective_loss or moment.centipawn_loss or 0) < 100:
            continue
        baseline = _material_balance(positions[moment.ply - 1], perspective)
        for materialization_ply in range(moment.ply + 1, min(len(records), moment.ply + 6) + 1):
            balance = _material_balance(positions[materialization_ply], perspective)
            adverse = baseline - balance
            if adverse < 100:
                continue
            record = by_ply[materialization_ply]
            links.append(
                RootCauseLink(
                    root_cause_ply=moment.ply,
                    materialization_ply=materialization_ply,
                    root_cause_san=moment.san,
                    materialization_san=record.san,
                    affected_side=perspective,
                    material_swing_cp=adverse,
                    plies_later=materialization_ply - moment.ply,
                )
            )
            break
    return links


def _uniqueness(gap: int | None) -> Literal["unknown", "low", "medium", "high"]:
    if gap is None:
        return "unknown"
    if gap >= 150:
        return "high"
    if gap >= 75:
        return "medium"
    return "low"


def _candidate_gap(top: list[Any], side: chess.Color) -> int | None:
    if len(top) < 2:
        return None
    perspective: Perspective = "white" if side == chess.WHITE else "black"
    first = _perspective_raw_value(top[0].cp, top[0].mate, perspective)
    second = _perspective_raw_value(top[1].cp, top[1].mate, perspective)
    return max(0, first - second)


def _strongest_reply_fact(board: chess.Board, ev: MCPEval) -> tuple[str, str, bool, bool] | None:
    if not ev.best_move or board.is_game_over(claim_draw=False):
        return None
    try:
        move = chess.Move.from_uci(ev.best_move.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if move not in board.legal_moves:
        return None
    return move.uci(), board.san(move), board.gives_check(move), board.is_capture(move)


def _piece_evidence_label(item: Any) -> str:
    return f"{item.color}_{item.piece}@{item.square}"


def _hanging_target_label(item: Any) -> str:
    return _piece_evidence_label(item.target)


async def _verify_critical_moments(
    critical: list[CriticalMoment],
    records: list[_PlyRecord],
    positions: list[chess.Board],
    *,
    perspective: Perspective,
    scan_depth: int,
    pool: Any,
    evaluate_positions: Callable[..., Awaitable[list[tuple[MCPEval, bool]]]],
) -> tuple[list[CriticalMoment], int, int | None]:
    if not critical:
        return critical, min(max(scan_depth + 4, 22), 26), None

    verification_depth = 22 if scan_depth <= 18 else 24 if scan_depth <= 20 else min(scan_depth + 2, 26)
    record_by_ply = {record.ply: record for record in records}
    unique_indices = sorted({index for moment in critical for index in (moment.ply - 1, moment.ply)})
    searched = await evaluate_positions(
        [positions[index] for index in unique_indices],
        verification_depth,
        pool,
        requested_depth=verification_depth,
        history_complete="complete",
    )
    verified = {index: item[0] for index, item in zip(unique_indices, searched, strict=True)}

    escalated_depth: int | None = None
    out: list[CriticalMoment] = []
    for moment in critical:
        record = record_by_ply[moment.ply]
        before_ev = verified[moment.ply - 1]
        after_ev = verified[moment.ply]
        score = score_played_move(
            record.board_before,
            record.move,
            before_ev,
            after_ev,
            record.board_after,
        )
        stable = score.move_class.value == moment.move_class
        effective_loss = score.effective_loss
        depth_used = verification_depth

        if not stable and verification_depth < 26:
            escalation = min(verification_depth + 2, 26)
            pair = await evaluate_positions(
                [positions[moment.ply - 1], positions[moment.ply]],
                escalation,
                pool,
                requested_depth=escalation,
                history_complete="complete",
            )
            before_ev, after_ev = pair[0][0], pair[1][0]
            score = score_played_move(
                record.board_before,
                record.move,
                before_ev,
                after_ev,
                record.board_after,
            )
            stable = score.move_class.value == moment.move_class
            effective_loss = score.effective_loss
            depth_used = escalation
            escalated_depth = max(escalated_depth or 0, escalation)

        gap: int | None = None
        try:
            top = await pool.top_moves(record.board_before, n=2, depth=depth_used)
            gap = _candidate_gap(list(top), record.board_before.turn)
        except Exception:
            gap = None

        before_snapshot = build_rich_tactical_snapshot(record.board_before)
        after_snapshot = build_rich_tactical_snapshot(record.board_after)
        before_en_prise = {
            _piece_evidence_label(item)
            for item in before_snapshot.en_prise_pieces
            if item.color == perspective
        }
        after_en_prise = {
            _piece_evidence_label(item)
            for item in after_snapshot.en_prise_pieces
            if item.color == perspective
        }
        newly_en_prise = sorted(after_en_prise - before_en_prise)

        before_hanging = {
            _hanging_target_label(item)
            for item in before_snapshot.tactically_hanging_candidates
            if item.target.color == perspective
        }
        after_hanging = {
            _hanging_target_label(item)
            for item in after_snapshot.tactically_hanging_candidates
            if item.target.color == perspective
        }
        newly_hanging = sorted(after_hanging - before_hanging)

        played_piece_obj = record.board_before.piece_at(record.move.from_square)
        played_piece = PIECE_NAMES.get(played_piece_obj.piece_type) if played_piece_obj else None
        only_move_missed = bool(
            gap is not None and gap >= 150 and not score.is_best_engine_move
        )

        signatures = list(moment.evidence_signatures)
        if newly_en_prise:
            signatures.append("NEW_EN_PRISE_PIECE_AFTER_MOVE")
        if newly_hanging:
            signatures.append("NEW_TACTICALLY_HANGING_CANDIDATE_AFTER_MOVE")
        if only_move_missed:
            signatures.append("ONLY_MOVE_MISSED_CANDIDATE")

        reply = _strongest_reply_fact(record.board_after, after_ev)
        update: dict[str, Any] = {
            "verification_depth": depth_used,
            "verified_move_class": score.move_class.value,
            "verified_effective_loss": effective_loss,
            "classification_stable": stable,
            "candidate_gap_effective_cp": gap,
            "resource_uniqueness": _uniqueness(gap),
            "played_piece": played_piece,
            "only_move_missed_candidate": only_move_missed,
            "newly_en_prise_user_pieces": newly_en_prise,
            "newly_tactically_hanging_user_targets": newly_hanging,
        }
        if reply is not None:
            uci, san, is_check, is_capture = reply
            update.update(
                {
                    "strongest_reply_uci": uci,
                    "strongest_reply_san": san,
                    "strongest_reply_is_check": is_check,
                    "strongest_reply_is_capture": is_capture,
                }
            )
            if is_check:
                signatures.append("FORCING_CHECK_REPLY")
            if is_capture:
                signatures.append("FORCING_CAPTURE_REPLY")
            if is_check and is_capture:
                signatures.append("CHECK_CAPTURE_REPLY")
            if (effective_loss or 0) >= 100 and (is_check or is_capture):
                signatures.append("MISSED_FORCING_REPLY_CANDIDATE")
            if (
                played_piece == "pawn"
                and (effective_loss or 0) >= 100
                and (is_check or is_capture)
            ):
                signatures.append("PAWN_MOVE_FORCING_PUNISHMENT")
            if moment.user_comment_raw and (is_check or is_capture):
                signatures.append("PLAYER_SELF_REPORT_WITH_FORCING_REPLY")
        if moment.user_comment_raw:
            signatures.append("PLAYER_SELF_REPORT_PRESENT")
        update["evidence_signatures"] = sorted(set(signatures))
        out.append(moment.model_copy(update=update))
    return out, verification_depth, escalated_depth


async def _verify_positive_moments(
    positive: list[PositiveMoment],
    records: list[_PlyRecord],
    *,
    pool: Any,
    depth: int,
) -> list[PositiveMoment]:
    by_ply = {record.ply: record for record in records}
    out: list[PositiveMoment] = []
    for moment in positive:
        record = by_ply[moment.ply]
        try:
            top = list(await pool.top_moves(record.board_before, n=2, depth=depth))
            gap = _candidate_gap(top, record.board_before.turn)
        except Exception:
            gap = None
        reason = moment.reason
        if gap is not None and gap >= 150:
            reason = "unique_resource"
        out.append(
            moment.model_copy(
                update={
                    "candidate_gap_effective_cp": gap,
                    "resource_uniqueness": _uniqueness(gap),
                    "reason": reason,
                }
            )
        )
    return out


async def _final_assessment(
    board: chess.Board,
    ev: MCPEval,
    *,
    perspective: Perspective,
    detail: Literal["coach", "forensic"],
    pool: Any,
    verification_depth: int | None,
) -> FinalPositionAssessment:
    legal_count = board.legal_moves.count()
    best_uci = ev.best_move
    best_san: str | None = None
    if best_uci:
        try:
            move = chess.Move.from_uci(best_uci.lower())
            if move in board.legal_moves:
                best_san = board.san(move)
            else:
                best_uci = None
        except (ValueError, chess.InvalidMoveError):
            best_uci = None

    reasonable_count: int | None = None
    if detail == "forensic" and not board.is_game_over(claim_draw=False):
        try:
            top = list(await pool.top_moves(board, n=min(5, legal_count), depth=verification_depth or 22))
            if top:
                side = board.turn
                side_perspective: Perspective = "white" if side == chess.WHITE else "black"
                values = [
                    _perspective_raw_value(item.cp, item.mate, side_perspective) for item in top
                ]
                best = values[0]
                reasonable_count = sum(1 for value in values if best - value <= 100)
        except Exception:
            reasonable_count = None

    return FinalPositionAssessment(
        perspective=perspective,
        position_terminal_by_rules=board.is_game_over(claim_draw=False),
        checkmate=board.is_checkmate(),
        stalemate=board.is_stalemate(),
        forced_mate=board.is_checkmate() or ev.mate is not None,
        mate_distance=ev.mate,
        effective_cp=_perspective_cp(ev, perspective),
        wdl=ev.wdl,
        side_to_move=_side_name(board.turn),
        legal_move_count=legal_count,
        best_move_uci=best_uci,
        best_move_san=best_san,
        defensive_resources_exist=not board.is_game_over(claim_draw=False) and legal_count > 0,
        reasonable_resource_count=reasonable_count,
        verification_depth=verification_depth if detail == "forensic" else None,
    )


async def build_game_coaching_evidence(
    *,
    positions: list[chess.Board],
    moves: list[chess.Move],
    evals: list[MCPEval],
    game: chess.pgn.Game,
    perspective: Perspective,
    detail: Literal["coach", "forensic"],
    max_critical_moments: int,
    scan_depth: int,
    pool: Any,
    evaluate_positions: Callable[..., Awaitable[list[tuple[MCPEval, bool]]]],
) -> GameCoachingEvidence:
    comments = _mainline_comments(game)
    records = _build_records(
        positions,
        moves,
        evals,
        perspective=perspective,
        comments=comments,
    )
    segments = _build_segments(evals, perspective=perspective)
    events = _build_advantage_events(records, evals, perspective=perspective)
    critical = _select_critical_moments(
        records,
        events,
        perspective=perspective,
        max_moments=max(1, min(max_critical_moments, 7)),
    )
    positive = _select_positive_moments(records, perspective=perspective)
    root_links = _build_root_cause_links(
        critical,
        records,
        positions,
        perspective=perspective,
    )

    verification_depth: int | None = None
    escalation_depth: int | None = None
    if detail == "forensic":
        critical, verification_depth, escalation_depth = await _verify_critical_moments(
            critical,
            records,
            positions,
            perspective=perspective,
            scan_depth=scan_depth,
            pool=pool,
            evaluate_positions=evaluate_positions,
        )
        positive = await _verify_positive_moments(
            positive,
            records,
            pool=pool,
            depth=verification_depth,
        )

    final_position = await _final_assessment(
        positions[-1],
        evals[-1],
        perspective=perspective,
        detail=detail,
        pool=pool,
        verification_depth=verification_depth,
    )

    return GameCoachingEvidence(
        detail=detail,
        perspective=perspective,
        critical_moments=critical,
        game_segments=segments,
        advantage_events=events,
        positive_moments=positive,
        root_cause_links=root_links,
        final_position=final_position,
        scan_depth=scan_depth,
        verification_depth=verification_depth,
        adaptive_escalation_depth=escalation_depth,
    )
