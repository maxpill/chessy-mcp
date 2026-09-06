"""Per-ply causal board-state trace for rich move coaching.

The engine PV says what both sides may play.  Coaching additionally needs to
show what each ply *changes*: material, defenders, pins, en-prise pieces,
activity, files, pawn structure and king pressure.  This module reconstructs
those deterministic board deltas from the already-returned principal variation.

No engine search is performed here.  The trace deliberately stops at the
adaptive forcing-resolution point when that evidence is available, otherwise it
is bounded by the returned PV and ``MAX_CAUSAL_TRACE_PLIES``.  It is evidence
for a coaching causal chain, not proof that one changed feature caused the
engine evaluation.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES
from mcp_server.analysis.position_integrity import (
    build_rich_position_delta,
    build_rich_tactical_snapshot,
)
from mcp_server.models.forensics import ForensicMoveAnalysis, PositionDelta, TacticalSnapshot

MAX_CAUSAL_TRACE_PLIES = 12
MAX_DETAIL_ITEMS = 10


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _piece_label(piece: chess.Piece | None, square: chess.Square | None = None) -> str | None:
    if piece is None:
        return None
    base = f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}"
    return f"{base}@{chess.square_name(square)}" if square is not None else base


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _mechanism_key(item: object) -> str:
    mechanism = str(getattr(item, "mechanism", ""))
    trigger = str(getattr(item, "trigger_uci", None) or "-")
    actor = str(getattr(item, "actor", None) or "-")
    targets = ",".join(str(value) for value in getattr(item, "targets", [])) or "-"
    return f"{mechanism}:trigger={trigger}:actor={actor}:targets={targets}"


def _hanging_key(item: object) -> str:
    target = getattr(item, "target", None)
    capture = getattr(item, "capture", None)
    if target is None or capture is None:
        return ""
    return (
        f"{getattr(target, 'color', '')}_{getattr(target, 'piece', '')}@"
        f"{getattr(target, 'square', '')}:capture={getattr(capture, 'uci', '')}:"
        f"reason={getattr(item, 'reason', '')}"
    )


def _piece_safety_changes(delta: PositionDelta) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for item in delta.piece_safety_changes[:MAX_DETAIL_ITEMS]:
        changes.append(
            {
                "target": item.target,
                "attackers": [item.attackers_before, item.attackers_after],
                "defenders": [item.defenders_before, item.defenders_after],
                "defender_loss": item.defenders_after < item.defenders_before,
                "new_attack": item.attackers_before == 0 and item.attackers_after > 0,
            }
        )
    return changes


def _mobility_changes(delta: PositionDelta) -> list[dict[str, Any]]:
    return [
        {
            "target": item.target,
            "mobility": [item.mobility_before, item.mobility_after],
            "gained_squares": item.gained_squares[:MAX_DETAIL_ITEMS],
            "lost_squares": item.lost_squares[:MAX_DETAIL_ITEMS],
        }
        for item in delta.piece_mobility_changes[:MAX_DETAIL_ITEMS]
    ]


def _control_changes(delta: PositionDelta) -> list[dict[str, Any]]:
    return [
        {
            "square": item.square,
            "white_attackers": [item.white_attackers_before, item.white_attackers_after],
            "black_attackers": [item.black_attackers_before, item.black_attackers_after],
        }
        for item in delta.strategic_square_control_changes[:MAX_DETAIL_ITEMS]
    ]


def _delta_summary(delta: PositionDelta) -> dict[str, Any]:
    return {
        "material_delta_white_cp": delta.material_delta_white,
        "material_delta_black_cp": delta.material_delta_black,
        "removed_pieces": delta.removed_pieces[:MAX_DETAIL_ITEMS],
        "added_pieces": delta.added_pieces[:MAX_DETAIL_ITEMS],
        "newly_loose_pieces": delta.newly_loose_pieces[:MAX_DETAIL_ITEMS],
        "newly_en_prise_pieces": delta.newly_en_prise_pieces[:MAX_DETAIL_ITEMS],
        "resolved_en_prise_pieces": delta.resolved_en_prise_pieces[:MAX_DETAIL_ITEMS],
        "newly_pinned_pieces": delta.newly_pinned_pieces[:MAX_DETAIL_ITEMS],
        "removed_pins": delta.removed_pins[:MAX_DETAIL_ITEMS],
        "piece_safety_changes": _piece_safety_changes(delta),
        "piece_mobility_changes": _mobility_changes(delta),
        "strategic_square_control_changes": _control_changes(delta),
        "opened_files": delta.opened_files,
        "closed_files": delta.closed_files,
        "pawn_structure_changes": delta.pawn_structure_changes[:MAX_DETAIL_ITEMS],
        "king_ring_attack_delta_white": delta.king_ring_attack_delta_white,
        "king_ring_attack_delta_black": delta.king_ring_attack_delta_black,
        "check_state_changed": delta.check_state_changed,
    }


def _causal_flags(delta: PositionDelta) -> list[str]:
    flags: list[str] = []
    if delta.material_delta_white or delta.material_delta_black:
        flags.append("material_changed")
    if delta.newly_en_prise_pieces:
        flags.append("new_en_prise_piece")
    if delta.resolved_en_prise_pieces:
        flags.append("en_prise_resolved")
    if delta.newly_pinned_pieces:
        flags.append("new_pin")
    if delta.removed_pins:
        flags.append("pin_removed")
    if any(item.defenders_after < item.defenders_before for item in delta.piece_safety_changes):
        flags.append("defender_count_dropped")
    if any(item.attackers_after > item.attackers_before for item in delta.piece_safety_changes):
        flags.append("attacker_count_increased")
    if delta.opened_files:
        flags.append("file_opened")
    if delta.closed_files:
        flags.append("file_closed")
    if delta.pawn_structure_changes:
        flags.append("pawn_structure_changed")
    if delta.king_ring_attack_delta_white or delta.king_ring_attack_delta_black:
        flags.append("king_ring_pressure_changed")
    return flags


def _adaptive_trace_limit(result: ForensicMoveAnalysis) -> int | None:
    evidence = result.forensics
    if evidence is None:
        return None
    for item in reversed(evidence.mechanism_evidence):
        if item.get("mechanism") != "adaptive_forcing_resolution":
            continue
        consumed = item.get("plies_consumed")
        if isinstance(consumed, int) and consumed >= 0:
            return consumed
    return None


def build_causal_position_trace(
    board_after_played: chess.Board,
    pv_uci: list[str],
    *,
    max_plies: int | None = None,
) -> dict[str, Any]:
    """Build a deterministic per-ply position-delta trace over a returned PV."""
    limit = MAX_CAUSAL_TRACE_PLIES if max_plies is None else max(0, min(max_plies, MAX_CAUSAL_TRACE_PLIES))
    work = board_after_played.copy(stack=True)
    steps: list[dict[str, Any]] = []
    termination_reason = "no_pv" if not pv_uci else "pv_exhausted"

    for ply, raw in enumerate(pv_uci[:limit], start=1):
        try:
            move = chess.Move.from_uci(str(raw).lower())
        except (ValueError, chess.InvalidMoveError):
            termination_reason = "invalid_pv_move"
            break
        if move not in work.legal_moves:
            termination_reason = "invalid_pv_move"
            break

        before = work.copy(stack=True)
        before_snapshot = build_rich_tactical_snapshot(before)
        side = _color_name(before.turn)
        san = before.san(move)
        captured = _captured_piece(before, move)
        is_check = before.gives_check(move)
        is_capture = before.is_capture(move)
        is_promotion = move.promotion is not None

        work.push(move)
        after_snapshot = build_rich_tactical_snapshot(work)
        delta = build_rich_position_delta(
            before,
            work,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )

        before_mechanisms = {_mechanism_key(item) for item in before_snapshot.mechanism_candidates}
        after_mechanisms = {_mechanism_key(item) for item in after_snapshot.mechanism_candidates}
        before_hanging = {_hanging_key(item) for item in before_snapshot.tactically_hanging_candidates}
        after_hanging = {_hanging_key(item) for item in after_snapshot.tactically_hanging_candidates}

        steps.append(
            {
                "ply": ply,
                "side": side,
                "uci": move.uci(),
                "san": san,
                "is_check": is_check,
                "is_capture": is_capture,
                "is_promotion": is_promotion,
                "captured_piece": _piece_label(captured),
                "position_after_fen": work.fen(),
                "position_after_hash": __import__("hashlib").sha256(work.fen().encode()).hexdigest()[:16],
                "delta": _delta_summary(delta),
                "causal_flags": _causal_flags(delta),
                "new_mechanism_candidates": sorted(after_mechanisms - before_mechanisms)[:MAX_DETAIL_ITEMS],
                "resolved_mechanism_candidates": sorted(before_mechanisms - after_mechanisms)[:MAX_DETAIL_ITEMS],
                "new_tactical_hanging_candidates": sorted(after_hanging - before_hanging)[:MAX_DETAIL_ITEMS],
                "resolved_tactical_hanging_candidates": sorted(before_hanging - after_hanging)[:MAX_DETAIL_ITEMS],
            }
        )

        if work.is_game_over(claim_draw=False):
            termination_reason = "terminal_position"
            break

    if len(pv_uci) > limit and termination_reason == "pv_exhausted":
        termination_reason = "trace_limit"

    return {
        "mechanism": "causal_position_delta_trace",
        "steps": steps,
        "pv_plies_available": len(pv_uci),
        "plies_traced": len(steps),
        "termination_reason": termination_reason,
        "proof_scope": (
            "Deterministic board-state deltas over the already-returned principal variation. "
            "The trace shows what changed after each ply but does not prove that a particular "
            "delta caused the engine-evaluation change or that the principal variation is forced."
        ),
    }


def apply_causal_position_trace(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    """Attach the causal trace to rich ``classify_move`` evidence."""
    evidence = result.forensics
    if evidence is None or result.action_type != "play_move" or played_move is None:
        return result
    if played_move not in board_before.legal_moves:
        return result

    board_after = board_before.copy(stack=True)
    board_after.push(played_move)
    adaptive_limit = _adaptive_trace_limit(result)
    if adaptive_limit is None:
        adaptive_limit = len(evidence.forced_line.uci)
    trace = build_causal_position_trace(
        board_after,
        list(evidence.forced_line.uci),
        max_plies=adaptive_limit,
    )

    mechanisms = list(evidence.mechanism_evidence)
    mechanisms.append(trace)
    signatures = list(evidence.evidence_signatures)
    steps = trace["steps"]
    if steps:
        signatures.append("CAUSAL_POSITION_DELTA_TRACE_AVAILABLE")
    if any("material_changed" in step["causal_flags"] for step in steps):
        signatures.append("FORCED_LINE_MATERIAL_CHANGE")
    if any("defender_count_dropped" in step["causal_flags"] for step in steps):
        signatures.append("FORCED_LINE_DEFENDER_COUNT_DROPPED")
    if any("new_en_prise_piece" in step["causal_flags"] for step in steps):
        signatures.append("FORCED_LINE_NEW_EN_PRISE_PIECE")
    if any("new_pin" in step["causal_flags"] for step in steps):
        signatures.append("FORCED_LINE_NEW_PIN")

    upgraded = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
