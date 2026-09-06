"""Evidence-bounded continuation endpoints for explicit candidate comparisons.

Candidate comparison is most useful when it compares positions that actually
arise after the returned continuation, not only the board one ply after each
root move. This module walks only already returned engine PV moves, stops at a
local forcing-resolution point when that evidence is available, and compares
those endpoint positions without performing another engine search.

The endpoint is principal-variation evidence. It is not a proof that the line is
forced, and a ``pv_exhausted`` endpoint must not be described as a quiet point.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensics import PIECE_NAMES, build_position_fingerprint
from mcp_server.analysis.position_integrity import (
    build_rich_position_delta,
    build_rich_tactical_snapshot,
)
from mcp_server.analysis.tactical_snapshot_extensions import extend_tactical_snapshot
from mcp_server.models.forensics import (
    CandidateContinuationEndpointDifference,
    CandidateContinuationEndpointEvidence,
    CandidateEvidence,
    PositionDelta,
    TacticalSnapshot,
)


def _legal_uci(board: chess.Board, raw: str | None) -> chess.Move | None:
    if not raw:
        return None
    try:
        move = chess.Move.from_uci(str(raw).lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    return move if move in board.legal_moves else None


def _forcing(move_board: chess.Board, move: chess.Move) -> bool:
    return (
        move_board.gives_check(move)
        or move_board.is_capture(move)
        or move.promotion is not None
    )


def _forcing_available(board: chess.Board) -> bool:
    return any(_forcing(board, move) for move in board.legal_moves)


def _captured_piece_label(board: chess.Board, move: chess.Move) -> str | None:
    if not board.is_capture(move):
        return None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    piece = board.piece_at(square)
    if piece is None:
        return None
    color = "white" if piece.color == chess.WHITE else "black"
    return f"{color}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def _irreversible_event(
    board: chess.Board,
    move: chess.Move,
    *,
    ply_from_root: int,
) -> dict[str, Any] | None:
    piece = board.piece_at(move.from_square)
    reasons: list[str] = []
    if piece is not None and piece.piece_type == chess.PAWN:
        reasons.append("pawn_move")
    if board.is_capture(move):
        reasons.append("capture")
    if move.promotion is not None:
        reasons.append("promotion")
    if board.is_castling(move):
        reasons.append("castling")

    before_rights = board.castling_xfen() or "-"
    post = board.copy(stack=True)
    post.push(move)
    after_rights = post.castling_xfen() or "-"
    if before_rights != after_rights:
        reasons.append(f"castling_rights:{before_rights}->{after_rights}")
    if not reasons:
        return None

    return {
        "ply_from_root": ply_from_root,
        "uci": move.uci(),
        "san": board.san(move),
        "reasons": reasons,
        "captured_piece": _captured_piece_label(board, move),
    }


def _walk_endpoint(
    root: chess.Board,
    candidate: CandidateEvidence,
) -> tuple[chess.Board, dict[str, Any], list[dict[str, Any]]]:
    """Walk candidate + returned PV to a bounded local endpoint."""
    root_move = _legal_uci(root, candidate.uci)
    if root_move is None:
        return root.copy(stack=True), {
            "plies_from_candidate": 0,
            "plies_from_root": 0,
            "termination_reason": "invalid_candidate_move",
            "tactical_sequence_resolved": False,
            "returned_pv_plies_available": len(candidate.continuation_uci),
        }, []

    events: list[dict[str, Any]] = []
    event = _irreversible_event(root, root_move, ply_from_root=1)
    if event is not None:
        events.append(event)

    forcing_seen = _forcing(root, root_move)
    capture_or_promotion_seen = root.is_capture(root_move) or root_move.promotion is not None
    work = root.copy(stack=True)
    work.push(root_move)

    if work.is_checkmate():
        return work, {
            "plies_from_candidate": 0,
            "plies_from_root": 1,
            "termination_reason": "forced_mate_in_returned_pv",
            "tactical_sequence_resolved": True,
            "returned_pv_plies_available": len(candidate.continuation_uci),
        }, events
    if work.is_game_over(claim_draw=False):
        return work, {
            "plies_from_candidate": 0,
            "plies_from_root": 1,
            "termination_reason": "terminal_position",
            "tactical_sequence_resolved": True,
            "returned_pv_plies_available": len(candidate.continuation_uci),
        }, events

    # A forcing root move can itself settle the local tactic. Do not consume a
    # quiet engine continuation merely to reach an arbitrary fixed PV length.
    if forcing_seen and not work.is_check() and not _forcing_available(work):
        return work, {
            "plies_from_candidate": 0,
            "plies_from_root": 1,
            "termination_reason": (
                "material_resolution" if capture_or_promotion_seen else "quiet_position"
            ),
            "tactical_sequence_resolved": True,
            "returned_pv_plies_available": len(candidate.continuation_uci),
        }, events

    walked = 0
    termination_reason = "no_pv" if not candidate.continuation_uci else "pv_exhausted"
    resolved = False
    for raw in candidate.continuation_uci:
        move = _legal_uci(work, raw)
        if move is None:
            termination_reason = "invalid_pv_move"
            break

        walked += 1
        event = _irreversible_event(work, move, ply_from_root=walked + 1)
        if event is not None:
            events.append(event)
        forcing = _forcing(work, move)
        forcing_seen = forcing_seen or forcing
        capture_or_promotion_seen = (
            capture_or_promotion_seen or work.is_capture(move) or move.promotion is not None
        )
        work.push(move)

        if work.is_checkmate():
            termination_reason = "forced_mate_in_returned_pv"
            resolved = True
            break
        if work.is_game_over(claim_draw=False):
            termination_reason = "terminal_position"
            resolved = True
            break
        if forcing_seen and not work.is_check() and not _forcing_available(work):
            termination_reason = (
                "material_resolution" if capture_or_promotion_seen else "quiet_position"
            )
            resolved = True
            break

    return work, {
        "plies_from_candidate": walked,
        "plies_from_root": walked + 1,
        "termination_reason": termination_reason,
        "tactical_sequence_resolved": resolved,
        "returned_pv_plies_available": len(candidate.continuation_uci),
    }, events


def attach_candidate_continuation_endpoint(
    root: chess.Board,
    candidate: CandidateEvidence,
    *,
    root_snapshot: TacticalSnapshot | None = None,
) -> CandidateEvidence:
    """Attach the endpoint reached by the candidate's already returned PV."""
    endpoint, meta, events = _walk_endpoint(root, candidate)
    if meta["termination_reason"] == "invalid_candidate_move":
        typed = CandidateContinuationEndpointEvidence(
            **meta,
            proof_scope=(
                "Candidate root move was not legal in the supplied root board; no "
                "continuation endpoint was reconstructed."
            ),
        )
        return candidate.model_copy(update={"continuation_endpoint": typed})

    root_snapshot = root_snapshot or extend_tactical_snapshot(
        root,
        build_rich_tactical_snapshot(root),
    )
    endpoint_snapshot = extend_tactical_snapshot(
        endpoint,
        build_rich_tactical_snapshot(endpoint),
    )
    delta = build_rich_position_delta(
        root,
        endpoint,
        before_snapshot=root_snapshot,
        after_snapshot=endpoint_snapshot,
    )
    typed = CandidateContinuationEndpointEvidence(
        **meta,
        endpoint_fen=endpoint.fen(),
        endpoint_position=build_position_fingerprint(endpoint),
        endpoint_tactical_snapshot=endpoint_snapshot,
        root_to_endpoint_delta=delta,
        irreversible_events=events,
        proof_scope=(
            "Endpoint uses only the candidate move and its already returned principal "
            "variation. It may stop early after a forcing sequence reaches a local state "
            "with no legal check, capture or promotion. This is principal-variation evidence, "
            "not proof that the line is forced. A pv_exhausted endpoint is only the end of the "
            "available PV and must not be described as a quiet position."
        ),
    )
    return candidate.model_copy(update={"continuation_endpoint": typed})


def _delta_from_endpoint(candidate: CandidateEvidence) -> PositionDelta | None:
    endpoint = candidate.continuation_endpoint
    if endpoint is None:
        return None
    return endpoint.root_to_endpoint_delta


def _material_effect(delta: PositionDelta, mover: chess.Color) -> int:
    white_net = delta.material_delta_white - delta.material_delta_black
    return white_net if mover == chess.WHITE else -white_net


def _event_labels(candidate: CandidateEvidence) -> list[str]:
    endpoint = candidate.continuation_endpoint
    if endpoint is None:
        return []
    labels: list[str] = []
    for raw in endpoint.irreversible_events:
        reasons = raw.get("reasons")
        reason_text = ",".join(str(item) for item in reasons) if isinstance(reasons, list) else ""
        labels.append(
            f"ply={raw.get('ply_from_root')}:{raw.get('san')}:{reason_text or '-'}"
        )
    return labels


def build_candidate_endpoint_difference(
    root: chess.Board,
    reference: CandidateEvidence,
    candidate: CandidateEvidence,
) -> CandidateContinuationEndpointDifference:
    """Compare two candidate continuation endpoints from the root mover's POV."""
    ref_endpoint = reference.continuation_endpoint
    cand_endpoint = candidate.continuation_endpoint
    ref_delta = _delta_from_endpoint(reference)
    cand_delta = _delta_from_endpoint(candidate)
    if ref_endpoint is None or cand_endpoint is None:
        return CandidateContinuationEndpointDifference(
            available=False,
            reason="continuation_endpoint_missing",
        )
    if ref_delta is None or cand_delta is None:
        return CandidateContinuationEndpointDifference(
            available=False,
            reason="endpoint_delta_missing_or_invalid",
        )

    return CandidateContinuationEndpointDifference(
        available=True,
        reference_endpoint_fen=ref_endpoint.endpoint_fen,
        candidate_endpoint_fen=cand_endpoint.endpoint_fen,
        reference_plies_from_root=ref_endpoint.plies_from_root,
        candidate_plies_from_root=cand_endpoint.plies_from_root,
        reference_termination_reason=ref_endpoint.termination_reason,
        candidate_termination_reason=cand_endpoint.termination_reason,
        reference_tactical_sequence_resolved=ref_endpoint.tactical_sequence_resolved,
        candidate_tactical_sequence_resolved=cand_endpoint.tactical_sequence_resolved,
        material_effect_difference_for_mover_cp=(
            _material_effect(cand_delta, root.turn) - _material_effect(ref_delta, root.turn)
        ),
        only_reference_newly_en_prise=sorted(
            set(ref_delta.newly_en_prise_pieces) - set(cand_delta.newly_en_prise_pieces)
        ),
        only_candidate_newly_en_prise=sorted(
            set(cand_delta.newly_en_prise_pieces) - set(ref_delta.newly_en_prise_pieces)
        ),
        only_reference_newly_pinned=sorted(
            set(ref_delta.newly_pinned_pieces) - set(cand_delta.newly_pinned_pieces)
        ),
        only_candidate_newly_pinned=sorted(
            set(cand_delta.newly_pinned_pieces) - set(ref_delta.newly_pinned_pieces)
        ),
        only_reference_opened_files=sorted(
            set(ref_delta.opened_files) - set(cand_delta.opened_files)
        ),
        only_candidate_opened_files=sorted(
            set(cand_delta.opened_files) - set(ref_delta.opened_files)
        ),
        only_reference_pawn_structure_changes=sorted(
            set(ref_delta.pawn_structure_changes) - set(cand_delta.pawn_structure_changes)
        ),
        only_candidate_pawn_structure_changes=sorted(
            set(cand_delta.pawn_structure_changes) - set(ref_delta.pawn_structure_changes)
        ),
        king_ring_attack_delta_difference_white=(
            cand_delta.king_ring_attack_delta_white - ref_delta.king_ring_attack_delta_white
        ),
        king_ring_attack_delta_difference_black=(
            cand_delta.king_ring_attack_delta_black - ref_delta.king_ring_attack_delta_black
        ),
        reference_irreversible_events=_event_labels(reference),
        candidate_irreversible_events=_event_labels(candidate),
        proof_scope=(
            "Compares the two evidence-bounded principal-variation endpoints. Different "
            "termination reasons or continuation lengths are reported explicitly, so the "
            "consumer must not treat unequal PV horizons as a controlled causal experiment."
        ),
    )
