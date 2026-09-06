"""Integrate rich position-geometry evidence into classify/top-move forensics.

The original forensic layer intentionally kept cheap board facts separate from
richer defender/motif geometry. This module joins those layers after engine
work has already completed. It adds no Stockfish search: every field below is
computed from legal board states, the already selected candidate moves and the
already returned strongest replies.
"""

from __future__ import annotations

from collections.abc import Iterable

import chess

from mcp_server.analysis.candidate_continuation import (
    attach_candidate_continuation_endpoint,
    build_candidate_endpoint_difference,
)
from mcp_server.analysis.forensic_extensions import apply_move_forensic_extensions
from mcp_server.analysis.forensics import build_position_fingerprint
from mcp_server.analysis.position_integrity import (
    build_rich_position_delta,
    build_rich_tactical_snapshot,
)
from mcp_server.analysis.tactical_snapshot_extensions import extend_tactical_snapshot
from mcp_server.models.forensics import (
    CandidateEvidence,
    CandidatePositionDifference,
    ForensicMoveAnalysis,
    ForensicTopMovesResult,
    PositionDelta,
    TacticalSnapshot,
)

MATE_VALUE = 100_000


def _legal_uci(board: chess.Board, raw: str | None) -> chess.Move | None:
    if not raw:
        return None
    try:
        move = chess.Move.from_uci(raw.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    return move if move in board.legal_moves else None


def _candidate_white_value(candidate: CandidateEvidence) -> int | None:
    if candidate.eval_mate is not None:
        mate = int(candidate.eval_mate)
        if mate == 0:
            return MATE_VALUE
        return (MATE_VALUE - min(abs(mate), MATE_VALUE - 1)) * (1 if mate > 0 else -1)
    return candidate.eval_cp


def _material_effect_for_mover(candidate: CandidateEvidence, mover: chess.Color) -> int:
    delta = candidate.position_delta
    if delta is None:
        return 0
    white_net = delta.material_delta_white - delta.material_delta_black
    return white_net if mover == chess.WHITE else -white_net


def _piece_safety_labels(delta: PositionDelta) -> set[str]:
    return {
        (
            f"{item.target}:attackers={item.attackers_before}->{item.attackers_after}:"
            f"defenders={item.defenders_before}->{item.defenders_after}"
        )
        for item in delta.piece_safety_changes
    }


def _piece_mobility_labels(delta: PositionDelta) -> set[str]:
    return {
        (
            f"{item.target}:mobility={item.mobility_before}->{item.mobility_after}:"
            f"gained={','.join(item.gained_squares) or '-'}:lost={','.join(item.lost_squares) or '-'}"
        )
        for item in delta.piece_mobility_changes
    }


def _square_control_labels(delta: PositionDelta) -> set[str]:
    return {
        (
            f"{item.square}:white={item.white_attackers_before}->{item.white_attackers_after}:"
            f"black={item.black_attackers_before}->{item.black_attackers_after}"
        )
        for item in delta.strategic_square_control_changes
    }


def _mechanism_labels(candidate: CandidateEvidence) -> set[str]:
    return {
        (
            f"{item.mechanism}:trigger={item.trigger_uci or '-'}:actor={item.actor or '-'}:"
            f"targets={','.join(item.targets) or '-'}"
        )
        for item in candidate.tactical_snapshot_after.mechanism_candidates
    }


def _forcing_label(item: object) -> str:
    uci = str(getattr(item, "uci", ""))
    san = str(getattr(item, "san", ""))
    is_check = int(bool(getattr(item, "is_check", False)))
    is_capture = int(bool(getattr(item, "is_capture", False)))
    promotion = str(getattr(item, "promotion", None) or "-")
    return (
        f"{uci}:{san}:check={is_check}:capture={is_capture}:"
        f"promotion={promotion}"
    )


def _immediate_reply_forcing_labels(candidate: CandidateEvidence) -> set[str]:
    snapshot = candidate.tactical_snapshot_after
    by_uci: dict[str, object] = {}
    for item in [*snapshot.checks, *snapshot.captures]:
        by_uci[item.uci] = item
    return {_forcing_label(item) for item in by_uci.values()}


def _root_threat_labels_if_reply_passes(candidate: CandidateEvidence) -> set[str]:
    return {
        _forcing_label(item)
        for item in candidate.tactical_snapshot_after.opponent_forcing_threats_if_pass
    }


def _root_irreversible_reasons(board: chess.Board, candidate: CandidateEvidence) -> list[str]:
    move = _legal_uci(board, candidate.uci)
    if move is None:
        return []
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
    if after_rights != before_rights:
        reasons.append(f"castling_rights:{before_rights}->{after_rights}")
    return reasons


def enrich_candidate_geometry(
    board: chess.Board,
    candidate: CandidateEvidence,
    *,
    root_snapshot: TacticalSnapshot | None = None,
) -> CandidateEvidence:
    """Attach root, reply and continuation-endpoint geometry to one candidate.

    The candidate's engine evaluation and PV already exist. This function only
    reconstructs legal board states so a coaching client can compare what the
    candidates leave on the board immediately and at evidence-bounded returned
    continuation endpoints, rather than merely comparing first-move scores.
    """
    move = _legal_uci(board, candidate.uci)
    if move is None:
        return candidate

    root_snapshot = root_snapshot or extend_tactical_snapshot(
        board,
        build_rich_tactical_snapshot(board),
    )
    post = board.copy(stack=True)
    post.push(move)
    post_snapshot = extend_tactical_snapshot(post, build_rich_tactical_snapshot(post))
    updates: dict[str, object] = {
        "resulting_fen": post.fen(),
        "position_after": build_position_fingerprint(post),
        "tactical_snapshot_after": post_snapshot,
        "position_delta": build_rich_position_delta(
            board,
            post,
            before_snapshot=root_snapshot,
            after_snapshot=post_snapshot,
        ),
    }

    reply = candidate.opponent_best_reply
    if reply is not None:
        reply_move = _legal_uci(post, reply.uci)
        if reply_move is not None:
            post_reply = post.copy(stack=True)
            post_reply.push(reply_move)
            reply_snapshot = extend_tactical_snapshot(
                post_reply,
                build_rich_tactical_snapshot(post_reply),
            )
            updates.update(
                {
                    "position_after_reply": build_position_fingerprint(post_reply),
                    "tactical_after_reply": reply_snapshot,
                    "reply_delta": build_rich_position_delta(
                        post,
                        post_reply,
                        before_snapshot=post_snapshot,
                        after_snapshot=reply_snapshot,
                    ),
                }
            )

    enriched = candidate.model_copy(update=updates)
    return attach_candidate_continuation_endpoint(
        board,
        enriched,
        root_snapshot=root_snapshot,
    )


def build_candidate_differences(
    board: chess.Board,
    candidates: Iterable[CandidateEvidence],
    *,
    reference_uci: str | None,
) -> list[CandidatePositionDifference]:
    """Compare every candidate with one explicit engine-reference candidate.

    Immediate resulting positions remain available for low-horizon explanations.
    ``continuation_endpoint_difference`` additionally compares each candidate's
    evidence-bounded returned-PV endpoint. The endpoint records unequal horizons
    and termination reasons explicitly, so it is useful evidence rather than a
    false claim that two PV endpoints form a controlled causal experiment.
    """
    items = list(candidates)
    if len(items) < 2:
        return []

    reference = next((item for item in items if item.uci == reference_uci), items[0])
    mover_sign = 1 if board.turn == chess.WHITE else -1
    reference_value = _candidate_white_value(reference)
    reference_delta = reference.position_delta
    if reference_delta is None:
        return []

    reference_safety = _piece_safety_labels(reference_delta)
    reference_mobility = _piece_mobility_labels(reference_delta)
    reference_control = _square_control_labels(reference_delta)
    reference_mechanisms = _mechanism_labels(reference)
    reference_immediate = _immediate_reply_forcing_labels(reference)
    reference_root_threats = _root_threat_labels_if_reply_passes(reference)
    reference_irreversible = _root_irreversible_reasons(board, reference)

    out: list[CandidatePositionDifference] = []
    for candidate in items:
        if candidate.uci == reference.uci:
            continue
        delta = candidate.position_delta
        if delta is None:
            continue

        candidate_value = _candidate_white_value(candidate)
        eval_gap: int | None = None
        if reference_value is not None and candidate_value is not None:
            eval_gap = mover_sign * (candidate_value - reference_value)

        candidate_safety = _piece_safety_labels(delta)
        candidate_mobility = _piece_mobility_labels(delta)
        candidate_control = _square_control_labels(delta)
        candidate_mechanisms = _mechanism_labels(candidate)
        candidate_immediate = _immediate_reply_forcing_labels(candidate)
        candidate_root_threats = _root_threat_labels_if_reply_passes(candidate)
        candidate_irreversible = _root_irreversible_reasons(board, candidate)

        out.append(
            CandidatePositionDifference(
                reference_uci=reference.uci,
                reference_san=reference.san,
                candidate_uci=candidate.uci,
                candidate_san=candidate.san,
                first_divergence_ply=1,
                first_divergence={
                    "reference": reference.san,
                    "candidate": candidate.san,
                },
                eval_gap_candidate_minus_reference_for_mover_cp=eval_gap,
                material_effect_difference_for_mover_cp=(
                    _material_effect_for_mover(candidate, board.turn)
                    - _material_effect_for_mover(reference, board.turn)
                ),
                reference_reply_is_forcing=(
                    reference.opponent_best_reply.is_forcing
                    if reference.opponent_best_reply is not None
                    else None
                ),
                candidate_reply_is_forcing=(
                    candidate.opponent_best_reply.is_forcing
                    if candidate.opponent_best_reply is not None
                    else None
                ),
                reference_root_move_irreversible=bool(reference_irreversible),
                candidate_root_move_irreversible=bool(candidate_irreversible),
                reference_root_irreversible_reasons=reference_irreversible,
                candidate_root_irreversible_reasons=candidate_irreversible,
                only_reference_newly_en_prise=sorted(
                    set(reference_delta.newly_en_prise_pieces)
                    - set(delta.newly_en_prise_pieces)
                ),
                only_candidate_newly_en_prise=sorted(
                    set(delta.newly_en_prise_pieces)
                    - set(reference_delta.newly_en_prise_pieces)
                ),
                only_reference_newly_pinned=sorted(
                    set(reference_delta.newly_pinned_pieces) - set(delta.newly_pinned_pieces)
                ),
                only_candidate_newly_pinned=sorted(
                    set(delta.newly_pinned_pieces) - set(reference_delta.newly_pinned_pieces)
                ),
                only_reference_opened_files=sorted(
                    set(reference_delta.opened_files) - set(delta.opened_files)
                ),
                only_candidate_opened_files=sorted(
                    set(delta.opened_files) - set(reference_delta.opened_files)
                ),
                only_reference_pawn_structure_changes=sorted(
                    set(reference_delta.pawn_structure_changes)
                    - set(delta.pawn_structure_changes)
                ),
                only_candidate_pawn_structure_changes=sorted(
                    set(delta.pawn_structure_changes) - set(reference_delta.pawn_structure_changes)
                ),
                only_reference_piece_safety_changes=sorted(reference_safety - candidate_safety),
                only_candidate_piece_safety_changes=sorted(candidate_safety - reference_safety),
                only_reference_piece_mobility_changes=sorted(
                    reference_mobility - candidate_mobility
                ),
                only_candidate_piece_mobility_changes=sorted(
                    candidate_mobility - reference_mobility
                ),
                only_reference_strategic_square_control_changes=sorted(
                    reference_control - candidate_control
                ),
                only_candidate_strategic_square_control_changes=sorted(
                    candidate_control - reference_control
                ),
                only_reference_mechanism_candidates=sorted(
                    reference_mechanisms - candidate_mechanisms
                ),
                only_candidate_mechanism_candidates=sorted(
                    candidate_mechanisms - reference_mechanisms
                ),
                only_reference_immediate_reply_forcing_moves=sorted(
                    reference_immediate - candidate_immediate
                ),
                only_candidate_immediate_reply_forcing_moves=sorted(
                    candidate_immediate - reference_immediate
                ),
                only_reference_root_forcing_threats_if_reply_passes=sorted(
                    reference_root_threats - candidate_root_threats
                ),
                only_candidate_root_forcing_threats_if_reply_passes=sorted(
                    candidate_root_threats - reference_root_threats
                ),
                king_ring_attack_delta_difference_white=(
                    delta.king_ring_attack_delta_white
                    - reference_delta.king_ring_attack_delta_white
                ),
                king_ring_attack_delta_difference_black=(
                    delta.king_ring_attack_delta_black
                    - reference_delta.king_ring_attack_delta_black
                ),
                continuation_endpoint_difference=build_candidate_endpoint_difference(
                    board,
                    reference,
                    candidate,
                ),
            )
        )
    return out


def upgrade_move_forensics(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    """Upgrade classify_move evidence to the same rich geometry as evaluate/top_moves."""
    evidence = result.forensics
    if evidence is None:
        return result

    board_after = board_before.copy(stack=True)
    if played_move is not None and played_move in board_before.legal_moves:
        board_after.push(played_move)

    tactical_before = extend_tactical_snapshot(
        board_before,
        build_rich_tactical_snapshot(board_before),
    )
    tactical_after = extend_tactical_snapshot(
        board_after,
        build_rich_tactical_snapshot(board_after),
    )
    updates: dict[str, object] = {
        "tactical_before": tactical_before,
        "tactical_after_played": tactical_after,
        "position_delta": build_rich_position_delta(
            board_before,
            board_after,
            before_snapshot=tactical_before,
            after_snapshot=tactical_after,
        ),
    }

    reply = evidence.strongest_reply
    if reply is not None:
        reply_move = _legal_uci(board_after, reply.uci)
        if reply_move is not None:
            after_reply = board_after.copy(stack=True)
            after_reply.push(reply_move)
            tactical_reply = extend_tactical_snapshot(
                after_reply,
                build_rich_tactical_snapshot(after_reply),
            )
            updates.update(
                {
                    "position_after_reply": build_position_fingerprint(after_reply),
                    "tactical_after_reply": tactical_reply,
                    "reply_delta": build_rich_position_delta(
                        board_after,
                        after_reply,
                        before_snapshot=tactical_after,
                        after_snapshot=tactical_reply,
                    ),
                }
            )

    candidates = [
        enrich_candidate_geometry(
            board_before,
            item,
            root_snapshot=tactical_before,
        )
        for item in evidence.candidate_comparisons
    ]
    reference_move = _legal_uci(board_before, result.eval_before.best_move)
    reference_uci = reference_move.uci() if reference_move is not None else None
    updates["candidate_comparisons"] = candidates
    updates["candidate_differences"] = build_candidate_differences(
        board_before,
        candidates,
        reference_uci=reference_uci,
    )

    upgraded = result.model_copy(update={"forensics": evidence.model_copy(update=updates)})
    return apply_move_forensic_extensions(
        upgraded,
        board_before,
        played_move=played_move,
    )


def upgrade_top_moves_forensics(
    result: ForensicTopMovesResult,
    board: chess.Board,
) -> ForensicTopMovesResult:
    """Attach rich immediate and returned-continuation candidate differences."""
    evidence = result.forensics
    if evidence is None:
        return result

    root_snapshot = extend_tactical_snapshot(board, evidence.tactical_snapshot)
    candidates = [
        enrich_candidate_geometry(
            board,
            item,
            root_snapshot=root_snapshot,
        )
        for item in evidence.candidate_comparisons
    ]
    reference_uci = candidates[0].uci if candidates else None
    upgraded = evidence.model_copy(
        update={
            "tactical_snapshot": root_snapshot,
            "candidate_comparisons": candidates,
            "candidate_differences": build_candidate_differences(
                board,
                candidates,
                reference_uci=reference_uci,
            ),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
