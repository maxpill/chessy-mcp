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

from mcp_server.analysis.forensics import build_position_fingerprint
from mcp_server.analysis.position_integrity import (
    build_rich_position_delta,
    build_rich_tactical_snapshot,
)
from mcp_server.models.forensics import (
    CandidateEvidence,
    CandidatePositionDifference,
    ForensicMoveAnalysis,
    ForensicTopMovesResult,
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


def enrich_candidate_geometry(board: chess.Board, candidate: CandidateEvidence) -> CandidateEvidence:
    """Attach resulting-position and reply geometry to one candidate.

    The candidate's engine evaluation already exists. This function only
    reconstructs legal board states so a coaching client can compare what the
    candidates *leave on the board*, not merely their first-move engine score.
    """
    move = _legal_uci(board, candidate.uci)
    if move is None:
        return candidate

    root_snapshot = build_rich_tactical_snapshot(board)
    post = board.copy(stack=True)
    post.push(move)
    post_snapshot = build_rich_tactical_snapshot(post)
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
            reply_snapshot = build_rich_tactical_snapshot(post_reply)
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

    return candidate.model_copy(update=updates)


def build_candidate_differences(
    board: chess.Board,
    candidates: Iterable[CandidateEvidence],
    *,
    reference_uci: str | None,
) -> list[CandidatePositionDifference]:
    """Compare every candidate with one explicit reference resulting position.

    The engine-best move should normally be supplied as ``reference_uci``. If it
    is absent from the candidate list, the first candidate becomes the reference.
    This makes questions such as "why g4 instead of gxh4?" directly answerable
    from feature deltas after both moves.
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

        out.append(
            CandidatePositionDifference(
                reference_uci=reference.uci,
                reference_san=reference.san,
                candidate_uci=candidate.uci,
                candidate_san=candidate.san,
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
                    set(delta.pawn_structure_changes)
                    - set(reference_delta.pawn_structure_changes)
                ),
                king_ring_attack_delta_difference_white=(
                    delta.king_ring_attack_delta_white
                    - reference_delta.king_ring_attack_delta_white
                ),
                king_ring_attack_delta_difference_black=(
                    delta.king_ring_attack_delta_black
                    - reference_delta.king_ring_attack_delta_black
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

    tactical_before = build_rich_tactical_snapshot(board_before)
    tactical_after = build_rich_tactical_snapshot(board_after)
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
            tactical_reply = build_rich_tactical_snapshot(after_reply)
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

    candidates = [enrich_candidate_geometry(board_before, item) for item in evidence.candidate_comparisons]
    reference_move = _legal_uci(board_before, result.eval_before.best_move)
    reference_uci = reference_move.uci() if reference_move is not None else None
    updates["candidate_comparisons"] = candidates
    updates["candidate_differences"] = build_candidate_differences(
        board_before,
        candidates,
        reference_uci=reference_uci,
    )

    return result.model_copy(update={"forensics": evidence.model_copy(update=updates)})


def upgrade_top_moves_forensics(
    result: ForensicTopMovesResult,
    board: chess.Board,
) -> ForensicTopMovesResult:
    """Attach rich resulting-position differences to top_moves candidates."""
    evidence = result.forensics
    if evidence is None:
        return result

    candidates = [enrich_candidate_geometry(board, item) for item in evidence.candidate_comparisons]
    reference_uci = candidates[0].uci if candidates else None
    upgraded = evidence.model_copy(
        update={
            "candidate_comparisons": candidates,
            "candidate_differences": build_candidate_differences(
                board,
                candidates,
                reference_uci=reference_uci,
            ),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
