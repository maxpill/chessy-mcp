"""Additional evidence-bounded classify-move forensics.

These helpers deliberately avoid extra engine searches. They refine the
already-returned principal variation and strongest reply into facts that are
more useful for coaching: where a forcing sequence becomes quiet, when material
loss first materializes, what forcing continuation exists if the opponent were
to pass, and whether the strongest reply itself has discovered-check or skewer
geometry.

None of the helpers claims to prove a player's thought process. Null-move
follow-up evidence is explicitly hypothetical and skewer evidence is geometric,
not a proof of a forced material win.
"""

from __future__ import annotations

from typing import Any, Literal

import chess

from mcp_server.analysis.forensics import PIECE_NAMES, PIECE_VALUES
from mcp_server.models.forensics import ForensicMoveAnalysis

MAX_FORCING_FOLLOWUPS = 16
MATE_VALUE = 100_000


def _color_name(color: chess.Color) -> Literal["white", "black"]:
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


def _material_balance(board: chess.Board, color: chess.Color) -> int:
    own = 0
    opponent = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        if piece.color == color:
            own += value
        else:
            opponent += value
    return own - opponent


def _forcing_moves(board: chess.Board) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for move in board.legal_moves:
        is_check = board.gives_check(move)
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None
        if not (is_check or is_capture or is_promotion):
            continue
        captured = _captured_piece(board, move)
        out.append(
            {
                "uci": move.uci(),
                "san": board.san(move),
                "is_check": is_check,
                "is_capture": is_capture,
                "promotion": PIECE_NAMES.get(move.promotion) if move.promotion else None,
                "captured_piece": _piece_label(captured),
            }
        )
    return sorted(out, key=lambda item: (not item["is_check"], not item["is_capture"], item["san"]))


def _white_value(cp: int | None, mate: int | None) -> int | None:
    if mate is not None:
        if mate == 0:
            return MATE_VALUE
        return (MATE_VALUE - min(abs(mate), MATE_VALUE - 1)) * (1 if mate > 0 else -1)
    return cp


def _mover_value(cp: int | None, mate: int | None, mover: chess.Color) -> int | None:
    value = _white_value(cp, mate)
    if value is None:
        return None
    return value if mover == chess.WHITE else -value


def build_adaptive_forcing_resolution(
    board: chess.Board,
    pv_uci: list[str],
    *,
    mover: chess.Color | None = None,
) -> dict[str, Any]:
    """Summarize a returned PV until its forcing sequence locally resolves.

    This does not extend the engine PV. It walks only the moves already supplied
    by the engine and stops early once at least one forcing move has occurred,
    the side to move is not currently in check, and that side has no legal
    check, capture or promotion. This keeps mandatory quiet check evasions inside
    the sequence. It is a coaching quiet-point heuristic, not proof that no
    quiet tactical threat exists.
    """
    work = board.copy(stack=True)
    perspective = mover if mover is not None else board.turn
    baseline_material = _material_balance(work, perspective)
    forcing_seen = False
    capture_or_promotion_seen = False
    steps: list[dict[str, Any]] = []
    termination_reason = "no_pv" if not pv_uci else "pv_exhausted"
    resolved = False
    first_material_loss_ply: int | None = None

    for ply, raw in enumerate(pv_uci, start=1):
        try:
            move = chess.Move.from_uci(str(raw).lower())
        except (ValueError, chess.InvalidMoveError):
            termination_reason = "invalid_pv_move"
            break
        if move not in work.legal_moves:
            termination_reason = "invalid_pv_move"
            break

        side = _color_name(work.turn)
        san = work.san(move)
        is_check = work.gives_check(move)
        is_capture = work.is_capture(move)
        is_promotion = move.promotion is not None
        captured = _captured_piece(work, move)
        forcing = is_check or is_capture or is_promotion
        forcing_seen = forcing_seen or forcing
        capture_or_promotion_seen = capture_or_promotion_seen or is_capture or is_promotion
        before_material = _material_balance(work, perspective)
        work.push(move)
        after_material = _material_balance(work, perspective)
        cumulative_material_change = after_material - baseline_material
        if first_material_loss_ply is None and cumulative_material_change <= -100:
            first_material_loss_ply = ply

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
                "material_change_for_mover_this_ply_cp": after_material - before_material,
                "cumulative_material_change_for_mover_cp": cumulative_material_change,
            }
        )

        if work.is_checkmate():
            termination_reason = "forced_mate_in_returned_pv"
            resolved = True
            break
        if work.is_game_over(claim_draw=False):
            termination_reason = (
                "repetition_or_draw_terminal"
                if work.is_repetition(3) or work.is_stalemate() or work.is_insufficient_material()
                else "terminal_position"
            )
            resolved = True
            break

        forcing_available = _forcing_moves(work)
        if forcing_seen and not work.is_check() and not forcing_available:
            termination_reason = (
                "material_resolution" if capture_or_promotion_seen else "quiet_position"
            )
            resolved = True
            break

    return {
        "mechanism": "adaptive_forcing_resolution",
        "moves": steps,
        "pv_plies_available": len(pv_uci),
        "plies_consumed": len(steps),
        "termination_reason": termination_reason,
        "tactical_sequence_resolved": resolved,
        "first_material_loss_ply_for_mover": first_material_loss_ply,
        "proof_scope": (
            "Uses only the returned principal variation and a local forcing-move quiet-point "
            "heuristic. Mandatory quiet check evasions remain inside the sequence. It does not "
            "extend the engine search and does not prove that a quiet position has no strategic "
            "or non-forcing tactical threat."
        ),
    }


def strongest_reply_followup_evidence(
    board_after_played: chess.Board,
    reply_uci: str,
) -> dict[str, Any] | None:
    """Return forcing continuations available if the mover passed after the reply."""
    try:
        reply = chess.Move.from_uci(reply_uci.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if reply not in board_after_played.legal_moves:
        return None

    before_forcing = _forcing_moves(board_after_played)
    after_reply = board_after_played.copy(stack=True)
    san = board_after_played.san(reply)
    after_reply.push(reply)
    if after_reply.is_game_over(claim_draw=False):
        return {
            "mechanism": "strongest_reply_forcing_followup_if_pass",
            "reply": san,
            "reply_uci": reply.uci(),
            "pass_hypothesis_available": False,
            "reason": "terminal_after_reply",
            "forcing_followups_if_opponent_passes": [],
            "proof_scope": "The game is terminal after the strongest reply.",
        }
    if after_reply.is_check():
        return {
            "mechanism": "strongest_reply_forcing_followup_if_pass",
            "reply": san,
            "reply_uci": reply.uci(),
            "pass_hypothesis_available": False,
            "reason": "opponent_in_check_after_reply",
            "forcing_followups_if_opponent_passes": [],
            "proof_scope": (
                "A null-move threat probe is intentionally not performed while the opponent "
                "is in check because passing would be illegal."
            ),
        }

    passed = after_reply.copy(stack=True)
    passed.push(chess.Move.null())
    followups = _forcing_moves(passed)[:MAX_FORCING_FOLLOWUPS]
    before_uci = {item["uci"] for item in before_forcing}
    new_followups = [item for item in followups if item["uci"] not in before_uci]
    return {
        "mechanism": "strongest_reply_forcing_followup_if_pass",
        "reply": san,
        "reply_uci": reply.uci(),
        "pass_hypothesis_available": True,
        "forcing_followups_if_opponent_passes": followups,
        "new_followups_vs_pre_reply": new_followups,
        "has_forcing_followup_if_pass": bool(followups),
        "has_new_forcing_followup_if_pass": bool(new_followups),
        "proof_scope": (
            "Hypothetical null-move probe only. A forcing continuation that exists after a "
            "pass is evidence of a threat candidate, not proof that the threat survives the "
            "opponent's best legal defense."
        ),
    }


def _discovered_check_evidence(board: chess.Board, move: chess.Move) -> dict[str, Any] | None:
    if move not in board.legal_moves or not board.gives_check(move):
        return None
    post = board.copy(stack=True)
    san = board.san(move)
    mover = board.turn
    post.push(move)
    enemy_king = post.king(not mover)
    if enemy_king is None:
        return None
    checkers = set(post.checkers())
    discovered_checkers = sorted(checkers - {move.to_square})
    if not discovered_checkers:
        return None
    return {
        "mechanism": "discovered_check",
        "trigger_uci": move.uci(),
        "trigger_san": san,
        "moved_piece_square": chess.square_name(move.to_square),
        "discovered_checker_squares": [chess.square_name(square) for square in discovered_checkers],
        "double_check": move.to_square in checkers,
        "target": f"{_color_name(not mover)}_king@{chess.square_name(enemy_king)}",
        "proof_scope": (
            "Deterministic check geometry: after the legal move, at least one checking piece "
            "is not the moved piece. This proves a discovered-check component, not a forced win."
        ),
    }


def _ray_squares(square: chess.Square, df: int, dr: int) -> list[chess.Square]:
    file_index = chess.square_file(square) + df
    rank_index = chess.square_rank(square) + dr
    out: list[chess.Square] = []
    while 0 <= file_index < 8 and 0 <= rank_index < 8:
        out.append(chess.square(file_index, rank_index))
        file_index += df
        rank_index += dr
    return out


def _slider_directions(piece_type: chess.PieceType) -> tuple[tuple[int, int], ...]:
    bishop = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    rook = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if piece_type == chess.BISHOP:
        return bishop
    if piece_type == chess.ROOK:
        return rook
    if piece_type == chess.QUEEN:
        return bishop + rook
    return ()


def _skewer_evidence(board: chess.Board, move: chess.Move) -> list[dict[str, Any]]:
    if move not in board.legal_moves:
        return []
    san = board.san(move)
    mover = board.turn
    post = board.copy(stack=True)
    post.push(move)
    actor = post.piece_at(move.to_square)
    if actor is None or actor.color != mover:
        return []
    directions = _slider_directions(actor.piece_type)
    if not directions:
        return []

    out: list[dict[str, Any]] = []
    for df, dr in directions:
        occupied: list[tuple[chess.Square, chess.Piece]] = []
        for square in _ray_squares(move.to_square, df, dr):
            piece = post.piece_at(square)
            if piece is None:
                continue
            occupied.append((square, piece))
            if len(occupied) == 2:
                break
        if len(occupied) < 2:
            continue
        (front_square, front), (rear_square, rear) = occupied
        if front.color == mover or rear.color == mover:
            continue
        front_value = MATE_VALUE if front.piece_type == chess.KING else PIECE_VALUES[front.piece_type]
        rear_value = PIECE_VALUES[rear.piece_type]
        if front.piece_type != chess.KING and front_value <= rear_value:
            continue
        out.append(
            {
                "mechanism": "skewer_candidate",
                "trigger_uci": move.uci(),
                "trigger_san": san,
                "actor": _piece_label(actor, move.to_square),
                "front_target": _piece_label(front, front_square),
                "rear_target": _piece_label(rear, rear_square),
                "front_value_cp": front_value if front.piece_type != chess.KING else None,
                "rear_value_cp": rear_value,
                "proof_scope": (
                    "Geometric skewer candidate after the legal move: a slider attacks a "
                    "higher-priority front target with a lower-value enemy piece behind it on "
                    "the same ray. This does not prove the front piece must move or material is won."
                ),
            }
        )
    return out


def strongest_reply_geometry_evidence(
    board_after_played: chess.Board,
    reply_uci: str,
) -> list[dict[str, Any]]:
    try:
        move = chess.Move.from_uci(reply_uci.lower())
    except (ValueError, chess.InvalidMoveError):
        return []
    if move not in board_after_played.legal_moves:
        return []
    out: list[dict[str, Any]] = []
    discovered = _discovered_check_evidence(board_after_played, move)
    if discovered is not None:
        out.append(discovered)
    out.extend(_skewer_evidence(board_after_played, move))
    return out


def _reply_loss_realization(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    adaptive: dict[str, Any],
) -> dict[str, Any] | None:
    evidence = result.forensics
    if evidence is None or evidence.strongest_reply is None:
        return None
    reply = evidence.strongest_reply
    mover = board_before.turn
    post_move_value = _mover_value(result.eval_after.cp, result.eval_after.mate, mover)
    post_reply_value = _mover_value(reply.eval_after_reply_cp, reply.eval_after_reply_mate, mover)
    research_shift = None
    if post_move_value is not None and post_reply_value is not None:
        research_shift = post_reply_value - post_move_value

    first_material_loss = adaptive.get("first_material_loss_ply_for_mover")
    realization_kind = "not_materialized_in_returned_pv"
    if adaptive.get("termination_reason") == "forced_mate_in_returned_pv":
        realization_kind = "mate_in_returned_pv"
    elif first_material_loss == 1:
        realization_kind = "immediate_material"
    elif isinstance(first_material_loss, int):
        realization_kind = "delayed_material_in_forcing_line"
    elif reply.is_forcing:
        realization_kind = "forcing_reply_without_material_resolution_in_returned_pv"

    return {
        "mechanism": "reply_loss_realization",
        "reply": reply.san,
        "reply_uci": reply.uci,
        "opponent_first_pv_move_forcing": reply.is_forcing,
        "loss_realized_within_plies": first_material_loss,
        "realization_kind": realization_kind,
        "post_move_eval_for_mover_effective_cp": post_move_value,
        "post_reply_research_eval_for_mover_effective_cp": post_reply_value,
        "post_reply_research_shift_for_mover_cp": research_shift,
        "share_of_total_loss_after_first_reply": None,
        "share_of_total_loss_semantics": "not_defined_for_minimax_post_move_evaluation",
        "inference_boundary": (
            "The post-move engine evaluation already assumes the opponent's best reply, so "
            "dividing total CPL into a first-reply share would double-count minimax information. "
            "loss_realized_within_plies reports board materialization in the returned PV instead."
        ),
    }


def apply_move_forensic_extensions(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    evidence = result.forensics
    if evidence is None:
        return result

    board_after = board_before.copy(stack=True)
    if played_move is not None and played_move in board_before.legal_moves:
        board_after.push(played_move)

    mechanisms = list(evidence.mechanism_evidence)
    signatures = list(evidence.evidence_signatures)
    adaptive = build_adaptive_forcing_resolution(
        board_after,
        list(evidence.forced_line.uci),
        mover=board_before.turn,
    )
    mechanisms.append(adaptive)
    if adaptive.get("tactical_sequence_resolved"):
        signatures.append("ADAPTIVE_FORCING_SEQUENCE_RESOLVED")
    if adaptive.get("first_material_loss_ply_for_mover") is not None:
        signatures.append("MATERIAL_LOSS_REALIZED_IN_RETURNED_PV")

    realization = _reply_loss_realization(result, board_before, adaptive)
    if realization is not None:
        mechanisms.append(realization)
        if realization.get("loss_realized_within_plies") == 1:
            signatures.append("LOSS_MATERIALIZED_ON_FIRST_REPLY")
        elif isinstance(realization.get("loss_realized_within_plies"), int):
            signatures.append("LOSS_MATERIALIZED_LATER_IN_FORCING_LINE")

    reply = evidence.strongest_reply
    if reply is not None:
        followup = strongest_reply_followup_evidence(board_after, reply.uci)
        if followup is not None:
            mechanisms.append(followup)
            if followup.get("has_forcing_followup_if_pass"):
                signatures.append("FORCING_FOLLOWUP_IF_OPPONENT_PASSES")
            if followup.get("has_new_forcing_followup_if_pass"):
                signatures.append("NEW_FORCING_FOLLOWUP_ENABLED_BY_REPLY_CANDIDATE")

        reply_geometry = strongest_reply_geometry_evidence(board_after, reply.uci)
        mechanisms.extend(reply_geometry)
        if any(item.get("mechanism") == "discovered_check" for item in reply_geometry):
            signatures.append("DISCOVERED_CHECK_REPLY")
        if any(item.get("mechanism") == "skewer_candidate" for item in reply_geometry):
            signatures.append("SKEWER_GEOMETRY_REPLY_CANDIDATE")

    upgraded = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
