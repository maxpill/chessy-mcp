"""Evidence-bounded threat/update extensions for rich ``classify_move`` output.

This module adds deterministic board facts that are especially useful for the
coach workflow:

* what new forcing continuation the opponent created with the previous move if
  the player were to pass;
* whether the played move actually removed that exact forcing continuation;
* whether the strongest reply is a capture after which a legal recapture exists
  but an intermediate forcing move is also available (zwischenzug candidate);
* relative-pin geometry created by the played move or by the strongest reply.

The null-move probe is deliberately labelled hypothetical. It is not an engine
threat search and does not prove that a forcing move survives best defense.
Exact-UCI comparisons always use the complete legal forcing sets; only response
lists are presentation-capped. Relative pins and zwischenzugs are
geometry/candidate evidence only.
"""

from __future__ import annotations

from typing import Any, Literal

import chess

from mcp_server.analysis.forensics import PIECE_NAMES, PIECE_VALUES
from mcp_server.models.forensics import ForensicMoveAnalysis

MAX_FORCING_FACTS = 24
MAX_RELATIVE_PINS = 16
MAX_ZWISCHENZUGS = 12


def _color_name(color: chess.Color) -> Literal["white", "black"]:
    return "white" if color == chess.WHITE else "black"


def _captured_piece_with_square(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Piece | None, chess.Square | None]:
    if not board.is_capture(move):
        return None, None
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        square = move.to_square + offset
    else:
        square = move.to_square
    return board.piece_at(square), square


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    return _captured_piece_with_square(board, move)[0]


def _piece_label(piece: chess.Piece | None, square: chess.Square | None = None) -> str | None:
    if piece is None:
        return None
    base = f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}"
    return f"{base}@{chess.square_name(square)}" if square is not None else base


def _forcing_moves(
    board: chess.Board,
    *,
    limit: int | None = MAX_FORCING_FACTS,
) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for move in board.legal_moves:
        is_check = board.gives_check(move)
        is_capture = board.is_capture(move)
        is_promotion = move.promotion is not None
        if not (is_check or is_capture or is_promotion):
            continue
        captured, captured_square = _captured_piece_with_square(board, move)
        facts.append(
            {
                "uci": move.uci(),
                "san": board.san(move),
                "is_check": is_check,
                "is_capture": is_capture,
                "promotion": PIECE_NAMES.get(move.promotion) if move.promotion else None,
                "captured_piece": _piece_label(captured, captured_square),
            }
        )
    facts.sort(
        key=lambda item: (
            not bool(item["is_check"]),
            not bool(item["is_capture"]),
            item["promotion"] is None,
            item["san"],
            item["uci"],
        )
    )
    if limit is None:
        return facts
    return facts[: max(0, limit)]


def _wire(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return items[:MAX_FORCING_FACTS]


def opponent_forcing_threat_update(
    board_before: chess.Board,
    played_move: chess.Move | None,
) -> dict[str, Any]:
    """Compare opponent forcing opportunities before/after their previous move.

    ``board_before`` is the position in which the player is about to move. When
    move-stack history exists, popping once reconstructs the position in which
    the opponent chose the previous move. We compare the opponent's forcing
    moves there with the forcing moves they would have on their *next* turn if
    the player passed now.

    This is a deterministic null-move threat candidate, not proof that the threat
    survives a legal defense. Exact-UCI persistence after the player's move is
    computed on complete legal forcing sets before presentation truncation.
    """
    if not board_before.move_stack:
        return {
            "mechanism": "opponent_forcing_threat_update",
            "history_available": False,
            "pass_probe_available": False,
            "reason": "no_previous_move_history",
            "inference_boundary": (
                "A naked FEN has no previous move history, so newly-created opponent "
                "threats cannot be attributed to a specific prior move."
            ),
        }

    before_opponent = board_before.copy(stack=True)
    opponent_move = before_opponent.pop()
    try:
        opponent_san = before_opponent.san(opponent_move)
    except (ValueError, AssertionError):
        opponent_san = opponent_move.uci()

    baseline_all = _forcing_moves(before_opponent, limit=None)
    if board_before.is_game_over(claim_draw=False):
        return {
            "mechanism": "opponent_forcing_threat_update",
            "history_available": True,
            "pass_probe_available": False,
            "opponent_move_uci": opponent_move.uci(),
            "opponent_move_san": opponent_san,
            "reason": "terminal_position",
            "forcing_moves_before_opponent_move": _wire(baseline_all),
            "forcing_move_counts": {"before_opponent_move_total": len(baseline_all)},
            "presentation_truncated": len(baseline_all) > MAX_FORCING_FACTS,
            "inference_boundary": "The current position is terminal, so a null-move probe is not meaningful.",
        }
    if board_before.is_check():
        return {
            "mechanism": "opponent_forcing_threat_update",
            "history_available": True,
            "pass_probe_available": False,
            "opponent_move_uci": opponent_move.uci(),
            "opponent_move_san": opponent_san,
            "reason": "player_in_check_after_opponent_move",
            "forcing_moves_before_opponent_move": _wire(baseline_all),
            "forcing_move_counts": {"before_opponent_move_total": len(baseline_all)},
            "presentation_truncated": len(baseline_all) > MAX_FORCING_FACTS,
            "inference_boundary": (
                "Passing while in check is illegal. The check itself is already an urgent "
                "position update, so no null-move threat probe is fabricated."
            ),
        }

    passed = board_before.copy(stack=True)
    passed.push(chess.Move.null())
    threats_all = _forcing_moves(passed, limit=None)
    baseline_uci = {item["uci"] for item in baseline_all}
    new_all = [item for item in threats_all if item["uci"] not in baseline_uci]

    unresolved_all: list[dict[str, Any]] = []
    addresses: bool | None = None
    current_all: list[dict[str, Any]] = []
    if played_move is not None and played_move in board_before.legal_moves:
        after_user = board_before.copy(stack=True)
        after_user.push(played_move)
        current_all = _forcing_moves(after_user, limit=None)
        current = {item["uci"]: item for item in current_all}
        unresolved_all = [current[item["uci"]] for item in new_all if item["uci"] in current]
        addresses = not unresolved_all

    counts = {
        "before_opponent_move_total": len(baseline_all),
        "if_player_passes_total": len(threats_all),
        "newly_enabled_total": len(new_all),
        "after_played_move_total": len(current_all),
        "unresolved_exact_total": len(unresolved_all),
    }
    truncated = {
        "before_opponent_move": len(baseline_all) > MAX_FORCING_FACTS,
        "if_player_passes": len(threats_all) > MAX_FORCING_FACTS,
        "newly_enabled": len(new_all) > MAX_FORCING_FACTS,
        "unresolved_exact": len(unresolved_all) > MAX_FORCING_FACTS,
    }
    return {
        "mechanism": "opponent_forcing_threat_update",
        "history_available": True,
        "pass_probe_available": True,
        "opponent_move_uci": opponent_move.uci(),
        "opponent_move_san": opponent_san,
        "forcing_moves_before_opponent_move": _wire(baseline_all),
        "opponent_forcing_moves_if_player_passes": _wire(threats_all),
        "newly_enabled_forcing_threats_if_pass": _wire(new_all),
        "played_move_addresses_exact_new_threats": addresses,
        "unresolved_exact_new_threats_after_played_move": _wire(unresolved_all),
        "forcing_move_counts": counts,
        "presentation_truncated": truncated,
        "proof_scope": (
            "Hypothetical null-move comparison only. It identifies forcing checks, captures "
            "and promotions that become available on the opponent's next turn if the player "
            "does nothing. It does not prove that these moves survive best legal defense. "
            "Exact-UCI threat creation/persistence is computed on complete legal forcing sets "
            "before wire truncation; returned lists are presentation-capped with total counts."
        ),
    }


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


def _ray(square: chess.Square, df: int, dr: int) -> list[chess.Square]:
    file_index = chess.square_file(square) + df
    rank_index = chess.square_rank(square) + dr
    out: list[chess.Square] = []
    while 0 <= file_index < 8 and 0 <= rank_index < 8:
        out.append(chess.square(file_index, rank_index))
        file_index += df
        rank_index += dr
    return out


def relative_pin_candidates(board: chess.Board) -> list[dict[str, Any]]:
    """Return slider geometry where a lower-value piece screens a higher-value one."""
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for actor_square, actor in board.piece_map().items():
        if actor.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            continue
        for df, dr in _slider_directions(actor.piece_type):
            occupied: list[tuple[chess.Square, chess.Piece]] = []
            for square in _ray(actor_square, df, dr):
                piece = board.piece_at(square)
                if piece is None:
                    continue
                occupied.append((square, piece))
                if len(occupied) == 2:
                    break
            if len(occupied) < 2:
                continue
            (front_square, front), (rear_square, rear) = occupied
            if front.color == actor.color or rear.color == actor.color:
                continue
            if front.piece_type == chess.KING or rear.piece_type == chess.KING:
                continue
            front_value = PIECE_VALUES[front.piece_type]
            rear_value = PIECE_VALUES[rear.piece_type]
            if front_value >= rear_value:
                continue
            actor_label = _piece_label(actor, actor_square)
            front_label = _piece_label(front, front_square)
            rear_label = _piece_label(rear, rear_square)
            if actor_label is None or front_label is None or rear_label is None:
                continue
            key = (actor_label, front_label, rear_label)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "mechanism": "relative_pin_candidate",
                    "actor": actor_label,
                    "front_target": front_label,
                    "rear_target": rear_label,
                    "front_value_cp": front_value,
                    "rear_value_cp": rear_value,
                    "proof_scope": (
                        "Deterministic slider geometry: a lower-value enemy piece is the first "
                        "blocker in front of a higher-value enemy piece on the same ray. This "
                        "is a relative-pin candidate, not proof that the front piece cannot move "
                        "or that material is won."
                    ),
                }
            )
    out.sort(key=lambda item: (item["actor"], item["front_target"], item["rear_target"]))
    return out[:MAX_RELATIVE_PINS]


def zwischenzug_candidate_after_reply(
    board_after_played: chess.Board,
    reply_uci: str,
) -> dict[str, Any] | None:
    """Detect an intermediate forcing-move candidate after a capturing reply.

    The detector requires that the strongest reply is a capture, that the player
    has at least one legal immediate recapture on the reply square, and that the
    player also has another forcing check/capture/promotion available. This is
    exactly the geometry that makes "recapture automatically" dangerous, but it
    does not prove the intermediate move is best.
    """
    try:
        reply = chess.Move.from_uci(reply_uci.lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if reply not in board_after_played.legal_moves or not board_after_played.is_capture(reply):
        return None

    reply_san = board_after_played.san(reply)
    post = board_after_played.copy(stack=True)
    post.push(reply)
    if post.is_game_over(claim_draw=False) or post.is_check():
        return None

    recaptures: list[dict[str, Any]] = []
    for move in post.legal_moves:
        if post.is_capture(move) and move.to_square == reply.to_square:
            recaptures.append(
                {
                    "uci": move.uci(),
                    "san": post.san(move),
                    "is_check": post.gives_check(move),
                }
            )
    if not recaptures:
        return None

    recapture_uci = {item["uci"] for item in recaptures}
    intermediate_all = [
        item
        for item in _forcing_moves(post, limit=None)
        if item["uci"] not in recapture_uci
    ]
    if not intermediate_all:
        return None

    return {
        "mechanism": "zwischenzug_candidate_after_reply",
        "reply_uci": reply.uci(),
        "reply_san": reply_san,
        "available_immediate_recaptures": recaptures,
        "intermediate_forcing_moves": intermediate_all[:MAX_ZWISCHENZUGS],
        "intermediate_forcing_move_count": len(intermediate_all),
        "presentation_truncated": len(intermediate_all) > MAX_ZWISCHENZUGS,
        "proof_scope": (
            "After the capturing reply, an immediate recapture is legal but at least one other "
            "forcing check/capture/promotion is also legal. Existence is checked against the "
            "complete legal forcing set; the returned list is presentation-capped. This "
            "establishes a zwischenzug candidate only, not that it is best or winning."
        ),
    }


def apply_threat_forensics(
    result: ForensicMoveAnalysis,
    board_before: chess.Board,
    *,
    played_move: chess.Move | None,
) -> ForensicMoveAnalysis:
    evidence = result.forensics
    if evidence is None:
        return result

    mechanisms = list(evidence.mechanism_evidence)
    signatures = list(evidence.evidence_signatures)

    threat_update = opponent_forcing_threat_update(board_before, played_move)
    mechanisms.append(threat_update)
    new_count = (
        threat_update.get("forcing_move_counts", {}).get("newly_enabled_total")
        if isinstance(threat_update.get("forcing_move_counts"), dict)
        else None
    )
    new_threats = threat_update.get("newly_enabled_forcing_threats_if_pass")
    has_new_threats = bool(new_count) if isinstance(new_count, int) else bool(new_threats)
    if has_new_threats:
        signatures.append("OPPONENT_MOVE_ENABLED_FORCING_THREAT_CANDIDATE")
        if threat_update.get("played_move_addresses_exact_new_threats") is False:
            signatures.append("FAILED_FORCING_THREAT_UPDATE_CANDIDATE")

    before_relative = relative_pin_candidates(board_before)
    board_after = board_before.copy(stack=True)
    if played_move is not None and played_move in board_before.legal_moves:
        board_after.push(played_move)
    after_relative = relative_pin_candidates(board_after)
    before_keys = {
        (item["actor"], item["front_target"], item["rear_target"])
        for item in before_relative
    }
    new_relative = [
        item
        for item in after_relative
        if (item["actor"], item["front_target"], item["rear_target"]) not in before_keys
    ]
    if new_relative:
        mechanisms.append(
            {
                "mechanism": "new_relative_pin_geometry_after_played_move",
                "candidates": new_relative,
                "proof_scope": (
                    "Geometry newly present after the played move. It is not by itself proof of "
                    "a tactical punishment or material loss."
                ),
            }
        )
        signatures.append("NEW_RELATIVE_PIN_GEOMETRY_AFTER_MOVE")

    reply = evidence.strongest_reply
    if reply is not None:
        zwischenzug = zwischenzug_candidate_after_reply(board_after, reply.uci)
        if zwischenzug is not None:
            mechanisms.append(zwischenzug)
            signatures.append("ZWISCHENZUG_RESPONSE_CANDIDATE")

        try:
            reply_move = chess.Move.from_uci(reply.uci.lower())
        except (ValueError, chess.InvalidMoveError):
            reply_move = None
        if reply_move is not None and reply_move in board_after.legal_moves:
            after_reply = board_after.copy(stack=True)
            after_reply.push(reply_move)
            reply_relative = relative_pin_candidates(after_reply)
            after_keys = {
                (item["actor"], item["front_target"], item["rear_target"])
                for item in after_relative
            }
            new_reply_relative = [
                item
                for item in reply_relative
                if (item["actor"], item["front_target"], item["rear_target"])
                not in after_keys
            ]
            if new_reply_relative:
                mechanisms.append(
                    {
                        "mechanism": "new_relative_pin_geometry_after_strongest_reply",
                        "candidates": new_reply_relative,
                        "proof_scope": (
                            "Geometry newly present after the engine's strongest reply. It does "
                            "not by itself prove the reply wins material."
                        ),
                    }
                )
                signatures.append("RELATIVE_PIN_GEOMETRY_CREATED_BY_STRONGEST_REPLY")

    upgraded = evidence.model_copy(
        update={
            "mechanism_evidence": mechanisms,
            "evidence_signatures": sorted(set(signatures)),
        }
    )
    return result.model_copy(update={"forensics": upgraded})
