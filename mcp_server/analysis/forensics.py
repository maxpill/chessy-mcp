"""Evidence-first forensic helpers for coaching-oriented move analysis.

This module reports board facts and engine continuations. It deliberately does
not claim why a human chose a move; the coaching layer can map evidence to a
process hypothesis when it also has player context.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

import chess

from mcp_server.models.forensics import (
    CandidateEvidence,
    ForcedLineEvidence,
    ForensicEvidence,
    ForensicMoveAnalysis,
    ForcingMoveEvidence,
    PieceEvidence,
    PositionDelta,
    PositionFingerprint,
    StrongestReplyEvidence,
    TacticalSnapshot,
)
from mcp_server.models.legacy import MCPMoveAnalysis

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


def _color_name(color: chess.Color) -> Literal["white", "black"]:
    return "white" if color == chess.WHITE else "black"


def _piece_label(piece: chess.Piece | None) -> str | None:
    if piece is None:
        return None
    return f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}"


def _piece_at_label(piece: chess.Piece, square: chess.Square) -> str:
    return f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _move_evidence(board: chess.Board, move: chess.Move) -> ForcingMoveEvidence:
    captured = _captured_piece(board, move)
    return ForcingMoveEvidence(
        uci=move.uci(),
        san=board.san(move),
        is_check=board.gives_check(move),
        is_capture=board.is_capture(move),
        captured_piece=_piece_label(captured),
        promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
    )


def build_position_fingerprint(board: chess.Board) -> PositionFingerprint:
    canonical_fen = board.fen()
    piece_map: dict[str, dict[str, list[str]]] = {
        "white": {name: [] for name in PIECE_NAMES.values()},
        "black": {name: [] for name in PIECE_NAMES.values()},
    }
    material = {"white": 0, "black": 0}
    for square, piece in board.piece_map().items():
        color = _color_name(piece.color)
        name = PIECE_NAMES[piece.piece_type]
        piece_map[color][name].append(chess.square_name(square))
        material[color] += PIECE_VALUES[piece.piece_type]
    for side in piece_map.values():
        for squares in side.values():
            squares.sort()
    ep = chess.square_name(board.ep_square) if board.ep_square is not None else None
    return PositionFingerprint(
        canonical_fen=canonical_fen,
        side_to_move=_color_name(board.turn),
        piece_map=piece_map,
        material=material,
        castling_rights=board.castling_xfen() or "-",
        en_passant=ep,
        in_check=board.is_check(),
        legal_move_count=board.legal_moves.count(),
        position_hash=hashlib.sha256(canonical_fen.encode()).hexdigest()[:16],
    )


def _piece_evidence(
    board: chess.Board,
    square: chess.Square,
    piece: chess.Piece,
) -> PieceEvidence:
    return PieceEvidence(
        color=_color_name(piece.color),
        piece=PIECE_NAMES[piece.piece_type],
        square=chess.square_name(square),
        attackers=len(board.attackers(not piece.color, square)),
        defenders=len(board.attackers(piece.color, square)),
    )


def _piece_sort_key(item: PieceEvidence) -> tuple[str, str, str]:
    return item.color, item.square, item.piece


def build_tactical_snapshot(board: chess.Board) -> TacticalSnapshot:
    checks: list[ForcingMoveEvidence] = []
    captures: list[ForcingMoveEvidence] = []
    en_prise_squares: set[chess.Square] = set()
    for move in board.legal_moves:
        if board.gives_check(move):
            checks.append(_move_evidence(board, move))
        if board.is_capture(move):
            captures.append(_move_evidence(board, move))
            if not board.is_en_passant(move) and board.piece_at(move.to_square) is not None:
                en_prise_squares.add(move.to_square)

    loose: list[PieceEvidence] = []
    en_prise: list[PieceEvidence] = []
    pinned: list[PieceEvidence] = []
    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        evidence = _piece_evidence(board, square, piece)
        if evidence.defenders == 0:
            loose.append(evidence)
        if square in en_prise_squares:
            en_prise.append(evidence)
        if board.is_pinned(piece.color, square):
            pinned.append(evidence)

    return TacticalSnapshot(
        side_to_move=_color_name(board.turn),
        checks=sorted(checks, key=lambda item: item.san),
        captures=sorted(captures, key=lambda item: item.san),
        loose_pieces=sorted(loose, key=_piece_sort_key),
        en_prise_pieces=sorted(en_prise, key=_piece_sort_key),
        pinned_pieces=sorted(pinned, key=_piece_sort_key),
    )


def parse_candidate_move(board: chess.Board, text: str) -> chess.Move:
    raw = text.strip()
    try:
        move = chess.Move.from_uci(raw.lower())
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass
    try:
        return board.parse_san(raw)
    except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError) as exc:
        raise ValueError(f"INVALID_COMPARE_MOVE: {text}") from exc


def build_position_delta(
    before: chess.Board,
    after: chess.Board,
    *,
    before_snapshot: TacticalSnapshot | None = None,
    after_snapshot: TacticalSnapshot | None = None,
) -> PositionDelta:
    before_fp = build_position_fingerprint(before)
    after_fp = build_position_fingerprint(after)
    before_snapshot = before_snapshot or build_tactical_snapshot(before)
    after_snapshot = after_snapshot or build_tactical_snapshot(after)

    before_pieces = {
        f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"
        for square, piece in before.piece_map().items()
    }
    after_pieces = {
        f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"
        for square, piece in after.piece_map().items()
    }
    before_loose = {f"{p.color}_{p.piece}@{p.square}" for p in before_snapshot.loose_pieces}
    after_loose = {f"{p.color}_{p.piece}@{p.square}" for p in after_snapshot.loose_pieces}
    before_pinned = {f"{p.color}_{p.piece}@{p.square}" for p in before_snapshot.pinned_pieces}
    after_pinned = {f"{p.color}_{p.piece}@{p.square}" for p in after_snapshot.pinned_pieces}

    return PositionDelta(
        material_delta_white=after_fp.material["white"] - before_fp.material["white"],
        material_delta_black=after_fp.material["black"] - before_fp.material["black"],
        removed_pieces=sorted(before_pieces - after_pieces),
        added_pieces=sorted(after_pieces - before_pieces),
        newly_loose_pieces=sorted(after_loose - before_loose),
        newly_pinned_pieces=sorted(after_pinned - before_pinned),
        check_state_changed=before.is_check() != after.is_check(),
    )


def _reply_from_eval(
    board: chess.Board,
    eval_obj: Any,
    *,
    eval_after_reply: Any | None = None,
) -> StrongestReplyEvidence | None:
    best = getattr(eval_obj, "best_move", None)
    if not best or board.is_game_over(claim_draw=False):
        return None
    try:
        move = chess.Move.from_uci(str(best).lower())
    except (ValueError, chess.InvalidMoveError):
        return None
    if move not in board.legal_moves:
        return None

    captured = _captured_piece(board, move)
    reply_board = board.copy(stack=True)
    san = board.san(move)
    is_check = board.gives_check(move)
    is_capture = board.is_capture(move)
    reply_board.push(move)
    return StrongestReplyEvidence(
        uci=move.uci(),
        san=san,
        is_check=is_check,
        is_capture=is_capture,
        is_forcing=is_check or is_capture,
        captured_piece=_piece_label(captured),
        resulting_fen=reply_board.fen(),
        eval_after_reply_cp=getattr(eval_after_reply, "cp", None),
        eval_after_reply_mate=getattr(eval_after_reply, "mate", None),
        searched_depth=getattr(eval_after_reply, "depth", None),
    )


async def _strongest_reply(
    board: chess.Board,
    eval_obj: Any,
    *,
    pool: Any,
    depth: int,
    deep: bool,
) -> StrongestReplyEvidence | None:
    base = _reply_from_eval(board, eval_obj)
    if base is None or not deep:
        return base
    try:
        move = chess.Move.from_uci(base.uci)
        reply_board = board.copy(stack=True)
        reply_board.push(move)
        if reply_board.is_game_over(claim_draw=False):
            return base
        post_eval = await pool.evaluate(reply_board, depth=min(depth + 2, 24))
    except Exception:
        return base
    return _reply_from_eval(board, eval_obj, eval_after_reply=post_eval)


def _principal_line(board: chess.Board, pv: list[str] | None) -> ForcedLineEvidence:
    if not pv:
        return ForcedLineEvidence()
    work = board.copy(stack=True)
    uci: list[str] = []
    san: list[str] = []
    for raw in pv[:12]:
        try:
            move = chess.Move.from_uci(str(raw).lower())
        except (ValueError, chess.InvalidMoveError):
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="invalid_pv_move",
            )
        if move not in work.legal_moves:
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="invalid_pv_move",
            )
        uci.append(move.uci())
        san.append(work.san(move))
        work.push(move)
        if work.is_game_over(claim_draw=False):
            return ForcedLineEvidence(
                uci=uci,
                san=san,
                termination_reason="terminal_position",
                tactical_sequence_resolved=True,
            )
    final_snapshot = build_tactical_snapshot(work)
    return ForcedLineEvidence(
        uci=uci,
        san=san,
        termination_reason="pv_exhausted",
        tactical_sequence_resolved=not final_snapshot.checks and not final_snapshot.captures,
    )


async def _candidate_evidence(
    board: chess.Board,
    requested: str,
    *,
    pool: Any,
    depth: int,
) -> CandidateEvidence:
    move = parse_candidate_move(board, requested)
    san = board.san(move)
    post = board.copy(stack=True)
    post.push(move)
    snapshot = build_tactical_snapshot(post)
    ev: Any | None = None
    if not post.is_game_over(claim_draw=False):
        ev = await pool.evaluate(post, depth=depth)
    return CandidateEvidence(
        requested=requested,
        uci=move.uci(),
        san=san,
        resulting_fen=post.fen(),
        eval_cp=getattr(ev, "cp", None),
        eval_mate=getattr(ev, "mate", None),
        searched_depth=getattr(ev, "depth", None),
        opponent_best_reply=_reply_from_eval(post, ev) if ev is not None else None,
        tactical_snapshot_after=snapshot,
    )


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


def _first_irreversible_event_ply(
    board: chess.Board,
    forced_line: ForcedLineEvidence,
) -> int | None:
    work = board.copy(stack=True)
    for ply, raw in enumerate(forced_line.uci, start=1):
        try:
            move = chess.Move.from_uci(raw)
        except (ValueError, chess.InvalidMoveError):
            return None
        if move not in work.legal_moves:
            return None
        irreversible = work.is_capture(move) or move.promotion is not None
        work.push(move)
        if irreversible or work.is_game_over(claim_draw=False):
            return ply
    return None


def _forcing_prefix_plies(board: chess.Board, forced_line: ForcedLineEvidence) -> int:
    work = board.copy(stack=True)
    count = 0
    for raw in forced_line.uci:
        try:
            move = chess.Move.from_uci(raw)
        except (ValueError, chess.InvalidMoveError):
            break
        if move not in work.legal_moves:
            break
        forcing = work.gives_check(move) or work.is_capture(move) or move.promotion is not None
        if not forcing:
            break
        count += 1
        work.push(move)
    return count


def _reply_failure_profile(
    board_before: chess.Board,
    board_after: chess.Board,
    reply: StrongestReplyEvidence | None,
    forced_line: ForcedLineEvidence,
) -> dict[str, Any] | None:
    if reply is None:
        return None
    try:
        move = chess.Move.from_uci(reply.uci)
    except (ValueError, chess.InvalidMoveError):
        return None
    if move not in board_after.legal_moves:
        return None

    mover = board_before.turn
    balance_before_reply = _material_balance(board_after, mover)
    post_reply = board_after.copy(stack=True)
    post_reply.push(move)
    balance_after_reply = _material_balance(post_reply, mover)
    immediate_material_swing = balance_after_reply - balance_before_reply
    first_irreversible = _first_irreversible_event_ply(board_after, forced_line)
    forcing_prefix = _forcing_prefix_plies(board_after, forced_line)

    if reply.is_check and reply.is_capture:
        reply_type = "check_capture"
    elif reply.is_check:
        reply_type = "check"
    elif reply.is_capture:
        reply_type = "capture"
    else:
        reply_type = "quiet"

    if post_reply.is_checkmate():
        realization = "immediate_mate"
    elif immediate_material_swing < 0:
        realization = "immediate_material"
    elif reply.is_forcing and first_irreversible is not None and first_irreversible > 1:
        realization = "forcing_sequence"
    elif reply.is_forcing:
        realization = "forcing_reply"
    else:
        realization = "quiet_reply"

    return {
        "mechanism": "reply_failure_profile",
        "reply": reply.san,
        "reply_type": reply_type,
        "opponent_first_pv_move_forcing": reply.is_forcing,
        "immediate_material_swing_for_mover_cp": immediate_material_swing,
        "first_irreversible_event_ply": first_irreversible,
        "forcing_prefix_plies": forcing_prefix,
        "tactical_sequence_resolved_in_returned_pv": forced_line.tactical_sequence_resolved,
        "realization_kind": realization,
        "eval_after_reply_cp": reply.eval_after_reply_cp,
        "eval_after_reply_mate": reply.eval_after_reply_mate,
        "inference_boundary": (
            "This is board/line evidence, not proof of what the player calculated. "
            "No share_of_total_loss_after_first_reply is emitted because a minimax "
            "evaluation after the played move already assumes the opponent's best reply."
        ),
    }


def _piece_attack_state(
    board: chess.Board,
    color: chess.Color,
) -> dict[str, tuple[int, int, bool]]:
    state: dict[str, tuple[int, int, bool]] = {}
    for square, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        state[_piece_at_label(piece, square)] = (
            len(board.attackers(not color, square)),
            len(board.attackers(color, square)),
            board.is_pinned(color, square),
        )
    return state


def _position_update_evidence(
    board_before: chess.Board,
    played_move: chess.Move | None,
) -> dict[str, Any]:
    if not board_before.move_stack:
        return {
            "mechanism": "position_update_after_opponent_move",
            "history_available": False,
            "inference_boundary": (
                "A naked FEN has no previous-move history, so failed-position-update "
                "evidence cannot be reconstructed."
            ),
        }

    before_opponent = board_before.copy(stack=True)
    opponent_move = before_opponent.pop()
    try:
        opponent_san = before_opponent.san(opponent_move)
    except (ValueError, AssertionError):
        opponent_san = opponent_move.uci()

    mover = board_before.turn
    prior = _piece_attack_state(before_opponent, mover)
    current = _piece_attack_state(board_before, mover)

    newly_attacked: list[str] = []
    newly_exposed: list[str] = []
    newly_pinned: list[str] = []
    defender_losses: list[str] = []
    for label, (attackers_after, defenders_after, pinned_after) in current.items():
        previous = prior.get(label)
        if previous is None:
            continue
        attackers_before, defenders_before, pinned_before = previous
        if attackers_before == 0 and attackers_after > 0:
            newly_attacked.append(label)
            if defenders_after == 0:
                newly_exposed.append(label)
        if defenders_after < defenders_before:
            defender_losses.append(
                f"{label}:{defenders_before}->{defenders_after}"
            )
        if not pinned_before and pinned_after:
            newly_pinned.append(label)

    king_in_check = board_before.is_check()
    urgent_targets = sorted(set(newly_exposed + newly_pinned))
    urgent_change = king_in_check or bool(urgent_targets)

    unresolved: list[str] = []
    addresses_change: bool | None = None
    if played_move is not None and played_move in board_before.legal_moves:
        after_user = board_before.copy(stack=True)
        after_user.push(played_move)
        after_state = _piece_attack_state(after_user, mover)
        for label in newly_exposed:
            state = after_state.get(label)
            if state is not None and state[0] > 0 and state[1] == 0:
                unresolved.append(label)
        for label in newly_pinned:
            state = after_state.get(label)
            if state is not None and state[2]:
                unresolved.append(label)
        if king_in_check and after_user.is_check():
            unresolved.append("king_in_check")
        addresses_change = not unresolved

    return {
        "mechanism": "position_update_after_opponent_move",
        "history_available": True,
        "opponent_move_uci": opponent_move.uci(),
        "opponent_move_san": opponent_san,
        "king_in_check_after_opponent_move": king_in_check,
        "newly_attacked_user_pieces": sorted(newly_attacked),
        "newly_exposed_user_pieces": sorted(newly_exposed),
        "newly_pinned_user_pieces": sorted(newly_pinned),
        "defender_count_losses": sorted(defender_losses),
        "opponent_move_created_urgent_change": urgent_change,
        "played_move_addresses_change": addresses_change,
        "unresolved_urgent_targets_after_played_move": sorted(set(unresolved)),
        "inference_boundary": (
            "urgent_change is a deterministic board-state proxy. It can support, "
            "but does not itself prove, a coaching label such as plan persistence."
        ),
    }


def _wdl_expectation(wdl: tuple[int, int, int] | None, color: chess.Color) -> float | None:
    if wdl is None:
        return None
    wins, draws, losses = wdl
    total = wins + draws + losses
    if total <= 0:
        return None
    white_expectation = (wins + 0.5 * draws) / total
    return white_expectation if color == chess.WHITE else 1.0 - white_expectation


def _stability_evidence(
    result: MCPMoveAnalysis,
    *,
    mover: chess.Color,
    reply: StrongestReplyEvidence | None,
    depth: int,
) -> dict[str, Any]:
    before_expectation = _wdl_expectation(result.eval_before.wdl, mover)
    after_expectation = _wdl_expectation(result.eval_after.wdl, mover)
    wdl_loss_pp = None
    if before_expectation is not None and after_expectation is not None:
        wdl_loss_pp = max(0.0, (before_expectation - after_expectation) * 100.0)

    mate_before = result.eval_before.mate
    mate_after = result.eval_after.mate
    mate_status_changed = (mate_before is None) != (mate_after is None)
    if mate_before is not None and mate_after is not None:
        mate_status_changed = mate_status_changed or ((mate_before > 0) != (mate_after > 0))

    forcing_punishment = bool(
        reply is not None
        and reply.is_forcing
        and result.move_class.value in {"mistake", "blunder"}
    )
    small_cp_loss = result.centipawn_loss is not None and result.centipawn_loss <= 30
    small_wdl_loss = wdl_loss_pp is not None and wdl_loss_pp <= 2.0
    practical_equivalent = bool(
        result.action_equivalent
        or result.is_engine_best
        or (
            result.move_class.value in {"best", "good"}
            and not forcing_punishment
            and not mate_status_changed
            and (small_wdl_loss or (wdl_loss_pp is None and small_cp_loss))
        )
    )

    if practical_equivalent:
        coach_priority = "negligible"
    elif result.move_class.value in {"mistake", "blunder"} and forcing_punishment:
        coach_priority = "high"
    elif result.move_class.value in {"mistake", "blunder"}:
        coach_priority = "medium"
    elif result.move_class.value == "inaccuracy":
        coach_priority = "low"
    else:
        coach_priority = "routine"

    return {
        "classification_verified": result.classification_verified,
        "action_equivalent": result.action_equivalent,
        "is_engine_best": result.is_engine_best,
        "requested_depth": depth,
        "searched_depth_before": result.eval_before.searched_depth or result.eval_before.depth,
        "searched_depth_after": result.eval_after.searched_depth or result.eval_after.depth,
        "wdl_loss_percentage_points": wdl_loss_pp,
        "mate_status_changed": mate_status_changed,
        "forcing_punishment": forcing_punishment,
        "practical_equivalent": practical_equivalent,
        "coach_priority": coach_priority,
        "practical_equivalence_basis": (
            "action-equivalence/engine-best first; otherwise a good-or-better move with "
            "<=2 WDL percentage-point loss (or <=30 cp when WDL is unavailable), no mate "
            "status transition, and no forcing tactical punishment."
        ),
    }


def _mechanism_evidence(
    result: MCPMoveAnalysis,
    reply: StrongestReplyEvidence | None,
    tactical_after: TacticalSnapshot,
) -> tuple[list[dict[str, Any]], list[str]]:
    evidence: list[dict[str, Any]] = []
    signatures: list[str] = []
    move_class = result.move_class.value

    if reply is not None and reply.is_forcing:
        evidence.append(
            {
                "mechanism": "forcing_reply",
                "reply": reply.san,
                "is_check": reply.is_check,
                "is_capture": reply.is_capture,
                "captured_piece": reply.captured_piece,
            }
        )
        if move_class in {"mistake", "blunder"}:
            signatures.append("MISSED_FORCING_REPLY_CANDIDATE")
        if reply.is_check:
            signatures.append("FORCING_CHECK_REPLY")
        if reply.is_capture:
            signatures.append("FORCING_CAPTURE_REPLY")
        if reply.is_check and reply.is_capture:
            signatures.append("CHECK_CAPTURE_REPLY")

    if reply is not None and reply.is_capture and reply.captured_piece:
        try:
            victim_square = chess.Move.from_uci(reply.uci).to_square
        except (ValueError, chess.InvalidMoveError):
            victim_square = None
        if victim_square is not None:
            square_name = chess.square_name(victim_square)
            matching = [p for p in tactical_after.loose_pieces if p.square == square_name]
            if matching:
                piece = matching[0]
                evidence.append(
                    {
                        "mechanism": "loose_piece_capture",
                        "target": f"{piece.color}_{piece.piece}@{piece.square}",
                        "defenders": piece.defenders,
                        "reply": reply.san,
                    }
                )
                signatures.append("LOOSE_PIECE_PUNISHED")

    if tactical_after.pinned_pieces:
        evidence.append(
            {
                "mechanism": "pin_present_after_played_move",
                "pieces": [
                    f"{piece.color}_{piece.piece}@{piece.square}"
                    for piece in tactical_after.pinned_pieces
                ],
            }
        )
    return evidence, sorted(set(signatures))


async def enrich_move_analysis(
    result: MCPMoveAnalysis,
    *,
    board_before: chess.Board,
    played_move: chess.Move | None,
    pool: Any,
    depth: int,
    detail: Literal["coach", "forensic"],
    compare_moves: list[str] | None = None,
) -> ForensicMoveAnalysis:
    board_after = board_before.copy(stack=True)
    if played_move is not None:
        board_after.push(played_move)

    fp_before = build_position_fingerprint(board_before)
    fp_after = build_position_fingerprint(board_after)
    tactical_before = build_tactical_snapshot(board_before)
    tactical_after = build_tactical_snapshot(board_after)
    delta = build_position_delta(
        board_before,
        board_after,
        before_snapshot=tactical_before,
        after_snapshot=tactical_after,
    )
    reply = await _strongest_reply(
        board_after,
        result.eval_after,
        pool=pool,
        depth=depth,
        deep=detail == "forensic",
    )
    forced_line = _principal_line(board_after, result.eval_after.pv)
    mechanisms, signatures = _mechanism_evidence(result, reply, tactical_after)

    reply_profile = _reply_failure_profile(board_before, board_after, reply, forced_line)
    if reply_profile is not None:
        mechanisms.append(reply_profile)
        immediate_material_swing = reply_profile["immediate_material_swing_for_mover_cp"]
        if isinstance(immediate_material_swing, int) and immediate_material_swing <= -100:
            signatures.append("IMMEDIATE_MATERIAL_PUNISHMENT")
        if reply_profile["tactical_sequence_resolved_in_returned_pv"]:
            signatures.append("TACTICAL_SEQUENCE_RESOLVED_IN_PV")

    update_evidence = _position_update_evidence(board_before, played_move)
    mechanisms.append(update_evidence)
    if update_evidence.get("opponent_move_created_urgent_change"):
        signatures.append("OPPONENT_MOVE_CREATED_URGENT_CHANGE")
        if update_evidence.get("played_move_addresses_change") is False:
            signatures.append("FAILED_POSITION_UPDATE_CANDIDATE")

    requested_candidates: list[str] = []
    if detail == "forensic":
        if result.played_san:
            requested_candidates.append(result.played_san)
        if result.best_move_san and result.best_move_san != result.played_san:
            requested_candidates.append(result.best_move_san)
    for requested in compare_moves or []:
        if requested not in requested_candidates:
            requested_candidates.append(requested)
    requested_candidates = requested_candidates[:8]

    comparisons: list[CandidateEvidence] = []
    for requested in requested_candidates:
        comparisons.append(
            await _candidate_evidence(board_before, requested, pool=pool, depth=depth)
        )

    forensic = ForensicEvidence(
        detail=detail,
        position_before=fp_before,
        position_after_played=fp_after,
        tactical_before=tactical_before,
        tactical_after_played=tactical_after,
        strongest_reply=reply,
        position_delta=delta,
        mechanism_evidence=mechanisms,
        evidence_signatures=sorted(set(signatures)),
        forced_line=forced_line,
        candidate_comparisons=comparisons,
        stability=_stability_evidence(
            result,
            mover=board_before.turn,
            reply=reply,
            depth=depth,
        ),
    )
    payload = result.model_dump(
        exclude={"same_action_type", "same_outcome", "within_cp_threshold"}
    )
    return ForensicMoveAnalysis(**payload, forensics=forensic)
