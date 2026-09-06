"""Extend rich tactical snapshots with additional evidence-bounded motifs.

The base position-integrity layer already exposes pins, forks, overloaded
pieces and defender-removal candidates. This module adds deterministic geometry
for discovered checks, skewers and relative pins, a bounded opponent-threat
probe, and a capture-only local exchange tree for defended pieces. No engine
search is performed here.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensic_extensions import (
    _discovered_check_evidence,
    _skewer_evidence,
)
from mcp_server.analysis.forensics import PIECE_NAMES, PIECE_VALUES
from mcp_server.analysis.threat_forensics import relative_pin_candidates
from mcp_server.models.forensics import (
    ForensicEval,
    ForcingMoveEvidence,
    MechanismCandidateEvidence,
    PieceEvidence,
    TacticalHangingEvidence,
    TacticalSnapshot,
)

MAX_EXTENDED_MECHANISM_CANDIDATES = 32
MAX_THREAT_PROBE_MOVES = 24
MAX_LOCAL_EXCHANGE_PLIES = 8


def _candidate_from_raw(raw: dict[str, Any]) -> MechanismCandidateEvidence:
    mechanism = str(raw["mechanism"])
    trigger_uci = raw.get("trigger_uci")
    trigger_san = raw.get("trigger_san")
    proof_scope = str(raw.get("proof_scope", "Deterministic board-geometry candidate."))

    if mechanism == "discovered_check":
        target = raw.get("target")
        discovered = list(raw.get("discovered_checker_squares", []))
        return MechanismCandidateEvidence(
            mechanism="discovered_check",
            trigger_uci=str(trigger_uci) if trigger_uci is not None else None,
            trigger_san=str(trigger_san) if trigger_san is not None else None,
            targets=[str(target)] if target is not None else [],
            evidence={
                "moved_piece_square": raw.get("moved_piece_square"),
                "discovered_checker_squares": discovered,
                "double_check": bool(raw.get("double_check", False)),
            },
            proof_scope=proof_scope,
        )

    return MechanismCandidateEvidence(
        mechanism="skewer_candidate",
        trigger_uci=str(trigger_uci) if trigger_uci is not None else None,
        trigger_san=str(trigger_san) if trigger_san is not None else None,
        actor=str(raw["actor"]) if raw.get("actor") is not None else None,
        targets=[
            str(value)
            for value in (raw.get("front_target"), raw.get("rear_target"))
            if value is not None
        ],
        evidence={
            "front_target": raw.get("front_target"),
            "rear_target": raw.get("rear_target"),
            "front_value_cp": raw.get("front_value_cp"),
            "rear_value_cp": raw.get("rear_value_cp"),
        },
        proof_scope=proof_scope,
    )


def _relative_pin_candidate(raw: dict[str, Any]) -> MechanismCandidateEvidence:
    return MechanismCandidateEvidence(
        mechanism="relative_pin_candidate",
        actor=str(raw["actor"]),
        targets=[str(raw["front_target"]), str(raw["rear_target"])],
        evidence={
            "front_target": raw["front_target"],
            "rear_target": raw["rear_target"],
            "front_value_cp": raw["front_value_cp"],
            "rear_value_cp": raw["rear_value_cp"],
        },
        proof_scope=str(raw["proof_scope"]),
    )


def _captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    if board.is_en_passant(move):
        offset = -8 if board.turn == chess.WHITE else 8
        return board.piece_at(move.to_square + offset)
    return board.piece_at(move.to_square)


def _piece_evidence(board: chess.Board, square: chess.Square, piece: chess.Piece) -> PieceEvidence:
    return PieceEvidence(
        color="white" if piece.color == chess.WHITE else "black",
        piece=PIECE_NAMES[piece.piece_type],
        square=chess.square_name(square),
        attackers=len(board.attackers(not piece.color, square)),
        defenders=len(board.attackers(piece.color, square)),
    )


def _capture_evidence(board: chess.Board, move: chess.Move) -> ForcingMoveEvidence:
    captured = _captured_piece(board, move)
    return ForcingMoveEvidence(
        uci=move.uci(),
        san=board.san(move),
        is_check=board.gives_check(move),
        is_capture=True,
        captured_piece=(
            f"{'white' if captured.color == chess.WHITE else 'black'}_"
            f"{PIECE_NAMES[captured.piece_type]}"
            if captured is not None
            else None
        ),
        promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
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


def _captures_to_square(board: chess.Board, square: chess.Square) -> list[chess.Move]:
    return sorted(
        (
            move
            for move in board.legal_moves
            if board.is_capture(move) and move.to_square == square
        ),
        key=lambda move: move.uci(),
    )


def _local_exchange_minimax(
    board: chess.Board,
    square: chess.Square,
    *,
    root_color: chess.Color,
    baseline_material: int,
    plies_left: int,
    memo: dict[tuple[str, int], tuple[int, list[str], list[str], bool]],
) -> tuple[int, list[str], list[str], bool]:
    """Solve the legal capture-only subtree on one square with a stop option.

    Both sides may decline another capture. The root side maximizes its material
    balance change; the opponent minimizes it. ``complete`` is true only when no
    explored frontier was truncated by ``MAX_LOCAL_EXCHANGE_PLIES``.
    """
    key = (board.fen(), plies_left)
    cached = memo.get(key)
    if cached is not None:
        return cached

    current_gain = _material_balance(board, root_color) - baseline_material
    captures = _captures_to_square(board, square)
    if not captures:
        result = (current_gain, [], [], True)
        memo[key] = result
        return result
    if plies_left <= 0:
        result = (current_gain, [], [], False)
        memo[key] = result
        return result

    branches: list[tuple[int, list[str], list[str], bool]] = [
        (current_gain, [], [], True)
    ]
    all_complete = True
    for move in captures:
        san = board.san(move)
        post = board.copy(stack=True)
        post.push(move)
        gain, child_uci, child_san, complete = _local_exchange_minimax(
            post,
            square,
            root_color=root_color,
            baseline_material=baseline_material,
            plies_left=plies_left - 1,
            memo=memo,
        )
        all_complete = all_complete and complete
        branches.append((gain, [move.uci(), *child_uci], [san, *child_san], complete))

    if board.turn == root_color:
        chosen = max(branches, key=lambda item: (item[0], tuple(item[1])))
    else:
        chosen = min(branches, key=lambda item: (item[0], tuple(item[1])))
    result = (chosen[0], chosen[1], chosen[2], all_complete)
    memo[key] = result
    return result


def _local_exchange_hanging_candidates(board: chess.Board) -> list[TacticalHangingEvidence]:
    """Find defended targets that still lose material in the local recapture tree.

    This extends the immediate-recapture test without pretending to be full SEE.
    Only legal captures that land on the original target square are explored;
    off-square checks, zwischenzugs and quiet tactical resources are deliberately
    outside the proof. A candidate is emitted only when the entire bounded local
    tree is exhausted and the capturer can guarantee at least one pawn of net
    material gain within that restricted tree.
    """
    root_color = board.turn
    baseline = _material_balance(board, root_color)
    out: list[TacticalHangingEvidence] = []
    for capture in list(board.legal_moves):
        if not board.is_capture(capture) or board.is_en_passant(capture):
            continue
        target = board.piece_at(capture.to_square)
        if target is None or target.piece_type == chess.KING:
            continue
        nominal_defenders = len(board.attackers(target.color, capture.to_square))
        if nominal_defenders <= 0:
            continue

        post = board.copy(stack=True)
        first_san = board.san(capture)
        post.push(capture)
        immediate_recaptures = _captures_to_square(post, capture.to_square)
        if not immediate_recaptures:
            continue

        gain, tail_uci, tail_san, complete = _local_exchange_minimax(
            post,
            capture.to_square,
            root_color=root_color,
            baseline_material=baseline,
            plies_left=MAX_LOCAL_EXCHANGE_PLIES - 1,
            memo={},
        )
        if not complete or gain < PIECE_VALUES[chess.PAWN]:
            continue
        out.append(
            TacticalHangingEvidence(
                target=_piece_evidence(board, capture.to_square, target),
                capture=_capture_evidence(board, capture),
                nominal_defenders=nominal_defenders,
                legal_immediate_recaptures=sorted(post.san(move) for move in immediate_recaptures),
                reason="local_capture_exchange_profitable",
                local_exchange_gain_cp=gain,
                local_exchange_line_uci=[capture.uci(), *tail_uci],
                local_exchange_line_san=[first_san, *tail_san],
                local_exchange_tree_complete=True,
                proof_scope=(
                    "Exhaustive legal capture-only minimax on the original target square, with "
                    "either side allowed to stop exchanging, up to eight capture plies. The "
                    "reported material gain is guaranteed only inside that local exchange tree. "
                    "Off-square checks, zwischenzugs, quiet resources and broader positional "
                    "consequences are not part of this proof."
                ),
            )
        )
    return sorted(out, key=lambda item: (item.target.square, item.capture.san))


def _threat_probe(board: chess.Board) -> tuple[list[ForcingMoveEvidence], bool, str | None, str]:
    """List opponent forcing moves after a hypothetical pass by side to move.

    The probe is intentionally unavailable while the side to move is in check or
    the board is terminal. Otherwise it uses a legal python-chess null move to
    give the opponent the turn and enumerates checks, captures and promotions.
    This is threat-candidate evidence only, not an engine proof that the move
    survives best defense.
    """
    scope = (
        "Hypothetical null-move probe. Returned checks, captures and promotions are "
        "opponent forcing-threat candidates if the side to move does nothing. The probe "
        "does not establish that any candidate survives the best legal defense."
    )
    if board.is_game_over(claim_draw=False):
        return [], False, "terminal_position", scope
    if board.is_check():
        return [], False, "side_to_move_in_check_pass_illegal", scope

    passed = board.copy(stack=True)
    passed.push(chess.Move.null())
    threats: list[ForcingMoveEvidence] = []
    for move in passed.legal_moves:
        is_check = passed.gives_check(move)
        is_capture = passed.is_capture(move)
        if not (is_check or is_capture or move.promotion is not None):
            continue
        captured = _captured_piece(passed, move)
        threats.append(
            ForcingMoveEvidence(
                uci=move.uci(),
                san=passed.san(move),
                is_check=is_check,
                is_capture=is_capture,
                captured_piece=(
                    f"{'white' if captured.color == chess.WHITE else 'black'}_"
                    f"{PIECE_NAMES[captured.piece_type]}"
                    if captured is not None
                    else None
                ),
                promotion=PIECE_NAMES.get(move.promotion) if move.promotion else None,
            )
        )
    threats.sort(key=lambda item: (not item.is_check, not item.is_capture, item.san))
    return threats[:MAX_THREAT_PROBE_MOVES], True, None, scope


def extend_tactical_snapshot(board: chess.Board, snapshot: TacticalSnapshot) -> TacticalSnapshot:
    """Append extended geometry and bounded opponent forcing-threat evidence.

    Candidates are deduplicated and globally sorted before the output cap is
    applied. This keeps the wire result deterministic even when python-chess's
    legal-move iteration order changes or a position has many motif candidates.
    """
    candidates = list(snapshot.mechanism_candidates)
    seen = {
        (item.mechanism, item.trigger_uci, item.actor, tuple(item.targets))
        for item in candidates
    }

    for move in board.legal_moves:
        raw_items: list[dict[str, Any]] = []
        discovered = _discovered_check_evidence(board, move)
        if discovered is not None:
            raw_items.append(discovered)
        raw_items.extend(_skewer_evidence(board, move))

        for raw in raw_items:
            candidate = _candidate_from_raw(raw)
            key = (
                candidate.mechanism,
                candidate.trigger_uci,
                candidate.actor,
                tuple(candidate.targets),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    for raw in relative_pin_candidates(board):
        candidate = _relative_pin_candidate(raw)
        key = (
            candidate.mechanism,
            candidate.trigger_uci,
            candidate.actor,
            tuple(candidate.targets),
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(candidate)

    candidates.sort(
        key=lambda item: (
            item.mechanism,
            item.trigger_san or "",
            item.actor or "",
            tuple(item.targets),
        )
    )
    hanging = list(snapshot.tactically_hanging_candidates)
    existing_hanging = {(item.capture.uci, item.reason) for item in hanging}
    for item in _local_exchange_hanging_candidates(board):
        key = (item.capture.uci, item.reason)
        if key not in existing_hanging:
            existing_hanging.add(key)
            hanging.append(item)
    hanging.sort(key=lambda item: (item.target.square, item.capture.san, item.reason))

    threats, probe_available, probe_reason, probe_scope = _threat_probe(board)
    return snapshot.model_copy(
        update={
            "tactically_hanging_candidates": hanging,
            "mechanism_candidates": candidates[:MAX_EXTENDED_MECHANISM_CANDIDATES],
            "opponent_forcing_threats_if_pass": threats,
            "threat_probe_available": probe_available,
            "threat_probe_reason": probe_reason,
            "threat_probe_scope": probe_scope,
        }
    )


def extend_position_eval(result: ForensicEval, board: chess.Board) -> ForensicEval:
    """Propagate extended motif/threat geometry through evaluate_position evidence."""
    evidence = result.forensics
    if evidence is None:
        return result

    updates: dict[str, object] = {
        "tactical_snapshot": extend_tactical_snapshot(board, evidence.tactical_snapshot),
    }
    if evidence.best_move_uci and evidence.tactical_after_best is not None:
        try:
            move = chess.Move.from_uci(evidence.best_move_uci.lower())
        except (ValueError, chess.InvalidMoveError):
            move = None
        if move is not None and move in board.legal_moves:
            post = board.copy(stack=True)
            post.push(move)
            updates["tactical_after_best"] = extend_tactical_snapshot(
                post,
                evidence.tactical_after_best,
            )

    return result.model_copy(update={"forensics": evidence.model_copy(update=updates)})
