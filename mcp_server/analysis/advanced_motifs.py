"""Additional evidence-bounded tactical geometry for rich snapshots.

The detectors in this module intentionally stop at board geometry. They do not
claim that a motif wins material or that a player saw or missed it. They add no
engine search and are safe to use in ``coach`` and ``forensic`` responses.
"""

from __future__ import annotations

import chess

from mcp_server.analysis.forensics import PIECE_NAMES
from mcp_server.models.forensics import MechanismCandidateEvidence

MAX_ADVANCED_MOTIFS = 24
_SLIDERS = {chess.BISHOP, chess.ROOK, chess.QUEEN}
_MAJOR_MINOR = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}


def _color_name(color: chess.Color) -> str:
    return "white" if color == chess.WHITE else "black"


def _label(piece: chess.Piece, square: chess.Square) -> str:
    return f"{_color_name(piece.color)}_{PIECE_NAMES[piece.piece_type]}@{chess.square_name(square)}"


def _between_contains(a: chess.Square, b: chess.Square, square: chess.Square) -> bool:
    return bool(chess.between(a, b) & chess.BB_SQUARES[square])


def discovered_attack_candidates(board: chess.Board) -> list[MechanismCandidateEvidence]:
    """Find moves that uncover a different same-color slider onto an enemy piece."""
    mover = board.turn
    out: list[MechanismCandidateEvidence] = []
    legal = list(board.legal_moves)
    for move in legal:
        moved_before = board.piece_at(move.from_square)
        if moved_before is None:
            continue
        san = board.san(move)
        post = board.copy(stack=True)
        post.push(move)
        for target_square, target in post.piece_map().items():
            if target.color == mover or target.piece_type == chess.KING:
                continue
            before_attackers = set(board.attackers(mover, target_square))
            after_attackers = set(post.attackers(mover, target_square))
            for attacker_square in sorted(after_attackers - before_attackers):
                if attacker_square == move.to_square:
                    continue
                attacker = post.piece_at(attacker_square)
                if attacker is None or attacker.color != mover or attacker.piece_type not in _SLIDERS:
                    continue
                if not _between_contains(attacker_square, target_square, move.from_square):
                    continue
                is_check = post.is_check()
                is_capture = board.is_capture(move)
                target_defenders = len(post.attackers(target.color, target_square))
                target_attackers = len(post.attackers(mover, target_square))
                target_en_prise = target_defenders < target_attackers
                is_pawn_move = moved_before.piece_type == chess.PAWN

                if is_check:
                    priority = "checking_move"
                    relevance = "tactically_actionable"
                elif is_capture:
                    priority = "capturing_move"
                    relevance = "tactically_actionable"
                elif target_en_prise:
                    priority = "new_en_prise"
                    relevance = "tactically_actionable"
                elif is_pawn_move:
                    priority = "pure_geometry"
                    relevance = "pure_geometry"
                else:
                    priority = "forced_reply"
                    relevance = "tactically_actionable"

                out.append(
                    MechanismCandidateEvidence(
                        mechanism="discovered_attack_candidate",
                        trigger_uci=move.uci(),
                        trigger_san=san,
                        actor=_label(moved_before, move.from_square),
                        targets=[_label(target, target_square)],
                        evidence={
                            "discovered_attacker": _label(attacker, attacker_square),
                            "vacated_square": chess.square_name(move.from_square),
                            "moved_to_square": chess.square_name(move.to_square),
                            "is_check": is_check,
                            "is_capture": is_capture,
                            "target_en_prise": target_en_prise,
                            "is_pawn_slider_opening": is_pawn_move and not (is_check or is_capture or target_en_prise),
                        },
                        presentation_priority=priority,
                        relevance=relevance,
                        proof_scope=(
                            "Deterministic discovered-attack geometry: the legal move vacates a "
                            "square on a slider ray and a different same-color bishop, rook or "
                            "queen newly attacks the enemy target. This does not prove material "
                            "is won or that the target lacks a tactical resource."
                        ),
                    )
                )
    return out[:MAX_ADVANCED_MOTIFS]


def interference_candidates(board: chess.Board) -> list[MechanismCandidateEvidence]:
    """Find legal interpositions that sever an enemy slider's defense of an attacked target."""
    mover = board.turn
    enemy = not mover
    relations: list[tuple[chess.Square, chess.Square]] = []
    for target_square, target in board.piece_map().items():
        if target.color != enemy or target.piece_type == chess.KING:
            continue
        if not board.is_attacked_by(mover, target_square):
            continue
        for defender_square in board.attackers(enemy, target_square):
            defender = board.piece_at(defender_square)
            if defender is None or defender.piece_type not in _SLIDERS:
                continue
            if chess.between(defender_square, target_square):
                relations.append((defender_square, target_square))

    out: list[MechanismCandidateEvidence] = []
    for move in board.legal_moves:
        moved = board.piece_at(move.from_square)
        if moved is None:
            continue
        for defender_square, target_square in relations:
            if not _between_contains(defender_square, target_square, move.to_square):
                continue
            defender = board.piece_at(defender_square)
            target = board.piece_at(target_square)
            if defender is None or target is None:
                continue
            san = board.san(move)
            post = board.copy(stack=True)
            post.push(move)
            post_defender = post.piece_at(defender_square)
            post_target = post.piece_at(target_square)
            if post_defender is None or post_target is None:
                continue
            if defender_square in post.attackers(enemy, target_square):
                continue
            if not post.is_attacked_by(mover, target_square):
                continue
            out.append(
                MechanismCandidateEvidence(
                    mechanism="interference_candidate",
                    trigger_uci=move.uci(),
                    trigger_san=san,
                    actor=_label(moved, move.from_square),
                    targets=[_label(defender, defender_square), _label(target, target_square)],
                    evidence={
                        "interference_square": chess.square_name(move.to_square),
                        "defender": _label(defender, defender_square),
                        "defended_target": _label(target, target_square),
                        "target_remains_attacked_after_move": True,
                    },
                    proof_scope=(
                        "Deterministic line-interference geometry: before the move an enemy "
                        "bishop, rook or queen geometrically defended an already attacked target; "
                        "the legal move interposes on that ray and removes that defender from the "
                        "target's attacker set while the target remains attacked. This does not "
                        "prove the target is lost because the defender may capture the blocker or "
                        "another tactical resource may exist."
                    ),
                )
            )
    return out[:MAX_ADVANCED_MOTIFS]


def trapped_piece_candidates(board: chess.Board) -> list[MechanismCandidateEvidence]:
    """Find attacked enemy pieces with no geometrically safe move after a hypothetical pass."""
    if board.is_game_over(claim_draw=False) or board.is_check():
        return []

    mover = board.turn
    enemy = not mover
    passed = board.copy(stack=True)
    passed.push(chess.Move.null())
    out: list[MechanismCandidateEvidence] = []

    for square, piece in board.piece_map().items():
        if piece.color != enemy or piece.piece_type not in _MAJOR_MINOR:
            continue
        if not board.is_attacked_by(mover, square):
            continue

        piece_moves = [move for move in passed.legal_moves if move.from_square == square]
        safe: list[str] = []
        attacked_destinations: list[str] = []
        legal_san: list[str] = []
        for move in piece_moves:
            legal_san.append(passed.san(move))
            post = passed.copy(stack=True)
            post.push(move)
            moved_piece = post.piece_at(move.to_square)
            if moved_piece is None or moved_piece.color != enemy:
                continue
            if post.is_attacked_by(mover, move.to_square):
                attacked_destinations.append(chess.square_name(move.to_square))
            else:
                safe.append(passed.san(move))

        if safe:
            continue
        out.append(
            MechanismCandidateEvidence(
                mechanism="trapped_piece_candidate",
                actor=None,
                targets=[_label(piece, square)],
                evidence={
                    "current_attackers": sorted(
                        chess.square_name(attacker) for attacker in board.attackers(mover, square)
                    ),
                    "legal_piece_moves_after_hypothetical_pass": sorted(legal_san),
                    "geometrically_safe_piece_moves": [],
                    "attacked_destination_squares": sorted(set(attacked_destinations)),
                    "pass_hypothesis_available": True,
                },
                proof_scope=(
                    "Hypothetical-pass mobility geometry only. The piece is currently attacked "
                    "and, if the side to move did nothing, every legal move by that piece would "
                    "land on a square geometrically attacked by the current side (or it has no "
                    "legal piece move). This is a trapped-piece candidate, not proof of material "
                    "loss: captures, counterchecks, pinned attackers, exchanges and other moves "
                    "by the defending side can change the tactical result."
                ),
            )
        )
    return out[:MAX_ADVANCED_MOTIFS]


def advanced_motif_candidates(board: chess.Board) -> list[MechanismCandidateEvidence]:
    """Return deterministic, deduplicated advanced motif candidates."""
    items = [
        *discovered_attack_candidates(board),
        *interference_candidates(board),
        *trapped_piece_candidates(board),
    ]
    deduped: dict[tuple[str, str | None, str | None, tuple[str, ...]], MechanismCandidateEvidence] = {}
    for item in items:
        key = (item.mechanism, item.trigger_uci, item.actor, tuple(item.targets))
        deduped.setdefault(key, item)
    return sorted(
        deduped.values(),
        key=lambda item: (
            item.mechanism,
            item.trigger_san or "",
            item.actor or "",
            tuple(item.targets),
        ),
    )[:MAX_ADVANCED_MOTIFS]
