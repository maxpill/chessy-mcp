"""Extend rich tactical snapshots with additional evidence-bounded motifs.

The base position-integrity layer already exposes pins, forks, overloaded
pieces and defender-removal candidates. This module adds two more deterministic
geometry classes that are especially useful when explaining tactical lines:
discovered checks and skewers. No engine search is performed here.
"""

from __future__ import annotations

from typing import Any

import chess

from mcp_server.analysis.forensic_extensions import (
    _discovered_check_evidence,
    _skewer_evidence,
)
from mcp_server.models.forensics import (
    ForensicEval,
    MechanismCandidateEvidence,
    TacticalSnapshot,
)

MAX_EXTENDED_MECHANISM_CANDIDATES = 32


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


def extend_tactical_snapshot(board: chess.Board, snapshot: TacticalSnapshot) -> TacticalSnapshot:
    """Append discovered-check/skewer geometry for all legal root moves."""
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
            if len(candidates) >= MAX_EXTENDED_MECHANISM_CANDIDATES:
                return snapshot.model_copy(update={"mechanism_candidates": candidates})

    candidates.sort(
        key=lambda item: (
            item.mechanism,
            item.trigger_san or "",
            item.actor or "",
            tuple(item.targets),
        )
    )
    return snapshot.model_copy(update={"mechanism_candidates": candidates})


def extend_position_eval(result: ForensicEval, board: chess.Board) -> ForensicEval:
    """Propagate extended motif geometry through evaluate_position rich evidence."""
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
