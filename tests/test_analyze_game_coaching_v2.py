from __future__ import annotations

import io
import json
from types import SimpleNamespace

import chess
import chess.pgn
import pytest

from mcp_server.analysis.game_coaching import build_game_coaching_evidence
from mcp_server.middleware.request_cost import estimate_mcp_request_cost
from mcp_server.models import MCPEval


PGN = """[Event \"Coach evidence\"]
[White \"mes77777\"]
[Black \"Opponent\"]
[Result \"*\"]

1. e4 e5 2. Qh5 Nc6 3. Qxe5+ {I did not see Nxe5.} Nxe5 *
"""


def _game_and_line() -> tuple[chess.pgn.Game, list[chess.Board], list[chess.Move]]:
    game = chess.pgn.read_game(io.StringIO(PGN))
    assert game is not None
    board = game.board()
    positions = [board.copy(stack=True)]
    moves: list[chess.Move] = []
    for move in game.mainline_moves():
        moves.append(move)
        board.push(move)
        positions.append(board.copy(stack=True))
    return game, positions, moves


def _evals(positions: list[chess.Board], *, depth: int = 18) -> list[MCPEval]:
    # White is roughly equal until Qxe5+??. The post-blunder position already
    # exposes Black's forcing Nxe5 reply, matching the coaching use case.
    cps = [0, 0, 0, 0, 0, -900, -900]
    best = [
        "e2e4",
        "e7e5",
        "g1f3",
        "b8c6",
        "g1f3",
        "c6e5",
        None,
    ]
    out: list[MCPEval] = []
    for board, cp, best_move in zip(positions, cps, best, strict=True):
        pv = [best_move] if best_move else []
        out.append(
            MCPEval(
                cp=cp,
                best_move=best_move,
                pv=pv,
                depth=depth,
                searched_depth=depth,
                canonical_fen=board.fen(),
                wdl=(100, 100, 800) if cp < -500 else (330, 340, 330),
            )
        )
    return out


class _Pool:
    async def top_moves(self, board: chess.Board, *, n: int = 2, depth: int = 20):
        legal = list(board.legal_moves)[:n]
        values = [0, -200]
        return [
            SimpleNamespace(
                cp=values[idx] if board.turn == chess.WHITE else -values[idx],
                mate=None,
                best_move=move.uci(),
                pv=[move.uci()],
                depth=depth,
            )
            for idx, move in enumerate(legal)
        ]


async def _never_evaluate(*args, **kwargs):
    raise AssertionError("coach mode must not launch deep verification searches")


@pytest.mark.asyncio
async def test_coach_mode_builds_story_comment_and_root_cause_without_extra_searches() -> None:
    game, positions, moves = _game_and_line()
    evidence = await build_game_coaching_evidence(
        positions=positions,
        moves=moves,
        evals=_evals(positions),
        game=game,
        perspective="white",
        detail="coach",
        max_critical_moments=6,
        scan_depth=18,
        pool=_Pool(),
        evaluate_positions=_never_evaluate,
    )

    assert evidence.detail == "coach"
    assert evidence.verification_depth is None
    assert evidence.game_segments[0].state == "approximately_equal"
    assert evidence.game_segments[-1].state == "decisively_worse"
    assert evidence.game_segments[-1].transition_cause_ply == 5
    assert evidence.game_segments[-1].transition_confirmed_ply == 5
    assert evidence.game_segments[-1].eval_trough_effective_cp == -900

    qxe5 = next(moment for moment in evidence.critical_moments if moment.san == "Qxe5+")
    assert qxe5.move_class in {"mistake", "blunder"}
    assert qxe5.user_comment_raw == "I did not see Nxe5."
    assert "player_self_report" in qxe5.reasons

    link = next(link for link in evidence.root_cause_links if link.root_cause_ply == qxe5.ply)
    assert link.materialization_san == "Nxe5"
    assert link.plies_later == 1
    assert link.material_swing_cp >= 700

    assert evidence.final_position.position_terminal_by_rules is False
    assert evidence.final_position.defensive_resources_exist is True


@pytest.mark.asyncio
async def test_forensic_mode_selectively_verifies_and_surfaces_forcing_reply() -> None:
    game, positions, moves = _game_and_line()
    base_evals = _evals(positions)
    by_fen = {board.fen(): ev for board, ev in zip(positions, base_evals, strict=True)}
    calls: list[tuple[int, int]] = []

    async def evaluate_positions(boards, depth, pool, *, requested_depth, history_complete):
        calls.append((len(boards), depth))
        result = []
        for board in boards:
            base = by_fen[board.fen()]
            result.append(
                (
                    base.model_copy(
                        update={
                            "depth": depth,
                            "searched_depth": depth,
                            "requested_depth": requested_depth,
                        }
                    ),
                    False,
                )
            )
        return result

    evidence = await build_game_coaching_evidence(
        positions=positions,
        moves=moves,
        evals=base_evals,
        game=game,
        perspective="white",
        detail="forensic",
        max_critical_moments=6,
        scan_depth=18,
        pool=_Pool(),
        evaluate_positions=evaluate_positions,
    )

    assert evidence.verification_depth == 22
    assert calls
    assert all(depth in {22, 24} for _count, depth in calls)

    qxe5 = next(moment for moment in evidence.critical_moments if moment.san == "Qxe5+")
    assert qxe5.verification_depth == 22
    assert qxe5.classification_stable is True
    assert qxe5.strongest_reply_san == "Nxe5"
    assert qxe5.strongest_reply_is_capture is True
    assert qxe5.played_piece == "queen"
    assert qxe5.only_move_missed_candidate is True
    assert qxe5.newly_en_prise_user_pieces == ["white_queen@e5"]
    assert "FORCING_CAPTURE_REPLY" in qxe5.evidence_signatures
    assert "MISSED_FORCING_REPLY_CANDIDATE" in qxe5.evidence_signatures
    assert "NEW_EN_PRISE_PIECE_AFTER_MOVE" in qxe5.evidence_signatures
    assert "ONLY_MOVE_MISSED_CANDIDATE" in qxe5.evidence_signatures
    assert "PLAYER_SELF_REPORT_WITH_FORCING_REPLY" in qxe5.evidence_signatures
    assert qxe5.candidate_gap_effective_cp == 200
    assert qxe5.resource_uniqueness == "high"


def _rpc(arguments: dict[str, object]) -> bytes:
    return json.dumps(
        {"params": {"name": "analyze_game", "arguments": arguments}}
    ).encode()


def test_request_cost_charges_forensic_game_verification() -> None:
    standard = estimate_mcp_request_cost(_rpc({"pgn": PGN, "depth": 18}))
    coach = estimate_mcp_request_cost(
        _rpc({"pgn": PGN, "depth": 18, "detail": "coach"})
    )
    forensic = estimate_mcp_request_cost(
        _rpc(
            {
                "pgn": PGN,
                "depth": 18,
                "detail": "forensic",
                "max_critical_moments": 6,
            }
        )
    )

    assert coach > standard
    assert forensic > coach
