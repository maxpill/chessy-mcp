from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from mcp_server.analysis.top_moves_forensics import (
    build_tactical_proof,
    enrich_top_moves_result,
)
from mcp_server.middleware.request_cost import estimate_mcp_request_cost
from mcp_server.models import MCPEval, TopMovesResult
from mcp_server.models.forensics import ForensicTopMovesResult


class _ProofPool:
    async def evaluate(self, board: chess.Board, *, depth: int, root_moves=None):
        legal = list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return SimpleNamespace(
            cp=25 if board.turn == chess.WHITE else -25,
            mate=None,
            depth=depth,
            best_move=best,
            pv=[best] if best else [],
        )

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int = 20):
        legal = list(board.legal_moves)[:n]
        return [
            SimpleNamespace(
                cp=10 + idx,
                mate=None,
                depth=depth,
                best_move=move.uci(),
                pv=[move.uci()],
            )
            for idx, move in enumerate(legal)
        ]


def _root_result(*moves: str) -> TopMovesResult:
    return TopMovesResult(
        returned_n=len(moves),
        result=[
            MCPEval(cp=50 - idx * 10, best_move=move, pv=[move], depth=12, searched_depth=12)
            for idx, move in enumerate(moves)
        ],
    )


@pytest.mark.asyncio
async def test_tactical_proof_samples_engine_ranked_defenses_when_tree_is_wide() -> None:
    board = chess.Board()
    result = _root_result("e2e4", "d2d4")

    proof = await build_tactical_proof(
        result,
        board,
        pool=_ProofPool(),
        depth=10,
        proof_defenses=3,
    )

    assert proof is not None
    assert proof.root_move_san == "e4"
    assert proof.proof_status == "sampled_top_defenses"
    assert proof.legal_defense_count == 20
    assert proof.analyzed_defense_count == 3
    assert [defense.rank for defense in proof.defenses] == [1, 2, 3]
    assert all(defense.continuation.proof_status == "principal_variation_only" for defense in proof.defenses)


@pytest.mark.asyncio
async def test_tactical_proof_labels_immediate_mate_terminal_not_exhaustive() -> None:
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
    move = chess.Move.from_uci("f7f8")
    assert move in board.legal_moves
    after = board.copy(stack=True)
    after.push(move)
    assert after.is_checkmate()

    proof = await build_tactical_proof(
        _root_result("f7f8"),
        board,
        pool=_ProofPool(),
        depth=10,
        proof_defenses=3,
    )

    assert proof is not None
    assert proof.root_move_san.endswith("#")
    assert proof.proof_status == "terminal_after_root"
    assert proof.legal_defense_count == 0
    assert proof.analyzed_defense_count == 0


@pytest.mark.asyncio
async def test_include_move_is_analyzed_even_when_outside_returned_top_n() -> None:
    board = chess.Board()
    result = _root_result("e2e4")

    enriched = await enrich_top_moves_result(
        result,
        board,
        pool=_ProofPool(),
        depth=8,
        detail="coach",
        include_moves=["a3"],
        proof_mode="none",
        proof_defenses=3,
    )

    assert isinstance(enriched, ForensicTopMovesResult)
    assert enriched.forensics is not None
    assert enriched.forensics.position.canonical_fen == board.fen()
    assert len(enriched.forensics.candidate_comparisons) == 1
    candidate = enriched.forensics.candidate_comparisons[0]
    assert candidate.san == "a3"
    assert candidate.uci == "a2a3"
    assert candidate.resulting_fen != board.fen()


def _rpc(arguments: dict[str, object]) -> bytes:
    import json

    return json.dumps(
        {
            "params": {
                "name": "top_moves",
                "arguments": arguments,
            }
        }
    ).encode()


def test_request_cost_charges_candidate_comparison_and_tactical_proof() -> None:
    standard = estimate_mcp_request_cost(_rpc({"depth": 20, "n": 3}))
    compared = estimate_mcp_request_cost(
        _rpc({"depth": 20, "n": 3, "include_moves": ["g4", "gxh4"]})
    )
    proof = estimate_mcp_request_cost(
        _rpc({"depth": 20, "n": 3, "proof_mode": "tactical", "proof_defenses": 5})
    )

    assert compared > standard
    assert proof > compared
