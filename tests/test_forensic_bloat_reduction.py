from __future__ import annotations

import pytest

from mcp_server.tools.analyze_game import analyze_game
from mcp_server.tools.classify_move import classify_move
from mcp_server.tools.evaluate_position import evaluate_position
from mcp_server.tools.top_moves import top_moves

TEST_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
TEST_PGN = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Ng5 d5 5. exd5 Nxd5 6. Nxf7 Kxf7 7. Qf3+ Ke6 8. Nc3"


@pytest.mark.asyncio
async def test_forensic_bloat_reduction_across_all_tools() -> None:
    # 1. classify_move
    cm_res = await classify_move(
        fen=TEST_FEN,
        move="f3g5",
        detail="forensic",
        depth=10,
    )
    cm_len = len(cm_res.model_dump_json())
    assert cm_len < 75_000
    assert cm_res.played == "f3g5"
    assert cm_res.played_san == "Ng5"
    assert cm_res.move_class is not None
    assert cm_res.forensics is not None
    assert cm_res.forensics.inference_boundary is None
    traces = [
        m for m in cm_res.forensics.mechanism_evidence
        if m.get("mechanism") == "causal_position_delta_trace"
    ]
    assert len(traces) == 1
    assert "transition_anchors" in traces[0]

    # 2. top_moves
    tm_full = await top_moves(
        fen=TEST_FEN,
        detail="forensic",
        verbosity="full",
        n=3,
        depth=10,
    )
    tm_comp = await top_moves(
        fen=TEST_FEN,
        detail="forensic",
        verbosity="compact",
        n=3,
        depth=10,
    )
    tm_full_len = len(tm_full.model_dump_json())
    tm_comp_len = len(tm_comp.model_dump_json())
    assert tm_comp_len < 75_000
    assert tm_comp_len < tm_full_len * 0.55
    assert len(tm_comp.result) == len(tm_full.result) == 3
    assert tm_comp.result[0].best_move == tm_full.result[0].best_move
    assert tm_comp.forensics is not None
    for cand in tm_comp.forensics.candidate_comparisons:
        if cand.continuation_endpoint is not None:
            assert cand.continuation_endpoint.endpoint_tactical_snapshot is None
            assert cand.continuation_endpoint.root_to_endpoint_delta is None

    # 3. evaluate_position
    ev_full = await evaluate_position(
        fen=TEST_FEN,
        detail="forensic",
        verbosity="full",
        depth=10,
    )
    ev_comp = await evaluate_position(
        fen=TEST_FEN,
        detail="forensic",
        verbosity="compact",
        depth=10,
    )
    ev_full_len = len(ev_full.model_dump_json())
    ev_comp_len = len(ev_comp.model_dump_json())
    assert ev_comp_len < 15_000
    assert ev_comp_len < ev_full_len * 0.60
    assert ev_comp.best_move == ev_full.best_move
    assert ev_comp.forensics is not None
    assert ev_comp.forensics.inference_boundary is None

    # 4. analyze_game
    ag_res = await analyze_game(
        pgn=TEST_PGN,
        detail="forensic",
        verbosity="compact",
        depth=10,
    )
    ag_len = len(ag_res.model_dump_json())
    assert ag_len < 15_000
    assert ag_res.coaching is not None
    assert ag_res.coaching.inference_boundary is not None
    for cm in ag_res.coaching.critical_moments:
        assert cm.inference_boundary is None
