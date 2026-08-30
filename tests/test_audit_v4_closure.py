from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, get_type_hints

import chess
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from core.engines.analysis import classify_move as core_classify_move
from core.engines.types import Eval, MoveAnalysis, MoveClass
from core.engines.pool import _EnginePool
from mcp_server import cache as cache_module
from mcp_server import config as config_module
from mcp_server import server as server_module
from mcp_server.actions import build_legal_actions
from mcp_server.models import MCPEval
from mcp_server.rules import RuleStatus, can_checkmate, evaluate_rule_status, validate_mating_possibility
from mcp_server.tcp_analyzer import _white_pov_wdl as tcp_white_pov_wdl


class FlatPool:
    name = "AuditFlat"
    engine_version = "AuditFlat"

    def __init__(self, cp: int = 0, best_move: str | None = None) -> None:
        self.cp = cp
        self.best_move = best_move
        self.eval_calls = 0

    async def evaluate(
        self,
        board: chess.Board,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        self.eval_calls += 1
        legal = list(root_moves) if root_moves else list(board.legal_moves)
        best = self.best_move
        if not best or not any(m.uci() == best for m in legal):
            best = legal[0].uci() if legal else None
        return Eval(
            cp=self.cp,
            best_move=best,
            pv=[best] if best else [],
            depth=depth,
            wdl=(333, 334, 333),
        )

    async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
        legal = list(board.legal_moves)[:n]
        return [
            Eval(cp=self.cp, best_move=m.uci(), pv=[m.uci()], depth=depth, wdl=(333, 334, 333))
            for m in legal
        ]

    async def classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        depth: int = 14,
    ) -> MoveAnalysis:
        ev_before = Eval(cp=self.cp, best_move=move.uci(), pv=[move.uci()], depth=depth)
        ev_after = Eval(cp=self.cp, best_move=None, pv=[], depth=depth)
        return MoveAnalysis(
            played=move.uci(),
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            eval_before=ev_before,
            eval_after=ev_after,
            best_move_san=board.san(move),
        )

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _isolated_server_state():
    old_pool = server_module._analyzer_pool
    await server_module._cache.clear()
    server_module._analyzer_pool = FlatPool()
    yield
    await server_module._cache.clear()
    server_module._analyzer_pool = old_pool


@pytest.mark.asyncio
async def test_01_unknown_action_type_rejected():
    with pytest.raises(ToolError, match="INVALID_ACTION_TYPE"):
        await server_module.classify_move("startpos", "e4", depth=1, action_type="banana")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_02_unavailable_claim_rejected():
    with pytest.raises(ToolError, match="ILLEGAL_ACTION"):
        await server_module.classify_move("startpos", "e4", depth=1, action_type="claim_draw")


@pytest.mark.asyncio
async def test_03_wrong_intended_claim_move_rejected():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 1"
    with pytest.raises(ToolError, match="ILLEGAL_ACTION"):
        await server_module.classify_move(
            fen,
            "e4",
            depth=1,
            action_type="claim_draw_with_intended_move",
        )


def test_04_fifty_claim_pawn_reset_rejected():
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 99 1")
    status = evaluate_rule_status(board, history_complete="incomplete")
    assert chess.Move.from_uci("e2e4") not in status.intended_claim_moves


def test_05_fifty_claim_capture_reset_rejected():
    board = chess.Board("7k/8/8/8/8/8/Rr6/K7 w - - 99 50")
    assert chess.Move.from_uci("a2b2") in board.legal_moves
    status = evaluate_rule_status(board, history_complete="incomplete")
    assert chess.Move.from_uci("a2b2") not in status.intended_claim_moves


@pytest.mark.asyncio
async def test_06_played_action_obj_matches_requested_action_type():
    play = await server_module.classify_move("startpos", "e4", depth=1)
    assert play.played_action_obj is not None
    assert play.played_action_obj["type"] == "play_move"

    fen_now = "7k/8/8/8/8/8/R7/K7 w - - 100 51"
    claim = await server_module.classify_move(
        fen_now,
        "Ra3",
        depth=1,
        action_type="claim_draw",
    )
    assert claim.played_action_obj is not None
    assert claim.played_action_obj["type"] == "claim_draw"

    fen_intended = "7k/8/8/8/8/8/R7/K7 w - - 99 51"
    intended = await server_module.classify_move(
        fen_intended,
        "Ra3",
        depth=1,
        action_type="claim_draw_with_intended_move",
    )
    assert intended.played_action_obj is not None
    assert intended.played_action_obj["type"] == "claim_draw_with_intended_move"
    assert intended.played_action_obj["intended_move"]["uci"] == "a2a3"


@pytest.mark.asyncio
async def test_07_is_best_action_requires_match_or_equivalence():
    fen = "7k/8/8/8/8/8/R7/K7 w - - 100 51"
    res = await server_module.classify_move(fen, "Ra3", depth=1, action_type="play_move")
    if res.best_action != "play_move" and not res.action_equivalent:
        assert res.is_best_action is False


@pytest.mark.asyncio
async def test_08_engine_best_alias_fields_match():
    res = await server_module.classify_move("startpos", "e4", depth=1)
    assert res.is_engine_best == res.is_best_engine_move


def test_09_startpos_zero_ply_history_complete():
    assert server_module._history_provenance_for_input("startpos", None) == "complete"


def test_10_full_pgn_history_complete():
    assert server_module._history_provenance_for_input("1. Nf3 Nf6 2. Ng1 Ng8", None) == "complete"


def test_11_arbitrary_fen_history_incomplete():
    assert (
        server_module._history_provenance_for_input(chess.STARTING_FEN, None)
        == "incomplete"
    )


def test_12_arbitrary_fen_plus_suffix_history_partial():
    assert (
        server_module._history_provenance_for_input(chess.STARTING_FEN, ["Nf3"])
        == "partial"
    )


def test_13_partial_history_can_prove_threefold():
    board = server_module._build_board(
        chess.STARTING_FEN,
        ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"],
    )
    status = evaluate_rule_status(board, history_complete="partial")
    assert status.repetition_status == "threefold_claimable"
    assert "threefold_repetition" in status.claim_reasons_now


def test_14_partial_history_cannot_disprove_prior_repetition():
    board = server_module._build_board(chess.STARTING_FEN, ["Nf3"])
    status = evaluate_rule_status(board, history_complete="partial")
    assert status.repetition_status == "unknown"


@pytest.mark.asyncio
async def test_15_endpoint_order_cache_invariance():
    pool = FlatPool(cp=17)
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    fen = chess.STARTING_FEN

    complete = await server_module.evaluate_position("startpos", depth=2)
    incomplete = await server_module.evaluate_position(fen, depth=2)
    assert complete.history_completeness == "complete"
    assert incomplete.history_completeness == "incomplete"
    assert pool.eval_calls == 2

    await server_module._cache.clear()
    pool.eval_calls = 0
    incomplete2 = await server_module.evaluate_position(fen, depth=2)
    complete2 = await server_module.evaluate_position("startpos", depth=2)
    assert incomplete2.history_completeness == "incomplete"
    assert complete2.history_completeness == "complete"
    assert pool.eval_calls == 2


@pytest.mark.asyncio
async def test_16_ponder_preserves_history_provenance_and_stack(monkeypatch: pytest.MonkeyPatch):
    board = chess.Board()
    board.push_san("Nf3")
    board.push_san("Nf6")
    seen: dict[str, Any] = {}

    async def fake_eval(
        b: chess.Board,
        depth: int,
        pool: Any,
        requested_depth: int | None = None,
        history_complete: str | bool = "incomplete",
        reuse_tt: bool = False,
        analyzer: object | None = None,
    ) -> tuple[MCPEval, bool]:
        seen["stack"] = [m.uci() for m in b.move_stack]
        seen["history"] = history_complete
        ev = MCPEval.from_eval(Eval(cp=0, depth=depth), b.fen(), board=b, history_complete=history_complete)
        return ev, False

    monkeypatch.setattr(server_module, "_evaluate_game_position_cached", fake_eval)
    await server_module._ponder_warm_cache(FlatPool(), board, 2, "complete")  # type: ignore[arg-type]
    assert seen["stack"] == ["g1f3", "g8f6"]
    assert seen["history"] == "complete"


def test_17_compact_semantic_invariance():
    board = chess.Board()
    full = MCPEval.from_eval(
        Eval(cp=12, best_move="e2e4", pv=["e2e4"], depth=4),
        board.fen(),
        board=board,
        history_complete="incomplete",
    )
    compact = server_module._compact_mcpeval(full)
    for field in (
        "cp",
        "mate",
        "best_move",
        "pv",
        "status",
        "winner",
        "recommended_action",
        "history_completeness",
        "repetition_status",
        "can_claim_now",
        "can_claim_draw",
    ):
        assert getattr(compact, field) == getattr(full, field)


def test_18_fifty_claim_not_threefold_status():
    board = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 100 51")
    status = evaluate_rule_status(board, history_complete="incomplete")
    assert status.can_claim_now is True
    assert "fifty_moves" in status.claim_reasons_now
    assert status.repetition_status != "threefold_claimable"


@pytest.mark.asyncio
async def test_19_cross_tool_root_action_policy():
    fen = "7k/8/8/8/8/8/R6q/K7 w - - 100 51"
    pool = FlatPool(cp=-500, best_move="a2a3")
    server_module._analyzer_pool = pool  # type: ignore[assignment]
    ev = await server_module.evaluate_position(fen, depth=2)
    top = await server_module.top_moves(fen, n=1, depth=2)
    assert ev.recommended_action == top.recommended_action
    assert ev.best_action_obj is not None
    assert top.best_action_obj is not None
    assert ev.best_action_obj["type"] == top.best_action_obj["type"]


def test_20_wdl_white_pov_white_to_move():
    assert tcp_white_pov_wdl((700, 200, 100), chess.WHITE) == (700, 200, 100)


def test_21_wdl_white_pov_black_to_move():
    assert tcp_white_pov_wdl((700, 200, 100), chess.BLACK) == (100, 200, 700)


def test_22_wdl_percentage_sum_is_100():
    board = chess.Board()
    out = MCPEval.from_eval(
        Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=2, wdl=(333, 333, 334)),
        board.fen(),
        board=board,
        history_complete="complete",
    )
    assert out.wdl_pct is not None
    assert sum(out.wdl_pct.values()) == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_23_candidate_post_winner():
    class MatePool(FlatPool):
        async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
            return [Eval(mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)]

    server_module._analyzer_pool = MatePool()  # type: ignore[assignment]
    res = await server_module.top_moves("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1", n=1, depth=2)
    assert len(res.result) == 1
    cand = res.result[0]
    assert cand.post_terminal_status == "checkmate"
    assert cand.post_position is not None
    assert cand.post_position["winner"] == "white"


def test_24_kb_vs_k_cannot_mate():
    board = chess.Board("7k/8/8/8/8/8/2B5/K7 w - - 0 1")
    assert can_checkmate(board, chess.WHITE) is False


def test_25_kn_mating_possibility_considers_opponent_material():
    board = chess.Board("7k/7p/8/8/8/8/2N5/K7 w - - 0 1")
    assert can_checkmate(board, chess.WHITE) is True


def test_26_time_forfeit_kb_vs_k_draw():
    board = chess.Board("7k/8/8/8/8/8/2B5/K7 w - - 0 1")
    result, warnings = validate_mating_possibility(board, "1-0", "Black lost on time")
    assert result == "1/2-1/2"
    assert warnings


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("White wins on time", "1-0"),
        ("Black wins on time", "0-1"),
        ("White wins by resignation", "1-0"),
        ("Black wins by resignation", "0-1"),
        ("White resigned", "0-1"),
        ("Black resigned", "1-0"),
    ],
)
def test_27_to_29_winner_and_loser_termination_grammar(text: str, expected: str):
    assert server_module._infer_result_from_termination(text) == expected


def test_30_normal_time_control_with_color_no_result_inference():
    assert server_module._infer_result_from_termination("Normal time control - White") is None
    assert server_module.normalize_termination("Normal time control - White") == "normal"


class FakeWorker:
    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_31_32_self_heal_alive_count_and_no_overgrowth():
    async def factory() -> object:
        return FakeWorker()

    pool = _EnginePool([], factory, acquire_timeout=0.1)
    pool._target_size = 1
    pool._self_heal_interval_s = 0.005
    pool._start_self_heal()
    await asyncio.sleep(0.04)
    assert pool._alive_count == 1
    assert pool._q.qsize() == 1
    await asyncio.sleep(0.04)
    assert pool._alive_count == 1
    assert pool._q.qsize() == 1
    await pool.close()


@pytest.mark.asyncio
async def test_33_optional_claim_not_auto_terminal_in_core_classifier():
    board = chess.Board("7k/8/8/8/8/8/R7/K7 w - - 100 51")
    move = chess.Move.from_uci("a2a3")
    assert move in board.legal_moves

    class Backend:
        name = "backend"

        def __init__(self) -> None:
            self.calls = 0

        async def evaluate(
            self,
            b: chess.Board,
            depth: int | None = None,
            root_moves: list[chess.Move] | None = None,
        ) -> Eval:
            self.calls += 1
            if self.calls == 1:
                return Eval(cp=200, best_move="a2a4", pv=["a2a4"], depth=2)
            return Eval(cp=150, best_move="h8g8", pv=["h8g8"], depth=2)

        async def top_moves(self, board: chess.Board, n: int = 3, depth: int | None = None) -> list[Eval]:
            return []

        async def close(self) -> None:
            return None

    backend = Backend()
    out = await core_classify_move(backend, board, move, depth=2)
    assert backend.calls == 2
    assert out.eval_after.cp == 150


def test_34_find_movetext_result_optional_return_type():
    hints = get_type_hints(server_module._find_movetext_result)
    assert hints["return"] == (str | None)


def test_35_castling_alias_parser_must_not_swallow_exceptions():
    for alias in ("O-O", "0-0"):
        board = server_module._build_board(
            f"1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. {alias} Be7 5. d3 {alias}",
            [],
        )
        assert board.king(chess.WHITE) == chess.G1
        assert board.king(chess.BLACK) == chess.G8


def test_36_en_passant_asserts_final_board():
    board = server_module._build_board("1. e4 a6 2. e5 d5 3. exd6 e.p.", [])
    assert board.piece_at(chess.D6) == chess.Piece(chess.PAWN, chess.WHITE)
    assert board.piece_at(chess.D5) is None
    assert board.move_stack[-1].uci() == "e5d6"


@pytest.mark.asyncio
async def test_37_black_ranking_asserts_order():
    class RankingPool(FlatPool):
        async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
            return [
                Eval(cp=50, best_move="e7e5", pv=["e7e5"], depth=depth),
                Eval(cp=-80, best_move="d7d5", pv=["d7d5"], depth=depth),
                Eval(cp=10, best_move="g8f6", pv=["g8f6"], depth=depth),
            ][:n]

    server_module._analyzer_pool = RankingPool()  # type: ignore[assignment]
    res = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 100 1",
        n=3,
        depth=2,
    )
    assert [x.best_move for x in res.result] == ["d7d5", "g8f6", "e7e5"]


def test_38_39_cache_version_tracks_action_and_transport_semantics():
    required = {
        "mcp_server/actions.py",
        "mcp_server/tcp_analyzer.py",
        "mcp_server/tcp_client.py",
        "core/engines/analyzer.py",
        "core/engines/analysis.py",
        "core/engines/grading.py",
        "core/winprob.py",
    }
    assert required <= set(cache_module._LOGIC_FILES)


@pytest.mark.asyncio
async def test_40_authenticated_client_boundary_rejects_spoofed_chatgpt_ua(monkeypatch: pytest.MonkeyPatch):
    called = False

    async def app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal called
        called = True
        body = b"ok"
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]})
        await send({"type": "http.response.body", "body": body})

    monkeypatch.setattr(
        config_module,
        "get_mcp_settings",
        lambda: SimpleNamespace(auth_token="secret"),
    )
    middleware = server_module.ASGIRequestLoggerMiddleware(app, restrict_to_chatgpt=True)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp",
        "headers": [(b"user-agent", b"openai-chatgpt"), (b"origin", b"https://chatgpt.com")],
        "client": ("203.0.113.5", 1234),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)  # type: ignore[arg-type]
    assert called is False
    assert sent[0]["status"] == 403


def test_41_compute_weighted_rate_limit():
    def rpc(name: str, arguments: dict[str, Any]) -> bytes:
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        ).encode()

    cheap = server_module._estimate_mcp_request_cost(
        rpc("evaluate_position", {"fen": "startpos", "depth": 1})
    )
    multipv = server_module._estimate_mcp_request_cost(
        rpc("top_moves", {"fen": "startpos", "depth": 30, "n": 20})
    )
    long_game = server_module._estimate_mcp_request_cost(
        rpc("analyze_game", {"pgn": "1. e4 e5 " * 200, "depth": 30})
    )
    assert cheap < multipv
    assert cheap < long_game


def test_extra_intended_claim_reasons_are_bound_per_move():
    status = RuleStatus(
        can_claim_with_intended_move=True,
        intended_claim_ucis=["a1a2", "a1b1"],
        intended_claim_sans=["Ka2", "Kb1"],
        claim_reasons=["fifty_moves", "threefold_repetition"],
        intended_claim_reasons_by_uci={
            "a1a2": ["fifty_moves"],
            "a1b1": ["threefold_repetition"],
        },
    )
    actions = build_legal_actions(status, None, chess.Board(), None)
    intended = [a for a in actions if a["type"] == "claim_draw_with_intended_move"]
    assert {(a["intended_move"]["uci"], a["reason"]) for a in intended} == {
        ("a1a2", "fifty_moves"),
        ("a1b1", "threefold_repetition"),
    }


def test_extra_cache_history_provenance_is_part_of_key():
    board = chess.Board()
    complete = cache_module.eval_cache_key(board, 4, history_completeness="complete")
    partial = cache_module.eval_cache_key(board, 4, history_completeness="partial")
    incomplete = cache_module.eval_cache_key(board, 4, history_completeness="incomplete")
    assert len({complete, partial, incomplete}) == 3


def test_extra_build_sha_is_injected_into_production_image_and_compose():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile.mcp").read_text(encoding="utf-8")
    compose = (root / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "ARG BUILD_SHA" in dockerfile
    assert "BUILD_SHA=${BUILD_SHA}" in dockerfile
    assert "BUILD_SHA: ${BUILD_SHA:-unknown}" in compose
    assert "BUILD_SHA=${BUILD_SHA:-unknown}" in compose
