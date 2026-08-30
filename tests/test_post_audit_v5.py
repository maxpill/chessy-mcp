from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import chess
import pytest

from core.engines.types import Eval, MoveClass
from mcp_server import config as config_module
from mcp_server import server as server_module
from mcp_server.models import MCPEval, score_played_move


def _claim_score(halfmove: int, action_type: str):
    board = chess.Board(f"7k/5Q2/6K1/8/8/8/8/8 w - - {halfmove} 51")
    move = chess.Move.from_uci("f7g7")
    assert move in board.legal_moves
    after = board.copy(stack=True)
    after.push(move)
    assert after.is_checkmate()

    before_eval = MCPEval.from_eval(
        Eval(mate=1, best_move="f7g7", pv=["f7g7"], depth=4),
        board.fen(),
        board=board,
        history_complete="incomplete",
    )
    after_eval = MCPEval.from_eval(
        Eval(mate=0, best_move=None, pv=[], depth=0),
        after.fen(),
        board=after,
        history_complete="incomplete",
    )
    return score_played_move(
        board,
        move,
        before_eval,
        after_eval,
        after,
        action_type=action_type,  # type: ignore[arg-type]
    )


def test_claim_now_does_not_become_mating_play_when_placeholder_move_mates():
    score = _claim_score(100, "claim_draw")
    assert score.move_class == MoveClass.BLUNDER
    assert score.best_action == "play_move"
    assert score.is_best_action is False
    assert score.raw_centipawn_delta == 0


def test_intended_claim_does_not_become_mating_play_when_intended_move_mates():
    score = _claim_score(99, "claim_draw_with_intended_move")
    assert score.move_class == MoveClass.BLUNDER
    assert score.best_action == "play_move"
    assert score.is_best_action is False
    assert score.raw_centipawn_delta == 0


@pytest.mark.parametrize(
    "value",
    [
        "300",
        "40/7200",
        "300+5",
        "*60",
        "40/7200:3600",
        "40/7200:3600+30",
        "40/7200:20/3600:900+30",
        "?",
        "-",
    ],
)
def test_pgn_time_control_accepts_standard_forms(value: str):
    assert server_module._is_valid_pgn_time_control(value) is True


@pytest.mark.parametrize("value", ["", "40//7200", "40/", ":300", "300:", "300++5", "abc"])
def test_pgn_time_control_rejects_malformed_forms(value: str):
    assert server_module._is_valid_pgn_time_control(value) is False


@pytest.mark.asyncio
async def test_unauthenticated_post_is_rejected_before_body_is_read(monkeypatch: pytest.MonkeyPatch):
    called = False

    async def app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(config_module, "get_mcp_settings", lambda: SimpleNamespace(auth_token="secret"))
    middleware = server_module.ASGIRequestLoggerMiddleware(app, restrict_to_chatgpt=True)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"content-length", b"1048576")],
        "client": ("203.0.113.9", 1234),
    }
    receives = 0

    async def receive() -> dict[str, Any]:
        nonlocal receives
        receives += 1
        return {"type": "http.request", "body": b"x" * 1024, "more_body": False}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)  # type: ignore[arg-type]
    assert called is False
    assert receives == 0
    assert sent[0]["status"] == 403


@pytest.mark.asyncio
async def test_oversized_declared_body_is_rejected_without_reading(monkeypatch: pytest.MonkeyPatch):
    called = False

    async def app(scope: Any, receive: Any, send: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(config_module, "get_mcp_settings", lambda: SimpleNamespace(auth_token="secret"))
    middleware = server_module.ASGIRequestLoggerMiddleware(app, restrict_to_chatgpt=True)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"authorization", b"Bearer secret"),
            (b"content-length", str(middleware.MAX_BUFFERED_BODY + 1).encode("ascii")),
        ],
        "client": ("203.0.113.10", 1234),
    }
    receives = 0

    async def receive() -> dict[str, Any]:
        nonlocal receives
        receives += 1
        return {"type": "http.request", "body": b"", "more_body": False}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await middleware(scope, receive, send)  # type: ignore[arg-type]
    assert called is False
    assert receives == 0
    assert sent[0]["status"] == 413
