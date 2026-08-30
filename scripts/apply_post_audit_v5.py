from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, got {count}")
    return text.replace(old, new, 1)


# 1. Procedural actions must be scored before consequences of the placeholder/intended move.
models_path = Path("mcp_server/models.py")
models = models_path.read_text(encoding="utf-8")
models = replace_once(
    models,
    '    # 1. Delivered Checkmate (mover wins)\n    if board_after.is_checkmate() and board_after.turn != board_before.turn:\n',
    '    # 1. Delivered Checkmate (mover wins). Only a play_move action actually\n'
    '    # executes ``move``. Draw claims are procedural actions made instead of\n'
    '    # playing the supplied move, so a mating placeholder/intended move must\n'
    '    # not cause a claim to be misclassified as a mating play.\n'
    '    if (\n'
    '        action_type == "play_move"\n'
    '        and board_after.is_checkmate()\n'
    '        and board_after.turn != board_before.turn\n'
    '    ):\n',
    "gate delivered mate on play_move",
)
models = replace_once(
    models,
    '    # 2. Checkmate delivered against mover (mover is in checkmate)\n    if board_after.is_checkmate() and board_after.turn == board_before.turn:\n',
    '    # 2. Checkmate delivered against mover. As above, post-move terminal\n'
    '    # consequences apply only when the requested action is play_move.\n'
    '    if (\n'
    '        action_type == "play_move"\n'
    '        and board_after.is_checkmate()\n'
    '        and board_after.turn == board_before.turn\n'
    '    ):\n',
    "gate losing mate on play_move",
)
# A claim does not execute the supplied move, so its raw board delta is zero.
claim_marker = '    # Procedural draw claim action — but ONLY honor it when the claim is legally\n'
claim_start = models.index(claim_marker)
claim_end = models.index('    # If position before was winning and move blundered into an automatic draw\n', claim_start)
claim_block = models[claim_start:claim_end]
claim_block = claim_block.replace('raw_centipawn_delta=raw_board_delta,', 'raw_centipawn_delta=0,')
models = models[:claim_start] + claim_block + models[claim_end:]
models_path.write_text(models, encoding="utf-8")


# 2. PGN TimeControl supports staged controls and hourglass notation.
server_path = Path("mcp_server/server.py")
server = server_path.read_text(encoding="utf-8")
server = replace_once(
    server,
    'import asyncio\nimport io\nimport ipaddress\n',
    'import asyncio\nimport hmac\nimport io\nimport ipaddress\n',
    "import hmac",
)
helper_anchor = '''    if re.search(r"\\bdraw\\s+by\\s+agreement\\b|\\bagreement\\b", t):
        return "draw_agreement"
    return None


def _find_movetext_result(text: str) -> str | None:
'''
helper_replacement = '''    if re.search(r"\\bdraw\\s+by\\s+agreement\\b|\\bagreement\\b", t):
        return "draw_agreement"
    return None


_TIME_CONTROL_STAGE_RE = re.compile(r"^(?:\\d+|\\d+/\\d+|\\d+\\+\\d+|\\*\\d+)$")


def _is_valid_pgn_time_control(value: str) -> bool:
    """Validate the PGN TimeControl tag grammar.

    PGN permits a single stage or colon-separated stages. A stage is one of:
    sudden-death seconds (``300``), moves/seconds (``40/7200``), Fischer
    seconds+increment (``300+5``), or hourglass (``*60``). ``?`` and ``-``
    are the standard unknown/unspecified markers.
    """
    text = value.strip()
    if text in {"?", "-"}:
        return True
    if not text:
        return False
    stages = text.split(":")
    return all(bool(stage) and _TIME_CONTROL_STAGE_RE.fullmatch(stage) is not None for stage in stages)


def _find_movetext_result(text: str) -> str | None:
'''
server = replace_once(server, helper_anchor, helper_replacement, "insert TimeControl validator")
old_tc = '''        if time_control_val is not None and time_control_val not in ("-", "?"):
            if not re.match(r"^\\d+(?:\\+\\d+)?$|^\\d+/\\d+$", time_control_val):
                metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")
'''
new_tc = '''        if time_control_val is not None and not _is_valid_pgn_time_control(time_control_val):
            metadata_warnings.append(f"Invalid TimeControl header tag '{time_control_val}'.")
'''
server = replace_once(server, old_tc, new_tc, "TimeControl metadata validation")

# 3. Authenticate before buffering a potentially large POST body, use constant-time
# token comparison, and reject oversized Content-Length without reading the body.
old_after_options = '''        request_cost = 1.0
        if method == "POST":
            chunks: list[bytes] = []
'''
new_after_options = '''        # Authentication is header-only. Reject an unauthenticated caller before
        # reading/buffering a potentially large POST body, otherwise the auth wall
        # itself can be used as a memory/CPU amplification point.
        if path != "/health" and self.restrict_to_chatgpt:
            from .config import get_mcp_settings

            auth_token = get_mcp_settings().auth_token
            provided = headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
            authorization = headers_dict.get(b"authorization", b"").decode("utf-8", "ignore").strip()
            bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""

            def token_matches(candidate: str) -> bool:
                return bool(auth_token) and hmac.compare_digest(candidate, auth_token)

            if not (token_matches(provided) or token_matches(bearer)):
                log.warning("Blocked unauthenticated client ip=%s ua=%r origin=%r", client_ip, ua, origin)
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: valid MCP auth token required"}}\\n'
                await send(cast(Message, {"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii"))]}))
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

        request_cost = 1.0
        if method == "POST":
            raw_content_length = headers_dict.get(b"content-length", b"").decode("ascii", "ignore").strip()
            if raw_content_length:
                try:
                    declared_length = int(raw_content_length)
                except ValueError:
                    declared_length = -1
                if declared_length > self.MAX_BUFFERED_BODY:
                    response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Request body too large"}}\\n'
                    await send(cast(Message, {"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii"))]}))
                    await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                    return

            chunks: list[bytes] = []
'''
server = replace_once(server, old_after_options, new_after_options, "early auth/body-size admission")
old_late_auth = '''        if self.restrict_to_chatgpt:
            from .config import get_mcp_settings

            auth_token = get_mcp_settings().auth_token
            provided = headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
            authorization = headers_dict.get(b"authorization", b"").decode("utf-8", "ignore").strip()
            bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            valid = bool(auth_token) and (provided == auth_token or bearer == auth_token)
            if not valid:
                log.warning("Blocked unauthenticated client ip=%s ua=%r origin=%r", client_ip, ua, origin)
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: valid MCP auth token required"}}\\n'
                await send(cast(Message, {"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii"))]}))
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

'''
server = replace_once(server, old_late_auth, '', "remove late auth block")
server_path.write_text(server, encoding="utf-8")


# Permanent v5 regression coverage.
test_path = Path("tests/test_post_audit_v5.py")
test_path.write_text(r'''from __future__ import annotations

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
''', encoding="utf-8")
