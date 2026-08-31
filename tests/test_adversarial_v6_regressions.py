from __future__ import annotations

import chess
import pytest
from mcp.server.mcpserver.exceptions import ToolError

from core.engines.types import Eval
from mcp_server import server as server_module


class DeterministicPool:
    name = "AdversarialV6"
    engine_version = "AdversarialV6"

    async def evaluate(
        self,
        board: chess.Board,
        depth: int = 14,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        legal = list(root_moves) if root_moves else list(board.legal_moves)
        best = legal[0].uci() if legal else None
        return Eval(
            cp=0,
            best_move=best,
            pv=[best] if best else [],
            depth=depth,
            wdl=(1, 998, 1),
        )

    async def top_moves(
        self,
        board: chess.Board,
        n: int = 3,
        depth: int = 14,
    ) -> list[Eval]:
        return [
            Eval(
                cp=0,
                best_move=move.uci(),
                pv=[move.uci()],
                depth=depth,
                wdl=(1, 998, 1),
            )
            for move in list(board.legal_moves)[:n]
        ]

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
async def _isolated_server_state():
    old_pool = server_module._analyzer_pool
    await server_module._cache.clear()
    server_module._analyzer_pool = DeterministicPool()  # type: ignore[assignment]
    yield
    await server_module._cache.clear()
    server_module._analyzer_pool = old_pool


@pytest.mark.asyncio
async def test_top_moves_compact_is_identical_on_fresh_and_cached_paths():
    fresh = await server_module.top_moves("startpos", n=3, depth=2, verbosity="compact")
    cached = await server_module.top_moves("startpos", n=3, depth=2, verbosity="compact")

    assert len(fresh.result) == len(cached.result) == 3
    for candidate in fresh.result + cached.result:
        assert candidate.lichess_url is None
        assert candidate.lichess_image is None
        assert candidate.decision_value is None
        assert candidate.engine_eval is None
        assert candidate.input_fen is None

    assert [item.model_dump() for item in fresh.result] == [
        item.model_dump() for item in cached.result
    ]


@pytest.mark.asyncio
async def test_top_moves_reports_fen_canonicalization_like_evaluate_position():
    fen = "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1"

    evaluated = await server_module.evaluate_position(fen, depth=2, verbosity="compact")
    candidates = await server_module.top_moves(fen, n=1, depth=2, verbosity="compact")

    assert evaluated.canonical_fen == candidates.canonical_fen
    assert evaluated.fen_was_canonicalized is True
    assert candidates.fen_was_canonicalized is True


@pytest.mark.asyncio
async def test_evaluate_position_does_not_confuse_replayed_moves_with_fen_rewrite():
    fen = chess.STARTING_FEN
    result = await server_module.evaluate_position(
        fen,
        moves=["e4"],
        depth=2,
        verbosity="full",
    )

    assert result.input_fen == fen
    assert result.canonical_fen != fen
    assert result.fen_was_canonicalized is False


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["evaluate_position", "top_moves"])
async def test_invalid_verbosity_uses_structured_tool_error(tool_name: str):
    tool = getattr(server_module, tool_name)
    kwargs = {"fen": "startpos", "depth": 1, "verbosity": "definitely-invalid"}
    if tool_name == "top_moves":
        kwargs["n"] = 1

    with pytest.raises(ToolError, match="INVALID_VERBOSITY"):
        await tool(**kwargs)


@pytest.mark.asyncio
async def test_strict_pgn_rejects_lexically_malformed_header():
    malformed = '[White "Alice]\n[Black "Bob"]\n[Result "*"]\n\n*'

    with pytest.raises(ToolError, match="STRICT_VALIDATION_ERROR"):
        await server_module.analyze_game(malformed, depth=1, strict=True)


@pytest.mark.asyncio
async def test_nonstrict_malformed_header_preserves_valid_metadata_and_warns():
    malformed = (
        '[White "Alice"]\n'
        '[Black "Bob"]\n'
        '[Foo bar]\n'
        '[Result "*"]\n\n'
        '1. e4 e5 *'
    )

    result = await server_module.analyze_game(malformed, depth=1, strict=False)

    assert result.white == "Alice"
    assert result.black == "Bob"
    assert result.result_header == "*"
    assert any("malformed" in warning.lower() for warning in result.metadata_warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fen",
    [
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ],
)
async def test_terminal_evaluate_has_same_typed_game_over_action_as_top_moves(fen: str):
    evaluated = await server_module.evaluate_position(fen, depth=2, verbosity="full")
    candidates = await server_module.top_moves(fen, n=3, depth=2, verbosity="full")

    assert evaluated.recommended_action == "game_over"
    assert evaluated.best_action_obj == candidates.best_action_obj
    assert evaluated.legal_actions == candidates.legal_actions
    assert evaluated.best_action_obj is not None
    assert evaluated.best_action_obj["type"] == "game_over"
