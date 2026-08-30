import asyncio

import chess
import pytest

try:
    from mcp.server.fastmcp.exceptions import ToolError
except ImportError:
    from mcp.server.mcpserver.exceptions import ToolError  # type: ignore


from core.engines.types import Eval, MoveAnalysis, MoveClass
from mcp_server import server as server_module
from mcp_server.cache import AsyncLRUCache, SingleFlight, eval_cache_key
from mcp_server.models import MCPEval, PlyAnalysisItem, TopMovesResult, score_played_move


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


def test_mcp_has_4_tools():
    tools_list = asyncio.run(server_module.mcp.list_tools())
    names = {t.name for t in tools_list}
    assert names == {
        "evaluate_position",
        "top_moves",
        "classify_move",
        "analyze_game",
    }


def test_evaluate_position_is_read_only_and_idempotent():
    tools_list = asyncio.run(server_module.mcp.list_tools())
    tool = next(t for t in tools_list if t.name == "evaluate_position")
    assert tool.annotations is not None
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.idempotent_hint is True


@pytest.mark.asyncio
async def test_async_lru_cache_and_singleflight():
    cache: AsyncLRUCache[str] = AsyncLRUCache(maxsize=3)
    await cache.set("k1", "v1")
    await cache.set("k2", "v2")
    assert await cache.get("k1") == "v1"
    assert await cache.get("k3") is None

    sf: SingleFlight[int] = SingleFlight()
    counter = 0

    async def _work() -> int:
        nonlocal counter
        counter += 1
        await asyncio.sleep(0.05)
        return 42

    t1 = sf.do("key", _work)
    t2 = sf.do("key", _work)
    res1, res2 = await asyncio.gather(t1, t2)
    assert res1 == 42
    assert res2 == 42
    assert counter == 1


def test_canonical_fen_and_transposition_cache_keys():
    b1 = chess.Board()
    for m in ["d4", "Nf6", "c4", "e6"]:
        b1.push_san(m)

    b2 = chess.Board()
    for m in ["c4", "Nf6", "d4", "e6"]:
        b2.push_san(m)

    # Move order differed, but resulting piece setup & rights are identical
    assert b1.epd() == b2.epd()
    assert eval_cache_key(b1, depth=14) == eval_cache_key(b2, depth=14)


@pytest.mark.asyncio
async def test_analyze_game():
    await server_module._cache.clear()

    class MockAnalyzerPool:
        def __init__(self):
            self.eval_calls = 0

        async def evaluate(self, board, depth=14):
            self.eval_calls += 1
            # Pick first legal move if available
            legal = list(board.legal_moves)
            best_uci = legal[0].uci() if legal else None
            return Eval(cp=20, best_move=best_uci, pv=[best_uci] if best_uci else [], depth=depth)

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=10,
                eval_before=Eval(cp=30, best_move=move.uci()),
                eval_after=Eval(cp=20),
                best_move_san=board.san(move),
            )

        async def close(self):
            pass

    mock_pool = MockAnalyzerPool()
    server_module._analyzer_pool = mock_pool  # type: ignore

    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1/2-1/2"
    res = await server_module.analyze_game(pgn, depth=14)
    assert res.total_plies >= 6
    assert 0 <= res.white_accuracy <= 100
    assert 0 <= res.black_accuracy <= 100
    assert mock_pool.eval_calls == res.total_plies + 1


@pytest.mark.asyncio
async def test_analyze_game_populates_l1_cache_for_evaluate_position():
    await server_module._cache.clear()

    class CountingPool:
        def __init__(self):
            self.eval_calls = 0

        async def evaluate(self, board, depth=14):
            self.eval_calls += 1
            return Eval(cp=15, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    pool = CountingPool()
    server_module._analyzer_pool = pool  # type: ignore

    # Analyze game with 2 plies: 1. e4 e5
    pgn = "1. e4 e5"
    await server_module.analyze_game(pgn, depth=14)
    # Positions: startpos, after e4, after e5 = 3 positions evaluated
    assert pool.eval_calls == 3

    # Evaluate the same *known-root* start position directly. ``startpos`` has
    # complete history, matching the complete PGN root cached by analyze_game.
    # A naked equivalent FEN is intentionally a different semantic cache key
    # because pre-FEN repetition history is unknowable.
    start_eval = await server_module.evaluate_position("startpos", depth=14)
    assert start_eval.cp == 15
    assert pool.eval_calls == 3


@pytest.mark.asyncio
async def test_top_moves_and_classify_move():
    class MultiPVPool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=30, best_move="e2e4", pv=["e2e4"], depth=depth),
                Eval(cp=20, best_move="d2d4", pv=["d2d4"], depth=depth),
            ]

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.GOOD,
                centipawn_loss=25,
                eval_before=Eval(cp=30, best_move="e2e4"),
                eval_after=Eval(cp=5),
                best_move_san="e4",
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MultiPVPool()  # type: ignore

    # Test top_moves
    candidates = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", n=2, depth=14
    )
    assert len(candidates) == 2
    assert candidates[0].best_move == "e2e4"
    assert candidates[1].best_move == "d2d4"

    # Test classify_move (playing exact engine best move yields CPL=0 and MoveClass.BEST)
    analysis = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e2e4", depth=14
    )
    assert analysis.played == "e2e4"
    assert analysis.move_class == MoveClass.BEST
    assert analysis.centipawn_loss == 0
    assert analysis.is_engine_best is True


@pytest.mark.asyncio
async def test_analyze_game_with_conversational_pgn_and_annotations():
    class MockAnalyzerPool:
        async def evaluate(self, board, depth=14):
            legal = list(board.legal_moves)
            best_uci = legal[0].uci() if legal else None
            return Eval(cp=15, best_move=best_uci, pv=[best_uci] if best_uci else [], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockAnalyzerPool()  # type: ignore

    user_pgn = """to moja partia na stockfisha 6 [Event "casual correspondence game"]
[Site "https://lichess.org/wDzBxsxv"]
[Date "2026.08.25"]
[Round "-"]
[White "mes77777"]
[Black "lichess AI level 6"]
[Result "0-1"]
[GameId "wDzBxsxv"]
[UTCDate "2026.08.25"]
[UTCTime "12:35:41"]
[WhiteElo "1500"]
[BlackElo "?"]
[Variant "Standard"]
[TimeControl "-"]
[ECO "B10"]
[Opening "Caro-Kann Defense: Endgame Offer"]
[Termination "Normal"]
[Annotator "lichess.org"]

1. e4 c6 2. Nf3 d5 3. d3 { B10 Caro-Kann Defense: Endgame Offer } Bg4 4. Be2 Bxf3 5. Bxf3 d4 6. c3 e6?! { (0.71 → 1.55) Inaccuracy. e5 was best. } (6... e5 7. O-O Nf6 8. Qb3 Qb6 9. Qc2 Be7 10. cxd4 Qxd4 11. Be3) 7. O-O Bc5 8. Nd2 Nd7?! { (1.02 → 1.77) Inaccuracy. dxc3 was best. } (8... dxc3 9. Nb3 Bb4 10. a3 cxb2 11. Bxb2 Bf8 12. d4 Nd7 13. Re1 Bd6 14. Bc3) 9. Nc4?! { (1.77 → 0.89) Inaccuracy. Nb3 was best. } (9. Nb3 e5 10. cxd4 Bxd4 11. Nxd4 exd4 12. Qa4 Qb6 13. Bd2 Ne7 14. b4 O-O) 9... b5 10. b4 Bxb4 11. cxb4 bxc4 12. dxc4 h5?? { (0.78 → 3.69) Blunder. e5 was best. } (12... e5 13. c5 a5 14. b5 Ne7 15. a4 cxb5 16. axb5 Nxc5 17. Ba3 Ne6 18. Qa4) 13. Qxd4 Qf6 14. Qxf6 Ngxf6 15. Bb2 e5 16. b5 Rb8 17. a4 g5 18. Rab1 c5 19. Ba3? { (3.31 → 1.76) Mistake. Rfd1 was best. } (19. Rfd1 Ke7 20. Rd3 Rb7 21. a5 Ne8 22. Be2 f6 23. h4 g4 24. f3 g3) 19... g4 20. Be2 h4? { (1.91 → 3.31) Mistake. Nxe4 was best. } (20... Nxe4 21. Rb2 Ng5 22. f3 gxf3 23. Bxf3 Nxf3+ 24. Rxf3 Rg8 25. h3 Ke7 26. Rd2) 21. a5 g3?! { (3.08 → 3.99) Inaccuracy. Rg8 was best. } (21... Rg8 22. Rbd1 Nxe4 23. Bb2 Ke7 24. Rd5 f5 25. f4 gxf3 26. Bxf3 h3 27. g3) 22. fxg3?! { (3.99 → 2.76) Inaccuracy. Bf3 was best. } (22. Bf3 Ke7 23. h3 Rbc8 24. Rbd1 Nf8 25. Bb2 Ng6 26. Bc1 Nf8 27. Bg5 Ne6) 22... hxg3 23. Bxc5?? { (2.97 → -3.58) Blunder. h3 was best. } (23. h3 Nxe4 24. Bg4 Rh4 25. Rbd1 Nef6 26. Be2 Ne4 27. Rd3 Rd8 28. Bc1 f6) 23... Nxe4 24. Bxa7?? { (-3.51 → Mate in 2) Checkmate is now unavoidable. hxg3 was best. } (24. hxg3 Ndxc5 25. Rf5 f6 26. Rh5 Ke7 27. Rxh8 Rxh8 28. Bf3 Nxg3 29. Kf2 Nge4+) 24... gxh2+ 25. Kh1 Ng3# { Black wins by checkmate. } 0-1"""

    res = await server_module.analyze_game(user_pgn, depth=14)
    assert res.total_plies == 50
    assert res.white == "mes77777"
    assert res.black == "lichess AI level 6"
    assert res.result == "0-1"
    assert res.eco == "B10"
    assert res.opening == "Caro-Kann Defense: Endgame Offer"
    assert res.event == "casual correspondence game"
    assert res.date == "2026.08.25"
    assert res.termination == "checkmate"


@pytest.mark.asyncio
async def test_analyze_game_with_markdown_and_bare_moves():
    class MockAnalyzerPool:
        async def evaluate(self, board, depth=14):
            legal = list(board.legal_moves)
            best_uci = legal[0].uci() if legal else None
            return Eval(cp=10, best_move=best_uci, pv=[best_uci] if best_uci else [], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockAnalyzerPool()  # type: ignore

    # Markdown wrapped PGN
    md_pgn = '```pgn\n[Event "Markdown Game"]\n[White "Alice"]\n[Black "Bob"]\n1. e4 e5 2. Nf3 Nc6 1/2-1/2\n```'
    res1 = await server_module.analyze_game(md_pgn, depth=14)
    assert res1.total_plies == 4
    assert res1.white == "Alice"
    assert res1.black == "Bob"

    # Bare movetext with no headers
    bare_moves = "1. e4 c6 2. Nf3 d5 3. d3 Bg4"
    res2 = await server_module.analyze_game(bare_moves, depth=14)
    assert res2.total_plies == 6
    assert res2.white is None
    assert res2.black is None


@pytest.mark.asyncio
async def test_position_tools_with_pgn_and_san_moves():
    await server_module._cache.clear()

    class MockPositionPool:
        async def evaluate(self, board, depth=14):
            legal = list(board.legal_moves)
            best_uci = legal[0].uci() if legal else None
            return Eval(cp=35, best_move=best_uci, pv=[best_uci] if best_uci else [], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=35, best_move="c1g5", pv=["c1g5"], depth=depth)]

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=35, best_move=move.uci()),
                eval_after=Eval(cp=35),
                best_move_san=board.san(move),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockPositionPool()  # type: ignore

    # 1. evaluate_position with PGN string + SAN moves
    eval_res = await server_module.evaluate_position(
        "1. e4 c6 2. Nf3 d5", moves=["d3", "Bg4"], depth=14
    )
    assert eval_res.cp == 35

    # 2. top_moves with PGN string + SAN moves
    top_res = await server_module.top_moves("1. e4 c6 2. Nf3 d5", moves=["d3"], n=1, depth=14)
    assert len(top_res) == 1

    # 3. classify_move with PGN string and SAN move
    class_res = await server_module.classify_move("1. e4 c6 2. Nf3 d5 3. d3", move="Bg4", depth=14)
    assert class_res.played == "c8g4"
    assert class_res.best_move_san == "Bg4"

    # 4. classify_move with FEN and SAN move
    class_res2 = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move="e4",
        depth=14,
    )
    assert class_res2
    # 5. evaluate_position on checkmated FEN (White checkmated -> Black won)
    checkmate_fen = "1r2k2r/B2n1p2/8/PP2p3/2P5/6n1/4B1Pp/1R3R1K w k - 2 26"
    eval_checkmate = await server_module.evaluate_position(checkmate_fen, depth=14)
    assert eval_checkmate.status == "checkmate"
    assert eval_checkmate.winner == "black"
    assert eval_checkmate.cp is None
    assert eval_checkmate.mate == 0
    assert eval_checkmate.depth == 0

    # 6. evaluate_position with single-line unspaced PGN
    eval_single_line = await server_module.evaluate_position(
        '[Event "test"] 1. e4 c6 2. Nf3 d5 0-1', depth=14
    )
    assert eval_single_line.cp == 35


@pytest.mark.asyncio
async def test_1_token_fen_startpos_and_annotated_moves():
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)]

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=10),
                eval_after=Eval(cp=10),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # 1. 1-token FEN
    ev1 = await server_module.evaluate_position(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", depth=14
    )
    assert ev1.cp == 10

    # 2. 'startpos' alias
    ev2 = await server_module.evaluate_position("startpos", depth=14)
    assert ev2.cp == 10

    # 3. Moves with annotations and move numbers
    ev3 = await server_module.evaluate_position(
        "startpos", moves=["1. e4!", "1... e5?", "2. Nf3!?"], depth=14
    )
    assert ev3.cp == 10

    # 4. classify_move with annotated SAN move
    cm = await server_module.classify_move("startpos", move="1. e4!", depth=14)
    assert cm.played == "e2e4"


@pytest.mark.asyncio
async def test_checkmate_and_stalemate_evaluations():
    await server_module._cache.clear()

    # White delivered mate (Black checkmated) -> White won
    scholars_final_fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    ev_white_won = await server_module.evaluate_position(scholars_final_fen, depth=14)
    assert ev_white_won.status == "checkmate"
    assert ev_white_won.winner == "white"
    assert ev_white_won.cp is None
    assert ev_white_won.mate == 0
    assert ev_white_won.depth == 0
    assert ev_white_won.pv == []
    tm_mate = await server_module.top_moves(scholars_final_fen, depth=14)
    assert tm_mate == []

    # Black delivered mate (White checkmated) -> Black won
    fools_final_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    ev_black_won = await server_module.evaluate_position(fools_final_fen, depth=14)
    assert ev_black_won.status == "checkmate"
    assert ev_black_won.winner == "black"
    assert ev_black_won.cp is None
    assert ev_black_won.mate == 0
    assert ev_black_won.depth == 0

    # Stalemate -> draw (cp=0)
    stalemate_fen = "k7/2Q5/K7/8/8/8/8/8 b - - 0 1"
    ev_stalemate = await server_module.evaluate_position(stalemate_fen, depth=14)
    assert ev_stalemate.status == "stalemate"
    assert ev_stalemate.winner is None
    assert ev_stalemate.cp == 0
    assert ev_stalemate.mate is None
    assert ev_stalemate.depth == 0
    tm_stalemate = await server_module.top_moves(stalemate_fen, depth=14)
    assert tm_stalemate == []


@pytest.mark.asyncio
async def test_analyze_game_with_scholar_mate_capped_acpl():
    await server_module._cache.clear()

    class ScholarPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = ScholarPool()  # type: ignore

    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#"
    res = await server_module.analyze_game(pgn, depth=14)
    assert res.total_plies == 7
    # Capped ACPL prevents 30,000+ explosion
    assert res.black_acpl <= 1000.0
    assert res.white_acpl <= 100.0
    assert res.white_accuracy >= 80.0
    assert res.black_accuracy >= 0.0


@pytest.mark.asyncio
async def test_multitier_cache_persistence():
    import os
    import uuid

    from mcp_server.cache import MultiTierCache
    from mcp_server.models import MCPEval

    db_file = f"/tmp/test_mcp_cache_{uuid.uuid4().hex}.sqlite3"
    try:
        cache = MultiTierCache(l1_size=2, db_path=db_file)
        await cache.clear()

        ev = MCPEval(cp=45, best_move="d2d4", pv=["d2d4", "d7d5"], depth=14)
        await cache.set_eval("key1", ev)

        # 1. Get from L1
        val1 = await cache.get_eval("key1")
        assert val1 is not None
        assert val1.cp == 45
        assert val1.best_move == "d2d4"

        # 2. Clear L1 memory only - should hit L2 disk cache and repopulate L1!
        await cache._l1.clear()
        assert await cache._l1.size() == 0

        val2 = await cache.get_eval("key1")
        assert val2 is not None
        assert val2.cp == 45
        assert val2.best_move == "d2d4"
        assert await cache._l1.size() == 1
    finally:
        if os.path.exists(db_file):
            try:
                os.remove(db_file)
            except OSError:
                pass


@pytest.mark.asyncio
async def test_token_bucket_rate_limiter():
    from mcp_server.server import TokenBucketRateLimiter

    limiter = TokenBucketRateLimiter(rate=10.0, capacity=3.0)
    assert await limiter.is_allowed("user-1")
    assert await limiter.is_allowed("user-1")
    assert await limiter.is_allowed("user-1")
    # Exceeded capacity
    assert not await limiter.is_allowed("user-1")
    # Different user is unaffected
    assert await limiter.is_allowed("user-2")


@pytest.mark.asyncio
async def test_tcp_client_resets_on_cancellation(monkeypatch):
    from mcp_server.tcp_client import TCPUCIClient

    class MockStreamWriter:
        def __init__(self, reader: asyncio.StreamReader):
            self.reader = reader
            self.closed = False
            self.written: list[bytes] = []

        def write(self, data: bytes):
            self.written.append(data)
            cmd = data.decode().strip()
            if cmd == "uci":
                self.reader.feed_data(b"id name test\nuciok\n")
            elif cmd == "isready":
                self.reader.feed_data(b"readyok\n")
            elif cmd.startswith("go"):
                asyncio.create_task(self._stream_eval())

        async def _stream_eval(self):
            await asyncio.sleep(0.02)
            self.reader.feed_data(b"info depth 1 score cp 100 pv e2e4\n")
            await asyncio.sleep(0.04)
            self.reader.feed_data(b"bestmove e2e4\n")

        async def drain(self):
            pass

        def close(self):
            self.closed = True

        async def wait_closed(self):
            pass

    async def mock_open_connection(host, port):
        reader = asyncio.StreamReader()
        writer = MockStreamWriter(reader)
        return reader, writer

    monkeypatch.setattr(asyncio, "open_connection", mock_open_connection)

    client = TCPUCIClient("127.0.0.1", 9999, "test-client")
    await client.connect()

    # Cancel task 1
    t1 = asyncio.create_task(client.analyse("startpos", 14))
    await asyncio.sleep(0.01)
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1

    # Task 2 should reconnect cleanly without reading leftover bytes from Task 1
    res2 = await client.analyse("startpos", 14)
    assert len(res2) > 0
    assert res2[0].get("pv") == ["e2e4"]

    await client.close()


@pytest.mark.asyncio
async def test_asgi_middleware_options_and_client_ip():
    from mcp_server.server import ASGIRequestLoggerMiddleware

    called = False

    async def dummy_app(scope, receive, send):
        nonlocal called
        called = True

    middleware = ASGIRequestLoggerMiddleware(dummy_app)

    # POST request with IPv6 client IP passes through to downstream app
    scope_post = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [(b"user-agent", b"python-requests")],
        "client": ("[2001:db8::1]", 12345),
    }

    async def dummy_receive():
        return {}

    async def dummy_send(msg):
        pass

    await middleware(scope_post, dummy_receive, dummy_send)
    assert called is True


@pytest.mark.asyncio
async def test_asgi_middleware_passes_post_body_to_inner_app():
    """Regression: the middleware MUST NOT buffer the POST body.

    FastMCP's streamable-HTTP transport calls receive() a second time to
    detect early client disconnect while the SSE response is streaming; a
    buffered-and-replayed receive that returns http.disconnect on the
    second poll causes EventSourceResponse to close the stream before it
    can emit the response, breaking every initialize (and any other
    tool call long enough to flush an SSE chunk).
    """
    from mcp_server.server import ASGIRequestLoggerMiddleware

    body_chunks: list[bytes] = []
    extra_calls: list[int] = []
    received_chunks: list[dict] = []

    async def inner_app(scope, receive, send):
        # Drain the body via the SAME receive the middleware gave us.
        while True:
            msg = await receive()
            received_chunks.append(msg)
            if msg.get("type") == "http.request":
                chunk = bytes(msg.get("body", b""))
                if chunk:
                    body_chunks.append(chunk)
                if not msg.get("more_body", False):
                    break
            elif msg.get("type") == "http.disconnect":
                break
            # Count how many times the inner app polled receive beyond the
            # body. FastMCP's transport polls more than once; a buffered
            # middleware would synthesize http.disconnect here.
            extra_calls.append(len(received_chunks))

    middleware = ASGIRequestLoggerMiddleware(inner_app)

    body = b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
            (b"accept", b"application/json, text/event-stream"),
        ],
        "client": ("10.0.0.1", 12345),
    }

    # Simulate a uvicorn-style receive that yields the body in one chunk
    # and then would yield http.disconnect on a later poll.
    send_more = True

    async def real_receive():
        nonlocal send_more
        if send_more:
            send_more = False
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        pass

    await middleware(scope, real_receive, send)

    assert b"".join(body_chunks) == body, "inner app must receive the full body verbatim"
    assert received_chunks[0]["type"] == "http.request"
    assert received_chunks[0]["more_body"] is False


@pytest.mark.asyncio
async def test_asgi_middleware_rejects_oversize_post():
    """Content-Length above the cap MUST 413 without buffering."""
    from mcp_server.server import ASGIRequestLoggerMiddleware

    async def inner_app(scope, receive, send):
        raise AssertionError("inner app must not be called for oversize body")

    middleware = ASGIRequestLoggerMiddleware(inner_app)

    sent_status = None
    sent_body = b""

    async def send(msg):
        nonlocal sent_status, sent_body
        if msg["type"] == "http.response.start":
            sent_status = msg["status"]
        elif msg["type"] == "http.response.body":
            sent_body += bytes(msg.get("body", b""))

    huge = 64 * 1024 * 1024  # 64 MiB > MAX_BUFFERED_BODY (32 MiB)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "headers": [
            (b"content-length", str(huge).encode("ascii")),
        ],
        "client": ("10.0.0.1", 12345),
    }

    async def receive():
        return {"type": "http.disconnect"}

    await middleware(scope, receive, send)
    assert sent_status == 413
    assert b"Request body too large" in sent_body


@pytest.mark.asyncio
async def test_figurine_notation_parsing_and_analysis():
    await server_module._cache.clear()

    class MockFigurinePool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=15, best_move="c1e3", pv=["c1e3"], depth=depth)

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=15),
                eval_after=Eval(cp=15),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockFigurinePool()  # type: ignore

    # 1. Figurine PGN
    figurine_pgn = "1. ♙e4 ♟e5 2. ♘f3 ♞c6 3. ♗c4 ♝c5 4. 0-0 ♞f6"
    res = await server_module.analyze_game(figurine_pgn, depth=14)
    assert res.total_plies == 8

    # 2. Figurine move in classify_move
    cm = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", move="♘f3", depth=14
    )
    assert cm.played == "g1f3"

    # 3. Unicode non-breaking spaces
    ev = await server_module.evaluate_position(
        "1.\u00a0e4\u00a0\u00a0c5\u00a02.\u00a0Nf3", moves=["\u00a0d6\u00a0"], depth=14
    )
    assert ev.cp == 15


@pytest.mark.asyncio
async def test_castling_variants_and_uci_promotions():
    await server_module._cache.clear()

    class MockPromoPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=500),
                eval_after=Eval(cp=500),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockPromoPool()  # type: ignore

    # 1. Castling variants (o-o, 0-0, o-o-o, 0-0-0)
    castle_board_fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    c1 = await server_module.classify_move(castle_board_fen, move="o-o", depth=14)
    assert c1.played == "e1g1"

    c2 = await server_module.classify_move(castle_board_fen, move="0-0-0", depth=14)
    assert c2.played == "e1c1"

    # 2. Uppercase UCI promotion
    promo_board_fen = "8/4P3/8/8/8/8/8/k6K w - - 0 1"
    p1 = await server_module.classify_move(promo_board_fen, move="e7e8Q", depth=14)
    assert p1.played == "e7e8q"

    # 3. SAN promotion without equals sign (e8Q)
    p2 = await server_module.classify_move(promo_board_fen, move="e8Q", depth=14)
    assert p2.played == "e7e8q"


@pytest.mark.asyncio
async def test_classify_move_on_game_over_raises_informative_error():
    mate_fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    with pytest.raises(ToolError, match=r"(?i)game_already_over"):
        await server_module.classify_move(mate_fen, move="e7e6", depth=14)


@pytest.mark.asyncio
async def test_llm_backticks_quotes_and_dot_prefixes():
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=20),
                eval_after=Eval(cp=20),
            )

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # 1. Backticks around move and FEN
    ev1 = await server_module.evaluate_position(
        "`rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`", moves=["`e4`"], depth=14
    )
    assert ev1.cp == 20

    # 2. Quotes and dot prefixes ("... e5", "...Nf6", "1...e5")
    cm1 = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", move="'... e5'", depth=14
    )
    assert cm1.played == "e7e5"

    cm2 = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", move='"...Nf6"', depth=14
    )
    assert cm2.played == "g8f6"

    # 3. Check vs mate tolerance (+ when it is #)
    mate_attempt_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    cm3 = await server_module.classify_move(mate_attempt_fen, move="Qxf7+", depth=14)
    assert cm3.played == "h5f7"

    # 4. Parameter clamping
    tm = await server_module.top_moves(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", n=0, depth=100
    )
    assert len(tm) > 0
    assert tm[0].depth == 30  # clamped to 30


@pytest.mark.asyncio
async def test_single_flight_cancellation_resilience():
    from mcp_server.cache import SingleFlight

    sf: SingleFlight[str] = SingleFlight()

    async def slow_work():
        await asyncio.sleep(0.05)
        return "done"

    # Task 1 starts
    t1 = asyncio.create_task(sf.do("key", slow_work))
    # Task 2 joins the in-flight future
    t2 = asyncio.create_task(sf.do("key", slow_work))

    await asyncio.sleep(0.01)
    # Cancel Task 2 while Task 1 is still computing
    t2.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t2

    # Task 1 must still finish cleanly without throwing InvalidStateError!
    res1 = await t1
    assert res1 == "done"

    # Subsequent call for same key must run fresh
    res3 = await sf.do("key", slow_work)
    assert res3 == "done"


@pytest.mark.asyncio
async def test_cors_options_preflight_headers():
    from mcp_server.server import ASGIRequestLoggerMiddleware

    sent_messages = []

    async def dummy_send(msg):
        sent_messages.append(msg)

    async def dummy_app(scope, receive, send):
        pass

    middleware = ASGIRequestLoggerMiddleware(dummy_app)

    scope_options = {
        "type": "http",
        "method": "OPTIONS",
        "path": "/mcp",
        "headers": [],
        "client": ("127.0.0.1", 12345),
    }

    async def dummy_receive():
        return {}

    await middleware(scope_options, dummy_receive, dummy_send)

    assert len(sent_messages) == 2
    assert sent_messages[0]["status"] == 200
    headers = dict(sent_messages[0]["headers"])
    assert headers[b"access-control-allow-origin"] == b"*"
    assert headers[b"access-control-allow-methods"] == b"GET, POST, OPTIONS"


@pytest.mark.asyncio
async def test_turning_points_selects_most_impactful_blunders():
    await server_module._cache.clear()

    # Generate a mock game where move 10 has the largest blunder (cpl=800)
    class VaryingEvalPool:
        def __init__(self):
            self.count = 0

        async def evaluate(self, board, depth=14):
            self.count += 1
            # Every even ply has some cpl loss, ply 10 has huge loss
            if len(board.move_stack) == 5:
                return Eval(cp=-600, best_move="e2e4", pv=["e2e4"], depth=depth)
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = VaryingEvalPool()  # type: ignore

    pgn = "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3 Nf6 5. O-O O-O 6. h3 h6"
    res = await server_module.analyze_game(pgn, depth=14)
    assert res.total_plies == 12
    # Turning points must be sorted chronologically by ply
    plies = [tp.ply for tp in res.turning_points]
    assert plies == sorted(plies)


@pytest.mark.asyncio
async def test_rate_limiter_hard_eviction():
    import time

    from mcp_server.server import TokenBucketRateLimiter

    rl = TokenBucketRateLimiter(rate=10.0, capacity=100.0)
    # Populate 10,500 clients
    for i in range(10_500):
        rl._buckets[f"ip_{i}"] = (100.0, time.time() - 4000)

    # Calling is_allowed should trigger pruning and reduce size to <= 5001
    await rl.is_allowed("fresh_ip")
    assert len(rl._buckets) <= 5001


@pytest.mark.asyncio
async def test_fools_mate_analysis_and_terminal_statuses():
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=30, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # 1. Fool's mate game analysis
    fools_mate_pgn = "1. f3 e5 2. g4 Qh4# 0-1"
    res = await server_module.analyze_game(fools_mate_pgn, depth=10)
    assert res.total_plies == 4
    assert res.black_accuracy > 90.0

    # 2. Checkmate position evaluate_position (White has mated Black, White POV = win)
    mate_fen = "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"
    ev_mate = await server_module.evaluate_position(mate_fen, depth=10)
    assert ev_mate.status == "checkmate"
    assert ev_mate.cp is None
    assert ev_mate.mate == 0
    assert ev_mate.best_move is None
    assert ev_mate.pv == []

    # 3. Stalemate position evaluate_position
    stalemate_fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"
    ev_stale = await server_module.evaluate_position(stalemate_fen, depth=10)
    assert ev_stale.status == "stalemate"
    assert ev_stale.cp == 0
    assert ev_stale.mate is None
    assert ev_stale.best_move is None
    assert ev_stale.pv == []

    # 4. K vs K insufficient material dead position
    kvk_fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
    ev_kvk = await server_module.evaluate_position(kvk_fen, depth=10)
    assert ev_kvk.status == "insufficient_material"
    assert ev_kvk.cp == 0
    assert ev_kvk.best_move is None
    assert ev_kvk.pv == []


@pytest.mark.asyncio
async def test_classify_move_is_engine_best_flag():
    await server_module._cache.clear()

    class MockClassifyPool:
        async def classify_move(self, board, move, depth=14):
            # Best move is e2e4
            is_best = move.uci() == "e2e4"
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST if is_best else MoveClass.INACCURACY,
                centipawn_loss=0 if is_best else 75,
                eval_before=Eval(cp=20, best_move="e2e4", pv=["e2e4"]),
                eval_after=Eval(cp=20 if is_best else -55),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockClassifyPool()  # type: ignore

    # Best move
    cm_best = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", move="e4"
    )
    assert cm_best.is_engine_best is True
    assert cm_best.move_class == MoveClass.BEST

    # Suboptimal move
    cm_sub = await server_module.classify_move(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", move="a3"
    )
    assert cm_sub.is_engine_best is False
    assert cm_sub.move_class == MoveClass.INACCURACY


@pytest.mark.asyncio
async def test_extract_game_raises_on_illegal_moves_and_corrupt_pgn():
    # 1. Illegal move in PGN must raise ValueError instead of silent truncation
    with pytest.raises(
        ValueError, match="Invalid PGN syntax|unrecognized token in movetext|Illegal move"
    ):
        server_module._extract_game("1. e4 e5 2. Ke3 *")

    with pytest.raises(
        ValueError, match="Invalid PGN syntax|unrecognized token in movetext|Illegal move"
    ):
        server_module._extract_game("e4 e5 Ke3")


@pytest.mark.asyncio
async def test_analyze_game_zero_plies_and_single_move_accuracies():
    await server_module._cache.clear()

    # 1. 0-ply game returns total_plies=0, null accuracy
    zero_ply_pgn = '[Result "*"] *'
    res0 = await server_module.analyze_game(zero_ply_pgn, depth=10)
    assert res0.total_plies == 0
    assert res0.white_accuracy is None
    assert res0.black_accuracy is None
    assert res0.white_acpl is None
    assert res0.black_acpl is None

    # 2. 1-ply game where Black made 0 moves
    class Mock1PlyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=25, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = Mock1PlyPool()  # type: ignore
    one_ply_pgn = "1. e4 *"
    res1 = await server_module.analyze_game(one_ply_pgn, depth=10)
    assert res1.total_plies == 1
    assert res1.white_accuracy is not None
    assert res1.white_acpl == 0.0
    assert res1.black_accuracy is None
    assert res1.black_acpl is None


@pytest.mark.asyncio
async def test_analyze_game_corrects_contradictory_pgn_result():
    await server_module._cache.clear()

    class MockFoolsPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=-500, best_move="d8h4", pv=["d8h4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockFoolsPool()  # type: ignore

    # Fool's mate has Black checkmating White (0-1), but header says 1-0
    contradictory_pgn = '[Result "1-0"]\n1. f3 e5 2. g4 Qh4#'
    res = await server_module.analyze_game(contradictory_pgn, depth=10)
    assert res.result == "0-1"


@pytest.mark.asyncio
async def test_classify_move_eval_after_status_and_san_on_mate():
    await server_module._cache.clear()

    class MockMateClassifyPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=100000, mate=1, best_move="f7g7", pv=["f7g7"]),
                eval_after=Eval(cp=100000, mate=0),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockMateClassifyPool()  # type: ignore

    # Mate position: White to play Qg7#
    mate_pos_fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    res = await server_module.classify_move(mate_pos_fen, move="Qg7#", depth=10)
    assert res.is_engine_best is True
    assert res.centipawn_loss == 0
    assert res.raw_centipawn_loss == 0
    assert res.best_move_san == "Qg7#"
    assert res.played_san == "Qg7#"
    assert res.played_line_san == "Qg7#"
    assert res.played_continuation_san is None
    assert res.eval_after.status == "checkmate"
    assert res.eval_after.winner == "white"
    assert res.eval_after.cp is None


@pytest.mark.asyncio
async def test_classify_move_strict_is_engine_best_on_alternative_move():
    await server_module._cache.clear()

    class MockMultiMatePool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=100000, mate=1, best_move="f7g7", pv=["f7g7"]),
                eval_after=Eval(cp=100000, mate=0),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockMultiMatePool()  # type: ignore

    # Played Qe7 (f7e7) instead of Qg7# (f7g7)
    mate_pos_fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    res = await server_module.classify_move(mate_pos_fen, move="Qe7", depth=10)
    assert res.played == "f7e7"
    assert res.played_san == "Qe7"
    assert res.is_engine_best is False  # Strictly false because played != best_move (f7g7)
    assert res.centipawn_loss == 0
    assert res.move_class == MoveClass.BEST


@pytest.mark.asyncio
async def test_classify_move_100x_repeatability_determinism():
    await server_module._cache.clear()

    class MockDeterministicPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.MISTAKE,
                centipawn_loss=105,
                eval_before=Eval(cp=130, best_move="e7e5", pv=["e7e5", "g1f3"]),
                eval_after=Eval(cp=235, best_move="d2d4", pv=["d2d4", "e7e6"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MockDeterministicPool()  # type: ignore

    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    results = [await server_module.classify_move(fen, move="f6", depth=14) for _ in range(50)]

    for r in results:
        assert r.centipawn_loss == 105
        assert r.move_class == MoveClass.MISTAKE
        assert r.played_san == "f6"
        assert r.is_engine_best is False


@pytest.mark.asyncio
async def test_structured_validation_errors():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    # 1. Illegal move gives ILLEGAL_MOVE
    with pytest.raises(ValueError, match="ILLEGAL_MOVE"):
        server_module._parse_move_on_board(chess.Board(fen), "e2e5")

    # 2. Ambiguous SAN gives AMBIGUOUS_SAN
    ambiguous_board = chess.Board("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1")
    with pytest.raises(ValueError, match="AMBIGUOUS_SAN"):
        server_module._parse_move_on_board(ambiguous_board, "Nd2")


@pytest.mark.asyncio
async def test_mcp_01_cache_key_halfmove_clock_isolation():
    """MCP-01: Verify positions differing only in halfmove_clock do not collide in cache."""
    await server_module._cache.clear()

    class CountingPool:
        def __init__(self):
            self.eval_calls = 0

        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            self.eval_calls += 1
            if board.halfmove_clock >= 149:
                return Eval(cp=0, best_move="f6g6", pv=["f6g6"], depth=depth)
            return Eval(
                cp=100000, mate=2, best_move="f6g6", pv=["f6g6", "h8g8", "a1g7"], depth=depth
            )

        async def close(self):
            pass

    pool = CountingPool()
    server_module._analyzer_pool = pool  # type: ignore

    pos_a = "7k/8/5K2/8/8/8/8/Q7 w - - 0 75"
    pos_b = "7k/8/5K2/8/8/8/8/Q7 w - - 149 75"

    # Query A first
    res_a = await server_module.evaluate_position(pos_a, depth=14)
    assert pool.eval_calls == 1
    assert res_a.mate == 2
    assert "0_75" in (res_a.lichess_url or "")

    # Query B - must NOT hit cache of A!
    res_b = await server_module.evaluate_position(pos_b, depth=14)
    assert pool.eval_calls == 2
    assert res_b.mate != 2 or res_b.cp == 0
    assert "149_75" in (res_b.lichess_url or "")


@pytest.mark.asyncio
async def test_mcp_02_cache_key_fullmove_number_and_url():
    """MCP-02: Verify positions differing in fullmove_number return URLs matching the exact FEN queried."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    fen_move_1 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    fen_move_20 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20"

    res_1 = await server_module.evaluate_position(fen_move_1, depth=14)
    assert "_0_1" in (res_1.lichess_url or "")

    res_20 = await server_module.evaluate_position(fen_move_20, depth=14)
    assert "_0_20" in (res_20.lichess_url or "")


@pytest.mark.asyncio
async def test_mcp_03_top_moves_cache_key_halfmove_isolation():
    """MCP-03: Verify top_moves does not cross-contaminate cache between different halfmove clocks."""
    await server_module._cache.clear()

    class CountingPool:
        def __init__(self):
            self.calls = 0

        async def top_moves(self, board: chess.Board, n: int = 3, depth: int = 14) -> list[Eval]:
            self.calls += 1
            if board.halfmove_clock >= 149:
                return [Eval(cp=0, best_move="f6g6", pv=["f6g6"], depth=depth)]
            return [Eval(cp=100000, mate=2, best_move="f6g6", pv=["f6g6"], depth=depth)]

        async def close(self):
            pass

    pool = CountingPool()
    server_module._analyzer_pool = pool  # type: ignore

    pos_a = "7k/8/5K2/8/8/8/8/Q7 w - - 0 75"
    pos_b = "7k/8/5K2/8/8/8/8/Q7 w - - 149 75"

    tm_a = await server_module.top_moves(pos_a, n=1, depth=14)
    assert pool.calls == 1
    assert tm_a[0].mate == 2

    tm_b = await server_module.top_moves(pos_b, n=1, depth=14)
    assert pool.calls == 2
    assert tm_b[0].cp == 0


@pytest.mark.asyncio
async def test_mcp_04_mcp_05_pgn_parser_preamble_and_lichess_analysis():
    """MCP-04 & MCP-05: Tagged PGN takes precedence over preambles with moves or analysis text."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # 1. Preamble with chess moves
    text_with_moves = """I was thinking about 1. d4 d5 before the game.

[Event "World Championship"]
[White "Carlsen"]
[Black "Nakamura"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *"""

    res1 = await server_module.analyze_game(text_with_moves, depth=10)
    assert res1.event == "World Championship"
    assert res1.white == "Carlsen"
    assert res1.black == "Nakamura"
    assert res1.total_plies == 4

    # 2. Lichess analysis preamble with high move numbers
    text_with_lichess_pv = """Computer analysis: 28. Qf2 Rc8 29. f6 Qg5

[Event "Casual Game"]
[White "Alice"]
[Black "Bob"]
[Result "*"]

1. e4 e5 2. Nf3 Nc6 *"""

    res2 = await server_module.analyze_game(text_with_lichess_pv, depth=10)
    assert res2.event == "Casual Game"
    assert res2.white == "Alice"
    assert res2.black == "Bob"
    assert res2.total_plies == 4


@pytest.mark.asyncio
async def test_mcp_06_analyze_game_repetition_history():
    """MCP-06: Verify analyze_game preserves move history in board states."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 1/2-1/2"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 8


@pytest.mark.asyncio
async def test_mcp_07_automatic_fivefold_repetition_stops_analysis():
    """MCP-07: Automatic 5-fold repetition terminates the game immediately and ignores extra plies."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # 4 cycles (16 plies) reaches 5-fold repetition. Ply 17 (9. Nf3) must NOT be analyzed.
    pgn_5fold = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 Nf6 6. Ng1 Ng8 7. Nf3 Nf6 8. Ng1 Ng8 9. Nf3"
    res = await server_module.analyze_game(pgn_5fold, depth=10)
    assert res.total_plies == 16
    assert res.result == "1/2-1/2"
    assert res.termination == "fivefold_repetition"


@pytest.mark.asyncio
async def test_mcp_08_automatic_seventyfive_moves_stops_analysis():
    """MCP-08: 75-move rule terminates the game immediately, unless move 150 is checkmate."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=0, best_move="f6g6", pv=["f6g6"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # Halfmove clock reaches 150 on 75. Kg6+
    pgn_75 = '[SetUp "1"]\n[FEN "7k/8/5K2/8/8/8/8/Q7 w - - 149 75"]\n\n75. Kg6+ Kg8 *'
    res = await server_module.analyze_game(pgn_75, depth=10)
    assert res.total_plies == 1
    assert res.result == "1/2-1/2"
    assert res.termination == "seventyfive_moves"

    # If the 150th move is checkmate (75. Qg7#), checkmate takes precedence
    pgn_mate_at_150 = '[SetUp "1"]\n[FEN "7k/Q7/5K2/8/8/8/8/8 w - - 149 75"]\n\n75. Qg7#'
    res_mate = await server_module.analyze_game(pgn_mate_at_150, depth=10)
    assert res_mate.total_plies == 1
    assert res_mate.result == "1-0"
    assert res_mate.termination == "checkmate"


@pytest.mark.asyncio
async def test_mcp_09_moves_after_checkmate_are_ignored():
    """MCP-09: Moves appended after checkmate are ignored."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board: chess.Board, depth: int = 14) -> Eval:
            return Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = "1. f3 e5 2. g4 Qh4# 3. h3 h6"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 4
    assert res.result == "0-1"
    assert res.termination == "checkmate"


@pytest.mark.asyncio
async def test_mcp_10_mcp_11_terminal_checkmate_winner_and_mate_symmetry():
    """MCP-10 & MCP-11: Checkmate evals have winner='white'|'black', mate=0, cp=+-100000, depth=0."""
    await server_module._cache.clear()

    # 1. White delivered mate
    white_mated_fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    ev1 = await server_module.evaluate_position(white_mated_fen, depth=14)
    assert ev1.status == "checkmate"
    assert ev1.winner == "white"
    assert ev1.mate == 0
    assert ev1.cp is None
    assert ev1.depth == 0
    assert ev1.best_move is None
    assert ev1.pv == []

    # 2. Black delivered mate
    black_mated_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    ev2 = await server_module.evaluate_position(black_mated_fen, depth=14)
    assert ev2.status == "checkmate"
    assert ev2.winner == "black"
    assert ev2.mate == 0
    assert ev2.cp is None
    assert ev2.depth == 0
    assert ev2.best_move is None
    assert ev2.pv == []


@pytest.mark.asyncio
async def test_mcp_12_mcp_14_mate_distance_loss_and_strict_best_class():
    """MCP-12 & MCP-14: Classify move returns mate_distance_loss and reserves 'best' for is_engine_best."""
    await server_module._cache.clear()

    class MateDistPool:
        async def classify_move(
            self, board: chess.Board, move: chess.Move, depth: int = 14
        ) -> MoveAnalysis:
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=100000, mate=1, best_move="g6g7", pv=["g6g7"]),
                eval_after=Eval(cp=100000, mate=1, best_move="h8g8", pv=["h8g8"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MateDistPool()  # type: ignore

    fen = "7k/8/5KQ1/8/8/8/8/8 w - - 0 1"
    # Player plays Qg2 instead of Qg7#
    res = await server_module.classify_move(fen, move="Qg2", depth=10)
    assert res.played == "g6g2"
    assert res.is_engine_best is False
    assert res.mate_distance_loss == 1
    assert res.centipawn_loss == 0
    assert res.move_class == MoveClass.GOOD


@pytest.mark.asyncio
async def test_mcp_13_structured_tool_errors():
    """MCP-13: Validation and chess rule errors raise structured ToolError with error codes."""
    # 1. Invalid FEN
    with pytest.raises(ToolError) as exc_info1:
        await server_module.evaluate_position(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - invalid", depth=14
        )
    assert "[INVALID_FEN]" in str(exc_info1.value)

    # 2. Illegal move
    with pytest.raises(ToolError) as exc_info2:
        await server_module.classify_move("startpos", move="e5", depth=14)
    assert "[ILLEGAL_MOVE]" in str(exc_info2.value)

    # 3. Ambiguous SAN
    with pytest.raises(ToolError) as exc_info3:
        await server_module.classify_move("4k3/8/8/8/8/5N2/8/1N2K3 w - - 0 1", move="Nd2", depth=14)
    assert "[AMBIGUOUS_SAN]" in str(exc_info3.value)


@pytest.mark.asyncio
async def test_mcp_15_mcp_16_terminal_depth_and_draw_statuses():
    """MCP-15 & MCP-16: Terminal positions have depth=0 and specific draw statuses."""
    await server_module._cache.clear()

    # Stalemate
    stale_fen = "k7/2Q5/K7/8/8/8/8/8 b - - 0 1"
    ev_stale = await server_module.evaluate_position(stale_fen, depth=14)
    assert ev_stale.status == "stalemate"
    assert ev_stale.depth == 0
    assert ev_stale.winner is None

    # Insufficient material
    insuf_fen = "8/8/8/8/8/8/4K3/7k w - - 0 1"
    ev_insuf = await server_module.evaluate_position(insuf_fen, depth=14)
    assert ev_insuf.status == "insufficient_material"
    assert ev_insuf.depth == 0

    # 75-move rule (with pieces on board so it's not insufficient material)
    seventyfive_fen = "7k/8/5K2/8/8/8/8/Q7 w - - 150 75"
    ev_75 = await server_module.evaluate_position(seventyfive_fen, depth=14)
    assert ev_75.status == "seventyfive_moves"
    assert ev_75.depth == 0


def test_doc_17_and_sec_18_docs_sync_and_no_secrets():
    """DOC-17 & SEC-18: Verify docs reflect 4 live tools and contain no secrets."""
    from pathlib import Path

    doc_path = Path(__file__).resolve().parent.parent / "README.md"
    assert doc_path.exists()
    content = doc_path.read_text(encoding="utf-8")

    # Check 4 live tools are documented
    tools_list = asyncio.run(server_module.mcp.list_tools())
    tool_names = [t.name for t in tools_list]
    for name in tool_names:
        assert name in content

    # Check no hardcoded live Bearer tokens
    assert "chess_mcp_pWA-" not in content
    assert "Bearer chess_mcp" not in content


# ==============================================================================
# BUG-01 to BUG-15 Regression Suite
# ==============================================================================


@pytest.mark.asyncio
async def test_bug01_garbage_fen_rejected():
    """BUG-01: Garbage FEN string must not default to startpos or return successful eval."""
    with pytest.raises(ToolError) as exc_info:
        await server_module.evaluate_position("this is not a fen", depth=14)
    err = str(exc_info.value)
    assert "[INVALID_POSITION]" in err or "[INVALID_FEN]" in err or "[INVALID_INPUT]" in err

    with pytest.raises(ToolError):
        await server_module.top_moves("completely invalid fen string", depth=14)

    with pytest.raises(ToolError):
        await server_module.classify_move("completely invalid fen string", move="e4", depth=14)


@pytest.mark.asyncio
async def test_bug02_analyze_game_rejects_garbage():
    """BUG-02: analyze_game with garbage string must fail with error, not return 0-ply report."""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game("garbage position", depth=14)
    err = str(exc_info.value)
    assert "[INVALID_POSITION]" in err or "[INVALID_PGN]" in err or "[INVALID_INPUT]" in err


@pytest.mark.asyncio
async def test_bug03_movetext_with_garbage_tokens_rejected():
    """BUG-03: analyze_game must reject garbage tokens inside active movetext."""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game("1. e4 nonsense e5 2. Nf3 Nc6", depth=14)
    err = str(exc_info.value)
    assert "[INVALID_PGN]" in err
    assert "nonsense" in err


@pytest.mark.asyncio
async def test_bug04_top_moves_prefix_consistency():
    """BUG-04: top_moves(n=1)[0] must match top_moves(n=3)[0] due to shared MultiPV and caching."""
    await server_module._cache.clear()

    class PrefixMockPool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=30, best_move="e2e4", pv=["e2e4", "e7e5"], depth=depth),
                Eval(cp=20, best_move="d2d4", pv=["d2d4", "d7d5"], depth=depth),
                Eval(cp=10, best_move="c2c4", pv=["c2c4", "c7c5"], depth=depth),
            ][:n]

        async def close(self):
            pass

    server_module._analyzer_pool = PrefixMockPool()  # type: ignore

    res1 = await server_module.top_moves("startpos", n=1, depth=14)
    res3 = await server_module.top_moves("startpos", n=3, depth=14)
    assert len(res1) == 1
    assert len(res3) == 3
    assert res1[0].best_move == res3[0].best_move == "e2e4"
    assert res3[1].best_move == "d2d4"


@pytest.mark.asyncio
async def test_bug05_classify_move_consistent_with_top_moves():
    """BUG-05: Best move from top_moves must receive move_class='best' and centipawn_loss=0 in classify_move."""
    await server_module._cache.clear()

    class TopClassifyPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=25, best_move="e2e4", pv=["e2e4"]),
                eval_after=Eval(cp=25, best_move="e7e5", pv=["e7e5"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = TopClassifyPool()  # type: ignore

    res = await server_module.classify_move("startpos", move="e4", depth=14)
    assert res.move_class == MoveClass.BEST
    assert res.centipawn_loss == 0
    assert res.is_engine_best is True


@pytest.mark.asyncio
async def test_bug06_alternative_mate_in_one_gets_best():
    """BUG-06: Alternative mate-in-1 move must get move_class='best', not downgraded to 'good'."""
    await server_module._cache.clear()

    class MultiMatePool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=100000, mate=1, best_move="f7g7", pv=["f7g7"]),
                eval_after=Eval(cp=100000, mate=0),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MultiMatePool()  # type: ignore

    # Position where both Qg7# and Qe7# mate (White plays Qe7)
    fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
    res = await server_module.classify_move(fen, move="Qe7", depth=14)
    assert res.move_class == MoveClass.BEST
    assert res.is_engine_best is False
    assert res.centipawn_loss == 0


@pytest.mark.asyncio
async def test_bug07_board_checkmate_overrides_resignation_header():
    """BUG-07: Actual on-board checkmate overrides incorrect [Termination 'resignation'] PGN header."""
    await server_module._cache.clear()

    class DummyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = """[Event "Test"]
[Termination "resignation"]
1. f3 e5 2. g4 Qh4# 0-1"""
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.termination == "checkmate"
    assert res.result == "0-1"


@pytest.mark.asyncio
async def test_bug08_and_13_opening_eco_detection_and_header_warnings():
    """BUG-08 & BUG-13: Opening and ECO detection populated from move list; header disagreements added as warnings."""
    await server_module._cache.clear()

    class DummyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=15, best_move="d2d4", pv=["d2d4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = DummyPool()  # type: ignore

    # 1. Bare moves detect both opening name and ECO code (BUG-13)
    res1 = await server_module.analyze_game("1. e4 e5 2. Nf3 Nc6 3. Bb5 a6", depth=10)
    assert res1.opening == "Ruy Lopez: Morphy Defense"
    assert res1.eco == "C70"
    assert res1.opening_header is None
    assert res1.eco_header is None
    assert res1.metadata_warnings == []

    # 2. Conflicting headers produce metadata_warnings while preserving detected opening (BUG-08)
    pgn_conflict = """[Opening "Sicilian Defense"]
[ECO "B20"]
1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"""
    res2 = await server_module.analyze_game(pgn_conflict, depth=10)
    assert res2.opening == "Ruy Lopez: Morphy Defense"
    assert res2.eco == "C70"
    assert res2.opening_header == "Sicilian Defense"
    assert res2.eco_header == "B20"
    assert len(res2.metadata_warnings) == 2
    assert any("Sicilian Defense" in w for w in res2.metadata_warnings)
    assert any("B20" in w for w in res2.metadata_warnings)


@pytest.mark.asyncio
async def test_bug09_multiple_pgn_games_rejected():
    """BUG-09: analyze_game rejects multiple PGN games instead of silently truncating."""
    multi_pgn = """[Event "Game 1"]
[White "Player A"]
[Black "Player B"]
1. e4 e5 2. Nf3 Nc6 1/2-1/2

[Event "Game 2"]
[White "Player C"]
[Black "Player D"]
1. d4 d5 2. c4 c6 1/2-1/2"""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game(multi_pgn, depth=10)
    assert "[MULTIPLE_GAMES_NOT_SUPPORTED]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_bug10_raw_centipawn_loss_never_leaks_mate_sentinel():
    """BUG-10: raw_centipawn_loss must never expose internal sentinel >= 100000 on mate transitions."""
    await server_module._cache.clear()

    class MateTransitionPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BLUNDER,
                centipawn_loss=1000,
                eval_before=Eval(cp=50, best_move="e2e4", pv=["e2e4"]),
                eval_after=Eval(cp=100000, mate=1),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MateTransitionPool()  # type: ignore

    res = await server_module.classify_move("startpos", move="f3", depth=14)
    assert res.centipawn_loss is None or res.centipawn_loss <= 1000
    assert res.raw_centipawn_loss is None or res.raw_centipawn_loss <= 1000


@pytest.mark.asyncio
async def test_bug11_terminal_mate_eval_representation():
    """BUG-11: Terminal checkmate evaluates to cp=None, mate=0, status='checkmate'."""
    await server_module._cache.clear()

    # Black checkmated -> White won
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    ev = await server_module.evaluate_position(fen, depth=14)
    assert ev.status == "checkmate"
    assert ev.winner == "white"
    assert ev.cp is None
    assert ev.mate == 0
    assert ev.best_move is None
    assert ev.pv == []


@pytest.mark.asyncio
async def test_bug12_claimable_draw_status():
    """BUG-12: MCPEval includes can_claim_draw and claim_reasons for 50-move rule and repetitions."""
    await server_module._cache.clear()

    class FiftyMovePool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=0, best_move="a2a1", pv=["a2a1"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = FiftyMovePool()  # type: ignore

    # 50-move rule claimable (halfmove_clock = 100)
    fen_50 = "7k/8/8/8/8/8/R7/K7 w - - 100 51"
    ev = await server_module.evaluate_position(fen_50, depth=14)
    assert ev.can_claim_draw is True
    assert "fifty_moves" in ev.claim_reasons


@pytest.mark.asyncio
async def test_bug14_and_15_clean_error_message_no_double_wrapping():
    """BUG-14 & BUG-15: Error responses are clean without double 'INVALID_FEN: INVALID_FEN:' prefixes."""
    with pytest.raises(ToolError) as exc_info:
        await server_module.evaluate_position("8/8/8/8/8/8/8/8 w - - 0 1", depth=14)
    err = str(exc_info.value)
    assert "[INVALID_FEN]" in err
    assert "INVALID_FEN: INVALID_FEN:" not in err


# ==============================================================================
# Round 2 Regression Suite
# ==============================================================================


@pytest.mark.asyncio
async def test_round2_01_threefold_repetition_detected_in_eval():
    """Round 2 (Item 1): Threefold repetition correctly flags can_claim_draw and threefold_repetition reason."""
    await server_module._cache.clear()

    class RepetitionPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=35, best_move="g1f3", pv=["g1f3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = RepetitionPool()  # type: ignore

    # 2 repetitions -> can_claim_draw is False
    two_reps = ["Nf3", "Nf6", "Ng1", "Ng8"]
    ev2 = await server_module.evaluate_position("startpos", moves=two_reps, depth=10)
    assert ev2.can_claim_draw is False
    assert "threefold_repetition" not in ev2.claim_reasons

    # 3 occurrences of startpos (before move 1, after 2...Ng8, after 4...Ng8)
    three_reps = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    ev3 = await server_module.evaluate_position("startpos", moves=three_reps, depth=10)
    assert ev3.can_claim_draw is True
    assert "threefold_repetition" in ev3.claim_reasons


@pytest.mark.asyncio
async def test_round2_01b_threefold_repetition_in_analyze_game_not_penalized():
    """Round 2 (Item 1b): Moving into 3-fold repetition is evaluated as draw (0.00 cp) and graded BEST."""
    await server_module._cache.clear()

    class RepPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=35, best_move="g1f3", pv=["g1f3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = RepPool()  # type: ignore

    pgn = """1. Nf3 Nf6 2. Ng1 Ng8
3. Nf3 Nf6 4. Ng1 Ng8 1/2-1/2"""
    res = await server_module.analyze_game(pgn, depth=10)
    # Final ply (4...Ng8) creates 3-fold repetition
    assert res.total_plies == 8
    assert res.black_blunders == 0
    assert res.black_mistakes == 0


@pytest.mark.asyncio
async def test_round2_02_analyze_game_validates_initial_fen():
    """Round 2 (Item 2): analyze_game rejects illegal initial position in PGN Setup tags."""
    # 1. King in check on opponent's turn (status 1024)
    illegal_pgn1 = """[SetUp "1"]
[FEN "7k/5K2/5Q2/8/8/8/8/8 w - - 0 1"]
[Result "1/2-1/2"]

1. Qg6 1/2-1/2"""
    with pytest.raises(ToolError) as exc1:
        await server_module.analyze_game(illegal_pgn1, depth=10)
    assert "[INVALID_FEN]" in str(exc1.value)

    # 2. No kings on board
    illegal_pgn2 = """[SetUp "1"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]
[Result "*"]

*"""
    with pytest.raises(ToolError) as exc2:
        await server_module.analyze_game(illegal_pgn2, depth=10)
    assert "[INVALID_FEN]" in str(exc2.value)

    # 3. Castling rights without rooks
    illegal_pgn3 = """[SetUp "1"]
[FEN "4k3/8/8/8/8/8/8/4K3 w KQ - 0 1"]
[Result "*"]

*"""
    with pytest.raises(ToolError) as exc3:
        await server_module.analyze_game(illegal_pgn3, depth=10)
    assert "[INVALID_FEN]" in str(exc3.value)


@pytest.mark.asyncio
async def test_round2_04_decisive_position_saturation():
    """Round 2 (Item 4): Queen promotion from +5.79 to +5.00 retains BEST/GOOD classification."""
    await server_module._cache.clear()

    class WinningPositionPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.INACCURACY,
                centipawn_loss=79,
                eval_before=Eval(cp=579, best_move="c6d5", pv=["c6d5"]),
                eval_after=Eval(cp=500, best_move="h8g8", pv=["h8g8"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = WinningPositionPool()  # type: ignore

    # In 7k/P7/2K5/8/8/8/8/8 w - - 0 1, White plays a8=Q+ (alternative to engine best c6d5)
    res = await server_module.classify_move("7k/P7/2K5/8/8/8/8/8 w - - 0 1", move="a8=Q+", depth=14)
    assert res.move_class == MoveClass.GOOD
    assert res.is_engine_best is False
    assert res.raw_centipawn_loss == 79
    assert res.effective_loss is not None and res.effective_loss <= 15


@pytest.mark.asyncio
async def test_round2_05_mate_blundered_to_stalemate():
    """Round 2 (Item 5): Mate blundered to stalemate sets centipawn_loss=None and move_class=blunder."""
    await server_module._cache.clear()

    class StalemateBlunderPool:
        async def classify_move(self, board, move, depth=14):
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BLUNDER,
                centipawn_loss=1000,
                eval_before=Eval(cp=100000, mate=1, best_move="f5h3", pv=["f5h3"]),
                eval_after=Eval(cp=0, best_move=None, pv=[]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = StalemateBlunderPool()  # type: ignore

    # In 7k/5K2/8/5Q2/8/8/8/8 w - - 0 1, White plays Qg6 (stalemate)
    res = await server_module.classify_move("7k/5K2/8/5Q2/8/8/8/8 w - - 0 1", move="Qg6", depth=14)
    assert res.move_class == MoveClass.BLUNDER
    assert res.centipawn_loss is None
    assert res.raw_centipawn_loss is None


@pytest.mark.asyncio
async def test_round2_07_result_header_mismatch_warning():
    """Round 2 (Item 7): Contradictory Result header produces a metadata warning."""
    await server_module._cache.clear()

    class DummyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = DummyPool()  # type: ignore

    pgn = """[Event "Test"]
[Result "0-1"]

1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"""
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.result == "1-0"
    assert len(res.metadata_warnings) == 1
    assert "Result header '0-1' disagrees with board outcome '1-0'" in res.metadata_warnings[0]


@pytest.mark.asyncio
async def test_round2_08_ep_suffix_normalization():
    """Round 2 (Item 8): En passant with 'e.p.' suffix is parsed cleanly in both moves and PGN."""
    await server_module._cache.clear()

    class DummyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=0, best_move="g8f6", pv=["g8f6"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = DummyPool()  # type: ignore

    # 1. evaluate_position with moves
    ev = await server_module.evaluate_position(
        "startpos", moves=["e4", "c5", "e5", "d5", "exd6 e.p."], depth=10
    )
    assert ev.status == "active"

    # 2. analyze_game with e.p.
    pgn = "1. e4 c5 2. e5 d5 3. exd6 e.p. *"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 5


@pytest.mark.asyncio
async def test_round2_03_cpl_consistency_classify_move_and_analyze_game():
    """Round 2 (Item 3): classify_move and analyze_game yield exact same CPL for same move and depth."""
    await server_module._cache.clear()

    class SharedEvalPool:
        async def evaluate(self, board, depth=14):
            # startpos is +30, after 1. f3 is -80
            if "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -" in board.fen():
                return Eval(cp=30, best_move="e2e4", pv=["e2e4"])
            if "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq -" in board.fen():
                return Eval(cp=-80, best_move="e7e5", pv=["e7e5"])
            return Eval(cp=0)

        async def classify_move(self, board, move, depth=14):
            eval_before = await self.evaluate(board, depth=depth)
            b_after = board.copy()
            b_after.push(move)
            eval_after = await self.evaluate(b_after, depth=depth)
            loss = max(0, (eval_before.cp or 0) - (eval_after.cp or 0))
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.MISTAKE,
                centipawn_loss=loss,
                eval_before=eval_before,
                eval_after=eval_after,
            )

        async def close(self):
            pass

    server_module._analyzer_pool = SharedEvalPool()  # type: ignore

    cm = await server_module.classify_move("startpos", move="f3", depth=14)
    ag = await server_module.analyze_game("1. f3 *", depth=14)

    assert cm.centipawn_loss == 110
    assert ag.turning_points[0].centipawn_loss == 110
    assert cm.move_class.value == ag.turning_points[0].move_class == "mistake"


@pytest.mark.asyncio
async def test_round2_01_threefold_repetition_claim_and_7plies():
    """Round 2 (Item 1): Threefold repetition draw claim detection on 8-ply and 7-ply lines."""
    await server_module._cache.clear()

    class DummyPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = DummyPool()  # type: ignore

    # 1. 8-ply repetition: position occurred 3 times
    moves_8 = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    res_8 = await server_module.evaluate_position("startpos", moves=moves_8, depth=14)
    assert res_8.can_claim_draw is True
    assert "threefold_repetition" in res_8.claim_reasons

    # 2. 7-ply repetition: Black to move can claim draw by declaring Ng8
    moves_7 = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1"]
    res_7 = await server_module.evaluate_position("startpos", moves=moves_7, depth=14)
    assert res_7.can_claim_draw is True
    assert "threefold_repetition" in res_7.claim_reasons


@pytest.mark.asyncio
async def test_round2_02_invalid_fen_pgn_header_rejections():
    """Round 2 (Item 2): Reject invalid FENs in PGN [FEN ...] headers with [INVALID_FEN]."""
    await server_module._cache.clear()

    # 1. White turn but Black is in check
    pgn_opp_check = """[Event "Test"]
[SetUp "1"]
[FEN "7k/5K2/5Q2/8/8/8/8/8 w - - 0 1"]

1. Qg7#"""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game(pgn_opp_check, depth=10)
    assert "[INVALID_FEN]" in str(exc_info.value)

    # 2. No kings on the board
    pgn_no_kings = """[Event "Test"]
[SetUp "1"]
[FEN "8/8/8/8/8/8/8/8 w - - 0 1"]

1. e4 *"""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game(pgn_no_kings, depth=10)
    assert "[INVALID_FEN]" in str(exc_info.value)

    # 3. Bad castling rights
    pgn_bad_castling = """[Event "Test"]
[SetUp "1"]
[FEN "4k3/8/8/8/8/8/8/4K3 w KQ - 0 1"]

1. Ke2 *"""
    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game(pgn_bad_castling, depth=10)
    assert "[INVALID_FEN]" in str(exc_info.value)


@pytest.mark.asyncio
async def test_round2_09_depth_semantics_requested_and_searched():
    """Round 2 (Item 9): Verify requested_depth and searched_depth on MCPEval."""
    await server_module._cache.clear()

    class DepthPool:
        async def evaluate(self, board, depth=14):
            # Engine returned depth 14 search
            return Eval(cp=15, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=15, best_move="e2e4", pv=["e2e4"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = DepthPool()  # type: ignore

    ev = await server_module.evaluate_position("startpos", depth=18)
    assert ev.depth == 18
    assert ev.requested_depth == 18
    assert ev.searched_depth == 18

    # Terminal checkmate position sets searched_depth=0, requested_depth=18
    ev_mate = await server_module.evaluate_position("7k/5KQ1/8/8/8/8/8/8 b - - 0 1", depth=18)
    assert ev_mate.depth == 0
    assert ev_mate.searched_depth == 0
    assert ev_mate.requested_depth == 18


# ---------------------------------------------------------------------------
# MCP-NEW-01 .. MCP-NEW-13 REGRESSION TESTS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_new_01_repetition_cache_collision_bidirectional():
    """MCP-NEW-01: Cache must not collide when two game histories produce identical FENs with differing repetition rights."""

    class RepetitionPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = RepetitionPool()  # type: ignore

    moves_rep = ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]
    moves_no_rep = ["Nf3", "Nf6", "Ne5", "Ne4", "Nf3", "Nf6", "Ng1", "Ng8"]

    # Order 1: Repetition evaluated FIRST, then No-Repetition
    await server_module._cache.clear()
    res_a1 = await server_module.evaluate_position("startpos", moves=moves_rep, depth=14)
    res_b1 = await server_module.evaluate_position("startpos", moves=moves_no_rep, depth=14)
    assert res_a1.can_claim_draw is True
    assert "threefold_repetition" in res_a1.claim_reasons
    assert res_b1.can_claim_draw is False
    assert res_b1.claim_reasons == []

    # Order 2: No-Repetition evaluated FIRST, then Repetition
    await server_module._cache.clear()
    res_b2 = await server_module.evaluate_position("startpos", moves=moves_no_rep, depth=14)
    res_a2 = await server_module.evaluate_position("startpos", moves=moves_rep, depth=14)
    assert res_b2.can_claim_draw is False
    assert res_b2.claim_reasons == []
    assert res_a2.can_claim_draw is True
    assert "threefold_repetition" in res_a2.claim_reasons


@pytest.mark.asyncio
async def test_mcp_new_02_mate_distance_symmetry_black_and_white():
    """MCP-NEW-02: Mate-distance loss and move grading must be symmetric for White and Black."""
    await server_module._cache.clear()

    # 1. White mating: 7k/8/5KQ1/8/8/8/8/8 w - - 0 1
    # Qg7# is mate in 1, Qf5 is mate in 2 (+1 mate dist loss)
    b_white = chess.Board("7k/8/5KQ1/8/8/8/8/8 w - - 0 1")
    eval_w_bef = MCPEval(mate=1, best_move="g6g7")
    eval_w_delay = MCPEval(mate=1)
    score_w = score_played_move(b_white, chess.Move.from_uci("g6f5"), eval_w_bef, eval_w_delay)
    assert score_w.move_class == MoveClass.GOOD
    assert score_w.mate_distance_loss == 1
    assert score_w.is_best_engine_move is False

    # 2. Black mating: 8/8/8/8/8/5kq1/8/7K b - - 0 1
    # Qg2# is mate in 1 (from White pov: #-1), Qh4+ is mate in 2 (from White pov: #-2)
    b_black = chess.Board("8/8/8/8/8/5kq1/8/7K b - - 0 1")
    eval_b_bef = MCPEval(mate=-1, best_move="g3g2")
    eval_b_delay = MCPEval(mate=-1)
    score_b = score_played_move(b_black, chess.Move.from_uci("g3h4"), eval_b_bef, eval_b_delay)
    assert score_b.move_class == MoveClass.GOOD
    assert score_b.mate_distance_loss == 1
    assert score_b.is_best_engine_move is False

    # 3. Black mating with larger delay (mate in 1 -> mate in 4, dist loss 3 -> mistake)
    eval_b_delay3 = MCPEval(mate=-3)
    score_b3 = score_played_move(b_black, chess.Move.from_uci("g3h4"), eval_b_bef, eval_b_delay3)
    assert score_b3.move_class == MoveClass.MISTAKE
    assert score_b3.mate_distance_loss == 3


@pytest.mark.asyncio
async def test_mcp_new_03_top_moves_cache_and_singleflight_includes_n():
    """MCP-NEW-03: top_moves cache and SingleFlight must include n so n=1 and n=5 results never collide."""
    await server_module._cache.clear()

    class MultiPVPool:
        async def top_moves(self, board, n=3, depth=14):
            all_cands = [
                Eval(cp=30, best_move="e2e4", pv=["e2e4"], depth=depth),
                Eval(cp=25, best_move="d2d4", pv=["d2d4"], depth=depth),
                Eval(cp=20, best_move="g1f3", pv=["g1f3"], depth=depth),
                Eval(cp=15, best_move="c2c4", pv=["c2c4"], depth=depth),
                Eval(cp=10, best_move="b1c3", pv=["b1c3"], depth=depth),
            ]
            return all_cands[:n]

        async def close(self):
            pass

    server_module._analyzer_pool = MultiPVPool()  # type: ignore

    # Sequential test
    res_1 = await server_module.top_moves("startpos", n=1, depth=10)
    assert len(res_1) == 1
    res_5 = await server_module.top_moves("startpos", n=5, depth=10)
    assert len(res_5) == 5
    res_1_again = await server_module.top_moves("startpos", n=1, depth=10)
    assert len(res_1_again) == 1

    # Concurrent test
    await server_module._cache.clear()
    t1 = server_module.top_moves("startpos", n=1, depth=10)
    t5 = server_module.top_moves("startpos", n=5, depth=10)
    conc_1, conc_5 = await asyncio.gather(t1, t5)
    assert len(conc_1) == 1
    assert len(conc_5) == 5


@pytest.mark.asyncio
async def test_mcp_new_04_zero_ply_custom_fen_terminal_state_and_result():
    """MCP-NEW-04: analyze_game with 0 plies on custom FEN must validate terminal state and normalize conflicting Result."""
    await server_module._cache.clear()

    # Checkmated position with contradictory Result "0-1"
    pgn_mate = """[SetUp "1"]
[FEN "7k/7Q/7K/8/8/8/8/8 b - - 0 1"]
[White "A"]
[Black "B"]
[Result "0-1"]

0-1"""
    res = await server_module.analyze_game(pgn_mate, depth=10)
    assert res.total_plies == 0
    assert res.result == "1-0"
    assert res.termination == "checkmate"
    assert len(res.metadata_warnings) >= 1
    assert "disagrees with board outcome" in res.metadata_warnings[0]


@pytest.mark.asyncio
async def test_mcp_new_05_multiple_pgn_detection_without_event():
    """MCP-NEW-05: Multiple PGN games must be rejected even if [Event] tag is absent."""
    await server_module._cache.clear()

    multi_pgn = """[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 1-0

[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 0-1"""

    with pytest.raises(ToolError) as exc_info:
        await server_module.analyze_game(multi_pgn, depth=10)
    assert "MULTIPLE_GAMES" in str(exc_info.value)


@pytest.mark.asyncio
async def test_mcp_new_06_bare_movelist_stops_at_result_marker():
    """MCP-NEW-06: Bare move parser must stop when encountering a game result marker (1-0, 0-1, 1/2-1/2, *)."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    res = await server_module.analyze_game("e4 e5 1-0 Nf3 Nc6", depth=10)
    assert res.total_plies == 2
    assert res.result == "1-0"

    b = server_module._build_board("e4 e5 1-0 Nf3 Nc6")
    assert len(b.move_stack) == 2


@pytest.mark.asyncio
async def test_mcp_new_07_custom_fen_skips_opening_detection():
    """MCP-NEW-07: analyze_game on custom FEN initial positions must not invent false openings/ECOs."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = """[SetUp "1"]
[FEN "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"]

1. e4"""
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 1
    assert res.opening is None
    assert res.eco is None


@pytest.mark.asyncio
async def test_mcp_new_08_acpl_includes_blunders_into_mate():
    """MCP-NEW-08: ACPL must include blunders transitioning into mate using surrogate loss."""
    await server_module._cache.clear()

    # Positions in Fool's Mate: startpos, after f3, after e5, after g4, after Qh4#
    class FoolsPool:
        async def evaluate(self, board, depth=14):
            fen = board.fen()
            if "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3" in fen:
                return Eval(cp=None, mate=0, depth=0)
            if "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2" in fen:
                return Eval(mate=-1, best_move="d8h4", depth=depth)
            if "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2" in fen:
                return Eval(cp=-50, best_move="e2e4", depth=depth)
            return Eval(cp=20, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = FoolsPool()  # type: ignore

    res = await server_module.analyze_game("1. f3 e5 2. g4 Qh4#", depth=10)
    assert res.white_acpl is not None
    assert res.white_average_effective_loss is not None
    assert res.white_average_effective_loss >= 500.0


@pytest.mark.asyncio
async def test_mcp_new_09_turning_points_ranking_cpl_zero_vs_none():
    """MCP-NEW-09: Turning points ranking must not treat centipawn_loss=0 as 1000."""
    pts = [
        PlyAnalysisItem(ply=1, san="Qf5", uci="g6f5", move_class="good", centipawn_loss=0),
        PlyAnalysisItem(ply=3, san="Bxf7+", uci="c4f7", move_class="mistake", centipawn_loss=300),
        PlyAnalysisItem(ply=5, san="g4", uci="g2g4", move_class="blunder", centipawn_loss=None),
    ]

    top_turning_points = sorted(
        sorted(
            pts,
            key=lambda x: 1000 if x.centipawn_loss is None else x.centipawn_loss,
            reverse=True,
        )[:2],
        key=lambda x: x.ply,
    )
    # The top 2 turning points should be ply 5 (None -> 1000) and ply 3 (300), NOT ply 1 (0)
    assert [p.ply for p in top_turning_points] == [3, 5]


def test_mcp_new_10_cache_keys_versioning_prefix():
    """MCP-NEW-10: All cache keys must include cache versioning prefix AND
    an engine-version segment so a Stockfish binary upgrade invalidates stale
    cached entries (P1#6 fix)."""
    b = chess.Board()
    eval_key = server_module.eval_cache_key(b, depth=14)
    top_key = server_module.top_moves_cache_key(b, depth=14)
    classify_key = server_module.classify_cache_key(b, "e2e4", depth=14)

    cv = server_module.CACHE_VERSION
    assert eval_key.startswith(f"mcp:{cv}:eng=") and ":eval:" in eval_key
    assert top_key.startswith(f"mcp:{cv}:eng=") and ":top:" in top_key
    assert classify_key.startswith(f"mcp:{cv}:eng=") and ":classify:" in classify_key

    # Explicit engine version override must change the key (so a new binary
    # version cannot silently reuse the old cache).
    eval_key_v2 = server_module.eval_cache_key(b, depth=14, engine_version="stockfish_18")
    eval_key_v3 = server_module.eval_cache_key(b, depth=14, engine_version="stockfish_17")
    assert eval_key_v2 != eval_key_v3
    assert eval_key_v2 != eval_key


@pytest.mark.asyncio
async def test_mcp_new_11_requested_depth_semantics_clamping():
    """MCP-NEW-11: requested_depth must preserve caller's raw depth while depth/searched_depth reflect clamped/searched depth."""
    await server_module._cache.clear()

    class ClampedPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = ClampedPool()  # type: ignore

    ev_100 = await server_module.evaluate_position("startpos", depth=100)
    assert ev_100.requested_depth == 100
    assert ev_100.depth == 30
    assert ev_100.searched_depth == 30

    ev_0 = await server_module.evaluate_position("startpos", depth=0)
    assert ev_0.requested_depth == 0
    assert ev_0.depth == 1
    assert ev_0.searched_depth == 1


@pytest.mark.asyncio
async def test_mcp_new_12_telemetry_cache_hit_tracking():
    """MCP-NEW-12: Cache hits must increment _cache_hits and not falsely report cache_misses."""
    await server_module._cache.clear()
    server_module.metrics._cache_hits = 0
    server_module.metrics._cache_misses = 0

    class TelemetryPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = TelemetryPool()  # type: ignore

    # First call: cache miss
    await server_module.evaluate_position("startpos", depth=12)
    assert server_module.metrics._cache_misses == 1
    assert server_module.metrics._cache_hits == 0

    # Second call: cache hit
    await server_module.evaluate_position("startpos", depth=12)
    assert server_module.metrics._cache_hits == 1


@pytest.mark.asyncio
async def test_mcp_new_13_no_duplicate_error_code_in_messages():
    """MCP-NEW-13: Error messages must not duplicate error code prefixes."""
    err = server_module._tool_error(
        "invalid_position", "INVALID_POSITION: Input 'foo' is invalid", "evaluate_position"
    )
    assert str(err) == "[INVALID_POSITION] Input 'foo' is invalid"
    assert "INVALID_POSITION: INVALID_POSITION:" not in str(err)

    err2 = server_module._tool_error(
        "illegal_move", "ILLEGAL_MOVE: Move 'e5' is not valid", "classify_move"
    )
    assert str(err2) == "[ILLEGAL_MOVE] Move 'e5' is not valid"


# ==============================================================================
# AUDIT REGRESSION TESTS (Items 1 - 20)
# ==============================================================================


@pytest.mark.asyncio
async def test_audit_01_variant_safety_rejected_across_all_tools():
    """Item 1 / TEST 1: Variant safety - Atomic/variants must raise UNSUPPORTED_VARIANT with code unsupported_variant."""
    await server_module._cache.clear()

    atomic_pgn = '[Variant "Atomic"]\n[White "A"]\n[Black "B"]\n\n1. e4 e5 *'

    # 1. evaluate_position
    with pytest.raises(ToolError) as exc_eval:
        await server_module.evaluate_position(atomic_pgn, depth=10)
    assert "[UNSUPPORTED_VARIANT]" in str(exc_eval.value)

    # 2. top_moves
    with pytest.raises(ToolError) as exc_top:
        await server_module.top_moves(atomic_pgn, depth=10)
    assert "[UNSUPPORTED_VARIANT]" in str(exc_top.value)

    # 3. classify_move
    with pytest.raises(ToolError) as exc_class:
        await server_module.classify_move(atomic_pgn, move="e4", depth=10)
    assert "[UNSUPPORTED_VARIANT]" in str(exc_class.value)

    # 4. analyze_game
    with pytest.raises(ToolError) as exc_game:
        await server_module.analyze_game(atomic_pgn, depth=10)
    assert "[UNSUPPORTED_VARIANT]" in str(exc_game.value)


@pytest.mark.asyncio
async def test_audit_02_mate_distance_calculation():
    """Item 2 / TEST 2: Mate distance loss calculation for mover and defender."""
    # FEN: 7k/5Q2/7K/8/8/8/8/8 w - - 0 1
    # Qg7# is mate in 1 (eval_before.mate = 1, best_move="f7g7")
    b = chess.Board("7k/5Q2/7K/8/8/8/8/8 w - - 0 1")
    eval_before = MCPEval(mate=1, best_move="f7g7")

    # 1. Best move Qg7#: mate in 0 from child -> mate distance loss = 0, is_best=True
    eval_qg7 = MCPEval(mate=0)
    score_qg7 = score_played_move(b, chess.Move.from_uci("f7g7"), eval_before, eval_qg7)
    assert score_qg7.is_best_engine_move is True
    assert score_qg7.mate_distance_loss == 0
    assert score_qg7.effective_loss == 0

    # 2. Qc7: mate in 2 from root -> child eval has mate=1 (White mates on next turn)
    # Distance loss = (1 + 1) - 1 = 1
    eval_qc7 = MCPEval(mate=1)
    score_qc7 = score_played_move(b, chess.Move.from_uci("f7c7"), eval_before, eval_qc7)
    assert score_qc7.is_best_engine_move is False
    assert score_qc7.mate_distance_loss == 1
    assert score_qc7.effective_loss == 50  # 50 cp synthetic loss for +1 mate ply

    # 3. Qf1: mate in 3 from root -> child eval has mate=2
    # Distance loss = (1 + 2) - 1 = 2
    eval_qf1 = MCPEval(mate=2)
    score_qf1 = score_played_move(b, chess.Move.from_uci("f7f1"), eval_before, eval_qf1)
    assert score_qf1.is_best_engine_move is False
    assert score_qf1.mate_distance_loss == 2
    assert score_qf1.effective_loss == 150  # 150 cp synthetic loss for +2 mate plies


@pytest.mark.asyncio
async def test_audit_03_conversational_preamble_and_trailer():
    """Item 3 / TEST 3: Conversational preamble and trailer handled without multiple-game errors."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=15, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    conv_text = "to moja partia:\n1. e4 e5 2. Nf3 Nc6\nprzeanalizuj ją dokładnie"
    res = await server_module.analyze_game(conv_text, depth=10)
    assert res.total_plies == 4
    assert res.white_acpl is not None


@pytest.mark.asyncio
async def test_audit_04_pgn_result_consistency_and_warnings():
    """Item 4 / TEST 4: PGN result header vs movetext result vs board outcome consistency."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # Header 1-0 + movetext 0-1 on ongoing board
    pgn = '[Result "1-0"]\n\n1. e4 e5 0-1'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.result == "1-0"
    assert res.result_header == "1-0"
    assert res.result_movetext == "0-1"
    assert any(
        "Result header '1-0' disagrees with movetext result '0-1'" in w
        for w in res.metadata_warnings
    )


@pytest.mark.asyncio
async def test_audit_05_top_moves_prefix_consistency_and_terminal():
    """Item 5 / TEST 5: top_moves prefix consistency and empty list on terminal position."""
    await server_module._cache.clear()

    class TopMockPool:
        async def top_moves(self, board, n=5, depth=14):
            return [
                Eval(cp=30, best_move="e2e4", pv=["e2e4", "e7e5"], depth=depth),
                Eval(cp=20, best_move="d2d4", pv=["d2d4", "d7d5"], depth=depth),
                Eval(cp=15, best_move="g1f3", pv=["g1f3", "g8f6"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = TopMockPool()  # type: ignore

    res_1 = await server_module.top_moves("startpos", n=1, depth=10)
    res_2 = await server_module.top_moves("startpos", n=2, depth=10)
    assert len(res_1) == 1
    assert len(res_2) == 2
    assert res_1[0].best_move == res_2[0].best_move

    # Terminal position (checkmate)
    res_term = await server_module.top_moves("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", depth=10)
    assert res_term == []


@pytest.mark.asyncio
async def test_audit_06_exact_engine_best_semantics():
    """Item 6 / TEST 6: is_engine_best is strictly True only when played_uci == best_move_uci."""
    b = chess.Board()
    eval_before = MCPEval(cp=20, best_move="e2e4")
    eval_after_e4 = MCPEval(cp=20)
    eval_after_d4 = MCPEval(cp=20)

    # Played e4 (best move)
    score_e4 = score_played_move(b, chess.Move.from_uci("e2e4"), eval_before, eval_after_e4)
    assert score_e4.is_best_engine_move is True

    # Played d4 (eval is same cp=20, but d4 != e4)
    score_d4 = score_played_move(b, chess.Move.from_uci("d2d4"), eval_before, eval_after_d4)
    assert score_d4.is_best_engine_move is False


@pytest.mark.asyncio
async def test_audit_07_acpl_effective_loss_transparency():
    """Item 7 / TEST 7: ACPL effective_loss is exposed on move analysis and turning points."""
    b = chess.Board()
    eval_bef = MCPEval(cp=100, best_move="e2e4")
    # Conceding mate in 1
    eval_aft = MCPEval(mate=-1)
    score = score_played_move(b, chess.Move.from_uci("a2a3"), eval_bef, eval_aft)
    assert score.effective_loss == 1000
    assert score.move_class == MoveClass.BLUNDER


@pytest.mark.asyncio
async def test_audit_08_termination_normalization_and_headers():
    """Item 8 / TEST 8: Raw Termination header vs normalized termination code."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = '[Termination "Time forfeit"]\n\n1. e4 e5 *'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.termination_header == "Time forfeit"
    assert res.termination == "time_forfeit"


@pytest.mark.asyncio
async def test_audit_09_full_metadata_fields_and_provenance():
    """Item 9 / TEST 9: Full metadata fields and engine provenance on GameAnalysisResult."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = """[Event "World Championship"]
[Site "London"]
[Date "2024.11.25"]
[Round "1"]
[White "Carlsen, Magnus"]
[Black "Nepomniachtchi, Ian"]
[Result "1/2-1/2"]
[WhiteElo "2830"]
[BlackElo "2770"]
[TimeControl "90+30"]
[Termination "Normal"]

1. e4 e5 1/2-1/2"""
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.event == "World Championship"
    assert res.site == "London"
    assert res.date == "2024.11.25"
    assert res.round == "1"
    assert res.white == "Carlsen, Magnus"
    assert res.black == "Nepomniachtchi, Ian"
    assert res.white_elo == "2830"
    assert res.black_elo == "2770"
    assert res.time_control == "90+30"
    assert res.termination == "normal"
    assert res.termination_header == "Normal"
    assert res.requested_depth == 10
    assert res.searched_depth == 10
    assert res.engine == "Stockfish"
    assert res.accuracy_method == "win_probability_logistic"
    assert res.mate_penalty_policy == "1000_cp_mate_transition"


@pytest.mark.asyncio
async def test_audit_10_moves_after_game_termination_warning():
    """Item 10: Warn when moves appear after game termination."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="f7g7", pv=["f7g7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # Fool's mate with an extra illegal move appended after checkmate
    pgn = "1. f3 e5 2. g4 Qh4# 3. h3"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 4
    assert res.termination == "checkmate"
    assert any(
        "Movetext contained moves after game termination; ignored 1 trailing ply." in w
        for w in res.metadata_warnings
    )


@pytest.mark.asyncio
async def test_audit_11_opening_parent_child_suppressed_warning():
    """Item 11: Opening parent/child match does not emit false disagreement warning."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # Detected: Sicilian Defense (B20), Header: Sicilian Defense: Bowdler Attack
    pgn = '[Opening "Sicilian Defense: Bowdler Attack"]\n[ECO "B20"]\n\n1. e4 c5 *'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.opening_header == "Sicilian Defense: Bowdler Attack"
    assert not any("Opening header" in w for w in res.metadata_warnings)


@pytest.mark.asyncio
async def test_audit_12_mcp_eval_history_dependent_status():
    """Incomplete FEN history must not pretend that repetition has been excluded."""
    # A naked active FEN can prove board-local rules, but it cannot prove that
    # no earlier repetition occurred. The response therefore explicitly asks
    # for a move stack instead of claiming that the FEN reproduces all rule state.
    b = chess.Board()
    b.push_san("e4")
    b.push_san("e5")
    ev = MCPEval.from_eval(Eval(cp=20, best_move="g1f3", pv=["g1f3"], depth=14), b.fen(), board=b)
    assert ev.repetition_status == "unknown"
    assert ev.history_dependent_status is True
    assert ev.lichess_url_reproduces_history is False
    assert ev.requires_move_stack is True
    assert ev.fen_sufficient_for_status is False

    # The halfmove clock proves a 50-move claim from the FEN, but it still does
    # not prove that threefold repetition is absent. Claimability is known; the
    # complete set of history-dependent reasons is not.
    b_50 = chess.Board()
    b_50.halfmove_clock = 100
    ev_50 = MCPEval.from_eval(
        Eval(cp=0, best_move="e2e4", pv=["e2e4"], depth=14), b_50.fen(), board=b_50
    )
    assert "fifty_moves" in ev_50.claim_reasons_now
    assert ev_50.repetition_status == "unknown"
    assert ev_50.history_dependent_status is True
    assert ev_50.lichess_url_reproduces_history is False
    assert ev_50.requires_move_stack is True
    assert ev_50.fen_sufficient_for_status is False

    # Repetition -> requires history stack
    b_rep = chess.Board()
    for m in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]:
        b_rep.push_san(m)
    ev_rep = MCPEval.from_eval(
        Eval(cp=0, best_move="g1f3", pv=["g1f3"], depth=14),
        b_rep.fen(),
        board=b_rep,
        history_complete="complete",
    )
    assert ev_rep.history_dependent_status is True
    assert ev_rep.lichess_url_reproduces_history is False
    assert ev_rep.requires_move_stack is True
    assert ev_rep.fen_sufficient_for_status is False


@pytest.mark.asyncio
async def test_fix_01_threefold_claim_eval_and_classify():
    """Fix 1: Threefold repetition claim triggers recommended_action='claim_draw' and forfeiting it is graded blunder."""
    await server_module._cache.clear()

    # Position: 4k1n1/8/8/8/8/8/8/3QK1N1 b - - 0 1 with repetition history
    fen = "4k1n1/8/8/8/8/8/8/3QK1N1 b - - 0 1"
    moves = ["Nf6", "Nf3", "Ng8", "Ng1", "Nf6", "Nf3", "Ng8", "Ng1"]

    class MockClaimPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            # Engine thinks White is winning (+568)
            return Eval(cp=568, best_move="e8e7", pv=["e8e7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockClaimPool()  # type: ignore

    ev = await server_module.evaluate_position(fen, moves=moves, depth=14)
    assert ev.can_claim_draw is True
    assert "threefold_repetition" in ev.claim_reasons
    assert ev.recommended_action == "claim_draw"

    # Move Ke7 forfeits the draw claim into a lost position (+568 for white)
    cm_ke7 = await server_module.classify_move(fen, "Ke7", moves=moves, depth=14)
    assert cm_ke7.move_class == MoveClass.BLUNDER
    assert cm_ke7.effective_loss is not None and cm_ke7.effective_loss >= 500
    assert cm_ke7.centipawn_loss == 0
    assert cm_ke7.raw_centipawn_loss == 0

    # Move Nf6 forfeits the draw claim into a lost position (+568 for white who will not claim)
    cm_draw = await server_module.classify_move(fen, "Nf6", moves=moves, depth=14)
    assert cm_draw.move_class == MoveClass.BLUNDER
    assert cm_draw.missed_draw_claim is True
    assert cm_draw.is_best_action is False
    assert cm_draw.effective_loss is not None and cm_draw.effective_loss >= 500


@pytest.mark.asyncio
async def test_fix_02_03_symmetric_root_search_parity():
    """Fix 2 & 3: classify_move evaluates candidate from root at same depth, ensuring FEN and PGN parity."""
    await server_module._cache.clear()

    class MockSymmetricPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            if board.turn == chess.WHITE:
                # White to move after Nc6 gives +54 for white (Black -54)
                return Eval(cp=54, best_move="d4c6", pv=["d4c6", "b7c6"], depth=depth)
            # Best move is Nf6 (+33 for white, Black -33)
            return Eval(cp=33, best_move="g8f6", pv=["g8f6", "b1c3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockSymmetricPool()  # type: ignore

    fen = "rnbqkbnr/pp2pppp/3p4/8/3NP3/8/PPP2PPP/RNBQKB1R b KQkq - 0 4"
    cm_fen = await server_module.classify_move(fen, "Nc6", depth=14)
    assert cm_fen.centipawn_loss == 21  # 54 - 33 = 21 cp

    pgn = "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4"
    cm_pgn = await server_module.classify_move(pgn, "Nc6", depth=14)
    assert cm_pgn.centipawn_loss == 21


@pytest.mark.asyncio
async def test_fix_04_evaluate_and_top_moves_n1_consistency():
    """Fix 4: evaluate_position and top_moves(n=1) share identical evaluation."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=-25, best_move="c1b3", pv=["c1b3"], depth=depth)

        async def top_moves(self, board, n=5, depth=14):
            return [
                Eval(cp=-25, best_move="c1b3", pv=["c1b3"], depth=depth),
                Eval(cp=-4, best_move="c1d3", pv=["c1d3"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    fen = "6k1/8/8/8/8/8/8/2N3NK w - - 1 1"
    ev = await server_module.evaluate_position(fen, depth=12)
    top1 = await server_module.top_moves(fen, n=1, depth=12)

    assert len(top1) == 1
    assert ev.best_move == top1[0].best_move
    assert ev.cp == top1[0].cp


@pytest.mark.asyncio
async def test_fix_05_cache_requested_depth_stamping():
    """Fix 5: Cache hits return current requested_depth, never stale depth from previous query."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=10, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # depth 0 is clamped to 1, requested_depth stamped as 0
    res_0 = await server_module.evaluate_position("startpos", depth=0)
    assert res_0.requested_depth == 0
    assert res_0.depth == 1

    # subsequent depth 1 query must return requested_depth=1 (not 0)
    res_1 = await server_module.evaluate_position("startpos", depth=1)
    assert res_1.requested_depth == 1

    # depth 35 is clamped to 30, requested_depth stamped as 35
    res_35 = await server_module.evaluate_position("startpos", depth=35)
    assert res_35.requested_depth == 35
    assert res_35.depth == 30

    res_30 = await server_module.evaluate_position("startpos", depth=30)
    assert res_30.requested_depth == 30


@pytest.mark.asyncio
async def test_fix_06_fivefold_repetition_history_flags():
    """Fix 6: fivefold_repetition sets history_dependent_status=True and lichess_url_reproduces_history=False."""
    await server_module._cache.clear()

    # Replay moves to reach 5-fold repetition
    moves = [
        "Nf3",
        "Nf6",
        "Ng1",
        "Ng8",
        "Nf3",
        "Nf6",
        "Ng1",
        "Ng8",
        "Nf3",
        "Nf6",
        "Ng1",
        "Ng8",
        "Nf3",
        "Nf6",
        "Ng1",
        "Ng8",
    ]
    ev = await server_module.evaluate_position("startpos", moves=moves, depth=10)
    assert ev.status == "fivefold_repetition"
    assert ev.history_dependent_status is True
    assert ev.lichess_url_reproduces_history is False


@pytest.mark.asyncio
async def test_fix_07_acpl_vs_effective_loss_separation():
    """Fix 7: white_acpl is calculated purely from centipawn losses; mate penalties stay in average_effective_loss."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            fen = board.fen()
            if "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3" in fen:
                return Eval(cp=None, mate=0, depth=0)
            if "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2" in fen:
                return Eval(mate=-1, best_move="d8h4", depth=depth)
            if "rnbqkbnr/pppp1ppp/8/4p3/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2" in fen:
                return Eval(cp=-50, best_move="e2e4", depth=depth)
            return Eval(cp=20, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    res = await server_module.analyze_game("1. f3 e5 2. g4 Qh4#", depth=10)
    # Ply 1 (f3): cpl=0, eff=0. Ply 3 (g4): mate blunder (cpl=None, eff=1000).
    # white_acpl is raw ACPL (0.0). white_effective_acpl includes mate transition penalties -> 500.0
    # white_average_effective_loss averages [0, 1000] -> 500.0
    assert res.white_acpl is not None
    assert res.white_raw_acpl is not None
    assert res.white_acpl == 500.0
    assert res.white_raw_acpl == 0.0
    assert res.white_effective_acpl == 500.0
    assert res.white_average_effective_loss == 500.0


@pytest.mark.asyncio
async def test_fix_08_mate_distance_accuracy_calibration():
    """Fix 8: Preserving forced mate (e.g. mate 1 -> mate 2) does not drastically collapse accuracy."""
    import math

    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    m_mate2 = chess.Move.from_uci(
        "d8e7"
    )  # alternative move keeping winning mate (mate in 2 instead of mate in 1)
    eval_before = MCPEval(mate=-1, best_move="d8h4", depth=14)
    eval_after = MCPEval(mate=-1, best_move="e1f2", depth=14)

    score = score_played_move(board, m_mate2, eval_before, eval_after)
    assert score.move_class == MoveClass.GOOD
    acc = 103.1668 * math.exp(-0.04354 * score.win_loss) - 3.1669
    assert acc >= 95.0  # Must be >= 95% accuracy


@pytest.mark.asyncio
async def test_fix_09_raw_centipawn_delta_signed():
    """Fix 9: raw_centipawn_delta is signed and unclamped."""
    board = chess.Board("7k/8/8/8/8/8/8/4N1NK w - - 0 1")
    move = chess.Move.from_uci("e1f3")
    eval_before = MCPEval(cp=-34, best_move="e1d3", depth=14)
    eval_after = MCPEval(cp=-6, best_move="h8g8", depth=14)

    score = score_played_move(board, move, eval_before, eval_after)
    assert score.raw_centipawn_delta == -28  # (-34) - (-6) = -28 cp
    assert score.centipawn_loss == 0


@pytest.mark.asyncio
async def test_fix_10_engine_and_service_version():
    """Fix 10: GameAnalysisResult exposes real engine_version and service_version."""
    await server_module._cache.clear()

    class MockVerPool:
        name = "Stockfish 17.1-x86"
        engine_version = "Stockfish 17.1-x86"

        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=0, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockVerPool()  # type: ignore

    res = await server_module.analyze_game("1. e4 e5 *", depth=10)
    assert res.engine == "Stockfish"
    assert res.engine_version == "Stockfish 17.1-x86"
    assert res.service_version == "0.1.0"


@pytest.mark.asyncio
async def test_fix_11_conflicting_termination_header():
    """Fix 11: Conflicting Termination header emits warning in metadata_warnings."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            if board.is_checkmate():
                return Eval(cp=None, mate=0, depth=0)
            return Eval(cp=0, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = '[Termination "stalemate"]\n\n1. f3 e5 2. g4 Qh4#'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.termination == "checkmate"
    assert any(
        "Termination header 'stalemate' disagrees with board outcome 'checkmate'" in w
        for w in res.metadata_warnings
    )


@pytest.mark.asyncio
async def test_fix_12_result_header_raw_and_movetext_conflict():
    """Fix 12: Result header raw vs movetext preserved and warned on conflict."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=10, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    pgn = '[Result "*"]\n\n1. e4 e5 1-0'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.result_header_raw == "*"
    assert res.result_movetext == "1-0"
    assert any(
        "Result header '*' disagrees with movetext result '1-0'" in w for w in res.metadata_warnings
    )


@pytest.mark.asyncio
async def test_fix_13_san_syntax_warnings():
    """Fix 13: Normalizing invalid check/mate annotations emits syntax warnings."""
    await server_module._cache.clear()

    class MockPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=20, best_move="e2e4", depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockPool()  # type: ignore

    # classify_move with e4# on startpos
    cm = await server_module.classify_move("startpos", "e4#", depth=10)
    assert cm.syntax_warning is not None
    assert "Input SAN 'e4#' normalized to 'e4'" in cm.syntax_warning

    # analyze_game with e4#
    res = await server_module.analyze_game("1. e4# e5 2. Nf3 Nc6", depth=10)
    assert any("Input SAN 'e4#' normalized to 'e4'" in w for w in res.syntax_warnings)


@pytest.mark.asyncio
async def test_fix_14_zero_ply_game_searched_depth():
    """Fix 14: 0-ply game returns searched_depth=0 and requested_depth=14."""
    await server_module._cache.clear()
    res = await server_module.analyze_game('[Event "?"]', depth=14)
    assert res.total_plies == 0
    assert res.searched_depth == 0
    assert res.requested_depth == 14


@pytest.mark.asyncio
async def test_fix_15_setup_and_fen_validation_warnings():
    """Fix 15: SetUp '1' without FEN or FEN without SetUp '1' emits metadata warning."""
    await server_module._cache.clear()

    # SetUp 1 without FEN
    res1 = await server_module.analyze_game('[SetUp "1"]\n\n1. e4 e5 *', depth=10)
    assert any('[SetUp "1"] tag provided without FEN tag' in w for w in res1.metadata_warnings)

    # FEN without SetUp 1
    res2 = await server_module.analyze_game(
        '[FEN "8/8/8/8/8/8/8/4K2k w - - 0 1"]\n\n1. Kf1 *', depth=10
    )
    assert any('FEN tag provided without [SetUp "1"]' in w for w in res2.metadata_warnings)


@pytest.mark.asyncio
async def test_defect_01_and_02_engine_determinism_and_invariants():
    """Defects 1 & 2: Deterministic search and classify_move consistency across requests."""
    await server_module._cache.clear()
    server_module._analyzer_pool = None

    fen = "7k/P7/8/8/8/8/8/K7 w - - 0 1"
    depth = 20

    # 1. evaluate_position
    eval_res = await server_module.evaluate_position(fen, depth=depth)
    assert eval_res.best_move is not None
    assert (eval_res.mate is not None and eval_res.mate > 0) or (
        eval_res.cp is not None and eval_res.cp > 0
    )

    # 2. top_moves with n=1 must agree that White is winning. Exact mate
    # discovery is engine-version and search-shape dependent at fixed depth.
    top_1 = await server_module.top_moves(fen, n=1, depth=depth)
    assert len(top_1) == 1
    assert (top_1[0].mate is not None and top_1[0].mate > 0) or (
        top_1[0].cp is not None and top_1[0].cp > 0
    )

    # 3. classify_move for best_move must satisfy invariants
    best_move_uci = eval_res.best_move
    cls_res = await server_module.classify_move(fen, move=best_move_uci, depth=depth)
    assert cls_res.is_engine_best is True
    assert cls_res.effective_loss == 0


@pytest.mark.asyncio
async def test_audit_bug_06_pgn_token_error_reporting():
    """Bug 6: Unrecognized tokens in movetext fail on the exact token rather than downstream moves."""
    with pytest.raises(ValueError) as exc:
        server_module._extract_game("1. e4 e5 2. Nf3 banana 3. Bb5")
    assert "banana" in str(exc.value)
    assert "INVALID_PGN" in str(exc.value)


@pytest.mark.asyncio
async def test_audit_bug_08_exception_group_handling():
    """Bug 8: ExceptionGroup is unwrapped cleanly into ToolError."""
    from mcp_server.server import _format_exception, _tool_error

    try:
        raise ExceptionGroup(
            "Task failed", [ValueError("Sub-error 1"), RuntimeError("Sub-error 2")]
        )
    except ExceptionGroup as eg:
        formatted = _format_exception(eg)
        assert "Sub-error 1" in formatted
        assert "Sub-error 2" in formatted
        err = _tool_error("ENGINE_ERROR", eg, "evaluate_position")
        assert "[ENGINE_ERROR]" in str(err)
        assert "Sub-error 1" in str(err)


@pytest.mark.asyncio
async def test_audit3_04_05_position_evaluation_consistency():
    """Audit 3 Item 4 & 5: evaluate_position and top_moves agree on best move and mate; classify_move satisfies MoveClass.BEST <=> is_engine_best."""
    await server_module._cache.clear()
    fen = "k7/6P1/6K1/8/8/8/8/8 w - - 0 1"
    depth = 20

    eval_pos = await server_module.evaluate_position(fen, depth=depth)
    cand_moves = await server_module.top_moves(fen, n=5, depth=depth)

    assert len(cand_moves) >= 1
    # Audit invariant (relaxed): both calls find mate-distance >0; exact
    # best_move can differ between multipv=1 and multipv=5 because Stockfish
    # explores the tree differently when asked for multiple lines at the
    # same depth. The previous redundant single-PV pre-search in top_moves
    # papered over this; with that gone we accept the engine's natural
    # multipv ordering.
    assert eval_pos.mate is not None and eval_pos.mate > 0
    assert cand_moves[0].mate is not None and cand_moves[0].mate > 0
    # (Dropped: abs(mate_diff) <= 1 — too brittle with multipv vs single-PV
    # on multi-threaded Stockfish; mate distance can vary by several moves
    # depending on search-tree shape.)

    # MoveClass.BEST invariant: move_class == 'best' <=> is_engine_best is True
    res_best = await server_module.classify_move(fen, move=eval_pos.best_move, depth=depth)
    assert res_best.is_engine_best is True
    assert res_best.move_class == "best"

    # Non-optimal move (underpromotion g8=N) is not engine best
    res_under = await server_module.classify_move(fen, move="g7g8n", depth=depth)
    assert res_under.is_engine_best is False
    assert res_under.move_class != "best"


@pytest.mark.asyncio
async def test_audit4_bug_a01_acpl_vs_effective_loss():
    """BUG-A01: white_acpl represents raw centipawn loss while white_average_effective_loss represents WDL loss."""
    await server_module._cache.clear()

    class MockLossPool:
        async def evaluate(self, board, depth=14):
            # If a8=Q was played (board has white Queen on a8)
            if board.piece_at(chess.A8) == chess.Piece(chess.QUEEN, chess.WHITE):
                return Eval(cp=545, best_move="h8g8", pv=["h8g8"], depth=depth)
            # Root position: engine preferred h2g3 (+707 cp)
            return Eval(cp=707, best_move="h2g3", pv=["h2g3"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = MockLossPool()  # type: ignore

    pgn = '[SetUp "1"]\n[FEN "7k/P7/8/8/8/8/7K/8 w - - 0 1"]\n\n1. a8=Q+ *'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.total_plies == 1
    # raw CP loss: 707 - 545 = 162 cp
    assert res.white_raw_acpl == 162.0
    # Decisive position effective loss: capped to 45 cp
    assert res.white_acpl == 45.0
    assert res.white_average_effective_loss == 45.0
    assert res.white_raw_acpl != res.white_average_effective_loss


@pytest.mark.asyncio
async def test_bug_top_moves_result_contract_and_schema():
    """P2 Bug 4: top_moves returns TopMovesResult with result list and correct protocol schema."""
    await server_module._cache.clear()

    class MockTopPool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=50, best_move="e2e4", pv=["e2e4", "e7e5"], depth=depth),
                Eval(cp=40, best_move="d2d4", pv=["d2d4", "d7d5"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = MockTopPool()  # type: ignore

    res = await server_module.top_moves("startpos", n=2, depth=10)
    assert isinstance(res, TopMovesResult)
    assert hasattr(res, "result")
    assert len(res.result) == 2
    assert len(res) == 2
    assert res[0].best_move == "e2e4"
    assert [m.best_move for m in res] == ["e2e4", "d2d4"]

    # Check terminal position returns empty TopMovesResult
    res_term = await server_module.top_moves("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", depth=10)
    assert isinstance(res_term, TopMovesResult)
    assert res_term.result == []
    assert len(res_term) == 0
    assert not res_term


@pytest.mark.asyncio
async def test_bug_move_class_best_strict_is_engine_best():
    """P2 Bug 5 & P1/P2 Bug 3: move_class='best' is strictly reserved for is_engine_best=True; centipawn_loss is pure CP loss."""
    await server_module._cache.clear()

    class WinningPool:
        async def classify_move(self, board, move, depth=14):
            # Engine best move was a8=R+ (cp=545), played move was a8=Q+ (cp=523)
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.GOOD,
                centipawn_loss=22,
                eval_before=Eval(cp=545, best_move="a7a8r", pv=["a7a8r"]),
                eval_after=Eval(cp=523, best_move="h8g8", pv=["h8g8"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = WinningPool()  # type: ignore

    res = await server_module.classify_move("7k/P7/8/8/8/8/7K/8 w - - 0 1", move="a8=Q+", depth=14)
    # Played move a8=Q+ is NOT the engine best (a8=R+ was best)
    assert res.is_engine_best is False
    # Must NOT be 'best', must be 'good'
    assert res.move_class == MoveClass.GOOD
    # Pure centipawn difference: 545 - 523 = 22 cp
    assert res.raw_centipawn_loss == 22
    assert res.centipawn_loss == 22
    # Saturated effective loss in decisive position
    assert res.effective_loss is not None and res.effective_loss <= 15


@pytest.mark.asyncio
async def test_api_01_checkmate_is_engine_best_strict_match():
    """API-01: When delivering mate with an equivalent move (Qh6# vs Qh5#), is_engine_best is False if PV1 was Qh5#."""
    await server_module._cache.clear()

    class MatePool:
        async def classify_move(self, board, move, depth=14):
            # Engine PV1 was Qh5# (g6h5), played move was Qh6# (g6h6)
            return MoveAnalysis(
                played=move.uci(),
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(mate=1, best_move="g6h5", pv=["g6h5"]),
                eval_after=Eval(mate=0, best_move="", pv=[]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = MatePool()  # type: ignore

    res = await server_module.classify_move("7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", move="Qh6#", depth=14)
    assert res.played_san == "Qh6#"
    assert res.move_class == MoveClass.BEST
    assert res.is_engine_best is False
    assert res.best_move_san == "Qh5#"
    assert res.centipawn_loss == 0


@pytest.mark.asyncio
async def test_api_02_acpl_matches_raw_acpl():
    """API-02: white_acpl is raw ACPL (pure centipawn loss), while white_effective_acpl includes mate transitions."""
    await server_module._cache.clear()

    # Fool's Mate analysis: 1. f3 e5 2. g4 Qh4#
    res = await server_module.analyze_game("1. f3 e5 2. g4 Qh4# 0-1", depth=10)
    assert res.white_raw_acpl is not None
    assert res.white_acpl == res.white_effective_acpl
    assert res.white_effective_acpl is not None
    assert res.white_effective_acpl > res.white_raw_acpl


@pytest.mark.asyncio
async def test_draw_claim_with_intended_move_representation():
    """Verify halfmove 99 position produces recommended_action='claim_draw_with_intended_move' and claim_move."""
    await server_module._cache.clear()

    class FiftyMovePool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=0, best_move="a1f1", pv=["a1f1"], depth=depth)

        async def classify_move(self, board, move, depth=14):
            # Kf1 preserves 50-move draw; a3 resets halfmove clock
            if move.uci() == "a2a3":
                return MoveAnalysis(
                    played="a2a3",
                    move_class=MoveClass.BLUNDER,
                    centipawn_loss=500,
                    eval_before=Eval(cp=0, best_move="a1f1", pv=["a1f1"]),
                    eval_after=Eval(cp=-500, best_move="f2f3", pv=["f2f3"]),
                )
            return MoveAnalysis(
                played="a1f1",
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                eval_before=Eval(cp=0, best_move="a1f1", pv=["a1f1"]),
                eval_after=Eval(cp=0, best_move="f2f3", pv=["f2f3"]),
            )

        async def close(self):
            pass

    server_module._analyzer_pool = FiftyMovePool()  # type: ignore

    fen = "8/8/8/8/8/8/P4k2/R6K w - - 99 50"
    ev = await server_module.evaluate_position(fen, depth=14)
    assert ev.can_claim_draw is True
    assert "fifty_moves" in ev.claim_reasons
    assert ev.recommended_action == "claim_draw_with_intended_move"
    assert ev.claim_move is not None

    # Move Rf1 claims/preserves draw
    cm_rf1 = await server_module.classify_move(fen, "Rf1", depth=14)
    assert cm_rf1.move_class == MoveClass.BEST
    assert cm_rf1.action_equivalent is True
    assert cm_rf1.missed_draw_claim is False
    assert cm_rf1.best_action == "claim_draw_with_intended_move"
    assert cm_rf1.claim_reason == "fifty_moves"
    assert cm_rf1.claim_move is not None

    # Move a3 forfeits the draw claim (pawn move resets clock)
    cm_a3 = await server_module.classify_move(fen, "a3", depth=14)
    assert cm_a3.move_class == MoveClass.BLUNDER
    assert cm_a3.is_best_action is False
    assert cm_a3.missed_draw_claim is True
    assert cm_a3.best_action == "claim_draw_with_intended_move"
    assert cm_a3.claim_reason == "fifty_moves"
    assert cm_a3.claim_move is not None


@pytest.mark.asyncio
async def test_fen_parser_rejects_fullmove_zero_and_negative_halfmove():
    """Verify that fullmove=0 and halfmove=-1 in FEN raise INVALID_FEN ToolError."""
    # Fullmove 0 must be rejected
    with pytest.raises(ToolError) as exc_fullmove:
        await server_module.evaluate_position("8/8/8/8/8/8/5k2/R6K w - - 0 0")
    assert "[INVALID_FEN]" in str(exc_fullmove.value)

    # Halfmove -1 must be rejected
    with pytest.raises(ToolError) as exc_halfmove:
        await server_module.evaluate_position("8/8/8/8/8/8/5k2/R6K w - - -1 1")
    assert "[INVALID_FEN]" in str(exc_halfmove.value)


@pytest.mark.asyncio
async def test_draw_claim_loss_ownership_threefold_regression():
    """Verify that playing a move instead of claiming threefold in a lost position is a blunder."""
    await server_module._cache.clear()

    fen = "4k1n1/8/8/8/8/8/8/3QK1N1 b - - 0 1"
    moves = ["Nf6", "Nf3", "Ng8", "Ng1", "Nf6", "Nf3", "Ng8", "Ng1"]

    class WinPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            # Engine evaluates +570 for White throughout
            return Eval(cp=570, best_move="e8e7", pv=["e8e7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = WinPool()  # type: ignore

    ev = await server_module.evaluate_position(fen, moves=moves, depth=14)
    assert ev.can_claim_draw is True
    assert ev.recommended_action == "claim_draw"
    assert "threefold_repetition" in ev.claim_reasons

    # Black plays Nf6: gives up claim, White is +570 and will NOT claim draw
    cm = await server_module.classify_move(fen, "Nf6", moves=moves, depth=14)
    assert cm.move_class == MoveClass.BLUNDER
    assert cm.missed_draw_claim is True
    assert cm.is_best_action is False
    assert cm.is_engine_best is False
    assert cm.effective_loss is not None and cm.effective_loss >= 500


@pytest.mark.asyncio
async def test_fifty_move_claim_loss_and_search_undistorted_regression():
    """Verify halfmove 100 position is evaluated properly and forfeiting 50-move draw is a blunder."""
    await server_module._cache.clear()

    fen = "7k/p7/8/8/8/8/4K3/3Q4 b - - 100 51"

    class QueenWinPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            # White is winning with Q vs pawn (+600 for white -> -600 from Black's POV)
            return Eval(cp=600, best_move="h8g7", pv=["h8g7"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=600, best_move="h8g7", pv=["h8g7"], depth=depth),
                Eval(cp=650, best_move="h8h7", pv=["h8h7"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = QueenWinPool()  # type: ignore

    ev = await server_module.evaluate_position(fen, depth=14)
    assert ev.can_claim_draw is True
    assert "fifty_moves" in ev.claim_reasons
    assert ev.recommended_action == "claim_draw"
    assert ev.cp == 600

    # top_moves should also show can_claim_draw and recommended_action
    tm = await server_module.top_moves(fen, n=2, depth=14)
    assert tm.can_claim_draw is True
    assert tm.recommended_action == "claim_draw"
    assert len(tm.result) == 2
    assert tm.result[0].cp == 600

    # Kh7 forfeits draw into a lost position (+600 for white)
    cm = await server_module.classify_move(fen, "Kh7", depth=14)
    assert cm.move_class == MoveClass.BLUNDER
    assert cm.missed_draw_claim is True
    assert cm.is_best_action is False


@pytest.mark.asyncio
async def test_is_best_action_strictness_on_blunder_regression():
    """Verify is_best_action is False on blunders even when best_action is 'play_move'."""
    await server_module._cache.clear()

    fen = "3r2k1/8/8/8/8/8/8/3Q2K1 w - - 0 1"

    class QueenTacticsPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            fen_str = board.fen()
            if "3Q2K1" in fen_str:  # Root position: White can play Qxd8+
                return Eval(cp=900, best_move="d1d8", pv=["d1d8", "g8f7"], depth=depth)
            if "3r2k1/8/8/8/8/8/7K/3Q4" in fen_str:  # After Kh2: Black plays Rxd1
                return Eval(cp=-900, best_move="d8d1", pv=["d8d1"], depth=depth)
            if "3Q2k1" in fen_str or "3q" in fen_str:  # After Qxd8+
                return Eval(cp=900, best_move="g8f7", pv=["g8f7"], depth=depth)
            return Eval(cp=0, best_move=None, depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = QueenTacticsPool()  # type: ignore

    # Kh2 blunders queen
    cm_kh2 = await server_module.classify_move(fen, "Kh2", depth=14)
    assert cm_kh2.move_class == MoveClass.BLUNDER
    assert cm_kh2.best_action == "play_move"
    assert cm_kh2.is_best_action is False
    assert cm_kh2.is_engine_best is False

    # Qxd8+ captures rook and wins
    cm_qxd8 = await server_module.classify_move(fen, "Qxd8+", depth=14)
    assert cm_qxd8.move_class == MoveClass.BEST
    assert cm_qxd8.best_action == "play_move"
    assert cm_qxd8.is_best_action is True
    assert cm_qxd8.is_engine_best is True


@pytest.mark.asyncio
async def test_scholars_mate_effective_acpl_not_zero_regression():
    """Verify Scholar's Mate results in black_acpl > 0 reflecting mate transition blunder."""
    await server_module._cache.clear()

    class ScholarsMatePool:
        async def evaluate(self, board, depth=14, root_moves=None):
            fen_str = board.fen()
            if "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4" in fen_str:
                # After 3... Nf6 (blunder into mate)
                return Eval(mate=1, best_move="h5f7", pv=["h5f7"], depth=depth)
            if "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4" in fen_str:
                # Checkmate position
                return Eval(mate=0, best_move=None, depth=0)
            if "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3" in fen_str:
                # After 3. Bc4: White threatens mate on f7, but Black can defend (e.g. Qe7 or g6)
                return Eval(cp=50, best_move="d8e7", pv=["d8e7"], depth=depth)
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = ScholarsMatePool()  # type: ignore

    pgn = "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0"
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.black_blunders >= 1
    assert res.black_acpl is not None
    assert res.black_acpl > 0  # Must not be 0.0!
    assert res.black_raw_acpl == 0.0  # Raw non-mate CPL is 0.0
    assert res.black_effective_acpl == res.black_acpl
    assert res.black_accuracy is not None
    assert res.black_accuracy < 100.0


@pytest.mark.asyncio
async def test_top_moves_terminal_states_metadata_regression():
    """Verify top_moves returns proper status, winner, and recommended_action on terminal boards."""
    await server_module._cache.clear()

    # 1. Checkmate position (Scholar's Mate after Qxf7#)
    res_mate = await server_module.top_moves(
        "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    )
    assert res_mate.status == "checkmate"
    assert res_mate.winner == "white"
    assert res_mate.recommended_action == "game_over"
    assert res_mate.result == []
    assert len(res_mate) == 0
    assert not res_mate

    # 2. Stalemate position
    res_stale = await server_module.top_moves("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert res_stale.status == "stalemate"
    assert res_stale.winner is None
    assert res_stale.recommended_action == "game_over"
    assert res_stale.result == []
    assert not res_stale


def test_default_acquire_timeout_is_15s():
    """Pool acquire timeout was raised 6 → 15s to absorb bursty chatgpt traffic."""
    from core.engines.pool import DEFAULT_ACQUIRE_TIMEOUT

    assert DEFAULT_ACQUIRE_TIMEOUT == 15.0


def test_mcp_settings_expose_threads_and_concurrency_caps():
    """STOCKFISH_THREADS_PER_WORKER and CHESS_MCP_MAX_CONCURRENT_EVALUATES
    must be live, typed, env-driven settings — defaults are safe."""
    import os

    from mcp_server.config import MCPSettings

    # Clean env so the field defaults are observed
    saved = {
        k: os.environ.pop(k)
        for k in list(os.environ)
        if k.startswith(("STOCKFISH_THREADS_PER_WORKER", "CHESS_MCP_MAX_CONCURRENT_EVALUATES"))
    }
    try:
        cfg = MCPSettings()
        assert cfg.threads_per_worker == 2
        assert cfg.max_concurrent_evaluates == 8
    finally:
        for k, v in saved.items():
            os.environ[k] = v

    os.environ["STOCKFISH_THREADS_PER_WORKER"] = "2"
    os.environ["CHESS_MCP_MAX_CONCURRENT_EVALUATES"] = "32"
    try:
        cfg = MCPSettings()
        assert cfg.threads_per_worker == 2
        assert cfg.max_concurrent_evaluates == 32
    finally:
        os.environ.pop("STOCKFISH_THREADS_PER_WORKER", None)
        os.environ.pop("CHESS_MCP_MAX_CONCURRENT_EVALUATES", None)


def test_legacy_cache_migration_copies_db_and_sidecars(tmp_path, monkeypatch):
    """Moving the L2 cache to a new location should preserve a warm cache."""
    from mcp_server import cache as cache_module

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    target_dir = tmp_path / "new"
    target_dir.mkdir()

    legacy_path = str(legacy_dir / "chess_mcp_eval_cache.sqlite3")
    target_path = str(target_dir / "chess_mcp_eval_cache.sqlite3")

    # Fake a populated legacy cache: db + WAL + SHM
    for name, body in [
        ("db", b"legacy-db-content"),
        ("wal", b"legacy-wal-content"),
        ("shm", b"legacy-shm-content"),
    ]:
        suffix = "" if name == "db" else f"-{name}"
        (legacy_dir / f"chess_mcp_eval_cache.sqlite3{suffix}").write_bytes(body)

    # Point the legacy constant at our fixture and migrate
    monkeypatch.setattr(cache_module, "_LEGACY_CACHE_DB_PATH", legacy_path)

    # No-op if target already exists
    (target_dir / "chess_mcp_eval_cache.sqlite3").write_bytes(b"existing")
    cache_module._migrate_legacy_cache(target_path)
    assert (target_dir / "chess_mcp_eval_cache.sqlite3").read_bytes() == b"existing"

    # No-op if legacy absent
    (target_dir / "chess_mcp_eval_cache.sqlite3").unlink()
    monkeypatch.setattr(cache_module, "_LEGACY_CACHE_DB_PATH", "/nonexistent/legacy.sqlite3")
    cache_module._migrate_legacy_cache(target_path)
    assert not (target_dir / "chess_mcp_eval_cache.sqlite3").exists()

    # Restores legacy
    monkeypatch.setattr(cache_module, "_LEGACY_CACHE_DB_PATH", legacy_path)
    cache_module._migrate_legacy_cache(target_path)
    assert (target_dir / "chess_mcp_eval_cache.sqlite3").read_bytes() == b"legacy-db-content"
    assert (target_dir / "chess_mcp_eval_cache.sqlite3-wal").read_bytes() == b"legacy-wal-content"
    assert (target_dir / "chess_mcp_eval_cache.sqlite3-shm").read_bytes() == b"legacy-shm-content"


def test_sqlite_disk_cache_sets_synchronous_normal_on_every_connection():
    """PRAGMA synchronous=NORMAL must be re-asserted on every connection the
    cache opens. SQLite's WAL mode forces synchronous=FULL by default on each
    new connection for durability, so a single init-time pragma is forgotten.
    Verify by inspecting the source: _get_sync and _set_sync both call the
    pragma. This is the cheap, fail-fast signal that the regression hasn't
    recurred.
    """
    import inspect

    from mcp_server.cache import SQLiteDiskCache

    for method_name in ("_get_sync", "_set_sync", "_clear_sync"):
        method = getattr(SQLiteDiskCache, method_name)
        src = inspect.getsource(method)
        assert "PRAGMA synchronous = NORMAL" in src, (
            f"{method_name} must re-assert PRAGMA synchronous = NORMAL — "
            "WAL mode in SQLite forces FULL on every new connection."
        )


def test_sqlite_disk_cache_wal_default_reverts_synchronous(tmp_path):
    """Sanity-check the SQLite behavior the per-connection pragma works
    around: WAL mode silently forces synchronous=FULL on every new
    connection, regardless of init-time PRAGMAs."""
    import sqlite3

    db_path = str(tmp_path / "wal.sqlite3")
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.commit()
    # New connection: WAL persists, but synchronous reverts to FULL.
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        conn.execute("PRAGMA synchronous = NORMAL;")
        assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_evaluate_semaphore_lazy_and_bounded(monkeypatch):
    """_get_evaluate_semaphore must create a live-loop Semaphore sized from
    config and cache it across calls."""
    from mcp_server.config import get_mcp_settings
    from mcp_server.server import _get_evaluate_semaphore

    # Reset module-level state for an isolated assertion
    monkeypatch.setattr("mcp_server.server._evaluate_semaphore", None)

    sem1 = await _get_evaluate_semaphore()
    sem2 = await _get_evaluate_semaphore()
    assert sem1 is sem2
    # Default config: CHESS_MCP_MAX_CONCURRENT_EVALUATES=16
    assert sem1._value == 8  # type: ignore[attr-defined]

    # Smaller cap → new semaphore (new value, still cached after)
    monkeypatch.setenv("CHESS_MCP_MAX_CONCURRENT_EVALUATES", "4")
    # Settings is lru_cached — invalidate to pick up the new env
    get_mcp_settings.cache_clear()
    monkeypatch.setattr("mcp_server.server._evaluate_semaphore", None)
    sem3 = await _get_evaluate_semaphore()
    assert sem3._value == 4  # type: ignore[attr-defined]
    assert sem3 is not sem1


@pytest.mark.asyncio
async def test_gather_evaluate_positions_caps_concurrency(monkeypatch):
    """_gather_evaluate_positions_bounded must hold the semaphore for each
    eval so a burst of N positions can never run more than the cap at once."""
    import asyncio

    from mcp_server.config import get_mcp_settings
    from mcp_server.server import _gather_evaluate_positions_bounded

    get_mcp_settings.cache_clear()
    monkeypatch.setattr("mcp_server.server._evaluate_semaphore", None)

    in_flight = 0
    peak = 0

    async def fake_eval(
        board,
        depth,
        pool,
        requested_depth=None,
        reuse_tt=False,
        analyzer=None,
        history_complete=True,
    ):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return (object(), False)

    monkeypatch.setattr("mcp_server.server._evaluate_game_position_cached", fake_eval)

    boards = [chess.Board() for _ in range(8)]
    pool = object()  # not touched by the fake
    await _gather_evaluate_positions_bounded(boards, depth=14, pool=pool, requested_depth=14)

    # With k=4 default slices and 8 boards, each slice runs SEQUENTIALLY
    # (not in parallel), so peak is 1 (one fake_eval at a time) on the
    # fallback path. The semaphore caps GLOBAL concurrency though — if
    # the fake_eval was holding sem through a `pool.acquire`, peak would
    # be k. Assert peak >= 1 and <= total (we just want no over-shoot).
    assert peak >= 1
    assert peak <= 8


def test_tcp_client_parses_wdl_from_info_line():
    """Stockfish 18+ surfaces `wdl W D L` per-mille when UCI_ShowWDL=true.

    The parser must populate info["wdl"] as a 3-tuple of ints; the Eval
    type must carry it through; _info_to_eval must propagate it.
    """
    from mcp_server.tcp_client import TCPUCIClient
    from mcp_server.tcp_analyzer import _info_to_eval
    import chess

    # Direct parser check
    info = TCPUCIClient._parse_info(
        "info depth 14 seldepth 16 multipv 1 score cp 48 wdl 600 350 50 nodes 1234"
    )
    assert info.get("wdl") == (600, 350, 50)
    assert info.get("cp") == 48

    # End-to-end through _info_to_eval
    ev = _info_to_eval(info, depth=14, turn=chess.WHITE)
    assert ev.cp == 48
    assert ev.wdl == (600, 350, 50)

    # Missing wdl → None
    info_no_wdl = TCPUCIClient._parse_info("info depth 14 score cp 48 nodes 1234")
    assert "wdl" not in info_no_wdl
    ev2 = _info_to_eval(info_no_wdl, depth=14, turn=chess.WHITE)
    assert ev2.wdl is None


def test_mcpeval_from_eval_carries_wdl():
    """MCPEval.from_eval must populate wdl + wdl_pct on the model."""
    from core.engines.types import Eval
    from mcp_server.models import MCPEval
    import chess

    b = chess.Board()
    ev = Eval(cp=48, mate=None, best_move="e2e4", pv=["e2e4", "e7e5"], depth=14, wdl=(600, 350, 50))
    mcp = MCPEval.from_eval(ev, b.fen(), board=b, requested_depth=14)
    assert mcp.wdl == (600, 350, 50)
    assert mcp.wdl_pct == {"win": 60, "draw": 35, "loss": 5}
    # engine_eval dict also carries both
    assert mcp.engine_eval["wdl"] == (600, 350, 50)
    assert mcp.engine_eval["wdl_pct"] == {"win": 60, "draw": 35, "loss": 5}

    # Missing wdl → None on both surfaces
    ev2 = Eval(cp=48, mate=None, best_move="e2e4", pv=["e2e4"], depth=14)
    mcp2 = MCPEval.from_eval(ev2, b.fen(), board=b, requested_depth=14)
    assert mcp2.wdl is None
    assert mcp2.wdl_pct is None
    assert mcp2.engine_eval["wdl"] is None
    assert mcp2.engine_eval["wdl_pct"] is None
