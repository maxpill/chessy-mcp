"""TCP-based analyzer — same interface as Analyzer, but connects over the network."""

from __future__ import annotations

import asyncio
from typing import Any

import chess

from core.engines.analysis import classify_move, probe_threat
from core.engines.pool import DEFAULT_ACQUIRE_TIMEOUT, EnginePool
from core.engines.types import Eval, MoveAnalysis
from mcp_server.tcp_client import TCPUCIClient


def _info_to_eval(info: dict[str, Any], depth: int, turn: chess.Color) -> Eval:
    """Convert a parsed UCI info dict to an Eval model."""
    pv: list[str] = [m for m in info.get("pv", []) if m != "(none)"]
    sign = 1 if turn == chess.WHITE else -1
    cp = info.get("cp")
    mate = info.get("mate")
    if mate is not None:
        return Eval(
            cp=None,
            mate=sign * mate if mate != 0 else 0,
            best_move=pv[0] if pv else None,
            pv=pv,
            depth=info.get("depth", depth),
        )
    return Eval(
        cp=sign * cp if cp is not None else None,
        mate=None,
        best_move=pv[0] if pv else None,
        pv=pv,
        depth=info.get("depth", depth),
    )


class TCPAnalyzer:
    """Stockfish or Maia over TCP — mirrors Analyzer's public interface.

    Pure transport: implements the wire protocol and the OperationResult → Eval
    conversion. Move classification and threat probing are delegated to
    `core.engines.analysis` so this backend shares the same cp-loss semantics
    as the local UCI subprocess backend.
    """

    def __init__(self, client: TCPUCIClient) -> None:
        self._client = client
        self.name: str = client.name
        self._lock = asyncio.Lock()
        self._depth: int = 14

    @classmethod
    async def create(
        cls, host: str, port: int, *, name: str = "stockfish", threads: int = 2, hash_mb: int = 128
    ) -> TCPAnalyzer:
        client = TCPUCIClient(host, port, name=name)
        await client.connect()
        await client.configure({"Threads": threads, "Hash": hash_mb})
        return cls(client)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        actual_depth = depth if depth is not None else self._depth
        if board.is_checkmate():
            return Eval(
                cp=None,
                mate=0,
                best_move=None,
                pv=[],
                depth=0,
            )
        if board.is_game_over():
            return Eval(cp=0, mate=None, best_move=None, pv=[], depth=0)

        fen_str = board.fen()

        searchmoves = [m.uci() for m in root_moves] if root_moves else None
        results = await self._client.analyse(fen_str, actual_depth, multipv=1, searchmoves=searchmoves)
        if not results:
            return Eval()
        return _info_to_eval(results[0], actual_depth, board.turn)

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int | None = None) -> list[Eval]:
        if board.is_game_over():
            return []
        actual_depth = depth if depth is not None else self._depth
        mpv = max(1, n)
        fen_str = board.fen()
        results = await self._client.analyse(fen_str, actual_depth, multipv=mpv)
        out: list[Eval] = []
        for info in results[:n]:
            if not info.get("pv") or info.get("pv") == ["(none)"]:
                continue
            out.append(_info_to_eval(info, actual_depth, board.turn))
        return out

    async def classify_move(
        self, board: chess.Board, move: chess.Move, *, depth: int | None = None
    ) -> MoveAnalysis:
        return await classify_move(self, board, move, depth=depth)

    async def probe_threat(self, board_after: chess.Board, *, depth: int | None = None) -> Eval | None:
        return await probe_threat(self, board_after, depth=depth)

    async def close(self) -> None:
        await self._client.close()


class TCPAnalyzerPool:
    """Drop-in pool for TCPAnalyzer: handles N concurrent TCP connections."""

    def __init__(self, pool: EnginePool, name: str = "Stockfish") -> None:
        self._pool = pool
        self.name = name
        self.engine_version = name

    @classmethod
    async def create(
        cls,
        host: str,
        port: int,
        size: int,
        *,
        name: str = "stockfish",
        threads: int = 1,
        hash_mb: int = 128,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> TCPAnalyzerPool:
        async def factory() -> object:
            return await TCPAnalyzer.create(host, port, name=name, threads=threads, hash_mb=hash_mb)

        instances = [await factory() for _ in range(max(1, size))]
        engine_name = getattr(instances[0], "name", name) if instances else name
        return cls(EnginePool(instances, factory, acquire_timeout), name=engine_name)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
    ) -> Eval:
        return await self._pool.run(lambda a: a.evaluate(board, depth=depth, root_moves=root_moves))  # type: ignore[attr-defined]

    async def top_moves(self, board: chess.Board, *, n: int = 3, depth: int | None = None) -> list[Eval]:
        return await self._pool.run(lambda a: a.top_moves(board, n=n, depth=depth))  # type: ignore[attr-defined]

    async def classify_move(
        self, board: chess.Board, move: chess.Move, *, depth: int | None = None
    ) -> MoveAnalysis:
        return await self._pool.run(lambda a: a.classify_move(board, move, depth=depth))  # type: ignore[attr-defined]

    async def probe_threat(self, board_after: chess.Board, *, depth: int | None = None) -> Eval | None:
        return await self._pool.run(lambda a: a.probe_threat(board_after, depth=depth))  # type: ignore[attr-defined]

    async def close(self) -> None:
        await self._pool.close()
