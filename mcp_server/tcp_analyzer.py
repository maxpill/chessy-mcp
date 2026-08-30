"""TCP-based analyzer with the same interface as the local Analyzer."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import chess

from core.engines.analysis import classify_move
from core.engines.pool import DEFAULT_ACQUIRE_TIMEOUT, EnginePool
from core.engines.types import Eval, MoveAnalysis
from mcp_server.tcp_client import TCPUCIClient


def _white_pov_wdl(
    raw: object,
    turn: chess.Color,
) -> tuple[int, int, int] | None:
    """Normalize UCI WDL from side-to-move POV into White POV."""
    if raw is None:
        return None
    try:
        win, draw, loss = int(raw[0]), int(raw[1]), int(raw[2])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None
    return (win, draw, loss) if turn == chess.WHITE else (loss, draw, win)


def _info_to_eval(info: dict[str, Any], depth: int, turn: chess.Color) -> Eval:
    """Convert parsed UCI info into the public White-POV Eval contract."""
    pv: list[str] = [m for m in info.get("pv", []) if m != "(none)"]
    sign = 1 if turn == chess.WHITE else -1
    cp = info.get("cp")
    mate = info.get("mate")
    wdl = _white_pov_wdl(info.get("wdl"), turn)
    if mate is not None:
        return Eval(
            cp=None,
            mate=sign * mate if mate != 0 else 0,
            best_move=pv[0] if pv else None,
            pv=pv,
            depth=info.get("depth", depth),
            wdl=wdl,
        )
    return Eval(
        cp=sign * cp if cp is not None else None,
        mate=None,
        best_move=pv[0] if pv else None,
        pv=pv,
        depth=info.get("depth", depth),
        wdl=wdl,
    )


class TCPAnalyzer:
    """Stockfish or Maia over TCP."""

    def __init__(self, client: TCPUCIClient) -> None:
        self._client = client
        self.name: str = client.name
        self._lock = asyncio.Lock()
        self._depth: int = 14

    @classmethod
    async def create(
        cls,
        host: str,
        port: int,
        *,
        name: str = "stockfish",
        threads: int = 2,
        hash_mb: int = 128,
        show_wdl: bool = False,
        syzygy_path: str | None = None,
    ) -> TCPAnalyzer:
        client = TCPUCIClient(host, port, name=name)
        await client.connect()
        options: dict[str, int | str] = {"Threads": threads, "Hash": hash_mb}
        if show_wdl:
            options["UCI_ShowWDL"] = "true"
        if syzygy_path:
            options["SyzygyPath"] = syzygy_path
            options["SyzygyProbeLimit"] = 7
        await client.configure(options)
        return cls(client)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
        reuse_tt: bool = False,
    ) -> Eval:
        actual_depth = depth if depth is not None else self._depth
        if board.is_checkmate():
            return Eval(cp=None, mate=0, best_move=None, pv=[], depth=0)
        if board.is_game_over(claim_draw=False):
            return Eval(cp=0, mate=None, best_move=None, pv=[], depth=0)

        searchmoves = [m.uci() for m in root_moves] if root_moves else None
        results = await self._client.analyse(
            board.fen(),
            actual_depth,
            multipv=1,
            searchmoves=searchmoves,
            reuse_tt=reuse_tt,
        )
        if not results:
            return Eval()
        return _info_to_eval(results[0], actual_depth, board.turn)

    async def top_moves(
        self,
        board: chess.Board,
        *,
        n: int = 3,
        depth: int | None = None,
        reuse_tt: bool = False,
    ) -> list[Eval]:
        if board.is_game_over(claim_draw=False):
            return []
        actual_depth = depth if depth is not None else self._depth
        results = await self._client.analyse(
            board.fen(),
            actual_depth,
            multipv=max(1, n),
            reuse_tt=reuse_tt,
        )
        out: list[Eval] = []
        for info in results[:n]:
            if not info.get("pv") or info.get("pv") == ["(none)"]:
                continue
            out.append(_info_to_eval(info, actual_depth, board.turn))
        return out

    async def classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        depth: int | None = None,
    ) -> MoveAnalysis:
        return await classify_move(self, board, move, depth=depth)

    async def close(self) -> None:
        await self._client.close()


class TCPAnalyzerPool:
    """Drop-in pool for TCPAnalyzer."""

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
        show_wdl: bool = False,
        syzygy_path: str | None = None,
        acquire_timeout: float = DEFAULT_ACQUIRE_TIMEOUT,
    ) -> TCPAnalyzerPool:
        async def factory() -> object:
            return await TCPAnalyzer.create(
                host,
                port,
                name=name,
                threads=threads,
                hash_mb=hash_mb,
                show_wdl=show_wdl,
                syzygy_path=syzygy_path,
            )

        instances = await asyncio.gather(*[factory() for _ in range(max(1, size))])
        engine_name = getattr(instances[0], "name", name) if instances else name
        return cls(EnginePool(instances, factory, acquire_timeout), name=engine_name)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
        reuse_tt: bool = False,
    ) -> Eval:
        return await self._pool.run(
            lambda a: cast(TCPAnalyzer, a).evaluate(
                board,
                depth=depth,
                root_moves=root_moves,
                reuse_tt=reuse_tt,
            )
        )

    async def top_moves(
        self,
        board: chess.Board,
        *,
        n: int = 3,
        depth: int | None = None,
    ) -> list[Eval]:
        return await self._pool.run(
            lambda a: cast(TCPAnalyzer, a).top_moves(board, n=n, depth=depth)
        )

    async def classify_move(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        depth: int | None = None,
    ) -> MoveAnalysis:
        return await self._pool.run(
            lambda a: cast(TCPAnalyzer, a).classify_move(board, move, depth=depth)
        )

    async def close(self) -> None:
        await self._pool.close()
