from __future__ import annotations

import asyncio

import chess
import chess.engine

from .analysis import classify_move, probe_threat, pv_to_san
from .types import Eval

__all__ = ["Analyzer", "pv_to_san"]


class Analyzer:
    """Full-strength Stockfish wrapper for evaluation and move grading.

    Pure transport: implements the chess engine UCI protocol. Move classification
    and threat probing are delegated to `core.engines.analysis` so a second
    transport (TCP) can reuse the same algorithm.
    """

    def __init__(
        self, transport: chess.engine.SubprocessTransport, engine: chess.engine.UciProtocol, depth: int
    ) -> None:
        self._transport = transport
        self._engine = engine
        self._depth = depth
        # one Stockfish process can't take concurrent UCI commands (pydantic-ai runs parallel tool calls
        # concurrently → corrupts the engine's info dict / kills it); serialize every analyse through this.
        self._lock = asyncio.Lock()
        self.name: str = engine.id.get("name", "stockfish")

    @classmethod
    async def create(cls, path: str, *, depth: int = 12, threads: int = 2, hash_mb: int = 128) -> Analyzer:
        transport, engine = await chess.engine.popen_uci(path)
        await engine.configure({"Threads": threads, "Hash": hash_mb})
        return cls(transport, engine, depth)

    async def evaluate(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        root_moves: list[chess.Move] | None = None,
        reuse_tt: bool = False,
    ) -> Eval:
        actual_depth = depth or self._depth
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

        board_for_sf = board

        async with self._lock:
            if not reuse_tt:
                self._engine.send_line("ucinewgame")
            await self._engine.ping()
            if root_moves:
                info = await self._engine.analyse(
                    board_for_sf, chess.engine.Limit(depth=actual_depth), multipv=1, root_moves=root_moves
                )
            else:
                info = await self._engine.analyse(
                    board_for_sf, chess.engine.Limit(depth=actual_depth), multipv=1
                )
        info_dict = info[0] if isinstance(info, list) else info
        score = info_dict["score"].white()
        pv: list[chess.Move] = info_dict.get("pv", [])
        mate_val = score.mate()
        cp_val = None if mate_val is not None else score.score()
        return Eval(
            cp=cp_val,
            mate=mate_val,
            best_move=pv[0].uci() if pv else None,
            pv=[m.uci() for m in pv],
            depth=info_dict.get("depth", actual_depth),
        )

    async def top_moves(
        self,
        board: chess.Board,
        *,
        n: int = 3,
        depth: int | None = None,
        reuse_tt: bool = False,
    ) -> list[Eval]:
        """The engine's top-n candidate moves (MultiPV), best first."""
        if board.is_game_over():
            return []
        actual_depth = depth or self._depth
        mpv = max(1, n)
        board_for_sf = board
        async with self._lock:
            if not reuse_tt:
                self._engine.send_line("ucinewgame")
            await self._engine.ping()
            infos = await self._engine.analyse(
                board_for_sf, chess.engine.Limit(depth=actual_depth), multipv=mpv
            )
        out: list[Eval] = []
        for info in infos[:n]:
            score = info["score"].white()
            pv: list[chess.Move] = info.get("pv", [])
            if not pv:
                continue
            mate_val = score.mate()
            cp_val = None if mate_val is not None else score.score()
            out.append(
                Eval(
                    cp=cp_val,
                    mate=mate_val,
                    best_move=pv[0].uci(),
                    pv=[m.uci() for m in pv],
                    depth=info.get("depth", actual_depth),
                )
            )
        return out

    async def classify_move(self, board: chess.Board, move: chess.Move, *, depth: int | None = None):
        return await classify_move(self, board, move, depth=depth)

    async def probe_threat(self, board_after: chess.Board, *, depth: int | None = None):
        return await probe_threat(self, board_after, depth=depth)

    async def close(self) -> None:
        await self._engine.quit()
