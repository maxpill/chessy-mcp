from __future__ import annotations

import asyncio
from typing import Protocol

import chess
import chess.engine

# Stockfish 18 dev accepts UCI_Elo in range 1320..3190.
STOCKFISH_ELO_MIN = 1320
STOCKFISH_ELO_MAX = 3190



class OpponentEngine(Protocol):
    """A swappable opponent. Stockfish today; lc0+Maia behind the same interface later."""

    name: str

    async def select_move(self, board: chess.Board, target_rating: int) -> chess.Move: ...

    async def close(self) -> None: ...


class StockfishOpponent:
    def __init__(self, transport: chess.engine.SubprocessTransport, engine: chess.engine.UciProtocol) -> None:
        self._transport = transport
        self._engine = engine
        # One Stockfish process cannot handle concurrent UCI commands — configure+play must be atomic.
        self._lock = asyncio.Lock()
        self.name: str = engine.id.get("name", "stockfish")

    @classmethod
    async def create(cls, path: str) -> StockfishOpponent:
        transport, engine = await chess.engine.popen_uci(path)
        return cls(transport, engine)

    async def select_move(self, board: chess.Board, target_rating: int) -> chess.Move:
        elo = max(STOCKFISH_ELO_MIN, min(STOCKFISH_ELO_MAX, target_rating))
        async with self._lock:
            await self._engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
            result = await self._engine.play(board, chess.engine.Limit(time=0.1))
        assert result.move is not None
        return result.move

    async def close(self) -> None:
        await self._engine.quit()
