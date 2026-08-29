"""Async UCI-over-TCP client for remote Stockfish/Maia engines."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class UCIError(Exception):
    pass


class TCPUCIClient:
    """Persistent UCI session over a TCP socket (e.g. socat → stockfish)."""

    def __init__(self, host: str, port: int, name: str = "engine") -> None:
        self._host = host
        self._port = port
        self.name = name
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._lock = asyncio.Lock()
        # Track the last applied UCI options so that a connection reset
        # (auto-reconnect on engine death) RE-APPLIES them. Without this,
        # a freshly respawned Stockfish silently keeps its compiled-in
        # defaults (MultiPV=1, etc.) while the client thinks the previous
        # options are still in effect — a classic source of "wrong top_n"
        # bugs after a network blip.
        self._current_multipv: int = 1
        self._applied_options: dict[str, int | str] = {}

    async def connect(self) -> None:
        try:
            self._reader, self._writer = await asyncio.open_connection(self._host, self._port)
            await self._send("uci")
            await self._wait_for("uciok", timeout=10)
            # Re-apply every previously configured UCI option. UCI options
            # are NOT persisted across process restarts, and a reconnected
            # engine comes back with its compiled-in defaults.
            if self._applied_options:
                for key, value in self._applied_options.items():
                    await self._send(f"setoption name {key} value {value}")
            await self._send("isready")
            await self._wait_for("readyok", timeout=10)
            logger.info("Connected to %s at %s:%d", self.name, self._host, self._port)
        except BaseException:
            await self._reset_connection()
            raise

    async def configure(self, options: dict[str, int | str]) -> None:
        async with self._lock:
            for key, value in options.items():
                await self._send(f"setoption name {key} value {value}")
                self._applied_options[key] = value
            await self._send("isready")
            await self._wait_for("readyok", timeout=10)

    async def _ensure_connected(self) -> None:
        if self._reader is None or self._writer is None:
            await self.connect()

    async def _reset_connection(self) -> None:
        if self._writer:
            try:
                self._send_no_lock("quit")
            except (ConnectionError, OSError):
                pass
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        self._reader = None
        self._writer = None
        # Reset MultiPV tracking: a new engine instance defaults to MultiPV=1
        # regardless of what we last sent. We KEEP _applied_options so connect()
        # re-applies Threads/Hash/etc.; only MultiPV is reset here because
        # it's the only one whose non-applied state matters for the next call.
        #
        # P1 audit fix: also drop MultiPV from _applied_options. Without this,
        # connect() re-applies the OLD MultiPV value to the new engine, so
        # _current_multipv=1 vs _applied_options["MultiPV"]=5 leaves a phantom
        # 5-line Stockfish while the client thinks the next multipv=1 call
        # needs no reset (1 != 1 → False). Drop the MultiPV entry so connect()
        # leaves the new engine at its compiled-in default of MultiPV=1.
        self._current_multipv = 1
        self._applied_options.pop("MultiPV", None)

    async def analyse(
        self,
        fen: str,
        depth: int,
        *,
        multipv: int = 1,
        searchmoves: list[str] | None = None,
        reuse_tt: bool = False,
    ) -> list[dict[str, Any]]:
        """Run `go depth` analysis, return list of info dicts (one per PV line).

        reuse_tt=False (default) issues `ucinewgame` first — Stockfish clears
        its transposition table, every search starts cold. Pass reuse_tt=True
        when consecutive calls on this connection share position-tree history
        (e.g. contiguous plies in analyze_game, before/after pair in
        classify_move): Stockfish accumulates the TT across calls, and
        positions 2..N of a chain get back cached nodes from the previous
        search. Typically 20–40% wall-time reduction at depth 14+ on long
        lines. Caller is responsible for the semantic correctness — only
        use when the previous call's FEN is the predecessor of the current.
        """
        async with self._lock:
            await self._ensure_connected()
            try:
                if not reuse_tt:
                    await self._send("ucinewgame")
                await self._send("isready")
                await self._wait_for("readyok", timeout=10)
                if multipv != self._current_multipv:
                    await self._send(f"setoption name MultiPV value {multipv}")
                    self._current_multipv = multipv
                    self._applied_options["MultiPV"] = multipv
                await self._send(f"position fen {fen}")
                cmd = f"go depth {depth}"
                if searchmoves:
                    cmd += f" searchmoves {' '.join(searchmoves)}"
                await self._send(cmd)

                pv_lines: list[dict[str, Any]] = []
                best_move: str | None = None

                while True:
                    line = await self._readline(timeout=120)
                    if line is None:
                        raise UCIError(f"Engine {self.name} disconnected during analysis")
                    if line.startswith("info "):
                        parsed = self._parse_info(line)
                        if "pv" in parsed:
                            pv_idx: int = parsed.get("multipv", 1)  # type: ignore[assignment]
                            while len(pv_lines) < pv_idx:
                                pv_lines.append({})
                            pv_lines[pv_idx - 1] = parsed
                    elif line.startswith("bestmove"):
                        parts = line.split()
                        best_move = parts[1] if len(parts) > 1 else None
                        break

                if not pv_lines and best_move and best_move != "(none)":
                    pv_lines = [{"pv": [best_move]}]

                return pv_lines
            except BaseException:
                await self._reset_connection()
                raise

    async def raw_uci_command(self, command: str) -> list[str]:
        """Send a raw UCI command and collect all response lines until 'readyok' or 'bestmove'."""
        async with self._lock:
            await self._ensure_connected()
            try:
                await self._send(command)
                lines: list[str] = []
                while True:
                    line = await self._readline(timeout=60)
                    if line is None:
                        raise UCIError(f"Engine {self.name} disconnected")
                    lines.append(line)
                    if line == "readyok" or line.startswith("bestmove"):
                        break
                return lines
            except BaseException:
                await self._reset_connection()
                raise

    async def close(self) -> None:
        if self._writer:
            try:
                self._send_no_lock("quit")
            except (ConnectionError, OSError):
                pass
            self._writer.close()

    async def _send(self, msg: str) -> None:
        logger.debug("→ %s: %s", self.name, msg)
        assert self._writer is not None
        self._writer.write((msg + "\n").encode())
        await self._writer.drain()

    def _send_no_lock(self, msg: str) -> None:
        if self._writer:
            self._writer.write((msg + "\n").encode())

    async def _readline(self, timeout: float = 30) -> str | None:
        assert self._reader is not None
        try:
            data = await asyncio.wait_for(self._reader.readline(), timeout=timeout)
        except TimeoutError as exc:
            raise UCIError(f"Timeout reading from {self.name}") from exc
        if not data:
            return None
        return data.decode().strip()

    async def _wait_for(self, token: str, timeout: float = 10) -> str:
        while True:
            line = await self._readline(timeout=timeout)
            if line is None:
                raise UCIError(f"Engine {self.name} disconnected while waiting for {token!r}")
            if line.startswith("id name "):
                self.name = line.removeprefix("id name ").strip()
            if token in line:
                return line

    @staticmethod
    def _parse_info(line: str) -> dict[str, Any]:
        parts = line.split()
        info: dict[str, Any] = {}
        i = 0
        while i < len(parts):
            key = parts[i]
            if key == "info":
                i += 1
                continue
            if key == "depth" and i + 1 < len(parts):
                info["depth"] = int(parts[i + 1])
                i += 2
            elif key == "multipv" and i + 1 < len(parts):
                info["multipv"] = int(parts[i + 1])
                i += 2
            elif key == "score" and i + 2 < len(parts):
                if parts[i + 1] == "cp":
                    info["cp"] = int(parts[i + 2])
                elif parts[i + 1] == "mate":
                    info["mate"] = int(parts[i + 2])
                i += 3
            elif key == "wdl" and i + 3 < len(parts):
                # Stockfish 18+ WDL in per-mille (W D L). Only present when
                # UCI_ShowWDL=true is set on the engine.
                try:
                    info["wdl"] = (int(parts[i + 1]), int(parts[i + 2]), int(parts[i + 3]))
                except ValueError:
                    pass
                i += 4
            elif key == "pv":
                info["pv"] = parts[i + 1 :]
                break
            else:
                i += 1
        return info
