from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field


_DEFAULT_OPTIONS: dict[str, str] = {
    "Threads": "6",
    "Hash": "1024",
    "MultiPV": "1",
    "UCI_LimitStrength": "false",
    "UCI_Elo": "2850",
    "Skill Level": "20",
    "UCI_ShowWDL": "true",
}


@dataclass(eq=False)
class ClientSession:
    writer: asyncio.StreamWriter
    options: dict[str, str] = field(default_factory=dict)
    position: str = "position startpos"
    priority: str = "interactive"
    new_game: bool = True
    search_task: asyncio.Task[None] | None = None
    closed: bool = False
    write_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send(self, line: str) -> None:
        if self.closed or self.writer.is_closing():
            return
        async with self.write_lock:
            self.writer.write(f"{line}\n".encode())
            await self.writer.drain()


class EngineBroker:
    """Expose one Stockfish subprocess as serialized UCI client sessions."""

    def __init__(
        self,
        command: Sequence[str] = ("/usr/local/bin/stockfish",),
        *,
        host: str = "0.0.0.0",
        port: int = 9550,
    ) -> None:
        self._command = tuple(command)
        self._host = host
        self._requested_port = port
        self._server: asyncio.Server | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._engine_reader: asyncio.StreamReader | None = None
        self._engine_writer: asyncio.StreamWriter | None = None
        self._uci_banner: list[str] = []
        self._engine_lock = asyncio.Lock()
        self._clients: set[ClientSession] = set()
        self._active_session: ClientSession | None = None

    @property
    def port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self._requested_port
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def engine_pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    async def start(self) -> None:
        await self._start_engine()
        self._server = await asyncio.start_server(
            self._serve_client, self._host, self._requested_port
        )

    async def close(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        for session in tuple(self._clients):
            await self._disconnect(session)
        await self._stop_engine()

    async def _start_engine(self) -> None:
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._engine_writer = self._process.stdin
        self._engine_reader = self._process.stdout
        await self._send_engine("uci")
        self._uci_banner = await self._read_until("uciok")

    async def _stop_engine(self) -> None:
        process = self._process
        self._process = None
        self._engine_reader = None
        self._engine_writer = None
        if process is None:
            return
        if process.returncode is None:
            with contextlib.suppress(BrokenPipeError, ConnectionError):
                process.stdin.write(b"quit\n") if process.stdin is not None else None
                if process.stdin is not None:
                    await process.stdin.drain()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except TimeoutError:
                process.terminate()
                await process.wait()

    async def _serve_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        session = ClientSession(writer=writer)
        self._clients.add(session)
        try:
            while not reader.at_eof():
                raw = await reader.readline()
                if not raw or not await self._handle_command(session, raw.decode().strip()):
                    break
        finally:
            await self._disconnect(session)

    async def _handle_command(self, session: ClientSession, command: str) -> bool:
        if command == "uci":
            for line in self._uci_banner:
                if line != "uciok":
                    await session.send(line)
            await session.send(
                "option name BrokerPriority type combo default interactive var interactive var background"
            )
            await session.send("uciok")
            return True
        if command == "isready":
            await session.send("readyok")
            return True
        if command == "ucinewgame":
            session.new_game = True
            return True
        if command.startswith("setoption "):
            self._set_option(session, command)
            return True
        if command.startswith("position "):
            session.position = command
            return True
        if command.startswith("go "):
            if session.search_task is not None and not session.search_task.done():
                await session.send(
                    "info string broker already has an active search for this client"
                )
            else:
                session.search_task = asyncio.create_task(self._run_search(session, command))
                await self._preempt_background_if_needed(session)
            return True
        if command == "stop":
            await self._stop_session_search(session)
            return True
        return command != "quit"

    @staticmethod
    def _set_option(session: ClientSession, command: str) -> None:
        payload = command.removeprefix("setoption ")
        if not payload.startswith("name "):
            return
        name_and_value = payload.removeprefix("name ").split(" value ", maxsplit=1)
        name = name_and_value[0].strip()
        value = name_and_value[1].strip() if len(name_and_value) == 2 else "true"
        if name == "BrokerPriority":
            session.priority = value if value in {"interactive", "background"} else "interactive"
        elif name:
            session.options[name] = value

    async def _preempt_background_if_needed(self, requester: ClientSession) -> None:
        active = self._active_session
        if (
            requester.priority == "interactive"
            and active is not None
            and active.priority == "background"
        ):
            await self._send_engine("stop")

    async def _stop_session_search(self, session: ClientSession) -> None:
        task = session.search_task
        if task is None or task.done():
            return
        if self._active_session is session:
            await self._send_engine("stop")
        else:
            task.cancel()

    async def _run_search(self, session: ClientSession, go_command: str) -> None:
        try:
            async with self._engine_lock:
                if session.closed:
                    return
                self._active_session = session
                await self._configure_engine(session)
                if session.new_game:
                    await self._send_engine("ucinewgame")
                    session.new_game = False
                await self._send_engine(session.position)
                await self._send_engine(go_command)
                while True:
                    line = await self._read_engine_line(timeout=180)
                    await session.send(line)
                    if line.startswith("bestmove"):
                        return
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionError, TimeoutError) as exc:
            await session.send(f"info string broker engine error: {exc}")
        finally:
            if self._active_session is session:
                self._active_session = None

    async def _configure_engine(self, session: ClientSession) -> None:
        options = {**_DEFAULT_OPTIONS, **session.options}
        for name, value in options.items():
            await self._send_engine(f"setoption name {name} value {value}")
        await self._send_engine("isready")
        await self._read_until("readyok")

    async def _disconnect(self, session: ClientSession) -> None:
        if session.closed:
            return
        session.closed = True
        self._clients.discard(session)
        await self._stop_session_search(session)
        task = session.search_task
        if task is not None and task is not asyncio.current_task():
            with contextlib.suppress(asyncio.CancelledError, BrokenPipeError, ConnectionError):
                await task
        session.writer.close()
        with contextlib.suppress(ConnectionError):
            await session.writer.wait_closed()

    async def _send_engine(self, command: str) -> None:
        writer = self._engine_writer
        if writer is None or writer.is_closing():
            raise ConnectionError("Stockfish process is unavailable")
        writer.write(f"{command}\n".encode())
        await writer.drain()

    async def _read_until(self, prefix: str) -> list[str]:
        lines: list[str] = []
        while True:
            line = await self._read_engine_line(timeout=10)
            lines.append(line)
            if line.startswith(prefix):
                return lines

    async def _read_engine_line(self, *, timeout: float) -> str:
        reader = self._engine_reader
        if reader is None:
            raise ConnectionError("Stockfish process is unavailable")
        data = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not data:
            raise ConnectionError("Stockfish process exited")
        return data.decode().strip()


async def _main() -> None:
    broker = EngineBroker()
    await broker.start()
    try:
        await asyncio.Future()
    finally:
        await broker.close()


if __name__ == "__main__":
    asyncio.run(_main())
