from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from broker.server import EngineBroker


FAKE_ENGINE = """
import select
import sys
import time

for raw in sys.stdin:
    command = raw.strip()
    if command == "uci":
        print("id name Fakefish", flush=True)
        print("option name Threads type spin default 1 min 1 max 16", flush=True)
        print("uciok", flush=True)
    elif command == "isready":
        print("readyok", flush=True)
    elif command.startswith("go "):
        deadline = time.monotonic() + (1.0 if "depth 99" in command else 0.05)
        while time.monotonic() < deadline:
            readable, _, _ = select.select([sys.stdin], [], [], 0.01)
            if readable and sys.stdin.readline().strip() == "stop":
                break
        print("info depth 1 score cp 17 pv e2e4", flush=True)
        print("bestmove e2e4", flush=True)
"""


async def _connect(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"uci\n")
    await writer.drain()
    while (
        line := (await asyncio.wait_for(reader.readline(), timeout=1)).decode().strip()
    ) != "uciok":
        assert line
    return reader, writer


async def _read_until(reader: asyncio.StreamReader, prefix: str) -> list[str]:
    lines: list[str] = []
    while True:
        line = (await asyncio.wait_for(reader.readline(), timeout=2)).decode().strip()
        lines.append(line)
        if line.startswith(prefix):
            return lines


@pytest.mark.asyncio
async def test_broker_serializes_clients_and_preempts_background(tmp_path: Path) -> None:
    engine = tmp_path / "fake_uci.py"
    engine.write_text(FAKE_ENGINE)
    broker = EngineBroker((sys.executable, "-u", str(engine)), port=0)
    await broker.start()
    assert broker.engine_pid is not None

    background_reader, background_writer = await _connect(broker.port)
    interactive_reader, interactive_writer = await _connect(broker.port)
    try:
        background_writer.write(
            b"setoption name BrokerPriority value background\nposition startpos\ngo depth 99\n"
        )
        await background_writer.drain()
        await asyncio.sleep(0.05)

        interactive_writer.write(
            b"setoption name BrokerPriority value interactive\nposition startpos\ngo depth 1\n"
        )
        await interactive_writer.drain()

        background_lines = await _read_until(background_reader, "bestmove")
        interactive_lines = await _read_until(interactive_reader, "bestmove")
        assert any(line.startswith("info ") for line in background_lines)
        assert any(line.startswith("info ") for line in interactive_lines)
        assert broker.engine_pid is not None
    finally:
        background_writer.close()
        interactive_writer.close()
        await background_writer.wait_closed()
        await interactive_writer.wait_closed()
        await broker.close()
