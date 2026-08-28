"""No-op usage accounting for the standalone chess-mcp server.

The MCP exposes Stockfish tools anonymously (no per-account billing on this
service), so we keep the import surface but make every operation a no-op.
The chessy app has its own `core.usage` that records per-account usage; that
is the source of truth there — MCP just needs to not crash when the
engine pool calls `usage.count("stockfish")`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Literal

log = logging.getLogger("chessy_mcp.usage")

UsageKind = Literal["llm", "stockfish", "clara"]


@dataclass
class UsageEvent:
    account_id: str | None
    feature: str
    kind: UsageKind
    model: str | None = None
    calls: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


Recorder = Callable[[UsageEvent], Awaitable[None]]

_recorder: Recorder | None = None


@dataclass
class _Tally:
    account_id: str | None
    feature: str
    counts: dict[UsageKind, int] = field(default_factory=dict)


_tally: ContextVar[_Tally | None] = ContextVar("chessy_mcp_usage_tally", default=None)


def set_usage_recorder(recorder: Recorder | None) -> None:
    global _recorder
    _recorder = recorder


def count(kind: UsageKind, n: int = 1) -> None:
    """Bump an in-context counter. No-op in MCP — caller is anonymous."""


async def record_llm(*, model: str, input_tokens: int, output_tokens: int) -> None:
    pass


async def _record(event: UsageEvent) -> None:
    if _recorder is not None:
        try:
            await _recorder(event)
        except Exception as exc:
            log.warning("usage recorder failed: %s", exc)


@asynccontextmanager
async def usage_context(
    feature: str, account_id: str | None = None
) -> AsyncIterator[None]:
    yield
