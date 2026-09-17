"""HTTP client for the Lichess Opening Explorer.

Endpoint surface (all OAuth-protected; the Bearer token is ``MCPSettings.lichess_explorer_token``):

    GET /{db}   db ∈ {"lichess", "masters", "player"}

``/lichess`` returns NDJSON (Lichess streams partial updates while the indexer warms up).
The final line carries the complete, cumulative payload — that is what we return.
``/masters`` and ``/player`` return a single JSON object.

The client owns a single ``httpx.AsyncClient`` (lazy, shared across calls). It retries
once on transient 5xx with a 100 ms backoff and raises typed exceptions that the
tool layer maps onto structured ``ToolError`` codes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Final

import httpx

from mcp_server.config import get_mcp_settings

__all__ = [
    "LICHESS_DBS",
    "LICHESS_DB_LICHESS",
    "LICHESS_DB_MASTERS",
    "LICHESS_DB_PLAYER",
    "LICHESS_EXPLORER_TIMEOUT_S",
    "LICHESS_VARIANTS",
    "LichessExplorerAuthError",
    "LichessExplorerClient",
    "LichessExplorerNotFound",
    "LichessExplorerRateLimited",
    "LichessExplorerResponseError",
    "LichessExplorerTimeout",
    "LichessExplorerUnavailable",
    "LichessExplorerUnreachable",
    "LichessVariant",
]


LICHESS_DB_LICHESS: Final = "lichess"
LICHESS_DB_MASTERS: Final = "masters"
LICHESS_DB_PLAYER: Final = "player"
LICHESS_DBS: Final[frozenset[str]] = frozenset(
    {LICHESS_DB_LICHESS, LICHESS_DB_MASTERS, LICHESS_DB_PLAYER}
)

# Matches lila's OpeningConfig + OpeningExplorer.requestTimeout.
LICHESS_EXPLORER_TIMEOUT_S: Final[float] = 4.0

LichessVariant = str
LICHESS_VARIANTS: Final[frozenset[str]] = frozenset(
    {
        "standard",
        "chess960",
        "fromPosition",
        "antichess",
        "atomic",
        "crazyhouse",
        "horde",
        "kingOfTheHill",
        "racingKings",
        "threeCheck",
    }
)

log = logging.getLogger("chessy_mcp.lichess_explorer")


class LichessExplorerError(Exception):
    """Base for every typed failure the tool layer maps onto a ToolError code."""


class LichessExplorerAuthError(LichessExplorerError):
    """401 — token missing, malformed, or revoked."""


class LichessExplorerRateLimited(LichessExplorerError):
    """429 — caller backs off; we do not auto-retry 429s."""


class LichessExplorerUnavailable(LichessExplorerError):
    """5xx — already retried once internally; surface to the caller."""


class LichessExplorerTimeout(LichessExplorerError):
    """4 s timeout exceeded."""


class LichessExplorerUnreachable(LichessExplorerError):
    """Network-level failure (DNS, connection refused, TLS, etc.)."""


class LichessExplorerNotFound(LichessExplorerError):
    """404 — typically a missing Lichess player when ``db=player``."""


class LichessExplorerResponseError(LichessExplorerError):
    """200 but the body is not parseable as JSON / NDJSON JSON."""


class LichessExplorerClient:
    """Shared async HTTP client for ``https://explorer.lichess.ovh``."""

    def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
        settings = get_mcp_settings()
        self._endpoint: str = (endpoint or settings.lichess_explorer_endpoint).rstrip("/")
        self._token: str = token if token is not None else settings.lichess_explorer_token
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def token(self) -> str:
        return self._token

    async def _http(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None:
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(LICHESS_EXPLORER_TIMEOUT_S),
                    headers={"Accept": "application/x-ndjson, application/json"},
                    http2=False,
                )
        return self._client

    async def aclose(self) -> None:
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def query(self, db: str, params: dict[str, Any]) -> dict[str, Any]:
        """Fire one Explorer request and return the parsed JSON body.

        ``params`` is the query string (already URL-encoded by httpx). NDJSON
        responses collapse to the last complete JSON object; non-NDJSON JSON
        parses as the single body.
        """
        if not self._token:
            raise LichessExplorerAuthError(
                "MISSING_TOKEN: LICHESS_EXPLORER_TOKEN is not configured."
            )
        url = f"{self._endpoint}/{db}"
        headers = {
            "Authorization": f"Bearer {self._token}",
            "User-Agent": "chessy-mcp/0.1.0 (lichess explorer client)",
        }
        last_exc: BaseException | None = None
        for attempt in (1, 2):
            try:
                client = await self._http()
                resp = await client.get(url, params=params, headers=headers)
            except httpx.TimeoutException as exc:
                raise LichessExplorerTimeout(
                    f"Lichess explorer timed out after {LICHESS_EXPLORER_TIMEOUT_S}s"
                ) from exc
            except httpx.HTTPError as exc:
                raise LichessExplorerUnreachable(str(exc)) from exc

            if resp.status_code == 200:
                return self._parse_body(resp)
            if resp.status_code == 401:
                raise LichessExplorerAuthError(
                    f"Lichess explorer returned 401: token rejected (attempt {attempt})"
                )
            if resp.status_code == 404:
                raise LichessExplorerNotFound(
                    f"Lichess explorer returned 404 for {db}: {resp.text[:200]}"
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                raise LichessExplorerRateLimited(
                    f"Lichess explorer returned 429 (Retry-After={retry_after})"
                )
            if 500 <= resp.status_code < 600:
                last_exc = LichessExplorerUnavailable(
                    f"Lichess explorer returned {resp.status_code} (attempt {attempt})"
                )
                if attempt == 1:
                    await asyncio.sleep(0.1)
                    continue
                raise last_exc
            raise LichessExplorerResponseError(
                f"Lichess explorer returned unexpected status {resp.status_code}: {resp.text[:200]}"
            )
        if last_exc is not None:
            raise last_exc
        raise LichessExplorerUnavailable("Lichess explorer retry loop exhausted")

    @staticmethod
    def _parse_body(resp: httpx.Response) -> dict[str, Any]:
        ctype = resp.headers.get("content-type", "")
        text = resp.text
        if "ndjson" in ctype:
            last: dict[str, Any] | None = None
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LichessExplorerResponseError(
                        f"NDJSON line failed to parse: {exc.msg}"
                    ) from exc
                if isinstance(obj, dict):
                    last = obj
            if last is None:
                raise LichessExplorerResponseError("NDJSON body contained no JSON objects")
            return last
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LichessExplorerResponseError(f"Body failed to parse as JSON: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise LichessExplorerResponseError("Body is not a JSON object")
        return obj
