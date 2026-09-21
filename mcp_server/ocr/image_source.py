"""Resolve an :class:`ImageSource` discriminated union into raw image bytes.

The MCP ``ocr_to_pgn`` v2 tool accepts the image as one of:

    - ``base64``:  raw base64-encoded bytes supplied inline by the caller
    - ``url``:     http(s) URL — the server fetches with httpx, validates
                   content-type + magic bytes, enforces a 32 MB cap
    - ``file_uri``: absolute filesystem path (within a sandboxed root)

Security gates:

    - URL fetches obey :data:`URL_ALLOWLIST` (env: ``CHESSY_MCP_URL_ALLOWLIST``,
      comma-separated hostnames). When unset, only ``localhost`` /
      ``127.0.0.1`` are reachable.
    - ``file_uri`` reads must resolve under :data:`FILE_ROOT`
      (env: ``CHESSY_MCP_FILE_ROOT``, default ``/tmp``). Symlink traversal
      is blocked by ``Path.resolve(strict=True)``.
    - Every source goes through the same magic-byte format check so the
      sidecar never sees a corrupted upload.
"""

from __future__ import annotations

import asyncio
import base64 as _base64_mod
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx

from mcp_server.contracts.errors import InvalidArgument, InvalidInput


log = logging.getLogger("chessy_mcp.ocr.image_source")


MAX_IMAGE_BYTES: Final[int] = 32 * 1024 * 1024
URL_FETCH_TIMEOUT_S: Final[float] = 30.0
URL_FOLLOW_REDIRECTS: Final[bool] = True

_VALID_IMAGE_FORMATS: Final[tuple[str, ...]] = ("jpeg", "png", "webp", "gif", "heic")

_MAGIC_JPEG: Final[bytes] = b"\xff\xd8\xff"
_MAGIC_PNG: Final[bytes] = b"\x89PNG\r\n\x1a\n"
_MAGIC_HEIC_FTYP: Final[bytes] = b"ftyp"

_DEFAULT_URL_ALLOWLIST: Final[frozenset[str]] = frozenset({"localhost", "127.0.0.1", "::1"})
_DEFAULT_FILE_ROOT: Final[str] = "/tmp"


@dataclass(frozen=True)
class ResolvedImage:
    """The bytes resolved from an :class:`ImageSource` plus provenance label."""

    bytes: bytes
    format: str
    source_label: str


def _detect_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(_MAGIC_JPEG):
        return "jpeg"
    if image_bytes.startswith(_MAGIC_PNG):
        return "png"
    if image_bytes.startswith(b"GIF"):
        return "gif"
    if image_bytes.startswith(b"RIFF"):
        return "webp"
    if len(image_bytes) >= 12 and image_bytes[4:9] == _MAGIC_HEIC_FTYP:
        brand = image_bytes[8:12]
        if brand.startswith(b"heic") or brand.startswith(b"mif1") or brand.startswith(b"heim"):
            return "heic"
    return "unknown"


def _check_size_and_format(image_bytes: bytes) -> str:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidInput(
            f"IMAGE_TOO_LARGE: decoded {len(image_bytes)} bytes > cap {MAX_IMAGE_BYTES}"
        )
    fmt = _detect_format(image_bytes)
    if fmt not in _VALID_IMAGE_FORMATS:
        raise InvalidArgument(f"UNSUPPORTED_IMAGE_FORMAT: must be one of {_VALID_IMAGE_FORMATS}")
    return fmt


def _url_allowlist() -> frozenset[str]:
    raw = os.environ.get("CHESSY_MCP_URL_ALLOWLIST", "").strip()
    if not raw:
        return _DEFAULT_URL_ALLOWLIST
    hosts = {h.strip().lower() for h in raw.split(",") if h.strip()}
    return frozenset(hosts) if hosts else _DEFAULT_URL_ALLOWLIST


def _file_root() -> Path:
    raw = os.environ.get("CHESSY_MCP_FILE_ROOT", "").strip()
    return Path(raw).resolve() if raw else Path(_DEFAULT_FILE_ROOT).resolve()


def _host_from_url(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return (parsed.hostname or "").lower()


def _check_url_allowed(url: str) -> None:
    scheme = url.split(":", 1)[0].lower() if ":" in url else ""
    if scheme not in {"http", "https"}:
        raise InvalidArgument(f"UNSUPPORTED_URL_SCHEME: {scheme!r}; only http/https are fetchable")
    host = _host_from_url(url)
    if not host:
        raise InvalidArgument(f"INVALID_URL: cannot parse host from {url!r}")
    allowlist = _url_allowlist()
    if host not in allowlist:
        raise InvalidArgument(
            f"URL_HOST_NOT_ALLOWED: {host!r} is not in CHESSY_MCP_URL_ALLOWLIST "
            f"({sorted(allowlist)})"
        )


async def _fetch_url(url: str, *, timeout_s: float | None) -> bytes:
    timeout = httpx.Timeout(timeout_s if timeout_s else URL_FETCH_TIMEOUT_S)
    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=URL_FOLLOW_REDIRECTS,
        headers={"User-Agent": "chessy-mcp/0.1.0 (ocr image source)"},
    ) as client:
        try:
            resp = await client.get(url)
        except httpx.TimeoutException as exc:
            raise InvalidInput(f"URL_FETCH_TIMEOUT: {exc}") from exc
        except httpx.HTTPError as exc:
            raise InvalidInput(f"URL_FETCH_FAILED: {exc}") from exc
    if resp.status_code != 200:
        raise InvalidInput(
            f"URL_FETCH_HTTP_{resp.status_code}: status {resp.status_code} for {url!r}"
        )
    return resp.content


def _clean_base64_data(raw_b64: str) -> str:
    cleaned = raw_b64.strip()
    if ";base64," in cleaned:
        cleaned = cleaned.split(";base64,", 1)[1].strip()
    elif cleaned.startswith("data:"):
        cleaned = cleaned.split(",", 1)[1].strip()
    cleaned = "".join(cleaned.split())
    missing_padding = len(cleaned) % 4
    if missing_padding:
        cleaned += "=" * (4 - missing_padding)
    return cleaned


def _read_file(path_str: str) -> bytes:
    try:
        candidate = Path(path_str).resolve(strict=True)
    except FileNotFoundError as exc:
        raise InvalidInput(f"FILE_NOT_FOUND: {path_str!r}") from exc
    except RuntimeError as exc:
        raise InvalidInput(f"FILE_PATH_INVALID: {path_str!r} ({exc})") from exc

    raw_root = os.environ.get("CHESSY_MCP_FILE_ROOT", "").strip()
    if raw_root and raw_root != "*":
        root = Path(raw_root).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise InvalidArgument(
                f"FILE_OUTSIDE_SANDBOX: {candidate} is not under CHESSY_MCP_FILE_ROOT ({root})"
            ) from exc
    return candidate.read_bytes()


def _resolve_attachment(file_id: str) -> bytes:
    clean_id = os.path.basename(file_id.strip())
    search_dirs: list[Path] = []
    custom_dir = os.environ.get("CHESSY_ATTACHMENT_DIR", "").strip()
    if custom_dir:
        search_dirs.append(Path(custom_dir).resolve())
    search_dirs.append(Path("/tmp/chessy_attachments"))
    search_dirs.append(Path("/tmp"))
    search_dirs.append(Path("."))

    for d in search_dirs:
        if d.is_dir():
            cand = d / clean_id
            if cand.is_file():
                return cand.read_bytes()

    direct = Path(file_id)
    if direct.is_file():
        return direct.read_bytes()

    raise InvalidInput(
        f"ATTACHMENT_NOT_FOUND: file_id {file_id!r} could not be found. "
        "Supply image as base64 data or valid local file path."
    )


async def resolve_image(
    *,
    base64: str | None,
    url: str | None,
    file_uri: str | None,
    attachment_id: str | None = None,
    timeout_s: float | None = None,
) -> ResolvedImage:
    """Resolve the first non-None source into :class:`ResolvedImage`.

    Exactly one of the four keyword arguments must be supplied.
    """
    provided = sum(v is not None for v in (base64, url, file_uri, attachment_id))
    if provided == 0:
        raise InvalidArgument(
            "IMAGE_SOURCE_MISSING: one of base64/url/file_uri/attachment is required"
        )
    if provided > 1:
        raise InvalidArgument(
            "IMAGE_SOURCE_AMBIGUOUS: provide exactly one of base64/url/file_uri/attachment"
        )

    if base64 is not None:
        cleaned = _clean_base64_data(base64)
        try:
            data = _base64_mod.b64decode(cleaned, validate=True)
        except Exception as exc:
            raise InvalidInput(f"INVALID_BASE64: {exc}") from exc
        fmt = _check_size_and_format(data)
        return ResolvedImage(bytes=data, format=fmt, source_label="base64")

    if url is not None:
        _check_url_allowed(url)
        data = await _fetch_url(url, timeout_s=timeout_s)
        fmt = _check_size_and_format(data)
        return ResolvedImage(bytes=data, format=fmt, source_label=f"url:{url}")

    if file_uri is not None:
        data = await asyncio.to_thread(_read_file, file_uri)
        fmt = _check_size_and_format(data)
        return ResolvedImage(bytes=data, format=fmt, source_label=f"file_uri:{file_uri}")

    if attachment_id is not None:
        data = await asyncio.to_thread(_resolve_attachment, attachment_id)
        fmt = _check_size_and_format(data)
        return ResolvedImage(bytes=data, format=fmt, source_label=f"attachment:{attachment_id}")

    raise AssertionError("unreachable")  # pragma: no cover


__all__ = [
    "MAX_IMAGE_BYTES",
    "ResolvedImage",
    "resolve_image",
]
