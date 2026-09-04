"""P3 regression tests: silent Unicode/control sanitization must warn.

Bug doc §14 — NUL / ZWSP / NBSP are silently stripped by parsers without
emitting a metadata_warning. After fix, at least one warning is surfaced
when these characters are removed.
"""

from __future__ import annotations

import pytest

from mcp_server import server as server_module


@pytest.fixture(autouse=True)
async def _close_analyzer_at_test_end():
    yield
    await server_module.close_analyzer_pool()


def test_nul_in_pgn_header_emits_warning():
    """§14.1: [White "A<NUL>B"] must produce a metadata_warning."""
    from mcp_server.parsers.pgn.extractor import extract_canonical_pgn_text

    pgn = '[White "A\x00B"]\n[Result "*"]\n*\n'
    warnings: list[str] = []
    cleaned = extract_canonical_pgn_text(pgn, warnings=warnings)
    assert "\x00" not in cleaned, "NUL must be stripped from cleaned text"
    assert any("NUL" in w for w in warnings), (
        f"expected metadata_warning naming stripped NUL; got warnings={warnings!r}"
    )


def test_zwsp_in_pgn_header_emits_warning():
    """§14.1: [White "A<ZWSP>B"] must produce a metadata_warning."""
    from mcp_server.parsers.pgn.extractor import extract_canonical_pgn_text

    pgn = '[White "A\u200bB"]\n[Result "*"]\n*\n'
    warnings: list[str] = []
    cleaned = extract_canonical_pgn_text(pgn, warnings=warnings)
    assert "\u200b" not in cleaned, "ZWSP must be stripped from cleaned text"
    assert any("ZWSP" in w for w in warnings), (
        f"expected metadata_warning naming stripped ZWSP; got warnings={warnings!r}"
    )
