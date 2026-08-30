from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one occurrence, got {count}")
    return text.replace(old, new, 1)


# 1. Transport security settings.
p = Path("mcp_server/config.py")
s = p.read_text(encoding="utf-8")
marker = "CHESS_MCP_DNS_REBINDING_PROTECTION"
if marker not in s:
    old = '''    # When true, every non-health HTTP request must present the configured
    # token. User-Agent and Origin strings are never treated as credentials.
    lock_chatgpt: bool = Field(default=False, validation_alias="CHESS_MCP_LOCK_CHATGPT")
'''
    new = old + '''
    # MCP transport DNS-rebinding protection. Enabled by default for HTTP.
    # Host/Origin allowlists use the exact and `host:*` wildcard-port semantics
    # implemented by the official MCP Python SDK.
    dns_rebinding_protection: bool = Field(
        default=True, validation_alias="CHESS_MCP_DNS_REBINDING_PROTECTION"
    )
    allowed_hosts: str = Field(
        default="localhost:*,127.0.0.1:*,[::1]:*,testserver,testserver:*",
        validation_alias="CHESS_MCP_ALLOWED_HOSTS",
    )
    allowed_origins: str = Field(
        default="", validation_alias="CHESS_MCP_ALLOWED_ORIGINS"
    )
'''
    s = replace_once(s, old, new, "config transport security")
    p.write_text(s, encoding="utf-8")


# 2. Server: DNS rebinding, trailing-PGN contract, nested candidate identity.
p = Path("mcp_server/server.py")
s = p.read_text(encoding="utf-8")

old = "def _parse_pgn_game_candidate(text: str, strict: bool = False) -> chess.pgn.Game | None:\n"
new = '''def _parse_pgn_game_candidate(
    text: str,
    strict: bool = False,
    allow_trailing_after_terminal: bool = False,
) -> chess.pgn.Game | None:
'''
if old in s:
    s = replace_once(s, old, new, "parse candidate signature")

old = '''            if game.errors:
                b = game.board()
                reached_game_over = False
                for node in game.mainline():
                    b.push(node.move)
                    if b.is_game_over(claim_draw=False):
                        reached_game_over = True
                        break
                if not reached_game_over:
                    raise ValueError(
                        f"Invalid PGN syntax or illegal move in game: {game.errors[0]}"
                    )
'''
new = '''            if game.errors:
                b = game.board()
                reached_game_over = False
                for node in game.mainline():
                    b.push(node.move)
                    if b.is_game_over(claim_draw=False):
                        reached_game_over = True
                        break
                if reached_game_over:
                    if not allow_trailing_after_terminal:
                        raise ValueError(
                            "INVALID_PGN: Movetext contains moves after automatic game termination."
                        )
                else:
                    raise ValueError(
                        f"INVALID_PGN: Invalid PGN syntax or illegal move in game: {game.errors[0]}"
                    )
'''
if old in s:
    s = replace_once(s, old, new, "game.errors trailing terminal")

old = '''def _extract_game(text: str, strict: bool = False) -> chess.pgn.Game:
    """Extract a chess.pgn.Game from raw, dirty, annotated, or conversational text."""
    _check_multiple_games(text)
    canonical = _extract_canonical_pgn_text(text)
    return _extract_game_inner(canonical, strict=strict)


def _extract_game_inner(cleaned: str, strict: bool = False) -> chess.pgn.Game:
'''
new = '''def _extract_game(
    text: str,
    strict: bool = False,
    allow_trailing_after_terminal: bool = False,
) -> chess.pgn.Game:
    """Extract one game and reject hidden post-terminal moves by default."""
    _check_multiple_games(text)
    canonical = _extract_canonical_pgn_text(text)
    return _extract_game_inner(
        canonical,
        strict=strict,
        allow_trailing_after_terminal=allow_trailing_after_terminal,
    )


def _extract_game_inner(
    cleaned: str,
    strict: bool = False,
    allow_trailing_after_terminal: bool = False,
) -> chess.pgn.Game:
'''
if old in s:
    s = replace_once(s, old, new, "extract signatures")

s = s.replace(
    "_parse_pgn_game_candidate(norm_text, strict=strict)",
    '''_parse_pgn_game_candidate(
            norm_text,
            strict=strict,
            allow_trailing_after_terminal=allow_trailing_after_terminal,
        )''',
)
s = s.replace(
    "_parse_pgn_game_candidate(sub_movetext, strict=strict)",
    '''_parse_pgn_game_candidate(
                sub_movetext,
                strict=strict,
                allow_trailing_after_terminal=allow_trailing_after_terminal,
            )''',
)

if "        game = _extract_game(cleaned)\n" in s:
    s = replace_once(
        s,
        "        game = _extract_game(cleaned)\n",
        "        game = _extract_game(cleaned, strict=strict)\n",
        "build_board strict forwarding",
    )

if "        game = _extract_game_inner(canonical_pgn)\n" in s:
    s = replace_once(
        s,
        "        game = _extract_game_inner(canonical_pgn)\n",
        '''        game = _extract_game_inner(
            canonical_pgn,
            strict=strict,
            allow_trailing_after_terminal=not strict,
        )
''',
        "analyze_game parser contract",
    )

identity_needle = '''                        "recommended_action": "game_over"
                        if cand_post_terminal is not None
                        else "play_move",
                        "post_position": {
'''
identity_repl = '''                        "recommended_action": "game_over"
                        if cand_post_terminal is not None
                        else "play_move",
                        "build_sha": _build_sha(),
                        "engine_config": _engine_config(pool),
                        "post_position": {
'''
if identity_needle in s:
    s = replace_once(s, identity_needle, identity_repl, "candidate identity")

security_old = '''    from starlette.middleware.gzip import GZipMiddleware

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    base = mcp.streamable_http_app(transport_security=security)
'''
security_new = '''    from starlette.middleware.gzip import GZipMiddleware
    from .config import get_mcp_settings

    cfg = get_mcp_settings()
    allowed_hosts = [item.strip() for item in cfg.allowed_hosts.split(",") if item.strip()]
    allowed_origins = [item.strip() for item in cfg.allowed_origins.split(",") if item.strip()]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=cfg.dns_rebinding_protection,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )
    base = mcp.streamable_http_app(transport_security=security)
'''
if security_old in s:
    s = replace_once(s, security_old, security_new, "DNS rebinding settings")

p.write_text(s, encoding="utf-8")


# 3. Production/development settings.
p = Path(".env.example")
s = p.read_text(encoding="utf-8")
if "CHESS_MCP_DNS_REBINDING_PROTECTION" not in s:
    insert = '''
# MCP HTTP Host/Origin validation. Keep enabled on every public deployment.
CHESS_MCP_DNS_REBINDING_PROTECTION=true
CHESS_MCP_ALLOWED_HOSTS=localhost:*,127.0.0.1:*,[::1]:*,testserver,testserver:*
# Origin may be absent for server-to-server clients. If present it must match.
CHESS_MCP_ALLOWED_ORIGINS=http://localhost:*,http://127.0.0.1:*
'''
    anchor = "CHESS_MCP_LOCK_CHATGPT=true\n"
    s = replace_once(s, anchor, anchor + insert, ".env security")
    p.write_text(s, encoding="utf-8")

p = Path("docker-compose.prod.yml")
s = p.read_text(encoding="utf-8")
if "CHESS_MCP_DNS_REBINDING_PROTECTION" not in s:
    anchor = "      - CHESS_MCP_LOCK_CHATGPT=true\n"
    add = '''      - CHESS_MCP_DNS_REBINDING_PROTECTION=true
      - CHESS_MCP_ALLOWED_HOSTS=mcp.trychessy.com,mcp.trychessy.com:*,mcp,mcp:8000,localhost:*,127.0.0.1:*
      - CHESS_MCP_ALLOWED_ORIGINS=https://chatgpt.com,https://chat.openai.com,https://platform.openai.com,http://mcp.trychessy.com,https://mcp.trychessy.com
'''
    s = replace_once(s, anchor, anchor + add, "compose security")
    p.write_text(s, encoding="utf-8")


# 4. Remove stale operational claims from Caddy/README.
p = Path("Caddyfile")
s = p.read_text(encoding="utf-8")
old = '''#  * flush_interval -1    — flush each SSE chunk to the client immediately.
#                           Critical for multi-position analyze_game responses
#                           where the client wants incremental progress, not a
#                           75s buffer until the last move completes.
'''
new = '''#  * flush_interval -1    - avoid proxy buffering for streamable HTTP/SSE
#                           traffic. analyze_game currently returns one completed
#                           tool result; this does not imply per-position progress.
'''
if old in s:
    s = replace_once(s, old, new, "Caddy progress comment")
    p.write_text(s, encoding="utf-8")

p = Path("README.md")
s = p.read_text(encoding="utf-8")
old = '''- **Stockfish pool size** defaults to `min(cpu_count, 8)` — sized to the
  OVH production box (8 cores Xeon E5-1620 v2).
'''
new = '''- **Stockfish pool size** defaults to `min(cpu_count, 4)` when not explicitly
  configured. Production pins 4 workers with 2 Stockfish threads each.
'''
if old in s:
    s = replace_once(s, old, new, "README pool default")
    p.write_text(s, encoding="utf-8")


# 5. Permanent regressions for residual v1-v3 findings.
p = Path("tests/test_ultra_stress_v5.py")
s = p.read_text(encoding="utf-8")
marker = "test_residual_dns_rebinding_settings_are_enforced"
if marker not in s:
    s += r'''

# Residual audit closure: findings present in v1-v3 but not explicit in the v4 41-test map.
def test_residual_dns_rebinding_settings_are_enforced(monkeypatch):
    from mcp.server.transport_security import TransportSecurityMiddleware
    from mcp_server.config import get_mcp_settings

    monkeypatch.setenv("CHESS_MCP_DNS_REBINDING_PROTECTION", "true")
    monkeypatch.setenv("CHESS_MCP_ALLOWED_HOSTS", "mcp.trychessy.com,localhost:*")
    monkeypatch.setenv("CHESS_MCP_ALLOWED_ORIGINS", "https://chatgpt.com")
    get_mcp_settings.cache_clear()
    try:
        cfg = get_mcp_settings()
        settings = server_module.TransportSecuritySettings(
            enable_dns_rebinding_protection=cfg.dns_rebinding_protection,
            allowed_hosts=[item.strip() for item in cfg.allowed_hosts.split(",") if item.strip()],
            allowed_origins=[item.strip() for item in cfg.allowed_origins.split(",") if item.strip()],
        )
        middleware = TransportSecurityMiddleware(settings)
        assert middleware._validate_host("mcp.trychessy.com") is True
        assert middleware._validate_host("localhost:8000") is True
        assert middleware._validate_host("attacker.example") is False
        assert middleware._validate_origin("https://chatgpt.com") is True
        assert middleware._validate_origin("https://evil.example") is False
    finally:
        get_mcp_settings.cache_clear()


@pytest.mark.asyncio
async def test_residual_position_tools_reject_moves_after_automatic_terminal():
    bad = "1. f3 e5 2. g4 Qh4# 3. e4"
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.evaluate_position(bad, depth=1)
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.top_moves(bad, n=1, depth=1)
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.classify_move(bad, "e4", depth=1)


@pytest.mark.asyncio
async def test_residual_analyze_game_warns_permissive_and_rejects_strict_trailing_move():
    bad = "1. f3 e5 2. g4 Qh4# 3. e4"
    permissive = await server_module.analyze_game(bad, depth=1, strict=False)
    assert permissive.total_plies == 4
    assert any("after game termination" in w for w in permissive.metadata_warnings)
    with pytest.raises(ToolError, match="INVALID_PGN"):
        await server_module.analyze_game(bad, depth=1, strict=True)


@pytest.mark.asyncio
async def test_residual_nested_candidates_share_outer_build_identity():
    result = await server_module.top_moves("startpos", n=3, depth=1)
    assert result.result
    for candidate in result.result:
        assert candidate.build_sha == result.build_sha
        assert candidate.engine_config == result.engine_config
    cached = await server_module.top_moves("startpos", n=3, depth=1)
    for candidate in cached.result:
        assert candidate.build_sha == cached.build_sha
        assert candidate.engine_config == cached.engine_config
'''
    p.write_text(s, encoding="utf-8")
