from __future__ import annotations

import re
from pathlib import Path


PATH = Path("mcp_server/server.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, got {count}")
    return text.replace(old, new, 1)


def replace_block(text: str, start: str, end: str, replacement: str, label: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"{label}: start marker not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"{label}: end marker not found: {end!r}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


text = PATH.read_text(encoding="utf-8")

# Imports and high-signal typing defects.
text = text.replace("import io\n", "import io\nimport ipaddress\nimport json\n", 1)
text = text.replace("from typing import Any, cast", "from typing import Any, Literal, cast", 1)
text = text.replace("from core.engines.types import MoveClass", "from core.engines.types import Eval, MoveClass", 1)
text = text.replace(
    "from mcp_server.metrics import metrics\n",
    "from mcp_server.config import MCPSettings\nfrom mcp_server.metrics import metrics\n",
    1,
)
text = text.replace(
    "from mcp_server.rules import (\n    evaluate_rule_status,",
    "from mcp_server.rules import (\n    choose_recommended_action,\n    evaluate_rule_status,",
    1,
)
text = text.replace(
    'def _find_movetext_result(text: str) -> str:',
    'def _find_movetext_result(text: str) -> str | None:',
    1,
)
text = text.replace(
    'clean_tok if "clean_tok" in locals() else clean_t,',
    'clean_t,',
    1,
)
text = text.replace(
    'env_sha = os.environ.get("CHESSY_BUILD_SHA")',
    'env_sha = os.environ.get("BUILD_SHA") or os.environ.get("CHESSY_BUILD_SHA")',
    1,
)

verbosity_block = '''def _resolve_verbosity(value: Any) -> str:
    """Normalize verbosity and reject unknown values instead of silently changing semantics."""
    if value is None:
        return VERBOSITY_FULL
    normalized = str(value).strip().lower()
    resolved = _VERBOSITY_ALIASES.get(normalized)
    if resolved is None:
        raise ValueError(
            f"INVALID_VERBOSITY: expected one of {sorted(_VERBOSITY_ALIASES)}, got {value!r}"
        )
    return resolved


def _compact_mcpeval(mcp_eval: MCPEval) -> MCPEval:
    """Strip verbose payload duplication without rewriting chess semantics."""
    return mcp_eval.model_copy(
        update={
            "lichess_url": None,
            "lichess_image": None,
            "decision_value": None,
            "engine_eval": None,
            "input_fen": None,
        }
    )
'''
text = replace_block(text, "def _resolve_verbosity", "_FIGURINE_MAP =", verbosity_block, "verbosity semantics")

# History provenance is a property of the input source, not merely whether a
# suffix move list happens to be non-empty.
provenance_helper = '''def _history_provenance_for_input(
    fen_or_pgn: str,
    moves: list[str] | None,
) -> str:
    """Return complete, partial or incomplete history provenance for an input."""
    cleaned = (
        fen_or_pgn.replace("\\u00a0", " ")
        .replace("\\u200b", "")
        .replace("\\ufeff", "")
        .strip("`'\\\" \\t\\r\\n")
    )
    if cleaned.lower() in ("startpos", "initial", "start"):
        return "complete"

    tokens = cleaned.split()
    looks_like_fen = (
        "/" in cleaned
        and 1 <= len(tokens) <= 6
        and not cleaned.startswith("[")
        and not tokens[0].endswith(".")
    )
    if looks_like_fen:
        return "partial" if moves else "incomplete"

    # A PGN/movetext input defines its own game root and therefore carries the
    # complete history represented by that game. Additional suffix moves keep it complete.
    return "complete"
'''
anchor = "def _build_board_with_metadata("
idx = text.find(anchor)
if idx < 0:
    raise RuntimeError("history provenance insertion anchor missing")
text = text[:idx] + provenance_helper + "\n\n" + text[idx:]

# Preserve move-stack provenance in the ponder path. Never reconstruct a FEN-only
# board and label it as complete history.
ponder = '''async def _ponder_warm_cache(
    pool: AnalyzerPool | TCPAnalyzerPool,
    predicted_board: chess.Board,
    depth: int,
    history_complete: str,
) -> None:
    """Background cache warmer that preserves the predicted board's move stack."""
    try:
        board = predicted_board.copy(stack=True)
        if board.is_game_over(claim_draw=False):
            return
        ckey = eval_cache_key(
            board,
            depth,
            engine_version=getattr(pool, "engine_version", None),
            history_completeness=history_complete,
        )
        if (await _cache.get_eval(ckey)) is not None:
            return
        await _evaluate_game_position_cached(
            board,
            depth,
            pool,
            requested_depth=depth,
            history_complete=history_complete,
        )
    except Exception as exc:
        log.debug("ponder pre-eval failed: %s", exc)


def _maybe_ponder_warm(
    pool: AnalyzerPool | TCPAnalyzerPool,
    board: chess.Board,
    best_move_uci: str | None,
    depth: int,
    ponder_enabled: bool,
    history_complete: str,
) -> None:
    """Schedule a provenance-preserving background pre-evaluation."""
    if not ponder_enabled or not best_move_uci:
        return
    try:
        next_board = board.copy(stack=True)
        next_board.push_uci(best_move_uci)
        if next_board.is_game_over(claim_draw=False):
            return
        asyncio.create_task(
            _ponder_warm_cache(pool, next_board, depth, history_complete),
            name="ponder-warm",
        )
    except Exception:
        pass
'''
text = replace_block(text, "async def _ponder_warm_cache", "mcp = MCPServer(", ponder, "ponder provenance")

# Cached rule projection is history-sensitive. The cache key must include the
# epistemic state and the helper must fail safe when provenance is omitted.
text = text.replace(
    "    history_complete: bool = True,",
    '    history_complete: str | bool = "incomplete",',
    1,
)
text = text.replace(
    "    req_d = requested_depth if requested_depth is not None else depth\n    canonical_fen_str = b.fen()",
    "    req_d = requested_depth if requested_depth is not None else depth\n    history_state = (\n        (\"complete\" if history_complete else \"incomplete\")\n        if isinstance(history_complete, bool)\n        else history_complete\n    )\n    canonical_fen_str = b.fen()",
    1,
)
text = text.replace(
    "rule_status = evaluate_rule_status(b, history_complete=history_complete)",
    "rule_status = evaluate_rule_status(b, history_complete=history_state)",
    1,
)
text = text.replace(
    "        engine_version=getattr(pool, \"engine_version\", None),\n    )\n    cached = await _cache.get_eval(ckey)",
    "        engine_version=getattr(pool, \"engine_version\", None),\n        history_completeness=history_state,\n    )\n    cached = await _cache.get_eval(ckey)",
    1,
)
text = text.replace(
    "            history_complete=history_complete,\n        )\n        # Stamp build identity",
    "            history_complete=history_state,\n        )\n        # Stamp build identity",
    1,
)
text = text.replace(
    "            ponder_enabled=getattr(pool, \"_mcp_ponder_enabled\", False),\n        )",
    "            ponder_enabled=getattr(pool, \"_mcp_ponder_enabled\", False),\n            history_complete=history_state,\n        )",
    1,
)

# Public endpoint provenance.
text = text.replace(
    "        history_complete = bool(moves)\n        res, is_hit = await _evaluate_game_position_cached(",
    "        history_complete = _history_provenance_for_input(fen, moves)\n        res, is_hit = await _evaluate_game_position_cached(",
    1,
)
text = text.replace(
    "        history_complete = bool(moves)\n        rule_status = evaluate_rule_status(board, history_complete=history_complete)",
    "        history_complete = _history_provenance_for_input(fen, moves)\n        rule_status = evaluate_rule_status(board, history_complete=history_complete)",
    1,
)
text = text.replace(
    "            engine_version=getattr(pool, \"engine_version\", None),\n        )\n\n        # sign = mover's perspective sign",
    "            engine_version=getattr(pool, \"engine_version\", None),\n            history_completeness=history_complete,\n        )\n\n        # sign = mover's perspective sign",
    1,
)

# One canonical root action policy. Do not let top_moves invent a different
# threshold from evaluate_position.
root_policy_pattern = re.compile(
    r"        def _pick_root_recommended_action\(items: list\[MCPEval\]\) -> str:[\s\S]*?\n\n        cached =",
)
root_policy_replacement = '''        def _pick_root_recommended_action(items: list[MCPEval]) -> str:
            if not items:
                return rule_status.recommended_action
            best = items[0]
            mover_score = (
                sign * best.cp
                if best.cp is not None
                else (sign * best.mate * 1000 if best.mate is not None else None)
            )
            mate_for_mover = sign * best.mate if best.mate is not None else None
            return choose_recommended_action(
                board,
                can_claim_now=rule_status.can_claim_now,
                can_claim_with_intended_move=rule_status.can_claim_with_intended_move,
                mover_score=mover_score,
                mate_for_mover=mate_for_mover,
            )

        cached ='''
text, n = root_policy_pattern.subn(root_policy_replacement, text, count=1)
if n != 1:
    raise RuntimeError(f"root action policy replacement count={n}")

# Candidate-specific winner must come from the candidate rule state.
text = text.replace(
    "                cand_post_terminal: str | None = None\n                cand_can_claim_now = False",
    "                cand_post_terminal: str | None = None\n                cand_winner: str | None = None\n                cand_can_claim_now = False",
    1,
)
text = text.replace(
    "                            cand_post_terminal = cand_rule.terminal\n                            cand_can_claim_now = cand_rule.can_claim_now",
    "                            cand_post_terminal = cand_rule.terminal\n                            cand_winner = cand_rule.winner\n                            cand_can_claim_now = cand_rule.can_claim_now",
    1,
)
text = text.replace(
    '"winner": rule_status.winner\n                            if cand_post_terminal == "checkmate"\n                            else None,',
    '"winner": cand_winner if cand_post_terminal == "checkmate" else None,',
    1,
)

# classify_move boundary: typed action, provenance, explicit legality validation,
# history-sensitive cache identity and typed played action payload.
text = text.replace(
    '    action_type: str = "play_move",',
    '    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",',
    1,
)
classify_setup_old = '''        board = _build_board(fen, moves or [], strict=strict)
        chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
        pool = await _get_analyzer_pool(ctx)

        cache_key = classify_cache_key(
'''
classify_setup_new = '''        if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:
            raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")
        board = _build_board(fen, moves or [], strict=strict)
        history_complete = _history_provenance_for_input(fen, moves)
        chess_move, syntax_warn = _parse_move_on_board_with_warning(board, move, strict=strict)
        rule_before = evaluate_rule_status(board, history_complete=history_complete)
        if action_type == "claim_draw" and not rule_before.can_claim_now:
            raise ValueError("ILLEGAL_ACTION: draw cannot be claimed now")
        if action_type == "claim_draw_with_intended_move" and chess_move.uci() not in rule_before.intended_claim_ucis:
            raise ValueError("ILLEGAL_ACTION: intended move does not create a legal draw claim")
        pool = await _get_analyzer_pool(ctx)

        cache_key = classify_cache_key(
'''
text = replace_once(text, classify_setup_old, classify_setup_new, "classify boundary")
text = text.replace(
    "            engine_version=getattr(pool, \"engine_version\", None),\n        )\n\n        cached = await _cache.get_classify(cache_key)",
    "            engine_version=getattr(pool, \"engine_version\", None),\n            history_completeness=history_complete,\n        )\n\n        cached = await _cache.get_classify(cache_key)",
    1,
)
text = text.replace(
    "                    action_type=action_type,\n                )\n\n            eval_before, _ = await _evaluate_game_position_cached(",
    "                    action_type=action_type,\n                    history_complete=history_complete,\n                )\n\n            eval_before, _ = await _evaluate_game_position_cached(",
    1,
)
# Every classify evaluation of the root, PV child and played child receives the
# same provenance. Replace only calls inside classify by editing the function slice.
classify_start = text.find("async def classify_move(")
classify_end = text.find("\ndef _compute_game_metrics(", classify_start)
if classify_start < 0 or classify_end < 0:
    raise RuntimeError("classify slice not found")
classify = text[classify_start:classify_end]
classify = re.sub(
    r"(_evaluate_game_position_cached\(\n\s*board, depth, pool, requested_depth=raw_requested_depth)(\n\s*\))",
    r"\1, history_complete=history_complete\2",
    classify,
)
classify = re.sub(
    r"(_evaluate_game_position_cached\(\n\s*board_after, depth, pool, requested_depth=raw_requested_depth)(\n\s*\))",
    r"\1, history_complete=history_complete\2",
    classify,
)
classify = classify.replace(
    "                            analyzer=None,\n                        )",
    "                            analyzer=None,\n                            history_complete=history_complete,\n                        )",
)
classify = classify.replace(
    "                        board, depth + 4, pool, requested_depth=raw_requested_depth + 4\n                    )",
    "                        board, depth + 4, pool, requested_depth=raw_requested_depth + 4,\n                        history_complete=history_complete,\n                    )",
)
classify = classify.replace(
    "                            requested_depth=raw_requested_depth,\n                        )",
    "                            requested_depth=raw_requested_depth,\n                            history_complete=history_complete,\n                        )",
    1,
)
classify = classify.replace(
    "                is_engine_best=score.is_best_engine_move,\n                centipawn_loss=score.centipawn_loss,",
    "                is_engine_best=score.is_best_engine_move,\n                is_best_engine_move=score.is_best_engine_move,\n                centipawn_loss=score.centipawn_loss,",
    1,
)
classify = classify.replace(
    "                action_equivalent=score.action_equivalent,\n                missed_draw_claim=score.missed_draw_claim,",
    "                action_equivalent=score.action_equivalent,\n                played_action_obj=build_played_action(\n                    action_type,\n                    move_uci=chess_move.uci(),\n                    move_san=played_san,\n                    rule_status=rule_before,\n                    cp=eval_after.cp,\n                    mate=eval_after.mate,\n                ),\n                best_action_obj=eval_before.best_action_obj,\n                missed_draw_claim=score.missed_draw_claim,",
    1,
)
text = text[:classify_start] + classify + text[classify_end:]

# Ensure action helpers are imported once at module level.
insert_after = "from core.engines.types import Eval, MoveClass\n"
if "from mcp_server.actions import build_played_action\n" not in text:
    text = text.replace(
        insert_after,
        insert_after + "from mcp_server.actions import build_played_action\n",
        1,
    )

# Explicit termination grammar. Winner-oriented phrases are matched before
# loser-oriented phrases, and generic color+time co-occurrence is forbidden.
termination_helper = '''def _infer_result_from_termination(termination: str | None) -> str | None:
    if not termination:
        return None
    t = re.sub(r"\\s+", " ", termination.strip().lower())
    if "normal time control" in t:
        return None

    winner_patterns = (
        (r"\\bwhite\\s+wins?\\b.*\\b(?:time|resignation|resigns?)\\b", "1-0"),
        (r"\\bblack\\s+wins?\\b.*\\b(?:time|resignation|resigns?)\\b", "0-1"),
        (r"\\bwon\\s+by\\s+white\\b", "1-0"),
        (r"\\bwon\\s+by\\s+black\\b", "0-1"),
    )
    for pattern, result in winner_patterns:
        if re.search(pattern, t):
            return result

    loser_patterns = (
        (r"\\bwhite\\s+(?:resign(?:s|ed)?|lost|loses)\\b", "0-1"),
        (r"\\bblack\\s+(?:resign(?:s|ed)?|lost|loses)\\b", "1-0"),
        (r"\\bwhite(?:'s)?\\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "0-1"),
        (r"\\bblack(?:'s)?\\s+(?:flag|clock).*(?:fell|expired|flagged|out of time)", "1-0"),
        (r"\\bwhite\\s+(?:lost|loses)\\s+on\\s+time\\b", "0-1"),
        (r"\\bblack\\s+(?:lost|loses)\\s+on\\s+time\\b", "1-0"),
    )
    for pattern, result in loser_patterns:
        if re.search(pattern, t):
            return result

    if re.search(r"\\bwhite\\b.*\\b(?:illegal move|rules? infraction)\\b", t):
        return "0-1"
    if re.search(r"\\bblack\\b.*\\b(?:illegal move|rules? infraction)\\b", t):
        return "1-0"
    return None
'''
analyze_anchor = "@mcp.tool(annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True))\nasync def analyze_game("
pos = text.find(analyze_anchor)
if pos < 0:
    raise RuntimeError("analyze_game anchor missing")
text = text[:pos] + termination_helper + "\n\n" + text[pos:]

term_pattern = re.compile(
    r"        # Infer result from termination header if result is unstated \('\*'\)[\s\S]*?\n\n        # Validate Resignation",
)
term_replacement = '''        # Infer only from explicit winner/loser grammar.
        if result_val == "*" or result_val is None:
            inferred = _infer_result_from_termination(termination_header_val)
            if inferred is not None:
                result_val = inferred

        # Validate Resignation'''
text, n = term_pattern.subn(term_replacement, text, count=1)
if n != 1:
    raise RuntimeError(f"termination inference replacement count={n}")

# Harden rate limiting, proxy identity and authentication. UA/Origin are
# observability only and never authentication credentials.
security = '''def _is_trusted_proxy_peer(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.strip().strip("[]"))
    except ValueError:
        return False
    return addr.is_loopback or addr.is_private


def _effective_client_ip(peer_ip: str, forwarded_for: str) -> str:
    peer = peer_ip.strip().strip("[]")
    if forwarded_for and _is_trusted_proxy_peer(peer):
        candidate = forwarded_for.split(",", 1)[0].strip().strip("[]")
        return candidate or peer
    return peer


def _estimate_mcp_request_cost(body: bytes) -> float:
    """Approximate CPU admission cost from tool, depth, MultiPV and PGN size."""
    try:
        payload = json.loads(body.decode("utf-8"))
        params = payload.get("params") if isinstance(payload, dict) else None
        params = params if isinstance(params, dict) else {}
        tool_name = params.get("name") or params.get("tool") or ""
        args = params.get("arguments")
        args = args if isinstance(args, dict) else {}
        depth = max(1, min(int(args.get("depth", 14)), 30))
        if tool_name == "evaluate_position":
            return 1.0 + depth / 14.0
        if tool_name == "top_moves":
            n = max(1, min(int(args.get("n", 3)), 20))
            return 1.0 + (depth * n) / 14.0
        if tool_name == "classify_move":
            return 2.0 + depth / 10.0
        if tool_name == "analyze_game":
            pgn = str(args.get("pgn", ""))
            estimated_plies = max(1.0, min(200.0, len(pgn) / 24.0))
            return 5.0 + (depth * estimated_plies) / 28.0
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        pass
    return 1.0


class TokenBucketRateLimiter:
    """In-memory weighted token bucket per effective client IP."""

    def __init__(self, rate: float = 20.0, capacity: float = 500.0) -> None:
        self.rate = rate
        self.capacity = capacity
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str, cost: float = 1.0) -> bool:
        now = time.time()
        async with self._lock:
            if len(self._buckets) > 10_000:
                cutoff = now - 3600
                self._buckets = {k: v for k, v in self._buckets.items() if v[1] > cutoff}
                if len(self._buckets) > 10_000:
                    self._buckets = dict(
                        sorted(
                            self._buckets.items(),
                            key=lambda item: item[1][1],
                            reverse=True,
                        )[:5000]
                    )
            tokens, last_time = self._buckets.get(client_id, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last_time) * self.rate)
            if tokens >= cost:
                self._buckets[client_id] = (tokens - cost, now)
                return True
            self._buckets[client_id] = (tokens, now)
            return False


class ASGIRequestLoggerMiddleware:
    """Request logging, weighted admission control and token authentication."""

    MAX_BUFFERED_BODY = 32 * 1024 * 1024

    def __init__(
        self,
        app: ASGIApp,
        restrict_to_chatgpt: bool = False,
        rate_limiter: TokenBucketRateLimiter | None = None,
    ) -> None:
        self.app = app
        self.restrict_to_chatgpt = restrict_to_chatgpt
        self.rate_limiter = rate_limiter or TokenBucketRateLimiter()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
        headers_dict: dict[bytes, bytes] = dict(raw_headers)
        ua = headers_dict.get(b"user-agent", b"").decode("utf-8", "ignore")
        origin = headers_dict.get(b"origin", b"").decode("utf-8", "ignore")
        host = headers_dict.get(b"host", b"").decode("utf-8", "ignore")
        path = str(scope.get("path", ""))
        method = str(scope.get("method", "")).upper()
        session_id = headers_dict.get(b"mcp-session-id", b"").decode("utf-8", "ignore")

        client_tuple = cast(tuple[str, int] | None, scope.get("client"))
        peer_ip = client_tuple[0] if client_tuple else "127.0.0.1"
        forwarded_for = headers_dict.get(b"x-forwarded-for", b"").decode("utf-8", "ignore")
        client_ip = _effective_client_ip(peer_ip, forwarded_for)

        if method == "OPTIONS":
            await send(
                cast(
                    Message,
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"access-control-allow-origin", b"*"),
                            (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
                            (b"access-control-allow-headers", b"*"),
                            (b"access-control-max-age", b"86400"),
                            (b"content-length", b"0"),
                        ],
                    },
                )
            )
            await send(cast(Message, {"type": "http.response.body", "body": b""}))
            return

        request_cost = 1.0
        if method == "POST":
            chunks: list[bytes] = []
            total = 0
            while True:
                message = await receive()
                if message.get("type") != "http.request":
                    break
                chunk = bytes(message.get("body", b""))
                total += len(chunk)
                if total > self.MAX_BUFFERED_BODY:
                    response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Request body too large"}}\\n'
                    await send(cast(Message, {"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii"))]}))
                    await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                    return
                chunks.append(chunk)
                if not message.get("more_body", False):
                    break
            buffered_body = b"".join(chunks)
            request_cost = _estimate_mcp_request_cost(buffered_body)
            replayed = False

            async def replay_receive() -> Message:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return cast(Message, {"type": "http.request", "body": buffered_body, "more_body": False})
                return cast(Message, {"type": "http.disconnect"})

            receive = replay_receive

        if not await self.rate_limiter.is_allowed(client_ip, cost=request_cost):
            response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Rate limit exceeded. Please slow down."}}\\n'
            await send(cast(Message, {"type": "http.response.start", "status": 429, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii")), (b"retry-after", b"2")]}))
            await send(cast(Message, {"type": "http.response.body", "body": response_body}))
            return

        log.info(
            "%s %s host=%s ip=%s session=%s origin=%s ua=%s cost=%.2f",
            method,
            path,
            host,
            client_ip,
            session_id,
            origin,
            ua,
            request_cost,
        )

        if path == "/health":
            await self.app(scope, receive, send)
            return

        if self.restrict_to_chatgpt:
            from .config import get_mcp_settings

            auth_token = get_mcp_settings().auth_token
            provided = headers_dict.get(b"x-chessy-auth", b"").decode("utf-8", "ignore").strip()
            authorization = headers_dict.get(b"authorization", b"").decode("utf-8", "ignore").strip()
            bearer = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            valid = bool(auth_token) and (provided == auth_token or bearer == auth_token)
            if not valid:
                log.warning("Blocked unauthenticated client ip=%s ua=%r origin=%r", client_ip, ua, origin)
                response_body = b'{"jsonrpc":"2.0","error":{"code":-32000,"message":"Forbidden: valid MCP auth token required"}}\\n'
                await send(cast(Message, {"type": "http.response.start", "status": 403, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(response_body)).encode("ascii"))]}))
                await send(cast(Message, {"type": "http.response.body", "body": response_body}))
                return

        await self.app(scope, receive, send)
'''
text = replace_block(text, "class TokenBucketRateLimiter:", "def _build_app(", security, "security boundary")

PATH.write_text(text, encoding="utf-8")
print("audit v4 server migration applied")
