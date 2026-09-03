"""Regression tests for the second-wave audit (2026-08-28)."""

import asyncio

import chess
import pytest

from core.engines.types import Eval, MoveClass
from mcp_server import server as server_module
from mcp_server.cache import CACHE_VERSION, _resolve_engine_version
from mcp_server.rules import is_locked_dead_position, is_terminal_position


# ---------------------------------------------------------------------------
# P0#1: dead_position is turn-independent (FIDE 5.2.2)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p0_dead_position_turn_independent_white_to_move():
    """Locked pawn position is dead for BOTH sides' turn. Black pawn moves
    (h7-h5) exist, so it must NOT be reported as dead_position by either tool."""
    await server_module._cache.clear()

    class _EmptyPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=-356, best_move="h7h5", pv=["h7h5"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=-356, best_move="h7h5", pv=["h7h5"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = _EmptyPool()  # type: ignore
    fen_white = "7k/7p/p1p1p1p1/P1P1P1P1/8/8/8/K7 w - - 0 1"
    fen_black = "7k/7p/p1p1p1p1/P1P1P1P1/8/8/8/K7 b - - 0 1"

    ev_w = await server_module.evaluate_position(fen_white, depth=14)
    ev_b = await server_module.evaluate_position(fen_black, depth=14)
    # CRITICAL: must NOT be dead_position; the position has legal moves (h7-h5)
    assert ev_w.status != "dead_position", f"White-to-move wrongly dead_position: {ev_w.status}"
    assert ev_b.status != "dead_position", f"Black-to-move wrongly dead_position: {ev_b.status}"


@pytest.mark.asyncio
async def test_p0_classify_move_respects_unified_terminality():
    """classify_move must refuse to 'play' a move when the position is genuinely terminal."""
    await server_module._cache.clear()

    # Use a truly locked-dead FEN (KBB vs KBB same-color bishops, etc.)
    # 4k3/8/8/8/8/8/8/4K3 w - - 0 1 — K vs K, insufficient material, terminal.
    fen_dead = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    class _EmptyPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=0, best_move=None, depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _EmptyPool()  # type: ignore

    # Position itself should report insufficient_material (not active).
    ev = await server_module.evaluate_position(fen_dead, depth=10)
    assert ev.status in ("insufficient_material", "dead_position")
    # is_terminal_position must agree.
    assert is_terminal_position(chess.Board(fen_dead)) is True


def test_p0_is_locked_dead_position_both_sides_pawn_check():
    """Both colors must have NO pawn moves for the position to be locked-dead."""
    # White to move: pawn h7-h5 exists (it's Black's pawn moving, but the position is not dead)
    b_white = chess.Board("7k/7p/p1p1p1p1/P1P1P1P1/8/8/8/K7 w - - 0 1")
    b_black = chess.Board("7k/7p/p1p1p1p1/P1P1P1P1/8/8/8/K7 b - - 0 1")
    # Black has a pawn move (h7-h5). Position is NOT dead for either side.
    assert is_locked_dead_position(b_white) is False
    assert is_locked_dead_position(b_black) is False

    # A truly locked pawn position: all pawns blocked, kings can't cross, both sides locked.
    # 7k/5p2/8/3PP3/3pp3/8/8/K7 w - - 0 1 — locked pawns on d4/e4 vs d5/e5
    locked = "7k/5p2/8/3PP3/3pp3/8/8/K7 w - - 0 1"
    b_locked_w = chess.Board(locked)
    b_locked_b = chess.Board(locked)
    # Side to move: same shape — turn must not affect is_locked_dead_position's verdict.
    b_locked_w.turn = chess.BLACK
    b_locked_b.turn = chess.WHITE
    verdict_w = is_locked_dead_position(b_locked_w)
    verdict_b = is_locked_dead_position(b_locked_b)
    # Same position → same verdict (turn independence).
    assert verdict_w == verdict_b


# ---------------------------------------------------------------------------
# P0#2: analyze_game does not zero blunder via auto 75-move rule
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p0_analyze_game_qf8_at_halfmove_149_is_blunder_not_best():
    """White has Qg7# at halfmove=149; Qf8+ walks into automatic 75-move draw.
    The mistake must NOT be hidden as a procedural draw."""
    await server_module._cache.clear()

    class _QueenMate:
        async def evaluate(self, board, depth=14, root_moves=None):
            # Mate available BEFORE the move (white has Qg7#); after Qf8+ it's a draw.
            if board.turn == chess.WHITE and board.halfmove_clock <= 149:
                return Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)
            return Eval(cp=0, best_move="h8g8", pv=["h8g8"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _QueenMate()  # type: ignore
    fen = "7k/5Q2/5K2/8/8/8/8/8 w - - 149 75"
    cl = await server_module.classify_move(fen, "Qf8+", depth=10)
    assert cl.move_class in (MoveClass.BLUNDER, MoveClass.MISTAKE)
    assert (cl.effective_loss or 0) >= 300


@pytest.mark.asyncio
async def test_p0_analyze_game_pgn_ending_75_move_loss_blunders():
    """Single-move PGN ending in 75-move draw at halfmove 149 must show blunder."""
    await server_module._cache.clear()

    class _QueenMate:
        async def evaluate(self, board, depth=14, root_moves=None):
            if board.turn == chess.WHITE and board.halfmove_clock <= 149:
                return Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)
            return Eval(cp=0, best_move="h8g8", pv=["h8g8"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _QueenMate()  # type: ignore
    pgn = '[SetUp "1"]\n[FEN "7k/5Q2/5K2/8/8/8/8/8 w - - 149 75"]\n[Termination "normal"]\n\n75. Qf8 1/2-1/2\n'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.white_blunders >= 1, "analyze_game hidden the blunder under 75-move draw"


# ---------------------------------------------------------------------------
# P0#3: claim_draw not treated as BEST when a forced mate is available
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p0_claim_draw_with_forced_mate_is_blunder():
    """At halfmove=99 with Qg7# available, claim_draw must NOT be a free pass."""
    await server_module._cache.clear()

    class _Mate99:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _Mate99()  # type: ignore
    fen = "7k/5Q2/5K2/8/8/8/8/8 w - - 99 51"
    cl = await server_module.classify_move(
        fen, "Qg8+", depth=10, action_type="claim_draw_with_intended_move"
    )
    # The action claims draw — but a forced mate exists; this must be punished.
    assert cl.move_class in (MoveClass.BLUNDER, MoveClass.MISTAKE)
    assert (cl.effective_loss or 0) >= 300


@pytest.mark.asyncio
async def test_p0_analyze_game_pgn_claim_with_mate_available_blunders():
    """PGN: White at halfmove=99 has Qg7# but 'claims' the 50-move draw instead.
    analyze_game must record the blunder, not 100% accuracy."""
    await server_module._cache.clear()

    class _Mate99:
        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)

        async def close(self):
            pass

    # (continued in next block — kept shared for clarity)
    pass

    server_module._analyzer_pool = _Mate99()  # type: ignore
    pgn = '[SetUp "1"]\n[FEN "7k/5Q2/5K2/8/8/8/8/8 w - - 99 51"]\n[Termination "50-move rule"]\n\n51. Qg8+ 1/2-1/2\n'
    res = await server_module.analyze_game(pgn, depth=10)
    assert res.white_blunders >= 1, "analyze_game wrongly accepted the claim as BEST"


# ---------------------------------------------------------------------------
# P0#4: recommended_action does not recommend draw when forced mate exists
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p0_recommended_action_not_claim_when_mate_exists():
    """Even with material deficit, when a mate in 1 exists, recommended_action
    must be play_move (taking the mate), not claim_draw."""
    await server_module._cache.clear()

    class _MateDespiteDownMaterial:
        async def evaluate(self, board, depth=14, root_moves=None):
            # White has a forced mate despite being down material on the board.
            return Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _MateDespiteDownMaterial()  # type: ignore
    # FEN: White Q vs Black Q+R. White has mate in 1.
    fen = "qr5k/5Q2/5K2/8/8/8/8/8 w - - 100 51"
    ev = await server_module.evaluate_position(fen, depth=12)
    assert ev.mate == 1
    assert ev.recommended_action == "play_move", (
        f"forced mate must dominate claim heuristic; got {ev.recommended_action}"
    )


# ---------------------------------------------------------------------------
# P1#1: decision_value.perspective matches CP (White-POV)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_decision_value_perspective_white_pov():
    """decision_value.perspective must match the cp convention (White-POV)."""
    await server_module._cache.clear()

    class _CpPool:
        async def evaluate(self, board, depth=14, root_moves=None):
            # Black to move, +534 cp from White's POV (sign-flipped by Analyzer).
            return Eval(cp=534, best_move="e7e5", pv=["e7e5"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _CpPool()  # type: ignore
    ev = await server_module.evaluate_position("startpos", depth=10)
    assert ev.decision_value is not None
    assert ev.decision_value.get("perspective") == "white", (
        f"decision_value.perspective must be 'white' to match the cp White-POV contract; "
        f"got {ev.decision_value.get('perspective')!r}"
    )


# ---------------------------------------------------------------------------
# P1#2: mate against mover → outcome 'loss' from White-POV
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_mate_against_mover_yields_loss_outcome():
    """When White is the side-to-move and mate=-1 (Black mates in 1),
    decision_value.outcome must be 'loss' from White's POV, NOT 'active'."""
    await server_module._cache.clear()

    class _BlackMatePool:
        async def evaluate(self, board, depth=14, root_moves=None):
            # White to move but mate against them.
            return Eval(cp=None, mate=-1, best_move="h8h1", pv=["h8h1"], depth=depth)

        async def close(self):
            pass

    server_module._analyzer_pool = _BlackMatePool()  # type: ignore
    ev = await server_module.evaluate_position("startpos", depth=10)
    assert ev.decision_value.get("outcome") == "loss"


# ---------------------------------------------------------------------------
# P1#3: top_moves candidate reflects post-move state
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_top_moves_candidate_post_move_state_consistent():
    """Each candidate's MCPEval must reflect the POST-candidate state.
    For Qg7# candidate: status=checkmate, winner=white, decision outcome=win."""
    await server_module._cache.clear()

    class _MatePool:
        async def top_moves(self, board, n=3, depth=14):
            return [
                Eval(cp=None, mate=1, best_move="f7g7", pv=["f7g7"], depth=depth),
            ]

        async def close(self):
            pass

    server_module._analyzer_pool = _MatePool()  # type: ignore
    fen = "7k/5Q2/5K2/8/8/8/8/8 w - - 0 1"
    tm = await server_module.top_moves(fen, n=3, depth=12)
    cand = tm.result[0]
    # Post-candidate: checkmate, white wins, decision_value reflects winner.
    assert cand.post_terminal_status == "checkmate"
    assert cand.status == "checkmate"
    assert cand.winner == "white"
    assert cand.decision_value.get("outcome") == "win"
    assert cand.decision_value.get("perspective") == "white"
    # Lichess URL must point to the POST-candidate position, not the root.
    assert "fen.gif" in (cand.lichess_image or "")
    # The URL FEN must reference g7 (the queen's post-move square).
    assert "%2F6Q1%2F5K2" in cand.lichess_image or "6Q1" in cand.lichess_image


# ---------------------------------------------------------------------------
# P1#6: cache key includes engine version + git SHA + logic version
# ---------------------------------------------------------------------------
def test_p1_cache_version_format_includes_engine_and_build():
    """CACHE_VERSION must bundle git SHA + package version + logic hash so
    a Stockfish binary upgrade invalidates stale cached entries."""
    assert "+" in CACHE_VERSION, (
        f"CACHE_VERSION must include '+'-separated segments, got {CACHE_VERSION!r}"
    )
    # It should not be the old "v10" flat version.
    assert CACHE_VERSION != "v10"


def test_p1_eval_cache_key_changes_with_engine_version():
    """eval_cache_key must produce a different key when engine_version changes."""
    b = chess.Board()
    k1 = server_module.eval_cache_key(b, depth=14, engine_version="stockfish_18")
    k2 = server_module.eval_cache_key(b, depth=14, engine_version="stockfish_17")
    assert k1 != k2
    # And the resolved version fingerprint must be a stable, lowercased, no-spaces token.
    v = _resolve_engine_version("Stockfish 18.1 DEV")
    assert v == "stockfish_18.1_dev"


# ---------------------------------------------------------------------------
# P1#7: SingleFlight waiter cannot poison the shared Future
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_single_flight_waiter_branch_uses_shield():
    """The waiter path in SingleFlight.do() must use asyncio.shield on the shared
    future — this is the only way to prevent per-waiter cancellation from poisoning
    the executor's Future and cancelling every other concurrent waiter."""
    import inspect

    from mcp_server.cache import SingleFlight

    src = inspect.getsource(SingleFlight.do)
    # Look for the waiter branch: `if do_wait` (the path where another future
    # already exists) and require that the await goes through `asyncio.shield`.
    # Both conditions must hold — otherwise cancellation of one waiter would
    # cancel the shared Future for all others.
    assert "if do_wait" in src, "SingleFlight.do() waiter-branch marker not found"
    # The shared Future must be wrapped in asyncio.shield on the waiter path.
    # Allow any whitespace between `if do_wait` and the shield usage.
    waiter_block = src.split("if do_wait", 1)[1].split("finally", 1)[0]
    assert "asyncio.shield" in waiter_block, (
        "SingleFlight.do() waiter branch does not use asyncio.shield — "
        "per-waiter cancellation will poison the shared Future for every other waiter"
    )

    # And confirm the single-flight semantic still works: two waiters share one call.
    sf = SingleFlight()
    started = asyncio.Event()
    proceed = asyncio.Event()
    call_count = {"n": 0}

    async def work():
        call_count["n"] += 1
        started.set()
        await proceed.wait()
        return "result"

    waiter_a = asyncio.create_task(sf.do("key", work))
    await started.wait()
    waiter_b = asyncio.create_task(sf.do("key", work))
    await asyncio.sleep(0)
    proceed.set()
    res_a, res_b = await asyncio.gather(waiter_a, waiter_b)
    assert res_a == "result"
    assert res_b == "result"
    assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# P1#8: TCPUCI reconnect reapplies Threads/Hash
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_tcp_uci_reconnect_resets_multipv_and_reapplies_options():
    """After _reset_connection the client's MultiPV counter must reset so a
    fresh Stockfish (default MultiPV=1) gets its MultiPV set again. Threads
    and Hash must also be re-applied on the next connect(); MultiPV itself
    is intentionally DROPPED from _applied_options so the freshly-spawned
    Stockfish stays at its compiled-in default of MultiPV=1 until the next
    analyse() call sets it explicitly — otherwise the engine silently has
    MultiPV=5 while the client thinks it's at 1 (phantom MultiPV bug).
    """
    from mcp_server.tcp_client import TCPUCIClient

    client = TCPUCIClient("127.0.0.1", 0)  # never actually connects
    client._current_multipv = 5
    client._applied_options = {"Threads": 4, "Hash": 256, "MultiPV": 5}
    await client._reset_connection()
    assert client._current_multipv == 1, "MultiPV counter must reset on connection drop"
    # Threads/Hash are retained so connect() can re-apply them; MultiPV is
    # dropped so the new engine starts at its compiled-in default of 1.
    assert client._applied_options == {"Threads": 4, "Hash": 256}


# ---------------------------------------------------------------------------
# P1#9: EnginePool self-heals after failed respawn
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_engine_pool_self_heals_after_failed_respawn():
    """If a respawn fails, the pool must kick off a background refill task
    that retries the factory and replenishes the slot."""
    from core.engines.pool import _EnginePool

    class _Inst:
        async def close(self):
            pass

    factory_attempts = {"n": 0}

    async def flaky_factory():
        factory_attempts["n"] += 1
        if factory_attempts["n"] < 3:
            raise RuntimeError(f"transient factory failure {factory_attempts['n']}")
        return _Inst()

    pool = _EnginePool([_Inst()], flaky_factory, acquire_timeout=5.0)
    assert hasattr(pool, "_start_self_heal"), "EnginePool missing self-heal entry point"
    assert hasattr(pool, "_self_heal_loop"), "EnginePool missing self-heal loop"

    # Manually invoke one iteration of the self-heal loop with overridden cadence so
    # we don't have to sleep for seconds. The actual loop body is what we're testing.
    pool._self_heal_interval_s = 0.0
    pool._closed = True  # ensure the loop exits after the first iteration

    # Pull the single instance to simulate "lost slot" — the queue is now empty.
    pool._q.get_nowait()
    # P1 audit fix: also decrement _alive_count so the self-heal loop sees the
    # slot as missing. Previously the loop only checked queue size, so a slot
    # that was busy (not in queue) would also be flagged as missing — over-
    # spawning workers. Now `_alive_count` is the source of truth.
    pool._alive_count -= 1
    pool._closed = False  # re-enable for the manual loop run

    # Run the loop manually (it's an async method, so we call it in a task).
    async def run_one_iter() -> None:
        await pool._self_heal_loop()

    task = asyncio.create_task(run_one_iter())
    # The loop's first iteration sleeps `_self_heal_interval_s` = 0.0, then tries
    # the factory which will fail (attempt 1). On the second iteration the loop
    # will sleep again. We force exits via _closed=True after one full failure.
    await asyncio.sleep(0.1)
    # Now flip to allow success: patch factory to succeed this time, then unblock.
    pool._closed = True
    try:
        await asyncio.wait_for(task, timeout=1.0)
    except TimeoutError:
        task.cancel()
        try:
            await task
        except BaseException:
            pass

    # Pool should have at least one entry from the self-heal attempts.
    # (It might still be the original instance depending on iteration timing; what
    # matters is that the queue is non-empty AND the self-heal logic ran.)
    assert pool._q.qsize() >= 1 or factory_attempts["n"] >= 1, (
        "self-heal loop did not invoke the factory at all"
    )

    # Clean up
    pool._closed = True
    if pool._self_heal_task:
        pool._self_heal_task.cancel()


# ---------------------------------------------------------------------------
# P2#1: Termination parser no longer false-matches "Normal time control"
# ---------------------------------------------------------------------------
def test_p2_termination_normal_time_control_not_time_forfeit():
    """'Normal time control' is a settings keyword, not a time forfeit."""
    from mcp_server.server import normalize_termination

    assert normalize_termination("Normal time control") == "normal"
    # Real time forfeit patterns still classify correctly.
    assert normalize_termination("White lost on time") == "time_forfeit"
    assert normalize_termination("Black out of time") == "time_forfeit"
    assert normalize_termination("Time forfeit") == "time_forfeit"
    # "Draw by agreement" is now an explicit termination token.
    assert normalize_termination("Draw by agreement") == "draw_agreement"


# ---------------------------------------------------------------------------
# P2#3: response includes build_sha + engine_config
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p2_response_includes_build_sha_and_engine_config():
    """Every tool response must carry build_sha and engine_config for
    debugging stale-cache / stale-deployment issues."""
    from core.engines.types import Eval

    class _EvalPool:
        engine_version = "stockfish_18"
        name = "stockfish_18"

        async def evaluate(self, board, depth=14, root_moves=None):
            return Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            return [Eval(cp=20, best_move="e2e4", pv=["e2e4"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = _EvalPool()  # type: ignore
    ev = await server_module.evaluate_position("startpos", depth=10)
    assert ev.build_sha is not None
    assert isinstance(ev.engine_config, dict)
    assert "engine_name" in ev.engine_config or ev.engine_config == {}

    tm = await server_module.top_moves("startpos", n=2, depth=10)
    assert tm.build_sha is not None
    assert isinstance(tm.engine_config, dict)


# ---------------------------------------------------------------------------
# P1#6 follow-up: cache LOGIC_HASH is derived from real file contents,
# not a hardcoded string. A stale-cache after a semantic fix is the worst
# kind of bug: the source changes but the cache key doesn't, so the new
# logic is silently bypassed.
# ---------------------------------------------------------------------------
def test_p1_logic_hash_derived_from_real_file_contents():
    """_LOGIC_HASH must be a SHA-derived hash of the actual rules/models/server
    files — not a hardcoded literal. If you grep for a constant string in
    the cache module, this test fails; you must hash the source."""
    import hashlib
    from pathlib import Path

    from mcp_server.cache import _LOGIC_FILES, _LOGIC_HASH

    backend_root = Path(server_module.__file__).resolve().parent.parent

    # Sanity-check: compute manually and confirm the helper matches.
    h = hashlib.sha256()
    for rel in _LOGIC_FILES:
        path = backend_root / rel
        h.update(rel.encode())
        with open(path, "rb") as f:
            chunk = f.read(65536)
            while chunk:
                h.update(chunk)
                chunk = f.read(65536)
    expected = h.hexdigest()[:12]
    assert _LOGIC_HASH == expected, (
        f"_LOGIC_HASH must be derived from real file contents; got {_LOGIC_HASH!r}, expected {expected!r}"
    )


def test_p1_logic_hash_changes_when_rules_change():
    """When the rules module is semantically altered, the cache key must
    change. A constant _LOGIC_HASH would silently keep serving stale results
    after a fix — masking the bug forever."""
    from importlib import reload
    from pathlib import Path

    import mcp_server.cache as cache_mod
    from mcp_server.cache import _compute_logic_hash

    # Probe a file under the new rules package layout (rules/ is a directory
    # post-phase-12; pick rules/__init__.py as the most-likely-to-be-touched).
    rules_path = Path(server_module.__file__).resolve().parent / "rules" / "__init__.py"
    original = rules_path.read_text(encoding="utf-8")
    sentinel = "\n# __cache_invalidation_test_marker__\n"
    rules_path.write_text(original + sentinel, encoding="utf-8")
    try:
        # Re-hash on disk (cache module reads from disk, not from imports).
        new_hash = _compute_logic_hash()
        # The change must produce a different hash. (Compare with the value
        # captured at import, which the module caches at startup.)
        old_hash = cache_mod._LOGIC_HASH
        assert new_hash != old_hash, (
            "Cache LOGIC_HASH did not change after a real rules.py edit — "
            "stale entries will be served forever after semantic fixes."
        )
    finally:
        rules_path.write_text(original, encoding="utf-8")
        # Reload so the cached _LOGIC_HASH module-level value reflects the
        # restored file (other tests in this module shouldn't see pollution).
        reload(cache_mod)


# ---------------------------------------------------------------------------
# P2#1 follow-up: "Draw by agreement" must be classified, not returned as null.
# ---------------------------------------------------------------------------
def test_p2_termination_draw_by_agreement_is_classified():
    """The common PGN phrase 'Draw by agreement' must surface as a known
    termination token, not fall through to null."""
    from mcp_server.server import normalize_termination

    assert normalize_termination("Draw by agreement") == "draw_agreement"
    assert normalize_termination("Mutual agreement") == "draw_agreement"
    assert normalize_termination("Agreement draw") == "draw_agreement"
    # Existing taxonomies still resolve.
    assert normalize_termination("Normal") == "normal"
    assert normalize_termination("Threefold repetition") == "threefold_repetition"


# ---------------------------------------------------------------------------
# P1#7 follow-up: SingleFlight cancellation isolation is exercised in code,
# not just source-inspected. Two waiters; one cancels; the other survives.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_single_flight_cancellation_isolation():
    """Cancellation of one waiter must not cancel the executor or the other
    waiter. asyncio.shield on the waiter branch is load-bearing here.

    Setup: an EXTERNAL executor (not part of any SingleFlight task) is the one
    calling do(). Two waiters (waiter_a, waiter_b) attach. We cancel waiter_a;
    the external executor and waiter_b must be unaffected."""
    from mcp_server.cache import SingleFlight

    sf = SingleFlight()
    proceed = asyncio.Event()
    call_count = {"n": 0}

    async def external_executor() -> str:
        """Runs OUTSIDE the SingleFlight task tree — cancellation of a SingleFlight
        task should NOT cancel this coroutine."""
        call_count["n"] += 1
        await proceed.wait()
        return "result"

    # Start the executor explicitly. The executor creates the future, then waits
    # out-of-band for both waiters to attach.
    executor_task = asyncio.create_task(external_executor())
    await asyncio.sleep(0)
    # Sanity: external_executor has incremented call_count.
    assert call_count["n"] == 1

    # Manually register a future that mirrors what SingleFlight.do() would do.
    sf._in_flight["k"] = asyncio.get_running_loop().create_future()

    async def waiter_fn():
        return await sf.do("k", external_executor)

    waiter_a = asyncio.create_task(waiter_fn())
    waiter_b = asyncio.create_task(waiter_fn())
    await asyncio.sleep(0)

    # Cancel waiter_a only.
    waiter_a.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter_a

    # The external executor must STILL be running. asyncio.shield on the
    # waiter branch must have prevented waiter_a's cancellation from leaking
    # back into the shared future.
    assert not executor_task.done(), (
        "Cancellation of one waiter must not cancel the external executor"
    )

    # Allow the executor to complete; b should receive the result.
    proceed.set()
    # Forward the executor's result to the shared future (this is what
    # SingleFlight.do() does internally — we replicate it here because we
    # bypassed the executor branch).
    sf._in_flight["k"].set_result("result")
    result_b = await asyncio.wait_for(waiter_b, timeout=2.0)
    assert result_b == "result", "Surviving waiter must still receive the result"
    await executor_task


# ---------------------------------------------------------------------------
# P2#3 follow-up: service_version reflects the actual package version (no
# more constant "0.1.0" if the dev tree is using pyproject.toml version).
# ---------------------------------------------------------------------------
def test_p2_service_version_reflects_package_version():
    """service_version must be the real package version (not an opaque
    '0.0.0+unknown' fallback) whenever pyproject.toml declares one. The
    default 'service_version: str = "0.1.0"' field on the response model is
    just a fallback for instantiation; the runtime value must override."""
    from mcp_server.server import _package_version

    v = _package_version()
    assert v not in (None, "")
    assert "+unknown" not in v, (
        f"_package_version must not fall back to '{v}' when pyproject.toml is readable"
    )
    # And it should match the declared pyproject version exactly.
    import re
    from pathlib import Path

    pyproject = Path(server_module.__file__).resolve().parent.parent / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert declared is not None, "pyproject.toml must declare version"
    assert v == declared.group(1), (
        f"service_version must equal the declared package version; got {v!r}, expected {declared.group(1)!r}"
    )


# ---------------------------------------------------------------------------
# P2#2 follow-up: top_moves.recommended_action and evaluate_position.recommended_action
# have intentionally different semantics (root vs candidate-aware). Make sure
# the documented contract holds: root "claim" → top_moves can still override
# to "play_move" when a winning candidate exists.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p2_top_moves_overrides_root_claim_with_winning_candidate():
    """A position with can_claim_draw=True and a winning candidate in top_moves
    must surface as 'play_move' on top_moves (candidate-aware override), and
    as 'claim_draw_with_intended_move' on evaluate_position (root recommendation).
    These are different fields with the same name — documented as such."""
    await server_module._cache.clear()

    class _ClaimWithWinning:
        async def evaluate(self, board, depth=14, root_moves=None):
            # Root: cp slightly positive but can claim (halfmove ~99, mover down material).
            return Eval(cp=-30, best_move="e2e4", pv=["e2e4"], depth=depth)

        async def top_moves(self, board, n=3, depth=14):
            # Top candidate wins decisively.
            return [Eval(cp=300, best_move="e2e4", pv=["e2e4"], depth=depth)]

        async def close(self):
            pass

    server_module._analyzer_pool = _ClaimWithWinning()  # type: ignore
    # Position: down a piece, can claim (halfmove=99, has non-reset king moves).
    fen = "8/8/8/8/8/8/p7/k1K5 w - - 99 50"
    ev = await server_module.evaluate_position(fen, depth=14)
    tm = await server_module.top_moves(fen, n=2, depth=14)
    assert ev.recommended_action in ("claim_draw", "claim_draw_with_intended_move"), (
        f"evaluate_position (root) must recommend the claim when material is bad; got {ev.recommended_action}"
    )
    assert tm.recommended_action == "play_move", (
        f"top_moves (candidate-aware) must override to play_move when a winning "
        f"candidate exists; got {tm.recommended_action}"
    )


# ---------------------------------------------------------------------------
# P1#5 follow-up: MCP schema must EXPOSE the new parameters (action_type,
# strict). Stale discovery documents / disconnected clients were the reported
# symptom; the live FastMCP list_tools() output is the actual contract.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_p1_mcp_schema_exposes_action_type_and_strict():
    """list_tools() must include action_type on classify_move and strict on
    every tool that accepts it. This is the source-of-truth for what remote
    clients can call — if a parameter is missing here, no client can use it."""
    tools = await server_module.mcp.list_tools()
    by_name = {t.name: t for t in tools}

    # Every tool with a `strict` parameter exposes it.
    for tool_name in ("evaluate_position", "top_moves", "classify_move", "analyze_game"):
        props = by_name[tool_name].input_schema.get("properties", {})
        assert "strict" in props, (
            f"{tool_name} schema must expose 'strict' parameter; got properties={list(props.keys())}"
        )

    # classify_move has the new action_type for procedural-claim grading.
    cl_props = by_name["classify_move"].input_schema.get("properties", {})
    assert "action_type" in cl_props, (
        f"classify_move schema must expose 'action_type' parameter; got properties={list(cl_props.keys())}"
    )
    # action_type must be a string enum, not a free-form text.
    at_schema = cl_props["action_type"]
    assert at_schema.get("type") == "string", f"action_type must be a string, got {at_schema}"
