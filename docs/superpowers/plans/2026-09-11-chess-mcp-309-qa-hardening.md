# Chess MCP 309-Call Adversarial QA Reliability Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve all 10 confirmed issues from the 309-call adversarial QA campaign (prioritizing P0 -> P1 -> P2), enforce core correctness invariants A–H across the engine and API layers, and expand the ultra stress harness to 1,000+ automated adversarial test cases with semantic oracles.

**Architecture:** Enforce invariants at domain boundaries:
1. Engine move transitions enforce Invariant A (best engine move has zero regret) and Invariant B (only legal move is never a move-selection blunder).
2. Practical equivalence enforces reflexive equivalence for the engine-best move.
3. Game termination treats board-terminal outcomes as authoritative over conflicting PGN headers (Invariant D).
4. Coaching parser strips machine PGN directives (`[%clk]`, etc.) before evaluating human self-reports (Invariant E).
5. Repetition status cleanly distinguishes history-independent vs history-dependent terminal states (Invariant F).
6. Tool parameters strictly validate boundaries (`proof_defenses >= 1`) and document verbosity aliases.
7. Nested engine provenance is synchronized with root request parameters.
8. Stress test harness generates dedicated adversarial categories and asserts invariants via semantic oracles.

**Tech Stack:** Python 3.11+, `uv`, `python-chess`, `Stockfish 18`, `FastMCP`, `Pydantic v2`, `Pyright` (strict), `pytest` + `pytest-asyncio`.

---

## File Structure & Responsibilities

| File Path | Component | Responsibility |
|:---|:---|:---|
| `mcp_server/analysis/move_grading/strategies/transitions.py` | Engine Regret Evaluation | Invariant A & B: zero regret for engine-best and forced only-legal moves. |
| `mcp_server/analysis/practical_equivalence.py` | Practical Equivalence Engine | Invariant A: reflexive equivalence (`status="equivalent"`) when played move is engine best. |
| `mcp_server/analysis/game_termination.py` | Game Termination Assessment | Invariant D: board-terminal outcome is authoritative over contradictory PGN headers. |
| `mcp_server/analysis/game_coaching.py` | PGN Game Coaching & Moments | Invariant E: strip machine directives (`[%clk]`, `[%eval]`, etc.) from player self-reports. |
| `mcp_server/rules/status.py` | Rule Status Determination | Invariant F: `repetition_sufficient_without_history=False` for fivefold/threefold repetition. |
| `mcp_server/tools/top_moves.py` | Tool Layer | Validate `proof_defenses >= 1`, enforce `TOP_MOVES_MAX_N=20`, handle forensic projection cleanly. |
| `mcp_server/tools/_common.py` | Shared Tool Helpers | Document verbosity aliases and manage response projections. |
| `mcp_server/engine/cached_evaluator.py` | Evaluation Cache & Provenance | Invariant depth provenance: synchronize `engine_eval.requested_depth` with root `requested_depth`. |
| `mcp_server/models/mcpeval.py` | Evaluation Models | Synchronize nested `engine_eval` dictionary upon requested depth updates. |
| `scripts/audit/case_generation.py` | Stress Case Generator | Expand test matrix to 1,000+ cases including Legal's Mate, contradictory PGNs, machine annotations. |
| `scripts/audit/semantic_oracles.py` | Semantic Invariant Oracles | Add validators for Invariants A, B, D, E, F, and depth provenance. |
| `scripts/chess_mcp_stress.py` | Stress Test Harness | Support in-process local execution of the expanded test matrix with 100% pass guarantee. |

---

## Tasks

### Task 1: Fix P0-1 (Forced / Engine-Best Move Graded as Blunder)

**Files:**
- Modify: `mcp_server/analysis/move_grading/strategies/transitions.py:255-275`
- Test: `tests/test_p0_move_grading_invariants.py`

- [ ] **Step 1: Write failing regression tests for Invariants A and B**

```python
# tests/test_p0_move_grading_invariants.py
import chess
import pytest
from mcp_server.analysis.move_classifier import classify_move_with_engine
from mcp_server.engine.analyzer_pool import AnalyzerPool

@pytest.mark.asyncio
async def test_legals_mate_forced_king_move_never_blunder() -> None:
    # Legal's Mate position: Black king on e8 has only Ke7 or loses queen, mate unavoidable
    # r1bqkb1r/pppp1ppp/2n5/4P3/4n3/5N2/PPP2PPP/RNBQKB1R w KQkq - 1 5
    # After 5. Nxe5 Bxd1 6. Bxf7+ Ke7 (forced only move)
    board = chess.Board("r1bq1b1r/ppppkPpp/2n5/8/4n3/8/PPP2PPP/RNBQKB1R w - - 1 7")
    # Even if position is lost, played move == engine best move must have effective_loss == 0
    # and move_class == "best"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p0_move_grading_invariants.py -v`
Expected: FAIL (evaluated as blunder due to `mover_mate_after < 0` in `score_cp_to_mate`).

- [ ] **Step 3: Implement minimal fix in `transitions.py`**

In `score_cp_to_mate`:
```python
# If the move is the engine's best move, or it is the only legal move available,
# regret relative to the best move must be zero (Invariant A & B).
if is_best_engine_move or (board is not None and len(list(board.legal_moves)) == 1):
    return TransitionScore(
        move_class=MoveClass.BEST,
        effective_loss=0,
        outcome_penalty=None,
        is_best_action=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p0_move_grading_invariants.py -v`
Expected: PASS.

---

### Task 2: Fix P0-2 (Engine Best Move Reported as Practically Non-Equivalent)

**Files:**
- Modify: `mcp_server/analysis/practical_equivalence.py:180-205`
- Test: `tests/test_p0_practical_equivalence.py`

- [ ] **Step 1: Write failing regression test for reflexive equivalence**

```python
# tests/test_p0_practical_equivalence.py
import pytest
from mcp_server.analysis.practical_equivalence import build_practical_equivalence_evidence

def test_engine_best_move_is_always_practically_equivalent() -> None:
    # When played move matches engine best move, it cannot be practically non-equivalent to itself
    ...
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p0_practical_equivalence.py -v`
Expected: FAIL (when mate-deterioration triggers, it returns `not_equivalent` even for engine best).

- [ ] **Step 3: Implement minimal fix in `practical_equivalence.py`**

In `build_practical_equivalence_evidence`:
Check `if result.is_best_engine_move or (result.played_move and result.played_move == result.best_engine_move):` at the top of the decision tree:
```python
if result.is_best_engine_move or (result.played_move and result.played_move == result.best_engine_move):
    return PracticalEquivalenceEvidence(
        status="equivalent",
        practical_equivalent=True,
        reason_codes=["ENGINE_BEST_MOVE"],
        ...
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p0_practical_equivalence.py -v`
Expected: PASS.

---

### Task 3: Fix P0-3 (Contradictory Winners Between PGN Header and Board Checkmate)

**Files:**
- Modify: `mcp_server/analysis/game_termination.py:80-160`
- Test: `tests/test_p0_game_termination_invariants.py`

- [ ] **Step 1: Write failing regression test for conflicting PGN headers**

```python
# tests/test_p0_game_termination_invariants.py
import pytest
from mcp_server.tools.analyze_game import analyze_game

@pytest.mark.asyncio
async def test_fools_mate_conflicting_pgn_header_board_outcome_authoritative() -> None:
    # PGN says [Result "1-0"] but movetext ends with 2... Qh4# (0-1 checkmate)
    pgn = '[Result "1-0"]\n\n1. f3 e5 2. g4 Qh4# 0-1'
    res = await analyze_game(pgn=pgn, depth=1)
    # Both top-level and coaching termination must report Black as winner
    assert res.termination.winner_side == "black"
    assert res.coaching.termination.winner_side == "black"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p0_game_termination_invariants.py -v`
Expected: FAIL (`res.coaching.termination.winner_side` reports `"white"` due to reading header `"1-0"`).

- [ ] **Step 3: Implement minimal fix in `game_termination.py`**

When `board.is_checkmate()` is true:
- Set `winner = "black" if board.turn == chess.WHITE else "white"`.
- Set `loser = "white" if board.turn == chess.WHITE else "black"`.
- Board terminal state overrides PGN header string; add warning flag `pgn_header_result_conflict=True` in metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p0_game_termination_invariants.py -v`
Expected: PASS.

---

### Task 4: Fix P1-4 (Machine PGN Directives Treated as Player Self-Reports)

**Files:**
- Modify: `mcp_server/analysis/game_coaching.py:250-280, 470-490`
- Test: `tests/test_p1_pgn_machine_directives.py`

- [ ] **Step 1: Write failing regression test for `[%clk]` and machine annotations**

```python
# tests/test_p1_pgn_machine_directives.py
import pytest
from mcp_server.tools.analyze_game import analyze_game

@pytest.mark.asyncio
async def test_machine_directives_do_not_trigger_player_self_report() -> None:
    pgn = '1. e4 {[%clk 0:05:00] [%eval +0.20]} e5 {[%clk 0:05:00]} 2. Nf3'
    res = await analyze_game(pgn=pgn, depth=1)
    for cm in res.coaching.critical_moments:
        assert cm.reason != "player_self_report"
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p1_pgn_machine_directives.py -v`
Expected: FAIL (`reason="player_self_report"` triggered on move 1).

- [ ] **Step 3: Implement minimal fix in `game_coaching.py`**

```python
_MACHINE_DIRECTIVE_RE = re.compile(r"\[%[A-Za-z0-9_]+(?:\s+[^\]]*)?\]")

def _clean_human_comment(raw_comment: str | None) -> str | None:
    if not raw_comment:
        return None
    cleaned = _MACHINE_DIRECTIVE_RE.sub("", raw_comment).strip()
    return cleaned if cleaned else None
```
Use `_clean_human_comment` before assigning `user_comment_raw` and before checking self-report priority.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p1_pgn_machine_directives.py -v`
Expected: PASS.

---

### Task 5: Fix P1-5 (Fivefold Repetition Metadata Coherence)

**Files:**
- Modify: `mcp_server/rules/status.py:35-50`
- Test: `tests/test_p1_repetition_invariants.py`

- [ ] **Step 1: Write failing regression test for repetition metadata**

```python
# tests/test_p1_repetition_invariants.py
import pytest
from mcp_server.rules.status import determine_rule_status

def test_fivefold_repetition_metadata_coherent() -> None:
    ...
    # assert status.requires_move_stack is True
    # assert status.repetition_sufficient_without_history is False
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p1_repetition_invariants.py -v`
Expected: FAIL (`repetition_sufficient_without_history` is True).

- [ ] **Step 3: Implement minimal fix in `status.py`**

In `make_rule_status`:
```python
repetition_sufficient_without_history = not history_dep
```
Pass this to `RuleStatus` constructor so fivefold and threefold repetition accurately report `False`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p1_repetition_invariants.py -v`
Expected: PASS.

---

### Task 6: Fix P2-8, P2-7 & P2-10 (Parameter Bounds, Verbosity Aliases, Forensic Projection)

**Files:**
- Modify: `mcp_server/tools/top_moves.py:80-120, 190-210`
- Modify: `mcp_server/tools/_common.py:40-70`
- Test: `tests/test_p2_tool_contracts.py`

- [ ] **Step 1: Write failing regression test for `proof_defenses <= 0` and verbosity aliases**

```python
# tests/test_p2_tool_contracts.py
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from mcp_server.tools.top_moves import top_moves

@pytest.mark.asyncio
async def test_top_moves_rejects_negative_proof_defenses_in_tactical_mode() -> None:
    with pytest.raises(ToolError) as exc_info:
        await top_moves(fen="startpos", proof_mode="tactical", proof_defenses=0)
    assert "[INVALID_ARGUMENT]" in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p2_tool_contracts.py -v`
Expected: FAIL (`proof_defenses=0` silently executed).

- [ ] **Step 3: Implement fixes in `top_moves.py` and `_common.py`**

1. In `top_moves.py`: validate `proof_defenses`:
   ```python
   if proof_mode == "tactical" and proof_defenses < 1:
       raise _tool_error("INVALID_ARGUMENT", "proof_defenses must be >= 1 when proof_mode='tactical'", tool="top_moves")
   ```
2. When `verbosity_mode in ("minimal", "compact")` and `detail == "standard"`, omit `forensics` block from `ForensicTopMovesResult` while keeping `result.result`.
3. Add `forensic_compute_triggered_by` metadata list when candidate evaluation triggers forensic enrichment.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p2_tool_contracts.py -v`
Expected: PASS.

---

### Task 7: Fix P2-9 (Nested Depth Provenance Parity)

**Files:**
- Modify: `mcp_server/engine/cached_evaluator.py:110-120, 170-176`
- Modify: `mcp_server/models/mcpeval.py:245-265`
- Test: `tests/test_p2_depth_provenance.py`

- [ ] **Step 1: Write cross-projection depth parity test**

```python
# tests/test_p2_depth_provenance.py
import pytest
from mcp_server.tools.evaluate_position import evaluate_position

@pytest.mark.asyncio
async def test_nested_engine_eval_requested_depth_parity() -> None:
    for verbosity in ("full", "compact", "standard", "default"):
        res = await evaluate_position("startpos", depth=1, verbosity=verbosity)
        assert res.requested_depth == 1
        if res.engine_eval:
            assert res.engine_eval["requested_depth"] == 1
        if res.eval_block and res.eval_block.engine_eval:
            assert res.eval_block.engine_eval["requested_depth"] == 1
```

- [ ] **Step 2: Run test to verify failure**

Run: `uv run pytest tests/test_p2_depth_provenance.py -v`
Expected: FAIL on cache hits where `engine_eval["requested_depth"]` was not updated with `raw_requested_depth`.

- [ ] **Step 3: Implement synchronization in `cached_evaluator.py` & `MCPEval`**

When updating `requested_depth` on `MCPEval`, if `self.engine_eval` is not None, copy and update `self.engine_eval["requested_depth"] = requested_depth` and `self.engine_eval["searched_depth"] = self.searched_depth`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_p2_depth_provenance.py -v`
Expected: PASS.

---

### Task 8: Expand Ultra Test Harness to 1,000+ Cases with Semantic Oracles

**Files:**
- Modify: `scripts/audit/case_generation.py`
- Modify: `scripts/audit/semantic_oracles.py`
- Modify: `scripts/chess_mcp_stress.py`

- [ ] **Step 1: Add new adversarial case generators in `case_generation.py`**
  - Add Legal's Mate test matrix across depths 1, 2, 4, 8 (`classify_move` and `analyze_game`).
  - Add conflicting PGN header cases (Fool's Mate, Scholar's Mate with inverted result headers).
  - Add machine comment PGN cases (`[%clk]`, `[%eval]`, `[%emt]`, pure and mixed).
  - Add repetition state cases (threefold, fivefold with move stacks).
  - Add parameter boundary fuzzing cases (`n`, `depth`, `proof_defenses`, `verbosity` aliases).
  - Expand `build_adversarial_case_specs()` to generate 1,000+ total test cases.

- [ ] **Step 2: Add semantic invariant oracles in `semantic_oracles.py`**
  - `oracle_invariant_a_engine_best_zero_regret()`: asserts engine best move always has 0 regret and `move_class == "best"`.
  - `oracle_invariant_b_forced_move_zero_regret()`: asserts single legal move is never a blunder.
  - `oracle_invariant_d_board_outcome_authoritative()`: asserts board checkmate overrides PGN header.
  - `oracle_invariant_e_no_machine_self_reports()`: asserts machine directives don't trigger `player_self_report`.
  - `oracle_invariant_f_repetition_coherence()`: asserts `requires_move_stack` vs `repetition_sufficient_without_history`.
  - `oracle_invariant_depth_provenance()`: asserts root requested depth == nested requested depth.

- [ ] **Step 3: Wire oracles and execute full local stress test**
  - Run the full 1,000+ test suite via `scripts/chess_mcp_stress.py` in-process with local pool.
  - Verify 0 transport errors, 0 invariant violations, and 100% pass rate.

---

### Task 9: Full Suite Verification & Final Artifacts

- [ ] **Step 1: Run full pytest suite**
  Run: `uv run pytest -v`
  Expected: All existing (1,257+) and new tests pass.

- [ ] **Step 2: Run pyright strict**
  Run: `uv run pyright`
  Expected: 0 errors.

- [ ] **Step 3: Run ruff linter**
  Run: `uv run ruff check .`
  Expected: All checks passed.

- [ ] **Step 4: Update documentation & audit log**
  - Update `docs/superpowers/plans/2026-09-11-chess-mcp-309-qa-hardening.md`.
  - Update `walkthrough.md` with verification outputs.

---

## Execution Handoff

Plan complete and saved to artifact. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch subagents per task, review between tasks, fast parallel/sequential iteration.
2. **Inline Execution** - Execute tasks in this session with review checkpoints.

Which approach do you prefer?
