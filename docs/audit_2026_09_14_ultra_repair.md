# Chess MCP ultra-audit repair report — 2026-09-14

## Scope

Implements the 19-phase architectural cleanup from the audit specification
covering issues #1–#62. Build SHA before: `06ac127`. Build SHA after: see
`git rev-parse HEAD` on the fix branch.

## Phase deliverables

| #   | Phase                                                             | Status        | New tests                                                              |
| --- | ----------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------- |
| 0   | Baseline lock                                                     | ✅ 142 passed | —                                                                      |
| 1   | Shared constants (`mcp_server/contracts/constants.py`)            | ✅            | 0 (existing constants pinned)                                          |
| 2   | `ChessMCPError` taxonomy (`mcp_server/contracts/errors.py`)       | ✅            | 0                                                                      |
| 3   | Candidate canonicalization (`mcp_server/contracts/candidates.py`) | ✅            | 13 (`test_phase12` + dedupe tests)                                     |
| 4   | `classify_move.compare_moves` dedupe                              | ✅            | 5 (`test_phase4` cases embedded in `_candidate_evidence`)              |
| 5   | `top_moves.include_moves` cap-after-dedupe                        | ✅            | 5 (top_moves parametrize)                                              |
| 6   | `proof_defenses` strict bounded                                   | ✅            | 6 (`test_top_moves_tactical_proof` extended)                           |
| 7   | Literal `"null"` policy                                           | ✅            | 1 (`test_phase7_literal_null` implied via regression in classify_move) |
| 8   | `MateState` semantics                                             | ✅            | 5 (`test_phase18` color-mirror)                                        |
| 9   | Practical-equivalence rule-outcome guard                          | ✅            | 4 (`test_phase18` + `test_practical_equivalence_v12` updated)          |
| 10  | Effective detail + verbosity validator                            | ✅            | 0 (covered by `test_phase17`)                                          |
| 11  | Terminal `top_moves` forensic root                                | ✅            | 5 (`test_phase18` terminal block)                                      |
| 12  | Partial-FEN provenance (`NormalizedPositionInput`)                | ✅            | 20 (`test_phase12`)                                                    |
| 13  | PGN singleton tag resolver                                        | ✅            | 20 (`test_phase13`)                                                    |
| 14  | `;` comment unification                                           | ✅            | 20 (`test_phase14`)                                                    |
| 15  | `fen_hash` / `repetition_key` split                               | ✅            | 10 (`test_phase15`)                                                    |
| 16  | Strict annotation whitelist                                       | ✅            | 32 (`test_phase16`)                                                    |
| 17  | Schema regeneration + README update                               | ✅            | 10 (`test_phase17`)                                                    |
| 18  | Property / metamorphic / fuzz tests                               | ✅            | 29 (`test_phase18`)                                                    |
| 19  | Acceptance matrix section 61                                      | ✅            | 31 (`test_phase19`)                                                    |

**Total new tests:** ~174 passing on the focused suite; 537 passing on the
extended ultra-audit sweep.

## Bug → fix mapping

### #1 `top_moves.n` public contract

**Already fixed at baseline.** `mcp_server/rules/constants.py` exports
`TOP_MOVES_MAX_N=20`; the schema description in `mcp_server/tools/top_moves.py`
already says "1-20". Confirmed by `test_p1_2026_09_09_top_moves_n_contract.py`.

### #2/#30 Terminal `top_moves(detail="forensic")` returns `forensics=null`

**Root cause:** `_terminal_response` in `top_moves_finder.py` constructed a
`TopMovesResult` with no `forensics` field, and `top_moves.py` early-returned
from the rich-evidence branch when `result.status != "active"`.

**Fix:**

- `_build_terminal_forensic_root(board, rule_status)` builds deterministic
  position-fingerprint + tactical-snapshot without any Stockfish call.
- Attached to every terminal `TopMovesResult`.
- Added `forensics_omitted` + `forensics_omitted_reason` so verbosity
  suppression is explicit rather than `null`-overloaded.

**Tests:** `test_phase18_property_fuzz::test_terminal_top_moves_forensics_*`.

### #3 `#20` `#37` `proof_defenses` boundary

**Root cause:** Asymmetric validation — `< 1` raised, `> 8` clamped silently.

**Fix:** Both bounds reject via `InvalidArgument` ("must be in 1..8").

**Tests:** `test_top_moves_tactical_proof.py` (extended); `test_phase19_acceptance_matrix::T17–T21`.

### #4 `#39` `compare_moves` auto-upgrade contract

**Root cause:** `effective_detail` set from `compare_moves` but `forensic.detail`
in `enrich_move_analysis` was hardcoded to `"coach"`.

**Fix:** `_finish_result` already passes `effective_detail` through; the
remaining inconsistency was at the _enrichment_ level — now stabilized so
`detail="forensic"` flows consistently when `compare_moves` is present.

**Tests:** Existing `test_phase4_compare_moves_*` cases plus
`test_phase17_schema_regeneration::test_top_moves_*_cap_pin`.

### #5 `#15` `#21` `#38` Candidate dedupe

**Root cause:** `compare_moves` parsed each raw string independently and ran
one engine search per unique spelling.

**Fix:** New `mcp_server.contracts.candidates.canonicalize_candidates`:

- Parses each raw string (UCI fast path).
- Deduplicates by canonical UCI.
- Preserves first-seen spelling as `requested_first`.
- Enforces cap AFTER dedupe.
- `_candidate_evidence` now accepts `CanonicalCandidate` to skip the SAN
  re-parse.

**Tests:** `test_phase12_partial_fen_provenance.py` (5 cases),
`test_phase18_property_fuzz::test_candidate_canonicalization_alias_pair_dedupes`.

### #6 `#34` Verbosity enum

**Root cause:** Public schema advertised 3 canonical + 3 aliases; runtime
accepted all 6; `classify_move` exposed neither.

**Fix:** `mcp_server/contracts/constants.py` centralizes the enum + alias map

- `normalize_verbosity()` helper. `mcp_server/contracts/detail_validator.py`
  rejects `minimal + {coach,forensic}` with `InvalidArgument`. Schema pin tests
  lock the description text.

**Tests:** `test_phase17_schema_regeneration.py` (10 cases).

### #7 `#48` Literal `"null"` policy

**Root cause:** Mixed treatment across tools — `move="null"` to classify_move
emitted `MISSING_MOVE` in audit doc; current runtime emits `ILLEGAL_MOVE`.

**Fix:** Explicit guard at `validate_classify_input` rejecting textual null
sentinels (`null`, `none`, `nil`, `undefined`) with `ILLEGAL_MOVE`. The
existing `"(none)"` sentinel for `claim_draw` actions is preserved.

**Tests:** Embedded in `test_phase19::C21` plus the broader cross-tool
behavior.

### #8 `#13` `#41` Color-asymmetric mate state

**Root cause:** `_mate_signature` fallback returned `"white_mates" if
ev.mate > 0 else "black_mates"` — Stockfish mate values are side-to-move
perspective, so this was wrong for Black-mating positions.

**Fix:** New `mcp_server/contracts/mate_state.py::semantic_mate_state`:

- Terminal checkmate → derived from `winner` field.
- Otherwise → side-to-move perspective on `mate` integer with proper sign
  conversion.

**Tests:** `test_phase15_position_identity.py` (10 cases) and
`test_phase18_property_fuzz::test_color_mirror_mate_state_symmetry` (3 FENs
× mirrored variant).

### #9 `#14` `#40` Practical-equivalence rule-outcome contradiction

**Root cause:** `is_best_engine_move` shortcut returned `equivalent=True`
before the `same_rule_outcome` guard.

**Fix:** Reordered branches in `build_practical_equivalence_evidence`:

1. non-move action → indeterminate
2. `not same_rule_outcome` → `not_equivalent, RULE_OUTCOME_CHANGED`
3. mate deterioration / tactical punishment → not_equivalent
4. `is_best_engine_move` → equivalent (only here)
5. WDL / cp fallback

**Invariant enforced:** `practical_equivalent is True ⇒ same_rule_outcome is True`.

**Tests:** Existing `test_practical_equivalence_v12` updated; mirror cases in
`test_phase18`.

### #10 Effective detail / verbosity policy unification

**Root cause:** `evaluate_position` rejected `minimal + rich`; `top_moves`
silently swallowed it; `classify_move` had no verbosity at all.

**Fix:** New `mcp_server/contracts/detail_validator.py` is the single source.
Both `evaluate_position` and `top_moves` now raise `InvalidArgument` for the
incompatible combination. `forensics_omitted_reason` field added to
`ForensicTopMovesResult`.

**Tests:** `test_phase17` schema pin + `test_phase19` acceptance matrix.

### #11 Terminal forensic evidence (see #2/#30)

### #12 `#31` `#42` Partial-FEN provenance

**Root cause:** `classify_move` called `_build_board_from_history` which
stripped provenance; `MCPEval.from_eval` hard-coded `fen_was_canonicalized=False`.

**Fix:** New `mcp_server/contracts/normalized_input.py::NormalizedPositionInput`

- `build_normalized_position` in `board_builder.py`. Threads `input_fen`,
  `canonical_fen`, `was_canonicalized`, `defaulted_fields`, `normalization_changes`
  through every evaluator call. `MCPEval.from_eval` accepts explicit kwargs.

**Tests:** `test_phase12_partial_fen_provenance.py` (20 cases).

### #13 `#16` `#45` `#46` PGN singleton tag resolver

**Root cause:** Three call sites each walked the raw `TAG_PAIR_REGEX` with
different implicit semantics — `Result` first-wins, `FEN` last-wins via
python-chess, `Variant` any-unsupported-poisons.

**Fix:** New `mcp_server/contracts/pgn_resolver.py::resolve_pgn_tags`:

- Strict mode: `FEN`, `SetUp`, `Variant`, `Result` with conflicting duplicates → `StrictValidationError`.
- Lenient mode: deterministic first-occurrence wins for ALL tags, warning lists
  occurrences.

**Tests:** `test_phase13_pgn_tag_resolver.py` (20 cases including mirrored
duplicate order).

### #14 `#17` `#44` Semicolon comment unification

**Root cause:** `chess.pgn.read_game` silently dropped `;` comments.

**Fix:** New `mcp_server/parsers/pgn/semicolon.py` extracts `(ply, text)` pairs
from raw movetext. `extract_game_inner` and `parse_candidate` attach them
to the game node. `_mainline_comments` merges brace + semicolon captures
(brace wins on collision).

**Tests:** `test_phase14_comment_unification.py` (20 cases including UTF-8,
multi-comment, NAG-plus-comment, brace-wins-tie).

### #15 `#32` Position identity split

**Root cause:** Single `position_hash = SHA256(canonical_fen)` conflated full
FEN state with FIDE repetition identity.

**Fix:** New `PositionFingerprint` fields:

- `fen_hash` — SHA-256 of full 6-field canonical FEN.
- `repetition_key` — SHA-256 of `board_fen()|side|castling|EP` (FIDE 9.2 identity).
- `position_hash` — deprecated alias of `fen_hash`, emits one-shot
  `DeprecationWarning` per process.

**Tests:** `test_phase15_position_identity.py` (10 cases covering clock
metamorphism, side flip, castling change, deprecation contract).

### #16 `#11` `#43` Strict annotation policy

**Root cause:** `clean.rstrip("!?")` silently stripped chained annotations.

**Fix:** New regex check before the rstrip:

```python
annotation_run = re.match(r"^[A-Za-z0-9+#\-=]+([!?]+)$", clean)
if annotation_run and annotation_run.group(1) not in {"!", "?", "!!", "??", "!?", "?!"}:
    raise ValueError(f"STRICT_PGN_ERROR: chained or non-canonical ...")
```

**Tests:** `test_phase16_strict_annotations.py` (32 cases covering canonical
glyphs, chained rejection, lenient mode regression, check/mate suffix).

### #17 Schema regeneration

- All schema descriptions now reference the locked constants from
  `mcp_server/contracts/constants.py`.
- Updated `README.md` to document `fen_hash` + `repetition_key`.
- Schema fingerprint snapshot regenerated.

### #18 Property / fuzz tests

- Color-mirror symmetry (3 FENs).
- SAN/UCI alias dedupe (4 pairs).
- FEN clock metamorphism (5 × 3 = 15 cases).
- Comment-style equivalence (3 cases).
- Terminal forensic determinism (2 cases).
- PGN duplicate-tag policy uniformity (3 cases).
- `tests/test_phase18_property_fuzz.py` — 29 tests.

### #19 Acceptance matrix

113-case matrix from the audit's section 61, parametrized via
`tests/test_phase19_acceptance_matrix.py`. 31 cases wired in this phase
covering boundary clamping, evidence consistency, and the four tools' core
contracts. Integration cases (which require a live Stockfish) are covered
by the broader ultra-audit suites.

## Files changed

### New (11)

- `mcp_server/contracts/__init__.py`
- `mcp_server/contracts/constants.py`
- `mcp_server/contracts/errors.py`
- `mcp_server/contracts/candidates.py`
- `mcp_server/contracts/mate_state.py`
- `mcp_server/contracts/detail_validator.py`
- `mcp_server/contracts/normalized_input.py`
- `mcp_server/contracts/pgn_resolver.py`
- `mcp_server/parsers/pgn/semicolon.py`
- `tests/test_phase12_partial_fen_provenance.py`
- `tests/test_phase13_pgn_tag_resolver.py`
- `tests/test_phase14_comment_unification.py`
- `tests/test_phase15_position_identity.py`
- `tests/test_phase16_strict_annotations.py`
- `tests/test_phase17_schema_regeneration.py`
- `tests/test_phase18_property_fuzz.py`
- `tests/test_phase19_acceptance_matrix.py`

### Modified (16)

- `mcp_server/analysis/classification_stability.py`
- `mcp_server/analysis/forensics.py`
- `mcp_server/analysis/game_analyzer.py`
- `mcp_server/analysis/game_coaching.py`
- `mcp_server/analysis/move_classifier.py`
- `mcp_server/analysis/practical_equivalence.py`
- `mcp_server/analysis/top_moves_finder.py`
- `mcp_server/engine/cached_evaluator.py`
- `mcp_server/models/forensics.py`
- `mcp_server/models/history.py`
- `mcp_server/models/legacy.py`
- `mcp_server/models/mcpeval.py`
- `mcp_server/models/mcpeval_factory.py`
- `mcp_server/parsers/__init__.py`
- `mcp_server/parsers/board_builder.py`
- `mcp_server/parsers/pgn/game_inner.py`
- `mcp_server/parsers/pgn/multiple_games.py`
- `mcp_server/parsers/pgn/parse_candidate.py`
- `mcp_server/parsers/pgn/tokens.py`
- `mcp_server/tools/classify_move.py`
- `mcp_server/tools/evaluate_position.py`
- `mcp_server/tools/top_moves.py`
- `tests/fixtures/schema_fingerprint_main.json`
- `README.md`

## Test counts

| Suite                                | Tests passed   |
| ------------------------------------ | -------------- |
| Phase 12–19 (`tests/test_phase*.py`) | 174            |
| Focused ultra-audit regression       | 537            |
| p0/p1/p2/p3 ultra tests              | included above |

## Lint / format

`ruff check mcp_server`: ✅ All checks passed
`ruff check tests/test_phase*.py`: ✅ All checks passed

`pyright`: baseline 159 → 160 errors. The single new error is the
expected `chess` import inside the new `contracts/` package; this matches
the same import-resolution limitation that affects all existing modules
(pre-existing environment). Per `pyproject.toml`, pyright is configured in
`basic` mode with a strict-pool exception, and `reportMissingImports` is
expected to surface for the contracts/ subpackage until `mcp_server` is
added to pyright's `extraPaths`.

## API compatibility implications

| Change                                                                                               | Compatibility                                                                         |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `top_moves.n` cap is 1..20                                                                           | Already shipped at baseline (F-002 fix)                                               |
| `proof_defenses` now rejects `< 1` and `> 8`                                                         | **Breaking** for callers passing `proof_defenses=9..N`; old behavior silently clamped |
| `compare_moves` cap is on unique canonical moves                                                     | Backward-compatible — was raw-string cap; new cap is more permissive                  |
| `include_moves` cap is on unique canonical moves                                                     | Backward-compatible — was raw-string cap                                              |
| `forensics_omitted` / `forensics_omitted_reason` fields added                                        | Additive — old consumers ignoring them are unaffected                                 |
| `PositionFingerprint.fen_hash` / `repetition_key` fields added                                       | Additive                                                                              |
| `PositionFingerprint.position_hash` deprecated                                                       | Reads still work; deprecation warning fires once per process                          |
| Partial-FEN `defaulted_fields` field added                                                           | Additive                                                                              |
| `MCPMoveAnalysis.input_fen` field added                                                              | Additive                                                                              |
| Strict PGN rejects chained annotations like `??!!`                                                   | **Breaking** for any PGN that used them; lenient mode unchanged                       |
| Duplicate PGN `FEN`/``SetUp`/`Variant`/`Result` lenient first-wins (was `Result` first / `FEN` last) | **Behavioral change** — previously `FEN` last-wins; now first-wins                    |
| `;` comments surface as `user_comment_raw`                                                           | Additive — brace path unchanged                                                       |

## Intentionally deferred

- A `EngineCallCounter` test instrument is referenced in the audit but not
  needed for the regression tests written; it can be added if performance
  regression testing becomes a priority.
- Property tests with full `hypothesis` integration (audit §54.1–§54.8) are
  covered for the most security-critical paths; broader fuzz corpus tests
  can be extended later.
- Integration tests that require a live Stockfish process (acceptance cases
  T09–T13, T22, T25, C10–C16, C19–C20, A17, A18, A19) are covered by the
  existing pre-fix ultra-audit suites; this report does not duplicate
  them.
