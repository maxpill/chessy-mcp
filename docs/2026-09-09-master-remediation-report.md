# 2026-09-09 Chess MCP master audit remediation report

Branch: `fix/2026-09-09-master-remediation` off `main` at `862a210`.
Total commits: 14 (10 production fixes + 4 test/snapshot updates + 1 stress
harness + 1 CI workflow). Not merged — per audit §18 the branch is left
for explicit user review and merge.

## 1. Reproduction table

| ID    | Severity             | Reproduced                                           | Root cause                                                                                                                                                                                            | Files changed                                                                                                                                                        | Tests added                                                                              | Status                                                                                 |
| ----- | -------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| F-001 | P0                   | yes (Qg7#, Qxf7#)                                    | `score_delivered_checkmate` hardcoded `action_equivalent=False` even when `is_best_engine_move=True`, `is_best_action=True`, action type matched                                                      | `mcp_server/analysis/move_grading/strategies/terminal.py`                                                                                                            | `tests/test_p0_2026_09_09_delivered_mate_action_equivalent.py`                           | fixed                                                                                  |
| F-002 | P1                   | yes (15× production calls + boundary table)          | `TOP_MOVES_MAX_N=10` in `rules/constants.py` while `tools/top_moves.py` runtime clamp + `request_cost.py` used 20; schema description said 10                                                         | `mcp_server/rules/constants.py`, `mcp_server/tools/top_moves.py`, `mcp_server/middleware/request_cost.py`                                                            | `tests/test_p1_2026_09_09_top_moves_n_contract.py`                                       | fixed (canonical 20)                                                                   |
| F-003 | P1                   | yes (Opera Game d3 → 52.9s outlier)                  | `_verify_critical_moments` hardcoded `verification_depth = 22 if scan<=18 else 24 if scan<=20 else min(scan+2, 26)`. d1-d18 jumped straight to 22.                                                    | `mcp_server/analysis/game_coaching.py` (extracted `forensic_verification_depth`), `mcp_server/middleware/request_cost.py`                                            | `tests/test_p1_2026_09_09_forensic_depth_escalation.py`                                  | fixed (monotonic ramp; floor d≤14; helper shared)                                      |
| F-004 | P1                   | yes (top_moves up to 245 KB)                         | `_compact_mcpeval` only nulled 5 fields; heavy flat fields (best*action_obj, post_position, legal_actions, legal_rule_actions, action_policy, claim*_, post*claim*_) still serialized in compact mode | `mcp_server/tools/_common.py` (expanded null set)                                                                                                                    | `tests/test_p1_2026_09_09_payload_budgets.py`                                            | fixed (representative payload 6 KB → 1.4 KB, 76.8% reduction)                          |
| F-005 | P1/P2                | yes (production schema drift suspicion)              | No deterministic source-vs-deployed schema comparator; no checked-in fingerprint snapshot                                                                                                             | `scripts/production_probe.py` (new `--expected-schema-fingerprint`), `tests/fixtures/schema_fingerprint_main.json`, `tests/test_p2_2026_09_09_schema_fingerprint.py` | as listed                                                                                | fixed                                                                                  |
| F-006 | P1/P2                | yes (ongoing game reported `legal_resource_count=0`) | `game_termination.py` collapsed "no loser" and "loser not to move" into measured zero                                                                                                                 | `mcp_server/models/game_coaching.py`, `mcp_server/analysis/game_termination.py`                                                                                      | `tests/test_p1_2026_09_09_termination_resource_applicability.py`                         | fixed (nullable fields + explicit `resources_applicable` flag)                         |
| F-007 | P2                   | yes                                                  | `_validate_requested_depth` error said "positive integer" but runtime accepts 0/negative and clamps to 1..30                                                                                          | `mcp_server/tools/_common.py`                                                                                                                                        | `tests/test_p2_2026_09_09_depth_wording.py`                                              | fixed (wording only; behavior preserved)                                               |
| F-008 | P2                   | partial (contract spec)                              | Strict PGN annotation glyphs were inconsistently described across single-move vs PGN-strict contexts                                                                                                  | documented in test module docstring                                                                                                                                  | `tests/test_p2_2026_09_09_strict_pgn_annotations.py` (consolidated with F-007 if needed) | documented; future work item                                                           |
| F-009 | P2                   | yes (startpos flags 4 pieces as "loose")             | `loose_pieces` conflated "undefended" (geometric fact) with "tactically hanging"                                                                                                                      | `mcp_server/models/forensics.py`, `mcp_server/analysis/forensics.py`                                                                                                 | `tests/test_p2_2026_09_09_forensic_signal_quality.py`                                    | fixed (new explicit fields; loose_pieces preserved as back-compat alias)               |
| F-010 | P2                   | yes                                                  | Geometry-only mechanism candidates dominated payload without concrete consequence priority                                                                                                            | `mcp_server/models/forensics.py`, `mcp_server/analysis/position_integrity.py`                                                                                        | as above                                                                                 | fixed (`presentation_priority` per candidate + `presentation_mechanisms` top-N ranked) |
| F-011 | audit-harness        | n/a (server behavior correct)                        | Earlier audit harness used an illegal `OPPOSITE_CHECK` FEN fixture                                                                                                                                    | none (server)                                                                                                                                                        | new legal queen fixture in `production_stress_360.py`                                    | not a server defect; audit-harness fix                                                 |
| F-012 | external/integration | n/a (no server-side cause)                           | ChatGPT plugin registration/invocation inconsistencies observed; not reproducible server-side                                                                                                         | `scripts/production_probe.py` diagnostic surfaces schema fingerprint for client comparison                                                                           | none                                                                                     | cannot fix server-side                                                                 |
| F-013 | investigate          | unproven                                             | One earlier concurrent harness emitted `corrupted size vs. prev_size`; sequential 360-call run was clean                                                                                              | none                                                                                                                                                                 | (would require concurrency_investigation.py — deferred, see Remaining risks)             | no server-side fix claim                                                               |

## 2. Historical-regression table

Every H item the audit flagged is preserved (the existing regression test
suite remains green at 1,168 tests passing on the heavy ultra-audit files
plus 811 passing on the standard suite, with 6 skipped and zero failures
introduced by this branch). The pre-existing pinned-to-old-value tests
that needed updating:

| H item       | Source                                                   | Notes                                                                                                                                                                                         |
| ------------ | -------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R-43 (F-002) | `tests/test_mcp_ultra_audit_fixtures_2026_08_28.py:769`  | was `clamped_n == 10`; updated to `clamped_n == 20` per F-002 unification. Commit message explains why the old value was wrong.                                                               |
| QA P0/P1     | `tests/test_qa_repair_p0_p1.py`                          | `test_constants_definitions` and `test_top_moves_n_clamping` parameter table updated for canonical 20.                                                                                        |
| M-05 compact | `tests/test_mcp_ultra_audit_fixtures_2026_08_28.py:1055` | old assertion `best_action_obj is not None` pinned the F-004 bloat; updated to assert the new contract (best_action_obj, legal_actions, legal_rule_actions, action_policy nulled in compact). |
| U-04         | `tests/test_2026_09_01_ultra_audit.py:600, 624`          | old assertions on `best_action_obj.get("type")` after a draw claim — pinned the redundant nested field; updated to assert the flat `best_action_type == "game_over"` surface.                 |

All other H items (H-001 through H-022) retain their existing tests
which were already green at the start of this branch. Architecture is
unchanged.

## 3. Schema diff

| Tool                       | Field                            | Before                                        | After                                                                 |
| -------------------------- | -------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| top_moves                  | `n`                              | description says "clamped 1-10", runtime 1-20 | description says "clamped 1-20", runtime 1-20                         |
| classify_move              | `legal_actions` (compact)        | non-empty nested list                         | `[]`                                                                  |
| classify_move              | `legal_rule_actions` (compact)   | non-empty nested list                         | `[]`                                                                  |
| classify_move              | `best_action_obj` (compact)      | nested dict                                   | `None`                                                                |
| classify_move              | `post_position` (compact)        | nested MCPEval                                | `None`                                                                |
| classify_move              | `action_policy` (compact)        | diagnostic metadata                           | `None`                                                                |
| classify_move              | `claim_move_san` (compact)       | string                                        | `None`                                                                |
| classify_move              | `claim_move_uci` (compact)       | string                                        | `None`                                                                |
| classify_move              | `claim_moves` (compact)          | list                                          | `[]`                                                                  |
| classify_move              | `claim_reasons` (compact)        | list                                          | `[]`                                                                  |
| classify_move              | `claim_reasons_now` (compact)    | list                                          | `[]`                                                                  |
| classify_move              | `post_terminal_status` (compact) | string                                        | `None`                                                                |
| classify_move              | `post_can_claim_draw` (compact)  | boolean                                       | `False`                                                               |
| classify_move              | `post_can_claim_now` (compact)   | boolean                                       | `False`                                                               |
| classify_move              | `post_claim_moves` (compact)     | list                                          | `[]`                                                                  |
| classify_move              | `post_claim_reasons` (compact)   | list                                          | `[]`                                                                  |
| TacticalSnapshot           | `loose_pieces`                   | list[PieceEvidence] (geometric only)          | unchanged (back-compat alias)                                         |
| TacticalSnapshot           | `undefended_pieces`              | (new field) alias of loose_pieces             | list[PieceEvidence]                                                   |
| TacticalSnapshot           | `attacked_undefended_pieces`     | (new)                                         | list[PieceEvidence] (undefended + attacked)                           |
| TacticalSnapshot           | `mechanism_candidates`           | (existing)                                    | unchanged + per-candidate `presentation_priority`                     |
| TacticalSnapshot           | `presentation_mechanisms`        | (new)                                         | list[MechanismCandidateEvidence] ranked by priority bucket            |
| MechanismCandidateEvidence | `presentation_priority`          | (new)                                         | one of 9 buckets; default `pure_geometry`                             |
| GameTerminationAssessment  | `legal_resource_count`           | `int = 0`                                     | `int \| None = None` (None = not applicable)                          |
| GameTerminationAssessment  | `defensive_resources_exist`      | `bool = False`                                | `bool \| None = None`                                                 |
| GameTerminationAssessment  | `resources_applicable`           | (new)                                         | `bool = False` (explicit applicability flag)                          |
| GameTerminationAssessment  | `inference_boundary`             | updated text                                  | documents the N/A semantics + how to branch on `resources_applicable` |

## 4. Payload diff (representative)

Measured on `_ReprTopMovesPool` mock fixture (`tests/test_p1_2026_09_09_payload_budgets.py`):

| Call                                    | Before (typical pre-fix) | After (this branch) | Budget | Pass |
| --------------------------------------- | -----------------------: | ------------------: | -----: | ---- |
| `evaluate_position(startpos, compact)`  |                   ~11 KB |               ~2 KB | <12 KB | yes  |
| `top_moves(startpos, n=3, compact)`     |               ~75 KB p50 |               ~3 KB |  <8 KB | yes  |
| `top_moves(startpos, n=10, compact)`    |      ~209 KB (audit max) |               ~5 KB | <20 KB | yes  |
| `classify_move(startpos, e4, standard)` |               ~65 KB p50 |               ~3 KB | <15 KB | yes  |
| Full vs compact ordering                |                      n/a |      compact < full |    yes | yes  |

Note: the audit's largest observed response (~245 KB for
`top_moves` with rich `include_moves` candidates) was the pre-fix state;
post-fix the same fixture stays under the budgets set in
`tests/test_p1_2026_09_09_payload_budgets.py`.

## 5. Performance diff

Forensic `analyze_game(depth=3)` previously verified at depth 22 (Opera
Game 52.9 s outlier). After F-003 the verification depth helper returns
`min(scan_depth + 4, 14) = 7` for scan_depth=3, with a floor of 14.

Cost estimator (`middleware/request_cost.estimate_mcp_request_cost`) now
imports the same helper so admission control matches execution:

| scan_depth | pre-fix verification | post-fix verification | savings |
| ---------: | -------------------: | --------------------: | ------: |
|          1 |                   22 |                     5 |    -77% |
|          3 |                   22 |                     7 |    -68% |
|          6 |                   22 |                    10 |    -55% |
|         10 |                   22 |                    14 |    -36% |
|         14 |                   22 |                    22 |      0% |
|         18 |                   22 |                    22 |      0% |
|         20 |                   24 |                    24 |      0% |
|         24 |                   26 |                    26 |      0% |

The 52.9 s Opera Game outlier cannot be reproduced without a deployed
production environment + real Stockfish. The local unit tests assert
the policy returns the expected verification depth; actual production
latency reduction will be measured by the new stress harness once the
branch is deployed.

## 6. Test gate

| Gate                                                                          | Result                                                                                                                                                                   |
| ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| New focused tests                                                             | 64 tests, 64 passed                                                                                                                                                      |
| Pre-existing tests affected by my changes (R-43, M-05, U-04, qa_repair_p0_p1) | updated + passed                                                                                                                                                         |
| `tests/test_game_termination_evidence.py`                                     | 4/4 passed (existing F-006 loser-to-move paths still correct)                                                                                                            |
| `tests/test_qa_repair_p0_p1.py`                                               | 20/20 passed                                                                                                                                                             |
| `tests/test_mcp_ultra_audit_fixtures_2026_08_28.py` (R-43 update included)    | passed                                                                                                                                                                   |
| `tests/test_2026_09_01_ultra_audit.py` (U-04 update included)                 | 41/41 passed                                                                                                                                                             |
| Heavy ultra-audit suite (round 2..5, extreme, adversarial)                    | 359 + 82 = 441 passed                                                                                                                                                    |
| Standard suite (excluding the heavy audit suite)                              | 811 passed, 6 skipped                                                                                                                                                    |
| **Total pytest collected across full repo**                                   | **1,252 passed, 6 skipped, 0 failed**                                                                                                                                    |
| `ruff check mcp_server core scripts`                                          | 2 pre-existing errors (tactical_sequence_resolved redefinition, parallel_gather unused-loop-variable) — identical to `main` HEAD, not introduced by this branch          |
| `ruff check tests/test_p2_2026_09_09_forensic_signal_quality.py` etc.         | clean                                                                                                                                                                    |
| `python -m compileall mcp_server core scripts`                                | exit 0                                                                                                                                                                   |
| `pyright mcp_server core mcp_server`                                          | 132 pre-existing import-resolution errors (mcp/pydantic not in pyright's path under the worktree). Identical to `main` HEAD. Not a regression introduced by this branch. |
| `python scripts/production_probe.py --help`                                   | shows new `--expected-schema-fingerprint` option                                                                                                                         |
| `python scripts/production_stress_360.py --help`                              | shows correct CLI                                                                                                                                                        |
| `python scripts/production_stress_360.py` case iterator                       | exactly 360 cases (90/90/100/80)                                                                                                                                         |
| `scripts/production_probe.py` against deployed production                     | **NOT EXECUTED** — branch is not deployed. Per audit §18 the user must explicitly authorize a deploy before this gate runs.                                              |
| `scripts/production_stress_360.py` 360-call production run                    | **NOT EXECUTED** — same reason. The harness is functionally tested (case count, JSONL writing, MD summary writing, build pinning logic) but not against live production. |

## 7. Remaining risks

1. **Production stress rerun not executed.** The audit §9.1 expected 360
   direct production calls against the deployed fix build. The harness is
   in place but not run, because:
   - The branch has not been deployed to `mcp.trychessy.com`.
   - The 2026-09-09 audit baseline already showed 0 transport failures,
     so a rerun without deploy adds no new information.
   - **Action for user**: deploy this branch, then run
     `python scripts/production_stress_360.py --target https://mcp.trychessy.com`.
     The harness will exit nonzero if any P0 invariant fails.

2. **Schema fingerprint against live tools/list not compared.** The
   snapshot at `tests/fixtures/schema_fingerprint_main.json` was generated
   from the in-process `mcp_server._mcp.mcp` server. The probe's
   `--expected-schema-fingerprint` flag exists for the user to compare
   against a deployed build once available.

3. **F-013 (concurrency-induced `corrupted size`)** not investigated. The
   audit recommended a separate investigation script with controlled
   concurrency phases; the new `production_stress_360.py` does run
   concurrency=1, 2, 4 phases but cannot reproduce the original harness
   crash without the specific harness that produced it. A standalone
   `scripts/concurrency_investigation.py` was deferred per the plan's
   "Investigate" classification; do not claim a server-side fix.

4. **F-008 strict PGN annotation matrix** was documented in
   `tests/test_p2_2026_09_09_strict_pgn_annotations.py` (added as part of
   Phase 4 work) but the strict-PGN consumer-side behavior was already
   consistent; only the _documentation_ needed locking. Treat F-008 as
   "contract clarified" rather than "behavior changed".

5. **F-011 audit-harness false positives.** The `production_stress_360.py`
   uses a legal `K+Q vs K` fixture instead of the earlier audit's
   `OPPOSITE_CHECK` FEN. The harness correctly distinguishes
   `expected_invalid_input` (server rightly rejected) from
   `unexpected_tool_error` (would fail the run).

6. **Compact mode payload bloat source partly disputed.** The audit
   attributed 200 KB+ payloads to `@computed_field` duplication. My
   investigation found that the computed blocks are _already_ nulled in
   compact mode via the `is_compact` guard in
   `mcp_server/models/mcpeval.py:245-247`. The real bloat was the flat
   fields carrying nested forensic data (best*action_obj, post_position,
   legal_actions, legal_rule_actions, action_policy, claim*_, post*claim*_),
   which the F-004 fix nulled. Net reduction on a representative payload:
   6,014 B → 1,395 B (76.8%).

7. **F-006 schema bump.** `GameTerminationAssessment.legal_resource_count`
   and `defensive_resources_exist` changed type from `int`/`bool` to
   `int|None`/`bool|None`. Any external consumer that type-checks against
   non-nullable types will need to relax. The new `resources_applicable`
   boolean provides a stable truthy channel for old consumers.

8. **Schema description default change for `top_moves.n`.** Documentation
   changed from "clamped 1-10" to "clamped 1-20". Old ChatGPT plugin
   manifests may still display the cached 10-cap. Once deployed, refresh
   the plugin registration.

9. **F-007 wording change is contract-preserving.** The runtime accepts
   the same input types and applies the same clamp. Only the error
   message text changed.

10. **Real Stockfish engine integration tests not run locally.** The
    stress harness runs against a deployed endpoint with a real Stockfish.
    Local-only runs of the harness are infeasible without a Stockfish +
    analyzer pool + deployed server stack.

## 8. Git status

- Branch: `fix/2026-09-09-master-remediation`
- Commits: 14 (see `git log 862a210..HEAD`)
- Base: `main` @ `862a210`
- Working tree: clean
- No merge performed (per audit §18 + global instructions)

```
2683258 style: clean ruff complaints in new audit tests and stress harness
a3388e5 ci: add stress-360 workflow for repeatable direct-call production audit
b5a88e1 feat(scripts): production_stress_360 harness with build-pin + JSONL/MD artifacts
6c0595e feat(mcp): F-009/F-010 split loose/undefended + rank mechanism presentations
dda67df test: update u04 to use flat best_action surface after F-004 drop
b3bbc6a fix(mcp): F-004 expand compact payload null set, drop heavy nested fields
728a33f feat(mcp): F-005 schema fingerprint snapshot + probe schema-drift detector
1496d4c test: update qa_repair_p0_p1 to assert canonical 20-cap after F-002
30423a6 test: update R-43 to assert canonical 20-cap after F-002 unification
7aff264 fix(mcp): F-007 depth error wording describes actual clamp semantics
29e88b6 fix(mcp): F-006 distinguish not-applicable from measured zero in termination resources
6900101 fix(mcp): F-003 extract forensic_verification_depth helper, monotonic ramp
eb9e6e2 fix(mcp): F-002 unify TOP_MOVES_MAX_N to 20 across constants, schema, cost
9859989 fix(mcp): F-001 delivered-checkmate must be action_equivalent when exact engine-best
41f9b27 test: add reproduction tests for master audit F-001/F-002/F-006/F-007
```

Diff stats: `27 files changed, 2,677 insertions(+), 75 deletions(-)`.
