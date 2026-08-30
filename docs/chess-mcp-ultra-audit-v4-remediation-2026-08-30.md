# Chess MCP ultra-audit v4 remediation report

Date: 2026-08-30

Repository: `maxpill/chessy-mcp`

Audit baseline: `faaedb8375e091dfd70ad0491cc8236181d5b7aa`

Remediation branch: `fix/chess-mcp-ultra-audit-2026-08-30`

This document is the durable closure record for the Chess MCP ultra-audit v4. It combines the original runtime/source findings, the remediation decisions, the permanent regression coverage, and the final GitHub Actions verification state.

## 1. Scope and verification model

The original audit combined a large live Chess MCP stress campaign with source-level inspection of the server, rules engine, action model, caches, local/TCP Stockfish adapters, engine pool, parser, metadata reconciliation, HTTP boundary, deployment configuration, and regression suite.

The remediation deliberately separates four kinds of correctness:

1. Board correctness: legal moves, terminal states, repetition, 50/75-move rules, castling, en passant, promotions, mate/stalemate and dead positions.
2. Procedural correctness: draw claims are actions, not merely board states. `claim_draw`, `claim_draw_with_intended_move`, and `play_move` must remain structurally distinct.
3. Epistemic correctness: a naked FEN does not contain arbitrary pre-FEN repetition history. The service must distinguish what it knows from what it cannot know.
4. Service correctness: cache identity, transport POV, worker recovery, authentication, rate limiting, build identity and CI must not invalidate otherwise correct chess logic.

The closure suite therefore tests not only individual outputs, but cross-tool invariants and information provenance.

## 2. P0 action-boundary findings

### 2.1 Free-form `action_type`

Original problem: `classify_move.action_type` accepted arbitrary strings. A value such as `banana` could reach ordinary classification code.

Fix:

- public `action_type` is constrained to `Literal["play_move", "claim_draw", "claim_draw_with_intended_move"]`;
- runtime validation remains in place as defense in depth;
- invalid action types map to a dedicated `invalid_action_type` tool error.

Permanent regression: `test_01_unknown_action_type_rejected`.

Status: fixed.

### 2.2 Illegal claims silently falling through to move grading

Original problem: an unavailable claim could fall through into ordinary `play_move` scoring. This changed the meaning of the caller's request.

Fix:

- `claim_draw` is rejected unless `can_claim_now` is true;
- `claim_draw_with_intended_move` is rejected unless the exact supplied move is in the legal intended-claim set;
- a pawn move or capture that resets the 50-move counter cannot be accepted as an intended 50-move claim;
- typed action builders also reject impossible action payloads.

Permanent regressions: tests 02 through 05 in `test_audit_v4_closure.py`.

Status: fixed.

### 2.3 `is_best_action` contradicting the requested action

Original problem: a move could be engine-best while the declared procedural action was not the canonical best game action, yet `is_best_action` could still be true.

Fix:

- `MCPMoveAnalysis` enforces that `is_best_action=true` requires matching action types or explicit action equivalence;
- engine-best move status is kept separate from best legal game-action status.

Permanent regression: test 07.

Status: fixed.

## 3. History provenance redesign

### 3.1 Incorrect `bool(moves)` model

Original problem: history completeness was derived from whether an optional move suffix was non-empty. This produced both false positives and false negatives:

- arbitrary FEN + one move was called complete;
- full PGN with no separate `moves` argument could be called incomplete;
- the initial position was treated as unknown history even though zero prior plies is a complete history.

Fix: first-class provenance states:

- `complete`: start position with its complete continuation, or a PGN/movetext game reconstructed from its root;
- `partial`: arbitrary FEN plus a supplied continuation;
- `incomplete`: arbitrary naked FEN;
- `not_required`: history-independent terminal facts such as checkmate or stalemate.

Permanent regressions: tests 09 through 14.

Status: fixed.

### 3.2 Partial history may prove but not disprove repetition

Original problem: a supplied suffix could incorrectly be used to assert `repetition_status=none` even though repetition before the supplied FEN was unknowable.

Fix:

- partial/incomplete history can report `threefold_claimable` if the supplied stack itself proves it;
- absence of proof under partial/incomplete history yields `repetition_status=unknown`, not `none`;
- unresolved repetition now explicitly sets `requires_move_stack=true`, `history_dependent_status=true`, and `fen_sufficient_for_status=false`.

Permanent regressions: tests 13, 14 and follow-up `test_unknown_repetition_explicitly_requires_move_stack`.

Status: fixed.

### 3.3 Cache-order contamination

Original problem: semantic evaluations could be reused across different history-certainty states because cache identity did not encode provenance. Calling endpoints in a different order could change the reported rule state.

Fix:

- `eval_cache_key`, `top_moves_cache_key`, and `classify_cache_key` include history completeness;
- reversible-history fingerprints use actual board history rather than an unsafe `(id(board), len(move_stack))` memo key;
- call-order invariance is tested explicitly.

Permanent regressions: tests 15 and `test_extra_cache_history_provenance_is_part_of_key`.

Status: fixed.

### 3.4 Ponder losing history

Original problem: the ponder path reconstructed a FEN-only board and could warm cache with falsely complete history.

Fix:

- ponder passes a stack-preserving `board.copy(stack=True)`;
- provenance is carried explicitly into the warm-cache call and cache key.

Permanent regression: test 16.

Status: fixed.

### 3.5 Compact verbosity changing truth

Original problem: compact serialization overwrote history values, turning `unknown/incomplete` into `none/complete`.

Fix:

- compact mode only removes verbose/duplicated presentation fields;
- semantic fields are preserved exactly;
- unknown verbosity values are rejected instead of silently falling back.

Permanent regression: test 17 plus existing verbosity regressions.

Status: fixed.

## 4. Draw-rule semantics

### 4.1 50-move claim reported as threefold

Original problem: a generic `can_claim_draw` branch could set `repetition_status=threefold_claimable` even when the only reason was the 50-move rule.

Fix:

- repetition status depends only on repetition evidence;
- 50-move claimability and repetition metadata are independent.

Permanent regression: test 18.

Status: fixed.

### 4.2 Intended-claim reason cross-product

Original problem: intended moves and claim reasons were stored as unrelated global lists. Mixed 50-move/threefold positions could advertise a reason for a move that did not create that reason.

Fix:

- rule state stores `intended_claim_reasons_by_uci`;
- typed legal actions are generated from exact move-to-reason mappings.

Permanent regression: `test_extra_intended_claim_reasons_are_bound_per_move`.

Status: fixed.

### 4.3 Optional claims treated as automatic terminal draws in core classifier

Original problem: transport-independent move classification treated `can_claim_draw()` as if the game had automatically ended.

Fix:

- only automatic game-over states short-circuit to terminal draw handling;
- procedural claims remain the responsibility of the action layer.

Permanent regression: test 33.

Status: fixed.

## 5. Cross-tool action policy

Original problem: `evaluate_position` and `top_moves` could recommend different root actions for the same claimable position at the same search depth.

Fix:

- one canonical action-selection policy is used for claim-vs-play decisions;
- typed `best_action_obj` exposes the chosen procedural action consistently.

Permanent regression: test 19 and real-Stockfish claim-policy integration test.

Status: fixed.

## 6. Typed action response contract

Original problems:

- direct classify path could omit `played_action_obj` / `best_action_obj`;
- alternate constructor could build a `play_move` payload for a draw claim;
- duplicate `is_engine_best` / `is_best_engine_move` booleans could diverge.

Fix:

- one typed played-action builder handles play, immediate claim and intended-move claim;
- model validation requires `played_action_obj.type == action_type`;
- engine-best aliases are synchronized;
- `is_best_action` is corrected if action semantics do not match.

Permanent regressions: tests 06 through 08.

Status: fixed.

## 7. WDL perspective and representation

### 7.1 TCP WDL perspective

Original problem: CP/mate were converted to White POV, but raw TCP WDL was copied without swapping wins/losses for black-to-move positions.

Fix:

- TCP WDL is normalized to White POV;
- black-to-move `(W,D,L)` becomes `(L,D,W)`.

Permanent regressions: tests 20 and 21.

Status: fixed.

### 7.2 Local subprocess WDL perspective and configuration parity

Original problem: local subprocess score used `.white()` while WDL/configuration behavior drifted from TCP. `show_wdl` and Syzygy settings were not forwarded through the local pool factory.

Fix:

- local WDL uses White POV;
- `Analyzer.create()` accepts `show_wdl` and `syzygy_path`;
- it configures `UCI_ShowWDL`, `SyzygyPath` and `SyzygyProbeLimit` when requested;
- `AnalyzerPool.create()` and the server pool factory forward the same options as the TCP path.

Permanent regression: `test_local_analyzer_pool_forwards_wdl_and_syzygy`.

Status: fixed.

### 7.3 WDL percentages summing to 99

Original problem: floor division converted permille values to integer percentages and could sum to 99.

Fix:

- percentages use `/ 10.0`, preserving the exact 100% total for a valid 1000-permille tuple.

Permanent regression: test 22.

Status: fixed.

## 8. Candidate post-position winner

Original problem: a top-move candidate computed candidate-specific rule state but nested `post_position.winner` could read the root winner.

Fix: post-position winner comes from the candidate's own rule status.

Permanent regression: test 23.

Status: fixed.

## 9. FIDE mating-possibility edge rules

### 9.1 K+B vs bare K

Original problem: the helper treated any bishop as potentially mating and therefore could preserve a win on time/resignation when mate was impossible.

Fix: K+B vs bare K is recognized as unable to mate.

Permanent regressions: tests 24 and 26.

Status: fixed.

### 9.2 K+N with opponent material

Original problem: K+N was treated as universally unable to mate, ignoring that opponent material can occupy escape squares in a possible legal mating sequence.

Fix: the predicate is conservative. It returns false only when impossibility is proven. K+N vs bare king remains impossible, while opponent non-king material prevents the false-negative draw conversion.

Permanent regression: test 25.

Status: fixed.

## 10. Termination and result inference

Original problem: free-form termination inference used broad color+keyword matching. Winner-oriented phrases such as `White wins on time` could be inverted. `Normal time control - White` could also accidentally imply a result.

Fix:

- explicit winner and loser grammar;
- `wins` and past-tense `won` are both recognized;
- `White/Black wins/won on time` normalizes to `time_forfeit`;
- `Normal time control` is explicitly non-forfeit and does not infer a result;
- winner-oriented time text still participates in FIDE mating-possibility validation.

Permanent regressions: tests 27 through 30 and follow-up winner/past-tense tests.

Status: fixed.

## 11. Best-move post-position evaluation

Original problem discovered during remediation: `classify_move` tried to avoid a second search for the engine's best move by replaying the root PV tail from the wrong frame. Even when repaired mechanically, using a later PV position or copying the root score would not be a sound representation of the immediate post-move position at finite depth.

Fix:

- server classification always obtains an evaluation of the actual immediate `board_after`;
- the transport-independent core classifier does the same for active positions;
- when the played move is the engine's root best move, centipawn loss remains zero by definition, but `eval_after` is no longer fabricated from the root result;
- cache/TT layers remain responsible for performance reuse rather than changing semantics.

Permanent regressions:

- `test_core_best_move_uses_real_immediate_post_evaluation`;
- `test_server_best_move_eval_after_is_immediate_post_position` using a real Stockfish binary.

Status: fixed.

## 12. EnginePool self-heal cardinality

Original problem: background recovery could enqueue a new worker without incrementing `_alive_count`, causing subsequent heal cycles to create extra workers indefinitely.

Fix:

- cardinality updates are protected by a lock;
- `_alive_count` increases only when a new worker is successfully accepted;
- excess fresh workers are discarded;
- both synchronous replacement and background healing respect `_target_size`.

Permanent regressions: tests 31 and 32.

Status: fixed.

## 13. Cache invalidation and build identity

### 13.1 Logic-bearing file coverage

Original problem: semantic cache versioning omitted files whose changes can alter evaluations/actions.

Fix: logic hash covers action, rules, model, server, TCP client/analyzer, local analyzer/core analysis/grading and win-probability logic.

Permanent regressions: tests 38 and 39.

Status: fixed.

### 13.2 Production BUILD_SHA

Original problem: Docker image did not necessarily have `.git`, so `git rev-parse` could return unknown despite deployment knowing the build SHA.

Fix:

- Docker/Compose propagate `BUILD_SHA`;
- server build metadata consumes the env value;
- cache `_git_sha()` now also consumes `BUILD_SHA` / `CHESSY_BUILD_SHA` before falling back to local git.

Permanent regressions: build-injection test in closure suite and follow-up cache build-SHA test.

Status: fixed.

## 14. HTTP authentication and overload controls

### 14.1 Spoofable ChatGPT/OpenAI User-Agent lock

Original problem: strings such as `User-Agent: openai` could satisfy the application-level lock.

Fix:

- when restriction is enabled, a valid configured token is the authentication credential;
- User-Agent/Origin are not treated as credentials.

Permanent regression: test 40.

Status: fixed at application layer. External reverse-proxy policy remains an infrastructure concern outside this repository.

### 14.2 Compute-unaware rate limiting

Original problem: a cheap depth-1 position eval and a long depth-30 game analysis consumed the same request budget.

Fix: admission cost depends on tool and expensive parameters such as depth, MultiPV count and estimated game size.

Permanent regression: test 41.

Status: fixed.

### 14.3 Forwarded client identity

The HTTP layer was hardened so forwarded client identity is only trusted according to configured proxy assumptions. Deployment must continue to ensure that the public path is through the trusted reverse proxy.

Status: hardened in application; infrastructure trust remains deployment-specific.

## 15. Parser and test-suite false positives

### 15.1 En passant tests

Original problem: tests could pass from a tautological `total_plies == 5` fallback without proving en passant board semantics.

Fix: final board is checked directly: capturing pawn on d6, d5 empty, final UCI `e5d6`. `e.p.` normalization is also reported.

Status: fixed.

### 15.2 Castling alias test

Original problem: a broad `except Exception: pass` allowed a broken parser to pass.

Fix: both `O-O` and `0-0` are parsed on legal fixtures and final king/rook squares are asserted.

Status: fixed.

### 15.3 Black ranking test

Original problem: the test only asserted that some candidate existed.

Fix: controlled White-POV evaluations are supplied for a black-to-move position and exact Black-utility ordering is asserted.

Status: fixed.

### 15.4 Strict typing

Original problem: strict typing was advertised but not continuously proven, and `_find_movetext_result` had an optional return without an optional annotation.

Fix:

- return type is `str | None`;
- `pyproject.toml` uses Pyright strict for `mcp_server` and `core`;
- CI runs Pyright on every PR/main change.

Permanent regression: test 34 plus CI.

Status: fixed.

## 16. Permanent closure suite mapping

The audit requested at least 41 explicit regressions. `tests/test_audit_v4_closure.py` maps directly to that list:

1. unknown action type rejected
2. unavailable immediate claim rejected
3. wrong intended-claim move rejected
4. pawn reset cannot create intended 50-move claim
5. capture reset cannot create intended 50-move claim
6. played typed action matches requested action
7. best-action boolean obeys action identity/equivalence
8. engine-best aliases agree
9. startpos zero-ply history complete
10. full PGN history complete
11. arbitrary FEN history incomplete
12. arbitrary FEN + suffix partial
13. partial history can prove threefold
14. partial history cannot disprove missing earlier repetition
15. endpoint/cache order invariant
16. ponder preserves history stack/provenance
17. compact mode preserves semantic values
18. 50-move claim does not become threefold status
19. cross-tool root action policy agrees
20. WDL White POV for White to move
21. WDL White POV for Black to move
22. WDL percentage total
23. candidate post-position winner
24. K+B vs K cannot mate
25. K+N mating possibility considers opponent material
26. time-forfeit K+B vs K normalizes to draw
27-29. winner/loser termination grammar
30. normal time-control text does not infer result
31-32. self-heal count/no-overgrowth
33. optional claim is not automatic terminal in core classifier
34. optional return-type contract
35. castling alias exceptions cannot be swallowed
36. en passant board semantics
37. Black candidate ranking
38-39. cache version tracks action/transport semantics
40. spoofed OpenAI/ChatGPT User-Agent is not authentication
41. rate-limit cost scales with work

Additional closure tests cover per-move intended-claim reasons, provenance in cache keys and BUILD_SHA propagation.

`tests/test_audit_v4_followup.py` then covers remediation findings discovered while closing v4, especially cache build identity, unresolved repetition metadata, past-tense result grammar, correct immediate post-position evaluation and local analyzer configuration parity.

## 17. Real Stockfish coverage

The permanent integration suite uses an installed Stockfish binary rather than mocks for:

- `evaluate_position` contract;
- distinct legal `top_moves`;
- classifying Stockfish's own cached best move;
- short full-game analysis;
- cross-tool claim-policy consistency.

The follow-up suite adds a real-engine regression proving that `classify_move.eval_after` describes the immediate post-move FEN for the engine's best move.

## 18. Final CI gate

Permanent workflow: `.github/workflows/ci.yml`.

Required checks:

- compile all production/test Python;
- Ruff on `mcp_server` and `core`;
- complete pytest suite with Stockfish installed;
- strict Pyright on `mcp_server` and `core`.

Final clean branch verification after removing all temporary remediation workflows/scripts:

- branch head: `bd3e33d4dccfa3b3914e11e8d9c10a58d174cc50`;
- GitHub Actions run: `33318022730`;
- compile: pass;
- Ruff: pass;
- pytest: **332 passed**;
- Pyright strict: **0 errors, 0 warnings**.

The run executed on GitHub's synthetic PR merge ref against the audit baseline `main`, which also verifies clean integration with current `main`.

## 19. Final assessment

All concrete repository-level findings listed by audit v4 have either been fixed and regression-tested or, where the concern depends on external infrastructure, the application-side boundary has been hardened and the remaining trust assumption documented.

The most important architectural changes are not isolated patches. They are invariant changes:

- actions are typed and procedural;
- history certainty is explicit;
- unknown history is not silently converted into certainty;
- cache identity carries semantic provenance;
- all tools share the same root-action decision model;
- engine evaluation is White-POV consistently across transports;
- move classification describes the real immediate post-move position;
- worker recovery cannot exceed pool cardinality;
- authentication is not inferred from spoofable client strings;
- CI permanently enforces the closure contract.

The remediation branch remains intentionally unmerged until explicit merge approval.