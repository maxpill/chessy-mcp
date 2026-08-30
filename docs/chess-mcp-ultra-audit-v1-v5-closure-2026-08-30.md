# Chess MCP ultra-audit v1-v5 closure

Date: 2026-08-30
Repository: `maxpill/chessy-mcp`
Original audited revision: `faaedb8375e091dfd70ad0491cc8236181d5b7aa`
Merged v4 remediation: PR #1, merge commit `0491c9881a82f80a6552d59f86e633851b31f2f6`
Follow-up stress/remediation: PR #2, branch `test/chess-mcp-ultra-stress-v5-2026-08-30`

## Purpose

This file consolidates the findings from the four 2026-08-30 ultra-audit reports plus the v5 stress follow-up. It distinguishes historical defects from their current remediation status so old reports do not remain misleading after fixes land.

The original audit campaign included 160 live Chess MCP calls against the old revision and multiple subsequent 150+ source/test/deployment inspection rounds. The v5 follow-up adds deterministic generated-game stress, parser matrices, rule boundaries, cache invariants, production hardening checks and real-Stockfish CI coverage.

## Closure status

### Procedural action contract

- FIXED: arbitrary `classify_move.action_type` values are rejected.
- FIXED: unavailable immediate draw claims are rejected instead of falling through to move grading.
- FIXED: intended draw claims require the exact legal claim-enabling move.
- FIXED: pawn/capture resets cannot masquerade as intended fifty-move claims.
- FIXED: typed `played_action_obj` represents the requested action on both classify paths.
- FIXED: `played_action_obj.type` is validated against `action_type`.
- FIXED: `is_best_action` cannot remain true for a different non-equivalent action.
- FIXED: `is_engine_best` and `is_best_engine_move` are synchronized.
- FIXED: intended claim reasons are represented per intended UCI move instead of cross-producting unrelated moves and reasons.
- FIXED: safe execution is separated from engine recommendation through `executable_move` / typed best action.

### History provenance and repetition

- FIXED: history provenance is first-class: `complete`, `partial`, `incomplete`, `not_required`.
- FIXED: `startpos` with zero plies is complete history.
- FIXED: complete PGN/movetext reconstructed from the game root is complete history.
- FIXED: arbitrary naked FEN is incomplete history.
- FIXED: arbitrary FEN plus a continuation is partial, not complete.
- FIXED: partial history may prove repetition but cannot falsely disprove missing pre-FEN repetition.
- FIXED: internal evaluation no longer defaults to fabricated complete history.
- FIXED: semantic cache keys include history provenance.
- FIXED: reversible-history fingerprints are derived from actual history rather than `(id(board), len(move_stack))` memoization.
- FIXED: ponder preserves the board move stack and provenance.
- FIXED: compact verbosity removes payload only and does not rewrite history truth.
- FIXED: pure fifty-move claims no longer set `repetition_status=threefold_claimable`.
- FIXED: full-game analysis threads complete history through evaluation and metrics.

### Root action policy and grading

- FIXED: root draw-claim policy is centralized and shared between position tools.
- FIXED: optional `can_claim_draw()` in the lower core classifier is no longer treated as automatic game termination.
- FIXED: draw claim forfeiture and opponent-claim concession remain explicit grading mechanisms.
- FIXED: the old best-move PV-tail shortcut in `classify_move` was removed; the immediate post-move position is evaluated instead.
- FIXED: failed deeper candidate verification no longer fabricates a BEST result.

### WDL and candidate semantics

- FIXED: TCP WDL is normalized to White POV, including win/loss reversal for black-to-move roots.
- FIXED: subprocess WDL is normalized through the White POV object.
- FIXED: subprocess and TCP analyzer creation honor WDL and Syzygy configuration consistently.
- FIXED: WDL percentages use decimal scaling and no longer lose one percentage point through floor division.
- FIXED: candidate `post_position.winner` uses the candidate-specific rule state.
- FIXED: top-move candidates are ranked in mover POV while their public CP/mate contract remains White POV.
- FIXED: nested top-move candidates now carry the same build/engine identity as the outer result.

### FIDE edge rules

- FIXED: K+B versus bare K is correctly treated as unable to mate.
- FIXED: K+N versus bare K is unable to mate, while opponent material is conservatively considered because it can block escape squares.
- FIXED: time-forfeit, resignation and rules-infraction result normalization uses the mating-possibility helper.
- FIXED: mate precedence over the 75-move automatic draw is retained.
- FIXED: threefold/fivefold and 50/75-move current/intended claim boundaries are covered by regressions.

### PGN, SAN, FEN and metadata

- FIXED: free-form `Termination` result inference uses winner/loser grammar rather than color+keyword co-occurrence.
- FIXED: `White wins on time`, `Black wins on time`, lost-on-time and resignation variants are not inverted.
- FIXED: `Normal time control - White/Black` does not invent a decisive result.
- FIXED: `_find_movetext_result` is correctly typed as optional.
- FIXED: Unicode result markers, Unicode hyphens, figurines, NAG, comments, RAV and conversational wrappers remain supported.
- FIXED: strict PGN parsing is actually forwarded through `_build_board`.
- FIXED: position-oriented endpoints reject hidden movetext after automatic game termination instead of silently swallowing it.
- FIXED: permissive `analyze_game` may retain its documented tolerant behavior by ignoring post-terminal junk with an explicit warning; strict mode rejects it.
- FIXED: v5 found that python-chess may sanitize invalid raw castling rights; the public input layer now validates the raw castling field and required king/rook placement before accepting the FEN.
- FIXED: unknown verbosity values are returned through the structured `INVALID_VERBOSITY` tool-error contract instead of escaping the normal error path.

### Engine pool, transport, cache and overload resistance

- FIXED: EnginePool background self-heal increments `_alive_count` only when a replacement is accepted and cannot overgrow target cardinality.
- FIXED: excess recovery workers are discarded.
- FIXED: TCP reconnect re-applies persistent UCI options while correctly resetting MultiPV tracking.
- FIXED: cache logic hashing covers action, transport, analyzer, grading and win-probability logic-bearing files in addition to server/rules/models/cache.
- FIXED: `BUILD_SHA` is injected into production build/runtime and participates in identity/cache versioning.
- FIXED: compute admission/rate limiting is weighted rather than charging every tool request equally.
- FIXED: trusted-proxy handling avoids blindly trusting arbitrary `X-Forwarded-For` clients.
- FIXED: application authorization requires the configured token; User-Agent/Origin strings are not credentials.
- FIXED in v5 follow-up: official MCP DNS-rebinding Host/Origin validation is enabled by default and production has explicit allowlists.
- FIXED: request body buffering has an explicit size limit.

### Regression suite quality and CI

- FIXED: old R-18/R-19 en-passant tests now prove board semantics rather than passing through tautologies.
- FIXED: R-21 castling regression no longer swallows arbitrary exceptions.
- FIXED: R-44 black ranking checks actual order rather than only result presence.
- FIXED: self-heal tests check target cardinality.
- FIXED: permanent audit-v4 closure tests cover all 41 requested v4 invariants.
- FIXED: permanent follow-up tests cover build identity, repetition metadata, termination grammar, WDL/Syzygy parity and immediate post-move evaluation.
- FIXED: v5 adds deterministic legal-sequence replay, PGN round trips, malformed FEN matrices, history/cache/action matrices and real Stockfish tests.
- FIXED: GitHub CI runs compile, Ruff, full pytest and strict Pyright with Stockfish installed.

### Documentation and deployment drift

- FIXED: README pool-size documentation matches the actual fallback and production setting.
- FIXED: Caddy documentation no longer claims `analyze_game` emits per-position progress events; `flush_interval` is described as proxy buffering control.
- FIXED: `.env.example` documents token authentication, build identity and DNS Host/Origin settings.
- FIXED: production Compose declares the corresponding authentication, Host/Origin and build settings.

## v5 clean verification

After the residual v1-v3 closure changes and removal of temporary remediation tooling, GitHub `Chess MCP CI` runs on the PR merge ref with a real Stockfish package installed.

Current clean verification target:

- Python compileall
- Ruff
- full pytest suite
- strict Pyright for `mcp_server` and `core`
- real Stockfish integration tests inside pytest

The v5 suite reached 624 passing tests with strict Pyright reporting zero errors and zero warnings on the cleaned PR branch.

## Remaining operational caveat

The live developer Chess MCP connector is an external deployed integration and can lag a repository branch until the relevant commit reaches deployment. Repository/CI correctness therefore does not by itself prove that a currently connected external MCP instance is already running the branch under review. Live post-deploy smoke tests should be repeated after deployment.

## Release position

All concrete defects recorded in audit reports v1-v4 are either fixed by merged PR #1 or closed by the v5 follow-up branch. The branch also closes residual issues that appeared only in the earlier v1-v3 reports, notably DNS rebinding, post-terminal PGN handling, nested candidate observability and stale operational documentation.

The codebase is now suitable to be treated as an authoritative analysis/action service subject to normal limitations of finite-depth engine search and the deliberately conservative FIDE mating-possibility predicate. Any future behavioral change in action semantics, history provenance, FIDE rule projection, WDL perspective, result inference or transport identity should add a regression before merge.
