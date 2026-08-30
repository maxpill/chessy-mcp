# Chess MCP post-audit v5 closure report

Date: 2026-08-30

Repository: `maxpill/chessy-mcp`

Base main revision: `0491c9881a82f80a6552d59f86e633851b31f2f6`

Remediation branch: `fix/chess-mcp-post-audit-v5-2026-08-30`

This follow-up started after the full ultra-audit v4 remediation had already been merged to `main`. It focuses on residual edge cases found by another source review of the corrected implementation.

## Verification constraint

The Chess MCP developer connector was requested for a fresh runtime pass, but the current conversation environment rejected the invocation with `FORBIDDEN: This conversation does not support developer MCPs`. No live MCP result is claimed in this report.

Runtime verification for this follow-up therefore comes from GitHub Actions using the repository test suite with a real Stockfish binary installed, plus strict static checks.

## Finding V5-01: procedural claims could inherit consequences of a placeholder move

### Problem

`score_played_move()` evaluated post-move checkmate branches before the procedural draw-claim branch.

For `claim_draw`, the supplied move is only a placeholder required by the classifier API. For `claim_draw_with_intended_move`, the move identifies the intended claim move, but the procedural claim is made instead of executing that move.

If the supplied placeholder/intended move happened to give checkmate, the old ordering could classify a legal claim request as a mating `play_move`. That violates the typed action contract and could also expose a meaningless raw board delta for an action that did not actually play a move.

### Fix

- post-move checkmate consequence branches now run only for `action_type == "play_move"`;
- legal draw claims remain in the procedural claim branch even when the supplied move would checkmate if executed;
- `raw_centipawn_delta` for a procedural claim is zero because no board move is executed by that action.

### Regressions

`tests/test_post_audit_v5.py` verifies both immediate and intended claims with a mating supplied move.

Status: fixed.

## Finding V5-02: PGN TimeControl validation rejected valid standard forms

### Problem

Metadata validation accepted only simple values such as:

- `300`
- `300+5`
- `40/7200`

It rejected valid compound or hourglass forms such as:

- `40/7200:3600`
- `40/7200:3600+30`
- `40/7200:20/3600:900+30`
- `*60`

This did not alter chess moves, but it generated false metadata warnings for valid PGNs.

### Fix

A dedicated `_is_valid_pgn_time_control()` validates colon-separated stages. Each stage may be:

- sudden death seconds: `300`;
- moves/seconds: `40/7200`;
- Fischer seconds+increment: `300+5`;
- hourglass: `*60`.

The standard unknown/unspecified markers `?` and `-` remain accepted.

Malformed forms are explicitly covered by negative tests.

Status: fixed.

## Finding V5-03: protected HTTP POSTs were buffered before authentication

### Problem

The middleware previously read and buffered a POST body, then calculated request cost, then authenticated the caller.

An unauthenticated client could therefore make the server receive and buffer a large body before receiving HTTP 403. The hard body cap prevented unbounded memory use, but authentication should reject such a caller before body work begins.

Token comparison also used ordinary string equality.

### Fix

- protected non-health requests authenticate from headers before reading the body;
- shared-secret comparison uses `hmac.compare_digest()`;
- an oversized declared `Content-Length` is rejected with HTTP 413 without reading the request body;
- streamed/chunked bodies remain protected by the existing incremental hard cap;
- the `/health` endpoint remains unauthenticated by design.

### Regressions

Tests prove:

- an unauthenticated POST is rejected before `receive()` is called;
- an authenticated request declaring a body above `MAX_BUFFERED_BODY` receives HTTP 413 before `receive()` is called.

Status: fixed.

## Verification

Temporary remediation workflow run `33325289471` completed successfully with Stockfish installed.

Results before committing the fixes:

- new v5 regressions: `20 passed`;
- full repository suite: `352 passed`;
- Ruff: all checks passed;
- Pyright strict: `0 errors, 0 warnings, 0 informations`;
- compileall: passed.

The verified implementation commit is:

`562cee5459d1c21bfa1ff14321eee599b1601fd5`

Temporary patching workflow and script were removed after the verified commit so they are not part of the intended final production change set.

## Relationship to audit v4

The v4 remediation remains the authoritative closure record for its original 41 required regressions and broader findings. This v5 document records only the additional issues found after v4 reached `main`.

The final merge gate for v5 is the permanent `.github/workflows/ci.yml` workflow on the pull request, which runs compile, Ruff, the complete pytest suite with Stockfish available, and strict Pyright.