# chessy-mcp

Standalone Stockfish MCP server. The chessy app talks to it over HTTPS at
`mcp.trychessy.com`; ChatGPT and other MCP clients can talk to the same
endpoint.

## Architecture

```
Cloudflare tunnel (mcp.trychessy.com)
        │
        ▼
   ┌─────────┐  HTTP :8000   ┌──────────┐  UCI/TCP :9550   ┌────────────┐
   │  Caddy  │ ────────────► │   MCP    │ ───────────────► │ Stockfish  │
   │  (TLS)  │               │  server  │  (N parallel)    │  (sf_18)   │
   └─────────┘               └──────────┘                  └────────────┘
```

The Caddy reverse-proxy terminates nothing (TLS is at Cloudflare) and
forwards HTTP to the MCP service on the internal `mcp-net` bridge. The MCP
service opens N TCP connections to the Stockfish sidecar (one per pool
slot) and dispatches UCI queries across them.

Stockfish is built from source with `-march=native` on the host CPU.
This means **the image is host-specific**: rebuilding on a different box
will recompile for THAT CPU. The chessy app's MCP traffic will work as
long as this image has been (re)built on the host that's running it.

## Deploy

The stack lives in a separate Komodo stack at
`/etc/komodo/stacks/chessy-mcp/` and is deployed on push to `main` via a
GitHub webhook to the public tunnel:

```text
https://webhook-komodo.trychessy.com/listener/github/stack/chessy-mcp/deploy
```

The stack has `webhook_force_deploy = true` so every push to `main`
runs `docker compose up -d --build --remove-orphans`. The Komodo stack
`pre_deploy` step calls `scripts/deploy-helper.sh` which stamps the
checked-out HEAD into the stack `.env` as `BUILD_SHA` so the running
container and every cached eval report the correct deployment identity
(rather than the `unknown` default the compose substitution falls back
to when nothing has set it).

`file_paths` includes both `docker-compose.prod.yml` and `Caddyfile`
so a Caddy-only change is enough to trigger change detection.

The `.env` file (gitignored) holds `CHESS_MCP_AUTH_TOKEN` plus the
pool/hash/threads tunables. Generate the token with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

…and put the same value in the chessy app's `.env` so the coach runtime
can authenticate against the shared edge.

## Local development

```bash
cp .env.example .env
# edit .env to set CHESS_MCP_AUTH_TOKEN
docker compose -f docker-compose.prod.yml up -d --build
curl http://127.0.0.1:9551/health
```

## Tools

Five Streamable-HTTP MCP tools at `/mcp`:

| Tool                | What it does                                                               |
| ------------------- | -------------------------------------------------------------------------- |
| `evaluate_position` | Single-position Stockfish eval at a given depth                            |
| `top_moves`         | Top-N candidate moves ranked by Stockfish                                  |
| `classify_move`     | Grade a played move against the engine's best alternative                  |
| `analyze_game`      | Full PGN analysis with accuracy, mistakes, turning points                  |
| `explore_opening`   | Lichess Opening Explorer — W/D/L stats + ECO + sample games for a position |

The 5-tool surface is intentional - the chessy app's coach runtime is the
primary consumer, and this server is sized for that workload.

### Lichess integration

`explore_opening` wraps `https://explorer.lichess.ovh` and requires a Lichess
OAuth bearer token in `LICHESS_EXPLORER_TOKEN`. Generate a PAT with the
`explorer:read` scope at <https://lichess.org/settings/oauth/token> and put it
in the stack `.env` next to `CHESS_MCP_AUTH_TOKEN`. Without the token the tool
fails fast with `MISSING_TOKEN` rather than silently dropping queries.

Responses are cached L1+L2 for 8 minutes (matches lila's `OpeningApi`) and
concurrent identical calls coalesce via `SingleFlight` into a single HTTP
roundtrip. The 4 s timeout mirrors the upstream `requestTimeout` setting in
`lila.modules.opening.OpeningExplorer`.

### Coaching forensics

`classify_move` keeps its default low-cost classification path, but coaching
clients can opt into structured evidence without adding another MCP tool:

```text
classify_move(..., detail="coach")
classify_move(..., detail="forensic", compare_moves=["g4", "gxh4"])
```

`detail="coach"` attaches deterministic position fingerprints, CCT-style
forcing-move snapshots, strongest-reply metadata, position deltas, mechanism
evidence and the principal continuation. `detail="forensic"` additionally
verifies the strongest reply one step deeper and evaluates the resulting
positions after the played move, engine-best move and explicitly requested
comparison moves.

`top_moves` exposes the same evidence-first approach for puzzle and candidate
questions:

```text
top_moves(..., detail="coach")
top_moves(..., detail="forensic", include_moves=["g4", "gxh4"])
top_moves(..., proof_mode="tactical", proof_defenses=5)
```

`include_moves` forces SAN/UCI alternatives into resulting-position analysis
even when they are outside the returned MultiPV top-N. Tactical proof mode
checks every opponent reply when there are at most eight legal defenses;
otherwise it samples the requested number of engine-ranked defenses. The
response carries `proof_status=exhaustive`, `sampled_top_defenses`, or
`terminal_after_root`, so a single PV is never mislabeled as a complete proof.
Each defense includes its resulting FEN, evaluation and principal continuation.

`analyze_game` can now produce a coaching post-mortem instead of only an engine
report:

```text
analyze_game(..., depth=18, detail="coach", perspective="black")
analyze_game(..., depth=18, detail="forensic", perspective="black", max_critical_moments=6)
```

Coach mode reuses the full-game eval scan and structures it into perspective-
relative `game_segments`, `advantage_events`, 1-7 `critical_moments`,
`positive_moments`, evidence-bounded `root_cause_links`, PGN mainline comments
and a final-position defensibility snapshot. It deliberately does not promote
every small engine delta into a lesson.

Forensic mode keeps the cheap whole-game scan, then automatically re-searches
only the selected critical positions. The default d18 scan is verified around
d22, d20 around d24, and an unstable classification can escalate only that
position to at most d26. Each verified moment can include classification
stability, a top-2 candidate gap, resource uniqueness and the strongest forcing
reply. This implements the intended `scan -> select -> verify` workflow without
paying d24+ for every ply in a long game.

Annotated mainline comments are preserved as `user_comment_raw`. That lets the
coaching layer combine engine evidence such as `FORCING_CAPTURE_REPLY` with a
player statement such as "I did not see Qxc5+" before assigning a process label.

The evidence layer intentionally does **not** claim what the player thought.
It emits machine-readable signatures such as `FORCING_CAPTURE_REPLY` and
`MISSED_FORCING_REPLY_CANDIDATE`; the coach/LLM combines those board facts
with the player's self-report before assigning process labels such as
"incomplete CCT" or "calculation stopped too early".

The response also echoes a canonical piece map, material, side to move,
castling/en-passant state and two deterministic position hashes. The legacy
single `position_hash` field is deprecated in favor of two explicitly-named
identities so downstream consumers can pick the right semantics:

- `fen_hash` — SHA-256 of the full 6-field canonical FEN (includes the
  halfmove + fullmove bookkeeping counters). Changes whenever any FEN field
  changes.
- `repetition_key` — SHA-256 of the 4-field FIDE repetition identity
  (piece placement + side to move + castling rights + en-passant square).
  Invariant to clock changes; matches FIDE's "same position" rule for
  threefold and fivefold repetition tracking.

The `position_hash` field is kept as a deprecated alias of `fen_hash` for one
release. Reading it emits a single `DeprecationWarning` per process.

These hashes give screen-based clients a position-verification handshake
before they explain a puzzle or game position.

## Performance

- **Stockfish pool size** defaults to `min(cpu_count, 8)` - sized to the
  OVH production box (8 cores Xeon E5-1620 v2).
- **Lifespan-managed pool**: the FastMCP `lifespan` context initializes
  the pool at startup so the first request doesn't pay a 100ms+
  cold-pool penalty.
- **GZip compression**: Starlette's `GZipMiddleware` cuts analyze_game
  payload sizes ~6× on the wire (`minimum_size=1024`).
- **uvloop + httptools**: replaces the default asyncio loop and
  h11 parser - measured ~30% throughput improvement on JSON-RPC workloads.
- **Single-flight cache coalescing**: concurrent identical evaluation
  requests share one Stockfish call via the `SingleFlight` helper.
- **Multi-tier cache (L1 memory LRU + L2 SQLite WAL)** absorbs repeat
  positions / top-moves queries.
