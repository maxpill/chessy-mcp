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

```
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

Four Streamable-HTTP MCP tools at `/mcp`:

| Tool                | What it does                                              |
| ------------------- | --------------------------------------------------------- |
| `evaluate_position` | Single-position Stockfish eval at a given depth           |
| `top_moves`         | Top-N candidate moves ranked by Stockfish                 |
| `classify_move`     | Grade a played move against the engine's best alternative |
| `analyze_game`      | Full PGN analysis with accuracy, mistakes, turning points |

The 4-tool surface is intentional — the chessy app's coach runtime is the
primary consumer, and this server is sized for that workload.

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

The evidence layer intentionally does **not** claim what the player thought.
It emits machine-readable signatures such as `FORCING_CAPTURE_REPLY` and
`MISSED_FORCING_REPLY_CANDIDATE`; the coach/LLM combines those board facts
with the player's self-report before assigning process labels such as
"incomplete CCT" or "calculation stopped too early".

The response also echoes a canonical piece map, material, side to move,
castling/en-passant state and a deterministic position hash. That gives
screen-based clients a position-verification handshake before they explain a
puzzle or game position.

## Performance

- **Stockfish pool size** defaults to `min(cpu_count, 8)` — sized to the
  OVH production box (8 cores Xeon E5-1620 v2).
- **Lifespan-managed pool**: the FastMCP `lifespan` context initializes
  the pool at startup so the first request doesn't pay a 100ms+
  cold-pool penalty.
- **GZip compression**: Starlette's `GZipMiddleware` cuts analyze_game
  payload sizes ~6× on the wire (`minimum_size=1024`).
- **uvloop + httptools**: replaces the default asyncio loop and
  h11 parser — measured ~30% throughput improvement on JSON-RPC workloads.
- **Single-flight cache coalescing**: concurrent identical evaluation
  requests share one Stockfish call via the `SingleFlight` helper.
- **Multi-tier cache (L1 memory LRU + L2 SQLite WAL)** absorbs repeat
  positions / top-moves queries.
