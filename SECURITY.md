# Security notes for chessy-mcp

## Secret rotation

### `n` (M3 multimodal API key)

The chess-ocr sidecar reads the M3 multimodal API key from the single-letter
environment variable `n`. This is the **only** API key the sidecar uses, and
it never reaches the MCP container or the public edge.

**Important**: treat `n` like any production API key:

1. **Never commit** the real value to git. `.env` is gitignored; the example
   in `.env.example` has an empty `n=`.
2. **Rotate immediately** if the key is ever exposed in chat, logs, screen
   share, screenshots, or any other unintended channel. Compromised keys
   should be revoked at the provider and a new one issued.
3. **Audit logs** before sharing anything that contains M3 responses — the
   sidecar never echoes the key back, but a misconfigured logger or a custom
   debug handler could.

### Operational checklist

- [ ] `n` is set in `.env` (not `.env.example`).
- [ ] `.env` is in `.gitignore` (already true in this repo).
- [ ] The chess-ocr sidecar refuses to boot when `n` is empty (HTTP 503
      `MISSING_API_KEY` on every OCR request).
- [ ] Rotation is logged in your team's secret-management tool.
- [ ] No logging library or middleware prints environment variables on
      startup.

## What `n` should NOT do

- It must **not** appear in any `git`-tracked file.
- It must **not** appear in MCP responses (the OCR tool returns only the
  extracted PGN text, never the key).
- It must **not** appear in tracebacks or exception messages — the sidecar
  references the env var name `n`, never the value.
- It must **not** appear in structlog output — the `OCRSettings.__repr__`
  redacts it explicitly.

## What we verify

The `tests/test_secret_redaction.py` test suite enforces:

- `OCRSettings` `__repr__` and `__str__` do not include the key.
- Logging a settings object does not include the key.
- Missing-key error messages reference `MISSING_API_KEY` but never the key.
- The system prompt sent to M3 contains no trace of the key.

## What we do not do (v1)

- We do not currently support key rotation without sidecar restart. A future
  v2 could subscribe to a secret manager and hot-reload `n`.
- We do not currently rate-limit per-key. Cloudflare's edge rate-limits the
  MCP container; the sidecar has no per-key quota.
