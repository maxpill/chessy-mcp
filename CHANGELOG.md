# Changelog

## 2026-09-21 — `ocr_to_pgn` v2

### Breaking

- `ocr_to_pgn` no longer accepts the `image_b64: str` parameter. Callers
  MUST pass an `image` discriminated-union object: `{"kind":"base64",...}`,
  `{"kind":"url",...}`, or `{"kind":"file_uri",...}`. Exactly one form.
- Default response `verbosity` is now `"minimal"`. The full forensic
  payload (`candidates`, `moves`, `raw_ocr_text`, etc.) is no longer
  sent unless the caller opts into `"compact"` or `"full"`.
- The `verify_with_stockfish` flag is removed. Use
  `engine_plausibility="off"` (engine never invoked) or the default
  `engine_plausibility="tiebreak_only"` (Stockfish is consulted only as
  a weak tiebreaker between visually similar candidates; never to
  invalidate a move solely on the basis of a bad eval).

### Additions

- **Image source union.** `httpx` server-side fetch for `http(s)` URLs
  with allowlist (`CHESSY_MCP_URL_ALLOWLIST`) and `file://`-style
  absolute path reads sandboxed under `CHESSY_MCP_FILE_ROOT`.
- **Adaptive preprocessing.** `mode="auto"` (default) applies CLAHE +
  auto-rotation always and the bilateral denoise only when image SNR
  (grayscale stdev) is below 18.0. `mode="explicit"` restores the
  caller-controlled flag semantics from v1.
- **Structured header extraction.** The sidecar now runs a deterministic
  M3 call returning the seven PGN header fields (White / Black / Round
  / Date / Event / Site / Result) with per-field confidence. The MCP
  tool surfaces them as `metadata: {white: {value, confidence}, ...}`
  and lets callers override detected values via the new `metadata`
  input parameter.
- **Legal-sequence beam search.** The sidecar aggregates per-cell SAN
  candidates across all OCR passes; the MCP tool reranks them via a
  deterministic beam search (`beam_width=5`) with a downstream-
  continuity bonus and an opt-in Stockfish tiebreak. The output is a
  single, validated `canonical_pgn`; ambiguous plies are surfaced in
  `uncertainties: [...]` rather than failing the call.
- **Resolve modes.** `resolve_ambiguities="auto" | "strict" |
"best_effort"`. `"auto"` (default) returns `status="needs_review"`
  with populated `uncertainties`; `"strict"` raises on any ambiguity;
  `"best_effort"` always returns the best-scoring legal sequence.

### Internal

- New modules: `mcp_server/ocr/image_source.py`,
  `mcp_server/ocr/beam_search.py`, `core/chess_ocr/header_extractor.py`.
- Sidecar schema additions in `core/chess_ocr/app.py` and
  `core/chess_ocr/consensus.py`: `extract_headers`, `beam_per_cell`,
  `metadata_hints` request fields; `cell_candidates`, `headers`
  response fields.
- `MinimaxEngine.extract_headers()` method (deterministic JSON-mode M3
  call).
- `OcrPgnResult` is now a verbosity-gated Pydantic model; v1 callers
  that read `candidates`, `moves`, `raw_ocr_text` should request
  `verbosity="full"`.
