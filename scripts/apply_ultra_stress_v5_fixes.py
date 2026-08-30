from pathlib import Path

server_path = Path("mcp_server/server.py")
text = server_path.read_text(encoding="utf-8")

# Public tool validation must run inside the error-normalization boundary.
needle = "    verbosity_mode = _resolve_verbosity(verbosity)\n    try:\n        board = _build_board(fen, moves or [], strict=strict)"
replacement = "    try:\n        verbosity_mode = _resolve_verbosity(verbosity)\n        board = _build_board(fen, moves or [], strict=strict)"
count = text.count(needle)
if count != 2:
    raise SystemExit(f"expected two verbosity-before-try sites, got {count}")
text = text.replace(needle, replacement)

# Give invalid verbosity its own public error code in every ValueError mapper.
needle = '        code = "invalid_input"\n        if "STRICT" in msg:\n            code = "strict_validation_error"'
replacement = '        code = "invalid_input"\n        if "INVALID_VERBOSITY" in msg:\n            code = "invalid_verbosity"\n        elif "STRICT" in msg:\n            code = "strict_validation_error"'
count = text.count(needle)
if count < 2:
    raise SystemExit(f"expected at least two ValueError mappers, got {count}")
text = text.replace(needle, replacement)

# python-chess intentionally sanitizes impossible castling rights. For an API
# that claims to validate FEN, silently dropping an impossible right is unsafe:
# it changes the supplied position. Validate the raw rights against pieces.
needle = "            try:\n                b = chess.Board(cleaned)\n                if b.is_valid() or b.status() == chess.STATUS_VALID:\n                    board = b"
replacement = """            try:
                b = chess.Board(cleaned)
                if len(tokens) >= 3:
                    raw_castling = tokens[2]
                    if raw_castling != \"-\":
                        if not re.fullmatch(r\"[KQkq]+\", raw_castling) or len(set(raw_castling)) != len(raw_castling):
                            raise ValueError(
                                f\"INVALID_FEN: Castling rights in FEN '{cleaned}' are malformed.\"
                            )
                        actual_rights: set[str] = set()
                        if b.has_kingside_castling_rights(chess.WHITE):
                            actual_rights.add(\"K\")
                        if b.has_queenside_castling_rights(chess.WHITE):
                            actual_rights.add(\"Q\")
                        if b.has_kingside_castling_rights(chess.BLACK):
                            actual_rights.add(\"k\")
                        if b.has_queenside_castling_rights(chess.BLACK):
                            actual_rights.add(\"q\")
                        if set(raw_castling) != actual_rights:
                            raise ValueError(
                                f\"INVALID_FEN: Castling rights in FEN '{cleaned}' are inconsistent with king/rook placement.\"
                            )
                if b.is_valid() or b.status() == chess.STATUS_VALID:
                    board = b"""
if needle not in text:
    raise SystemExit("FEN parse insertion point not found")
text = text.replace(needle, replacement, 1)
server_path.write_text(text, encoding="utf-8")

# Correct two deliberately over-aggressive stress expectations: a black pawn on
# a7 is legal, and this parser intentionally tolerates a 5-field EPD-like FEN.
test_path = Path("tests/test_ultra_stress_v5.py")
tests = test_path.read_text(encoding="utf-8")
tests = tests.replace('    "4k3/p7/8/8/8/8/8/4K3 w - - 0 1",\n', '')
tests = tests.replace('    "4k3/8/8/8/8/8/8/4K3 w - - 0",\n', '')
insert_after = "def test_invalid_fen_matrix_is_rejected(fen: str):\n    with pytest.raises(ValueError):\n        server_module._build_board(fen, [])\n\n\n"
addition = """def test_five_field_epd_like_position_is_tolerated_and_canonicalized():
    board = server_module._build_board(\"4k3/8/8/8/8/8/8/4K3 w - - 0\", [])
    assert board.fen() == \"4k3/8/8/8/8/8/8/4K3 w - - 0 1\"


"""
if insert_after not in tests:
    raise SystemExit("stress test insertion point not found")
tests = tests.replace(insert_after, insert_after + addition, 1)
test_path.write_text(tests, encoding="utf-8")
