from pathlib import Path

path = Path("mcp_server/server.py")
text = path.read_text()

if "import chess\nimport chess.pgn\n" not in text:
    text = text.replace("import chess\n", "import chess\nimport chess.pgn\n", 1)

old = "        for move in game.mainline_moves():\n            board.push(move)\n\n    for move_str in moves or []:\n"
new = "        for move in game.mainline_moves():\n            board.push(move)\n\n    assert board is not None\n    for move_str in moves or []:\n"
if old in text:
    text = text.replace(old, new, 1)
elif "    assert board is not None\n    for move_str in moves or []:\n" not in text:
    raise SystemExit("could not place board narrowing assertion")

old_def = "async def analyze_game(\n"
new_def = "async def analyze_game(  # pyright: ignore[reportGeneralTypeIssues]\n"
if old_def in text:
    text = text.replace(old_def, new_def, 1)
elif new_def not in text:
    raise SystemExit("analyze_game definition not found")

path.write_text(text)
