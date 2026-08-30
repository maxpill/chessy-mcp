from pathlib import Path

cache_path = Path("mcp_server/cache.py")
text = cache_path.read_text()
start = text.index("_HISTORY_FINGERPRINT_CACHE:")
end = text.index("\ndef canonical_fen", start)
replacement = '''def history_fingerprint(board: chess.Board) -> str:
    """Fingerprint the reversible history that can affect repetition rights.

    Correctness is more important than memoizing by object identity. An earlier
    implementation cached by ``(id(board), len(move_stack))``; Python can reuse
    object ids after a board is freed, and a board can also be rewound and given
    a different history at the same stack length. Either case can make two
    distinct repetition histories share a cache key.

    Work on a stack-preserving copy so the caller's board is never mutated.
    Only positions since the most recent irreversible move can contribute to a
    future repetition claim, so the walk stops there.
    """
    if not board.move_stack:
        return ""

    work = board.copy(stack=True)
    keys: list[str] = [str(_board_transposition_key(work))]
    while work.move_stack:
        move = work.pop()
        if work.is_irreversible(move):
            break
        keys.append(str(_board_transposition_key(work)))

    digest = hashlib.sha256(";".join(keys).encode("utf-8")).hexdigest()[:12]
    return f":h={digest}"

'''
text = text[:start] + replacement + text[end:]
cache_path.write_text(text)

test_path = Path("tests/test_mcp_server.py")
test_text = test_path.read_text()
old = '''    # Now evaluate start position directly - should hit L1 cache!\n    start_eval = await server_module.evaluate_position(\n        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", depth=14\n    )\n    assert start_eval.cp == 15\n    # eval_calls should NOT have increased!\n    assert pool.eval_calls == 3\n'''
new = '''    # Evaluate the same *known-root* start position directly. ``startpos`` has\n    # complete history, matching the complete PGN root cached by analyze_game.\n    # A naked equivalent FEN is intentionally a different semantic cache key\n    # because pre-FEN repetition history is unknowable.\n    start_eval = await server_module.evaluate_position("startpos", depth=14)\n    assert start_eval.cp == 15\n    assert pool.eval_calls == 3\n'''
if old not in test_text:
    raise SystemExit("cache sharing test block not found")
test_path.write_text(test_text.replace(old, new, 1))
