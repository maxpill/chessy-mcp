from pathlib import Path

path = Path("mcp_server/server.py")
text = path.read_text()

old_sig = '''async def _gather_evaluate_positions_bounded(
    positions: list[chess.Board],
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    *,
    requested_depth: int,
) -> list[tuple[MCPEval, bool]]:
'''
new_sig = '''async def _gather_evaluate_positions_bounded(
    positions: list[chess.Board],
    depth: int,
    pool: AnalyzerPool | TCPAnalyzerPool,
    *,
    requested_depth: int,
    history_complete: str = "complete",
) -> list[tuple[MCPEval, bool]]:
'''
if old_sig not in text:
    raise SystemExit("gather signature not found")
text = text.replace(old_sig, new_sig, 1)

old_worker = '''                            reuse_tt=(j > 0),
                            analyzer=analyzer,
                        )
'''
new_worker = '''                            reuse_tt=(j > 0),
                            analyzer=analyzer,
                            history_complete=history_complete,
                        )
'''
if old_worker not in text:
    raise SystemExit("worker eval call not found")
text = text.replace(old_worker, new_worker, 1)

old_fallback = '''                    requested_depth=requested_depth,
                    reuse_tt=False,
                )
'''
new_fallback = '''                    requested_depth=requested_depth,
                    reuse_tt=False,
                    history_complete=history_complete,
                )
'''
if old_fallback not in text:
    raise SystemExit("fallback eval call not found")
text = text.replace(old_fallback, new_fallback, 1)

old_call = '''        eval_pairs = await _gather_evaluate_positions_bounded(
            positions, depth, pool, requested_depth=raw_requested_depth
        )
'''
new_call = '''        eval_pairs = await _gather_evaluate_positions_bounded(
            positions,
            depth,
            pool,
            requested_depth=raw_requested_depth,
            history_complete="complete",
        )
'''
if old_call not in text:
    raise SystemExit("analyze gather call not found")
text = text.replace(old_call, new_call, 1)

old_metric_rule = '''                rule_before = evaluate_rule_status(board_before)
'''
new_metric_rule = '''                rule_before = evaluate_rule_status(
                    board_before, history_complete="complete"
                )
'''
if old_metric_rule not in text:
    raise SystemExit("metrics rule call not found")
text = text.replace(old_metric_rule, new_metric_rule, 1)

path.write_text(text)
