from pathlib import Path

path = Path("mcp_server/server.py")
text = path.read_text()
old = "        rule_final = evaluate_rule_status(final_board)\n"
new = "        # positions[] was reconstructed from the complete PGN mainline, so repetition\n        # history is authoritative here. Do not downgrade a previously detected fivefold\n        # repetition to generic game_over during final result reconciliation.\n        rule_final = evaluate_rule_status(final_board, history_complete=\"complete\")\n"
if old not in text:
    raise SystemExit("target final rule-status call not found")
text = text.replace(old, new, 1)
path.write_text(text)
