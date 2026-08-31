from pathlib import Path

p = Path("mcp_server/server.py")
s = p.read_text(encoding="utf-8")
old = '''            if invalid_tokens:
                raise ValueError(
                    f"INVALID_PGN: Invalid PGN syntax or unrecognized token in movetext: {invalid_tokens[0]!r}"
                )
'''
new = '''            if invalid_tokens:
                error_prefix = "STRICT_PGN_ERROR" if strict else "INVALID_PGN"
                raise ValueError(
                    f"{error_prefix}: Invalid PGN syntax or unrecognized token in movetext: {invalid_tokens[0]!r}"
                )
'''
if old in s:
    s = s.replace(old, new, 1)
elif new not in s:
    raise RuntimeError("strict PGN token error patch target not found")
p.write_text(s, encoding="utf-8")
print("strict PGN token error patch applied")
