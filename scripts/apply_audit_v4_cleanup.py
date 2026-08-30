from __future__ import annotations

from pathlib import Path


def dedupe_rule_policy() -> None:
    path = Path("mcp_server/rules.py")
    text = path.read_text(encoding="utf-8")
    marker = "def choose_recommended_action("
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("choose_recommended_action not found")
    end = text.find("def evaluate_rule_status(", start)
    if end < 0:
        raise RuntimeError("evaluate_rule_status not found")
    region = text[start:end]
    duplicate = region.find(marker, len(marker))
    if duplicate >= 0:
        region = region[:duplicate].rstrip() + "\n\n"
        text = text[:start] + region + text[end:]
    path.write_text(text, encoding="utf-8")


def clean_server() -> None:
    path = Path("mcp_server/server.py")
    text = path.read_text(encoding="utf-8")

    # Module imports must precede executable statements so strict linting does
    # not report every project import as E402.
    log_stmt = 'log = logging.getLogger("chessy_mcp.server")\n\n'
    if log_stmt in text:
        text = text.replace(log_stmt, "", 1)
        anchor = "from mcp_server.urls import lichess_urls\n"
        if anchor not in text:
            raise RuntimeError("server import anchor not found")
        text = text.replace(anchor, anchor + "\n\n" + log_stmt.rstrip() + "\n", 1)

    text = text.replace(
        "except Exception as exc:  # noqa: BLE001 — log and keep looping",
        "except Exception as exc:",
    )

    # asyncio keeps only weak references to tasks. Retain ponder tasks until
    # completion to make the background warmer deterministic and silence RUF006.
    marker = "def _maybe_ponder_warm("
    if marker in text and "_background_tasks: set[asyncio.Task[Any]]" not in text:
        text = text.replace(
            marker,
            "_background_tasks: set[asyncio.Task[Any]] = set()\n\n\n" + marker,
            1,
        )
    old = '''        asyncio.create_task(\n            _ponder_warm_cache(pool, next_board, depth, history_complete),\n            name="ponder-warm",\n        )\n'''
    new = '''        task = asyncio.create_task(\n            _ponder_warm_cache(pool, next_board, depth, history_complete),\n            name="ponder-warm",\n        )\n        _background_tasks.add(task)\n        task.add_done_callback(_background_tasks.discard)\n'''
    if old in text:
        text = text.replace(old, new, 1)

    path.write_text(text, encoding="utf-8")


dedupe_rule_policy()
clean_server()
print("audit v4 cleanup applied")
