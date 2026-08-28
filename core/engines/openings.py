from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chess

_DATA = Path(__file__).resolve().parent.parent / "data" / "openings.json"


@lru_cache(maxsize=1)
def _openings() -> list[tuple[str, str, str | None]]:
    rows = json.loads(_DATA.read_text(encoding="utf-8"))
    items = [(r["uci"], r["name"], r.get("eco")) for r in rows]
    items.sort(key=lambda x: len(x[0]), reverse=True)  # longest prefix wins
    return items


@lru_cache(maxsize=1)
def _openings_by_epd() -> dict[str, tuple[str, str | None, list[str], int]]:
    """Index openings by final board EPD for transposition-aware detection."""
    epd_map: dict[str, tuple[str, str | None, list[str], int]] = {}
    for uci, name, eco in _openings():
        b = chess.Board()
        valid = True
        move_list = uci.split()
        for m_str in move_list:
            try:
                m = chess.Move.from_uci(m_str)
                if m in b.legal_moves:
                    b.push(m)
                else:
                    valid = False
                    break
            except Exception:
                valid = False
                break
        if valid:
            epd = b.epd()
            # If not yet seen or this entry has deeper move length, store it
            if epd not in epd_map or len(move_list) > epd_map[epd][3]:
                epd_map[epd] = (name, eco, move_list, len(move_list))
    return epd_map


def lookup_opening(moves: list[str]) -> tuple[str | None, str | None, list[str]]:
    played = " ".join(moves)
    if not played:
        return None, None, []

    # 1. Direct prefix match
    prefix_match: tuple[str | None, str | None, list[str]] = (None, None, [])
    prefix_len = 0
    for uci, name, eco in _openings():
        if played == uci or played.startswith(uci + " "):
            prefix_match = (name, eco, uci.split())
            prefix_len = len(uci.split())
            break

    # 2. Transposition-aware EPD match by replaying moves
    epd_map = _openings_by_epd()
    b = chess.Board()
    epds_visited: list[tuple[str, int]] = []
    for i, m_str in enumerate(moves, start=1):
        try:
            m = chess.Move.from_uci(m_str)
            if m in b.legal_moves:
                b.push(m)
                epds_visited.append((b.epd(), i))
            else:
                break
        except Exception:
            break

    # Check from deepest position backwards
    for epd, _depth in reversed(epds_visited):
        if epd in epd_map:
            t_name, t_eco, t_moves, t_len = epd_map[epd]
            if t_len >= prefix_len:
                return t_name, t_eco, t_moves

    return prefix_match


def name_opening(board: chess.Board) -> str | None:
    moves = [m.uci() for m in board.move_stack]
    name, _, _ = lookup_opening(moves)
    return name
