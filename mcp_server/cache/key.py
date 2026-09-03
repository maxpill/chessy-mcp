"""Cache key derivation.

All cache keys are constructed here so the prefix structure can be inspected
in one place. The key format is:

    ``mcp:{CACHE_VERSION}:eng={engine}:{tool_kind}:hist={history}:{fen}{fp}:{args}``

Where:
    - ``CACHE_VERSION`` comes from ``mcp_server.cache.version`` and bundles
      build SHA + package version + logic-hash.
    - ``engine`` is the Stockfish fingerprint (overridable per call).
    - ``tool_kind`` is ``eval`` / ``top`` / ``classify``.
    - ``history`` is the explicit ``history_complete`` flag, which lets us
      promote an ``incomplete`` cached entry only when the new request also
      reports ``complete`` (the history fingerprint is then a superset).
    - ``fp`` is the ``history_fingerprint`` suffix that captures the
      reversible moves since the last irreversible.
"""

from __future__ import annotations

import hashlib
from typing import Any

import chess

from mcp_server.cache.version import CACHE_VERSION, _resolve_engine_version

__all__ = [
    "canonical_fen",
    "classify_cache_key",
    "eval_cache_key",
    "history_fingerprint",
    "top_moves_cache_key",
]


def _board_transposition_key(b: chess.Board) -> tuple[Any, ...]:
    return (
        b.pawns,
        b.knights,
        b.bishops,
        b.rooks,
        b.queens,
        b.kings,
        b.occupied_co[chess.WHITE],
        b.occupied_co[chess.BLACK],
        b.turn,
        b.clean_castling_rights(),
        b.ep_square if b.has_legal_en_passant() else None,
    )


def history_fingerprint(board: chess.Board) -> str:
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


def canonical_fen(board: chess.Board) -> str:
    """Return full 6-field FEN position key."""
    return board.fen()


def eval_cache_key(
    board: chess.Board,
    depth: int,
    engine_version: str | None = None,
    history_completeness: str = "incomplete",
) -> str:
    """Generate canonical cache key for position evaluation."""
    fp = history_fingerprint(board)
    ev = _resolve_engine_version(engine_version)
    return (
        f"mcp:{CACHE_VERSION}:eng={ev}:eval:hist={history_completeness}:{board.fen()}{fp}:{depth}"
    )


def top_moves_cache_key(
    board: chess.Board,
    depth: int,
    n: int = 1,
    engine_version: str | None = None,
    history_completeness: str = "incomplete",
) -> str:
    """Generate canonical cache key for MultiPV top moves."""
    fp = history_fingerprint(board)
    n_part = f":n={n}" if n is not None else ""
    ev = _resolve_engine_version(engine_version)
    return f"mcp:{CACHE_VERSION}:eng={ev}:top:hist={history_completeness}:{board.fen()}{fp}:{depth}{n_part}"


def classify_cache_key(
    board: chess.Board,
    move_uci: str,
    depth: int,
    action_type: str = "play_move",
    engine_version: str | None = None,
    history_completeness: str = "incomplete",
) -> str:
    """Generate canonical cache key for move classification."""
    fp = history_fingerprint(board)
    act_part = f":{action_type}" if action_type and action_type != "play_move" else ""
    ev = _resolve_engine_version(engine_version)
    return (
        f"mcp:{CACHE_VERSION}:eng={ev}:classify:hist={history_completeness}:"
        f"{board.fen()}{fp}:{move_uci}:{depth}{act_part}"
    )
