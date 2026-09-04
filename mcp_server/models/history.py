"""``HistoryBlock`` aggregate — position provenance of :class:`MCPEval`.

Groups the FEN provenance fields (input → canonical), history-dependent
status flags, repetition status, and the post-position FEN surface.
"""

from __future__ import annotations

from pydantic import BaseModel


class HistoryBlock(BaseModel):
    """Position-history fields of a single position evaluation."""

    input_fen: str | None = None
    canonical_fen: str | None = None
    fen_was_canonicalized: bool = False
    post_fen: str | None = None
    history_dependent_status: bool = False
    lichess_url_reproduces_history: bool = True
    requires_move_stack: bool = False
    fen_sufficient_for_status: bool = True
    history_completeness: str = "incomplete"  # complete | partial | incomplete | not_required
    repetition_status: str = "none"  # "unknown" | "none" | "threefold_claimable" | "fivefold"
