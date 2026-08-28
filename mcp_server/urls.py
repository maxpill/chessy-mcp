from __future__ import annotations

from urllib.parse import quote


def lichess_urls(fen: str) -> tuple[str, str]:
    """Return (analysis_url, image_url) for a FEN string."""
    fen_lichess = fen.replace(" ", "_")
    fen_encoded = quote(fen, safe="")
    return (
        f"https://lichess.org/analysis/standard/{fen_lichess}",
        f"https://lichess1.org/export/fen.gif?fen={fen_encoded}",
    )
