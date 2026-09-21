"""Validate every PGN in ``~/Desktop/pgny_pgns/`` with a sequential Stockfish pass.

For each game, every legal move is replayed; after each ply we ask
Stockfish (depth 8, 200 ms budget) for the position evaluation and
compare it to the eval *before* the move. A swing of >=300 cp against
the side that just moved is treated as a probable OCR transcription
error and flagged.

Output:
    /tmp/chessy_validation/validation.json
    /tmp/chessy_validation/VALIDATION.md
"""

from __future__ import annotations

import io
import json
import pathlib
import time
from dataclasses import asdict, dataclass, field

import chess
import chess.engine
import chess.pgn


PGN_DIR = pathlib.Path.home() / "Desktop" / "pgny_pgns"
OUTPUT_DIR = pathlib.Path("/tmp/chessy_validation")
STOCKFISH_BIN = "/Users/max/.local/bin/stockfish"
DEPTH = 8
BUDGET_S = 0.2
BLUNDER_THRESHOLD_CP = 300


@dataclass
class GameReport:
    file: str
    moves_played: int = 0
    moves_legal: int = 0
    flagged_blunders: list[dict] = field(default_factory=list)
    final_fen: str = ""
    result_token: str = "*"
    parse_error: str = ""
    engine_errors: int = 0
    total_eval_ms: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.parse_error
            and self.moves_illegal == 0
            and self.moves_played > 0
            and len(self.flagged_blunders) <= 2
        )

    @property
    def moves_illegal(self) -> int:
        return self.moves_played - self.moves_legal


def _info_to_cp(info, stm: chess.Color) -> int:
    """Convert engine Info (dict-like) to a centipawn score from the
    side-to-move perspective at the eval position.
    """
    # `engine.analyse` returns a dict-like (chess.engine.Info).
    if isinstance(info, dict):
        score = info.get("score")
    else:
        score = getattr(info, "score", None)
    if score is None:
        return 0
    # PovScore.white() returns the Score from White's POV.
    s = score.white()
    if s.is_mate():
        n = s.mate() or 0
        return (100000 - abs(n)) * (1 if n > 0 else -1)
    cp = s.score(mate_score=100000)
    # Now flip sign if stm is BLACK (Stockfish's score is always White's POV).
    if stm == chess.BLACK:
        cp = -cp if cp is not None else None
    return int(cp) if cp is not None else 0


def validate_one(pgn_path: pathlib.Path, engine: chess.engine.SimpleEngine) -> GameReport:
    """Replay one PGN under Stockfish and flag suspicious moves."""
    report = GameReport(file=pgn_path.name)
    text = pgn_path.read_text()
    try:
        game = chess.pgn.read_game(io.StringIO(text))
    except Exception as exc:
        report.parse_error = f"{type(exc).__name__}: {exc}"
        return report

    if game is None:
        report.parse_error = "python-chess returned None"
        return report

    report.result_token = game.headers.get("Result", "*")
    board = game.board()

    for ply, move in enumerate(game.mainline_moves(), start=1):
        report.moves_played = ply
        san = board.san(move)
        if san in {"?", "??"}:
            report.notes.append(f"ply {ply}: SAN placeholder '{san}' skipped")
            continue

        # Eval BEFORE the move, from the perspective of the side-to-move.
        t0 = time.time()
        try:
            info_before = engine.analyse(
                board,
                limit=chess.engine.Limit(depth=DEPTH, time=BUDGET_S),
            )
        except chess.engine.EngineError as exc:
            report.engine_errors += 1
            report.notes.append(f"ply {ply}: engine error before: {exc}")
            board.push(move)
            continue
        cmp_time = time.time() - t0

        cp_before = _info_to_cp(info_before, board.turn)
        board.push(move)
        report.moves_legal += 1

        # Eval AFTER the move (still from the perspective of the side
        # that just moved, so flip when it's now the opponent's turn).
        t0 = time.time()
        try:
            info_after = engine.analyse(
                board,
                limit=chess.engine.Limit(depth=DEPTH, time=BUDGET_S),
            )
        except chess.engine.EngineError as exc:
            report.engine_errors += 1
            report.notes.append(f"ply {ply}: engine error after: {exc}")
            continue
        cmp_time += time.time() - t0
        report.total_eval_ms += cmp_time * 1000.0

        cp_after = _info_to_cp(info_after, board.turn)
        swing = cp_before - cp_after
        if swing >= BLUNDER_THRESHOLD_CP:
            report.flagged_blunders.append(
                {
                    "ply": ply,
                    "san": san,
                    "cp_before": cp_before,
                    "cp_after": cp_after,
                    "swing_cp": swing,
                }
            )

    report.final_fen = board.fen()
    return report


def build_markdown(reports: list[GameReport], elapsed: float) -> str:
    total = len(reports)
    parsed_ok = [r for r in reports if not r.parse_error]
    clean = [r for r in reports if r.ok]
    blunders = [r for r in reports if r.flagged_blunders]
    failed = [r for r in reports if r.parse_error]

    def pct(n: int) -> str:
        return f"{n / total * 100:.1f}%"

    lines = [
        "# Stockfish validation — 40 OCR-transcribed PGNs",
        "",
        f"**Engine**: `{STOCKFISH_BIN}` (depth {DEPTH}, {BUDGET_S * 1000:.0f} ms/eval)",
        f"**Blunder threshold**: >= {BLUNDER_THRESHOLD_CP} cp swing against mover",
        f"**Total wall-clock**: {elapsed:.0f}s",
        "",
        "## Headline",
        "",
        f"- **{len(clean)}/{total} ({pct(len(clean))})** clean (no flagged blunders, no parse errors)",
        f"- **{len(blunders)}** with at least one flagged blunder (probable OCR error)",
        f"- **{len(failed)}** failed to parse through python-chess",
        f"- **{sum(len(r.flagged_blunders) for r in reports)}** total flagged plies across all games",
        "",
        "## Per-game verdict",
        "",
        "| Game | Status | Legal/Total | Blunders | Total eval time |",
        "|------|--------|-------------|----------|------------------|",
    ]
    for r in sorted(reports, key=lambda x: x.file):
        if r.parse_error:
            status = "PARSE-FAIL"
        elif r.flagged_blunders:
            status = "BLUNDER"
        elif r.ok:
            status = "OK"
        else:
            status = "?"
        lines.append(
            f"| `{r.file}` | {status} | {r.moves_legal}/{r.moves_played} | "
            f"{len(r.flagged_blunders)} | {r.total_eval_ms:.0f}ms |"
        )

    if blunders:
        lines += ["", "## Flagged blunders (per move, >= 300 cp swing)", ""]
        for r in blunders:
            lines.append(f"### `{r.file}`")
            for b in r.flagged_blunders[:5]:
                lines.append(
                    f"  - ply {b['ply']:>3}: **{b['san']}**  "
                    f"(before {b['cp_before']:+d}cp -> after {b['cp_after']:+d}cp, "
                    f"swing {b['swing_cp']:+d}cp)"
                )
            if len(r.flagged_blunders) > 5:
                lines.append(f"  - ... and {len(r.flagged_blunders) - 5} more")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pgns = sorted(PGN_DIR.rglob("*.pgn"))
    print(f"Validating {len(pgns)} PGNs from {PGN_DIR}")
    print(f"Engine: {STOCKFISH_BIN} (depth {DEPTH}, {BUDGET_S * 1000:.0f}ms/eval)")

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_BIN)
    t0 = time.time()
    reports: list[GameReport] = []
    try:
        for i, p in enumerate(pgns, start=1):
            print(f"[{i:>2}/{len(pgns)}] {p.name} ...", end=" ", flush=True)
            rep = validate_one(p, engine)
            reports.append(rep)
            status = "OK" if rep.ok else ("BLUNDER" if rep.flagged_blunders else "PARSE-FAIL")
            print(
                f"{status:11} moves={rep.moves_legal}/{rep.moves_played}  "
                f"blunders={len(rep.flagged_blunders)}  eval={rep.total_eval_ms:.0f}ms"
            )
    finally:
        engine.quit()

    elapsed = time.time() - t0

    json_path = OUTPUT_DIR / "validation.json"
    json_path.write_text(
        json.dumps(
            [{"file": r.file, **asdict(r)} for r in reports],
            indent=2,
            default=str,
        )
    )
    md_path = OUTPUT_DIR / "VALIDATION.md"
    md_path.write_text(build_markdown(reports, elapsed))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Total wall-clock: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
