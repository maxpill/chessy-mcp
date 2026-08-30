from __future__ import annotations

import re
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, got {count}")
    return text.replace(old, new, 1)


def replace_block(text: str, start: str, end: str, replacement: str, label: str) -> str:
    i = text.find(start)
    if i < 0:
        raise RuntimeError(f"{label}: start marker not found: {start!r}")
    j = text.find(end, i)
    if j < 0:
        raise RuntimeError(f"{label}: end marker not found: {end!r}")
    return text[:i] + replacement.rstrip() + "\n\n" + text[j:]


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------
models_path = Path("mcp_server/models.py")
models = models_path.read_text(encoding="utf-8")
models = models.replace(
    "from pydantic import BaseModel, Field",
    "from pydantic import BaseModel, Field, model_validator",
    1,
)
models = models.replace(
    "from mcp_server.actions import (\n    build_best_action,\n    build_legal_actions,\n)",
    "from mcp_server.actions import (\n    build_best_action,\n    build_legal_actions,\n    build_played_action,\n)",
    1,
)
models = models.replace(
    'history_completeness: str = "complete"  # "complete" | "incomplete" | "not_required"',
    'history_completeness: str = "incomplete"  # complete | partial | incomplete | not_required',
    1,
)
models = models.replace(
    "wdl_pct: dict[str, int] | None = None",
    "wdl_pct: dict[str, float] | None = None",
    1,
)
models = models.replace(
    "history_complete: bool = True,",
    'history_complete: str | bool = "incomplete",',
    1,
)
models = models.replace(
    "wdl_pct_dict: dict[str, int] | None = (\n            {\"win\": wdl_tuple[0] // 10, \"draw\": wdl_tuple[1] // 10, \"loss\": wdl_tuple[2] // 10}\n            if wdl_tuple is not None\n            else None\n        )",
    "wdl_pct_dict: dict[str, float] | None = (\n            {\n                \"win\": wdl_tuple[0] / 10.0,\n                \"draw\": wdl_tuple[1] / 10.0,\n                \"loss\": wdl_tuple[2] / 10.0,\n            }\n            if wdl_tuple is not None\n            else None\n        )",
    1,
)
models = models.replace(
    '    action_type: str = "play_move",\n) -> PlayedMoveScore:',
    '    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",\n) -> PlayedMoveScore:',
    1,
)
models = models.replace(
    '    """Unified, rule-aware single source of truth for move grading and loss across all tools."""\n    if board_after is None:',
    '    """Unified, rule-aware single source of truth for move grading and loss across all tools."""\n    if action_type not in {"play_move", "claim_draw", "claim_draw_with_intended_move"}:\n        raise ValueError(f"INVALID_ACTION_TYPE: {action_type}")\n    if board_after is None:',
    1,
)
models = models.replace(
    "    rule_before = evaluate_rule_status(\n        board_before,\n        mover_score=before_mover_score,\n        mate_for_mover=mover_mate_before,\n    )\n    rule_after = evaluate_rule_status(board_after)",
    "    history_state = eval_before.history_completeness\n    rule_before = evaluate_rule_status(\n        board_before,\n        mover_score=before_mover_score,\n        mate_for_mover=mover_mate_before,\n        history_complete=history_state,\n    )\n    rule_after = evaluate_rule_status(\n        board_after, history_complete=history_state\n    )",
    1,
)

claim_old = '''        claim_legal = (is_claim_now_action and rule_before.can_claim_now) or (
            not is_claim_now_action
            and rule_before.can_claim_with_intended_move
            and played_uci in [u.lower() for u in rule_before.intended_claim_ucis]
        )
        # Forced-win-for-mover check: never accept a claim when the position
        # has a winning mate or large CP advantage in mover's favor.
        is_mover_forced_win = (mover_mate_before is not None and mover_mate_before > 0) or before_mover >= 200
        if not claim_legal or is_mover_forced_win:
            # Fall through to play_move scoring; this move is a real blunder.
            pass
        else:
            claim_r = rule_before.claim_reasons[0] if rule_before.claim_reasons else None
            return PlayedMoveScore(
                move_class=MoveClass.BEST,
                centipawn_loss=0,
                raw_centipawn_loss=0,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=0,
                effective_loss=0,
                loss_kind="none",
                is_best_engine_move=is_best_engine_move,
                win_loss=0.0,
                best_action=canonical_best_action,
                is_best_action=True,
                action_equivalent=False,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=claim_r,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )
'''
claim_new = '''        claim_legal = (is_claim_now_action and rule_before.can_claim_now) or (
            not is_claim_now_action
            and rule_before.can_claim_with_intended_move
            and played_uci in [u.lower() for u in rule_before.intended_claim_ucis]
        )
        if not claim_legal:
            raise ValueError("ILLEGAL_ACTION: requested draw claim is not legally available")

        if is_claim_now_action:
            reasons = rule_before.claim_reasons_now
        else:
            reason_map = rule_before.intended_claim_reasons_by_uci
            reasons = reason_map.get(move.uci(), [])
        claim_r = reasons[0] if reasons else None

        is_mover_forced_win = (
            (mover_mate_before is not None and mover_mate_before > 0)
            or before_mover >= 200
        )
        if is_mover_forced_win:
            return PlayedMoveScore(
                move_class=MoveClass.BLUNDER,
                centipawn_loss=None,
                raw_centipawn_loss=None,
                raw_centipawn_delta=raw_board_delta,
                mate_distance_loss=None,
                effective_loss=1000,
                loss_kind="outcome_penalty",
                outcome_penalty=1000,
                is_best_engine_move=is_best_engine_move,
                win_loss=50.0,
                best_action=canonical_best_action,
                is_best_action=False,
                action_equivalent=False,
                missed_draw_claim=False,
                conceded_draw_claim=False,
                claim_reason=claim_r,
                claim_move=rule_before.claim_move,
                can_claim_now=rule_before.can_claim_now,
                can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
                claim_moves=rule_before.claim_moves,
            )

        return PlayedMoveScore(
            move_class=MoveClass.BEST,
            centipawn_loss=0,
            raw_centipawn_loss=0,
            raw_centipawn_delta=raw_board_delta,
            mate_distance_loss=0,
            effective_loss=0,
            loss_kind="none",
            is_best_engine_move=is_best_engine_move,
            win_loss=0.0,
            best_action=canonical_best_action,
            is_best_action=canonical_best_action == action_type,
            action_equivalent=canonical_best_action != action_type,
            missed_draw_claim=False,
            conceded_draw_claim=False,
            claim_reason=claim_r,
            claim_move=rule_before.claim_move,
            can_claim_now=rule_before.can_claim_now,
            can_claim_with_intended_move=rule_before.can_claim_with_intended_move,
            claim_moves=rule_before.claim_moves,
        )
'''
models = replace_once(models, claim_old, claim_new, "strict procedural claim scoring")

models = models.replace(
    '    action_type: str = "play_move"\n    best_action: str = "play_move"',
    '    action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move"\n    best_action: str = "play_move"',
    1,
)
validator_anchor = "    classification_verified: bool = False\n\n    @classmethod\n"
validator = '''    classification_verified: bool = False

    @model_validator(mode="after")
    def _enforce_action_invariants(self) -> MCPMoveAnalysis:
        engine_best = bool(self.is_engine_best or self.is_best_engine_move)
        self.is_engine_best = engine_best
        self.is_best_engine_move = engine_best
        if self.played_action_obj is not None:
            obj_type = self.played_action_obj.get("type")
            if obj_type != self.action_type:
                raise ValueError(
                    f"played_action_obj.type={obj_type!r} does not match action_type={self.action_type!r}"
                )
        if self.is_best_action and self.action_type != self.best_action and not self.action_equivalent:
            self.is_best_action = False
        return self

    @classmethod
'''
models = replace_once(models, validator_anchor, validator, "move analysis invariants")
models = models.replace(
    '        action_type: str = "play_move",\n    ) -> MCPMoveAnalysis:',
    '        action_type: Literal["play_move", "claim_draw", "claim_draw_with_intended_move"] = "play_move",\n        history_complete: str | bool = "incomplete",\n    ) -> MCPMoveAnalysis:',
    1,
)
models = models.replace(
    "        eval_bef = MCPEval.from_eval(ma.eval_before, fen_before, board=board_before)\n        eval_aft = MCPEval.from_eval(ma.eval_after, fen_after, board=board_after)",
    "        eval_bef = MCPEval.from_eval(\n            ma.eval_before, fen_before, board=board_before, history_complete=history_complete\n        )\n        eval_aft = MCPEval.from_eval(\n            ma.eval_after, fen_after, board=board_after, history_complete=history_complete\n        )",
    1,
)
played_block_pattern = re.compile(
    r"        # Build typed action payloads \(audit 10\.2\)\n"
    r"        from mcp_server\.actions import \([\s\S]*?\n"
    r"        # Build best_action from eval_before's typed state\n"
    r"        best_action_payload = eval_bef\.best_action_obj\n"
)
played_replacement = '''        rule_before = evaluate_rule_status(
            b_bef, history_complete=history_complete
        )
        played_action_obj = build_played_action(
            action_type,
            move_uci=ma.played,
            move_san=played_san,
            rule_status=rule_before,
            cp=eval_aft.cp,
            mate=eval_aft.mate,
        )
        best_action_payload = eval_bef.best_action_obj
'''
models, n = played_block_pattern.subn(played_replacement, models, count=1)
if n != 1:
    raise RuntimeError(f"typed from_analysis action payload: expected 1 replacement, got {n}")
models_path.write_text(models, encoding="utf-8")


# ---------------------------------------------------------------------------
# cache.py
# ---------------------------------------------------------------------------
cache_path = Path("mcp_server/cache.py")
cache = cache_path.read_text(encoding="utf-8")
old_logic = '''_LOGIC_FILES = (
    "mcp_server/cache.py",
    "mcp_server/rules.py",
    "mcp_server/models.py",
    "mcp_server/server.py",
)'''
new_logic = '''_LOGIC_FILES = (
    "mcp_server/cache.py",
    "mcp_server/rules.py",
    "mcp_server/models.py",
    "mcp_server/actions.py",
    "mcp_server/server.py",
    "mcp_server/tcp_analyzer.py",
    "mcp_server/tcp_client.py",
    "core/engines/analyzer.py",
    "core/engines/analysis.py",
    "core/engines/grading.py",
    "core/winprob.py",
)'''
cache = replace_once(cache, old_logic, new_logic, "cache logic files")
cache = cache.replace('return f"v13+', 'return f"v14+', 1)

cache = cache.replace(
    "def eval_cache_key(board: chess.Board, depth: int, engine_version: str | None = None) -> str:",
    "def eval_cache_key(\n    board: chess.Board,\n    depth: int,\n    engine_version: str | None = None,\n    history_completeness: str = \"incomplete\",\n) -> str:",
    1,
)
cache = cache.replace(
    'return f"mcp:{CACHE_VERSION}:eng={ev}:eval:{board.fen()}{fp}:{depth}"',
    'return f"mcp:{CACHE_VERSION}:eng={ev}:eval:hist={history_completeness}:{board.fen()}{fp}:{depth}"',
    1,
)
cache = cache.replace(
    "def top_moves_cache_key(board: chess.Board, depth: int, n: int = 1, engine_version: str | None = None) -> str:",
    "def top_moves_cache_key(\n    board: chess.Board,\n    depth: int,\n    n: int = 1,\n    engine_version: str | None = None,\n    history_completeness: str = \"incomplete\",\n) -> str:",
    1,
)
cache = cache.replace(
    'return f"mcp:{CACHE_VERSION}:eng={ev}:top:{board.fen()}{fp}:{depth}{n_part}"',
    'return f"mcp:{CACHE_VERSION}:eng={ev}:top:hist={history_completeness}:{board.fen()}{fp}:{depth}{n_part}"',
    1,
)
cache = cache.replace(
    '    engine_version: str | None = None,\n) -> str:\n    """Generate canonical cache key for move classification."""',
    '    engine_version: str | None = None,\n    history_completeness: str = "incomplete",\n) -> str:\n    """Generate canonical cache key for move classification."""',
    1,
)
cache = cache.replace(
    'return f"mcp:{CACHE_VERSION}:eng={ev}:classify:{board.fen()}{fp}:{move_uci}:{depth}{act_part}"',
    'return f"mcp:{CACHE_VERSION}:eng={ev}:classify:hist={history_completeness}:{board.fen()}{fp}:{move_uci}:{depth}{act_part}"',
    1,
)
cache_path.write_text(cache, encoding="utf-8")

print("audit v4 semantic migration applied")
